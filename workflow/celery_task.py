import json

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from autotune.redis import redis_conn
from workflow.models import Task, Dataset
from workflow.task import call_llm_generate, construct_prompt
from workflow.utils import upload_dataset_to_hf, get_workflow_config


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def process_subtask(self, task_id):
    try:
        task = Task.objects.get(id=task_id)
        task.status = "Processing"
        task.save()

        config = get_workflow_config(task.workflow.workflow_type)
        prompt_text = construct_prompt(task.workflow, config, True, task.total_number)
        generated_data = call_llm_generate(prompt_text, task.workflow, {})
        task.temp_data = json.dumps(generated_data)
        task.status = "Completed"
        task.save()

        if task.parent_task:
            redis_conn.hincrby(f"task_progress:{task.parent_task.id}", "completed", 1)
            check_and_update_main_task(task.parent_task_id)

    except Task.DoesNotExist:
        print(f"Task {task_id} does not exist.")
    except Exception as exc:
        task.status = "Failed"
        task.save()
        raise self.retry(exc=exc)


def check_and_update_main_task(main_task_id):
    main_task = Task.objects.get(id=main_task_id)
    if all(subtask.status == "Completed" for subtask in main_task.subtasks.all()):
        # Merge data from all subtasks
        combined_data = []
        for subtask in main_task.subtasks.all():
            combined_data.extend(json.loads(subtask.temp_data))

        dataset, created = Dataset.objects.get_or_create(
            id=main_task.dataset_id,
            defaults={'name': f"Dataset for {main_task.name}"}
        )

        dataset_id, commit_hash = upload_dataset_to_hf(combined_data, dataset.name)

        dataset.huggingface_id = dataset_id
        dataset.latest_commit_hash = commit_hash
        dataset.uploaded_at = timezone.now()
        dataset.save()

        main_task.status = "Completed"
        main_task.save()


def create_and_dispatch_subtasks(main_task_id, workflow_id, number_of_subtasks):
    main_task = Task.objects.get(id=main_task_id)

    # Resetting Redis progress for the main task
    redis_conn.hmset(f"task_progress:{main_task_id}", {"total": number_of_subtasks, "completed": 0})

    for _ in range(number_of_subtasks):
        subtask = Task.objects.create(
            name=f"Subtask for {main_task.name}",
            status="Starting",
            workflow=main_task.workflow,
            parent_task=main_task
        )
        process_subtask.delay(subtask.id, workflow_id)
