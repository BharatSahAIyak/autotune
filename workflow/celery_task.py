import logging

from celery import shared_task
from django.conf import settings

from .models import Task
from .task import DataFetcher

logger = logging.getLogger(__name__)

batch_size = int(getattr(settings, "MAX_BATCH_SIZE", 10))
max_iterations = int(getattr(settings, "MAX_ITERATIONS", 100))


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def process_task(self, task_id):
    task = Task.objects.get(id=task_id)
    workflow = task.workflow
    task.status = "Processing"
    task.save()
    fetcher = DataFetcher()
    fetcher.generate_or_refine(
        workflow_id=workflow.workflow_id,
        total_examples=workflow.total_examples,
        workflow_type=workflow.workflow_type,
        llm_model=workflow.llm_model,
        refine=True,
        task_id=task_id,
        iteration=1,
        batch=1,
    )


# def check_and_update_main_task(main_task_id):
#     main_task = Task.objects.get(id=main_task_id)
#     if all(subtask.status == "Completed" for subtask in main_task.subtasks.all()):
#         # Merge data from all subtasks
#         combined_data = []
#         for subtask in main_task.subtasks.all():
#             combined_data.extend(json.loads(subtask.temp_data))

#         dataset, created = Dataset.objects.get_or_create(
#             id=main_task.dataset_id, defaults={"name": f"Dataset for {main_task.name}"}
#         )

#         dataset_id, commit_hash = upload_dataset_to_hf(combined_data, dataset.name)

#         dataset.huggingface_id = dataset_id
#         dataset.latest_commit_hash = commit_hash
#         dataset.uploaded_at = timezone.now()
#         dataset.save()

#         main_task.status = "Completed"
#         main_task.save()
