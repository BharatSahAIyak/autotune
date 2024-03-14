from celery import shared_task
from django.conf import settings

from autotune.redis import redis_conn
from workflow.models import Task


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def process_subtask(self, task_id, workflow_id):
    try:
        task = Task.objects.get(id=task_id)
        task.status = "Processing"
        task.save()

        # print("DUMMY")

        # Upon successful completion
        task.status = "Completed"
        task.save()

        # Updating Redis for progress tracking
        redis_conn.hincrby(f"workflow_progress:{workflow_id}", "completed", 1)
    except Task.DoesNotExist:
        print(f"Task {task_id} does not exist.")
    except Exception as exc:
        task.status = "Failed"
        task.save()
        # Do we need to update Redis to reflect the failed attempt or leave it for manual review ?
        raise self.retry(exc=exc)


def create_and_dispatch_subtasks(task_ids, workflow_id):
    # Initialize progress tracking in Redis for the new workflow
    redis_conn.hmset(f"workflow_progress:{workflow_id}", {"completed": 0})

    # Dispatch each task to Celery
    for task_id in task_ids:
        process_subtask.delay(task_id, workflow_id)
