import io
import json
import os
import shutil

from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from django.utils import timezone
from django_pandas.io import read_frame
from huggingface_hub import CommitOperationAdd, HfApi, login
from transformers import TrainerCallback

from workflow.models import Task
from workflow.force_alignment.alignment import ForceAligner


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def align_task(self,req_data):

    task_id=self.request.id 
    task=Task.objects.get(id=task_id)
    task.status="ALIGNING"
    task.save()
    dataset=req_data["dataset"]
    if "time_duration" in req_data:
        time_duration=req_data["time_duration"]
    else:
        time_duration=None
    
    api_key=settings.HUGGING_FACE_TOKEN
    alignment_object=ForceAligner()
    alignment_object.align_dataset(dataset,alignment_duration=time_duration)
    alignment_object.push_to_hub(req_data["save_path"],api_key)
    task.status="PUSHING"
    task.save()