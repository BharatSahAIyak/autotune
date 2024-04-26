import json
import logging
import re
from datetime import datetime

import pandas as pd
from celery import shared_task
from django.conf import settings
from huggingface_hub import CommitOperationAdd, HfApi

from .models import Examples, Task, Workflows
from .task import DataFetcher

logger = logging.getLogger(__name__)

max_iterations = int(getattr(settings, "MAX_ITERATIONS", 100))


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def process_task(self, task_id):
    task = Task.objects.get(id=task_id)
    workflow: Workflows = task.workflow

    workflow.status = "GENERATION"
    workflow.save()
    task.status = "Processing"
    task.total_samples = workflow.total_examples
    task.save()

    fetcher = DataFetcher()
    fetcher.generate_or_refine(
        workflow_id=workflow.workflow_id,
        total_examples=workflow.total_examples,
        workflow_config=workflow.workflow_config,
        llm_model=workflow.llm_model,
        refine=True,
        task_id=task_id,
        iteration=1,
    )

    workflow.status = "PUSHING_DATASET"
    workflow.save()
    task.status = "Uploading"
    task.save()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    username = settings.HUGGING_FACE_USERNAME
    repo_name = f"{workflow.workflow_name}_{timestamp}"
    repo_name = re.sub(r"\s+", "_", repo_name)
    repo_id = f"{username}/{repo_name}"

    upload_datasets_to_hf(task_id, workflow.split, repo_id)

    workflow.status = "IDLE"
    workflow.save()
    task.status = "Completed"
    task.save()


def upload_datasets_to_hf(task_id, split, repo_id):
    hf_api = HfApi(token=settings.HUGGING_FACE_TOKEN)

    repo_url = hf_api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
    )

    logger.info(f"Created repo: {repo_url}")
    examples = Examples.objects.filter(task_id=task_id)
    data = []
    for example in examples:
        pairs = {}
        for key, value in example.text.items():
            pairs[key] = value
        data.append(pairs)

    df = pd.DataFrame(data)
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    total_samples = len(df)
    train_end = int(total_samples * split[0] / 100)
    validation_end = train_end + int(total_samples * split[1] / 100)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    splits = [train_df, validation_df, test_df]
    split_name = ["train", "validation", "test"]

    for i, split in enumerate(splits):
        csv_data = split.to_csv()
        file_data = csv_data.encode("utf-8")
        operation = CommitOperationAdd(f"{split_name[i]}.csv", file_data)
        hf_api.create_commit(
            repo_id=repo_id,
            operations=[operation],
            commit_message=f"Adding {split_name[i]} csv file",
            repo_type="dataset",
        )
        logger.info(f"pushed {split_name[i]} csv file")
