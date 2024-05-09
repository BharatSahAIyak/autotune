import io
import json
import logging
import re
from datetime import datetime
from decimal import Decimal, getcontext

import pandas as pd
from celery import shared_task
from django.conf import settings
from huggingface_hub import CommitOperationAdd, HfApi

from workflowV2.utils import minio_client

from .dataFetcher import DataFetcher
from .models import Dataset, Examples, Task, Workflows
from .utils import create_pydantic_model, get_model_cost

logger = logging.getLogger(__name__)

max_iterations = int(getattr(settings, "MAX_ITERATIONS", 100))


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def process_task(self, task_id, max_iterations, max_concurrent_fetches, batch_size):
    task = Task.objects.get(id=task_id)
    workflow: Workflows = task.workflow

    workflow.status = "GENERATION"
    workflow.save()
    task.status = "Processing"
    task.total_samples = workflow.total_examples
    task.save()

    Model, _ = create_pydantic_model(workflow.workflow_config.schema_example)

    fetcher = DataFetcher(
        max_iterations=max_iterations,
        max_concurrent_fetches=max_concurrent_fetches,
        batch_size=batch_size,
    )
    fetcher.generate_or_refine(
        workflow_id=workflow.workflow_id,
        total_examples=workflow.total_examples,
        workflow_config_id=workflow.workflow_config.id,
        llm_model=workflow.llm_model,
        Model=Model,
        refine=True,
        task_id=task_id,
        iteration=1,
    )

    task.refresh_from_db()
    print(f"generated samples= {task.generated_samples} in celery")

    costs = get_model_cost(workflow.llm_model)

    getcontext().prec = 6

    input_cost = Decimal(fetcher.input_tokens * costs["input"]) / Decimal(1000)
    output_cost = Decimal(fetcher.output_tokens * costs["output"]) / Decimal(1000)

    iteration_cost = input_cost + output_cost
    iteration_cost = iteration_cost.quantize(Decimal("0.0001"))
    workflow.cost += iteration_cost
    workflow.cost = workflow.cost.quantize(Decimal("0.0001"))

    workflow.status = "PUSHING_DATASET"
    workflow.save()
    task.status = "Uploading"
    task.save()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    username = settings.HUGGING_FACE_USERNAME
    repo_name = f"{workflow.workflow_name}_{timestamp}"
    repo_name = re.sub(r"\s+", "_", repo_name)
    repo_id = f"{username}/{repo_name}"

    dataset_info = upload_datasets_to_hf(task_id, workflow.split, repo_id)
    dataset = Dataset.objects.create(
        huggingface_id=repo_id,
        uploaded_at=dataset_info["uploaded_at"],
        latest_commit_hash=dataset_info["latest_commit_hash"],
        name=workflow.workflow_name,
        workflow=workflow,
    )

    workflow.status = "IDLE"
    workflow.save()
    task.status = "Completed"
    task.dataset = dataset
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
        pairs["example_id"] = example.example_id
        for key, value in example.text.items():
            pairs[key] = value
        data.append(pairs)

    jsonData = {"data": data}
    json_str = json.dumps(jsonData)
    json_bytes = json_str.encode("utf-8")
    buffer = io.BytesIO(json_bytes)
    buffer.seek(0)

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

    uploaded_at = datetime.now()
    for i, split in enumerate(splits):
        csv_data = split.to_csv(index=False)
        file_data = csv_data.encode("utf-8")
        operation = CommitOperationAdd(f"{split_name[i]}.csv", file_data)
        commit_info = hf_api.create_commit(
            repo_id=repo_id,
            operations=[operation],
            commit_message=f"Adding {split_name[i]} csv file",
            repo_type="dataset",
        )
        logger.info(f"pushed {split_name[i]} csv file")

    latest_commit_hash = commit_info.split("/")[-1]

    minio_file_name = f"{repo_id.split('/')[1]}/{latest_commit_hash}/data.json"

    minio_client.put_object(
        bucket_name=settings.MINIO_BUCKET_NAME,
        object_name=minio_file_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
        content_type="utf-8",
    )

    return {
        "repo_url": repo_url,
        "latest_commit_hash": latest_commit_hash,
        "uploaded_at": uploaded_at,
    }
