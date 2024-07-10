import io
import json
import logging
import re
from datetime import datetime
from decimal import Decimal, getcontext
from typing import List

import pandas as pd
from celery import shared_task
from django.conf import settings
from gevent import joinall, spawn
from huggingface_hub import CommitOperationAdd, HfApi

from workflow.models import Dataset, Examples, Prompt, Task, Workflows
from workflow.utils import create_pydantic_model, get_model_cost
from workflowV2.utils import minio_client

from .dataFetcher import DataFetcher

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def process_task(
    self,
    task_id: str,
    max_iterations: int,
    max_concurrent_fetches: int,
    batch_size: int,
    prompts: List[str],
):
    task = Task.objects.get(id=task_id)
    workflow: Workflows = task.workflow
    workflow.status = "GENERATION"
    workflow.save()
    task.status = "PROCESSING"
    task.total_samples = workflow.total_examples
    task.save()

    Model, _ = create_pydantic_model(workflow.workflow_config.schema_example)

    if len(prompts) > 0:
        generator = GenerateMultiplePrompts(
            workflow=workflow,
            prompts=prompts,
            max_iterations=max_iterations,
            max_concurrent_fetches=max_concurrent_fetches,
            batch_size=batch_size,
            task=task,
            Model=Model,
        )
        generator.controller()
    else:
        fetcher = DataFetcher(
            max_iterations=max_iterations,
            max_concurrent_fetches=max_concurrent_fetches,
            batch_size=batch_size,
        )
        prompt: Prompt = workflow.latest_prompt
        fetcher.generate_or_refine(
            workflow_id=workflow.workflow_id,
            total_examples=workflow.total_examples,
            workflow_config_id=workflow.workflow_config.id,
            llm_model=workflow.llm_model,
            prompt=prompt.user_prompt,
            prompt_id=prompt.id,
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
        workflow.save()

    workflow.status = "PUSHING_DATASET"
    workflow.save()
    task.status = "UPLOADING_DATASET"
    task.save()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    username = settings.HUGGING_FACE_USERNAME
    repo_name = f"{workflow.workflow_name}_{timestamp}"
    repo_name = re.sub(r"\s+", "_", repo_name)
    repo_id = f"{username}/{repo_name}"

    dataset_info = upload_datasets_to_hf(
        task_id, workflow.split, repo_id, workflow.total_examples
    )
    dataset = Dataset.objects.create(
        huggingface_id=repo_id,
        uploaded_at=dataset_info["uploaded_at"],
        latest_commit_hash=dataset_info["latest_commit_hash"],
        name=workflow.workflow_name,
        workflow=workflow,
    )

    workflow.status = "IDLE"
    workflow.save()
    task.status = "COMPLETED"
    task.dataset = dataset
    task.save()


class GenerateMultiplePrompts:
    def __init__(
        self,
        prompts,
        workflow,
        max_iterations,
        max_concurrent_fetches,
        batch_size,
        task,
        Model,
    ):
        self.workflow: Workflows = workflow
        self.completed_prompts = []
        self.pending_prompts = []
        self.max_iterations: int = max_iterations
        self.max_concurrent_fetches: int = max_concurrent_fetches
        self.batch_size: int = batch_size
        self.task: Task = task
        self.Model = Model
        for prompt in prompts:
            prompt = Prompt.objects.create(workflow=workflow, user_prompt=prompt)
            self.pending_prompts.append({"prompt": prompt.user_prompt, "id": prompt.id})

    def controller(self):
        greenlets = [
            spawn(
                self.request_and_save,
                prompt["prompt"],
                prompt["id"],
            )
            for prompt in [
                self.pending_prompts.pop(0)
                for _ in range(
                    min(len(self.pending_prompts), self.max_concurrent_fetches)
                )
            ]
        ]
        joinall(greenlets)

        if len(self.pending_prompts) > 0:
            self.controller()

    def request_and_save(self, user_prompt, prompt_id):
        print(f"requesting for user_prompt \n{user_prompt}")
        fetcher = DataFetcher(
            max_iterations=self.max_iterations,
            max_concurrent_fetches=self.max_concurrent_fetches,
            batch_size=self.batch_size,
        )

        fetcher.generate_or_refine(
            workflow_id=self.workflow.workflow_id,
            total_examples=self.workflow.total_examples,
            workflow_config_id=self.workflow.workflow_config.id,
            llm_model=self.workflow.llm_model,
            prompt=user_prompt,
            prompt_id=prompt_id,
            Model=self.Model,
            refine=True,
            task_id=self.task.id,
            iteration=1,
        )

        self.task.refresh_from_db()
        print(f"generated samples= {self.task.generated_samples} in celery")

        costs = get_model_cost(self.workflow.llm_model)

        getcontext().prec = 6

        input_cost = Decimal(fetcher.input_tokens * costs["input"]) / Decimal(1000)
        output_cost = Decimal(fetcher.output_tokens * costs["output"]) / Decimal(1000)

        iteration_cost = input_cost + output_cost
        iteration_cost = iteration_cost.quantize(Decimal("0.0001"))
        self.workflow.cost += iteration_cost
        self.workflow.cost = self.workflow.cost.quantize(Decimal("0.0001"))
        self.workflow.save()


def upload_datasets_to_hf(task_id, split, repo_id, total_examples):
    hf_api = HfApi(token=settings.HUGGING_FACE_TOKEN)

    repo_url = hf_api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
    )

    logger.info(f"Created repo: {repo_url}")
    examples = Examples.objects.filter(task_id=task_id)
    data = []
    examples = examples[:total_examples]
    for example in examples:
        pairs = {}
        pairs["example_id"] = str(example.example_id)
        pairs["prompt_id"] = str(example.prompt.id)

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
