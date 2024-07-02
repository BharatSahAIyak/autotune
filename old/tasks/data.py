import json
import logging
from io import StringIO

import coloredlogs
import pandas as pd
from fastapi import HTTPException
from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem
from sklearn.preprocessing import LabelEncoder

from old.models import GenerationAndCommitRequest, GenerationAndUpdateRequest
from old.tasks.data_fetcher import DataFetcher
from old.utils import split_data

logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.

coloredlogs.install(logger=logger)
logger.propagate = False


async def generate_data(redis, task_id, req: GenerationAndCommitRequest, openai_key):
    data = {"data": []}
    try:
        fetcher = DataFetcher(req, openai_key, redis, task_id)
        d = await fetcher.fetch()
        data["data"] = d
    except Exception as e:
        detail = f"Failed to generate data: {str(e)}"
        await redis.hset(
            task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail}
        )
        raise HTTPException(status_code=500, detail=detail)

    logger.info("Generated %s samples", len(data["data"]))
    logger.info("Saving data to redis")
    data["data"] = data["data"][: req.num_samples]
    detail = {}
    detail["data"] = data["data"] if len(data["data"]) < 50 else data["data"][:50]
    await redis.hset(
        task_id, mapping={"status": "Generated", "Detail": json.dumps(detail)}
    )
    logger.info("Task %s completed", task_id)
    return data


async def generate_and_push_data(
    redis, task_id, req: GenerationAndCommitRequest, openai_key, huggingface_key
):
    data = await generate_data(redis, task_id, req, openai_key)
    await redis.hset(
        task_id,
        mapping={
            "status": "Completed",
            "Progress": "None",
            "Detail": json.dumps(data),
        },
    )
    await push_dataset_to_hf(redis, task_id, req, huggingface_key, data)


async def push_dataset_to_hf(redis, task_id, req, huggingface_key, data):
    train, test, val = {}, {}, {}
    train["data"], val["data"], test["data"] = split_data(data["data"], req.split)
    repo_id = req.repo + "-" + task_id

    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=huggingface_key)
    except Exception as e:
        detail = f"Failed to connect to HF: {str(e)}"
        await redis.hset(
            task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail}
        )
        raise HTTPException(status_code=500, detail=detail)

    try:
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        detail = f"Failed to create repo in HF: {str(e)}"
        await redis.hset(
            task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail}
        )
        raise HTTPException(status_code=500, detail=detail)

    for split, d in zip(["train", "validation", "test"], [train, val, test]):
        df = pd.DataFrame(d["data"])
        df["label_string"] = df["label"]
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])

        csv_data = df.to_csv()
        file_data = csv_data.encode("utf-8")
        operation = CommitOperationAdd(f"{split}.csv", file_data)
        try:
            hf_api.create_commit(
                repo_id=repo_id,
                operations=[operation],
                commit_message=f"Adding {split} csv file",
                repo_type="dataset",
            )
        except Exception as e:
            detail = f"Failed to commit to repo in HF: {str(e)}"
            await redis.hset(
                task_id,
                mapping={"status": "Error", "Progress": "None", "Detail": detail},
            )
            raise HTTPException(status_code=500, detail=detail)

    await redis.hset(
        task_id, mapping={"status": "Completed", "Progress": "None", "Detail": "None"}
    )


async def generate_and_update_data(
    redis, task_id, req: GenerationAndUpdateRequest, openai_key, huggingface_key
):
    data = await generate_data(redis, task_id, req, openai_key)

    try:
        fs = HfFileSystem(token=huggingface_key)
        path = f"datasets/{req.repo}/{req.split}.csv"

        original_data = fs.read_text(path)
        original_data = pd.read_csv(StringIO(original_data))
        columns_to_keep = ["text", "label"]
        original_data = original_data[columns_to_keep]

        df = pd.DataFrame(data["data"])

        combined_df = pd.concat([df, original_data]).reset_index(drop=True)

        with fs.open(path, "w") as f:
            combined_df.to_csv(f)
    except Exception as e:
        detail = f"Failed to update repo in HF: {str(e)}"
        await redis.hset(
            task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail}
        )
        raise HTTPException(status_code=500, detail=detail)

    await redis.hset(
        task_id, mapping={"status": "Completed", "Progress": "None", "Detail": "None"}
    )


def delete_hf_repo(repo, token):
    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=token)
        hf_api.delete_repo(repo_id=repo, repo_type="dataset")
    except Exception as e:
        detail = f"Failed to delete repo in HF: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)


def check_if_hf_repo_exists(repo, token):
    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=token)
        hf_api.dataset_info(repo_id=repo)
        return True
    except Exception as e:
        return False
