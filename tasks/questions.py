import json
import logging
from io import StringIO

import coloredlogs
import pandas as pd
from fastapi import HTTPException
from huggingface_hub import CommitOperationAdd, HfApi, HfFileSystem
from sklearn.preprocessing import LabelEncoder

from models import QuestionCreationRequest, QuestionUpdationRequest
from tasks.data_fetcher import DataFetcher
from utils import split_data, upload

logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.

coloredlogs.install(logger=logger)
logger.propagate = False


async def generate_questions(
    redis,
    task_id,
    req: QuestionCreationRequest,
    openai_key,
):
    data = {"data": []}
    try:
        fetcher = DataFetcher(req, openai_key, redis, task_id, True)
        d = await fetcher.fetch()
        data["data"] = d
    except Exception as e:
        detail = f"Failed to generate data: {str(e)}"
        await redis.hset(
            task_id,
            mapping={
                "status": "Error",
                "content_row": req.combined_index if req.multiple_chunks else req.index,
                "Progress": "None",
                "Detail": detail,
            },
        )
        raise HTTPException(status_code=500, detail=detail)

    logger.info("Generated %s samples", len(data["data"]))
    logger.info("Saving data to redis")
    data["data"] = data["data"][: req.num_samples]
    detail = {}
    detail["data"] = data["data"] if len(data["data"]) < 50 else data["data"][:50]

    await redis.hset(
        task_id,
        mapping={
            "status": "Generated",
            "content_row": req.combined_index if req.multiple_chunks else req.index,
            "Detail": json.dumps(detail),
        },
    )

    logger.info("Task %s completed", task_id)
    return data


async def generate_and_push_questions(
    redis, task_id, req: QuestionCreationRequest, openai_key, huggingface_key
):
    data = await generate_questions(redis, task_id, req, openai_key)
    await redis.hset(
        task_id,
        mapping={
            "status": "Completed",
            "content_row": req.combined_index if req.multiple_chunks else req.index,
            "Progress": "None",
            "Detail": json.dumps(data),
        },
    )
    await push_questionset_to_hf(redis, task_id, req, huggingface_key, data)


async def push_questionset_to_hf(redis, task_id, req, huggingface_key, data):
    train, test, val = {}, {}, {}
    train["data"], val["data"], test["data"] = split_data(data["data"], req.split)
    repo_id = req.repo + "-" + task_id

    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=huggingface_key)
    except Exception as e:
        detail = f"Failed to connect to HF: {str(e)}"
        await redis.hset(
            task_id,
            mapping={
                "status": "Error",
                "content_row": req.combined_index if req.multiple_chunks else req.index,
                "Progress": "None",
                "Detail": detail,
            },
        )
        raise HTTPException(status_code=500, detail=detail)

    logger.info("Connected to HF")

    try:
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        detail = f"Failed to create repo in HF: {str(e)}"
        await redis.hset(
            task_id,
            mapping={
                "status": "Error",
                "content_row": req.combined_index if req.multiple_chunks else req.index,
                "Progress": "None",
                "Detail": detail,
            },
        )
        raise HTTPException(status_code=500, detail=detail)

    logger.info("created repo in HF")

    for split, d in zip(["train", "validation", "test"], [train, val, test]):
        df = pd.DataFrame(d["data"])
        if not req.multiple_chunks:
            df["content_row"] = req.index
        else:
            df["content_row"] = req.combined_index

        csv_data = df.to_csv()
        file_data = csv_data.encode("utf-8")

        operation = CommitOperationAdd(f"{split}.csv", file_data)
        try:
            hf_api.create_commit(
                repo_id=repo_id,
                operations=[operation],
                commit_message=f"Adding {split} csv file",
                token=huggingface_key,
                repo_type="dataset",
            )
            logger.info("Pushed %s data to HF", split)
        except Exception as e:
            logger.error("Failed to push %s data to HF", split)
            logger.error(f"Failed to push data to HF {str(e)}")
            detail = f"Failed to commit to repo in HF: {str(e)}"
            await redis.hset(
                task_id,
                mapping={"status": "Error", "Progress": "None", "Detail": detail},
            )
            raise HTTPException(status_code=500, detail=detail)


async def generate_and_update_questions(
    redis,
    task_id,
    req: QuestionUpdationRequest,
    openai_key,
    huggingface_key,
):
    data = await generate_questions(redis, task_id, req, openai_key)

    if req.bulk_process:  # for multiple put requests, update HF once every 50 requests
        if req.index % 50 == 0:
            await upload.get_redis_keys_status(redis, req, huggingface_key, task_id)
    else:
        logger.info("Updating HF now")
        try:
            fs = HfFileSystem(token=huggingface_key)

            train, test, val = {}, {}, {}
            train["data"], val["data"], test["data"] = split_data(
                data["data"], req.split
            )
            for split, d in zip(["train", "validation", "test"], [train, val, test]):
                path = f"datasets/{req.repo}/{split}.csv"

                original_data = fs.read_text(path)
                original_data = pd.read_csv(StringIO(original_data))
                original_data = original_data[["question", "answer", "content_row"]]

                df = pd.DataFrame(d["data"])
                df["content_row"] = req.index

                combined_df = pd.concat([df, original_data]).reset_index(drop=True)
                with fs.open(path, "w") as f:
                    combined_df.to_csv(f)
                logger.info(
                    f"Pushed data to HF for index {req.index} and split {split}"
                )

        except Exception as e:
            detail = f"Failed to update repo in HF: {str(e)}"
            await redis.hset(
                task_id,
                mapping={"status": "Error", "Progress": "None", "Detail": detail},
            )
            raise HTTPException(status_code=500, detail=detail)

        await redis.hset(
            task_id,
            mapping={"status": "Completed", "Progress": "None", "Detail": "None"},
        )
