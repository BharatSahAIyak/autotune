import json
import logging
from io import StringIO

import aioredis
import coloredlogs
import pandas as pd
from fastapi import HTTPException
from huggingface_hub import HfFileSystem

logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.

coloredlogs.install(logger=logger)
logger.propagate = False


def split_data(res, split):
    assert sum(split) == 100, "Split should sum to 100"
    train_end = int(len(res) * split[0] / 100)
    val_end = train_end + int(len(res) * split[1] / 100)
    test_end = val_end + int(len(res) * split[2] / 100)
    train = res[:train_end]
    val = res[train_end:val_end]
    test = res[val_end:test_end]
    return train, val, test


async def prep_for_upload(train, val, test, data, req, content_row):
    new_train, new_val, new_test = {}, {}, {}

    try:
        new_train["data"], new_val["data"], new_test["data"] = split_data(
            data["data"], req.split
        )
    except Exception as e:
        logger.error(f"Error splitting data: {e}")

    for i, (old, new) in enumerate(
        zip([train, val, test], [new_train, new_val, new_test])
    ):
        df = pd.DataFrame(new["data"])
        df["content_row"] = content_row
        combined_df = pd.concat([old, df]).reset_index(drop=True)

        old = combined_df
        if i == 0:
            train = combined_df
        elif i == 1:
            val = combined_df
        else:
            test = combined_df

    return train, val, test


async def get_redis_keys_status(redis_pool, req, huggingface_key, task_id):
    try:
        r = await redis_pool
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return

    try:
        keys = await r.keys("*")
    except Exception as e:
        logger.error(f"Error getting keys from Redis: {e}")
        return

    train, test, val = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for key in keys:
        try:
            key_type = await r.type(key)
            if key_type == "hash":
                hash_values = await r.hgetall(key)
                content_row = hash_values["content_row"]
                data = hash_values["data"]
                data = json.loads(data)
                if hash_values["status"] == "Generated":
                    train, val, test = await prep_for_upload(
                        train, test, val, data, req, content_row
                    )
                    print("prepped for upload")

                await r.hset(key, "status", "SUCCESS")

        except Exception as e:
            logger.error(f"Error checking status for key {key}: {e}")

    await push_dataset_to_hf(
        redis_pool, task_id, req, huggingface_key, train, val, test, content_row
    )


async def push_dataset_to_hf(
    redis, task_id, req, huggingface_key, train, val, test, content_row
):
    try:
        fs = HfFileSystem(token=huggingface_key)
        for split, df in zip(["train", "validation", "test"], [train, val, test]):
            path = f"datasets/{req.repo}/{split}.csv"

            original_data = fs.read_text(path)
            original_data = pd.read_csv(StringIO(original_data))
            original_data = original_data[["question", "answer", "content_row"]]

            combined_df = pd.concat([df, original_data]).reset_index(drop=True)

            with fs.open(path, "w") as f:
                combined_df.to_csv(f)
            logger.info(f"Pushed data to HF for split {split}")

    except Exception as e:
        detail = f"Failed to update repo in HF: {str(e)}"
        await redis.hset(
            task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail}
        )
        raise HTTPException(status_code=500, detail=detail)


if __name__ == "__main__":
    redis_url = "redis://localhost"
    redis_pool = aioredis.from_url(redis_url, decode_responses=True)

    import asyncio

    asyncio.run(get_redis_keys_status(redis_pool))
