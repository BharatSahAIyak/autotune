from fastapi import HTTPException
from huggingface_hub import HfApi, CommitOperationAdd, HfFileSystem
import pandas as pd
from io import StringIO

from utils import split_data, get_data, get_cols
from models import GenerationAndCommitRequest, GenerationAndUpdateRequest


async def generate_data(redis, task_id, req: GenerationAndCommitRequest, openai_key):
    data = {"data": []}

    try:
        while len(data["data"]) < req.num_samples:
            res = await get_data(req.prompt, openai_key, req.task, req.num_labels)
            data["data"].extend(res)
            progress = min(100, len(data["data"]) / req.num_samples * 100)
            await redis.hset(task_id, mapping={"status": "Processing", "Progress": f"{progress}%", "Detail": "Generating data"})
    except Exception as e:
        detail = f"Failed to generate data: {str(e)}"
        await redis.hset(task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail})
        raise HTTPException(status_code=500, detail=detail)
    
    data["data"] = data["data"][:req.num_samples]
    return data


async def generate_and_push_data(redis, task_id, req: GenerationAndCommitRequest, 
                                 openai_key, huggingface_key
                                 ):
    data = await generate_data(redis, task_id, req, openai_key)
    train, test, val = {}, {}, {}
    train["data"], val["data"], test["data"] = split_data(data["data"], req.split)

    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=huggingface_key)
    except Exception as e:
        detail = f"Failed to connect to HF: {str(e)}"
        await redis.hset(task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail})
        raise HTTPException(status_code=500, detail=detail)

    try:
        hf_api.create_repo(repo_id=req.repo, repo_type="dataset")
    except Exception as e:
        detail = f"Failed to create repo in HF: {str(e)}"
        await redis.hset(task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail})
        raise HTTPException(status_code=500, detail=detail)

    for split, d in zip(["train", "validation", "test"], [train, val, test]):
        df = pd.DataFrame(d["data"])
        csv_data = df.to_csv()
        file_data = csv_data.encode("utf-8")
        operation = CommitOperationAdd(f"{split}.csv", file_data)
        try:
            hf_api.create_commit(
                repo_id=req.repo,
                operations=[operation],
                commit_message=f"Adding {split} csv file",
                repo_type="dataset"
            )
        except Exception as e:
            detail = f"Failed to commit to repo in HF: {str(e)}"
            await redis.hset(task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail})
            raise HTTPException(status_code=500, detail=detail)

    await redis.hset(task_id, mapping={"status": "Completed", "Progress": "None", "Detail": "None"})


async def generate_and_update_data(redis, task_id, req: GenerationAndUpdateRequest, 
                                   openai_key, huggingface_key
                                   ):
    
    data = await generate_data(redis, task_id, req, openai_key)

    try:
        fs = HfFileSystem(token=huggingface_key)
        path = f"datasets/{req.repo}/{req.split}.csv"

        original_data = fs.read_text(path)
        original_data = pd.read_csv(StringIO(original_data))

        df = pd.DataFrame(data["data"])
        
        combined_df = pd.concat([df, original_data])
        combined_df.reset_index(drop=True, inplace=True)

        columns_to_keep = get_cols(req.task)
        combined_df = combined_df[columns_to_keep]

        with fs.open(path, "w") as f:
            combined_df.to_csv(f)
    except Exception as e:
        detail = f"Failed to update repo in HF: {str(e)}"
        await redis.hset(task_id, mapping={"status": "Error", "Progress": "None", "Detail": detail})
        raise HTTPException(status_code=500, detail=detail)

    await redis.hset(task_id, mapping={"status": "Completed", "Progress": "None", "Detail": "None"})
