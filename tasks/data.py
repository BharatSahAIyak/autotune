from fastapi import HTTPException
from huggingface_hub import HfApi, CommitOperationAdd
import pandas as pd

from utils import split_data, get_data
from models import GenerationAndCommitRequest


async def generate_and_push_data(redis, task_id, req: GenerationAndCommitRequest, 
                                 openai_key, huggingface_key
                                 ):
    data = {"data": []}
    train, test, val = {}, {}, {}

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

    for split, d in zip(["train", "val", "test"], [train, val, test]):
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
