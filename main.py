from fastapi import FastAPI, Security, HTTPException
from fastapi.security.api_key import APIKey, APIKeyHeader
from huggingface_hub import HfApi, CommitOperationAdd
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from utils import *


app = FastAPI()

# Security schemes
openai_key_scheme = APIKeyHeader(name="X-OpenAI-Key")
huggingface_key_scheme = APIKeyHeader(name="X-HuggingFace-Key")


class GenData(BaseModel):
    prompt: str
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]


@app.post("/data")
async def chat_completion(req: GenData,
                          openai_key: APIKey = Security(openai_key_scheme), 
                          huggingface_key: APIKey = Security(huggingface_key_scheme)
                          ):
    
    data = {"data": []}
    train, test, val = {}, {}, {}

    try:
        while len(data["data"]) < req.num_samples:
            res = get_data(req.prompt, api_key=openai_key)
            data["data"].extend(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to gen data: {str(e)}")


    data["data"] = data["data"][:req.num_samples]
    train["data"], val["data"], test["data"] = split_data(data["data"], req.split)

    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=huggingface_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to authenicate HF key: {str(e)}")

    try:
        hf_api.create_repo(repo_id=req.repo, repo_type="dataset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create repo: {str(e)}")

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
            raise HTTPException(status_code=500, detail=f"Failed to commit to repo: {str(e)}")

    return {"status": "success", "data": test["data"]}

