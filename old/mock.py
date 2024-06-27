import json
import os

from fastapi import FastAPI, Form, Query

app = FastAPI()


@app.get("/workflow/", status_code=202)
async def show_workflow(
    tag: str = Query(None, title="tag"), name: str = Query(None, title="name")
):
    """
    Get a list of Workflows based on certain parameters
    """
    with open(
        os.path.join(os.path.curdir, "docs", "mocks", "get.workflow.json"), "r"
    ) as f:
        data = json.load(f)
        return data


@app.get("/workflow/status/", status_code=202)
async def workflow(workflow_id=Query(title="workflow_id")):
    """
    Find the status of dataset generation
    """
    with open(
        os.path.join(os.path.curdir, "docs", "mocks", "get.workflow.status.json"), "r"
    ) as f:
        data = json.load(f)
        return data


from typing import Optional

from pydantic import BaseModel


class IterationRequest(BaseModel):
    prompt: str
    labels: list[str]
    valid_examples: Optional[list[str]] = []
    invalid_examples: Optional[list[str]] = []
    num_samples: Optional[int] = 5
    model: str = "gpt-3.5-turbo"


@app.post("/workflow/iterate/{id}", status_code=202)
async def workflow(id: str, req: IterationRequest):
    print(req)
    """
    Start generating sampled from GPT and return a job_id
    """
    with open(
        os.path.join(os.path.curdir, "docs", "mocks", "post.workflow.iterate.id.json"),
        "r",
    ) as f:
        data = json.load(f)
        return data


class GenerationRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    num_samples: int
    labels: list[str]
    valid_examples: Optional[list[str]] = []
    invalid_examples: Optional[list[str]] = []
    num_samples: Optional[int] = 5
    model: str = "gpt-3.5-turbo"


@app.post("/workflow/generate/{id}", status_code=202)
async def workflow(id: str, req: GenerationRequest):
    """
    Generate n-number of samples for a given prompt
    """
    with open(
        os.path.join(os.path.curdir, "docs", "mocks", "post.workflow.generate.id.json"),
        "r",
    ) as f:
        data = json.load(f)
        return data


@app.put("/workflow/", status_code=202)
async def workflow():
    """
    Create a new workflow,edit an existing one or duplciate a workflow
    """
    with open(
        os.path.join(os.path.curdir, "docs", "mocks", "put.workflow.json"), "r"
    ) as f:
        data = json.load(f)
        return data


@app.put("/dataset/{id}", status_code=202)
async def workflow(id: str):
    """
    Push the data for an existing workflow to HF
    """
    with open(
        os.path.join(os.path.curdir, "docs", "mocks", "put.dataset.id.json"), "r"
    ) as f:
        data = json.load(f)
        return data
