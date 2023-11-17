import json
import logging
import random
import string
import time
import uuid

import aioredis
import celery.signals
import coloredlogs
from celery.result import AsyncResult
from fastapi import BackgroundTasks, FastAPI, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKey, APIKeyHeader
from huggingface_hub import HfApi

from models import (
    ChatViewRequest,
    GenerationAndCommitRequest,
    GenerationAndUpdateRequest,
    ModelData,
)
from tasks import generate_and_push_data, generate_and_update_data, generate_data
from worker import celery_app

# import logging.config


@celery.signals.setup_logging.connect
def on_celery_setup_logging(**kwargs):
    pass


app = FastAPI()

# setup loggers
# logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.

coloredlogs.install(logger=logger)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs incoming HTTP requests and their responses.

    Parameters:
        request (Request): The incoming HTTP request.
        call_next (Callable): The next middleware or the endpoint handler.

    Returns:
        Response: The HTTP response returned by the endpoint handler or the next middleware.
    """
    idem = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info("rid=%s start request path=%s", idem, request.url.path)
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    logger.info(
        "rid=%s completed_in=%sms status_code=%s",
        idem,
        formatted_process_time,
        response.status_code,
    )

    return response


# Security schemes
openai_key_scheme = APIKeyHeader(name="X-OpenAI-Key")
huggingface_key_scheme = APIKeyHeader(name="X-HuggingFace-Key")

# Redis connection pool
redis_pool = None


@app.on_event("startup")
async def startup_event():
    global redis_pool
    redis_pool = aioredis.from_url("redis://localhost", decode_responses=True)


@app.on_event("shutdown")
async def shutdown_event():
    await redis_pool.close()

@app.post("/sample", status_code=202)
async def sample_generation(
    req: GenerationAndCommitRequest,
    openai_key: APIKey = Security(openai_key_scheme),
):
    task_id = str(uuid.uuid4())
    await redis_pool.hset(
        task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"}
    )
    data = await generate_data(redis_pool, task_id, req, openai_key)
    return {"response": data}



@app.post("/data/view", status_code=202)
async def chat_view(
    req: ChatViewRequest,
    background_tasks: BackgroundTasks,
    openai_key: APIKey = Security(openai_key_scheme),
):
    task_id = str(uuid.uuid4())
    await redis_pool.hset(
        task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"}
    )
    background_tasks.add_task(generate_data, redis_pool, task_id, req, openai_key)
    return {"status": "Accepted", "task_id": task_id}


@app.post("/data", status_code=202)
async def chat_completion(
    req: GenerationAndCommitRequest,
    background_tasks: BackgroundTasks,
    openai_key: APIKey = Security(openai_key_scheme),
    huggingface_key: APIKey = Security(huggingface_key_scheme),
):
    task_id = str(uuid.uuid4())
    await redis_pool.hset(
        task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"}
    )
    background_tasks.add_task(
        generate_and_push_data, redis_pool, task_id, req, openai_key, huggingface_key
    )
    return {"status": "Accepted", "task_id": task_id}


@app.put("/data", status_code=202)
async def chat_updation(
    req: GenerationAndUpdateRequest,
    background_tasks: BackgroundTasks,
    openai_key: APIKey = Security(openai_key_scheme),
    huggingface_key: APIKey = Security(huggingface_key_scheme),
):
    task_id = str(uuid.uuid4())
    await redis_pool.hset(
        task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"}
    )
    background_tasks.add_task(
        generate_and_update_data, redis_pool, task_id, req, openai_key, huggingface_key
    )
    return {"status": "Accepted", "task_id": task_id}


@app.post("/train", status_code=202)
async def train_model(
    req: ModelData, huggingface_key: APIKey = Security(huggingface_key_scheme)
):
    task = celery_app.send_task("worker.train_task", args=[dict(req), huggingface_key])
    await redis_pool.hset(
        str(task.id), mapping={"status": "Acknowledged", "handler": "Celery"}
    )
    return {"task_id": str(task.id)}


@app.get("/commit")
async def commit(repo_id: str, response: Response):
    api = HfApi()
    try:
        commit_info = api.list_repo_commits(repo_id)
    except Exception as e:
        response.status_code = 404
        return {"response": str(e)}
    commit_info = [
        {"version": item.commit_id, "date": item.created_at}
        for item in commit_info
        if "pytorch_model.bin" in item.title
    ]
    return {"response": commit_info}


@app.get("/track/{task_id}")
async def get_progress(task_id: str, response: Response):
    res = await redis_pool.hgetall(task_id)
    logger.debug("CP-1")
    if res == {}:
        response.status_code = 404
    else:
        response.status_code = 200
        logger.debug("CP-2")
        if "handler" in res and res["handler"] == "Celery":
            cres = AsyncResult(task_id, app=celery_app)
            logger.debug("CP-3")
            if str(cres.status) == "SUCCESS":
                if isinstance(res["logs"], str):
                    logs = json.loads(res["logs"])
                else:
                    logs = res["logs"]
                return {"status": res["status"], "response": logs}
            return {"status": str(cres.status), "response": cres.info}
        else:
            try:
                if isinstance(res["Detail"], str):
                    detail = json.loads(res["Detail"])
                    return {"status": res["status"], "response": detail["data"]}
            except:
                pass
    return {
        "status": res["status"],
        "response": {"Detail": res["Detail"], "Progress": res["Progress"]},
    }
