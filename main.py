from fastapi import FastAPI, Security, BackgroundTasks, Response
from fastapi.security.api_key import APIKey, APIKeyHeader
from celery.result import AsyncResult
from huggingface_hub import HfApi
import aioredis
import uuid
import json

from models import GenerationAndCommitRequest, GenerationAndUpdateRequest, ChatViewRequest, ModelData
from tasks import generate_and_push_data, generate_and_update_data, generate_data
from worker import celery_app


app = FastAPI()

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

@app.post("/data/view", status_code=202)
async def chat_view(req: ChatViewRequest,
                    background_tasks: BackgroundTasks,
                    openai_key: APIKey = Security(openai_key_scheme)
                    ):
    task_id = str(uuid.uuid4())
    await redis_pool.hset(task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"})
    background_tasks.add_task(generate_data, redis_pool, task_id, req, openai_key)
    return {"status": "Accepted", "task_id": task_id}

@app.post("/data", status_code=202)
async def chat_completion(req: GenerationAndCommitRequest,
                          background_tasks: BackgroundTasks,
                          openai_key: APIKey = Security(openai_key_scheme), 
                          huggingface_key: APIKey = Security(huggingface_key_scheme)
                          ):
    task_id = str(uuid.uuid4()) 
    await redis_pool.hset(task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"})
    background_tasks.add_task(generate_and_push_data, redis_pool, task_id, req, openai_key, huggingface_key)
    return {"status": "Accepted", "task_id": task_id}

@app.put("/data", status_code=202)
async def chat_updation(req: GenerationAndUpdateRequest,
                        background_tasks: BackgroundTasks,
                        openai_key: APIKey = Security(openai_key_scheme),
                        huggingface_key: APIKey = Security(huggingface_key_scheme)
                        ):
    task_id = str(uuid.uuid4())
    await redis_pool.hset(task_id, mapping={"status": "Starting", "Progress": "None", "Detail": "None"})
    background_tasks.add_task(generate_and_update_data, redis_pool, task_id, req, openai_key, huggingface_key)
    return {"status": "Accepted", "task_id": task_id}

@app.post("/train", status_code=202)
async def train_model(req: ModelData, 
                      huggingface_key: APIKey = Security(huggingface_key_scheme)
                      ):
    task = celery_app.send_task('worker.train_task', args=[dict(req), huggingface_key])
    await redis_pool.hset(str(task.id), mapping={"status": "Acknowledged", "handler": "Celery"})
    return {'task_id': str(task.id)}

@app.get("/commit")
async def commit(repo_id: str, response: Response):
    api = HfApi()
    try:
        commit_info = api.list_repo_commits(repo_id)
    except Exception as e:
        response.status_code = 404
        return {"response": str(e)}
    commit_info = [{"version": item.commit_id, "date": item.created_at} for item in commit_info if "pytorch_model.bin" in item.title]
    return {"response": commit_info}

@app.get("/track/{task_id}")
async def get_progress(task_id: str, response: Response):
    res = await redis_pool.hgetall(task_id)
    if res == {}:
        response.status_code = 404
    else:
        response.status_code = 200
        if "handler" in res and res["handler"] == "Celery":
            cres = AsyncResult(task_id, app=celery_app)
            if str(cres.status) == 'SUCCESS':
                if isinstance(res['logs'], str):
                    logs = json.loads(res['logs'])
                else:
                    logs = res['logs']
                return {"status": res['status'], "response": logs}
            return {'status': str(cres.status), 'response': cres.info}
        else:
            try:
                if isinstance(res['Detail'], str):
                    detail = json.loads(res['Detail'])
                    return {"status": res['status'], "response": detail["data"]}
            except:
                pass
    return {"response": res["response"]}


