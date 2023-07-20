from fastapi import FastAPI, Security, BackgroundTasks, Response
from fastapi.security.api_key import APIKey, APIKeyHeader
import aioredis
import uuid

from models import GenerationAndCommitRequest
from tasks import generate_and_push_data


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


@app.post("/data", status_code=202)
async def chat_completion(req: GenerationAndCommitRequest,
                          background_tasks: BackgroundTasks,
                          openai_key: APIKey = Security(openai_key_scheme), 
                          huggingface_key: APIKey = Security(huggingface_key_scheme)
                          ):
    task_id = str(uuid.uuid4()) 
    await redis_pool.hset(task_id, mapping={"status": "Starting", "Progress": "None", "Details": "None"})
    background_tasks.add_task(generate_and_push_data, redis_pool, task_id, req, openai_key, huggingface_key)
    return {"status": "Accepted", "task_id": task_id}


@app.get("/track/{task_id}")
async def get_progress(task_id: str, response: Response):
    res = await redis_pool.hgetall(task_id)
    if res == {}:
        response.status_code = 404
    else:
        response.status_code = 200
    return {"task_id": task_id, "response": res}

