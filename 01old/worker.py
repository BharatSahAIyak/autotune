import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

print("PYTORCH_ENABLE_MPS_FALLBACK", os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

import aioredis
from celery import Celery

from tasks import train_model

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery_app.conf.update(worker_hijack_root_logger=False)


@celery_app.task(bind=True)
def train_task(self, req, api_key):
    meta = train_model(self, req, api_key)
    loop = asyncio.get_event_loop()
    redis_pool = aioredis.from_url("redis://localhost", decode_responses=True)
    meta = json.dumps(meta)
    task = loop.create_task(
        redis_pool.hset(
            str(self.request.id), mapping={"status": "COMPLETED", "logs": meta}
        )
    )
    loop.run_until_complete(task)
    task = loop.create_task(redis_pool.close())
    loop.run_until_complete(task)
