from celery import Celery

from tasks import train_model


celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')


@celery_app.task(bind=True)
def train_task(self, req, api_key):
    train_model(self, req, api_key)
