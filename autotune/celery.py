import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autotune.settings")
app = Celery("django_celery")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


# celery -A autotune worker --loglevel=info

