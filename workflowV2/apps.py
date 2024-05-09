from django.apps import AppConfig
from django.conf import settings

from .utils import minio_client


class Workflowv2Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "workflowV2"

    def ready(self):
        import workflow.signals

        bucket_name = settings.MINIO_BUCKET_NAME
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"created bucket {bucket_name}")
        else:
            print(f"bucket {bucket_name} already exists")
