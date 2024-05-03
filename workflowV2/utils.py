from django.conf import settings
from minio import Minio

minio_client = Minio(
    settings.MINIO_BASE_URL,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE_CONN,
)
