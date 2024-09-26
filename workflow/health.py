import dj_database_url
import openai
import psycopg2
import redis
from celery import Celery
from django.conf import settings
from huggingface_hub import HfApi
from minio import Minio
from minio.error import S3Error
from openai import OpenAI


class HealthCheck:
    def openai(self):
        try:
            openai_key = settings.OPENAI_API_KEY
            if not openai_key:
                return self.create_health_status(
                    "OpenAI API",
                    "external",
                    "openai.com",
                    "Synthetic data generation will be impacted",
                    {"isAvailable": False, "error": "OPENAI_API_KEY is not set"},
                    {"timeForResolutionInMinutes": 20, "priority": 0},
                )
            client = OpenAI(api_key=openai_key)
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Say hello"}],
            )
            return self.create_health_status(
                "OpenAI API",
                "external",
                "openai.com",
                "Synthetic data generation will be impacted",
                {"isAvailable": True},
                None,
            )
        except Exception as e:
            return self.create_health_status(
                "OpenAI API",
                "external",
                "openai.com",
                "Synthetic data generation will be impacted",
                {"isAvailable": False, "error": str(e)},
                {"timeForResolutionInMinutes": 20, "priority": 0},
            )

    def redis(self):
        try:
            redis_client = redis.StrictRedis.from_url(settings.REDIS_URL)
            redis_client.ping()
            return self.create_health_status(
                "Redis",
                "internal",
                None,
                "Caching, Data Generation and Model training impacted",
                {"isAvailable": True},
                None,
            )
        except Exception as e:
            return self.create_health_status(
                "Redis",
                "internal",
                None,
                "Caching, Data Generation and Model training impacted",
                {"isAvailable": False, "error": str(e)},
                {"timeForResolutionInMinutes": 20, "priority": 0},
            )

    def celery_workers(self):
        try:
            app = Celery("django_celery")
            app.config_from_object("django.conf:settings", namespace="CELERY")
            inspector = app.control.inspect()
            stats = inspector.stats()
            if not stats:
                return self.create_health_status(
                    "Celery Workers",
                    "internal",
                    None,
                    "Task processing will be impacted",
                    {
                        "isAvailable": False,
                        "error": "No running Celery workers were found",
                    },
                    {"timeForResolutionInMinutes": 20, "priority": 0},
                )
            return self.create_health_status(
                "Celery Workers",
                "internal",
                None,
                "Task processing will be impacted",
                {"isAvailable": True},
                None,
            )
        except Exception as e:
            return self.create_health_status(
                "Celery Workers",
                "internal",
                None,
                "Task processing will be impacted",
                {"isAvailable": False, "error": str(e)},
                {"timeForResolutionInMinutes": 20, "priority": 0},
            )

    def postgres(self):
        try:
            conn_params = dj_database_url.parse(settings.AUTOTUNE_DATABASE_URL)
            conn = psycopg2.connect(
                dbname=conn_params["NAME"],
                user=conn_params["USER"],
                password=conn_params["PASSWORD"],
                host=conn_params["HOST"],
                port=conn_params["PORT"],
            )
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            return self.create_health_status(
                "PostgreSQL",
                "internal",
                None,
                "All core functionalities impacted",
                {"isAvailable": True},
                None,
            )
        except Exception as e:
            return self.create_health_status(
                "PostgreSQL",
                "internal",
                None,
                "All core functionalities impacted",
                {"isAvailable": False, "error": str(e)},
                {"timeForResolutionInMinutes": 20, "priority": 0},
            )

    def huggingface(self):
        try:
            api = HfApi()
            token = settings.HUGGING_FACE_TOKEN
            if not token:
                raise ValueError("HUGGING_FACE_TOKEN is not set")
            api.list_models(token=token)
            return self.create_health_status(
                "Hugging Face API",
                "external",
                "huggingface.co",
                "All core functionalities of autotune impacted",
                {"isAvailable": True},
                None,
            )
        except Exception as e:
            return self.create_health_status(
                "Hugging Face API",
                "external",
                "huggingface.co",
                "All core functionalities of autotune impacted",
                {"isAvailable": False, "error": str(e)},
                {"timeForResolutionInMinutes": 20, "priority": 0},
            )

    def minio(self):
        try:
            minio_client = Minio(
                settings.MINIO_BASE_URL,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE_CONN,
            )
            minio_client.list_buckets()
            return self.create_health_status(
                "Minio",
                "internal",
                None,
                "Mass prompt handling will be impacted, along with download of synthetic data json and csv",
                {"isAvailable": True},
                None,
            )
        except Exception as e:
            return self.create_health_status(
                "Minio",
                "internal",
                None,
                "Mass prompt handling will be impacted, along with download of synthetic data json and csv",
                {"isAvailable": False, "error": str(e)},
                {"timeForResolutionInMinutes": 20, "priority": 0},
            )

    def create_health_status(self, name, type, endpoint, impact_message, status, sla):
        status_dict = {
            "name": name,
            "type": type,
            "impactMessage": impact_message,
            "status": status,
        }
        if endpoint:
            status_dict["endpoint"] = endpoint
        if status["isAvailable"]:
            status_dict["sla"] = sla
        return status_dict
