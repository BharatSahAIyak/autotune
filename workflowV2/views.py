import io

import pandas as pd
from django.conf import settings
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from huggingface_hub import HfApi
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from workflow.mixins import UserIDMixin
from workflow.models import Dataset, Task, User, WorkflowConfig, Workflows
from workflow.serializers import (
    PromptSerializer,
    WorkflowConfigSerializer,
    WorkflowDetailSerializer,
    WorkflowSerializer,
)
from workflow.utils import create_pydantic_model
from workflow.views import GenerateTaskView, IterateWorkflowView

from .utils import minio_client


class WorkflowListView(UserIDMixin, APIView):
    """
    List all workflows or create a new workflow.
    """

    def get(self, request, *args, **kwargs):
        user_id = request.META["user"].user_id
        workflows = Workflows.objects.filter(user_id=user_id)
        serializer = WorkflowDetailSerializer(workflows, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        serializer = WorkflowSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.META["user"])
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class WorkflowDetailView(UserIDMixin, APIView):
    """
    Retrieve, update, or delete a workflow instance.
    """

    def get(self, request, workflow_id, *args, **kwargs):
        user_id = request.META["user"].user_id

        workflow = get_object_or_404(
            Workflows, workflow_id=workflow_id, user_id=user_id
        )
        serializer = WorkflowDetailSerializer(workflow)
        return Response(serializer.data)

    def put(self, request, workflow_id, *args, **kwargs):
        user_id = request.META["user"].user_id

        workflow = get_object_or_404(
            Workflows, workflow_id=workflow_id, user_id=user_id
        )
        serializer = WorkflowSerializer(workflow, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, workflow_id, *args, **kwargs):
        user_id = request.META["user"].user_id

        workflow = get_object_or_404(
            Workflows, workflow_id=workflow_id, user_id=user_id
        )
        workflow.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@method_decorator(csrf_exempt, name="dispatch")
class WorkflowConfigCreateView(UserIDMixin, APIView):
    def post(self, request, *args, **kwargs):
        with transaction.atomic():
            user: User = request.META["user"]

            config_data = request.data.get("config", {})

            if "schema_example" not in config_data:
                return Response(
                    {"message": "Schema Example is required!"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if "config_name" in config_data:
                config_data["name"] = config_data.pop("config_name")

            Model, model_string = create_pydantic_model(config_data["schema_example"])
            field_names = list(Model.__fields__.keys())
            field_info = list(Model.__fields__.values())

            fields = [
                {name: info.annotation.__name__}
                for name, info in zip(field_names, field_info)
            ]

            combined_config_data = {
                "model_string": model_string,
                "fields": fields,
                **config_data,  # Ensure other fields are included
            }

            workflow_config_serializer = WorkflowConfigSerializer(
                data=combined_config_data
            )
            if workflow_config_serializer.is_valid():
                workflow_config: WorkflowConfig = workflow_config_serializer.save()
            else:
                return Response(
                    workflow_config_serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST,
                )

            workflow_data = request.data.get("workflow", {})

            if "user_prompt" in workflow_data:
                user_prompt = workflow_data.pop("user_prompt")
            else:
                return Response(
                    {"message": "User Prompt is required!"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            workflow_data["user"] = user.user_id
            workflow_data["workflow_config"] = workflow_config.id

            workflow_serializer = WorkflowSerializer(data=workflow_data)
            if workflow_serializer.is_valid():
                workflow: Workflows = workflow_serializer.save()

            else:
                return Response(
                    workflow_serializer.errors, status=status.HTTP_400_BAD_REQUEST
                )

            prompt_data = {
                "user_prompt": user_prompt,
                "system_prompt": workflow_config.system_prompt,
                "workflow": workflow.workflow_id,
            }

            prompt_serializer = PromptSerializer(data=prompt_data)

            if prompt_serializer.is_valid():
                prompt_serializer.save()
            else:
                return Response(
                    prompt_serializer.errors, status=status.HTTP_400_BAD_REQUEST
                )

            workflow_complete_data = WorkflowDetailSerializer(workflow).data

            return Response(
                {
                    "workflow": workflow_complete_data,
                    "config": workflow_config_serializer.data,
                    "prompt": prompt_data,
                },
                status=status.HTTP_201_CREATED,
            )


@method_decorator(csrf_exempt, name="dispatch")
class WorkflowIterateView(UserIDMixin, APIView):
    def post(self, request, workflow_id):
        iterator = IterateWorkflowView()
        return iterator.post(request, workflow_id)


@method_decorator(csrf_exempt, name="dispatch")
class WorkflowGenerateView(UserIDMixin, APIView):
    def post(self, request, workflow_id):
        generator = GenerateTaskView()
        return generator.post(request, workflow_id)


@method_decorator(csrf_exempt, name="dispatch")
class GetDataView(UserIDMixin, APIView):
    def post(self, request):
        data = request.data
        workflow_id = data.get("workflow_id")
        task_id = data.get("task_id")
        user_id = request.META["user"].user_id

        format = data.get("format")

        if format and format != "csv" and format != "json":
            return Response(
                {"error": "Invalid format. supported formats are 'csv' or 'json'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            if task_id:
                task = get_object_or_404(Task, pk=task_id, workflow__user_id=user_id)
                percentage = task.generated_samples / task.total_samples * 100
                if percentage > 100:
                    percentage = 100.0
                return Response(
                    {
                        "workflow_id": str(task.workflow_id),
                        "data": [
                            {
                                "task_id": task_id,
                                "percentage": percentage,
                                "links": self.get_dataset_links(task.dataset, format),
                            }
                        ],
                    },
                    status=status.HTTP_200_OK,
                )

            elif workflow_id:

                workflow = get_object_or_404(
                    Workflows, workflow_id=workflow_id, user_id=user_id
                )
                tasks = Task.objects.filter(workflow=workflow)
                data = []
                for task in tasks:
                    percentage = task.generated_samples / task.total_samples * 100
                    if percentage > 100:
                        percentage = 100.0
                    data.append(
                        {
                            "task_id": task.id,
                            "percentage": percentage,
                            "links": self.get_dataset_links(task.dataset, format),
                        }
                    )
                return Response(
                    {"workflow_id": workflow_id, "data": data},
                    status=status.HTTP_200_OK,
                )

            else:
                return Response(
                    {"error": "Either workflow_id or task_id must be provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_dataset_links(self, dataset: Dataset, format: str):
        if dataset is None:
            return {}
        hf_api = HfApi(token=settings.HUGGING_FACE_TOKEN)

        files = ["test", "validation", "train"]
        csv_links = {}
        minio_file_name = f"{dataset.huggingface_id.split('/')[1]}/{dataset.latest_commit_hash}/data.json"
        json_url = minio_client.presigned_get_object(
            bucket_name=settings.MINIO_BUCKET_NAME,
            object_name=minio_file_name,
        )
        if format is None or format == "csv":
            for file in files:
                if hf_api.file_exists(
                    repo_id=dataset.huggingface_id,
                    filename=f"{file}.csv",
                    repo_type="dataset",
                    revision=dataset.latest_commit_hash,
                ):
                    minio_file_name = f"{dataset.huggingface_id.split('/')[1]}/{dataset.latest_commit_hash}/{file}.csv"
                    try:
                        minio_client.stat_object(
                            settings.MINIO_BUCKET_NAME, minio_file_name
                        )
                        print(
                            f"File {minio_file_name} already exists in MinIO, generating presigned URL."
                        )
                    except Exception:
                        # If the file does not exist in MinIO, download and upload it
                        file_path = hf_api.hf_hub_download(
                            repo_id=dataset.huggingface_id,
                            filename=f"{file}.csv",
                            repo_type="dataset",
                            revision=dataset.latest_commit_hash,
                        )
                        df = pd.read_csv(file_path)
                        df.drop(df.columns[0], axis=1)
                        buffer = io.BytesIO()
                        df.to_csv(buffer, index=False)
                        buffer.seek(0)
                        minio_client.put_object(
                            bucket_name=settings.MINIO_BUCKET_NAME,
                            object_name=minio_file_name,
                            data=buffer,
                            length=buffer.getbuffer().nbytes,
                            content_type="application/csv",
                        )
                    presigned_url = minio_client.presigned_get_object(
                        bucket_name=settings.MINIO_BUCKET_NAME,
                        object_name=minio_file_name,
                    )

                    csv_links[file] = presigned_url

        if format == "csv":
            return {"csv": csv_links}
        elif format == "json":
            return {"json": json_url}
        else:
            return {"csv": csv_links, "json": json_url}


class StatusView(UserIDMixin, APIView):

    def get(self, request):
        workflow_id = request.query_params.get("workflow-id")
        task_id = request.query_params.get("task-id")

        if workflow_id:
            workflow = get_object_or_404(Workflows, workflow_id=workflow_id)
            tasks = workflow.tasks.all()
            tasks_status = []
            for task in tasks:
                percentage = task.generated_samples / task.total_samples * 100
                if percentage > 100:
                    percentage = 100.0
                dataset_links = GetDataView().get_dataset_links(task.dataset, None)
                tasks_status.append(
                    {
                        "task_id": task.id,
                        "status": task.status,
                        "percentage": percentage,
                        "dataset": dataset_links,
                    }
                )

            response_data = {
                "workflow": {
                    "workflow_id": workflow.workflow_id,
                    "workflow_status": workflow.status,
                    "workflow_cost": f"${workflow.cost}",
                },
                "tasks": tasks_status,
            }
            return Response(response_data, status=status.HTTP_200_OK)

        elif task_id:
            task = get_object_or_404(Task, id=task_id)
            workflow = task.workflow
            dataset_links = GetDataView().get_dataset_links(task.dataset, None)

            response_data = {
                "workflow_id": workflow.workflow_id,
                "workflow_status": workflow.status,
                "tasks": [
                    {
                        "task_id": task.id,
                        "status": task.status,
                        "name": task.name,
                        "dataset": dataset_links,
                    }
                ],
            }
            return Response(response_data, status=status.HTTP_200_OK)

        else:
            return Response(
                {"error": "Either workflow_id or task_id must be provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )
