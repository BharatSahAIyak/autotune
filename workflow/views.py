import ast
import io
import logging
from decimal import Decimal, getcontext

import pandas as pd
from django.core.exceptions import FieldDoesNotExist
from django.db import transaction
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, UpdateAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from workflow.generator.dataFetcher import DataFetcher
from workflow.generator.generate import process_task
from workflow.health import HealthCheck
from workflow.training.deploy import deploy_model
from workflow.training.train import train

from .align_tasks import align_task
from .mixins import CacheDatasetMixin, CreateMLBaseMixin, UserIDMixin
from .models import (
    Dataset,
    DatasetData,
    Examples,
    MLModel,
    MLModelConfig,
    Prompt,
    Task,
    User,
    WorkflowConfig,
    Workflows,
)
from .serializers import (
    AudioDatasetSerializer,
    DatasetDataSerializer,
    ExampleSerializer,
    MLModelSerializer,
    ModelDataSerializer,
    ModelDeploySerializer,
    PromptSerializer,
    UserSerializer,
    WorkflowConfigSerializer,
    WorkflowDetailSerializer,
    WorkflowSerializer,
)
from .utils import (
    create_pydantic_model,
    dehydrate_cache,
    get_model_cost,
    get_task_config,
    get_task_mapping,
    paginate_queryset,
    validate_and_save_examples,
)

logger = logging.getLogger(__name__)


def index():
    return HttpResponse("Hello, world. You're at the workflow index.")


class CreateWorkflowView(UserIDMixin, APIView):

    @swagger_auto_schema(
        operation_description="Create a new workflow and associated prompt",
        request_body=WorkflowSerializer,
        responses={
            201: openapi.Response(
                description="Workflow and prompt created successfully",
                schema=WorkflowSerializer,
            ),
            400: openapi.Response(description="Invalid data for workflow or prompt"),
        },
    )
    def post(self, request):
        with transaction.atomic():
            user: User = request.META["user"]

            workflow_data = request.data.get("workflow")
            workflow_data["user"] = user.user_id
            workflow_serializer = WorkflowSerializer(
                data=request.data.get("workflow", {})
            )
            if workflow_serializer.is_valid(raise_exception=True):
                workflow = workflow_serializer.save()

                prompt_data = {
                    "user_prompt": request.data.get("user_prompt", ""),
                    "workflow": workflow.pk,
                }

                prompt_serializer = PromptSerializer(data=prompt_data)
                if prompt_serializer.is_valid(raise_exception=True):
                    prompt_serializer.save()

                    return Response(
                        {
                            "workflow": workflow_serializer.data,
                            "prompt": prompt_serializer.data,
                        },
                        status=status.HTTP_201_CREATED,
                    )

        return Response(
            {
                "error": "Invalid data for workflow or prompt",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )


@method_decorator(csrf_exempt, name="dispatch")
class IterateWorkflowView(UserIDMixin, APIView):
    def post(self, request, workflow_id, *args, **kwargs):
        user_id = request.META["user"].user_id

        workflow = get_object_or_404(
            Workflows, workflow_id=workflow_id, user_id=user_id
        )
        workflow.status = "ITERATION"
        workflow.save()
        examples_data = request.data.get("examples", [])

        examples_exist = (
            Examples.objects.filter(
                workflow_id=workflow_id, label__isnull=False
            ).exists()
            or len(examples_data) > 0
        )

        Model, _ = create_pydantic_model(workflow.workflow_config.schema_example)

        success, result = validate_and_save_examples(examples_data, Model, workflow)

        if not success:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        user_prompt = request.data.get("user_prompt")
        if user_prompt:
            Prompt.objects.create(user_prompt=user_prompt, workflow=workflow)

        total_examples = request.data.get("total_examples", 10)
        max_iterations = request.data.get("max_iterations", 50)
        max_concurrent_fetches = request.data.get("max_concurrent_fetches", 100)
        batch_size = request.data.get("batch_size", 5)

        fetcher = DataFetcher(
            max_iterations=int(max_iterations),
            max_concurrent_fetches=int(max_concurrent_fetches),
            batch_size=int(batch_size),
        )
        prompt: Prompt = workflow.latest_prompt
        fetcher.generate_or_refine(
            workflow_id=workflow.workflow_id,
            total_examples=total_examples,
            workflow_config_id=workflow.workflow_config.id,
            llm_model=workflow.llm_model,
            Model=Model,
            prompt=prompt.user_prompt,
            prompt_id=prompt.id,
            refine=examples_exist,
            iteration=1,
        )

        costs = get_model_cost(workflow.llm_model)

        getcontext().prec = 6

        input_cost = Decimal(fetcher.input_tokens * costs["input"]) / Decimal(1000)
        output_cost = Decimal(fetcher.output_tokens * costs["output"]) / Decimal(1000)

        iteration_cost = input_cost + output_cost
        iteration_cost = iteration_cost.quantize(Decimal("0.0001"))
        workflow.cost += iteration_cost
        workflow.cost = workflow.cost.quantize(Decimal("0.0001"))

        total_batches = max(
            1,
            (workflow.total_examples + batch_size - 1) // batch_size,
        )

        workflow.estimated_dataset_cost = Decimal(
            Decimal(1.25) * iteration_cost * total_batches
        )

        workflow.estimated_dataset_cost = workflow.estimated_dataset_cost.quantize(
            Decimal("0.0001")
        )

        workflow.status = "IDLE"
        workflow.save()
        return Response(
            {
                "workflow_cost": f"${workflow.cost}",
                "iteration_cost": f"${iteration_cost}",
                "estimated_dataset_cost": f"${workflow.estimated_dataset_cost}",
                "data": fetcher.examples,
            }
        )


class WorkflowListView(APIView):

    @swagger_auto_schema(
        operation_description="Retrieve a list of workflows",
        responses={200: WorkflowDetailSerializer(many=True)},
    )
    def get(self, request, *args, **kwargs):
        workflows = Workflows.objects.all()
        serializer = WorkflowDetailSerializer(workflows, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        operation_description="Create a new workflow",
        request_body=WorkflowSerializer,
        responses={201: WorkflowSerializer(), 400: "Invalid data"},
    )
    def post(self, request, *args, **kwargs):
        serializer = WorkflowSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SingleWorkflowView(APIView):

    @swagger_auto_schema(
        operation_description="Retrieve a specific workflow by ID",
        responses={200: WorkflowDetailSerializer()},
    )
    def get(self, request, workflow_id, *args, **kwargs):
        workflow = get_object_or_404(Workflows, workflow_id=workflow_id)
        serializer = WorkflowDetailSerializer(workflow)
        return Response(serializer.data)

    @swagger_auto_schema(
        operation_description="Update a specific workflow by ID",
        request_body=WorkflowSerializer,
        responses={200: WorkflowSerializer(), 400: "Invalid data"},
    )
    def put(self, request, workflow_id, *args, **kwargs):
        workflow = get_object_or_404(Workflows, workflow_id=workflow_id)
        serializer = WorkflowSerializer(workflow, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Delete a specific workflow by ID", responses={204: None}
    )
    def delete(self, request, workflow_id, *args, **kwargs):
        workflow = get_object_or_404(Workflows, workflow_id=workflow_id)
        workflow.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class PromptViewSet(APIView):

    @swagger_auto_schema(
        operation_description="Retrieve all prompts associated with a specific workflow",
        responses={200: PromptSerializer(many=True)},
    )
    def get(self, request, workflow_id):
        workflow = get_object_or_404(Workflows, pk=workflow_id)
        prompts = (
            workflow.prompts.all()
        )  # Get all prompts associated with this workflow
        return Response(PromptSerializer(prompts, many=True).data)

    @swagger_auto_schema(
        operation_description="Create a new prompt in a specific workflow",
        request_body=PromptSerializer,
        responses={201: PromptSerializer(), 400: "Invalid data"},
    )
    def post(self, request, workflow_id):
        workflow = get_object_or_404(Workflows, pk=workflow_id)
        if not request.data.get("user_prompt"):
            return Response(
                {"message": "user_prompt is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        prompt_data = {
            "user_prompt": request.data.get("user_prompt"),
            "workflow": workflow.pk,
        }
        serializer = PromptSerializer(data=prompt_data)
        if serializer.is_valid():
            prompt = serializer.save(workflow=workflow)

            # Update the latest_prompt field on the workflow to this new prompt
            workflow.latest_prompt = prompt
            workflow.save()

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ExamplesView(APIView):

    @swagger_auto_schema(
        operation_description="Retrieve examples; filter by workflow if ID is provided.",
        responses={200: ExampleSerializer(many=True)},
    )
    def get(self, request, workflow_id=None):
        if workflow_id:
            examples = Examples.objects.filter(
                workflow_id=workflow_id, task_id__isnull=True
            )
        else:
            examples = Examples.objects.all()

        serialized_examples = ExampleSerializer(examples, many=True)
        return Response(serialized_examples.data, status=status.HTTP_200_OK)

    @swagger_auto_schema(
        operation_description="Post examples for a specific workflow.",
        request_body=ExampleSerializer(many=True),
        responses={201: "Examples updated successfully", 400: "Invalid data"},
    )
    def post(self, request, workflow_id):
        workflow = get_object_or_404(Workflows, pk=workflow_id)
        examples_data = request.data.get("examples", [])

        Model, _ = create_pydantic_model(workflow.workflow_config.schema_example)

        success, result = validate_and_save_examples(examples_data, Model, workflow)

        if not success:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)

        return Response({"message": "Examples updated successfully"}, status=201)


class WorkflowUpdateView(UpdateAPIView):
    queryset = Workflows.objects.all()
    serializer_class = WorkflowSerializer
    lookup_field = "workflow_id"

    @swagger_auto_schema(
        operation_description="Update a specific workflow by ID",
        request_body=WorkflowSerializer,
        responses={200: WorkflowSerializer(), 400: "Invalid data"},
    )
    def put(self, request, *args, **kwargs):
        return super().put(request, *args, **kwargs)


class WorkflowDuplicateView(APIView):

    @swagger_auto_schema(
        operation_description="Duplicate a workflow by ID",
        responses={201: WorkflowSerializer(), 404: "Not Found"},
    )
    def put(self, request, workflow_id):
        workflow = get_object_or_404(Workflows, workflow_id=workflow_id)
        workflow.pk = None
        workflow.save()
        serializer = WorkflowSerializer(workflow)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class WorkflowStatusView(APIView):

    @swagger_auto_schema(
        operation_description="Retrieve the status of a workflow by ID",
        responses={200: "Status of the workflow", 404: "Not Found"},
    )
    def get(self, request, workflow_id):
        workflow = get_object_or_404(Workflows, workflow_id=workflow_id)
        return Response({"status": workflow.status})


class WorkflowSearchView(ListAPIView):
    serializer_class = WorkflowSerializer

    @swagger_auto_schema(
        operation_description="Search for workflows by tags",
        manual_parameters=[
            openapi.Parameter(
                "tags",
                openapi.IN_QUERY,
                description="Comma-separated list of tags to filter workflows",
                type=openapi.TYPE_STRING,
            )
        ],
        responses={200: WorkflowSerializer(many=True)},
    )
    def get_queryset(self):
        tags_param = self.request.query_params.get("tags", "")
        tags_query = tags_param.split(",") if tags_param else []
        query = Q(tags__overlap=tags_query) if tags_query else Q()
        return Workflows.objects.filter(query)


class TaskView(APIView):

    @swagger_auto_schema(
        operation_description="Retrieve the status and progress of a task by ID",
        responses={
            200: "A JSON object with the task status and optional progress percentage",
            404: "Not Found",
        },
    )
    def get(self, request, task_id):
        task = get_object_or_404(Task, pk=task_id)
        response_data = {"status": task.status}

        if not task.name.strip().startswith("Training Workflow"):
            percentage = task.generated_samples / task.total_samples
            response_data["percentage"] = percentage

        return Response(response_data)


@method_decorator(csrf_exempt, name="dispatch")
class GenerateTaskView(UserIDMixin, APIView):

    @swagger_auto_schema(
        operation_description="Create a new task for the specified workflow.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "prompts": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="A list of prompts as a string representation of a list (JSON format).",
                ),
                "file": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Items(type=openapi.TYPE_STRING, format="binary"),
                    description="List of CSV files containing prompts. Picks the prompts column from the CSV file.",
                ),
                "total_examples": openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="The total number of examples to generate.",
                ),
                "batch_size": openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="The batch size for processing.",
                ),
                "example_per_prompt": openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Number of examples per prompt. REQUIRED when files are uploaded or prompts are provided.",
                ),
                "max_iterations": openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Maximum number of iterations for the task.",
                    default=100,
                ),
                "max_concurrent_fetches": openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="Maximum number of concurrent fetches.",
                    default=100,
                ),
            },
        ),
        responses={
            202: openapi.Response(description="Task creation initiated"),
            400: openapi.Response(description="Invalid data"),
        },
    )
    def post(self, request, workflow_id, *args, **kwargs):
        user_id = request.META["user"].user_id

        workflow = get_object_or_404(
            Workflows, workflow_id=workflow_id, user_id=user_id
        )

        uploaded_files = request.FILES.getlist("file")
        prompts_str = request.data.get("prompts")

        prompts = []
        total_examples = request.data.get("total_examples")
        batch_size = request.data.get("batch_size")

        if uploaded_files and prompts_str:
            return JsonResponse(
                {
                    "error": "Both prompts and file uploads are not allowed. Please provide either prompts or a file."
                },
                status=400,
            )
        if len(uploaded_files) > 0:
            if not request.data.get("example_per_prompt"):
                return JsonResponse(
                    {"error": "example_per_prompt is required when uploading files."},
                    status=400,
                )
            total_examples = request.data.get("example_per_prompt")
            batch_size = total_examples

            series = []
            for file in uploaded_files:
                if not file.name.lower().endswith(".csv"):
                    return JsonResponse(
                        {
                            "error": f"Invalid file extension for {file.name}. Only .csv files are allowed."
                        },
                        status=400,
                    )

                try:
                    csv_file = io.BytesIO(file.read())
                    df = pd.read_csv(csv_file)
                    if "prompts" in df:
                        series.append(df["prompts"])
                    else:
                        return JsonResponse(
                            {"error": " `prompts` column not found in the CSV file"},
                            status=400,
                        )
                except Exception as e:
                    return JsonResponse(
                        {"error": f"Error reading CSV file: {str(e)}"}, status=400
                    )
            prompts = pd.concat(series, ignore_index=True).tolist()

        if prompts_str:
            if not request.data.get("example_per_prompt"):
                return JsonResponse(
                    {"error": "example_per_prompt is required when providing prompts."},
                    status=400,
                )
            total_examples = request.data.get("example_per_prompt")
            batch_size = total_examples

            try:
                prompts = ast.literal_eval(prompts_str)
                if not isinstance(prompts, list):
                    raise ValueError("Prompts is not a valid list.")
            except (ValueError, SyntaxError) as e:
                return JsonResponse(
                    {"error": f"Invalid prompts format: {str(e)}"}, status=400
                )

        if total_examples:
            workflow.total_examples = total_examples
            workflow.save()

        max_iterations = request.data.get("max_iterations", 50)
        max_concurrent_fetches = request.data.get("max_concurrent_fetches", 100)
        batch_size = batch_size if batch_size else 5

        task = Task.objects.create(
            name=f"Batch Task for Workflow {workflow_id}",
            status="STARTING",
            workflow=workflow,
        )

        process_task.delay(
            task.id,
            int(max_iterations),
            int(max_concurrent_fetches),
            int(batch_size),
            prompts,
        )

        estimated_cost = workflow.estimated_dataset_cost

        if estimated_cost == None:
            estimated_cost = "Not available without iterations being completed"

        return JsonResponse(
            {
                "message": "Tasks creation initiated",
                "task_id": task.id,
                "workflow_id": workflow.workflow_id,
                "expeced_cost": estimated_cost,
            },
            status=202,
        )


@api_view(["GET"])
def dehydrate_cache_view(request, key_pattern):
    """
    A simple view to dehydrate cache entries based on a key pattern.
    """
    dehydrate_cache(key_pattern)
    return JsonResponse(
        {"status": "success", "message": "Cache dehydrated successfully."}
    )


class WorkflowConfigView(APIView):
    """
    Class-based view for managing WorkflowConfig.
    """

    @swagger_auto_schema(
        operation_description="Retrieve all WorkflowConfig objects",
        responses={200: WorkflowConfigSerializer(many=True)},
    )
    def get(self, request):
        """
        Retrieve all WorkflowConfig objects.
        """
        configs = WorkflowConfig.objects.all()
        serializer = WorkflowConfigSerializer(configs, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        operation_description="Create a new WorkflowConfig",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=["schema_example"],
            properties={
                "schema_example": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="The example schema for the workflow configuration",
                )
            },
        ),
        responses={
            201: openapi.Response(
                description="Workflow config created successfully",
                schema=WorkflowConfigSerializer,
            ),
            400: "Invalid data",
        },
    )
    def post(self, request):
        """
        Create a new WorkflowConfig.
        """
        if request.data.get("schema_example") is None:
            return Response(
                {"message": "Schema Example is required!"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        Model, model_string = create_pydantic_model(request.data.get("schema_example"))
        field_names = list(Model.__fields__.keys())
        field_info = list(Model.__fields__.values())

        fields = []

        for i in range(len(field_names)):
            fields.append({field_names[i]: field_info[i].annotation.__name__})

        data = request.data

        data["model_string"] = model_string
        data["fields"] = fields

        serializer = WorkflowConfigSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {
                    "message": "Workflow config created successfully!",
                    "config": serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Update an existing WorkflowConfig based on its ID",
        request_body=WorkflowConfigSerializer,
        responses={
            200: WorkflowConfigSerializer,
            400: "Invalid data",
            404: "Not found",
        },
    )
    def patch(self, request, config_id):
        """
        Update an existing WorkflowConfig based on its ID.
        """
        config = get_object_or_404(WorkflowConfig, id=config_id)
        serializer = WorkflowConfigSerializer(config, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Delete a WorkflowConfig based on its ID",
        responses={204: "Workflow config deleted successfully", 404: "Not found"},
    )
    def delete(self, request, config_id):
        """
        Delete a WorkflowConfig based on its ID.
        """
        config = get_object_or_404(WorkflowConfig, id=config_id)
        config.delete()
        return Response(
            {"message": "Workflow config deleted successfully!"},
            status=status.HTTP_204_NO_CONTENT,
        )


@api_view(["POST"])
def add_user(request):
    serializer = UserSerializer(data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response(
            {"message": "User created successfully!", "user": serializer.data},
            status=status.HTTP_201_CREATED,
        )
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TrainModelView(UserIDMixin, CreateMLBaseMixin, CacheDatasetMixin, APIView):

    @swagger_auto_schema(
        operation_description="Start training a model with the provided data.",
        request_body=ModelDataSerializer,
        responses={
            202: openapi.Response(description="Training task initiated successfully."),
            400: openapi.Response(description="Invalid data"),
        },
    )
    def post(self, request, *args, **kwargs):
        serializer = ModelDataSerializer(data=request.data)
        user_id = request.META["user"].user_id

        if serializer.is_valid():
            data = serializer.validated_data
            logger.info(f"Training model with data: {data}")
            workflow_id = request.META["workflow_id"]

            training_task = request.data.get("task_type")

            task = Task.objects.create(
                name=f"Training Workflow {workflow_id}",
                status="STARTING",
                workflow_id=workflow_id,
            )

            cached_dataset_id = request.META.get("cached_dataset_id", None)

            train.apply_async(
                args=[data, user_id, training_task, cached_dataset_id],
                task_id=str(task.id),
            )

            return Response(
                {"workflow_id": request.META["workflow_id"], "task_id": task.id},
                status=status.HTTP_202_ACCEPTED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MLModelListView(UserIDMixin, CreateMLBaseMixin, APIView):

    def get(self, request, format=None):
        user_id = request.META["user"].user_id
        models = MLModel.objects.filter(user_id=user_id)
        serializer = MLModelSerializer(models, many=True)
        return Response(serializer.data)


class MLModelDetailView(APIView):
    @swagger_auto_schema(
        operation_description="Retrieve a list of all ML models.",
        responses={200: MLModelSerializer(many=True)},
    )
    def get(self, request, model_id, format=None):
        try:
            model = MLModel.objects.get(id=model_id)
            serializer = MLModelSerializer(model)
            return Response(serializer.data)
        except MLModel.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)


class DatasetView(UserIDMixin, CreateMLBaseMixin, CacheDatasetMixin, APIView):

    @swagger_auto_schema(
        operation_description="Fetches CSV files from a Hugging Face dataset repository, with pagination and optional file-specific fetching.",
        manual_parameters=[
            openapi.Parameter(
                "page",
                openapi.IN_QUERY,
                description="The page number for the dataset.",
                type=openapi.TYPE_INTEGER,
                default=1,
            ),
            openapi.Parameter(
                "perPage",
                openapi.IN_QUERY,
                description="The number of records per page.",
                type=openapi.TYPE_INTEGER,
                default=10,
            ),
            openapi.Parameter(
                "file",
                openapi.IN_QUERY,
                description="Specific file name to fetch from the dataset.",
                type=openapi.TYPE_STRING,
                required=False,
            ),
        ],
        responses={
            200: openapi.Response(
                description="Data retrieved successfully.",
                schema=DatasetDataSerializer(many=True),
            ),
            204: openapi.Response(description="No content."),
            400: openapi.Response(description="Invalid data."),
        },
    )
    def get(self, request):
        """
        Fetches CSV files from a Hugging Face dataset repository, with pagination and optional file-specific fetching.

        If 'dataset' is provided, downloads CSV files from the specified Hugging Face dataset (format: {username}/{dataset_name}).
        If 'file' is provided, only this file's data is returned.

        If no Hugging Face dataset is provided, then the dataset generated at autotune is returned, and if no dataset is available,
        HTTP Status No Content is returned.

        If order is provided and no field, then bad request is returned.
        order- should be either 'asc' or 'desc'

        Parameters:
         - workflow_id(UUID): The ID of the workflow for which the dataset is to be fetched.
         - page(int): The page number for the dataset - Optional.
         - page_size(int): The number of records per page - Optional.
         - dataset(str): The Hugging Face dataset to be fetched - Optional.
         - file(str): Specific file name to fetch from the dataset - Optional.
         - field(str): field on which we want to sort the data - Optional.
         - order(str): order in which we want to sort the data - Optional.
        """
        page = request.query_params.get("page", 1)
        page_size = request.query_params.get("perPage", 10)
        file = request.query_params.get("file", None)
        field = request.query_params.get("field", None)
        order = request.query_params.get("order", None)

        # convert order to lowercase
        if order:
            order = order.lower()
            if order not in ["asc", "desc"]:
                return Response(
                    {
                        "error": "Order should be either 'asc' or 'desc'.",
                        "workflow_id": request.META.get("workflow_id"),
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if order and not field:
            return Response(
                {
                    "error": "Field is required when order is provided.",
                    "workflow_id": request.META.get("workflow_id"),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if field:
            # Check if the field is valid for DatasetData model
            try:
                DatasetData._meta.get_field(field)
            except FieldDoesNotExist:
                allowed_fields = [f.name for f in DatasetData._meta.get_fields()]
                return Response(
                    {
                        "error": f"Invalid field: {field}. Allowed fields are: {', '.join(allowed_fields)}",
                        "workflow_id": request.META.get("workflow_id"),
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        try:
            page = int(page)
            page_size = int(page_size)
        except ValueError:
            return Response(
                {
                    "error": "Page and page size must be integers.",
                    "workflow_id": request.META.get("workflow_id"),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        cached_dataset_id = request.META.get("cached_dataset_id")
        data = DatasetData.objects.filter(dataset_id=cached_dataset_id)

        if file:
            data = data.filter(file=file)

        if order and field:
            if order == "asc":
                data = data.order_by(field)
            elif order == "desc":
                data = data.order_by(f"-{field}")

        paginated_data, total_count, total_pages = paginate_queryset(
            data, page, page_size
        )

        if not paginated_data:
            return Response(
                {
                    "workflow_id": request.META.get("workflow_id"),
                    "pagination": {
                        "page": page,
                        "perPage": page_size,
                        "totalPages": total_pages,
                        "totalCount": total_count,
                    },
                    "data": [],
                },
                status=status.HTTP_204_NO_CONTENT,
            )

        serializer = DatasetDataSerializer(paginated_data, many=True)
        return Response(
            {
                "workflow_id": request.META.get("workflow_id"),
                "pagination": {
                    "page": page,
                    "perPage": page_size,
                    "totalPages": total_pages,
                    "totalCount": total_count,
                },
                "data": serializer.data,
            },
            status=status.HTTP_200_OK,
        )

    @swagger_auto_schema(
        operation_description="Fetches CSV files from a Hugging Face dataset repository, with pagination and optional file-specific fetching.",
        manual_parameters=[
            openapi.Parameter(
                "page",
                openapi.IN_QUERY,
                description="The page number for the dataset.",
                type=openapi.TYPE_INTEGER,
                default=1,
            ),
            openapi.Parameter(
                "perPage",
                openapi.IN_QUERY,
                description="The number of records per page.",
                type=openapi.TYPE_INTEGER,
                default=10,
            ),
            openapi.Parameter(
                "file",
                openapi.IN_QUERY,
                description="Specific file name to fetch from the dataset.",
                type=openapi.TYPE_STRING,
                required=False,
            ),
        ],
        responses={
            200: openapi.Response(
                description="Data retrieved successfully.",
                schema=DatasetDataSerializer(many=True),
            ),
            204: openapi.Response(description="No content."),
            400: openapi.Response(description="Invalid data."),
        },
    )
    def post(self, request):
        """
        Gets a dataset from huggingface, and stores it in the local cache if not already not locally cached, and stores any changes in the dataset till it is committed
        to HF just before training is triggered

        Parameters:
        - workflow_id(UUID): The ID of the workflow for which the dataset is to be fetched.
        Body:
        - dataset(str): The Hugging Face dataset. If this is not provided, then fall back to the workflow dataset- Optional.
        """
        data = request.data.get("data", None)
        user = request.META["user"]

        # each value of data should be a JSON with input and output
        if not data:
            return Response(
                {"error": "Data is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        for d in data:
            if not d.get("input") or not d.get("output"):
                return Response(
                    {"error": "Input and Output are required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # will be a valid dataset id, handled in the mixin
        cached_dataset_id = request.META.get("cached_dataset_id")
        dataset_object = Dataset.objects.get(id=cached_dataset_id)
        task = dataset_object.type
        task_mapping = get_task_mapping(task)
        keys = list(task_mapping.keys())

        dataset_data_to_create = []
        for d in data:
            record_data = DatasetData(
                dataset=dataset_object, file="train.csv", user=user
            )
            setattr(record_data, keys[0], d.get("input"))
            setattr(record_data, keys[1], d.get("output"))
            dataset_data_to_create.append(record_data)
        DatasetData.objects.bulk_create(dataset_data_to_create)

        return Response(
            {
                "message": "Dataset data saved successfully.",
                "workflow_id": request.META.get("workflow_id"),
                "dataset_id": cached_dataset_id,
            },
            status=status.HTTP_201_CREATED,
        )


class ConfigView(APIView):
    @swagger_auto_schema(
        operation_description="Returns the config of all the tasks or a specific task if provided.",
        manual_parameters=[
            openapi.Parameter(
                "task",
                openapi.IN_QUERY,
                description="Task to get the config for",
                type=openapi.TYPE_STRING,
                required=False,
            )
        ],
        responses={
            200: openapi.Response(
                description="Configuration data retrieved successfully."
            ),
            404: openapi.Response(description="Task not found."),
            400: openapi.Response(description="Invalid request."),
        },
    )
    def get(self, request):
        """
        Returns the config of all the tasks or a specific task if provided.

        Args:
            task: task to get the config for -OPTIONAL

        Returns:
            Array of the configs for all the tasks or a single task in an array
        """
        task = request.query_params.get("task", None)
        if task is None:
            return Response({"data": get_task_config()}, status=status.HTTP_200_OK)
        else:
            task_mapping = get_task_mapping(task)
            if task_mapping:
                return Response({"data": task_mapping}, status=status.HTTP_200_OK)
            else:
                return Response(
                    {"error": "Task not found"}, status=status.HTTP_400_BAD_REQUEST
                )


class ModelDeployView(UserIDMixin, APIView):
    def post(self, request):
        serializer = ModelDeploySerializer(data=request.data)

        if serializer.is_valid():
            data = serializer.data

            logger.info(f"data: {serializer.data}")
            logger.info(f"Deploying model with data: {data['finetuned_model']}")

            deploy_model.apply_async(
                args=[data],
            )

            return Response(
                status=status.HTTP_202_ACCEPTED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ForceAlignmentView(UserIDMixin, CacheDatasetMixin, APIView):

    def post(self, request, *args, **kwargs):

        serializer = AudioDatasetSerializer(data=request.data)

        if serializer.is_valid():
            data = serializer.validated_data
            logger.info(f"Force-Aligning with data: {data}")
            workflow_id = request.data["workflow_id"]

            task = Task.objects.create(
                name=f"Force Alignment Workflow {workflow_id}",
                status="STARTING",
                workflow_id=workflow_id,
            )
            if data["transcript_available"]:
                align_task.apply_async(
                    args=[data],
                    task_id=str(task.id),
                )
            else:
                # TODO: provide data to asr pipeline
                pass

            return Response(
                {"workflow_id": request.data["workflow_id"], "task_id": task.id},
                status=status.HTTP_202_ACCEPTED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class HealthCheckView(APIView):

    def get(self, request):

        health_checker = HealthCheck()

        services = [
            health_checker.openai(),
            health_checker.redis(),
            health_checker.celery_workers(),
            health_checker.postgres(),
            health_checker.huggingface(),
            health_checker.minio(),
        ]

        all_services_healthy = all(
            service["status"]["isAvailable"] for service in services
        )
        response = {
            "status": "ok" if all_services_healthy else "unhealthy",
            "upstreamServices": services,
        }
        return Response(response)


class PingCheckView(APIView):
    def get(self, request):
        resp = {"status": "ok", "details": {"autotune": {"status": "up"}}}
        return Response(resp, status=status.HTTP_200_OK)


from workflow.generator.generator_model import ModelDataFetcher


class ModelIterationView(UserIDMixin, CreateMLBaseMixin, APIView):
    """
    Custom implementation of the iteration logic suited for samples during model training
    """

    def post(self, request, *args, **kwargs):
        user_id = request.META["user"].user_id

        task_type = request.data.get("task_type")
        dataset = request.data.get("dataset")
        input = request.data.get("input")
        output = request.data.get("output")

        model: MLModel = get_object_or_404(
            MLModel, user_id=user_id, task=task_type, config__dataset_path=dataset
        )

        model_config: MLModelConfig = model.config

        pydantic_model, _ = create_pydantic_model(model_config.schema_example)

        data_fetcher = ModelDataFetcher(model_config, model, pydantic_model)

        data_fetcher.generate_or_refine(
            input=input,
            output=output,
            task_type=task_type,
        )

        return Response(data_fetcher.examples)
