import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from django.core.cache import cache
from django.shortcuts import get_object_or_404
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

from .models import Examples, WorkflowConfig
from .serializers import ExampleSerializer

logger = logging.getLogger(__name__)


def get_workflow_config(workflow_config):
    """
    Fetches a WorkflowConfig object from the cache or database by workflow_config.

    :param workflow_config: The type of the workflow to fetch the config for.
    :return: WorkflowConfig instance
    raises: HTTPError: Http 404 if no workflow config found in db.
    """
    cache_key = f"workflow_config_{workflow_config}"
    config = cache.get(cache_key)

    if config is None:
        get_object_or_404(WorkflowConfig, id=workflow_config)

    return config


def dehydrate_cache(key_pattern):
    """
    Dehydrates (clears) cache entries based on a given key pattern.
    This function can be used to invalidate specific cache entries manually,
    especially after database updates, to ensure cache consistency.

    Parameters:
    - key_pattern (str): The cache key pattern to clear. This can be a specific cache key
      or a pattern representing a group of keys.

    Returns:
    - None
    """
    if hasattr(cache, "delete_pattern"):
        cache.delete_pattern(key_pattern)
    else:
        cache.delete(key_pattern)


def create_pydantic_model(jsonData):
    Model = None
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=os.getcwd(), suffix=".json", delete=True
    ) as tmp_json:
        json.dump(jsonData, tmp_json)
        tmp_json.flush()

        with tempfile.NamedTemporaryFile(
            mode="w+", dir=os.getcwd(), suffix=".py", delete=True
        ) as tmp_py:
            command = (
                f"datamodel-codegen --input {tmp_json.name} --output {tmp_py.name}"
            )

            try:
                subprocess.run(command, check=True, shell=True)
                print("model generation successful")
            except subprocess.CalledProcessError as e:
                print("An error occurred while generating the pydantic model:", e)

            tmp_py_path = Path(tmp_py.name)
            Model = import_model_from_generated_file(tmp_py_path)

            class_string = get_classes_from_module(tmp_py_path, base_class=BaseModel)

    return Model, class_string


def import_model_from_generated_file(file_path):
    directory, module_name = os.path.split(file_path)
    module_name = os.path.splitext(module_name)[0]

    if directory not in sys.path:
        sys.path.append(directory)

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Model = getattr(module, "Model", None)
    return Model


def import_module_from_path(path):
    """Import a module from the given file path"""
    if isinstance(path, str):
        path = Path(path)  # Convert string to Path if necessary
    module_name = path.stem  # Using path.stem to get a module name from the file name
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_classes_from_module(path, base_class):
    module = import_module_from_path(path)
    class_details = ""
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, base_class) and obj.__module__ == module.__name__:
            class_details += f"\nclass {name}({base_class.__name__}):\n"
            for field_name, field_type in obj.__annotations__.items():
                field_type_name = (
                    field_type.__name__
                    if hasattr(field_type, "__name__")
                    else repr(field_type)
                )
                class_details += f"  {field_name}: {field_type_name}\n"

    return class_details


def validate_and_save_examples(examples_data, Model, workflow):
    examples = []
    for example_data in examples_data:
        serializer = ExampleSerializer(data=example_data)

        print(serializer.is_valid())

        if serializer.is_valid():
            example_id = serializer.validated_data.get("example_id", None)
            text = serializer.validated_data["text"]
            label = serializer.validated_data["label"]
            reason = serializer.validated_data["reason"]

            try:
                Model(**text)
            except PydanticValidationError as e:
                logger.error("huh")
                return False, {"error": True, "message": e.errors()}

            if example_id:
                example, created = Examples.objects.get_or_create(
                    example_id=example_id,
                    defaults={
                        "workflow": workflow,
                        "text": text,
                        "label": label,
                        "reason": reason,
                    },
                )

                if not created:
                    example.text = text
                    example.label = label
                    example.reason = reason
                    example.save()
            else:
                example = Examples.objects.create(
                    workflow=workflow,
                    text=text,
                    label=label,
                    reason=reason,
                )

            examples.append(
                {
                    "example_id": str(example.example_id),
                    "text": example.text,
                    "label": example.label,
                    "reason": example.reason,
                }
            )

        else:
            return False, serializer.errors

    return True, examples


# stores the cost per 1000 tokens used in USD
def get_model_cost(model):
    costs = {
        "gpt-4-0125-preview": {"input": 0.0100, "output": 0.0300},
        "gpt-4-1106-preview": {"input": 0.0100, "output": 0.0300},
        "gpt-4-vision-preview": {"input": 0.0100, "output": 0.0300},
        "gpt-3.5-turbo-1106": {"input": 0.0010, "output": 0.0020},
        "gpt-3.5-turbo-0613": {"input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo-16k-0613": {"input": 0.0030, "output": 0.0040},
        "gpt-3.5-turbo-0301": {"input": 0.0015, "output": 0.0020},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo": {"input": 0.0030, "output": 0.0060},
    }

    return costs[model]


def get_task_config(task=None):
    task_config = {
        "text_classification": {
            "model": "BERT",
            "task": "text_classification",
            "labelStudioElement": {
                "name": "Text Classification",
                "config": {
                    "choices": [
                        "pest",
                        "agricultural_scheme",
                        "agriculture",
                        "seed",
                        "weather",
                        "price",
                        "non_agri",
                    ]
                },
            },
            "telemetryDataField": {"input": "query", "output": None},
        },
        "ner": {
            "model": "distilbert-finetuned",
            "task": "NER",
            "labelStudioElement": {
                "name": "Named Entity Recognition",
                "config": {
                    "labels": [
                        {"name": "pest", "value": "pest"},
                        {"name": "crop", "value": "crop"},
                        {"name": "seed_type", "value": "seed_type"},
                        {"name": "email", "value": "email"},
                        {"name": "phone_number", "value": "phone_number"},
                        {"name": "time", "value": "time"},
                        {"name": "date", "value": "date"},
                    ]
                },
            },
            "telemetryDataField": {"input": "query", "output": "NER"},
        },
        "neural_coreference": {
            "model": "FCoref",
            "task": "Neural Coreference",
            "labelStudioElement": {
                "name": "Translation",
                "config": {
                    "leftHeader": "Read the previous conversation",
                    "rightHeader": "Provide coreferenced text",
                    "leftTextAreaName": "Previous conversation",
                    "rightTextAreaName": "Coreferenced text",
                },
            },
            "telemetryDataField": {"input": "query", "output": "coreferencedText"},
        },
    }

    keys = list(task_config.keys())
    res = []
    for key in keys:
        res.append(task_config[key])

    if task:
        if task in task_config:
            return [task_config[task]]
        else:
            return None
    else:
        return res


# to get the mapping between the dataset columns and the input columns
# task:{input_column in db: dataset_column,output_column in db: dataset_column}
def get_task_mapping(task):
    mapping = {
        "text_classification": {"input_string": "text", "output_string": "class"},
        "ner": {"input_string": "Input", "output_string": "Output"},
    }
    if task in mapping:
        return mapping[task]
    else:
        return None


def paginate_queryset(queryset, page, page_size):
    total_count = queryset.count()
    total_pages = (total_count + page_size - 1) // page_size

    if page > total_pages or page < 1:
        return [], total_count, total_pages

    start = (page - 1) * page_size
    end = start + page_size
    return queryset[start:end], total_count, total_pages
