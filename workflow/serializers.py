from jsonschema import Validator
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from .models import Dataset, Examples, MLModel, Prompt, User, WorkflowConfig, Workflows


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("user_id", "user_name")


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = ("id", "name", "is_locally_cached")


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ("id", "name", "is_locally_cached")


class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = (
            "id",
            "created_at",
            "updated_at",
            "user_prompt",
            "system_prompt",
            "source",
            "workflow",
        )


class ExampleSerializer(serializers.ModelSerializer):
    example_id = serializers.UUIDField(required=False)
    prompt = PromptSerializer(read_only=True)
    text = serializers.JSONField(required=True)
    label = serializers.CharField(required=True)
    reason = serializers.CharField(required=True)

    class Meta:
        model = Examples
        fields = ("example_id", "prompt", "text", "label", "reason")


class WorkflowDetailSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    model = MLModelSerializer(read_only=True)
    dataset = DatasetSerializer(read_only=True)
    latest_prompt = PromptSerializer(read_only=True)
    prompts = PromptSerializer(many=True, read_only=True)
    examples = ExampleSerializer(many=True, read_only=True)
    workflow_config = serializers.PrimaryKeyRelatedField(
        queryset=WorkflowConfig.objects.all()
    )

    class Meta:
        model = Workflows
        fields = (
            "workflow_id",
            "workflow_name",
            "total_examples",
            "split",
            "llm_model",
            "cost",
            "estimated_dataset_cost",
            "tags",
            "user",
            "dataset",
            "model",
            "examples",
            "latest_prompt",  # Ensures details of the latest prompt are shown
            "prompts",  # Lists all associated prompts
            "workflow_config",
        )


class WorkflowSerializer(serializers.ModelSerializer):
    examples = ExampleSerializer(many=True, required=False)
    workflow_config = serializers.PrimaryKeyRelatedField(
        queryset=WorkflowConfig.objects.all()
    )

    class Meta:
        model = Workflows
        fields = (
            "workflow_id",
            "workflow_name",
            "total_examples",
            "split",
            "llm_model",
            "tags",
            "user",
            "examples",
            "latest_prompt",
            "workflow_config",
        )
        extra_kwargs = {
            "cost": {"default": 0},
            "estimated_dataset_cost": {"default": "NULL till first iteration"},
        }

    def create(self, validated_data):
        examples_data = validated_data.pop("examples", [])
        workflow = Workflows.objects.create(**validated_data)

        for example_data in examples_data:
            Examples.objects.create(workflow=workflow, **example_data)

        return workflow


class WorkflowConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowConfig
        fields = (
            "id",
            "name",
            "system_prompt",
            "user_prompt_template",
            "schema_example",
            "temperature",
            "fields",
            "model_string",
        )
