from jsonschema import Validator
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from .models import (
    Dataset,
    DatasetData,
    Examples,
    MLModel,
    Prompt,
    User,
    WorkflowConfig,
    Workflows,
)


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


class ModelDataSerializer(serializers.Serializer):
    dataset = serializers.CharField(max_length=255, required=False, allow_blank=True)
    model = serializers.CharField(
        max_length=255
    )  # TODO: needs to be a valid model on huggingface
    model_id = serializers.UUIDField(
        required=False, allow_null=True
    )  # for an existing model
    epochs = serializers.FloatField(required=False, default=1)
    save_path = serializers.CharField(max_length=255)
    task_type = serializers.ChoiceField(
        choices=["text_classification", "seq2seq", "embedding"]
    )
    version = serializers.CharField(max_length=50, required=False, default="main")
    workflow_id = serializers.UUIDField(required=False, allow_null=True)
    args = serializers.JSONField(required=False, default={}, allow_null=True)

    def validate(self, data):
        # TODO: needs to be a valid dataset on huggingface
        # TODO: Add validation for model_id
        dataset = data.get("dataset")
        workflow_id = data.get("workflow_id")

        if not dataset and not workflow_id:
            raise serializers.ValidationError(
                "Either dataset or workflow_id must be provided."
            )

        if not dataset and workflow_id:
            workflow_dataset = Dataset.objects.filter(workflow_id=workflow_id).first()
            if not workflow_dataset:
                raise serializers.ValidationError(
                    "No dataset associated with the provided workflow_id."
                )
            data["dataset"] = workflow_dataset.name

        return data


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = (
            "__all__"  # You can list fields individually if you want to exclude some
        )


class DatasetDataSerializer(serializers.ModelSerializer):
    dataset_id = serializers.SerializerMethodField()

    class Meta:
        model = DatasetData
        fields = [
            "created_at",
            "updated_at",
            "id",
            "file",
            "input_string",
            "output_string",
            "input_json",
            "output_json",
            "dataset_id",
        ]

    def get_dataset_id(self, obj):
        return obj.dataset_id

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        return {
            key: value for key, value in representation.items() if value is not None
        }

class AudioDatasetSerializer(serializers.Serializer):
    dataset = serializers.CharField(max_length=255,required=False, allow_blank=True)
    workflow_id = serializers.UUIDField(required=False, allow_null=True)
    save_path = serializers.CharField(max_length=255)
    transcript_available=serializers.CharField(max_length=255,required=False,allow_blank=True)
    time_duration=serializers.FloatField(required=False, default=None)


    def validate(self, data):
        dataset_url = data.get("dataset")
        workflow_id = data.get("workflow_id")
        save_path=data.get("save_path")

        if not dataset_url and not workflow_id:

            raise serializers.ValidationError(
                "Either dataset_url or workflow_id must be provided"
            )
        
        if not dataset_url and worflow_id:
            worflow_dataset=Dataset.objects.filter(workflow_id=workflow_id).first()

            if not workflow_dataset:
                raise serializers.ValidationError(
                    "No dataset associated with the provided workflow_id."
                )
                data["dataset"] = workflow_dataset.urlpatterns
        if not save_path:
            raise serializers.ValidationError(
                "save_path must be provided"
            )
        
        return data 
