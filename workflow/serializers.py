from rest_framework import serializers

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


class ExampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Examples
        fields = ("text", "label", "reason")


class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = "__all__"


class WorkflowDetailSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    model = MLModelSerializer(read_only=True)
    dataset = DatasetSerializer(read_only=True)
    prompt = PromptSerializer(read_only=True)
    examples = ExampleSerializer(many=True, read_only=True, source="examples_set")

    class Meta:
        model = Workflows
        fields = (
            "workflow_id",
            "workflow_name",
            "total_examples",
            "split",
            "llm_model",
            "cost",
            "tags",
            "user",
            "dataset",
            "model",
            "examples",
            "prompt",
            "workflow_type",
        )


class WorkflowSerializer(serializers.ModelSerializer):
    examples = ExampleSerializer(many=True, required=False)

    class Meta:
        model = Workflows
        fields = (
            "workflow_name",
            "total_examples",
            "split",
            "llm_model",
            "cost",
            "tags",
            "user",
            "examples",
            "workflow_type",
        )

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
            "name",
            "system_prompt",
            "user_prompt_template",
            "json_schema",
            "parameters",
        )
