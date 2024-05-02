from rest_framework import serializers

from workflow.models import Examples, WorkflowConfig, Workflows


class ConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowConfig
        fields = (
            "name",
            "system_prompt",
            "user_prompt_template",
            "temperature",
            "schema_example",
        )


class ExampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Examples
        fields = ("text", "label", "reason", "example_id")


class WorkflowSerializer(serializers.ModelSerializer):
    config = ConfigSerializer(source="workflow_config", read_only=True)
    examples = ExampleSerializer(many=True, read_only=True)
    user_id = serializers.UUIDField(source="user.user_id", read_only=True)
    user_prompt = serializers.CharField(default="")

    class Meta:
        model = Workflows
        fields = (
            "workflow_id",
            "workflow_name",
            "user_id",
            "config",
            "split",
            "llm_model",
            "tags",
            "user_prompt",
            "cost",
            "estimated_dataset_cost",
            "examples",
        )
        extra_kwargs = {
            "cost": {"default": ""},
            "estimated_dataset_cost": {"default": ""},
        }
