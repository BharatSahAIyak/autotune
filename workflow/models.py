import json
import uuid

from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils.translation import gettext_lazy as _


def validate_split(value):
    if len(value) != 3:
        raise ValidationError("Exactly three values are required.")
    if sum(value) != 100:
        raise ValidationError("The sum of the values must be 100.")


def default_split():
    return [80, 10, 10]


LLM_MODELS = [
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
]


class User(models.Model):
    user_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    huggingface_user_id = models.CharField(max_length=255)


class MLModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    huggingface_id = models.UUIDField(null=True, blank=True)
    uploaded_at = models.DateTimeField(null=True, blank=True)
    latest_commit_hash = models.UUIDField(null=True, blank=True)
    is_trained_at_autotune = models.BooleanField(default=False)
    name = models.CharField(max_length=255)
    is_locally_cached = models.BooleanField(default=False)


class Dataset(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    huggingface_id = models.UUIDField(null=True, blank=True)
    uploaded_at = models.DateTimeField(null=True, blank=True)
    is_generated_at_autotune = models.BooleanField(default=False)
    latest_commit_hash = models.UUIDField(null=True, blank=True)
    name = models.CharField(max_length=255)
    is_locally_cached = models.BooleanField(default=False)


class Workflows(models.Model):
    class WorkflowType(models.TextChoices):
        QNA_EXAMPLE = 'QnA', _('QnA Example')

    class WorkflowStatus(models.TextChoices):
        SETUP = 'SETUP', _('Setup')
        ITERATION = 'ITERATION', _('Iteration')
        GENERATION = 'GENERATION', _('Generation')
        TRAINING = 'TRAINING', _('Training')
        PUSHING_DATASET = 'PUSHING_DATASET', _('Pushing Dataset')
        PUSHING_MODEL = 'PUSHING_MODEL', _('Pushing Model')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    workflow_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow_name = models.CharField(max_length=255)
    workflow_type = models.CharField(
        max_length=50,
        choices=WorkflowType.choices,
        default=WorkflowType.QNA_EXAMPLE,
    )
    tags = ArrayField(models.CharField(max_length=255))
    total_examples = models.IntegerField()
    split = ArrayField(
        models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(100)]),
        default=default_split,
        validators=[validate_split],
    )
    llm_model = models.CharField(
        max_length=255, choices=[(model, model) for model in LLM_MODELS]
    )
    cost = models.IntegerField(default=0)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="workflow")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="workflow", blank=True, null=True)
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="+", blank=True, null=True)

    # Status of Workflow
    status = models.CharField(
        max_length=20,
        choices=WorkflowStatus.choices,
        default=WorkflowStatus.SETUP,
    )

    status_details = models.JSONField(default=dict)


class Examples(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    workflow = models.ForeignKey(
        Workflows, on_delete=models.CASCADE, related_name="examples"
    )
    example_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    text = models.JSONField(default=dict)
    label = models.CharField(max_length=255)
    reason = models.TextField(max_length=255)


class Prompt(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.TextField(blank=True, null=True)
    source = models.TextField(blank=True, null=True)
    workflow = models.OneToOneField(Workflows, on_delete=models.CASCADE, related_name='prompt')


class Task(models.Model):
    """
    This has the information about the different formats the response from the LLM model can be in.
    """
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    format = models.JSONField(default=dict)
    status = models.CharField(max_length=255, default="Starting")
    parent_task = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='subtasks')
    dataset = models.ForeignKey(Dataset, on_delete=models.SET_NULL, null=True, blank=True, related_name='tasks')
    temp_data = models.TextField(blank=True, null=True)
    total_number = models.IntegerField(default=1)
    workflow = models.OneToOneField(
        "Workflows", on_delete=models.CASCADE, related_name="task"
    )

    def set_temp_data(self, data):
        self.temp_data = json.dumps(data)
        self.save()

    def get_temp_data(self):
        return json.loads(self.temp_data) if self.temp_data else None


class Log(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    api_url = models.CharField(max_length=255)
    model = models.CharField(max_length=255)
    system = models.TextField()
    user = models.TextField()
    text = models.TextField()
    result = models.TextField()
    latency_ms = models.IntegerField(default=-1)


class WorkflowConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    system_prompt = models.TextField()
    user_prompt_template = models.TextField()
    json_schema = models.JSONField(default=dict, blank=True, null=True)
    parameters = models.JSONField(default=dict, blank=True, null=True)

    def get_json_schema_as_dict(self):
        return json.loads(self.json_schema)

    def __str__(self):
        return self.name
