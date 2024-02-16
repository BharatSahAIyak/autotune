import uuid

from django.contrib.postgres.fields import ArrayField
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


def validate_split(value):
    if len(value) != 3:
        raise ValidationError("Exactly three values are required.")
    if sum(value) != 100:
        raise ValidationError("The sum of the values must be 100.")


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
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    huggingface_user_id = models.CharField(max_length=255)


class Workflows(models.Model):
    workflow_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow_name = models.CharField(max_length=255)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    split = ArrayField(
        models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(100)]),
        default=[80, 10, 10],
        validators=[validate_split],
    )
    llm_model = models.CharField(
        max_length=255, choices=[(model, model) for model in LLM_MODELS]
    )
    cost = models.IntegerField(default=0)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="workflows")


class Examples(models.Model):
    workflow = models.OneToOneField(
        Workflows, on_delete=models.CASCADE, related_name="examples"
    )
    example_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    text = models.TextField()
    label = models.CharField(max_length=255)
    reason = models.TextField(max_length=255)


class Tasks(models.Model):
    """
    This has the information about the different formats the response from the LLM model can be in.
    """

    task_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_name = models.CharField(max_length=255)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    format = models.JSONField(default=dict)
    workflow = models.OneToOneField(
        "Workflows", on_delete=models.CASCADE, related_name="task"
    )
