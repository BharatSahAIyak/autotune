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
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo",
]


class User(models.Model):
    user_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_name = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    huggingface_user_id = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)  # for django's authentication
    role = models.CharField(max_length=255, default="user")


class Dataset(models.Model):
    class DatasetType(models.TextChoices):
        ASR = "ASR", _("ASR")
        ASR_NGRAM = "ASR_NGRAM", _("ASR_NGRAM")
        TRANSLATE = "TRANSLATE", _("TRANSLATE")
        SPELL_CHECK = "SPELL_CHECK", _("SPELL_CHECK")
        NER = "NER", _("NER")
        EMBEDDING = "EMBEDDING", _("EMBEDDING")
        RERANKER = "RERANKER", _("RERANKER")
        CLASSIFIER = "CLASSIFIER", _("CLASSIFIER")
        SEMANTIC_CHUNKING = "SEMANTIC_CHUNKING", _("SEMANTIC_CHUNKING")
        NEURAL_COREF = "NEURAL_COREF", _("NEURAL_COREF")
        LANGUAGE_DETECTION = "LANGUAGE_DETECTION", _("LANGUAGE_DETECTION")
        SYNTHETIC = "SYNTHETIC", _("SYNTHETIC")  # For dataset generated at autotune

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    huggingface_id = models.CharField(null=True, blank=True)
    uploaded_at = models.DateTimeField(null=True, blank=True)
    is_generated_at_autotune = models.BooleanField(default=False)
    latest_commit_hash = models.CharField(
        null=True, blank=True
    )  # not a uuid on huggingface
    name = models.CharField(max_length=255)
    is_locally_cached = models.BooleanField(default=False)
    workflow = models.ForeignKey(
        "Workflows", on_delete=models.CASCADE, related_name="datasets"
    )
    type = models.CharField(
        max_length=50, choices=DatasetType.choices, default=DatasetType.SYNTHETIC
    )


class MLModelConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_save_path = models.TextField()
    dataset_path = models.TextField()
    type = models.CharField(max_length=255)
    system_prompt = models.TextField()
    user_prompt_template = models.TextField()
    schema_example = models.JSONField(default=dict)
    temperature = models.IntegerField(
        default=1, validators=[MinValueValidator(0), MaxValueValidator(2)]
    )
    model_string = models.TextField()


class MLModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    name = models.CharField(max_length=255)
    huggingface_id = models.CharField(null=True, blank=True)
    last_trained = models.DateTimeField(null=True, blank=True)
    latest_commit_hash = models.UUIDField(null=True, blank=True)
    is_trained_at_autotune = models.BooleanField(default=False)
    is_locally_cached = models.BooleanField(default=False)
    trained_on = models.ForeignKey(
        Dataset,
        related_name="trained_on",
        on_delete=models.PROTECT,
        null=True,
        blank=True,
    )
    label_studio_element = models.JSONField(null=True, blank=True)
    telemetry_data_field = models.JSONField(null=True, blank=True)
    deployed_at = models.DateTimeField(null=True, blank=True)
    config = models.ForeignKey(
        MLModelConfig,
        on_delete=models.CASCADE,
        related_name="models",
        null=True,
        blank=True,
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="ml_model")
    task = models.CharField(max_length=255, null=True, blank=True)


class TrainingMetadata(models.Model):
    trained_at = models.DateTimeField(auto_now_add=True)
    model = models.ForeignKey(
        MLModel, on_delete=models.CASCADE, related_name="metadata"
    )
    logs = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)


class WorkflowConfig(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    system_prompt = models.TextField()
    user_prompt_template = models.TextField()
    schema_example = models.JSONField(default=dict)
    temperature = models.IntegerField(
        default=1, validators=[MinValueValidator(0), MaxValueValidator(2)]
    )
    fields = models.JSONField(default=dict)
    model_string = models.TextField()

    def __str__(self):
        return self.name


class Workflows(models.Model):

    class WorkflowStatus(models.TextChoices):
        SETUP = "SETUP", _("Setup")
        ITERATION = "ITERATION", _("Iteration")
        GENERATION = "GENERATION", _("Generation")
        TRAINING = "TRAINING", _("Training")
        PUSHING_DATASET = "PUSHING_DATASET", _("Pushing Dataset")
        PUSHING_MODEL = "PUSHING_MODEL", _("Pushing Model")
        IDLE = "IDLE", _("Idle")

    class WorkflowType(models.TextChoices):
        COMPLETE = "COMPLETE", _("Complete")
        TRAINING = "TRAINING", _("Training")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    workflow_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow_name = models.CharField(max_length=255)
    workflow_config = models.ForeignKey(
        WorkflowConfig,
        on_delete=models.CASCADE,
        related_name="workflows",
        null=True,
        blank=True,
    )
    tags = ArrayField(models.CharField(max_length=255), null=True, blank=True)
    total_examples = models.IntegerField(null=True, blank=True)
    split = ArrayField(
        models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(100)]),
        default=default_split,
        validators=[validate_split],
        null=True,
        blank=True,
    )
    llm_model = models.CharField(
        max_length=255,
        choices=[(model, model) for model in LLM_MODELS],
        default="gpt-3.5-turbo",
    )
    cost = models.DecimalField(decimal_places=4, max_digits=10, default=0)
    estimated_dataset_cost = models.DecimalField(
        decimal_places=4, max_digits=10, null=True, blank=True
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="workflow")
    model = models.ForeignKey(
        MLModel, on_delete=models.CASCADE, related_name="+", blank=True, null=True
    )

    # Status of Workflow
    status = models.CharField(
        max_length=20,
        choices=WorkflowStatus.choices,
        default=WorkflowStatus.SETUP,
    )

    type = models.CharField(
        max_length=50, choices=WorkflowType.choices, default=WorkflowType.COMPLETE
    )

    status_details = models.JSONField(default=dict)
    latest_prompt = models.ForeignKey(
        "Prompt",
        on_delete=models.SET_NULL,
        null=True,
        related_name="latest_for_workflow",
    )


class DatasetData(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.CharField(max_length=255, null=True, blank=True)
    input_string = models.TextField(blank=True, null=True)
    output_string = models.TextField(blank=True, null=True)
    input_json = models.JSONField(blank=True, null=True)
    output_json = models.JSONField(blank=True, null=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="data")

    def save(self, *args, **kwargs):
        # if self.dataset.type == Dataset.DatasetType.ASR:
        # We can define the fields needed for different dataset types here

        super().save(*args, **kwargs)


class Prompt(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user_prompt = models.TextField()
    system_prompt = models.TextField(blank=True, null=True)
    source = models.TextField(blank=True, null=True)
    workflow = models.ForeignKey(
        Workflows, on_delete=models.CASCADE, related_name="prompts"
    )


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
    task_id = models.UUIDField(null=True, blank=True)
    prompt = models.ForeignKey(
        Prompt, on_delete=models.CASCADE, related_name="examples", null=True, blank=True
    )


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
    dataset = models.OneToOneField(
        "Dataset", on_delete=models.CASCADE, related_name="task", null=True, blank=True
    )
    workflow = models.ForeignKey(
        "Workflows", on_delete=models.CASCADE, related_name="tasks"
    )
    generated_samples = models.IntegerField(default=0)
    total_samples = models.IntegerField(default=0)


class Log(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    api_url = models.CharField(max_length=255)
    model = models.CharField(max_length=255)
    system = models.TextField()
    user = models.TextField()
    text = models.TextField()
    result = models.TextField()
    latency_ms = models.IntegerField(default=-1)
