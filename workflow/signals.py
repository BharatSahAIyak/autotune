from django.core.exceptions import ValidationError
from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver

from .models import DatasetData, Prompt, Workflows


@receiver(post_save, sender=Prompt)
def update_latest_prompt(sender, instance, created, **kwargs):
    """
    Updates the latest_prompt field on the Workflows model whenever a new prompt is created or updated.
    """
    workflow = instance.workflow
    workflow.latest_prompt = instance
    workflow.save()


@receiver(pre_save, sender=Workflows)
def validate_workflow(sender, instance, **kwargs):
    """
    Checks that required fields are present for workflows of type 'COMPLETE'.
    """
    if instance.type == Workflows.WorkflowType.COMPLETE:
        if not instance.workflow_config:
            raise ValidationError(
                "Workflow configuration is required for COMPLETE type workflows."
            )
        if not instance.latest_prompt:
            raise ValidationError(
                "Latest prompt is required for COMPLETE type workflows."
            )
        if not instance.split:
            raise ValidationError("Split is required for COMPLETE type workflows.")
        if not instance.total_examples:
            raise ValidationError(
                "Total examples are required for COMPLETE type workflows."
            )
        if not instance.tags:
            raise ValidationError("Tags are required for COMPLETE type workflows.")


@receiver(pre_save, sender=DatasetData)
def validate_dataset_fields(sender, instance, **kwargs):
    """
    validates the required fields for different dataset types.
    """
    # TODO: implement this logic when more clarity on the field types needed for each dataset type
