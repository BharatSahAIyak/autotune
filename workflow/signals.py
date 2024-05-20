from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Prompt, Workflows


@receiver(post_save, sender=Prompt)
def update_latest_prompt(sender, instance, created, **kwargs):
    """
    Updates the latest_prompt field on the Workflows model whenever a new prompt is created or updated.
    """
    workflow = instance.workflow
    workflow.latest_prompt = instance
    workflow.save()
