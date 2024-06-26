# Generated by Django 4.2.11 on 2024-05-02 12:10

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0014_examples_prompt_workflows_estimated_dataset_cost_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="workflowconfig",
            name="parameters",
        ),
        migrations.AddField(
            model_name="workflowconfig",
            name="temperature",
            field=models.IntegerField(
                default=1,
                validators=[
                    django.core.validators.MinValueValidator(0),
                    django.core.validators.MaxValueValidator(2),
                ],
            ),
        ),
    ]
