# Generated by Django 4.2.11 on 2024-04-26 17:33

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0010_task_generated_samples_task_total_samples"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="workflows",
            name="workflow_type",
        ),
        migrations.AddField(
            model_name="workflows",
            name="workflow_config",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="workflows",
                to="workflow.workflowconfig",
            ),
            preserve_default=False,
        ),
    ]
