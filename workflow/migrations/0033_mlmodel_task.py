# Generated by Django 4.2.13 on 2024-07-01 06:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0032_rename_uploaded_at_mlmodel_last_trained"),
    ]

    operations = [
        migrations.AddField(
            model_name="mlmodel",
            name="task",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
