# Generated by Django 4.2.13 on 2024-07-01 06:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0031_remove_mlmodel_label_studio_comp_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="mlmodel",
            old_name="uploaded_at",
            new_name="last_trained",
        ),
    ]
