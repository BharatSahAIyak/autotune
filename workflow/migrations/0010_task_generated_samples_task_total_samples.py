# Generated by Django 4.2.11 on 2024-04-25 16:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('workflow', '0009_remove_task_parent_task_remove_task_temp_data_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='task',
            name='generated_samples',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='task',
            name='total_samples',
            field=models.IntegerField(default=0),
        ),
    ]