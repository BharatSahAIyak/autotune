from django.apps import AppConfig


class Workflowv2Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "workflowV2"

    def ready(self):
        import workflow.signals