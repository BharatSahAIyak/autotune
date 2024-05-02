from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from .views import WorkflowConfigCreateView, WorkflowDetailView, WorkflowListView

urlpatterns = [
    path("", WorkflowListView.as_view(), name="workflow-list-v2"),
    path(
        "<uuid:workflow_id>/",
        WorkflowDetailView.as_view(),
        name="workflow-detail-v2",
    ),
    path(
        "create/",
        csrf_exempt(WorkflowConfigCreateView.as_view()),
        name="create_workflow",
    ),
]
