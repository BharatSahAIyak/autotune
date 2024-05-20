from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from workflow.views import ExamplesView

from .views import (
    GetDataView,
    StatusView,
    WorkflowConfigCreateView,
    WorkflowDetailView,
    WorkflowGenerateView,
    WorkflowIterateView,
    WorkflowListView,
)

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
    path(
        "iterate/<uuid:workflow_id>/",
        WorkflowIterateView.as_view(),
        name="workflow-iterate",
    ),
    path(
        "generate/<uuid:workflow_id>/",
        WorkflowGenerateView.as_view(),
        name="workflow-iterate",
    ),
    path(
        "data/",
        GetDataView.as_view(),
        name="get_data",
    ),
    path("status", StatusView.as_view(), name="status"),
    path(
        "examples/<uuid:workflow_id>/",
        ExamplesView.as_view(),
        name="examples-by-workflow",
    ),
]
