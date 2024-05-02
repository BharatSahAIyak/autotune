from django.urls import path

from .views import WorkflowDetailView, WorkflowListView

urlpatterns = [
    path("", WorkflowListView.as_view(), name="workflow-list-v2"),
    path(
        "<uuid:workflow_id>/",
        WorkflowDetailView.as_view(),
        name="workflow-detail-v2",
    ),
]
