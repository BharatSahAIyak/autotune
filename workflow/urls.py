from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views
from .views import (
    ExamplesView,
    PromptViewSet,
    SingleWorkflowView,
    WorkflowConfigView,
    WorkflowDuplicateView,
    WorkflowListView,
    WorkflowSearchView,
    WorkflowStatusView,
    WorkflowUpdateView,
    create_workflow_with_prompt,
)

urlpatterns = [
    # General routes
    path("", WorkflowListView.as_view(), name="workflow-list"),
    path("create/", create_workflow_with_prompt, name="create_workflow"),
    # Workflow-related routes
    path(
        "<uuid:workflow_id>/",
        SingleWorkflowView.as_view(),
        name="workflow-detail",
    ),
    path(
        "iterate/<uuid:workflow_id>/", views.iterate_workflow, name="iterate-workflow"
    ),
    path("prompt/<uuid:workflow_id>/", PromptViewSet.as_view(), name="prompt"),
    path(
        "update/<uuid:workflow_id>/",
        WorkflowUpdateView.as_view(),
        name="update-workflow",
    ),
    path(
        "duplicate/<uuid:workflow_id>/",
        WorkflowDuplicateView.as_view(),
        name="duplicate-workflow",
    ),
    path(
        "status/<uuid:workflow_id>/",
        WorkflowStatusView.as_view(),
        name="workflow-status",
    ),
    path("generate/<uuid:workflow_id>/", views.generate_task, name="generate-task"),
    # Examples routes
    path("examples/", ExamplesView.as_view(), name="examples"),
    path(
        "examples/<uuid:workflow_id>/",
        ExamplesView.as_view(),
        name="examples-by-workflow",
    ),
    # Search and config routes
    path("q/", WorkflowSearchView.as_view(), name="search-workflow"),
    path("config/", WorkflowConfigView.as_view(), name="workflow-config-list"),
    path(
        "config/<uuid:config_id>/",
        WorkflowConfigView.as_view(),
        name="workflow-config-detail",
    ),
    # Miscellaneous routes
    path(
        "dehydrate-cache/<str:key_pattern>/",
        views.dehydrate_cache_view,
        name="dehydrate-cache",
    ),
    path("user/", views.add_user, name="add-user"),
]
