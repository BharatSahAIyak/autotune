from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views
from .views import (
    GenerateTaskView,
    TaskProgressView,
    WorkflowDetailView,
    WorkflowDuplicateView,
    WorkflowSearchView,
    WorkflowStatusView,
    WorkflowUpdateView,
    create_workflow_with_prompt,
)

# router = DefaultRouter()
# router.register(r'prompts', PromptViewSet, basename='prompt')

urlpatterns = [
    path("", views.index, name="index"),
    path("create/", create_workflow_with_prompt, name="create_workflow"),
    path("<uuid:workflow_id>/", WorkflowDetailView.as_view(), name="workflow-detail"),
    path("prompt/<uuid:workflow_id>/", views.retrieve_prompt, name="retrieve-prompt"),
    path(
        "iterate/<uuid:workflow_id>/", views.iterate_workflow, name="iterate-workflow"
    ),
    path("prompt/update<uuid:workflow_id>/", views.update_prompt, name="update-prompt"),
    path("update/<workflow_id>", WorkflowUpdateView.as_view(), name="update-workflow"),
    path(
        "duplicate/<workflow_id>/",
        WorkflowDuplicateView.as_view(),
        name="duplicate-workflow",
    ),
    path("q/", WorkflowSearchView.as_view(), name="search-workflow"),
    path("status/<workflow_id>/", WorkflowStatusView.as_view(), name="workflow-status"),
    path(
        "generate/<uuid:workflow_id>/", GenerateTaskView.as_view(), name="generate-task"
    ),
    path(
        "progress/<uuid:workflow_id>/", TaskProgressView.as_view(), name="task-progress"
    ),
    path(
        "dehydrate-cache/<str:key_pattern>/",
        views.dehydrate_cache_view,
        name="dehydrate-cache",
    ),
    path("config/create/", views.create_workflow_config, name="create-config"),
    path("user/", views.add_user, name="add-user"),
    path("config/", views.create_workflow_config, name="create-config"),
    path("config/<uuid:config_id>", views.update_workflow_config, name="update-config"),
    # path('', include(router.urls)),
]
