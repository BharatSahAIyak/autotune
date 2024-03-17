from django.urls import path, include
from rest_framework.routers import DefaultRouter

from . import views
from .views import create_workflow_with_prompt, WorkflowDetailView, WorkflowDuplicateView, \
    WorkflowStatusView, WorkflowUpdateView, WorkflowSearchView, TaskProgressView, GenerateTaskView

# router = DefaultRouter()
# router.register(r'prompts', PromptViewSet, basename='prompt')

urlpatterns = [
    path("", views.index, name="index"),
    path('create/', create_workflow_with_prompt, name='create_workflow'),
    path('<uuid:workflow_id>/', WorkflowDetailView.as_view(), name='workflow-detail'),
    path('<uuid:workflow_id>/prompt/', views.retrieve_prompt, name='retrieve-prompt'),
    path('<uuid:workflow_id>/iterate/', views.iterate_workflow, name='iterate-workflow'),
    path('<uuid:workflow_id>/prompt/update/', views.update_prompt, name='update-prompt'),
    path('<workflow_id>/update/', WorkflowUpdateView.as_view(), name='update-workflow'),
    path('<workflow_id>/duplicate/', WorkflowDuplicateView.as_view(), name='duplicate-workflow'),
    path('q/', WorkflowSearchView.as_view(), name='search-workflow'),
    path('status/<workflow_id>/', WorkflowStatusView.as_view(), name='workflow-status'),
    path('generate/<uuid:workflow_id>/', GenerateTaskView.as_view(), name='generate-task'),
    path('progress/<uuid:workflow_id>/', TaskProgressView.as_view(), name='task-progress'),
    # path('', include(router.urls)),
]