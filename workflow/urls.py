from django.urls import path, include
from rest_framework.routers import DefaultRouter

from . import views
from .views import create_workflow_with_prompt, WorkflowDetailView, PromptViewSet, WorkflowDuplicateView, \
    WorkflowStatusView, WorkflowUpdateView, WorkflowSearchView

# router = DefaultRouter()
# router.register(r'prompts', PromptViewSet, basename='prompt')

urlpatterns = [
    path("", views.index, name="index"),
    path('workflow/', create_workflow_with_prompt, name='create_workflow'),
    path('workflow/<uuid:workflow_id>/', WorkflowDetailView.as_view(), name='workflow-detail'),
    path('workflows/<int:workflow_id>/prompt/', views.retrieve_prompt, name='retrieve-prompt'),
    path('workflows/<int:workflow_id>/prompt/update/', views.update_prompt, name='update-prompt'),
    path('workflow/<workflow_id>/update/', WorkflowUpdateView.as_view(), name='update-workflow'),
    path('workflow/<workflow_id>/duplicate/', WorkflowDuplicateView.as_view(), name='duplicate-workflow'),
    path('workflow/q/', WorkflowSearchView.as_view(), name='search-workflow'),
    path('workflow/status/<workflow_id>/', WorkflowStatusView.as_view(), name='workflow-status'),
    # path('', include(router.urls)),
]