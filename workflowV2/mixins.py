import uuid

from django.http import JsonResponse

from workflow.models import User


class UserIDMixin:
    def dispatch(self, request, *args, **kwargs):
        user_id = request.headers.get("user")
        try:
            user_id = uuid.UUID(user_id)
            user = User.objects.get(user_id=user_id)
            request.user = user
        except (ValueError, User.DoesNotExist):
            return JsonResponse({"error": "Invalid or missing user_id"}, status=401)

        return super().dispatch(request, *args, **kwargs)
