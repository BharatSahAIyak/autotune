import uuid

from django.core.cache import cache
from django.http import JsonResponse

from workflow.models import User


class UserIDMixin:
    def dispatch(self, request, *args, **kwargs):
        user_id = request.headers.get("user")
        if not user_id:
            return JsonResponse({"error": "User ID must be provided."}, status=401)

        try:
            user_id = uuid.UUID(user_id)
        except ValueError:
            return JsonResponse({"error": "Invalid user ID format."}, status=401)

        user = cache.get(f"user_{user_id}")
        if not user:
            try:
                user = User.objects.get(user_id=user_id)
                cache.set(f"user_{user_id}", user, timeout=3600)  # cache for 1hr
            except User.DoesNotExist:
                return JsonResponse({"error": "User not found."}, status=404)

        request.user = user
        return super().dispatch(request, *args, **kwargs)
