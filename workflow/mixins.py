import logging
import uuid

from django.core.cache import cache
from django.http import JsonResponse

from workflow.models import User

logger = logging.getLogger(__name__)


class LoggingMixin:
    """
    Provides full logging of requests
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("django.request")

    def initial(self, request, *args, **kwargs):
        try:
            extra = {
                "request": request.data,
                "method": request.method,
                "endpoint": request.path,
                "user": request.user.username,
                "ip_address": request.META.get("REMOTE_ADDR"),
                "user_agent": request.META.get("HTTP_USER_AGENT"),
                "headers": dict(request.headers),
            }
            self.logger.info(f"Request received: {extra}")
        except Exception:
            self.logger.exception("Error logging request data")

        super().initial(request, *args, **kwargs)


class UserIDMixin:
    def dispatch(self, request, *args, **kwargs):
        user_id = request.headers.get("User-Id")
        if not user_id:
            return JsonResponse({"error": "User ID must be provided."}, status=401)

        try:
            user_id = uuid.UUID(user_id)
        except ValueError:
            return JsonResponse({"error": "Invalid user ID format."}, status=401)

        role = request.headers.get("role")

        if not role:
            return JsonResponse({"error": "Role must be provided."}, status=401)

        user = cache.get(f"user_{user_id}")
        if not user:
            try:
                user = User.objects.get(user_id=user_id)
                cache.set(f"user_{user_id}", user, timeout=3600)  # Cache for 1hr
            except User.DoesNotExist:
                logger.info(
                    f"User {user_id} not found in the database. Creating new user."
                )
                user = User.objects.create(
                    user_id=user_id,
                    role=role,
                    is_active=True,
                )
                cache.set(f"user_{user_id}", user, timeout=3600)
        else:
            logger.info(f"User {user_id} found in cache.")

        request.META["user"] = user
        return super().dispatch(request, *args, **kwargs)
