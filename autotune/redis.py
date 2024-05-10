import redis
from django.conf import settings

redis_conn = redis.Redis.from_url(settings.REDIS_URL)
