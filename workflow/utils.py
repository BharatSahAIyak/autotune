from django.core.cache import cache
from django.shortcuts import get_object_or_404

from workflow.models import WorkflowConfig


def get_workflow_config(workflow_config):
    """
    Fetches a WorkflowConfig object from the cache or database by workflow_config.

    :param workflow_config: The type of the workflow to fetch the config for.
    :return: WorkflowConfig instance
    raises: HTTPError: Http 404 if no workflow config found in db.
    """
    cache_key = f"workflow_config_{workflow_config}"
    config = cache.get(cache_key)

    if config is None:
        get_object_or_404(WorkflowConfig, id=workflow_config)

    return config


def dehydrate_cache(key_pattern):
    """
    Dehydrates (clears) cache entries based on a given key pattern.
    This function can be used to invalidate specific cache entries manually,
    especially after database updates, to ensure cache consistency.

    Parameters:
    - key_pattern (str): The cache key pattern to clear. This can be a specific cache key
      or a pattern representing a group of keys.

    Returns:
    - None
    """
    if hasattr(cache, "delete_pattern"):
        cache.delete_pattern(key_pattern)
    else:
        cache.delete(key_pattern)
