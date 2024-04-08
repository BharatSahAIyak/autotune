from django.conf import settings
from django.core.cache import cache
from django.shortcuts import get_object_or_404
from huggingface_hub import HfApi, Repository
import pandas as pd
from datetime import datetime
import os

from workflow.models import WorkflowConfig


def upload_dataset_to_hf(combined_data, dataset_name):
    """
        Uploads a dataset to Hugging Face under a generated repository name based on the dataset name and timestamp.
        Parameters:
            - combined_data (list of dict): The dataset to upload, represented as a list of dictionaries where each dictionary represents a row in the dataset.
            - dataset_name (str): The base name for the dataset repository on Hugging Face. The actual repository name will also include a timestamp.
        Returns:
            - str: The name of the created repository on Hugging Face, which is the `dataset_name` appended with a timestamp.
        Raises:
            - HTTPError: An error occurred while trying to create the repository or upload the file to Hugging Face.
    """
    huggingface_token = settings.HUGGING_FACE_TOKEN
    df = pd.DataFrame(combined_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_name = f"{dataset_name}_{timestamp}"

    csv_path = f"/tmp/{repo_name}.csv"
    df.to_csv(csv_path, index=False)

    hf_api = HfApi()

    repo_url = hf_api.create_repo(token=huggingface_token,
                                  name=repo_name,
                                  organization=None,
                                  repo_type="dataset",
                                  private=False)

    repo = Repository(local_dir=f"/tmp/{repo_name}", clone_from=repo_url, use_auth_token=huggingface_token)

    os.rename(csv_path, f"/tmp/{repo_name}/{repo_name}.csv")
    repo.git_add()
    repo.git_commit("Add dataset")

    repo.git_push()

    return repo_name


def get_workflow_config(workflow_type):
    """
    Fetches a WorkflowConfig object from the cache or database by workflow_type.

    :param workflow_type: The type of the workflow to fetch the config for.
    :return: WorkflowConfig instance
    raises: HTTPError: Http 404 if no workflow config found in db.
    """
    cache_key = f"workflow_config_{workflow_type}"
    config = cache.get(cache_key)

    if config is None:
        get_object_or_404(WorkflowConfig, name=workflow_type)

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
    if hasattr(cache, 'delete_pattern'):
        cache.delete_pattern(key_pattern)
    else:
        cache.delete(key_pattern)