"""
This module contains utility functions for AutoTuneNLP e2e tests.
This calls the server APIs with the given parameters and returns the response object.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()


HF_API_KEY = os.environ["HF_API_KEY"]
OPEN_AI_API_KEY = os.environ["OPENAI_API_KEY"]
TIMEOUT = 100


def dataview(prompt, num_samples, task, num_labels):
    """
    Calls the Data View API with the given parameters.

    Args:
        prompt (str): The prompt to use for generating samples.
        num_samples (int): The number of samples to generate.
        task (str): The task to use for generating samples.
        num_labels (int): The number of labels to use for the task.

    Returns:
        The response object returned by the API.
    """
    return requests.post(
        "http://localhost:8000/data/view",
        headers={"X-OpenAI-Key": OPEN_AI_API_KEY},
        json={
            "prompt": prompt,
            "num_samples": num_samples,
            "task": task,
            "num_labels": num_labels,
        },
        timeout=TIMEOUT,
    )


def dataset_generation(
    prompt,
    num_samples,
    repo,
    split,
    task,
    num_labels,
    labels,
    valid_data=None,
    invalid_data=None,
):
    """
    Calls the Data API with the given parameters.

    Args:
        prompt (str): The prompt to use for generating samples.
        num_samples (int): The number of samples to generate.
        repo (str): The repository to push the generated data to.
        split (list[int]): The split to use for the generated data.
        task (str): The task to use for generating samples.
        num_labels (int): The number of labels to use for the task.

    Returns:
        The response object returned by the API.
    """
    return requests.post(
        "http://localhost:8000/data",
        headers={
            "X-OpenAI-Key": OPEN_AI_API_KEY,
            "X-HuggingFace-Key": HF_API_KEY,
        },
        json={
            "prompt": prompt,
            "num_samples": num_samples,
            "repo": repo,
            "split": split,
            "task": task,
            "num_labels": num_labels,
            "labels": labels,
            "valid_data": valid_data,
            "invalid_data": invalid_data,
        },
        timeout=TIMEOUT,
    ).json()


def dataset_updation(prompt, num_samples, repo, split, task, num_labels):
    """
    Calls the Data API with the given parameters.

    Args:
        prompt (str): The prompt to use for generating samples.
        num_samples (int): The number of samples to generate.
        repo (str): The repository to push the generated data to.
        split (list[int]): The split to use for the generated data.
        task (str): The task to use for generating samples.
        num_labels (int): The number of labels to use for the task.

    Returns:
        The response object returned by the API.
    """
    return requests.put(
        "http://localhost:8000/data",
        headers={
            "X-OpenAI-Key": OPEN_AI_API_KEY,
            "X-HuggingFace-Key": HF_API_KEY,
        },
        json={
            "prompt": prompt,
            "num_samples": num_samples,
            "repo": repo,
            "split": split,
            "task": task,
            "num_labels": num_labels,
        },
        timeout=TIMEOUT,
    )


def model_training(dataset, model, epochs, save_path, task, version="main"):
    """
    Calls the Train API with the given parameters.

    Args:
        dataset (str): The dataset to use for finetuning.
        model (str): The model to use for finetuning.
        epochs (int): The number of epochs to use for finetuning.
        save_path (str): The repository to push the finetuned model to.
        task (str): The task to use for finetuning.
        version (str): The version of the model to use for finetuning.

    Returns:
        The response object returned by the API.
    """
    return requests.post(
        "http://localhost:8000/train",
        headers={"X-HuggingFace-Key": HF_API_KEY},
        json={
            "dataset": dataset,
            "model": model,
            "epochs": epochs,
            "save_path": save_path,
            "task": task,
            "version": version,
        },
        timeout=TIMEOUT,
    ).json()


def track_task(task_id):
    """
    Tracks a task based on the task ID and returns the result associated with the task.

    Args:
        task_id (str): The ID of the task to track.

    Returns:
        The response object returned by the API.
    """
    return requests.get(
        f"http://localhost:8000/track/{task_id}",
        headers={"X-HuggingFace-Key": HF_API_KEY},
        timeout=TIMEOUT,
    ).json()


def sample_generation(
    prompt,
    num_samples,
    repo,
    split,
    task,
    num_labels,
    labels,
    valid_data=None,
    invalid_data=None,
):
    """
    Calls the Data API with the given parameters.

    Args:
        prompt (str): The prompt to use for generating samples.
        num_samples (int): The number of samples to generate.
        repo (str): The repository to push the generated data to.
        split (list[int]): The split to use for the generated data.
        task (str): The task to use for generating samples.
        num_labels (int): The number of labels to use for the task.

    Returns:
        The response object returned by the API.
    """
    return requests.post(
        "http://localhost:8000/sample",
        headers={
            "X-OpenAI-Key": OPEN_AI_API_KEY,
            "X-HuggingFace-Key": HF_API_KEY,
        },
        json={
            "prompt": prompt,
            "num_samples": num_samples,
            "repo": repo,
            "split": split,
            "task": task,
            "num_labels": num_labels,
            "labels": labels,
            "valid_data": valid_data,
            "invalid_data": invalid_data,
        },
        timeout=TIMEOUT,
    ).json()
