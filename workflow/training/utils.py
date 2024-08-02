from django.conf import settings
from huggingface_hub import HfApi, snapshot_download

from workflow.training import Colbert, NamedEntityRecognition, TextClassification, WhisperFineTuning
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModel,
    WhisperForConditionalGeneration
)

def download_model(repo_id):
    """
    Downloads a model from the Hugging Face Model Hub.
    Args:
        repo_id (str): The ID of the model repository.
        repo_type (str): The type of the model repository.
    Returns:
        str: The path to the downloaded model.
    Examples:
        >>> download_model("bert-base-uncased")
        '/path/to/downloaded/model'
    """
    return download_hf_repo(
        repo_id=repo_id,
        repo_type="model",
    )


def download_hf_repo(repo_id, repo_type):
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=settings.HUGGING_FACE_TOKEN,
    )


def push_to_hub(folder_path, repo_id, repo_type):
    api = HfApi(endpoint="https://huggingface.co", token=settings.HUGGING_FACE_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type=repo_type)


def get_task_class(task):
    tasks = {
        "text_classification": TextClassification,
        "embedding": Colbert,
        "ner": NamedEntityRecognition,
        "whisper_finetuning": WhisperFineTuning,
    }

    task_class = tasks.get(task)
    return task_class

def get_model_class(task_type):
    task_to_model = {
        "text_classification": AutoModelForSequenceClassification,
        "question_answering": AutoModelForQuestionAnswering,
        "token_classification": AutoModelForTokenClassification,
        "causal_lm": AutoModelForCausalLM,
        "embedding": AutoModel,
        "whisper_finetuning": WhisperForConditionalGeneration
    }
    return task_to_model.get(task_type)
