import io
import json
import os
import shutil

from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from django.utils import timezone
from django_pandas.io import read_frame
from huggingface_hub import CommitOperationAdd, HfApi, login
from transformers import TrainerCallback

from workflow.models import Dataset, DatasetData, MLModel, Task, TrainingMetadata, User
from .utils import get_task_class, get_model_class
from workflow.utils import get_task_mapping
from workflow.training.quantize_model import quantize_model

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def train(self, req_data, user_id, training_task, cached_dataset_id):
    upload_cache(cached_dataset_id, training_task, req_data["dataset"])

    task_id = self.request.id
    task = Task.objects.get(id=task_id)
    task.status = "TRAINING"
    task.save()
    meta = train_model(self, req_data, task_id)
    task.status = "PUSHING"
    task.save()
    try:
        huggingface_id = req_data.get("save_path").split("/")[0]
        model_name = req_data.get("save_path").split("/")[-1]

        model_id = req_data.get("model_id")
        if model_id:
            # Check if the model exists
            try:
                ml_model = MLModel.objects.get(id=model_id)
                # Update the model if it exists
                ml_model.name = model_name
                ml_model.is_trained_at_autotune = True
                ml_model.is_locally_cached = True
                ml_model.last_trained = timezone.now()
                ml_model.latest_commit_hash = meta.get("latest_commit_hash")
                ml_model.huggingface_id = huggingface_id
                ml_model.save()
                logger.info(f"Updated MLModel with ID {model_id}")
            except MLModel.DoesNotExist:
                logger.error(f"MLModel with ID {model_id} does not exist.")
                return  # Optionally, you can handle this case differently
        else:
            ml_model = MLModel.objects.create(
                name=model_name,
                is_trained_at_autotune=True,
                is_locally_cached=True,
                last_trained=timezone.now(),
                latest_commit_hash=meta.get("latest_commit_hash"),
                huggingface_id=huggingface_id,
            )
            logger.info("Created a new MLModel")
        user = User.objects.get(user_id=user_id)

        TrainingMetadata.objects.create(
            model=ml_model, logs=meta["logs"], metrics=meta["metrics"]
        )
        logger.info("Created TrainingMetadata")
    except Exception as e:
        logger.error(f"Failed to update model and log: {str(e)}")

    if "quantization_type" in req_data and req_data["quantization_type"]:
        task.status = "QUANTIZING"
        task.save()
        model_class = get_model_class(req_data["task_type"])
        if model_class:
            try:
                quantized_model = quantize_model(
                    model_name = req_data["save_path"],
                    model_class = model_class,
                    quantization_type = req_data["quantization_type"],
                    test_text = req_data["test_text"] #user has option to send this
                )
                if quantized_model:
                    quantized_save_path = f"{req_data['save_path']}_quantized"
                    api_key = settings.HUGGING_FACE_TOKEN
                    login(token=api_key)
                    quantized_model.push_to_hub(quantized_save_path, hf_token=api_key)
                    logger.info(f"Quantized model saved to {quantized_save_path}")
                else:
                    logger.warning("Quantization process did not return a model")
            except ValueError as ve:
                logger.error(f"ValueError during quantization: {str(ve)}")
            except RuntimeError as re:
                logger.error(f"RuntimeError during quantization: {str(re)}")
            except Exception as e:
                logger.error(f"Failed to quantize model: {str(e)}")
        else:
            logger.error(f"Unsupported task for quantization: {req_data['task_type']}")

    task.status = "COMPLETE"
    task.save()


class CeleryProgressCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task

    def on_log(self, args, state, control, logs, **kwargs):
        self.task.update_state(state="TRAINING", meta=state.log_history)


def train_model(celery, req_data, task_id):
    req_data["task_id"] = task_id
    task_class = get_task_class(req_data["task_type"])
    # TODO: differentiate between workflow dataset and request dataset
    task = task_class(req_data["model"], req_data["version"], args=req_data["args"])
    dataset = task.load_dataset(req_data["dataset"])
    training_args = task.get_training_args(req_data, dataset)

    trainer = task.Trainer(
        model=task.model, args=training_args, callbacks=[CeleryProgressCallback(celery)]
    )

    trainer.train()

    metrics = trainer.evaluate()
    json_metrics = json.dumps(metrics)
    json_bytes = json_metrics.encode("utf-8")
    fileObj = io.BytesIO(json_bytes)

    meta = {"logs": trainer.state.log_history, "metrics": metrics}
    logger.info(metrics)
    celery.update_state(state="PUSHING", meta=meta)

    api_key = settings.HUGGING_FACE_TOKEN
    login(token=api_key)
    task.push_to_hub(trainer, req_data["save_path"], hf_token=api_key)

    hfApi = HfApi(endpoint="https://huggingface.co", token=api_key)
    upload = hfApi.upload_file(
        path_or_fileobj=fileObj,
        path_in_repo="metrics.json",
        repo_id=req_data["save_path"],
        repo_type="model",
    )

    meta["latest_commit_hash"] = upload.commit_url.split("/")[-1]

    # if os.path.exists(f"./results_{celery.request.id}"):
    #     shutil.rmtree(f"./results_{celery.request.id}")
    # if os.path.exists(f"./logs_{celery.request.id}"):
    #     shutil.rmtree(f"./logs_{celery.request.id}")

    logger.info("Training complete")

    return meta


def upload_cache(cached_dataset_id, training_task, dataset):
    # upload the cached dataset to HF
    if(training_task=="whisper_finetuning"):   # caching dataset not yet implemented for whisper finetuning
        return
    hf_api = HfApi(token=settings.HUGGING_FACE_TOKEN)

    cached_dataset = Dataset.objects.get(id=cached_dataset_id)
    cachedDataEntries = DatasetData.objects.filter(dataset=cached_dataset)

    task_mapping = get_task_mapping(training_task)
    additional_fields = list(task_mapping.keys())
    fieldNames = ["record_id", "file"]
    fieldNames.extend(additional_fields)
    df = read_frame(cachedDataEntries, fieldnames=fieldNames)

    grouped = df.groupby("file")

    # Upload each group as a CSV to Hugging Face
    for file_name, group in grouped:
        # drop the file column
        group.drop(columns=["file"], inplace=True)

        # rename the columns for the dataset column names
        for field in additional_fields:
            group.rename(columns={field: task_mapping[field]}, inplace=True)

        csv_data = group.to_csv(index=False)
        file_data = csv_data.encode("utf-8")

        operation = CommitOperationAdd(file_name, file_data)
        commit_info = hf_api.create_commit(
            repo_id=dataset,
            operations=[operation],
            commit_message=f"Updating {file_name} file",
            repo_type="dataset",
        )
        logger.info(f"pushed {file_name} file: {commit_info}")

    logger.info("Uploaded cached dataset to HF")

    # delete the cached dataset
    cached_dataset.delete()
    logger.info("Deleted cached dataset")
    return
