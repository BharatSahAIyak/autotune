import io
import json
import logging
import os
import shutil

from celery import shared_task
from datasets import load_dataset
from django.conf import settings

logger = logging.getLogger(__name__)

from huggingface_hub import HfApi, login
from transformers import TrainerCallback

from .tasks import get_task_class


@shared_task(bind=True, max_retries=settings.CELERY_MAX_RETRIES, retry_backoff=True)
def train(self, req_data):
    task_id = self.request.id
    meta = train_model(self, req_data, task_id)


class CeleryProgressCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task

    def on_log(self, args, state, control, logs, **kwargs):
        self.task.update_state(state="TRAINING", meta=state.log_history)


def train_model(celery, req_data, task_id):
    task_class = get_task_class(req_data["task"])
    # todo: differentiate between workflow dataset and request dataset
    dataset = load_dataset(req_data["dataset"]).shuffle()
    task = task_class(req_data["model"], dataset, req_data["version"])

    api_key = settings.HUGGING_FACE_TOKEN

    training_args = task.TrainingArguments(
        output_dir=f"./results_{task_id}",
        num_train_epochs=req_data["epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir=f"./logs_{task_id}",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_steps=500,
        weight_decay=0.01,
        do_predict=True,
    )

    trainer = task.Trainer(
        args=training_args, callbacks=[CeleryProgressCallback(celery)]
    )

    trainer.train()

    _, _, metrics = trainer.predict(task.tokenized_dataset["test"])
    json_metrics = json.dumps(metrics)
    json_bytes = json_metrics.encode("utf-8")
    fileObj = io.BytesIO(json_bytes)

    meta = {"logs": trainer.state.log_history, "metrics": metrics}
    celery.update_state(state="PUSHING", meta=meta)

    login(token=api_key)
    task.model.push_to_hub(
        req_data["save_path"], commit_message="pytorch_model.bin upload/update"
    )
    task.tokenizer.push_to_hub(req_data["save_path"])

    # quantized_model = quantize_dynamic(task.model, {torch.nn.Linear}, dtype=torch.qint8)

    ort_model = task.onnx.from_pretrained(
        task.model, export=True
    )  # revision = req['version']
    ort_model.save_pretrained(f"./results_{celery.request.id}/onnx")
    ort_model.push_to_hub(
        f"./results_{celery.request.id}/onnx",
        repository_id=req_data["save_path"],
        use_auth_token=True,
    )

    hfApi = HfApi(endpoint="https://huggingface.co", token=api_key)
    hfApi.upload_file(
        path_or_fileobj=fileObj,
        path_in_repo="metrics.json",
        repo_id=req_data["save_path"],
        repo_type="model",
    )

    if os.path.exists(f"./results_{celery.request.id}"):
        shutil.rmtree(f"./results_{celery.request.id}")
    if os.path.exists(f"./logs_{celery.request.id}"):
        shutil.rmtree(f"./logs_{celery.request.id}")

    return meta
