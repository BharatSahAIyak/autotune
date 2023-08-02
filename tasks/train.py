from datasets import load_dataset
from huggingface_hub import login, HfApi
import shutil
import json
import io
import os

from utils import CeleryProgressCallback, get_task_class


def train_model(celery, req, api_key):
    task_class = get_task_class(req['task'])
    dataset = load_dataset(req['dataset']).shuffle()

    task = task_class(req['model'], dataset)

    celery.update_state(state='TRAINING')

    training_args = task.TrainingArguments(
        output_dir=f'./results_{celery.request.id}',
        num_train_epochs=req['epochs'],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir=f'./logs_{celery.request.id}',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        warmup_steps=500, 
        weight_decay=0.01,
        do_predict=True
    )

    trainer = task.Trainer(
        args=training_args, 
        callbacks=[CeleryProgressCallback(celery)]
    )

    trainer.train()

    _, _, metrics = trainer.predict(task.tokenized_dataset['test'])
    json_metrics = json.dumps(metrics)
    json_bytes = json_metrics.encode('utf-8')
    fileObj = io.BytesIO(json_bytes)

    meta = {"logs": trainer.state.log_history, "metrics": metrics}
    celery.update_state(state='PUSHING', meta=meta)

    login(token=api_key)
    task.model.push_to_hub(req['save_path'], commit_message="pytorch_model.bin upload/update")
    task.tokenizer.push_to_hub(req['save_path'])

    ort_model = task.onnx.from_pretrained(req['save_path'], export=True) # revision = req['version']
    ort_model.save_pretrained(f"./results_{celery.request.id}/onnx")
    ort_model.push_to_hub(
        f"./results_{celery.request.id}/onnx",
        repository_id=req['save_path'], 
        use_auth_token=True,
    )

    hfApi = HfApi(endpoint='https://huggingface.co', token=api_key)
    hfApi.upload_file(
        path_or_fileobj=fileObj,
        path_in_repo="metrics.json",
        repo_id=req['save_path'],
        repo_type="model"
    )

    if os.path.exists(f'./results_{celery.request.id}'):
        shutil.rmtree(f'./results_{celery.request.id}')
    if os.path.exists(f'./logs_{celery.request.id}'):
        shutil.rmtree(f'./logs_{celery.request.id}')   

    return meta