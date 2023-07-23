from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from huggingface_hub import login
from functools import partial
import shutil
import os

from utils import tokenize_data, CeleryProgressCallback


def train_model(celery, req, api_key):
    tokenizer = AutoTokenizer.from_pretrained(req['model'])
    model = AutoModelForSeq2SeqLM.from_pretrained(req['model'])
    
    dataset = load_dataset(req['dataset'])
    train_data = dataset['train']
    eval_data = dataset['validation']
    test_data = dataset['test']

    partial_tokenize_data = partial(tokenize_data, tokenizer=tokenizer)

    train_data = train_data.map(partial_tokenize_data, batched=True)
    eval_data = eval_data.map(partial_tokenize_data, batched=True)
    test_data = test_data.map(partial_tokenize_data, batched=True)

    celery.update_state(state='TRAINING')
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'./results_{celery.request.id}',
        num_train_epochs=req['epochs'],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir=f'./logs_{celery.request.id}',
        save_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        warmup_steps=500, 
        weight_decay=0.01,
    )

    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args, 
        train_dataset=train_data, 
        eval_dataset=eval_data,
        callbacks=[CeleryProgressCallback(celery)]
    )

    trainer.train()
    login(token=api_key)

    model.push_to_hub(req['save_path'])
    tokenizer.push_to_hub(req['save_path'])

    celery.update_state(state='COMPLETED', meta=trainer.state.log_history)

    if os.path.exists(f'./results_{celery.request.id}'):
        shutil.rmtree(f'./results_{celery.request.id}')
    if os.path.exists(f'./logs_{celery.request.id}'):
        shutil.rmtree(f'./logs_{celery.request.id}')   
