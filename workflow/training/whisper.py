from workflow.training.tasks import Tasks
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from functools import partial
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperFineTuning(Tasks):
    def __init__(self, model_name: str, version: str, args):
        super().__init__("whisper_finetuning", model_name, version)
        self.metrics = evaluate.load("wer")
        self.processor = WhisperProcessor.from_pretrained(model_name, language=args["language"], task="transcribe")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.args = args

    def load_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        train_dataset = dataset["train"]
        
        if "audio" not in train_dataset.column_names:
            audio_column = train_dataset.column_names[0]
            train_dataset = train_dataset.rename_column(audio_column, "audio")

        if "sentence" not in train_dataset.column_names:
            sentence_column = train_dataset.column_names[1]
            train_dataset = train_dataset.rename_column(sentence_column, "sentence")
        
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = train_dataset.train_test_split(test_size=0.1)
        self.train_dataset = dataset["train"]
        self.eval_dataset = dataset["test"]
        self._load_model()
        return dataset
    
    def _prepare_dataset(self):
        pass

    def _load_model_requirements(self):
        pass

    def _load_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.config.dropout = 0.1

        self.Trainer = partial(
            Seq2SeqTrainer,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.DataCollatorSpeechSeq2SeqWithPadding(
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                processor=self.processor,
                decoder_start_token_id=self.model.config.decoder_start_token_id
            )
        )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        feature_extractor: Any
        tokenizer: Any
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": self.feature_extractor(f["audio"]["array"], sampling_rate=f["audio"]["sampling_rate"]).input_features[0]} for f in features]
            labels = [{"input_ids": self.tokenizer(f["sentence"]).input_ids} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
            labels_batch["input_ids"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels_batch["input_ids"][:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels_batch["input_ids"] = labels_batch["input_ids"][:, 1:]

            batch["labels"] = labels_batch["input_ids"]
            return batch

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metrics.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def get_training_args(self, req_data, dataset):
        return Seq2SeqTrainingArguments(
            output_dir=f"./results_{req_data['task_id']}",
            num_train_epochs=req_data["epochs"],
            per_device_train_batch_size=self.args.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=self.args.get("per_device_eval_batch_size", 8),
            warmup_ratio=self.args.get("warmup_ratio", 0.1),
            gradient_accumulation_steps=self.args.get("gradient_accumulation_steps", 2),
            learning_rate=self.args.get("learning_rate", 3.75e-5),
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=self.args.get("eval_steps", 50),
            logging_steps=self.args.get("logging_steps", 25),
            save_steps=self.args.get("save_steps", 50),
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,
            lr_scheduler_type=self.args.get("lr_scheduler_type", "constant"),
        )

    def push_to_hub(self, trainer, save_path, hf_token=None):
        trainer.model.push_to_hub(
            save_path, commit_message="pytorch_model.bin upload/update"
        )
