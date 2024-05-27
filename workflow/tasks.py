from abc import ABC, abstractmethod
from functools import partial

import evaluate
import numpy as np
from datasets import Dataset
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForSequenceClassification
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)


def get_task_class(task):
    tasks = {"text_classification": TextClassification, "seq2seq": Seq2Seq}

    task_class = tasks.get(task)
    return task_class


class Tasks(ABC):
    def __init__(self, task: str, model_name: str, dataset: Dataset, version: str):
        self.task = task
        self.model_name = model_name
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=version)
        self.onnx = None
        self.model = None
        self.data_collator = None
        self.Trainer = None
        self.TrainingArguments = None
        self.metrics = None
        self.tokenized_dataset = None
        self._load_model_requirements()
        self._prepare_dataset()
        self.Trainer = partial(
            self.Trainer,
            model=self.model,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    @abstractmethod
    def _load_model_requirements(self):
        pass

    @abstractmethod
    def _prepare_dataset(self):
        pass

    @abstractmethod
    def compute_metrics(self, eval_pred):
        pass


# needs train and validation in the dataset
# needs 'class'/'label' column in the dataset


class TextClassification(Tasks):
    def __init__(self, model_name: str, dataset: Dataset, version: str):
        super().__init__("text_classification", model_name, dataset, version)
        self.metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
        self.onnx = ORTModelForSequenceClassification

    def _load_model_requirements(self):
        num_labels = len(self.dataset["train"].unique("class"))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.Trainer = Trainer
        self.TrainingArguments = TrainingArguments

    def _prepare_dataset(self):
        # assume label column is 'label' and text column is 'text' in the dataset
        self.tokenized_dataset = self.dataset.map(
            self.__preprocess_function, batched=True
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metrics.compute(predictions=predictions, references=labels)


class Seq2Seq(Tasks):
    def __init__(self, model_name: str, dataset: Dataset, version: str):
        super().__init__("seq2seq", model_name, dataset, version)
        self.metrics = evaluate.combine(["rouge", "bleu"])
        self.onnx = ORTModelForSeq2SeqLM

    def _load_model_requirements(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.Trainer = Seq2SeqTrainer
        self.TrainingArguments = partial(
            Seq2SeqTrainingArguments, predict_with_generate=True
        )

    def _prepare_dataset(self):
        # assume column names are 'Input' and 'Output' in the dataset
        self.tokenized_dataset = self.dataset.map(
            self.__preprocess_function, batched=True
        )
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

    def __preprocess_function(self, examples):
        inputs = self.tokenizer(examples["Input"], truncation=True)
        outputs = self.tokenizer(examples["Output"], truncation=True)
        inputs["labels"] = outputs["input_ids"]
        return inputs

    def __postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = self.__postprocess_text(
            decoded_preds, decoded_labels
        )

        result = self.metrics.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id)
            for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        # return {k: round(v, 4) for k, v in result.items()}
        return result
