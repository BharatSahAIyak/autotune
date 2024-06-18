from workflow.training.tasks import Tasks
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from functools import partial
import numpy as np
from sklearn.preprocessing import LabelEncoder


# needs train and validation in the dataset
# needs 'class'/'label' column in the dataset
class TextClassification(Tasks):
    def __init__(self, model_name: str, version: str, args):
        super().__init__("text_classification", model_name, version)
        self.metrics = evaluate.load("f1")
        self.le = LabelEncoder()
        self.label2id = None
        if "label2id" in args and len(args["label2id"]) != 0:
            self.label2id = args["label2id"]

    def load_dataset(self, dataset):
        self.dataset = load_dataset(dataset).shuffle()
        self._prepare_dataset()
        self._load_model()
        return self.dataset

    def _load_model(self):
        num_labels = len(self.dataset["train"].unique("class"))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.Trainer = partial(
            self.Trainer,
            model=self.model,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def _load_model_requirements(self):
        self.Trainer = Trainer
        self.TrainingArguments = TrainingArguments

    def __label_encoder(self, examples):
        if self.label2id is not None:
            encoded_labels = np.array(
                [self.label2id[label] for label in examples["class"]]
            )

        else:
            encoded_labels = self.le.fit_transform(np.array(examples["class"]))
        return {"text": examples["text"], "label": encoded_labels}

    def _prepare_dataset(self):
        # assume label column is 'class' and text column is 'text' in the dataset
        self.dataset = self.dataset.map(self.__label_encoder, batched=True)
        self.tokenized_dataset = self.dataset.map(
            self.__preprocess_function, batched=True
        )

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metrics.compute(
            predictions=predictions, references=labels, average="macro"
        )

    def push_to_hub(self, trainer, save_path, hf_token=None):
        trainer.model.push_to_hub(
            save_path, commit_message="pytorch_model.bin upload/update"
        )
        trainer.tokenizer.push_to_hub(save_path)

    def get_training_args(self, req_data, dataset):

        return TrainingArguments(
            output_dir=f"./results_{req_data['task_id']}",
            num_train_epochs=req_data["epochs"],
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir=f"./logs_{req_data['task_id']}",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            warmup_steps=500,
            weight_decay=0.01,
            do_predict=True,
        )
