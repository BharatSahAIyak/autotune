from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass
import logging
import evaluate
import numpy as np
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import HfApi

from colbert.training.utils import print_progress, manage_checkpoints
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from huggingface_hub import snapshot_download
import torch
from colbert.utils.amp import MixedPrecisionManager
import pandas as pd
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.training.rerank_batcher import RerankBatcher
from colbert.training.lazy_batcher import LazyBatcher
from colbert.utils.utils import print_message
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def get_task_class(task):
    tasks = {
        "text_classification": TextClassification,
        "embedding": Colbert,
    }

    task_class = tasks.get(task)
    return task_class


class Tasks(ABC):
    def __init__(self, task: str, model_name: str, version: str):
        self.task = task
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=version)
        self.onnx = None
        self.model = None
        self.data_collator = None
        self.Trainer = None
        self.TrainingArguments = None
        self.metrics = None
        self.tokenized_dataset = None
        self._load_model_requirements()
        # self.Trainer = partial(
        #     self.Trainer,
        #     model=self.model,
        #     train_dataset=self.tokenized_dataset["train"],
        #     eval_dataset=self.tokenized_dataset["test"],
        #     tokenizer=self.tokenizer,
        #     data_collator=self.data_collator,
        #     compute_metrics=self.compute_metrics,
        # )

    @abstractmethod
    def _load_model_requirements(self):
        pass

    @abstractmethod
    def _prepare_dataset(self):
        pass

    @abstractmethod
    def compute_metrics(self, eval_pred):
        pass

    @abstractmethod
    def push_to_hub(self, trainer, save_path):
        pass


# needs train and validation in the dataset
# needs 'class'/'label' column in the dataset
class TextClassification(Tasks):
    def __init__(self, model_name: str, version: str, args):
        super().__init__("text_classification", model_name, version)
        self.metrics = evaluate.load("f1")
        self.onnx = ORTModelForSequenceClassification
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
            self.model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
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
        # self.model.label2id = self.model.id2label

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
        if self.label2id is None:
            self.label2id = dict(
                zip(self.le.classes_, map(str, self.le.transform(self.le.classes_)))
            )
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metrics.compute(
            predictions=predictions, references=labels, average="macro"
        )

    def push_to_hub(self, trainer, save_path, hf_token=None):
        logger.info(self.label2id)
        logger.info(self.id2label)
        trainer.model.push_to_hub(save_path, commit_message="Updated Model")
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


class Colbert(Tasks):
    def __init__(self, model_name: str, version: str, kwargs):
        super().__init__("embedding", model_name, version)
        self.metrics = evaluate.combine(["cosine_similarity"])
        self.model_name = model_name
        self.checkpoint = model_name
        self._load_model_requirements()

    def load_dataset(self, dataset):
        path = snapshot_download(
            repo_id=dataset, repo_type="dataset", cache_dir="./data/"
        )
        return path

    def _load_model_requirements(self):
        """Initalise Trainer class"""
        self.Trainer = ColBERTTrainer
        self.TrainingArgument = ColBERTTrainingArgument
        self.model = ColBERT

    def _prepare_dataset(self):
        # TODO: Initalize dataset related fields
        pass

    def __preprocess_function(self, examples):
        # TODO: Do dataset preparation related thing, output inputs of the model
        pass

    def __postprocess_text(self, preds, labels):
        # TODO: Do post processing on preds
        # preds = [pred.strip() for pred in preds]
        # labels = [[label.strip()] for label in labels]
        # return preds, labels
        pass

    def get_training_args(self, request, dataset):
        return ColBERTTrainingArgument(
            request["model"],
            triples=os.path.join(dataset, "triples.train.colbert.jsonl"),
            queries_path=os.path.join(dataset, "queries.train.colbert.tsv"),
            collection_path=os.path.join(dataset, "corpus.train.colbert.tsv"),
            num_epochs=int(request["epochs"]),
        )

    def compute_metrics(self, eval_pred):
        # TODO: Output metrics of the models
        pass
        # predictions, labels = eval_pred
        # decoded_preds = self.tokenizer.batch_decode(
        #     predictions, skip_special_tokens=True
        # )
        # labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # decoded_preds, decoded_labels = self.__postprocess_text(
        #     decoded_preds, decoded_labels
        # )

        # result = self.metrics.compute(
        #     predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        # )

        # prediction_lens = [
        #     np.count_nonzero(pred != self.tokenizer.pad_token_id)
        #     for pred in predictions
        # ]
        # result["gen_len"] = np.mean(prediction_lens)

        # # return {k: round(v, 4) for k, v in result.items()}
        # return result

    def push_to_hub(self, trainer, save_path, hf_token=None):
        trainer.push_to_hub(save_path, hf_token)


@dataclass
class ColBERTTrainingArgument:
    triples = "data/triples.train.colbert.jsonl"
    queries_path = "data/queries.train.colbert.tsv"
    collection_path = "data/corpus.train.colbert.tsv"
    config = None

    def __init__(
        self, model_checkpoint, triples, queries_path, collection_path, num_epochs=1
    ):
        self.triples = triples
        self.queries_path = queries_path
        self.collection_path = collection_path
        self.num_epochs = num_epochs
        self.config = ColBERTConfig(
            bsize=1,
            lr=1e-5,
            warmup=3000,
            doc_maxlen=180,
            dim=128,
            attend_to_mask_tokens=False,
            nway=2,
            accumsteps=1,
            similarity="cosine",
            use_ib_negatives=True,
            checkpoint=model_checkpoint,
        )

        assert self.config.bsize % self.config.nranks == 0, (
            self.config.bsize,
            self.config.nranks,
        )
        self.config.bsize = self.config.bsize // self.config.nranks


class ColBERTTrainer:
    def __init__(self, model, args, callbacks=None) -> None:
        self.args = args
        self.model = model
        self.config = self.args.config
        self.ColBERT = model
        self.optimizer = None
        self.scheduler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model()

    def _init_model(self):
        self.model = self.ColBERT(
            name=self.config.checkpoint, colbert_config=self.config
        )
        self.model = self.model.to(self.device)
        self.model.train()  # Set the module in training mode.

    def _init_optimiser(self):
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.lr,
            eps=1e-8,
        )
        self.optimizer.zero_grad()

    def _init_scheduler(self):
        # print(
        #     f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps."
        # )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup,
            num_training_steps=self.config.maxsteps,
        )

    def _get_reader(self):
        if self.args.collection_path is not None:
            if self.config.reranker:
                self.reader = RerankBatcher(
                    self.config,
                    self.args.triples,
                    self.args.queries_path,
                    self.args.collection_path,
                    (0 if self.config.rank == -1 else self.config.rank),
                    self.config.nranks,
                )
            else:
                self.reader = LazyBatcher(
                    self.config,
                    self.args.triples,
                    self.args.queries_path,
                    self.args.collection_path,
                    (0 if self.config.rank == -1 else self.config.rank),
                    self.config.nranks,
                )
        else:
            raise NotImplementedError()

    def train(self):
        self._init_model()
        self._init_optimiser()
        if self.config.warmup is not None:
            self._init_scheduler()

        amp = MixedPrecisionManager(self.config.amp)
        labels = torch.zeros(self.config.bsize, dtype=torch.long, device=self.device)

        train_loss = None
        train_loss_mu = 0.999
        start_batch_idx = 0

        loss_df = pd.DataFrame(columns=["Epoch", "Step", "Loss"])
        loss_history = []
        checkpoint_paths = []

        for epoch in range(self.args.num_epochs):
            print(f"Starting Epoch {epoch + 1}/{self.args.num_epochs}")

            # Reinitialize or reset the reader here if necessary
            self._get_reader()

            for batch_idx, BatchSteps in zip(range(start_batch_idx, 256), self.reader):
                this_batch_loss = 0.0

                for batch in BatchSteps:
                    with amp.context():
                        try:
                            queries, passages, target_scores = batch
                            encoding = [queries, passages]
                        except:
                            encoding, target_scores = batch
                            encoding = [encoding.to(self.device)]

                        scores = self.model(*encoding)

                        if self.config.use_ib_negatives:
                            scores, ib_loss = scores

                        scores = scores.view(-1, self.config.nway)

                        if len(target_scores) and not self.config.ignore_scores:
                            target_scores = (
                                torch.tensor(target_scores)
                                .view(-1, self.config.nway)
                                .to(self.device)
                            )
                            target_scores = (
                                target_scores * self.config.distillation_alpha
                            )
                            target_scores = torch.nn.functional.log_softmax(
                                target_scores, dim=-1
                            )

                            log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                            loss = torch.nn.KLDivLoss(
                                reduction="batchmean", log_target=True
                            )(log_scores, target_scores)
                        else:
                            loss = torch.nn.CrossEntropyLoss()(
                                scores, labels[: scores.size(0)]
                            )

                        if self.config.use_ib_negatives:
                            if self.config.rank < 1:
                                print(
                                    "EPOCH ",
                                    epoch,
                                    " \t\t\t\t",
                                    loss.item(),
                                    ib_loss.item(),
                                )

                            loss += ib_loss

                        loss = loss / self.config.accumsteps

                    if self.config.rank < 1:
                        print_progress(scores)

                    amp.backward(loss)
                    this_batch_loss += loss.item()

                    if batch_idx % 500 == 0:
                        formatted_loss = "{:.6e}".format(
                            this_batch_loss
                        )  # Adjust the precision (e.g., 6) as needed
                        loss_history.append((epoch + 1, batch_idx + 1, formatted_loss))
                        loss_df = pd.DataFrame(
                            loss_history, columns=["Epoch", "Step", "Loss"]
                        )
                        loss_df.to_csv("loss_history.csv", index=False)

                train_loss = this_batch_loss if train_loss is None else train_loss
                train_loss = (
                    train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss
                )

                amp.step(self.model, self.optimizer, self.scheduler)

            if self.config.rank < 1:
                print_message(batch_idx, train_loss)
                epoch_save_path = f"./model_checkpoint/epoch_{epoch}/"
                os.makedirs(epoch_save_path, exist_ok=True)
                checkpoint_filename = f"checkpoint_batch_{batch_idx+1}.pt"
                full_checkpoint_path = os.path.join(
                    epoch_save_path, checkpoint_filename
                )
                manage_checkpoints(
                    self.config,
                    self.model,
                    self.optimizer,
                    batch_idx + 1,
                    savepath=full_checkpoint_path,
                    consumed_all_triples=True,
                )
                self.config.checkpoint = full_checkpoint_path + "/colbert/"
                checkpoint_paths.append(full_checkpoint_path + "/colbert/")

        if self.config.rank < 1:
            print_message("#> Done with all triples!")
            ckpt_path = manage_checkpoints(
                self.config,
                self.model,
                self.optimizer,
                batch_idx + 1,
                savepath="./output/model_checkpoint/",
                consumed_all_triples=True,
            )

        self.checkpoint_paths = checkpoint_paths
        self.losses = loss_history

        print(self.losses)

    def predict(self):
        # FIXME: Implement prediction
        pass

    @property
    def log_history(self):
        return dict(losses=self.losses, checkpoint_paths=self.checkpoint_paths)

    def push_to_hub(self, save_path, hf_token=None):
        api = HfApi(endpoint="https://huggingface.co", token=hf_token)
        api.create_repo(repo_id=save_path, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=self.checkpoint_paths[-1],
            repo_id=save_path,
            repo_type="model",
        )
