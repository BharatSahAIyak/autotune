from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
)
import logging
import torch

logger = logging.getLogger(__name__)


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
        # self._load_model_requirements()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    def load_dataset(self, dataset, **kwargs):
        """Load the dataset and prepare it for training. If Model requires any specific dataset preparation, do it here."""
        pass

    @abstractmethod
    def get_training_args(self, req_data, dataset):
        """Return the TrainingArguments for the model which will be passed to the Trainer class."""
        pass

    @abstractmethod
    def compute_metrics(self, eval_pred):
        pass

    @abstractmethod
    def push_to_hub(self, trainer, save_path):
        pass
