from workflow.training.tasks import Tasks
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
import logging
from functools import partial

logger = logging.getLogger(__name__)

class EmbeddingFineTuning(Tasks):
    def __init__(self, model_name: str, version: str, args):
        super().__init__("embedding_finetuning", model_name, version)
        self.guide_model = args.get("guide_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.args = args

    def load_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        if "train" not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        
        train_dataset = dataset["train"]
        
        if "question" not in train_dataset.column_names or "positive" not in train_dataset.column_names:
            raise ValueError("Dataset must contain 'question' and 'positive' columns")

        split_dataset = train_dataset.train_test_split(test_size=0.1)
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]

        self._prepare_dataset()
        self._load_model()
        return split_dataset

    def _prepare_dataset(self):
        self.val_queries = {idx: item['question'] for idx, item in enumerate(self.eval_dataset)}
        self.val_corpus = {idx: item['positive'] for idx, item in enumerate(self.eval_dataset)}
        self.val_relevant_docs = {idx: set([idx]) for idx in range(len(self.eval_dataset))}

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)
        self.guide = SentenceTransformer(self.guide_model)
        
        self.train_loss = losses.GISTEmbedLoss(model=self.model, guide=self.guide, temperature=self.args.get("temperature", 0.01))
        
        self.evaluator = InformationRetrievalEvaluator(
            self.val_queries, self.val_corpus, self.val_relevant_docs, name='val_evaluator',
            mrr_at_k=[5, 10, 100], ndcg_at_k=[5, 10, 100], accuracy_at_k=[1, 5, 10],
            precision_recall_at_k=[1, 5, 10]
        )

        self.Trainer = partial(
            SentenceTransformerTrainer,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            evaluator=self.evaluator,
            loss=self.train_loss,
        )

    def _load_model_requirements(self):
        pass

    def compute_metrics(self, eval_pred):
        pass

    def get_training_args(self, req_data, dataset):
        return SentenceTransformerTrainingArguments(
            output_dir=f"./results_{req_data['task_id']}",
            num_train_epochs=req_data["epochs"],
            per_device_train_batch_size=self.args.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=self.args.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=self.args.get("gradient_accumulation_steps", 4),
            evaluation_strategy="steps",
            eval_steps=self.args.get("eval_steps", 500),
            logging_steps=self.args.get("logging_steps", 500),
            save_steps=self.args.get("save_steps", 500),
            learning_rate=self.args.get("learning_rate", 1e-5),
            weight_decay=self.args.get("weight_decay", 0.01),
            warmup_ratio=self.args.get("warmup_ratio", 0.1),
            save_total_limit=3,
            load_best_model_at_end=True,
        )

    def push_to_hub(self, trainer, save_path, hf_token=None):
        trainer.model.push_to_hub(
            repo_id=save_path, commit_message="pytorch_model.bin upload/update", token=hf_token, exist_ok=True
        )
