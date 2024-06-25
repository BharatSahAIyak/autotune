from functools import partial
from workflow.training.tasks import Tasks
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, DistilBertForTokenClassification
import pandas as pd
import evaluate
import torch
import ast


class NamedEntityRecognition(Tasks):

    def __init__(self, model_name: str, version: str, args):
        super().__init__("ner", model_name, version)
        self.metrics = {
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
            "f1": evaluate.load("f1"),
        }
        if "entity_labels" in args:
            self.entity_labels = args["entity_labels"]  # List of entity labels/Tags
        else:
            self.entity_labels = None

        self.labels = None  # List of labels for training - After adding "B-" and "I-" to entity labels
        self.tokenized_dataset = {"train": None, "test": None}

    def _load_model_requirements(self):
        self.model = DistilBertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.Trainer = partial(
            Trainer,
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._get_compute_metrics(),
        )

    def load_dataset(self, dataset, **kwargs):
        train_df, test_df = self._prepare_dataset(dataset)
        if self.entity_labels is None:
            self.entity_labels = list(self._extract_entity_labels(train_df))
        extract_values = self._make_extraction(self.entity_labels)
        self.labels = self._get_labels()
        train_df[(self.entity_labels)] = train_df["Output"].apply(
            lambda x: pd.Series(extract_values(x))
        )
        test_df[self.entity_labels] = test_df["Output"].apply(
            lambda x: pd.Series(extract_values(x))
        )

        train_encodings = self._prepare_encodings(train_df)
        val_encodings = self._prepare_encodings(test_df)

        self.train_dataset = NERDataset(
            train_encodings, self.device, self.default_label_id
        )
        self.eval_dataset = NERDataset(
            val_encodings, self.device, self.default_label_id
        )

        self.tokenized_dataset["train"] = self.train_dataset
        self.tokenized_dataset["test"] = self.train_dataset

    def _extract_entity_labels(self, df):
        entity_labels = set()
        for output in df["Output"]:
            output_dict = ast.literal_eval(output)
            entity_labels.update(output_dict.keys())
        return entity_labels

    def get_training_args(self, req_data, dataset):
        self._load_model_requirements()
        training_args = TrainingArguments(
            output_dir=f"./tempp/results_NER_{req_data['task_id']}",
            num_train_epochs=req_data.get("num_train_epochs", 3),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"./logs/NER_{req_data['task_id']}",
            eval_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",  # Save at the end of each epoch
            save_total_limit=2,
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        return training_args

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = torch.argmax(predictions, axis=2)

        return {
            k: metric.compute(
                predictions=predictions.ravel(),
                references=labels.ravel(),
                average="weighted",
            )
            for k, metric in self.metrics.items()
        }

    def _get_compute_metrics(self):

        def compute_metrics(p):
            predictions, labels = p
            predictions = torch.argmax(torch.tensor(predictions), axis=2)

            return {
                k: metric.compute(
                    predictions=predictions.ravel(),
                    references=labels.ravel(),
                    average="weighted",
                )
                for k, metric in self.metrics.items()
            }

        return compute_metrics

    def _prepare_dataset(self, dataset_name):
        data_files = {"train": "train.csv", "test": "test.csv"}
        columns = ["Input", "Output"]
        df = load_dataset(dataset_name, data_files=data_files)
        train_df = pd.DataFrame(df["train"])[columns]
        test_df = pd.DataFrame(df["train"])[columns]
        train_df = train_df.rename(columns={"Input": "sentences"})
        test_df = test_df.rename(columns={"Input": "sentences"})
        return train_df, test_df

    def _make_extraction(self, entity_labels):

        def extract_values(output_str):
            output_dict = ast.literal_eval(output_str)
            entity_row = {}
            for entity in entity_labels:
                entity_row[entity] = output_dict.get(entity, None)

            return (val for val in entity_row.values())

        return extract_values

    def _get_labels(self):
        labels = ["O"]
        entity_labels = set(self.entity_labels)
        for l in entity_labels:
            labels.append("B-" + l)
            labels.append("I-" + l)

        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.default_label_id = self.label2id["O"]
        return labels

    def _create_tags(
        self, word_token_mapping, phrase, type_agri_term="PEST", tags=None
    ):
        if pd.isnull(phrase):
            return tags
        elif phrase == "":
            return tags
        else:
            phrase_words = phrase.split()
            # Iterate over the word_token_mapping to find the phrase
            for i in range(len(word_token_mapping) - len(phrase_words) + 1):
                # Check if current word matches the first word of the phrase
                if word_token_mapping[i][0] == phrase_words[0]:
                    match = True
                    for j in range(1, len(phrase_words)):
                        if (
                            i + j >= len(word_token_mapping)
                            or word_token_mapping[i + j][0] != phrase_words[j]
                        ):
                            match = False
                            break
                    # If we found a match, tag the tokens accordingly
                    if match:
                        for j, word in enumerate(phrase_words):
                            is_first_token = j == 0
                            for _, index in word_token_mapping[i + j][1]:
                                if is_first_token:
                                    tags[index] = "B-" + type_agri_term
                                    is_first_token = False
                                else:
                                    tags[index] = "I-" + type_agri_term

        return tags

    def _create_word_token_mapping(self, sentence, tokenized_list):
        # Create a copy of the tokenized_list removing [CLS], [SEP], and [PAD], but remember their original indices
        filtered_tokens_with_indices = [
            (token, idx)
            for idx, token in enumerate(tokenized_list)
            if token not in ["[CLS]", "[SEP]", "[PAD]"]
        ]

        word_token_mapping = []

        for word in sentence.replace(".", " .").replace("?", " ?").split():
            current_word_tokens = []
            reconstructed_word = ""

            while filtered_tokens_with_indices and reconstructed_word != word:
                token, original_idx = filtered_tokens_with_indices.pop(
                    0
                )  # Take the first token from the list
                current_word_tokens.append((token, original_idx))
                reconstructed_word += token.replace("#", "")

            if reconstructed_word != word:
                raise ValueError(
                    f"Token mismatch for word '{word}'! Failed to reconstruct from tokens."
                )

            word_token_mapping.append((word, current_word_tokens))

        return word_token_mapping

    def _prepare_encodings(self, df):
        sentences = df["sentences"].apply(lambda x: [x.lower()]).to_list()
        encodings = self.tokenizer(
            sentences,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encodings["labels"] = torch.full_like(
            encodings["input_ids"], self.default_label_id
        ).to(self.device)

        for i in range(0, len(df)):
            row = df.iloc[i]
            sentence = row["sentences"].lower()

            input_id = encodings["input_ids"][i]
            tokens = self.tokenizer.convert_ids_to_tokens(input_id)
            word_token_mapping = self._create_word_token_mapping(sentence, tokens)

            tags = ["O"] * len(tokens)
            for entity in self.entity_labels:
                tags = self._create_tags(
                    word_token_mapping,
                    row[entity],
                    type_agri_term=entity,
                    tags=tags,
                )

            current_labels = [self.label2id[tag] for tag in tags] + [
                self.label2id["O"]
            ] * (len(input_id) - len(tags))
            encodings["labels"][i] = torch.tensor(current_labels)

        return encodings

    def push_to_hub(self, trainer: Trainer, save_path, hf_token):
        trainer.model.push_to_hub(
            save_path, commit_message="pytorch_model.bin upload/update"
        )
        trainer.tokenizer.push_to_hub(save_path)


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, device, default_label_id):
        encodings["labels"] = torch.full_like(
            encodings["input_ids"], default_label_id
        ).to(
            device
        )  # Ensure labels are also moved
        self.encodings = encodings
        self.device = device

    def __getitem__(self, idx):
        return {key: val[idx].to(self.device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])
