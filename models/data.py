from typing import Literal, Optional

from pydantic import BaseModel


class GenerationAndCommitRequest(BaseModel):
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]
    task: Literal["text_classification", "seq2seq"]
    labels: Optional[list[str]]
    valid_data: Optional[list[dict]]
    invalid_data: Optional[list[dict]]


class GenerationAndUpdateRequest(BaseModel):
    num_samples: int
    repo: str
    split: Literal["train", "validation", "test"]
    task: Literal["text_classification", "seq2seq"]
    valid_data: Optional[list[dict]]
    invalid_data: Optional[list[dict]]


class ChatViewRequest(BaseModel):
    prompt: str
    num_samples: int
    task: Literal["text_classification", "seq2seq"]
    num_labels: Optional[int] = 2
