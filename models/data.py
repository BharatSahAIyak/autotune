from typing import Literal, Optional

from pydantic import BaseModel


class GenerationAndCommitRequest(BaseModel):
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]
    labels: Optional[list[str]]
    valid_data: Optional[list[dict]]
    invalid_data: Optional[list[dict]]


class GenerationAndUpdateRequest(BaseModel):
    num_samples: int
    repo: str
    split: Literal["train", "validation", "test"]
    labels: Optional[list[str]]
    valid_data: Optional[list[dict]]
    invalid_data: Optional[list[dict]]


class ChatViewRequest(BaseModel):
    prompt: str
    num_samples: int
    task: Literal["text_classification", "seq2seq"]
    num_labels: Optional[int] = 2


class QuestionCreationRequest(BaseModel):
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    content: list[str]
    index: Optional[int]  # the row of the content in the csv file
    model: Optional[str] = "gpt-3.5-turbo"
    multiple_chunks: bool = False
    combined_index: Optional[str] = None  # required when sending in multiple chunks


class QuestionUpdationRequest(BaseModel):
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    content: list[str]
    index: Optional[int]  # the row of the content in the csv file
    model: Optional[str] = "gpt-3.5-turbo"
    multiple_chunks: bool = False
    combined_index: Optional[str] = None  # required when sending in multiple chunks
    bulk_process: bool = False
