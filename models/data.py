from pydantic import BaseModel
from typing import Optional


class GenerationAndCommitRequest(BaseModel):
    prompt: str
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]