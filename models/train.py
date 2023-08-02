from pydantic import BaseModel
from typing import Optional, Literal


class ModelData(BaseModel):
    dataset: str
    model: str
    epochs: Optional[float] = 1
    save_path: str
    task: Literal['text_classification', 'seq2seq']
    version: Optional[str] = "main"