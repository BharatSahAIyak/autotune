from typing import Literal, Optional

from pydantic import BaseModel


class ModelData(BaseModel):
    dataset: str
    model: str
    epochs: Optional[float] = 1
    save_path: str
    task: Literal["text_classification", "seq2seq"]
    version: Optional[str] = "main"
