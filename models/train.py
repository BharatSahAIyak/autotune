from pydantic import BaseModel
from typing import Optional


class ModelData(BaseModel):
    dataset: str
    model: str
    epochs: Optional[int] = 1
    save_path: str