from pydantic import BaseModel, Field


class QAPair(BaseModel):
    question: str = Field(..., description="The generated question")
    answer: str = Field(..., description="The generated answer")


class QAResponse(BaseModel):
    qa_pairs: list[QAPair] = Field(..., description="List of question and answer pairs")
