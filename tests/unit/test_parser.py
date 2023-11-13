import json

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import dirtyjson
from utils.data import parse

OUTPUT = """[
    {"text": "I absolutely love this product!", "label": "positive"},
    {"text": "This is the worst experience I've ever had.", "label": "negative"},
    {"text": "The service was exceptional and fast.", "label": "positive"},
    {"text": "I can't believe how affordable this item is.", "label": "positive"},
    {"text": "The quality of this product is top-notch.", "label": "positive"},
    {
        "text": "I'm extremely disappointed with the customer service.",
        "label": "negative",
    },
    {"text": "The shipping took way longer than expected.", "label": "negative"},
    {
        "text": "This is the best purchase I've made in a long time.",
        "label": "positive",
    },
    {"text": "I would never recommend this product to anyone.", "label": "negative"},
    {"text": "The packaging was damaged when it arrived.", "label": "negative"},
    {"text": "I'm amazed at how well this product works.", "label": "positive"},
    {"text": "The customer support team was very helpful.", "label": "positive"},
    {"text": "This item exceeded my expectations.", "label": "positive"},
    {"text": "I've had nothing but issues with this product.", "label": "negative"},
    {"text": "The price of this item is outrageous.", "label": "negative"},
    {"text": "I'm completely satisfied with my purchase.", "label": "positive"},
    {"text": "This company has the worst return policy.", "label": "negative"},
    {"text": "The color of this item is not as described.", "label": "negative"},
    {
        "text": "I'm very impressed with the quality of this product.",
        "label": "positive",
    },
    {"text": "The instructions for this item were unclear.", "label": "negative"},
]
"""


class LabeledDataset(BaseModel):
    text: str = Field(..., description="The text of the sample generated")
    label: str = Field(..., description="The label of the sample generated")


def test_parser():
    parser = PydanticOutputParser(pydantic_object=LabeledDataset)
    parsed = parse(OUTPUT, parser)
    print(parsed)

    assert len(parsed) == 20
