import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi

from old.utils import split_data


class HFException(Exception):
    pass


DATASET = [
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


async def push_dataset_to_hf(
    data,
    repo,
    huggingface_key,
    split=[80, 10, 10],
):
    train, test, val = {}, {}, {}
    train["data"], val["data"], test["data"] = split_data(data["data"], split)

    try:
        hf_api = HfApi(endpoint="https://huggingface.co", token=huggingface_key)
    except Exception as e:
        detail = f"Failed to connect to HF: {str(e)}"
        return HFException(detail)

    try:
        hf_api.create_repo(repo_id=repo, repo_type="dataset")
    except Exception as e:
        detail = f"Failed to create repo in HF: {str(e)}"
        return HFException(detail)

    for split, d in zip(["train", "validation", "test"], [train, val, test]):
        df = pd.DataFrame(d["data"])
        csv_data = df.to_csv()
        file_data = csv_data.encode("utf-8")
        operation = CommitOperationAdd(f"{split}.csv", file_data)
        try:
            hf_api.create_commit(
                repo_id=repo,
                operations=[operation],
                commit_message=f"Adding {split} csv file",
                repo_type="dataset",
            )
        except Exception as e:
            detail = f"Failed to commit to repo in HF: {str(e)}"
            return HFException(detail)
