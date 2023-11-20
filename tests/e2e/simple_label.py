import os
import time

from tasks.data import check_if_hf_repo_exists, delete_hf_repo

from .utils import dataset_generation, model_training, sample_generation, track_task

HF_USERNAME = os.getenv("HF_USERNAME")
HF_SEPARATOR = "/"
HF_API_KEY = os.getenv("HF_API_KEY")

sample_reponse_v1 = sample_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=5,
    repo=HF_USERNAME + HF_SEPARATOR + "test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
)

sample_reponse_v2 = sample_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=5,
    repo=HF_USERNAME + HF_SEPARATOR + "test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
    valid_data=sample_reponse_v1["response"]["data"][
        : len(sample_reponse_v1["response"]["data"]) // 2
    ],
    invalid_data=sample_reponse_v1["response"]["data"][
        len(sample_reponse_v1["response"]["data"]) // 2 :
    ],
)

dataset_response = dataset_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=1000,
    repo=HF_USERNAME + HF_SEPARATOR + "test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
    valid_data=sample_reponse_v2["response"]["data"],
)

print(dataset_response)

while True:
    task_id = dataset_response["task_id"]
    task = track_task(task_id)
    print(task)
    if task["status"] == "Completed":
        print(
            "Deleting repo {repo} in 30 seconds".format(
                repo=HF_USERNAME + HF_SEPARATOR + "test"
            )
        )
        time.sleep(30)
        repo = HF_USERNAME + HF_SEPARATOR + "test" + "-" + task_id

        # delete_hf_repo(repo, HF_API_KEY)
        # is_exists = check_if_hf_repo_exists(repo, HF_API_KEY)
        # if is_exists:
        #     print("Couldn't delete repo earlier")
        break
    print(
        "Still waiting for task {task_id} to complete...".format(task_id=task_id),
    )
    time.sleep(2)
