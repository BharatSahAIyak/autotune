from .utils import dataset_generation, track_task, sample_generation
import time

sample_reponse_v1 = sample_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=5,
    repo="test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
)

sample_reponse_v2 = sample_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=5,
    repo="test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
    valid_data = sample_reponse_v1["response"]["data"][:len(sample_reponse_v1["response"]["data"]) // 2],
    invalid_data = sample_reponse_v1["response"]["data"][len(sample_reponse_v1["response"]["data"]) // 2:]
)

dataset_response = dataset_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=20,
    repo="test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
    valid_data = sample_reponse_v2["response"]["data"]
)

print(dataset_response)

while True:
    task_id = dataset_response["task_id"]
    task = track_task(task_id)
    print(task)
    if task["status"] == "Completed":
        break
    print(
        "Still waiting for task {task_id} to complete...".format(task_id=task_id),
    )
    time.sleep(2)
