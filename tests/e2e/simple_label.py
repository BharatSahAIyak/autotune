import time

from .utils import dataset_generation, track_task

dataset_response = dataset_generation(
    prompt="Create a dataset for text classification for sentiment analysis. The sentiments should be either positive or negative.",
    num_samples=10000,
    repo="test",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["positive", "negative"],
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
