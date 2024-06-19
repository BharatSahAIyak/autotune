import os
import time

from .utils import model_training, sample_generation, track_task

HF_USERNAME = os.getenv("HF_USERNAME")
HF_SEPARATOR = "/"
HF_API_KEY = os.getenv("HF_API_KEY")
task_id = "2fd961bc-aec9-4cec-8f2a-9e5ce676127a"

repo = HF_USERNAME + HF_SEPARATOR + "test" + "-" + task_id

# Training
model_training_task_id = model_training(
    dataset=repo,
    model="bert-base-uncased",  # HF Based models only
    epochs=10,
    save_path=repo,
    task="text_classification",
)

print(model_training_task_id)

while True:
    task_id = model_training_task_id["task_id"]
    task = track_task(task_id)
    print(task)
    if task["status"] == "Completed":
        break
    print(
        "Still waiting for task {task_id} to complete...".format(task_id=task_id),
    )
    time.sleep(2)
