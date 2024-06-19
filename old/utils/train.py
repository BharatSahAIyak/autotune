from transformers import TrainerCallback

from .tasks import Seq2Seq, TextClassification


class CeleryProgressCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task

    def on_log(self, args, state, control, logs, **kwargs):
        self.task.update_state(state="TRAINING", meta=state.log_history)


def get_task_class(task):
    tasks = {"text_classification": TextClassification, "seq2seq": Seq2Seq}

    task_class = tasks.get(task)
    return task_class
