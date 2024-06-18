from workflow.training.classification import TextClassification
from workflow.training.colbert import Colbert
from workflow.training.ner import NamedEntityRecognition


def get_task_class(task):
    tasks = {
        "text_classification": TextClassification,
        "embedding": Colbert,
        "ner": NamedEntityRecognition,
    }

    task_class = tasks.get(task)
    return task_class
