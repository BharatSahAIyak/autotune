from transformers import TrainerCallback


class CeleryProgressCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task

    def on_log(self, args, state, control, logs, **kwargs):
        self.task.update_state(state='TRAINING', meta=state.log_history)


def tokenize_data(batch, tokenizer):
    inputs = tokenizer(batch['Input'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    outputs = tokenizer(batch['Output'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    inputs['labels'] = outputs['input_ids']
    return inputs