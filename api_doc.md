# for viewing data
@app.post("/data/view")
Header(name="X-OpenAI-Key")
- params:
    prompt: str
    num_samples: int
    task: Literal['text_classification', 'seq2seq']
    num_labels: Optional[int] = 2
- return {"status": "Accepted", "task_id": task_id}

# for generating and pushing data to hugging face
@app.post("/data")
Header(name="X-OpenAI-Key")
Header(name="X-HuggingFace-Key")
- params:
    prompt: str
    num_samples: int
    repo: str
    split: Optional[list[int]] = [80, 10, 10]
    task: Literal['text_classification', 'seq2seq']
    num_labels: Optional[int] = 2
- return {"status": "Accepted", "task_id": task_id}

# for update already in existing data card on hugging face
@app.put("/data")
Header(name="X-OpenAI-Key")
Header(name="X-HuggingFace-Key")
- params:
    prompt: str
    num_samples: int
    repo: str
    split: Literal['train', 'validation', 'test']
    task: Literal['text_classification', 'seq2seq']
    num_labels: Optional[int] = 2
- return {"status": "Accepted", "task_id": task_id}

# for finetuning a model
@app.post("/train")
Header(name="X-HuggingFace-Key")
- params:
    dataset: str
    model: str
    epochs: Optional[float] = 1
    save_path: str
    task: Literal['text_classification', 'seq2seq']
    version: Optional[str] = "main"
- return {'task_id': str(task.id)}

# tracks task based on task id and returns the result associated with the task
@app.get("/track/{task_id}")
- params:
    task_id: str
- return {"response": ___}
