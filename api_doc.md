# Post `/data/view` - Viewing Data

### Header

    X-OpenAI-Key

### Params:

    prompt: str
    num_samples: int
    task: Literal['text_classification', 'seq2seq']
    num_labels: Optional[int] = 2

### Return

    {"status": "Accepted", "task_id": task_id}

# POST `/data/` - for generating and pushing data to hugging face

### Headers

     X-OpenAI-Key : OpenAI API key
     X-HuggingFace-Key : Hugging Face API Key

### Params:

    num_samples: int
    repo: str
    split: Optional [list[int]] = [80, 10, 10]
    labels: Optional [list[dict]]
    valid_data: Optional [list[dict]]
    invalid_data: Optional [list[dict]]

### return

    {"status": "Accepted", "task_id": task_id}

# PUT `/data` - For updating an already existing data card on Hugging Face

### Headers

    X-OpenAI-Key
    X-HuggingFace-Key

### Params:

    num_samples: int
    repo: str
    split: Literal['train', 'validation', 'test']
    labels: Optional [list[dict]]
    valid_data: Optional [list[dict]]
    invalid_data: Optional [list[dict]]

### Return

    {"status": "Accepted", "task_id": task_id}

# POST `/question/` - for generating and pushing questions to hugging face

### Headers

     X-OpenAI-Key : OpenAI API key
     X-HuggingFace-Key : Hugging Face API Key

### Params:

    num_samples: int - The number of questions to generate
    repo: str
    split: Optional [list[int]] = [80, 10, 10]
    content: str  - The content over which we want to generate questions
    index: int  - The index of the original chunk
    multiple_chunks: bool - Want to use prompt for single chunk or 2 chunks combined

### return

    {"status": "Accepted", "task_id": task_id}

# PUT `/question` - For updating an already existing data card on Hugging Face

### Headers

    X-OpenAI-Key
    X-HuggingFace-Key

### Params:

    num_samples: int
    repo: str
    split: Literal['train', 'validation', 'test']
    content: str  - The content over which we want to generate questions
    index: int  - The index of the original chunk
    multiple_chunks: bool - Want to use prompt for single chunk or 2 chunks combined
    bulk_process: bool - True when we want to generate question-answers for multiple set of chunks

### Return

    {"status": "Accepted", "task_id": task_id}

# POST `/train` - For finetuning a model

### Header

    X-OpenAI-Key

### Params:

    dataset: str
    model: str
    epochs: Optional[float] = 1
    save_path: str
    task: Literal['text_classification', 'seq2seq']
    version: Optional[str] = "main"

### Return

    {'task_id': str(task.id)}

# GET `/track/{task_id}` - Tracks task based on task id and returns the result associated with the task

### Params:

    task_id: str

### Return

    {"response": \_\_\_}
