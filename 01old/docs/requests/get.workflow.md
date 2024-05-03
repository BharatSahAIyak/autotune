### Request

```bash
curl --location 'localhost:8000/workflow?name=test01'
```

### Response

```json
[
  {
    "name": "workflow_name_01",
    "id": "<WORKFLOW_ID_01>",
    "created_at": "<Creation Time>",
    "updated_at": "<Last Updated Time>",
    "user": "user_01",
    "task": "task_01",
    "labels": ["label01", "label02"],
    "examples": [
      {
        "text": "text for example 01",
        "label": "<LABEL>",
        "reason": "<REASON>"
      }
    ],
    "model": "gpt-3.5-turbo",
    "dataset_size": 100,
    "split": [80, 10, 10]
  }
]
```
