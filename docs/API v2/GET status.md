# Endpoint

GET /status-v2/

# Request

### Headers

- workflow_id
- task_id

```bash
curl --location 'localhost:8000/status-v2/'
```

# Response

### task_id is provided

```json
{
  "task_id": "UUID",
  "status": "",
  "percentage": ""
}
```

### workflow_id is provided

```json
{
  "workflow_id": "UUID",
  "status": "",
  "task_ids": []
}
```

### If both task_id and workflow_id is provided

```json
{
  "workflow": {
    "workflow_id": "UUID",
    "status": "",
    "task_ids": []
  },
  "task": {
    "task_id": "UUID",
    "status": "",
    "percentage": ""
  }
}
```
