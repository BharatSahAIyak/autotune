### Request

```bash
curl --location --request PUT 'localhost:8000/dataset?workflow=<WORKFLOW_ID>'
```

### Response

```json
{
  "message": "Saved the dataset to HuggingFace",
  "dataset_id": "<DATASET_ID>"
}
```
