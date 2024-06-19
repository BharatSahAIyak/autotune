### Request

```bash
curl --location --request PUT 'localhost:8000/workflow/?workflow_id=<WORKFLOW_ID>&action="duplicate"'
```

### Response

```json
{
  "message": "Saved/Created/Duplicated the workflow locally",
  "workflow_id": "<WORKFLOW_ID"
}
```

- action=duplicate => creates a new workflow with the same local ID
- workflow_id => save to a workflow
- No workflow_id and action => create a new workflow
