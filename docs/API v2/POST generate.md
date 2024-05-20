# Endpoint

/workflow-v2/generate/{workflow_id}

# Request

### Headers

- User Auth

# Response

```json
{
  "expected_cost": "",
  "task_id": ""
}
```

## Implementation Details

The prompt which goes to GPT will only include those examples which have been provided a reason.
