- Will return the examples generated based on the user_prompt defined earlier and the examples which have a reason associated with them
- includes examples in the prompt which have been previously generated and given a reason and label

# Endpoint

POST /workflow-v2/iterate/{workflow_id}

# Request

### Headers:

- Auth User

### Request Body

```json
{
  "examples": [
    {
      "text": {
        "question": "",
        "answer": ""
      },
      "label": "Sample label",
      "reason": "sample reason"
    }
  ],
  "user_prompt": "",
  "num_samples": 20
}
```

All fields in this are optional.
If a user_prompt is specified in this request, it will replace the older user_prompt.
num_samples: Optional with default 10

# Response

```json
{
  "data": [
    {
      "example_id": "UUID",
      "text": {},
      "label": "",
      "reason": ""
    },
    {
      "example_id": "UUID",
      "text": {},
      "label": "",
      "reason": ""
    }
  ]
}
```

## Internal implementation detail:

- The actual request which goes to OpenAI has the following format:

### System Prompt

```
config.system_prompt
Ensure that the JSON output adheres to the provided structure
Pydantic classes for json structure:
<PYDANTIC MODEL>"
```

### User Prompt

```
{user_prompt_template}.

Based on the examples below, refine and generate 10 new examples.

{
    {
        "text":{},
        "label":"",
        "reason":""
    }
}
```
