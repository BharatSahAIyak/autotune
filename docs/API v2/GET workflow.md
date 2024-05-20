- Will return the workflows for a given user

# Endpoint

GET /workflow-v2 and GET /workflow-v2/{workflow_id}

# Request

### Headers

- User Auth

# Response

```json
[
  {
    "workflow_id": "<UUID>",
    "workflow_name": "String",
    "user_id": "<UUID>",
    "config": {
      "config_name": "QnA",
      "system_prompt": "You are a helpful data generation assistant working as a teacher. You are an expert in this field. Don't Hallucinate",
      "user_prompt_template": "{{workflow.user_prompt}}",
      "temperature": 1,
      "schema_example": {
        "question": "4 + 5",
        "choices": [
          {
            "text": "9",
            "score": "1"
          },
          {
            "text": "4",
            "score": "0"
          },
          {
            "text": "2",
            "score": "0"
          },
          {
            "text": "1",
            "score": "0"
          }
        ]
      }
    },
    "split": [100, 0, 0],
    "llm_model": "LLM Models",
    "tags": [],
    "user_prompt": "",
    "cost": "",
    "estimated_dataset_cost": "",
    "examples": [
      {
        "text": {
          "question": "question text",
          "choices": [
            {
              "text": "9",
              "score": "1"
            },
            {
              "text": "4",
              "score": "0"
            },
            {
              "text": "2",
              "score": "0"
            },
            {
              "text": "1",
              "score": "0"
            }
          ]
        }
      }
    ]
  }
]
```

Dataset Cost is estimated based on the tokens used in the previous iteration.
