Creates a workflow

# Endpoint

POST /workflow-v2

# Request

```json
{
  "workflow_name": "String",
  "split": [100, 0, 0],
  "llm_model": "LLM Models",
  "tags": [],
  "user_prompt": "",
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
```

Optional fields: Examples and Tags

### LLM Models

```
[
"gpt-4-0125-preview",
"gpt-4-turbo",
"gpt-4-turbo-preview",
"gpt-4-1106-preview",
"gpt-4-vision-preview",
"gpt-3.5-turbo-0125",
"gpt-3.5-turbo",
]
```

# Response

```json
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
  "prompt_id": "",
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
```
