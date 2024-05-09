Data which can be retrieved against a workflow_id

## Workflow

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

- Tags: These are the tags against which a user can filter workflows they have created
- Split: An array giving the ratio splitting the dataset into training, validation and testing. the default is [80,10,10]
- Examples: always needs to be a JSON object. it can be in any format. Needs to confirm to the same structure as defined by the user in the config's `schema_example`
- User Prompt: This is the prompt which will get replaced in place of `{{workflow.user_prompt}}` of the `user_prompt_template` in a given config.
- Cost: maintains the actual cost incurred by the workflow
- Expected workflow cost: NULL till the first iteration, post that, based on the cost incurred in the last iteration performed.

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
