### Config

This defines the basic structure of the requests which will be made to openAI.

```json
{
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
}
```
