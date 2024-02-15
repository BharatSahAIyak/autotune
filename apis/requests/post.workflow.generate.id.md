### Request

```bash
curl --location 'localhost:8000/workflow/generate/workflow_key_01' \
--header 'Content-Type: application/json' \
--data '{
    "system_prompt": "This is the prompt which tells GPT its role while generating responses and specific parameters to adhere to while responding.",
    "user_prompt":"defines the actual task to be performed. Additionally, it can be used to reinforce a constraint defined in system prompt",
    "num_samples": 1000,
    "labels": ["positive", "negative"],
    "valid_data": [
        {"input": "The customer support team was very helpful", "output": "positive","reason":""},
        {"input": "The price of this item is outrageous.", "output": "negative","reason":""}
    ],
    "invalid_data": [
        {"input": "This item exceeded my expectations", "output": "negative","reason":""},
        {"input": "I'\''ve had nothing but issues with this product.", "output": "positive","reason":""}
    ],
    "llm_model":"gpt-4-0125-preview",
    "model":"this is the model the response JSON needs to adhere to"
}'
```

### Response

```json
{
  "workflow_name": "<WORKFLOW_NAME>",
  "workflow_id": "<UUID>",
  "status": "Successfully started generation"
}
```
