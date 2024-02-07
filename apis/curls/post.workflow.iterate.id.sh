curl --location 'localhost:8000/workflow/iterate/workflow_key_01' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "Additional context given to GPT regarding the exact specification being used for labelling the data",
    "labels": ["positive", "negative"],
    "valid_data": [
        {"input": "The customer support team was very helpful", "output": "positive","reason":""},
        {"input": "The price of this item is outrageous.", "output": "negative","reason":""}
    ],
    "invalid_data": [
        {"input": "This item exceeded my expectations", "output": "negative","reason":""},
        {"input": "I'\''ve had nothing but issues with this product.", "output": "positive","reason":""}
    ],
    "llm_model": "gpt-4-0125-preview"
}'