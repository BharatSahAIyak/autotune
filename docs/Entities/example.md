- Example provided by the user must have the following fields

1. Text: This must be a JSON object
2. Label: What kind of example is this in the given context. String field
3. Reason: WHy we are giving the label to the example in the given context

### Sample example

```json
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
```
