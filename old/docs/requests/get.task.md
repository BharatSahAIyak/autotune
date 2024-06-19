### Request

```bash
curl --location 'localhost:8000/task?name="label"'
```

### Response

```json
[
  {
    "name": "label",
    "format": {
      "qna": [
        {
          "text": "<>",
          "label": "<LABEL>"
        }
      ]
    }
  }
]
```
