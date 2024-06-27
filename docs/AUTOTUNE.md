# INTRODUCTION

## Entities in the system

### WORKFLOWS

Every action taken by a user in autotune is mapped to a workflow. Autotune has two broad functions which is housed in the same place: `Synthetic Data Generation` and `Model Training`. These are built as two separate functions, with interoperability provided in autotune.
Based on this, there are two types of workflows in autotune: `training` and `complete`. A complete workflow indicates that the entire process from the data generation to the training is being performed at autotune. A `training` workflow can be used to perform a subset of operations of a complete workflow.
In training workflows, user can provide a HuggingFace dataset for training/fine tuning a model of a model.

Autotune has the assumption that a given user will have only one workflow for training a given model type like `Text Classification`, `Named entity recognition`, etc.

### CONFIG

Configs are re-usable components which provides metadata and various other fixed aspects of a workflow.

Overall config items which can be stored are:

- temperature: OpenAI temperature used in dataset generation
- system_prompt: System prompt which is passed to OpenAI API.
- user_prompt_template: A template with replaceable values according to workflow needs.
- schema_example: A sample JSON which we want the generated data to follow. We can create dynamic models of any structure we like, with validation using dynamically created pydantic models

### TASKS

### TRAINING

## Development Journey

## Models Supported

- Text Classification
- Colbert training
- Force Alignment

# SETUP

## API specifications

There are two versions of the APIs, with the core functionality accross both the APIs the same

### POST /v1/workflow/config

- REQUEST:

- RESPONSE:

### POST /v1/workflow/create

- REQUEST:

- RESPONSE:

### POST /v1/workflow/iterate/<UUID>

- REQUEST:

- RESPONSE:

### POST /v1/workflow/generate/<UUID>

- REQUEST:

- RESPONSE:

### POST /v1/workflow/status/<WORKFLOW_ID>

- REQUEST:

- RESPONSE:

### GET /health

- REQUEST:

```bash
curl --location 'localhost:8000/health'
```

- RESPONSE:

```json
{
  "health": "healthy",
  "upstreamServices": [
    {
      "name": "OpenAI API",
      "type": "external",
      "impactMessage": "Synthetic data generation will be impacted",
      "status": {
        "isAvailable": true
      },
      "endpoint": "openai.com",
      "sla": null
    },
    {
      "name": "Redis",
      "type": "internal",
      "impactMessage": "Caching, Data Generation and Model training impacted",
      "status": {
        "isAvailable": true
      },
      "sla": null
    },
    {
      "name": "Celery Workers",
      "type": "internal",
      "impactMessage": "Task processing will be impacted",
      "status": {
        "isAvailable": true
      },
      "sla": null
    },
    {
      "name": "PostgreSQL",
      "type": "internal",
      "impactMessage": "All core functionalities impacted",
      "status": {
        "isAvailable": true
      },
      "sla": null
    },
    {
      "name": "Hugging Face API",
      "type": "external",
      "impactMessage": "All core functionalities of autotune impacted",
      "status": {
        "isAvailable": true
      },
      "endpoint": "huggingface.co",
      "sla": null
    },
    {
      "name": "Minio",
      "type": "internal",
      "impactMessage": "Mass prompt handling will be impacted, along with download of synthetic data json and csv",
      "status": {
        "isAvailable": true
      },
      "sla": null
    }
  ]
}
```

### GET /health/ping

- REQUEST:

```bash
curl --location 'localhost:8000/health/ping'
```

- RESPONSE:

```json
{
  "status": "ok",
  "details": {
    "autotune": {
      "status": "up"
    }
  }
}
```
