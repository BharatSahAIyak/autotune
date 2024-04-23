1. User Management - link with userService later
2. Workflow
    - Create Workflow - `PUT /workflow`
    - Iterate - `POST /workflow/<id>/iterate`
    - Update Prompt/Provide Feedback for Workflow - `POST /wokflow/<id>`
    - Generate - `POST /workflow/<id>/generate` => Dataset Local ID
    - Save - `PUT /workflow`
    - Duplicate - `PUT /workflow`
    - List all workflows - `GET /workflow`
    - Search - `GET /wokflow/q={tag or name}`
3. Dataset
    - Save to huggingface - `PUT /dataset/<localID>`
4. Models
    - Define a training task <WIP>
    - Train <WIP>