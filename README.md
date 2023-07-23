# AutoTuneNLP
A comprehensive toolkit for seamless data generation and fine-tuning of NLP models, all conveniently packed into a single block.

# Setup

Clone the repo and cd to project root.

## Environment
1. Create and activate venv. Ex:
(on windows)
```
python -m venv venv
.\venv\Scripts\activate
```
2. This project uses [poetry](https://python-poetry.org/docs/basic-usage/).
```
pip install poetry==1.5.1
poetry install
```
3. Install [torch](https://pytorch.org/) from here based on your cuda version for GPU support else skip this step. Ex:
```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## API
1. Start your docker engine and run a redis image on port 6379.
```
docker run --name autotunenlp-redis -p 6379:6379 -d redis
```
2. Start celery worker.
```
celery -A worker worker --loglevel=info
```
- If you are running on windows, the above command won't work since celery is not supported on windows, but you can use the below command for testing (caveat: it's capabilities are lost).
```
celery -A worker worker --loglevel=info --pool=solo
```
3. Specify a port number and start the application.
```
uvicorn main:app --port PORT_NUMBER --reload
```


