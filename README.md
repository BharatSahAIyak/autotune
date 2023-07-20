# AutoTuneNLP
A comprehensive toolkit for seamless data generation and fine-tuning of NLP models, all conveniently packed into a single block.

# Setup

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

## API
1. Start your docker engine and run a redis image on port 6379.
```
docker run --name autotunenlp-redis -p 6379:6379 -d redis
```
2. Specify a port number and start the application.
```
uvicorn main:app --port PORT_NUMBER --reload
```


