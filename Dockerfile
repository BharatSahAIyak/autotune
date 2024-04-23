FROM python:3.10

WORKDIR /app

RUN pip install poetry==1.5.1 && poetry config virtualenvs.create false 

COPY pyproject.toml poetry.lock ./

RUN poetry install 

COPY . .
