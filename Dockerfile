FROM python:3.10

WORKDIR /app

RUN pip install poetry==1.6.0 && poetry config virtualenvs.create false 

COPY pyproject.toml poetry.lock ./

# RUN poetry install 

COPY . .

HEALTHCHECK --interval=5m --timeout=3s \
  CMD curl -f http://localhost/ || exit 1 