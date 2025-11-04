# syntax=docker/dockerfile:1.5

FROM python:3.11-slim as base

ENV POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml README.md ./
RUN poetry install --no-root --no-interaction --no-ansi

COPY . .

CMD ["python", "-m", "cloud.training.pipelines.daily_retrain"]
