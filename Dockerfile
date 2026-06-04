FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates docker-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --system .

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home app

USER app

CMD ["coding-assistant-manager", "--help"]
