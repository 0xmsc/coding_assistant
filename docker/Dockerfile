FROM python:3.12-slim
ARG CODING_ASSISTANT_REVISION=unknown

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    CODING_ASSISTANT_REVISION="${CODING_ASSISTANT_REVISION}" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        docker-cli \
        git \
        jq \
        poppler-utils \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --locked --no-editable --no-install-project

COPY src ./src
COPY alembic.ini .
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
RUN uv sync --locked --no-editable

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home app \
    && mkdir /data \
    && chown app:app /data

USER app

CMD ["/bin/sh", "/app/entrypoint.sh"]
