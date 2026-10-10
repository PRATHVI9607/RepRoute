# syntax=docker/dockerfile:1
# Lab plane: the full locked research environment (CPU torch).
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git make unar iproute2 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra lab --extra cpu --all-groups --frozen --no-install-project
COPY . .
RUN uv sync --extra lab --extra cpu --all-groups --frozen
ENV PATH="/app/.venv/bin:$PATH" PYTHONHASHSEED=0
CMD ["nox", "-s", "lint", "types", "tests"]
