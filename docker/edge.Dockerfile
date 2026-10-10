# syntax=docker/dockerfile:1
# Edge plane: numpy + onnxruntime only, no torch. Pin the base image by digest before release.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra edge --no-dev --frozen --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --extra edge --no-dev --frozen
ENV PATH="/app/.venv/bin:$PATH" PYTHONHASHSEED=0
COPY docker/netem/shape.sh /usr/local/bin/shape
CMD ["kai", "edge", "--help"]
