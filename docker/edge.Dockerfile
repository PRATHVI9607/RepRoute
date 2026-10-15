# syntax=docker/dockerfile:1
# Edge plane: numpy + onnxruntime only, no torch. Base images are pinned by digest (multi-arch index, 2026-10-14).
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
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
