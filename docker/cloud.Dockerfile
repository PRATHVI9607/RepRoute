# syntax=docker/dockerfile:1
# Cloud plane: onnxruntime + lightgbm head zoo, no torch.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra cloud --no-dev --frozen --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --extra cloud --no-dev --frozen
ENV PATH="/app/.venv/bin:$PATH" PYTHONHASHSEED=0
EXPOSE 7400
CMD ["kai", "cloud", "/bundle"]
