# Dockerfile
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /workspace

# System deps (optional, kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install uv

# Copy sources
COPY .. ./

# UV sync
RUN uv sync

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/src