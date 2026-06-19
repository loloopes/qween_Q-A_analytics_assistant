# LLM Q&A API (langchain.ipynb → server.py)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    PDF_PATH=/app/data/analytics_engineer.pdf \
    BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    ADAPTER_REPO=Glccampos/llm_qween

WORKDIR /app

# System libs occasionally needed by torch / tokenizers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# CPU PyTorch (smaller image; use a CUDA base + GPU torch for GPU hosts)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY service.py service_langgraph.py server.py server_langgraph.py semantic_cache.py ./
COPY scripts/prefetch_models.py ./scripts/prefetch_models.py
COPY ["Analytics Engineer .pdf", "/app/data/analytics_engineer.pdf"]

# Optional: bake models into the image (pass --build-arg HF_TOKEN=... for private repos)
ARG HF_TOKEN
ARG SKIP_PREFETCH=true
RUN if [ "$SKIP_PREFETCH" != "true" ] && [ -n "$HF_TOKEN" ]; then \
      HF_TOKEN="$HF_TOKEN" BASE_MODEL="$BASE_MODEL" ADAPTER_REPO="$ADAPTER_REPO" \
      python scripts/prefetch_models.py; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${API_PORT}/health" || exit 1

CMD ["python", "server.py"]
