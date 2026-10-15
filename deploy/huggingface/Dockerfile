
# GasSafe AI - Hugging Face Spaces Dockerfile
#
# Build from the repository root:
# docker build -f deploy/huggingface/Dockerfile .
#
# Cloud deployment preparation only.
# Do not use confidential documents.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GGML_NATIVE=OFF \
    HF_HOME=/app/.cache/huggingface \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Native build dependencies for llama-cpp-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the existing application dependencies.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        -r /app/requirements.txt

# Copy application source code.
COPY app.py rag.py llm.py ingest.py config.py /app/

# Create writable runtime directories.
RUN mkdir -p \
        /app/models \
        /app/chroma_db \
        /app/data/safety_manuals \
        /app/.cache/huggingface \
    && useradd --uid 1000 \
        --create-home \
        --shell /bin/bash \
        gassafe \
    && chown -R 1000:1000 \
        /app/models \
        /app/chroma_db \
        /app/data \
        /app/.cache

# Run the application as a non-root user.
USER 1000

EXPOSE 7860

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl --fail \
        http://127.0.0.1:7860/_stcore/health \
        || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.headless=true"]
     
