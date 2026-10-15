
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    GGML_NATIVE=OFF \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install native build dependencies for llama-cpp-python
# and curl for the container health check.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source code
# to improve Docker build caching.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy application files.
COPY . /app

# Create a non-root user and persistent storage paths.
RUN groupadd --system gassafe && \
    useradd \
        --system \
        --gid gassafe \
        --home-dir /app \
        --shell /usr/sbin/nologin \
        gassafe && \
    mkdir -p \
        /app/models \
        /app/chroma_db \
        /app/data/safety_manuals \
        /app/.cache/huggingface && \
    chown -R gassafe:gassafe \
        /app/models \
        /app/chroma_db \
        /app/data \
        /app/.cache

# Store Hugging Face downloads in a writable directory.
ENV HF_HOME=/app/.cache/huggingface

# Run the application without root privileges.
USER gassafe

EXPOSE 8501

# Confirm that Streamlit is responding.
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD curl --fail --silent \
        http://127.0.0.1:8501/_stcore/health \
        || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
     
