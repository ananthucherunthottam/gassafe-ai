
#!/bin/sh

# GasSafe AI - Cloud Startup Script
# Runs PDF ingestion before starting Streamlit.

set -eu

echo "================================"
echo "GasSafe AI Cloud Startup"
echo "================================"

cd /app

PDF_DIR="/app/data/safety_manuals"

echo "Checking PDF directory..."

if [ ! -d "$PDF_DIR" ]; then
    echo "ERROR: PDF directory not found."
    exit 1
fi

PDF_COUNT=$(find "$PDF_DIR" \
    -maxdepth 1 \
    -type f \
    -iname '*.pdf' \
    | wc -l)

echo "PDF files found: $PDF_COUNT"

if [ "$PDF_COUNT" -gt 0 ]; then
    echo "Starting PDF ingestion..."

    if python ingest.py; then
        echo "PDF ingestion completed."
    else
        echo "ERROR: PDF ingestion failed."
        exit 1
    fi
else
    echo "No PDF documents found."
    echo "Skipping ingestion."
    echo "The chatbot requires indexed PDFs."
fi

echo "Starting Streamlit..."

exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true
    
