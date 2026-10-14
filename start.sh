
#!/usr/bin/env bash
set -euo pipefail

echo "=================================="
echo "       GasSafe AI Startup"
echo "=================================="

# Run from the project directory
cd "$(dirname "$0")"

# Check Docker Compose availability
if ! docker compose version >/dev/null 2>&1; then
    echo "Error: Docker Compose is not available."
    echo "Install Docker Desktop or Docker Engine with Compose."
    exit 1
fi

# Check for approved PDF documents
shopt -s nullglob
pdf_files=(data/safety_manuals/*.pdf)

if [ ${#pdf_files[@]} -eq 0 ]; then
    echo "No PDF documents found."
    echo "Add approved PDFs to data/safety_manuals/"
    exit 1
fi

echo "Step 1: Building GasSafe AI..."
docker compose build

echo "Step 2: Indexing PDF documents..."
docker compose run --rm --no-deps gassafe-ai python ingest.py

echo "Step 3: Starting the chatbot..."
docker compose up -d --no-build

echo ""
echo "GasSafe AI startup commands completed."
echo "Open: http://localhost:8501"
echo "Check status: docker compose ps"
echo "View logs: docker compose logs -f gassafe-ai"
  
