
$ErrorActionPreference = "Stop"

Write-Host "=================================="
Write-Host "       GasSafe AI Startup"
Write-Host "=================================="

# Switch to the folder containing this script
Set-Location $PSScriptRoot

# Check Docker Compose availability
docker compose version | Out-Null

if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose is not available. Install Docker Desktop."
}

# Check for approved PDF documents
$pdfFiles = @(Get-ChildItem -Path "./data/safety_manuals" -Filter "*.pdf" -File -ErrorAction SilentlyContinue)

if ($pdfFiles.Count -eq 0) {
    throw "No PDF documents found. Add approved PDFs to data/safety_manuals/"
}

Write-Host "Step 1: Building GasSafe AI..."
docker compose build

if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed."
}

Write-Host "Step 2: Indexing PDF documents..."
docker compose run --rm --no-deps gassafe-ai python ingest.py

if ($LASTEXITCODE -ne 0) {
    throw "PDF indexing failed."
}

Write-Host "Step 3: Starting the chatbot..."
docker compose up -d --no-build

if ($LASTEXITCODE -ne 0) {
    throw "GasSafe AI startup failed."
}

Write-Host ""
Write-Host "GasSafe AI startup commands completed."
Write-Host "Open: http://localhost:8501"
Write-Host "Check status: docker compose ps"
Write-Host "View logs: docker compose logs -f gassafe-ai"
  
