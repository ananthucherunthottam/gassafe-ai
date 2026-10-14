# GasSafe AI

**A local-first, PDF-based gas safety knowledge assistant powered by Retrieval-Augmented Generation (RAG).**

GasSafe AI helps users search approved gas safety manuals, retrieve relevant passages, and generate answers with document references.

## Project Status

Under development. The application and Docker configuration have not yet been fully tested.

## Key Features

- Chat with approved gas safety PDF documents
- Local Qwen2.5-1.5B-Instruct language model
- Sentence Transformers for semantic search
- ChromaDB for vector storage
- Streamlit chatbot interface
- PDF filename and page references
- CPU-based inference without paid LLM APIs
- Docker-based deployment for supported computers
- Local-first design with future cloud deployment options

## Technology Stack

| Component | Technology |
| --- | --- |
| Frontend | Streamlit |
| Language model | Qwen2.5-1.5B-Instruct GGUF |
| LLM runtime | llama-cpp-python |
| Embeddings | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| PDF extraction | pypdf |
| Programming language | Python 3.11 |
| Containerization | Docker and Docker Compose |

## Architecture

```text
Gas Safety PDFs
       |
       v
    ingest.py
       |
       v
PDF Text Extraction
       |
       v
Text Chunking
       |
       v
MiniLM Embeddings
       |
       v
ChromaDB Vector Database
       |
       v
    rag.py <--- User Question
       |
       v
Relevant PDF Passages
       |
       v
    llm.py
       |
       v
Local Qwen Language Model
       |
       v
Answer + PDF Sources
       |
       v
Streamlit Chat Interface
```

## Project Structure

```text
gassafe-ai/
|
|-- app.py
|-- rag.py
|-- llm.py
|-- ingest.py
|-- config.py
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .gitignore
|-- start.sh
|-- start.ps1
|-- README.md
|
|-- data/
|   |-- safety_manuals/
|       |-- .gitkeep
|
|-- models/          # Created at runtime
|-- chroma_db/       # Created at runtime
```

## Requirements

For Docker deployment:

- Windows, Linux, or macOS computer
- Docker Desktop or Docker Engine with Compose
- Internet connection for initial model and dependency downloads
- Sufficient free storage and RAM for the AI models
- Approved, text-readable gas safety PDFs

Docker is not required for direct Python development, but the installation steps for that option will be documented after testing.

## Quick Start

### 1. Clone the repository

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd gassafe-ai
```

Replace the placeholder with your own GitHub repository URL.

### 2. Add approved PDF documents

Place PDF files inside:

```text
data/safety_manuals/
```

Use only public documents or documents you are explicitly authorized to process.

Do not commit confidential company documents to GitHub.

### 3. Start on Windows

Open PowerShell in the project folder:

```powershell
.\start.ps1
```

### 4. Start on Linux or macOS

Open a terminal in the project folder:

```bash
bash start.sh
```

The startup scripts are intended to:

1. Verify Docker Compose and PDF availability
2. Build the Docker image
3. Extract and index PDF text
4. Create embeddings in ChromaDB
5. Start the Streamlit chatbot

### 5. Open GasSafe AI

Visit:

http://localhost:8501

The initial setup may take considerable time. The Qwen model is downloaded when first needed for answer generation.

## Manual Docker Commands

Build the application:

```bash
docker compose build
```

Index the PDF documents:

```bash
docker compose run --rm --no-deps gassafe-ai python ingest.py
```

Start the application:

```bash
docker compose up -d
```

View logs:

```bash
docker compose logs -f gassafe-ai
```

Stop the application:

```bash
docker compose down
```

## Persistent Storage

Docker Compose mounts these local directories:

| Directory | Purpose |
| --- | --- |
| data/safety_manuals | Approved source PDFs (read-only in container) |
| models | Downloaded Qwen model |
| chroma_db | PDF embeddings and vector index |

The application image does not include these files.

## Example Questions

- What safety precautions are described for pipeline maintenance?
- What inspection requirements are listed in the manual?
- What does the document say about corrosion monitoring?
- Which section describes leak detection procedures?

Answers depend on the information available in the uploaded PDFs.

## Safety and Security

GasSafe AI is a document-search and research assistant, not a certified gas safety decision system.

- Do not use AI-generated answers as the sole basis for operational or emergency decisions.
- Verify all critical information against current approved procedures.
- Do not rely on the chatbot for live pipeline monitoring.
- Do not upload confidential documents without authorization.
- For an actual gas leak or immediate danger, leave the area and contact emergency services or the gas utility emergency line.
- The default Docker configuration binds Streamlit to localhost only.
- Authentication, HTTPS, and access controls are required before exposing the application to other users.

## Current Limitations

- Scanned PDFs without extractable text require OCR.
- The application does not provide live gas pipeline data.
- Retrieved passages may not always be relevant.
- AI-generated answers may contain errors.
- Source references are not proof that every claim is correct.
- The model and dependency versions still require compatibility testing.
- Mobile browser access requires additional deployment and security configuration.

## Development Roadmap

- [x] Create GitHub project structure
- [x] Implement PDF ingestion code
- [x] Implement RAG retrieval code
- [x] Implement local Qwen integration code
- [x] Create Streamlit interface
- [x] Centralize configuration
- [x] Add Docker configuration
- [x] Add startup scripts
- [ ] Verify dependencies and model filename
- [ ] Run first end-to-end test
- [ ] Validate PDF retrieval accuracy
- [ ] Test Windows/Linux/macOS compatibility
- [ ] Add secure mobile browser access
- [ ] Evaluate cloud deployment options

## License

No open-source license has been selected. This repository is private unless its owner changes its visibility.
