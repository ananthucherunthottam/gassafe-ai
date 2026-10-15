
# GasSafe AI

**A local-first, AI-powered gas safety PDF knowledge assistant using Retrieval-Augmented Generation (RAG).**

GasSafe AI helps users search approved gas safety manuals, retrieve relevant document passages, and generate AI-assisted answers with PDF filename and page references.

The application uses a locally running Qwen language model, MiniLM embeddings, ChromaDB, and a Streamlit web interface.

---

## Project Status

**Status: End-to-end integration validated. Local MVP release preparation in progress.**

Completed validation:

- 58 automated Python unit tests passed
- Docker image build passed
- Streamlit startup and health checks passed
- PDF ingestion using real MiniLM embeddings passed
- ChromaDB persistence validation passed
- PDF source citation validation passed
- Real Qwen CPU inference passed
- Full PDF-to-Qwen integration test passed

These tests were performed through GitHub Actions using synthetic documents.

### Remaining validation

- Test installation on the intended computer
- Verify the interactive Streamlit browser experience
- Evaluate answer accuracy using representative approved manuals
- Complete the final security and reliability review

**GasSafe AI is not certified for operational gas safety decisions.**

---

## Key Features

- Chat with indexed gas safety PDF documents
- Retrieve relevant passages using semantic search
- Generate answers with Qwen2.5-1.5B-Instruct
- Display PDF filenames and page numbers
- Show retrieved source excerpts
- Preserve conversation history during a browser session
- Clear conversation history
- Use example questions
- Run AI inference locally on CPU
- Avoid paid LLM APIs
- Deploy using Docker Compose
- Preserve downloaded models and ChromaDB data

---

## Technology Stack

| Component | Technology |
| --- | --- |
| Frontend | Streamlit |
| Language model | Qwen2.5-1.5B-Instruct GGUF |
| LLM runtime | llama-cpp-python |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| PDF extraction | pypdf |
| Programming language | Python 3.11 |
| Containerization | Docker and Docker Compose |
| Automated testing | unittest and GitHub Actions |

---

## Architecture

```text
Approved Gas Safety PDFs
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
  ChromaDB Vector Index
          |
          v
        rag.py
          ^
          |
     User Question
          |
          v
   Semantic Retrieval
          |
          v
   Relevant PDF Passages
          |
          v
        llm.py
          |
          v
    Local Qwen Model
          |
          v
 Answer + PDF References
          |
          v
  Streamlit Chat Interface
```

---

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
|
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .gitignore
|
|-- start.sh
|-- start.ps1
|-- README.md
|
|-- data/
|   |-- safety_manuals/
|       |-- .gitkeep
|
|-- tests/
|
|-- .github/
|   |-- workflows/
|       |-- validate.yml
|       |-- docker-validate.yml
|       |-- ingestion-validate.yml
|       |-- qwen-validate.yml
|       |-- e2e-validate.yml
```

Runtime model files and ChromaDB data are stored in persistent Docker volumes.

---

## System Requirements

To run GasSafe AI locally:

- Windows, Linux, or macOS computer
- Docker Engine with Docker Compose support
- Internet connection for initial downloads
- Sufficient available RAM and disk space
- Approved, text-readable PDF documents

The Qwen model runs on CPU.

Performance depends on the processor, available RAM, and the size of the indexed document collection.

A computer with at least 8 GB RAM is a reasonable starting point for experimentation, although this is not a guaranteed minimum requirement.

More RAM may be necessary for larger workloads.

The initial Docker build and model downloads can take considerable time.

---

## Quick Start

### Step 1: Obtain the repository

Clone your private GitHub repository:

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd gassafe-ai
```

Replace the placeholder with your repository's actual HTTPS or SSH clone URL.

You must have permission to access the private repository.

Alternatively, download the repository ZIP from GitHub and extract it.

### Step 2: Add approved PDF documents

Place approved PDF files inside:

```text
data/safety_manuals/
```

Example:

```text
data/
  safety_manuals/
    pipeline_safety.pdf
    inspection_manual.pdf
    maintenance_guidelines.pdf
```

Use only documents you are authorized to process.

**Do not commit confidential company documents to GitHub.**

The application currently supports PDFs containing extractable text.

Image-only scanned PDFs require OCR before ingestion.

### Step 3: Build the Docker image

Open a terminal in the project directory.

Run:

```bash
docker compose build
```

This installs the required Python and native dependencies.

The build may take several minutes or longer, especially when compiling llama-cpp-python.

### Step 4: Index PDF documents

Run:

```bash
docker compose run --rm --no-deps gassafe-ai python ingest.py
```

The ingestion process:

1. Reads PDFs from the safety_manuals directory
2. Extracts text
3. Splits text into smaller passages
4. Generates MiniLM embeddings
5. Creates the ChromaDB index
6. Stores source filenames and page numbers

The first embedding-model download requires internet access.

Wait for ingestion to complete successfully before starting the chatbot.

### Step 5: Start GasSafe AI

Run:

```bash
docker compose up -d
```

Check the container status:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f gassafe-ai
```

### Step 6: Open GasSafe AI

On the computer running Docker, open:

http://localhost:8501

You should see the GasSafe AI Streamlit chatbot.

Enter a question about your indexed documents.

The first question requiring AI generation may trigger the Qwen model download.

Later questions can reuse the downloaded model.

---

## Alternative Startup Scripts

The repository includes:

- `start.ps1` for Windows PowerShell
- `start.sh` for Linux and macOS

Windows:

```powershell
.\start.ps1
```

Linux/macOS:

```bash
bash start.sh
```

These scripts provide an alternative startup method.

Their behavior and operating-system compatibility must be verified before relying on them.

If a script fails, use the manual Docker commands above.

---

## Common Docker Commands

### Build

```bash
docker compose build
```

### Start

```bash
docker compose up -d
```

### Check status

```bash
docker compose ps
```

### View logs

```bash
docker compose logs -f gassafe-ai
```

### Rebuild the PDF index

```bash
docker compose run --rm --no-deps gassafe-ai python ingest.py
```

### Restart

```bash
docker compose restart gassafe-ai
```

### Stop

```bash
docker compose down
```

**Important:** Do not run `docker compose down -v` unless you intentionally want to remove persistent Docker volumes.

Removing volumes can delete your ChromaDB index and downloaded models.

---

## Persistent Storage

GasSafe AI uses the following Docker storage configuration.

| Storage | Container path | Purpose |
| --- | --- | --- |
| Host PDF directory | `/app/data/safety_manuals` | Source PDFs, read-only |
| `gassafe_chroma` | `/app/chroma_db` | Persistent ChromaDB index |
| `gassafe_models` | `/app/models` | Downloaded Qwen model |
| `gassafe_hf_cache` | `/app/.cache/huggingface` | Hugging Face model cache |

The PDF directory is mounted from:

```text
./data/safety_manuals
```

The remaining locations use Docker-managed named volumes.

Docker Compose may prefix volume names with the project name.

### Updating PDF documents

When adding, replacing, or removing PDFs:

1. Update the files in `data/safety_manuals/`
2. Run the ingestion command again
3. Wait for indexing to complete
4. Restart the application if necessary

Avoid querying the application while rebuilding the index.

The current index replacement procedure is not fully atomic for concurrent readers.

---

## Example Questions

- What inspection requirements are listed in the manual?
- What does the document say about corrosion monitoring?
- Which sections discuss leak detection?
- What maintenance precautions are described?
- Which PDF contains information about pipeline inspections?

Answers depend on the information available in the indexed documents.

If relevant information cannot be retrieved, GasSafe AI should return a fallback rather than invent an answer.

However, AI-generated answers are not guaranteed to be accurate.

---

## How PDF Citations Work

When a relevant passage is retrieved, GasSafe AI records:

- PDF filename
- PDF page number
- Excerpt from the retrieved passage

The application displays these details under PDF references.

Page numbers refer to extracted PDF page positions and may differ from printed page labels inside the document.

**A citation indicates a retrieved passage. It does not independently prove that every statement in the generated answer is supported.**

Always verify important information against the original approved document.

---

## Automated Testing

The repository contains Python unit tests and GitHub Actions workflows.

### Python unit tests

Workflow:

```text
.github/workflows/validate.yml
```

Runs the automated Python test suite.

### Docker validation

Workflow:

```text
.github/workflows/docker-validate.yml
```

Checks Docker image build, Streamlit startup, and container health.

### PDF ingestion validation

Workflow:

```text
.github/workflows/ingestion-validate.yml
```

Tests PDF extraction, MiniLM embeddings, ChromaDB persistence, retrieval, and citations.

### Qwen inference validation

Workflow:

```text
.github/workflows/qwen-validate.yml
```

Tests real Qwen model download, CPU inference, and model persistence.

### End-to-end validation

Workflow:

```text
.github/workflows/e2e-validate.yml
```

Tests the complete pipeline:

```text
PDF
 -> MiniLM
 -> ChromaDB
 -> Retrieval
 -> Qwen
 -> Answer and Sources
```

Some workflows must be started manually from GitHub Actions.

The successful integration tests used synthetic documents rather than production gas safety manuals.

---

## Troubleshooting

### Streamlit page does not open

Check:

```bash
docker compose ps
```

Then:

```bash
docker compose logs --tail=100 gassafe-ai
```

Confirm Docker is running and port 8501 is available.

### PDF database not found

Run:

```bash
docker compose run --rm --no-deps gassafe-ai python ingest.py
```

Ensure the PDF directory contains at least one text-readable PDF.

### No relevant answer found

Check whether:

- The document contains extractable text
- The relevant PDF was included during ingestion
- The question matches information in the document
- The retrieval distance threshold excluded the passage

Do not automatically increase the retrieval threshold without evaluating retrieval quality.

### Model download fails

Check:

- Internet connectivity
- Available disk space
- Access to Hugging Face
- Application logs
- Model repository and filename configuration

The initial model downloads require network access.

### Permission denied when writing to ChromaDB

Check Docker volume permissions and the container's non-root user configuration.

Do not disable security restrictions without understanding the underlying permission issue.

### Docker build fails

Review the build output:

```bash
docker compose build --progress=plain
```

Look for errors involving:

- Python package installation
- Native compilation
- CMake
- llama-cpp-python
- Network downloads

### Slow AI responses

Qwen runs on CPU.

Response speed depends on:

- Processor performance
- Available RAM
- Context length
- Number of generated tokens
- Other workloads on the computer

The first response may be slower because the model must load into memory.

### Scanned PDFs return no content

pypdf extracts embedded text but does not perform OCR.

Convert scanned PDFs into searchable PDFs using an approved OCR process before ingestion.

---

## Security and Privacy

GasSafe AI is designed for local-first use.

The default Docker configuration binds Streamlit to:

```text
127.0.0.1:8501
```

This limits access to the computer running Docker.

### Security precautions

- Keep the GitHub repository private when appropriate
- Do not commit confidential PDFs
- Do not commit downloaded model files or ChromaDB data
- Use only authorized documents
- Keep Docker and dependencies updated
- Restrict access to the host computer
- Review application logs before sharing them
- Avoid exposing Streamlit directly to the internet

The application does not currently provide production-grade authentication, authorization, or HTTPS termination.

These controls must be added before exposing it to other users or networks.

### Local-first limitations

The application runs model inference locally after downloading the required files.

Initial dependency and model downloads require network access.

Local-first operation does not mean that every deployment environment is automatically private or secure.

---

## Safety Disclaimer

**GasSafe AI is a document-search and research assistant, not a certified gas safety decision system.**

- Do not use AI-generated answers as the sole basis for operational decisions
- Verify all critical information against current approved procedures
- Do not rely on the chatbot for live pipeline monitoring
- Do not use it to authorize maintenance or emergency actions
- Do not assume retrieved citations prove answer accuracy
- Do not process confidential documents without authorization

For an actual gas leak or immediate danger, leave the affected area and contact emergency services or the gas utility emergency line.

GasSafe AI is not a substitute for qualified emergency personnel.

---

## Current Limitations

- Scanned PDFs require OCR
- The application does not access live pipeline telemetry
- Retrieval can miss relevant passages
- The model can generate incorrect or unsupported statements
- Source citations require human verification
- CPU inference may be slow
- PDF indexing must be rerun after document changes
- Concurrent queries during index replacement may fail
- Local installation has not yet been tested on the intended computer
- The interactive browser experience still requires validation
- Public deployment requires additional security controls

---

## Development Roadmap

### Completed

- [x] Create GitHub project structure
- [x] Implement PDF ingestion
- [x] Implement RAG retrieval
- [x] Integrate local Qwen model
- [x] Build Streamlit chatbot interface
- [x] Centralize configuration
- [x] Add Docker configuration
- [x] Add startup scripts
- [x] Add automated Python tests
- [x] Validate Docker build and startup
- [x] Validate PDF ingestion
- [x] Validate ChromaDB persistence
- [x] Validate PDF citations
- [x] Validate real Qwen CPU inference
- [x] Pass full end-to-end integration test

### Remaining

- [ ] Validate interactive Streamlit interface
- [ ] Test installation on the intended computer
- [ ] Evaluate answers against representative approved manuals
- [ ] Complete security and reliability review
- [ ] Prepare local MVP release checklist

### Future Enhancements

- Secure authenticated web deployment
- Document upload and management interface
- OCR for scanned PDFs
- Improved citation verification
- Multi-document evaluation
- Role-based access control
- Controlled integration with enterprise utility systems
- Human-approved AI agent workflows

---

## License

No open-source license has been selected.

The repository remains private unless its owner changes its visibility.

All third-party libraries and model files remain subject to their respective licenses.
