
import os
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# PDF storage
PDF_DIR = BASE_DIR / "data" / "safety_manuals"

# ChromaDB vector database
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "gas_safety_pdf"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local Qwen model storage
MODEL_DIR = BASE_DIR / "models"
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# PDF processing
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 64

# RAG retrieval
TOP_K = 3
MAX_DISTANCE = 0.65

# CPU inference
N_THREADS = max(1, min(4, os.cpu_count() or 2))
N_CTX = 4096
MAX_NEW_TOKENS = 300
  
