
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_DIR = Path("data/safety_manuals")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "gas_safety_pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 64


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    text = " ".join(text.split())
    step = chunk_size - overlap

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
        if text[start:start + chunk_size].strip()
    ]


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        print("Add an approved PDF and run this script again.")
        return

    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            for chunk_index, chunk in enumerate(split_text(text)):
                documents.append(chunk)

                metadatas.append({
                    "source": pdf_path.name,
                    "page": page_index + 1,
                })

                ids.append(
                    f"{pdf_path.stem}_p{page_index + 1}_c{chunk_index}"
                )

    if not documents:
        print("No extractable text found in the PDFs.")
        return

    print(f"Creating embeddings for {len(documents)} chunks...")

    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = documents[start:end]

        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=ids[start:end],
            documents=batch,
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )

        print(f"Indexed {min(end, len(documents))}/{len(documents)}")

    print("PDF indexing completed successfully.")
    print(f"Vector database saved in: {DB_DIR}")


if __name__ == "__main__":
    main()
  
