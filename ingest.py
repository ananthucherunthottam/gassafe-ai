
import hashlib

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    PDF_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("Chunk size must be greater than overlap.")

    text = " ".join(text.split())
    step = chunk_size - overlap

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
        if text[start:start + chunk_size].strip()
    ]


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        print("Add an approved PDF and run this script again.")
        return

    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))
        file_id = hashlib.sha256(
            pdf_path.name.encode("utf-8")
        ).hexdigest()[:16]

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            for chunk_index, chunk in enumerate(split_text(text)):
                documents.append(chunk)

                metadatas.append({
                    "source": pdf_path.name,
                    "page": page_index + 1,
                })

                ids.append(
                    f"{file_id}_p{page_index + 1}_c{chunk_index}"
                )

    if not documents:
        print("No extractable text found in the PDFs.")
        print("Scanned PDFs may require OCR.")
        return

    print(f"Creating embeddings for {len(documents)} chunks...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild the collection using the current PDF documents.
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

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

        print(
            f"Indexed {min(end, len(documents))}/{len(documents)}"
        )

    print("PDF indexing completed successfully.")
    print(f"Vector database saved in: {DB_DIR}")


if __name__ == "__main__":
    main()
  
