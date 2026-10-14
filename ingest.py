
import hashlib
import uuid

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
        raise ValueError(
            "Chunk size must be greater than overlap."
        )

    text = " ".join(text.split())
    step = chunk_size - overlap

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), step)
        if text[start:start + chunk_size].strip()
    ]


def collection_exists(client, name):
    """Check whether a ChromaDB collection exists."""
    return any(
        collection.name == name
        for collection in client.list_collections()
    )


def read_pdf_chunks(pdf_files):
    """Extract text chunks and source metadata from PDFs."""
    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))

            file_id = hashlib.sha256(
                pdf_path.name.encode("utf-8")
            ).hexdigest()[:16]

            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                for chunk_index, chunk in enumerate(
                    split_text(text)
                ):
                    documents.append(chunk)

                    metadatas.append({
                        "source": pdf_path.name,
                        "page": page_index + 1,
                    })

                    ids.append(
                        f"{file_id}_p{page_index + 1}_c{chunk_index}"
                    )

        except Exception as error:
            raise RuntimeError(
                f"Failed to process PDF: {pdf_path.name}"
            ) from error

    return documents, metadatas, ids


def build_collection(
    client,
    collection_name,
    documents,
    metadatas,
    ids,
    model,
):
    """Build and verify a new collection."""
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )

    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
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
            f"Indexed {end}/{len(documents)} chunks"
        )

    actual_count = collection.count()

    if actual_count != len(documents):
        raise RuntimeError(
            "Index verification failed: "
            f"expected {len(documents)} chunks, "
            f"found {actual_count}."
        )

    print("New index verified successfully.")


def promote_collection(client, temporary_name):
    """
    Promote the completed index while preserving
    the old collection for rollback.
    """
    old_exists = collection_exists(
        client,
        COLLECTION_NAME,
    )

    backup_name = None

    if old_exists:
        backup_name = (
            f"backup_{uuid.uuid4().hex[:12]}"
        )

        print("Backing up the previous index...")

        old_collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )

        old_collection.modify(name=backup_name)

    try:
        print("Activating the new index...")

        temporary_collection = client.get_collection(
            name=temporary_name,
            embedding_function=None,
        )

        temporary_collection.modify(
            name=COLLECTION_NAME
        )

    except Exception:
        if backup_name:
            print("Restoring the previous index...")

            backup_collection = client.get_collection(
                name=backup_name,
                embedding_function=None,
            )

            backup_collection.modify(
                name=COLLECTION_NAME
            )

        raise

    if backup_name:
        try:
            client.delete_collection(
                name=backup_name
            )
            print("Previous index backup removed.")
        except Exception as error:
            print(
                "Warning: Could not remove backup "
                f"collection: {error}"
            )


def main():
    """Safely rebuild the GasSafe AI PDF index."""
    PDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        print(
            "Existing index has not been changed."
        )
        return

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be greater than zero."
        )

    documents, metadatas, ids = read_pdf_chunks(
        pdf_files
    )

    if not documents:
        print(
            "No extractable text found in the PDFs."
        )
        print(
            "Existing index has not been changed."
        )
        return

    print(
        f"Preparing {len(documents)} document chunks..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    temporary_name = (
        f"staging_{uuid.uuid4().hex[:12]}"
    )

    try:
        build_collection(
            client=client,
            collection_name=temporary_name,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            model=model,
        )

        promote_collection(
            client,
            temporary_name,
        )

    except Exception:
        print("Index rebuild failed.")

        # Only remove the staging collection.
        # Do not delete the active collection.
        if collection_exists(
            client,
            temporary_name,
        ):
            try:
                client.delete_collection(
                    name=temporary_name
                )
            except Exception as cleanup_error:
                print(
                    "Warning: Could not clean up "
                    f"staging collection: {cleanup_error}"
                )

        raise

    print(
        "GasSafe AI PDF indexing completed successfully."
    )
    print(
        f"Active collection: {COLLECTION_NAME}"
    )
    print(
        f"Database location: {DB_DIR}"
    )


if __name__ == "__main__":
    main()
    
