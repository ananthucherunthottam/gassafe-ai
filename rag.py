
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    MAX_DISTANCE,
)
from llm import generate_answer

NOT_FOUND = (
    "I could not find this information in the provided documents."
)


@lru_cache(maxsize=1)
def load_embeddings():
    """Load the embedding model once."""
    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )


@lru_cache(maxsize=1)
def load_collection():
    """Open the existing ChromaDB collection."""
    if not DB_DIR.exists():
        raise FileNotFoundError(
            "PDF database not found. Run: python ingest.py"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def ask_gassafe(question: str) -> dict:
    """Retrieve PDF passages and generate a grounded answer."""
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    collection = load_collection()
    document_count = collection.count()

    if document_count == 0:
        return {
            "answer": NOT_FOUND,
            "sources": [],
        }

    model = load_embeddings()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(TOP_K, document_count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        if distance > MAX_DISTANCE:
            continue

        filename = metadata.get("source", "Unknown PDF")
        page = metadata.get("page", "?")

        context_parts.append(
            f"Source: {filename}, Page {page}\n{document}"
        )

        sources.append({
            "file": filename,
            "page": page,
            "excerpt": document[:350],
        })

    if not context_parts:
        return {
            "answer": NOT_FOUND,
            "sources": [],
        }

    context = "\n\n---\n\n".join(context_parts)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    question = input("Ask GasSafe AI: ")
    result = ask_gassafe(question)

    print("\nAnswer:\n", result["answer"])
    print("\nSources:")

    for source in result["sources"]:
        print(
            f"- {source['file']}, page {source['page']}"
        )
  
