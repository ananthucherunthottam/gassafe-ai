
from functools import lru_cache
import math

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

GENERATION_ERROR = (
    "I found relevant information, but could not generate "
    "an answer. Please try again."
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

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def empty_result(message=NOT_FOUND):
    """Return a consistent response structure."""
    return {
        "answer": message,
        "sources": [],
    }


def safe_first_list(results, key):
    """
    Safely retrieve the first list from a
    ChromaDB query response.
    """
    if not isinstance(results, dict):
        return []

    values = results.get(key)

    if not isinstance(values, (list, tuple)):
        return []

    if not values:
        return []

    first = values[0]

    if not isinstance(first, (list, tuple)):
        return []

    return first


def valid_distance(distance):
    """Check whether a distance is usable."""
    if isinstance(distance, bool):
        return False

    if not isinstance(distance, (int, float)):
        return False

    if not math.isfinite(distance):
        return False

    return distance <= MAX_DISTANCE


def extract_passages(results):
    """
    Extract valid, relevant passages and
    their source information.
    """
    documents = safe_first_list(
        results,
        "documents",
    )

    metadatas = safe_first_list(
        results,
        "metadatas",
    )

    distances = safe_first_list(
        results,
        "distances",
    )

    context_parts = []
    sources = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if not valid_distance(distance):
            continue

        if not isinstance(document, str):
            continue

        document = document.strip()

        if not document:
            continue

        if not isinstance(metadata, dict):
            metadata = {}

        filename = metadata.get(
            "source",
            "Unknown PDF",
        )

        page = metadata.get(
            "page",
            "?",
        )

        if not isinstance(filename, str):
            filename = "Unknown PDF"

        filename = filename.strip()

        if not filename:
            filename = "Unknown PDF"

        if not isinstance(page, (int, str)):
            page = "?"

        if isinstance(page, str):
            page = page.strip() or "?"

        context_parts.append(
            f"Source: {filename}, Page {page}\n"
            f"{document}"
        )

        sources.append({
            "file": filename,
            "page": page,
            "excerpt": document[:350],
        })

    return context_parts, sources


def ask_gassafe(question: str) -> dict:
    """
    Retrieve relevant PDF passages and
    generate a grounded answer.
    """
    if not isinstance(question, str):
        return empty_result(
            "Please enter a valid question."
        )

    question = question.strip()

    if not question:
        return empty_result(
            "Please enter a question."
        )

    collection = load_collection()

    document_count = collection.count()

    if document_count <= 0:
        return empty_result()

    model = load_embeddings()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(TOP_K, document_count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    context_parts, sources = extract_passages(
        results
    )

    if not context_parts:
        return empty_result()

    context = "\n\n---\n\n".join(
        context_parts
    )

    try:
        answer = generate_answer(
            question,
            context,
        )
    except Exception:
        return empty_result(
            GENERATION_ERROR
        )

    if not isinstance(answer, str):
        return empty_result(
            GENERATION_ERROR
        )

    answer = answer.strip()

    if not answer:
        return empty_result(
            GENERATION_ERROR
        )

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    question = input(
        "Ask GasSafe AI: "
    )

    result = ask_gassafe(
        question
    )

    print(
        "\nAnswer:\n",
        result["answer"],
    )

    print("\nSources:")

    for source in result["sources"]:
        print(
            f"- {source['file']}, "
            f"page {source['page']}"
        )
        
