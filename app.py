
import logging

import streamlit as st

from config import PDF_DIR, DB_DIR, COLLECTION_NAME
from rag import ask_gassafe


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GasSafe AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

NOT_FOUND = (
    "I could not find this information "
    "in the provided documents."
)

GENERAL_ERROR = (
    "GasSafe AI could not process your question. "
    "Please try again."
)

INDEX_ERROR = (
    "The document index is unavailable. "
    "Please check the application logs."
)


@st.cache_resource(show_spinner=False)
def initialize_document_index():
    """
    Check the existing ChromaDB collection.

    Run the project's existing PDF ingestion
    process only if the collection is missing
    or empty.
    """
    import chromadb

    pdf_files = (
        sorted(PDF_DIR.glob("*.pdf"))
        if PDF_DIR.exists()
        else []
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in "
            "data/safety_manuals/"
        )

    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        if collection.count() > 0:
            logger.info(
                "Existing PDF index is ready."
            )
            return True

    except (
        chromadb.errors.NotFoundError,
        ValueError,
    ):
        logger.info(
            "No existing PDF index found."
        )

    logger.info(
        "Building PDF index from %s files.",
        len(pdf_files),
    )

    from ingest import main as ingest_main

    ingest_main()

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise RuntimeError(
            "PDF ingestion produced an empty index."
        )

    logger.info(
        "PDF indexing completed successfully."
    )

    return True


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def clear_conversation():
    st.session_state.messages = []
    st.session_state.pending_question = None


def display_sources(sources):
    if not sources:
        return

    with st.expander(
        f"📄 PDF references ({len(sources)})"
    ):
        for index, source in enumerate(
            sources,
            start=1,
        ):
            if not isinstance(source, dict):
                continue

            filename = source.get(
                "file",
                "Unknown PDF",
            )

            page = source.get(
                "page",
                "?",
            )

            excerpt = source.get(
                "excerpt",
                "",
            )

            if not isinstance(filename, str):
                filename = "Unknown PDF"

            if not isinstance(excerpt, str):
                excerpt = ""

            st.markdown(
                f"**{index}. {filename} "
                f"— Page {page}**"
            )

            if excerpt.strip():
                st.caption(
                    excerpt.strip()
                )

            if index < len(sources):
                st.divider()


def process_question(question):
    try:
        result = ask_gassafe(question)

        if not isinstance(result, dict):
            logger.error(
                "Invalid RAG response."
            )
            return GENERAL_ERROR, []

        answer = result.get(
            "answer",
            NOT_FOUND,
        )

        sources = result.get(
            "sources",
            [],
        )

        if not isinstance(answer, str):
            return GENERAL_ERROR, []

        answer = answer.strip()

        if not answer:
            return NOT_FOUND, []

        if not isinstance(sources, list):
            sources = []

        return answer, sources

    except FileNotFoundError as error:
        logger.warning(
            "Document index missing: %s",
            error,
        )
        return INDEX_ERROR, []

    except Exception:
        logger.exception(
            "Question processing failed."
        )
        return GENERAL_ERROR, []


initialize_session()

st.title("🛡️ GasSafe AI")

st.caption(
    "AI-powered Gas Safety "
    "Document Intelligence"
)

st.info(
    "GasSafe AI uses Retrieval-Augmented "
    "Generation to answer questions "
    "from indexed PDF documents. "
    "AI-generated answers may be inaccurate."
)

st.warning(
    "Synthetic demonstration only. "
    "Not for operational or emergency "
    "decision-making. For a suspected "
    "gas leak or immediate danger, "
    "leave the affected area and contact "
    "emergency services or your gas "
    "utility emergency line."
)

index_ready = False

with st.status(
    "Initializing document intelligence...",
    expanded=False,
) as status:
    try:
        index_ready = initialize_document_index()

        status.update(
            label="Document index ready",
            state="complete",
            expanded=False,
        )

    except Exception:
        logger.exception(
            "Document initialization failed."
        )

        status.update(
            label="Document initialization failed",
            state="error",
            expanded=True,
        )

        st.error(
            "Unable to initialize the PDF "
            "database. Check application logs."
        )


with st.sidebar:
    st.header("About GasSafe AI")

    st.write(
        "GasSafe AI retrieves relevant PDF "
        "passages and uses a local language "
        "model to generate answers."
    )

    st.markdown(
        "**Language model:** "
        "Qwen2.5-1.5B-Instruct"
    )

    st.markdown(
        "**Embedding model:** MiniLM"
    )

    st.markdown(
        "**Vector database:** ChromaDB"
    )

    st.markdown(
        "**Inference:** CPU"
    )

    st.divider()

    if index_ready:
        st.success("Document index ready")
    else:
        st.error("Document index unavailable")

    st.button(
        "🗑️ Clear conversation",
        on_click=clear_conversation,
        use_container_width=True,
    )

    st.caption(
        "Models may be downloaded when "
        "first initialized. Free cloud "
        "hosting resources are limited."
    )


st.subheader("Ask a gas safety question")

examples = [
    "What is the inspection reference "
    "for Blue Pine Segment A?",
    "Summarize the corrosion "
    "monitoring record.",
    "What is the status of maintenance "
    "work order WO-DEMO-104?",
    "What information is recorded "
    "during document review?",
]

if not st.session_state.messages:
    st.write("**Try an example question:**")

    for index, example in enumerate(examples):
        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
            disabled=not index_ready,
        ):
            st.session_state.pending_question = (
                example
            )


for message in st.session_state.messages:
    with st.chat_message(
        message.get("role", "assistant")
    ):
        st.markdown(
            message.get("content", "")
        )

        display_sources(
            message.get("sources", [])
        )


typed_question = st.chat_input(
    "Ask about the synthetic safety PDF...",
    disabled=not index_ready,
)

pending_question = (
    st.session_state.pending_question
)

question = typed_question or pending_question

if question and index_ready:
    st.session_state.pending_question = None

    question = question.strip()

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Retrieving documents and "
                "generating an AI answer..."
            ):
                answer, sources = (
                    process_question(question)
                )

            st.markdown(answer)

            display_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


st.divider()

st.caption(
    "GasSafe AI | Qwen + MiniLM + ChromaDB "
    "| Synthetic demonstration"
)
