
import logging

import streamlit as st

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
    "I could not find this information in the provided documents."
)

GENERAL_ERROR = (
    "GasSafe AI could not process your question. "
    "Please try again or contact the application administrator."
)

INDEX_ERROR = (
    "The gas safety document index is unavailable. "
    "Please run the PDF ingestion process before asking questions."
)


def initialize_session():
    """Initialize conversation state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def clear_conversation():
    """Reset conversation history."""
    st.session_state.messages = []
    st.session_state.pending_question = None


def display_sources(sources):
    """Display retrieved PDF references."""
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
                f"**{index}. {filename} — Page {page}**"
            )

            if excerpt.strip():
                st.caption(
                    excerpt.strip()
                )

            if index < len(sources):
                st.divider()


def process_question(question):
    """Run the RAG pipeline and normalize its response."""
    try:
        result = ask_gassafe(question)

        if not isinstance(result, dict):
            logger.error(
                "RAG returned an invalid response type."
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
            answer = GENERAL_ERROR
            sources = []

        answer = answer.strip()

        if not answer:
            answer = NOT_FOUND
            sources = []

        if not isinstance(sources, list):
            sources = []

        return answer, sources

    except (
        FileNotFoundError,
        ) as error:
        logger.warning(
            "Document index not found: %s",
            error,
        )
        return INDEX_ERROR, []

    except Exception:
        logger.exception(
            "GasSafe AI failed to process the question."
        )
        return GENERAL_ERROR, []


initialize_session()


st.title("🛡️ GasSafe AI")

st.caption(
    "Local AI-powered gas safety document assistant"
)

st.info(
    "GasSafe AI retrieves information from indexed PDF documents. "
    "AI-generated answers may be incomplete or inaccurate. "
    "Always verify information against approved safety procedures."
)

st.warning(
    "For an active gas leak or immediate danger, "
    "leave the affected area and contact emergency services "
    "or your gas utility emergency line. "
    "Do not rely on this chatbot for emergency decisions."
)


with st.sidebar:
    st.header("About GasSafe AI")

    st.write(
        "GasSafe AI uses Retrieval-Augmented Generation (RAG) "
        "to answer questions from indexed gas safety PDFs."
    )

    st.markdown(
        "**Language model:** Qwen2.5-1.5B-Instruct"
    )

    st.markdown(
        "**Embedding model:** MiniLM"
    )

    st.markdown(
        "**Vector database:** ChromaDB"
    )

    st.markdown(
        "**Inference:** Local CPU"
    )

    st.divider()

    st.button(
        "🗑️ Clear conversation",
        on_click=clear_conversation,
        use_container_width=True,
    )

    st.caption(
        "Documents and model inference remain local "
        "after required models have been downloaded."
    )


st.subheader("Ask a gas safety question")

examples = [
    "What are common gas pipeline safety hazards?",
    "What precautions are required before maintenance?",
    "What does the document say about leak detection?",
]

if not st.session_state.messages:
    st.write("**Try an example question:**")

    for index, example in enumerate(examples):
        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example


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
    "Ask a question about your safety documents..."
)

pending_question = st.session_state.pending_question

question = typed_question or pending_question

if question:
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
                "Searching safety documents..."
            ):
                answer, sources = process_question(
                    question
                )

            st.markdown(answer)

            display_sources(sources)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        
