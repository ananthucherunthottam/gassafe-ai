
import streamlit as st

from rag import ask_gassafe

st.set_page_config(
    page_title="GasSafe AI",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ GasSafe AI")
st.caption("AI-powered gas safety document assistant")

st.info(
    "GasSafe AI provides information from indexed documents. "
    "It is not a substitute for approved safety procedures "
    "or qualified emergency personnel."
)

with st.sidebar:
    st.header("About GasSafe AI")
    st.write(
        "GasSafe AI uses Retrieval-Augmented Generation (RAG) "
        "to answer questions from gas safety PDFs."
    )

    st.markdown("**AI model:** Qwen2.5-1.5B-Instruct")
    st.markdown("**Embeddings:** MiniLM")
    st.markdown("**Vector database:** ChromaDB")
    st.markdown("**Inference:** Local CPU")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.subheader("Ask a gas safety question")

examples = [
    "What are common gas pipeline safety hazards?",
    "What precautions are required before maintenance?",
    "What does the document say about leak detection?",
]

st.write("**Example questions**")

for index, example in enumerate(examples):
    if st.button(example, key=f"example_{index}"):
        st.session_state.pending_question = example

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📄 PDF references"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['file']} — Page {source['page']}**"
                    )
                    st.caption(source["excerpt"])

typed_question = st.chat_input(
    "Ask GasSafe AI about gas safety..."
)

pending_question = st.session_state.pop(
    "pending_question", None
)

question = typed_question or pending_question

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask_gassafe(question)
                answer = result["answer"]
                sources = result["sources"]
            except Exception:
                answer = (
                    "GasSafe AI could not process the question. "
                    "Please check that the PDF database is indexed "
                    "and the AI model is available."
                )
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📄 PDF references"):
                for source in sources:
                    st.markdown(
                        f"**{source['file']} — Page {source['page']}**"
                    )
                    st.caption(source["excerpt"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
  
