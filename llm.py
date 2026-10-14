
import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_DIR = Path("models")


@lru_cache(maxsize=1)
def load_model():
    """Download the model once and load it into memory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
    )

    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=max(1, min(4, os.cpu_count() or 2)),
        n_gpu_layers=0,
        chat_format="chatml",
        verbose=False,
    )


def generate_answer(question: str, context: str) -> str:
    """Answer using only the retrieved PDF context."""
    llm = load_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are GasSafe AI, a gas safety document assistant. "
                "Answer only using the supplied document context. "
                "If the context does not contain the answer, say "
                "'I could not find this information in the provided documents.' "
                "Never invent safety procedures, pressure limits, "
                "emergency instructions, or regulatory requirements. "
                "For an actual gas leak or immediate danger, direct the "
                "user to leave the area and contact emergency services "
                "or the gas utility's emergency line."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document context:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=300,
        temperature=0.1,
    )

    return response["choices"][0]["message"]["content"].strip()
  
