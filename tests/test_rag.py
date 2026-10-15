
import ast
import math
import unittest

from pathlib import Path
from unittest.mock import MagicMock


RAG_PATH = (
    Path(__file__).resolve().parents[1] / "rag.py"
)

NOT_FOUND = (
    "I could not find this information in the provided documents."
)

GENERATION_ERROR = (
    "I found relevant information, but could not generate "
    "an answer. Please try again."
)


def load_rag_functions():
    """
    Load RAG functions directly from rag.py
    without importing heavyweight AI dependencies.
    """
    source = RAG_PATH.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    required_functions = {
        "empty_result",
        "safe_first_list",
        "valid_distance",
        "extract_passages",
        "ask_gassafe",
    }

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in required_functions
    ]

    found_functions = {
        node.name for node in functions
    }

    missing = required_functions - found_functions

    if missing:
        raise RuntimeError(
            f"Missing RAG functions: {sorted(missing)}"
        )

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    return compile(
        module,
        str(RAG_PATH),
        "exec",
    )


class TestRAGRetrieval(unittest.TestCase):

    def setUp(self):
        self.collection = MagicMock()

        self.embedding_model = MagicMock()

        self.embedding_model.encode.return_value.tolist.return_value = [
            [1.0, 0.0, 0.0]
        ]

        self.collection.count.return_value = 2

        self.collection.query.return_value = {
            "documents": [[
                "Inspect gas pipelines for corrosion."
            ]],
            "metadatas": [[
                {
                    "source": "gas_manual.pdf",
                    "page": 3,
                }
            ]],
            "distances": [[0.15]],
        }

        self.generate_answer = MagicMock(
            return_value=(
                "Inspect the pipeline for corrosion."
            )
        )

        self.namespace = {
            "load_collection": (
                lambda: self.collection
            ),
            "load_embeddings": (
                lambda: self.embedding_model
            ),
            "generate_answer": self.generate_answer,
            "NOT_FOUND": NOT_FOUND,
            "GENERATION_ERROR": GENERATION_ERROR,
            "TOP_K": 3,
            "MAX_DISTANCE": 0.65,
            "math": math,
        }

        exec(
            load_rag_functions(),
            self.namespace,
        )

        self.ask_gassafe = self.namespace[
            "ask_gassafe"
        ]

    def test_relevant_document_retrieval(self):
        """
        Relevant PDF content should produce
        an answer with a source.
        """
        result = self.ask_gassafe(
            "How should gas pipelines be inspected?"
        )

        self.assertIn(
            "corrosion",
            result["answer"].lower(),
        )

        self.assertEqual(
            len(result["sources"]),
            1,
        )

        self.assertEqual(
            result["sources"][0]["file"],
            "gas_manual.pdf",
        )

        self.assertEqual(
            result["sources"][0]["page"],
            3,
        )

        self.collection.query.assert_called_once()

        self.generate_answer.assert_called_once()

    def test_no_relevant_documents(self):
        """
        Irrelevant passages must not be
        sent to the language model.
        """
        self.collection.query.return_value = {
            "documents": [[
                "Unrelated operational information."
            ]],
            "metadatas": [[
                {
                    "source": "unrelated.pdf",
                    "page": 1,
                }
            ]],
            "distances": [[0.95]],
        }

        result = self.ask_gassafe(
            "What is the gas pipeline inspection procedure?"
        )

        self.assertEqual(
            result["answer"],
            NOT_FOUND,
        )

        self.assertEqual(
            result["sources"],
            [],
        )

        self.generate_answer.assert_not_called()

    def test_empty_database(self):
        """
        An empty vector database should
        return a safe fallback.
        """
        self.collection.count.return_value = 0

        result = self.ask_gassafe(
            "What are the pipeline safety requirements?"
        )

        self.assertEqual(
            result["answer"],
            NOT_FOUND,
        )

        self.assertEqual(
            result["sources"],
            [],
        )

        self.collection.query.assert_not_called()

        self.generate_answer.assert_not_called()

    def test_empty_question(self):
        """
        Blank questions should not
        trigger document retrieval.
        """
        result = self.ask_gassafe("   ")

        self.assertEqual(
            result["answer"],
            "Please enter a question.",
        )

        self.assertEqual(
            result["sources"],
            [],
        )

        self.collection.query.assert_not_called()

        self.generate_answer.assert_not_called()

    def test_missing_metadata_is_handled(self):
        """
        Missing metadata should not crash
        the retrieval process.
        """
        self.collection.query.return_value = {
            "documents": [[
                "Inspect the pipeline before operation."
            ]],
            "metadatas": [[None]],
            "distances": [[0.10]],
        }

        result = self.ask_gassafe(
            "What should happen before operation?"
        )

        self.assertEqual(
            result["sources"][0]["file"],
            "Unknown PDF",
        )

        self.assertEqual(
            result["sources"][0]["page"],
            "?",
        )

        self.generate_answer.assert_called_once()

    def test_llm_failure_returns_safe_message(self):
        """
        A language model error should
        return a controlled response.
        """
        self.generate_answer.side_effect = (
            RuntimeError("Simulated LLM failure")
        )

        result = self.ask_gassafe(
            "How should gas pipelines be inspected?"
        )

        self.assertEqual(
            result["answer"],
            GENERATION_ERROR,
        )

        self.assertEqual(
            result["sources"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
    
