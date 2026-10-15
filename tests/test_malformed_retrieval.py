
import ast
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock


RAG_PATH = Path(__file__).resolve().parents[1] / "rag.py"

NOT_FOUND = (
    "I could not find this information in the provided documents."
)

GENERATION_ERROR = (
    "I found relevant information, but could not generate "
    "an answer. Please try again."
)


def load_rag_functions():
    """
    Load actual RAG functions without importing
    ChromaDB, sentence-transformers, or the LLM.
    """
    tree = ast.parse(
        RAG_PATH.read_text(encoding="utf-8")
    )

    required = {
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
        and node.name in required
    ]

    found = {node.name for node in functions}

    missing = required - found

    if missing:
        raise RuntimeError(
            f"Missing RAG functions: {sorted(missing)}"
        )

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "math": math,
        "NOT_FOUND": NOT_FOUND,
        "GENERATION_ERROR": GENERATION_ERROR,
        "MAX_DISTANCE": 0.65,
        "TOP_K": 3,
    }

    exec(
        compile(module, str(RAG_PATH), "exec"),
        namespace,
    )

    return namespace


class TestMalformedRetrieval(unittest.TestCase):

    def setUp(self):
        self.namespace = load_rag_functions()

        self.collection = MagicMock()
        self.embedding_model = MagicMock()

        self.embedding_model.encode.return_value.tolist.return_value = [
            [1.0, 0.0, 0.0]
        ]

        self.collection.count.return_value = 3

        self.generate_answer = MagicMock(
            return_value="Follow the approved procedure."
        )

        self.namespace["load_collection"] = (
            lambda: self.collection
        )

        self.namespace["load_embeddings"] = (
            lambda: self.embedding_model
        )

        self.namespace["generate_answer"] = (
            self.generate_answer
        )

        self.ask_gassafe = self.namespace[
            "ask_gassafe"
        ]

    def set_results(
        self,
        documents,
        metadatas,
        distances,
    ):
        self.collection.query.return_value = {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def assert_safe_fallback(self):
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

    def test_none_metadata(self):
        """None metadata must use safe fallback labels."""
        self.set_results(
            ["Inspect the pipeline."],
            [None],
            [0.10],
        )

        result = self.ask_gassafe(
            "How should the pipeline be inspected?"
        )

        self.assertEqual(
            result["sources"][0]["file"],
            "Unknown PDF",
        )

        self.assertEqual(
            result["sources"][0]["page"],
            "?",
        )

    def test_missing_distance(self):
        """A missing distance must reject the passage."""
        self.set_results(
            ["Inspect the pipeline."],
            [{"source": "manual.pdf", "page": 1}],
            [None],
        )

        self.assert_safe_fallback()

    def test_nan_distance(self):
        """NaN must not pass the distance threshold."""
        self.set_results(
            ["Inspect the pipeline."],
            [{"source": "manual.pdf", "page": 1}],
            [float("nan")],
        )

        self.assert_safe_fallback()

    def test_infinite_distance(self):
        """Infinity must not pass the distance threshold."""
        self.set_results(
            ["Inspect the pipeline."],
            [{"source": "manual.pdf", "page": 1}],
            [float("inf")],
        )

        self.assert_safe_fallback()

    def test_string_distance(self):
        """Non-numeric distance values must be rejected."""
        self.set_results(
            ["Inspect the pipeline."],
            [{"source": "manual.pdf", "page": 1}],
            ["0.10"],
        )

        self.assert_safe_fallback()

    def test_empty_document(self):
        """Empty document text must not reach the LLM."""
        self.set_results(
            [""],
            [{"source": "manual.pdf", "page": 1}],
            [0.10],
        )

        self.assert_safe_fallback()

    def test_whitespace_document(self):
        """Whitespace-only documents must be rejected."""
        self.set_results(
            ["   \n   "],
            [{"source": "manual.pdf", "page": 1}],
            [0.10],
        )

        self.assert_safe_fallback()

    def test_non_string_document(self):
        """Non-text documents must be rejected."""
        self.set_results(
            [12345],
            [{"source": "manual.pdf", "page": 1}],
            [0.10],
        )

        self.assert_safe_fallback()

    def test_missing_result_fields(self):
        """Missing ChromaDB fields must not cause a crash."""
        self.collection.query.return_value = {
            "documents": [["Inspect the pipeline."]]
        }

        self.assert_safe_fallback()

    def test_mixed_valid_and_invalid_passages(self):
        """
        Only valid passages should be included
        in the generated answer and citations.
        """
        self.set_results(
            documents=[
                "Inspect the pipeline for corrosion.",
                "",
                "Unrelated information.",
            ],
            metadatas=[
                {"source": "inspection.pdf", "page": 2},
                {"source": "empty.pdf", "page": 1},
                {"source": "unrelated.pdf", "page": 5},
            ],
            distances=[
                0.10,
                0.20,
                0.95,
            ],
        )

        result = self.ask_gassafe(
            "How should the pipeline be inspected?"
        )

        self.assertEqual(
            len(result["sources"]),
            1,
        )

        self.assertEqual(
            result["sources"][0]["file"],
            "inspection.pdf",
        )

        self.generate_answer.assert_called_once()

        context = self.generate_answer.call_args.args[1]

        self.assertIn(
            "Inspect the pipeline for corrosion.",
            context,
        )

        self.assertNotIn(
            "Unrelated information.",
            context,
        )


if __name__ == "__main__":
    unittest.main()
  
