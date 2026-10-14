
import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock


RAG_PATH = Path(__file__).resolve().parents[1] / "rag.py"

NOT_FOUND = (
    "I could not find this information in the provided documents."
)


def load_ask_gassafe():
    """
    Extract ask_gassafe without importing
    ChromaDB, MiniLM, or the local LLM.
    """
    source = RAG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ask_gassafe"
    ]

    if len(functions) != 1:
        raise RuntimeError(
            "Expected one ask_gassafe function in rag.py."
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


class TestRAGEdgeCases(unittest.TestCase):

    def setUp(self):
        self.collection = MagicMock()
        self.embedding_model = MagicMock()
        self.generate_answer = MagicMock(
            return_value="Follow the documented inspection procedure."
        )

        self.embedding_model.encode.return_value.tolist.return_value = [
            [1.0, 0.0, 0.0]
        ]

        self.collection.count.return_value = 3

        self.namespace = {
            "load_collection": lambda: self.collection,
            "load_embeddings": lambda: self.embedding_model,
            "generate_answer": self.generate_answer,
            "NOT_FOUND": NOT_FOUND,
            "TOP_K": 3,
            "MAX_DISTANCE": 0.65,
        }

        exec(
            load_ask_gassafe(),
            self.namespace,
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

    def test_empty_retrieval_results(self):
        """No retrieved documents means no generated answer."""
        self.set_results([], [], [])

        result = self.ask_gassafe(
            "What are the pipeline inspection requirements?"
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

    def test_missing_metadata_fields(self):
        """Missing source and page fields use fallback labels."""
        self.set_results(
            documents=[
                "Inspect the pipeline for corrosion."
            ],
            metadatas=[{}],
            distances=[0.10],
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

        self.generate_answer.assert_called_once()

    def test_multiple_pdf_sources(self):
        """Relevant passages from different PDFs are preserved."""
        self.set_results(
            documents=[
                "Inspect pipelines for corrosion.",
                "Record the inspection findings.",
            ],
            metadatas=[
                {"source": "inspection.pdf", "page": 2},
                {"source": "maintenance.pdf", "page": 5},
            ],
            distances=[0.10, 0.20],
        )

        result = self.ask_gassafe(
            "What should be done during an inspection?"
        )

        self.assertEqual(
            len(result["sources"]),
            2,
        )

        self.assertEqual(
            result["sources"][0]["file"],
            "inspection.pdf",
        )

        self.assertEqual(
            result["sources"][1]["file"],
            "maintenance.pdf",
        )

        self.assertEqual(
            result["sources"][0]["page"],
            2,
        )

        self.assertEqual(
            result["sources"][1]["page"],
            5,
        )

        self.generate_answer.assert_called_once()

        context = self.generate_answer.call_args.args[1]

        self.assertIn(
            "inspection.pdf",
            context,
        )

        self.assertIn(
            "maintenance.pdf",
            context,
        )

    def test_all_documents_above_distance_threshold(self):
        """Irrelevant passages must not reach the LLM."""
        self.set_results(
            documents=[
                "Unrelated operational information."
            ],
            metadatas=[
                {"source": "unrelated.pdf", "page": 1}
            ],
            distances=[0.95],
        )

        result = self.ask_gassafe(
            "What is the pipeline safety procedure?"
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

    def test_distance_exactly_at_threshold(self):
        """A passage at the threshold is accepted."""
        self.set_results(
            documents=[
                "Inspect the pipeline before operation."
            ],
            metadatas=[
                {"source": "safety.pdf", "page": 4}
            ],
            distances=[0.65],
        )

        result = self.ask_gassafe(
            "What should happen before operation?"
        )

        self.assertEqual(
            len(result["sources"]),
            1,
        )

        self.generate_answer.assert_called_once()

    def test_retrieval_respects_top_k(self):
        """The query must not request more than TOP_K results."""
        self.collection.count.return_value = 10

        self.set_results([], [], [])

        self.ask_gassafe(
            "What are the inspection requirements?"
        )

        self.assertEqual(
            self.collection.query.call_args.kwargs[
                "n_results"
            ],
            3,
        )

    def test_retrieval_respects_document_count(self):
        """The query must not request more documents than exist."""
        self.collection.count.return_value = 1

        self.set_results([], [], [])

        self.ask_gassafe(
            "What are the inspection requirements?"
        )

        self.assertEqual(
            self.collection.query.call_args.kwargs[
                "n_results"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
  
