
import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def load_ask_gassafe():
    """
    Load ask_gassafe from rag.py without importing
    heavyweight AI dependencies.
    """
    source_path = Path(__file__).resolve().parents[1] / "rag.py"

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "ask_gassafe"
    )

    module = ast.Module(
        body=[function],
        type_ignores=[]
    )

    ast.fix_missing_locations(module)

    return compile(
        module,
        str(source_path),
        "exec"
    )


class TestRAGRetrieval(unittest.TestCase):

    def setUp(self):
        """Prepare mocked RAG dependencies."""

        self.collection = MagicMock()
        self.embedding_model = MagicMock()

        self.embedding_model.encode.return_value.tolist.return_value = [
            1.0, 0.0, 0.0
        ]

        self.collection.count.return_value = 2

        self.collection.query.return_value = {
            "documents": [[
                "Inspect gas pipelines for corrosion."
            ]],
            "metadatas": [[{
                "source": "gas_manual.pdf",
                "page": 3
            }]],
            "distances": [[0.15]]
        }

        self.generate_answer = MagicMock(
            return_value="Inspect the pipeline for corrosion."
        )

        self.namespace = {
            "load_collection": lambda: self.collection,
            "get_collection": lambda: self.collection,
            "load_embeddings": lambda: self.embedding_model,
            "load_embedding_model": lambda: self.embedding_model,
            "get_embedding_model": lambda: self.embedding_model,
            "generate_answer": self.generate_answer,
            "NOT_FOUND": (
                "No relevant information found in the "
                "uploaded documents."
            ),
            "TOP_K": 3,
            "MAX_DISTANCE": 0.65,
        }

        exec(
            load_ask_gassafe(),
            self.namespace
        )

        self.ask_gassafe = self.namespace["ask_gassafe"]

    def test_relevant_document_retrieval(self):
        """Relevant documents should be used as context."""

        result = self.ask_gassafe(
            "How should gas pipelines be inspected?"
        )

        self.assertIn(
            "corrosion",
            str(result).lower()
        )

        self.collection.query.assert_called_once()
        self.generate_answer.assert_called_once()

    def test_no_relevant_documents(self):
        """Reject matches beyond the similarity threshold."""

        self.collection.query.return_value = {
            "documents": [[
                "Unrelated operational information."
            ]],
            "metadatas": [[{
                "source": "unrelated.pdf",
                "page": 1
            }]],
            "distances": [[0.95]]
        }

        result = self.ask_gassafe(
            "What is the gas pipeline inspection procedure?"
        )

        self.assertIsNotNone(result)
        self.generate_answer.assert_not_called()

    def test_empty_database(self):
        """Handle an empty document index safely."""

        self.collection.count.return_value = 0

        result = self.ask_gassafe(
            "What are the pipeline safety requirements?"
        )

        self.assertIsNotNone(result)

        self.collection.query.assert_not_called()
        self.generate_answer.assert_not_called()

    def test_empty_question(self):
        """Blank questions must not trigger retrieval."""

        result = self.ask_gassafe("   ")

        self.assertIsNotNone(result)

        self.collection.query.assert_not_called()
        self.generate_answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
    
