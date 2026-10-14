
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter


SOURCE_PATH = Path(__file__).resolve().parents[1] / "ingest.py"


def load_ingestion_functions():
    """
    Load ingestion functions without importing
    ChromaDB or heavyweight embedding libraries.
    """
    tree = ast.parse(
        SOURCE_PATH.read_text(encoding="utf-8")
    )

    names = {
        "split_text",
        "read_pdf_chunks",
        "main",
    }

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    if len(functions) != len(names):
        raise RuntimeError(
            "Required ingestion functions not found."
        )

    # Explicit arguments will be supplied to split_text
    # in these tests, so configuration defaults are removed.
    for function in functions:
        if function.name == "split_text":
            function.args.defaults = []

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    import hashlib
    from pypdf import PdfReader

    namespace = {
        "hashlib": hashlib,
        "PdfReader": PdfReader,
        "CHUNK_SIZE": 800,
        "CHUNK_OVERLAP": 150,
        "BATCH_SIZE": 64,
    }

    exec(
        compile(module, str(SOURCE_PATH), "exec"),
        namespace,
    )

    return namespace


class TestIngestionErrors(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_dir = Path(self.temp_dir.name) / "pdfs"
        self.db_dir = Path(self.temp_dir.name) / "db"

        self.pdf_dir.mkdir()

        self.namespace = load_ingestion_functions()

        self.namespace["PDF_DIR"] = self.pdf_dir
        self.namespace["DB_DIR"] = self.db_dir
        self.namespace["COLLECTION_NAME"] = "gas_safety_pdf"

        self.namespace["SentenceTransformer"] = MagicMock()
        self.namespace["chromadb"] = MagicMock()
        self.namespace["build_collection"] = MagicMock()
        self.namespace["promote_collection"] = MagicMock()

        self.read_pdf_chunks = self.namespace[
            "read_pdf_chunks"
        ]

        self.main = self.namespace["main"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_pdfs(self):
        """No PDFs should leave the existing index untouched."""
        self.main()

        self.namespace[
            "SentenceTransformer"
        ].assert_not_called()

        self.namespace[
            "chromadb"
        ].PersistentClient.assert_not_called()

    def test_corrupted_pdf(self):
        """Invalid PDF content should raise a useful error."""
        bad_pdf = self.pdf_dir / "corrupted.pdf"
        bad_pdf.write_bytes(
            b"This is not a valid PDF file."
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to process PDF: corrupted.pdf",
        ):
            self.read_pdf_chunks([bad_pdf])

    def test_pdf_without_extractable_text(self):
        """A blank PDF must not produce document chunks."""
        blank_pdf = self.pdf_dir / "blank.pdf"

        writer = PdfWriter()
        writer.add_blank_page(
            width=300,
            height=300,
        )

        with blank_pdf.open("wb") as file:
            writer.write(file)

        documents, metadatas, ids = (
            self.read_pdf_chunks([blank_pdf])
        )

        self.assertEqual(documents, [])
        self.assertEqual(metadatas, [])
        self.assertEqual(ids, [])

    def test_invalid_batch_size(self):
        """Invalid batch size must be rejected."""
        sample_pdf = self.pdf_dir / "sample.pdf"
        sample_pdf.write_bytes(b"placeholder")

        with patch.dict(
            self.namespace,
            {"BATCH_SIZE": 0},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "BATCH_SIZE must be greater than zero",
            ):
                self.main()

        self.namespace[
            "SentenceTransformer"
        ].assert_not_called()

    def test_no_text_preserves_existing_index(self):
        """A blank PDF must not trigger index replacement."""
        blank_pdf = self.pdf_dir / "blank.pdf"

        writer = PdfWriter()
        writer.add_blank_page(
            width=300,
            height=300,
        )

        with blank_pdf.open("wb") as file:
            writer.write(file)

        self.main()

        self.namespace[
            "chromadb"
        ].PersistentClient.assert_not_called()

        self.namespace[
            "promote_collection"
        ].assert_not_called()


if __name__ == "__main__":
    unittest.main()
  
