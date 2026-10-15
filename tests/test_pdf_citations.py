
import ast
import hashlib
import math
import tempfile
import unittest

from pathlib import Path

import chromadb
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]
INGEST_PATH = ROOT / "ingest.py"
RAG_PATH = ROOT / "rag.py"


def load_functions(path, names, namespace):
    """Load selected functions from project source."""
    tree = ast.parse(
        path.read_text(encoding="utf-8")
    )

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    found = {node.name for node in functions}

    missing = set(names) - found

    if missing:
        raise RuntimeError(
            f"Missing functions: {sorted(missing)}"
        )

    for function in functions:
        if function.name == "split_text":
            function.args.defaults = [
                ast.Constant(value=800),
                ast.Constant(value=150),
            ]

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    exec(
        compile(module, str(path), "exec"),
        namespace,
    )

    return namespace


class TestPDFCitations(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.root = Path(self.temp_dir.name)

        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()

        self.db_dir = self.root / "chroma"

        ingest_namespace = {
            "hashlib": hashlib,
            "PdfReader": PdfReader,
            "CHUNK_SIZE": 800,
            "CHUNK_OVERLAP": 150,
        }

        load_functions(
            INGEST_PATH,
            {"split_text", "read_pdf_chunks"},
            ingest_namespace,
        )

        self.read_pdf_chunks = ingest_namespace[
            "read_pdf_chunks"
        ]

        rag_namespace = {
            "math": math,
            "MAX_DISTANCE": 0.65,
            "NOT_FOUND": (
                "I could not find this information "
                "in the provided documents."
            ),
        }

        load_functions(
            RAG_PATH,
            {
                "safe_first_list",
                "valid_distance",
                "extract_passages",
            },
            rag_namespace,
        )

        self.extract_passages = rag_namespace[
            "extract_passages"
        ]

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir)
        )

        self.collection = self.client.create_collection(
            name="test_pdf_citations",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def tearDown(self):
        self.collection = None
        self.client = None
        self.temp_dir.cleanup()

    def create_pdf(self, filename, pages=1):
        """Create a valid temporary PDF."""
        path = self.pdf_dir / filename

        writer = PdfWriter()

        for _ in range(pages):
            writer.add_blank_page(
                width=300,
                height=300,
            )

        with path.open("wb") as file:
            writer.write(file)

        return path

    def add_chunk(self, chunk_id, text, metadata, embedding):
        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def test_pdf_file_is_readable(self):
        """A generated PDF can be opened by pypdf."""
        pdf_path = self.create_pdf(
            "inspection.pdf",
            pages=2,
        )

        reader = PdfReader(str(pdf_path))

        self.assertEqual(
            len(reader.pages),
            2,
        )

    def test_blank_pdf_produces_no_citations(self):
        """Blank PDFs must not create document chunks."""
        pdf_path = self.create_pdf(
            "blank.pdf",
            pages=1,
        )

        documents, metadatas, ids = (
            self.read_pdf_chunks([pdf_path])
        )

        self.assertEqual(documents, [])
        self.assertEqual(metadatas, [])
        self.assertEqual(ids, [])

    def test_chromadb_preserves_filename(self):
        """ChromaDB must preserve the source PDF filename."""
        self.add_chunk(
            "chunk_1",
            "Inspect the pipeline for corrosion.",
            {
                "source": "inspection.pdf",
                "page": 2,
            },
            [1.0, 0.0, 0.0],
        )

        stored = self.collection.get(
            ids=["chunk_1"],
            include=["metadatas"],
        )

        self.assertEqual(
            stored["metadatas"][0]["source"],
            "inspection.pdf",
        )

    def test_chromadb_preserves_page_number(self):
        """ChromaDB must preserve the original page number."""
        self.add_chunk(
            "chunk_2",
            "Record all inspection findings.",
            {
                "source": "maintenance.pdf",
                "page": 5,
            },
            [0.0, 1.0, 0.0],
        )

        stored = self.collection.get(
            ids=["chunk_2"],
            include=["metadatas"],
        )

        self.assertEqual(
            stored["metadatas"][0]["page"],
            5,
        )

    def test_retrieval_preserves_source_citation(self):
        """Retrieved passages must retain source information."""
        self.add_chunk(
            "chunk_3",
            "Inspect pipelines for corrosion.",
            {
                "source": "safety_manual.pdf",
                "page": 3,
            },
            [1.0, 0.0, 0.0],
        )

        results = self.collection.query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=1,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        context_parts, sources = (
            self.extract_passages(results)
        )

        self.assertEqual(
            len(sources),
            1,
        )

        self.assertEqual(
            sources[0]["file"],
            "safety_manual.pdf",
        )

        self.assertEqual(
            sources[0]["page"],
            3,
        )

        self.assertIn(
            "Inspect pipelines for corrosion.",
            context_parts[0],
        )

    def test_irrelevant_retrieval_has_no_citation(self):
        """Passages beyond the distance threshold are excluded."""
        self.add_chunk(
            "chunk_4",
            "Unrelated document information.",
            {
                "source": "unrelated.pdf",
                "page": 1,
            },
            [1.0, 0.0, 0.0],
        )

        results = self.collection.query(
            query_embeddings=[[0.0, 1.0, 0.0]],
            n_results=1,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        context_parts, sources = (
            self.extract_passages(results)
        )

        self.assertEqual(context_parts, [])
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
  
