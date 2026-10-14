
import tempfile
import unittest
from pathlib import Path

import chromadb


class TestChromaDB(unittest.TestCase):

    def setUp(self):
        """Create an isolated database for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()

        self.client = chromadb.PersistentClient(
            path=str(Path(self.temp_dir.name) / "chroma_db")
        )

        self.collection = self.client.create_collection(
            name="gassafe_test",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def tearDown(self):
        """Clean up the temporary database."""
        self.client = None
        self.collection = None
        self.temp_dir.cleanup()

    def test_store_document_chunk(self):
        """Verify that a document chunk can be stored."""
        self.collection.add(
            ids=["chunk_1"],
            documents=["Inspect gas pipelines for corrosion."],
            metadatas=[{
                "source": "sample_manual.pdf",
                "page": 1,
            }],
            embeddings=[[1.0, 0.0, 0.0]],
        )

        self.assertEqual(self.collection.count(), 1)

    def test_retrieve_document_chunk(self):
        """Verify that stored text can be retrieved."""
        self.collection.add(
            ids=["chunk_1"],
            documents=["Inspect gas pipelines for corrosion."],
            metadatas=[{
                "source": "sample_manual.pdf",
                "page": 1,
            }],
            embeddings=[[1.0, 0.0, 0.0]],
        )

        results = self.collection.get(
            ids=["chunk_1"],
            include=["documents", "metadatas"],
        )

        self.assertEqual(
            results["documents"][0],
            "Inspect gas pipelines for corrosion.",
        )

        self.assertEqual(
            results["metadatas"][0]["page"],
            1,
        )

    def test_similarity_search(self):
        """Verify that the nearest vector is retrieved."""
        self.collection.add(
            ids=["corrosion", "pressure"],
            documents=[
                "Inspect pipelines for corrosion.",
                "Monitor gas pressure levels.",
            ],
            metadatas=[
                {"source": "manual.pdf", "page": 1},
                {"source": "manual.pdf", "page": 2},
            ],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        results = self.collection.query(
            query_embeddings=[[0.9, 0.1, 0.0]],
            n_results=1,
            include=["documents", "distances"],
        )

        self.assertEqual(
            results["ids"][0][0],
            "corrosion",
        )

        self.assertLess(
            results["distances"][0][0],
            0.1,
        )

    def test_collection_persistence(self):
        """Verify that stored chunks survive reopening."""
        self.collection.add(
            ids=["chunk_1"],
            documents=["Gas pipeline inspection procedure."],
            embeddings=[[1.0, 0.0, 0.0]],
        )

        reopened_client = chromadb.PersistentClient(
            path=str(Path(self.temp_dir.name) / "chroma_db")
        )

        reopened_collection = reopened_client.get_collection(
            name="gassafe_test",
            embedding_function=None,
        )

        self.assertEqual(reopened_collection.count(), 1)


if __name__ == "__main__":
    unittest.main()
  
