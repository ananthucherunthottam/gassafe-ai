
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import chromadb


SOURCE_PATH = Path(__file__).resolve().parents[1] / "ingest.py"
ACTIVE_NAME = "gas_safety_pdf"


def load_indexing_functions():
    """
    Load indexing functions without importing
    heavyweight AI dependencies.
    """
    tree = ast.parse(
        SOURCE_PATH.read_text(encoding="utf-8")
    )

    names = {
        "collection_exists",
        "build_collection",
        "promote_collection",
    }

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    if len(functions) != len(names):
        raise RuntimeError(
            "Required indexing functions not found."
        )

    module = ast.Module(
        body=functions,
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    namespace = {
        "COLLECTION_NAME": ACTIVE_NAME,
        "BATCH_SIZE": 2,
    }

    import uuid
    namespace["uuid"] = uuid

    exec(
        compile(module, str(SOURCE_PATH), "exec"),
        namespace,
    )

    return namespace


class TestSafeIndexing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.client = chromadb.PersistentClient(
            path=str(Path(self.temp_dir.name) / "database")
        )

        functions = load_indexing_functions()

        self.collection_exists = functions[
            "collection_exists"
        ]

        self.build_collection = functions[
            "build_collection"
        ]

        self.promote_collection = functions[
            "promote_collection"
        ]

        self.client.create_collection(
            name=ACTIVE_NAME,
            embedding_function=None,
        ).add(
            ids=["old_chunk"],
            documents=["Previous approved safety procedure."],
            embeddings=[[1.0, 0.0, 0.0]],
        )

    def tearDown(self):
        self.client = None
        self.temp_dir.cleanup()

    def create_staging_collection(self):
        collection = self.client.create_collection(
            name="staging_test",
            embedding_function=None,
        )

        collection.add(
            ids=["new_chunk"],
            documents=["Updated safety inspection procedure."],
            embeddings=[[0.0, 1.0, 0.0]],
        )

        return collection

    def test_existing_collection_detection(self):
        """Check collection existence detection."""
        self.assertTrue(
            self.collection_exists(
                self.client,
                ACTIVE_NAME,
            )
        )

        self.assertFalse(
            self.collection_exists(
                self.client,
                "missing_collection",
            )
        )

    def test_staging_does_not_replace_active_index(self):
        """Building staging must preserve active data."""
        self.create_staging_collection()

        active = self.client.get_collection(
            name=ACTIVE_NAME,
            embedding_function=None,
        )

        self.assertEqual(
            active.get()["ids"],
            ["old_chunk"],
        )

    def test_successful_promotion(self):
        """Promote a verified staging collection."""
        self.create_staging_collection()

        self.promote_collection(
            self.client,
            "staging_test",
        )

        active = self.client.get_collection(
            name=ACTIVE_NAME,
            embedding_function=None,
        )

        self.assertEqual(
            active.get()["ids"],
            ["new_chunk"],
        )

        self.assertFalse(
            self.collection_exists(
                self.client,
                "staging_test",
            )
        )

    def test_failed_build_preserves_old_index(self):
        """A failed embedding operation preserves old data."""
        model = MagicMock()

        model.encode.side_effect = RuntimeError(
            "Simulated embedding failure"
        )

        with self.assertRaises(RuntimeError):
            self.build_collection(
                client=self.client,
                collection_name="staging_failure",
                documents=["New safety procedure."],
                metadatas=[{
                    "source": "test.pdf",
                    "page": 1,
                }],
                ids=["new_chunk"],
                model=model,
            )

        active = self.client.get_collection(
            name=ACTIVE_NAME,
            embedding_function=None,
        )

        self.assertEqual(
            active.get()["ids"],
            ["old_chunk"],
        )

    def test_failed_promotion_restores_old_index(self):
        """Simulate a promotion failure and verify rollback."""
        self.create_staging_collection()

        real_get_collection = self.client.get_collection

        client_proxy = MagicMock(
            wraps=self.client
        )

        def mocked_get_collection(*args, **kwargs):
            name = kwargs.get("name")

            if name is None and args:
                name = args[0]

            if name == "staging_test":
                collection = real_get_collection(
                    name=name,
                    embedding_function=None,
                )

                def fail_rename(*args, **kwargs):
                    raise RuntimeError(
                        "Simulated promotion failure"
                    )

                collection.modify = fail_rename
                return collection

            return real_get_collection(
                *args,
                **kwargs,
            )

        client_proxy.get_collection.side_effect = (
            mocked_get_collection
        )

        with self.assertRaises(RuntimeError):
            self.promote_collection(
                client_proxy,
                "staging_test",
            )

        active = self.client.get_collection(
            name=ACTIVE_NAME,
            embedding_function=None,
        )

        self.assertEqual(
            active.get()["ids"],
            ["old_chunk"],
        )


if __name__ == "__main__":
    unittest.main()
  
