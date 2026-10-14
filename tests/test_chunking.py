
import ast
import unittest
from pathlib import Path


def load_split_text():
    """Load split_text without importing heavyweight AI libraries."""
    source_path = Path(__file__).resolve().parents[1] / "ingest.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "split_text"
    )

    # Remove defaults that refer to config.py constants.
    function.args.defaults = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["split_text"]


split_text = load_split_text()


class TestTextChunking(unittest.TestCase):

    def test_short_text(self):
        result = split_text("Gas safety", 20, 5)
        self.assertEqual(result, ["Gas safety"])

    def test_empty_text(self):
        result = split_text("", 20, 5)
        self.assertEqual(result, [])

    def test_whitespace_normalization(self):
        result = split_text("Gas   pipeline\n  safety", 50, 10)
        self.assertEqual(result, ["Gas pipeline safety"])

    def test_overlapping_chunks(self):
        result = split_text("ABCDEFGHIJ", 6, 2)
        self.assertEqual(result, ["ABCDEF", "EFGHIJ", "IJ"])

    def test_invalid_chunk_settings(self):
        with self.assertRaises(ValueError):
            split_text("Safety text", 5, 5)

        with self.assertRaises(ValueError):
            split_text("Safety text", 5, -1)


if __name__ == "__main__":
    unittest.main()
  
