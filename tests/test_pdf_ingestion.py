
import ast
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader


def load_split_text():
    """Load the existing chunking function without AI dependencies."""
    source_path = Path(__file__).resolve().parents[1] / "ingest.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "split_text"
    )

    function.args.defaults = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["split_text"]


split_text = load_split_text()


def create_test_pdf(path):
    """Create a minimal one-page PDF containing searchable text."""
    text = "Gas pipeline safety inspection"
    stream = f"BT /F1 12 Tf 50 150 Td ({text}) Tj ET".encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii")
        + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(
            f"{index} 0 obj\n".encode("ascii")
            + obj + b"\nendobj\n"
        )

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    path.write_bytes(pdf)


class TestPDFIngestion(unittest.TestCase):

    def test_pdf_text_extraction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            create_test_pdf(pdf_path)

            reader = PdfReader(str(pdf_path))
            self.assertEqual(len(reader.pages), 1)

            text = reader.pages[0].extract_text()
            self.assertIn("Gas pipeline safety inspection", text)

    def test_pdf_text_chunking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            create_test_pdf(pdf_path)

            reader = PdfReader(str(pdf_path))
            text = reader.pages[0].extract_text() or ""

            chunks = split_text(text, 20, 5)

            self.assertGreater(len(chunks), 1)
            self.assertTrue(
                all(len(chunk) <= 20 for chunk in chunks)
            )


if __name__ == "__main__":
    unittest.main()
  
