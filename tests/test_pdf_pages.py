import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pypdf import PdfWriter

from pdf_pages import PdfPageCountError, count_layout_units, count_pdf_pages


class PdfPageTests(unittest.TestCase):
    def test_counts_pages_in_pdf(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "layout.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.add_blank_page(width=100, height=100)
            with path.open("wb") as stream:
                writer.write(stream)

            self.assertEqual(count_pdf_pages(path), 2)
            self.assertEqual(count_layout_units([path])[0], 2)

    def test_non_pdf_layout_counts_as_one_unit(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "layout.jpg"
            path.write_bytes(b"image")
            self.assertEqual(count_layout_units([path])[0], 1)

    def test_broken_pdf_is_rejected(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "broken.pdf"
            path.write_bytes(b"not a pdf")
            with self.assertRaises(PdfPageCountError):
                count_pdf_pages(path)


if __name__ == "__main__":
    unittest.main()
