import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from classifier import classify_face_back, classify_face_back_paths
from file_discovery import list_layout_files
from pypdf import PdfWriter


class ClassifierTests(unittest.TestCase):
    def test_common_face_and_back_markers(self):
        cases = [
            ("лице.tif", "оборот.tif"),
            ("02_Лицо.tif", "02_Оборот.tif"),
            ("макет лицьова.tif", "макет зворотна.tif"),
            ("Сторона А.tif", "Сторона Б.tif"),
            ("макет титул.tif", "макет зворот.tif"),
            ("front.pdf", "back.pdf"),
            ("card-cover.pdf", "card-reverse.pdf"),
            ("card-F.tif", "card-B.tif"),
            ("card_1.tif", "card_2.tif"),
            ("макет лиц.tif", "макет обор.tif"),
            ("обложка.tif", "изнанка.tif"),
            ("первая сторона.tif", "вторая сторона.tif"),
            ("обкладинка.tif", "виворіт.tif"),
            ("перша сторона.tif", "друга сторона.tif"),
            ("obverse.tif", "verso.tif"),
            ("page_1.tif", "page_2.tif"),
            ("макет-прд.tif", "макет-звр.tif"),
            ("card-f1.tif", "card-b2.tif"),
            ("макет(1).tif", "макет(2).tif"),
            ("макет-a.tif", "макет-z.tif"),
            ("cardfrontfinal.tif", "cardreversefinal.tif"),
            ("cover-design.tif", "businesscardback.tif"),
            ("Deluxe_(Flyer)_Outer.pdf", "Deluxe_(Flyer)_Inner.pdf"),
            ("10_лицо (2).jpg", "10_оборот (1).jpg"),
        ]
        for face, back in cases:
            with self.subTest(face=face, back=back):
                self.assertEqual(
                    classify_face_back([back, face]),
                    {"face": face, "back": back},
                )

    def test_preview_is_not_treated_as_print_layout(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            face = root / "Emika_лице.tif"
            back = root / "Emika_оборот.tif"
            preview = root / "Emika_ПРВЕЬЮ.jpg"
            for path in (face, back, preview):
                path.write_bytes(b"data")

            files = list_layout_files(root)

            self.assertEqual(files, [face, back])
            self.assertEqual(
                classify_face_back_paths(files),
                {"face": face, "back": back},
            )

    def test_unknown_pair_stays_ambiguous(self):
        self.assertEqual(
            classify_face_back(["alpha.tif", "beta.tif"]),
            {"face": None, "back": None},
        )

    def test_one_explicit_marker_determines_the_other_file(self):
        self.assertEqual(
            classify_face_back(["design.tif", "оборот.tif"]),
            {"face": "design.tif", "back": "оборот.tif"},
        )

    def test_short_markers_do_not_match_inside_words_or_numbers(self):
        self.assertEqual(
            classify_face_back(["brief.tif", "job-12341.tif"]),
            {"face": None, "back": None},
        )

    def test_pdf_preview_is_not_treated_as_print_layout(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            layout = root / "лицо.tif"
            preview = root / "preview.pdf"
            layout.write_bytes(b"layout")
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with preview.open("wb") as stream:
                writer.write(stream)

            self.assertEqual(list_layout_files(root), [layout])


if __name__ == "__main__":
    unittest.main()
