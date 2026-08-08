import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pypdf import PdfWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api_core import (
    AmbiguousSidesError,
    ApiCoreError,
    _manual_reason_category,
    _name_for_side,
    build_api_mapping,
    files_per_suborder,
    process_archives_with_api,
)
from local_llm import LlmProposal, LocalLlmError, SideAssignment
from website_parser import SiteAccessError


def pdf_bytes(page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


class ApiCoreTests(unittest.TestCase):
    def test_llm_refusal_is_reported_as_ambiguous_sides(self):
        self.assertEqual(
            _manual_reason_category(
                "Недостаточно информации для назначения конкретного файла "
                "на лицевую или обратную сторону заказа"
            ),
            "спорные стороны",
        )

    def test_two_sided_mapping_uses_suborder_and_face_back(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            first = folder / "1.tif"
            second = folder / "2.tif"
            first.write_bytes(b"face")
            second.write_bytes(b"back")

            mapping = build_api_mapping(folder, [second, first], ["90001"])

            self.assertEqual(
                [(source.name, destination.name) for source, destination in mapping],
                [
                    ("1.tif", "job_4-4_(1-90001)_face.tif"),
                    ("2.tif", "job_4-4_(1-90001)_back.tif"),
                ],
            )

    def test_count_mismatch_is_rejected_before_rename(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            layout = folder / "1.pdf"
            layout.write_bytes(pdf_bytes())

            with self.assertRaises(ApiCoreError):
                build_api_mapping(folder, [layout], ["90001"])

            self.assertTrue(layout.exists())

    def test_two_sided_mapping_respects_explicit_cyrillic_side_names(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            back = folder / "зворот.tif"
            face = folder / "лице.tif"
            back.write_bytes(b"back")
            face.write_bytes(b"face")

            mapping = build_api_mapping(folder, [back, face], ["12345"])

            self.assertEqual(
                [(source.name, destination.name) for source, destination in mapping],
                [
                    ("лице.tif", "job_4-4_(1-12345)_face.tif"),
                    ("зворот.tif", "job_4-4_(1-12345)_back.tif"),
                ],
            )

    def test_ambiguous_two_sided_files_are_not_renamed_by_sort_order(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            alpha = folder / "alpha.tif"
            beta = folder / "beta.tif"
            alpha.write_bytes(b"a")
            beta.write_bytes(b"b")

            with self.assertRaises(AmbiguousSidesError):
                build_api_mapping(folder, [alpha, beta], ["12345"])

            self.assertTrue(alpha.exists())
            self.assertTrue(beta.exists())

    def test_numbered_pairs_map_first_side_to_face_and_second_to_back(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_1-1_(1-12345)_face"
            folder.mkdir()
            files = []
            for group in range(1, 9):
                nested = folder / "8 шт проба"
                nested.mkdir(exist_ok=True)
                for side in (1, 2):
                    path = nested / f"{group}_{side}.jpeg"
                    path.write_bytes(b"image")
                    files.append(path)

            order_ids = [str(90000 + index) for index in range(1, 9)]
            mapping = build_api_mapping(folder, list(reversed(files)), order_ids)

            self.assertEqual(len(mapping), 16)
            for index, order_id in enumerate(order_ids, start=1):
                pair = mapping[(index - 1) * 2:index * 2]
                self.assertEqual([source.name for source, _ in pair], [f"{index}_1.jpeg", f"{index}_2.jpeg"])
                self.assertIn(f"(1-{order_id})_face.jpeg", pair[0][1].name)
                self.assertIn(f"(1-{order_id})_back.jpeg", pair[1][1].name)

    def test_two_files_ending_in_one_and_two_map_to_face_and_back(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-25699063)_face"
            folder.mkdir()
            back = folder / "макет флаєр 2.pdf"
            face = folder / "макет флаєра 1.pdf"
            back.write_bytes(pdf_bytes())
            face.write_bytes(pdf_bytes())

            mapping = build_api_mapping(folder, [back, face], ["25699063"])

            self.assertEqual(
                [(source.name, destination.name) for source, destination in mapping],
                [
                    ("макет флаєра 1.pdf", "job_4-4_(1-25699063)_face.pdf"),
                    ("макет флаєр 2.pdf", "job_4-4_(1-25699063)_back.pdf"),
                ],
            )

    def test_two_page_pdf_is_one_valid_two_sided_order(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            layout = folder / "layout.pdf"
            layout.write_bytes(pdf_bytes(page_count=2))

            mapping = build_api_mapping(folder, [layout], ["12345"])

            self.assertEqual(
                [(source.name, destination.name) for source, destination in mapping],
                [("layout.pdf", "job_4-4_(1-12345)_face_1.pdf")],
            )

    def test_two_two_page_pdfs_match_two_two_sided_orders(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            first = folder / "layout_12345.pdf"
            second = folder / "layout_12346.pdf"
            first.write_bytes(pdf_bytes(page_count=2))
            second.write_bytes(pdf_bytes(page_count=2))

            mapping = build_api_mapping(folder, [first, second], ["12345", "12346"])

            self.assertEqual(len(mapping), 2)
            self.assertIn("12345", mapping[0][1].name)
            self.assertIn("12346", mapping[1][1].name)

    def test_one_page_pdf_is_rejected_for_two_sided_order(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            layout = folder / "layout.pdf"
            layout.write_bytes(pdf_bytes(page_count=1))

            with self.assertRaisesRegex(ApiCoreError, "не хватает печатной стороны"):
                build_api_mapping(folder, [layout], ["12345"])

    def test_two_one_page_pdfs_are_face_and_back_of_one_order(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            face = folder / "лицо.pdf"
            back = folder / "оборот.pdf"
            face.write_bytes(pdf_bytes(page_count=1))
            back.write_bytes(pdf_bytes(page_count=1))

            mapping = build_api_mapping(folder, [back, face], ["12345"])

            self.assertEqual(
                [(source.name, destination.name) for source, destination in mapping],
                [
                    ("лицо.pdf", "job_4-4_(1-12345)_face.pdf"),
                    ("оборот.pdf", "job_4-4_(1-12345)_back.pdf"),
                ],
            )

    def test_one_page_pdf_and_tiff_can_form_face_back_pair(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            face = folder / "лицо.pdf"
            back = folder / "оборот.tif"
            face.write_bytes(pdf_bytes(page_count=1))
            back.write_bytes(b"back")

            mapping = build_api_mapping(folder, [back, face], ["12345"])

            self.assertEqual(
                [(source.name, destination.suffix) for source, destination in mapping],
                [("лицо.pdf", ".pdf"), ("оборот.tif", ".tif")],
            )

    def test_two_one_page_pdfs_match_two_one_sided_orders(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-0_(1-12345)_face"
            folder.mkdir()
            first = folder / "layout_12345.pdf"
            second = folder / "layout_12346.pdf"
            first.write_bytes(pdf_bytes(page_count=1))
            second.write_bytes(pdf_bytes(page_count=1))

            mapping = build_api_mapping(folder, [first, second], ["12345", "12346"])

            self.assertEqual(len(mapping), 2)
            self.assertIn("12345", mapping[0][1].name)
            self.assertIn("12346", mapping[1][1].name)

    def test_multiple_orders_without_explicit_ids_use_sequential_grouping(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-0_(1-12345)_face"
            folder.mkdir()
            first = folder / "version_1_size_90.pdf"
            second = folder / "version_2_size_90.pdf"
            first.write_bytes(pdf_bytes())
            second.write_bytes(pdf_bytes())

            mapping = build_api_mapping(
                folder, [second, first], ["12345", "12346"]
            )

            self.assertEqual([source for source, _ in mapping], [first, second])
            self.assertIn("12345", mapping[0][1].name)
            self.assertIn("12346", mapping[1][1].name)

    def test_strict_mode_requires_explicit_order_ids(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_4-0_(1-12345)_face"
            folder.mkdir()
            first = folder / "version_1_size_90.pdf"
            second = folder / "version_2_size_90.pdf"
            first.write_bytes(pdf_bytes())
            second.write_bytes(pdf_bytes())

            with self.assertRaisesRegex(ApiCoreError, "полный номер заказа"):
                build_api_mapping(
                    folder,
                    [first, second],
                    ["12345", "12346"],
                    strict_order_ids=True,
                )

    def test_zero_one_is_one_back_side(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "job_0-1_(1-12345)_face"
            folder.mkdir()
            layout = folder / "layout.tif"
            layout.write_bytes(b"back")

            mapping = build_api_mapping(folder, [layout], ["90001"])

            self.assertEqual(files_per_suborder(folder.name), 1)
            self.assertEqual(
                mapping[0][1].name,
                "job_0-1_(1-90001)_back.tif",
            )

    def test_order_replacement_only_changes_structural_order_block(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "12345_job_4-0_(1-12345)_face"
            folder.mkdir()
            layout = folder / "layout.tif"
            layout.write_bytes(b"face")

            mapping = build_api_mapping(folder, [layout], ["90001"])

            self.assertEqual(
                mapping[0][1].name,
                "12345_job_4-0_(1-90001)_face_1.tif",
            )

    def test_side_marker_does_not_replace_part_of_word(self):
        self.assertEqual(
            _name_for_side("job_interface_1-face", "back"),
            "job_interface_1-back",
        )
        self.assertEqual(
            _name_for_side("job_backing", "face"),
            "job_backing-face",
        )

    @patch("api_core.unpack_archives")
    @patch("api_core.fetch_suborders", return_value=[])
    def test_pipeline_renames_and_marks_order_done(self, fetch, unpack):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "job_4-0_(1-12345)_face"
            folder.mkdir()
            (folder / "layout.pdf").write_bytes(pdf_bytes())

            summary = process_archives_with_api(root, api_key="test-key")

            self.assertTrue(summary.ok)
            self.assertEqual(summary.processed, [folder.name])
            self.assertTrue((folder / "job_4-0_(1-12345)_face_1.pdf").exists())
            self.assertTrue((folder / ".done").exists())
            fetch.assert_called_once_with("12345", api_key="test-key", timeout=10)

    @patch("api_core.fetch_suborders", return_value=[])
    def test_pipeline_finds_zip_and_keeps_it_inside_order_folder(self, fetch):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive_name = "job_4-0_(1-12345)_face.zip"
            with zipfile.ZipFile(root / archive_name, "w") as archive:
                archive.writestr("layout.pdf", pdf_bytes())

            summary = process_archives_with_api(root, api_key="test-key")

            folder = root / "job_4-0_(1-12345)_face"
            self.assertTrue(summary.ok)
            self.assertTrue((folder / "job_4-0_(1-12345)_face_1.pdf").exists())
            self.assertTrue((folder / ".source_archive" / archive_name).exists())
            self.assertFalse((root / archive_name).exists())

    @patch("api_core.unpack_archives")
    @patch("api_core.fetch_suborders", side_effect=SiteAccessError("timeout"))
    def test_api_failure_leaves_order_untouched_for_retry(self, fetch, unpack):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "job_4-0_(1-12345)_face"
            folder.mkdir()
            layout = folder / "layout.pdf"
            layout.write_bytes(pdf_bytes())

            summary = process_archives_with_api(root, api_key="test-key")

            self.assertFalse(summary.ok)
            self.assertTrue(layout.exists())
            self.assertFalse((folder / ".done").exists())

    @patch("api_core.unpack_archives")
    @patch("api_core.fetch_suborders", return_value=[])
    def test_manual_summary_groups_count_mismatch_reason(self, fetch, unpack):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "job_4-4_(1-12345)_face"
            folder.mkdir()
            (folder / "layout.pdf").write_bytes(pdf_bytes(page_count=1))

            summary = process_archives_with_api(root, api_key="test-key")

            self.assertEqual(
                summary.manual_reason_counts,
                {"несовпадение количества/страниц": 1},
            )
            self.assertTrue(
                (root / "_REQUIRES_MANUAL_CHECK_" / folder.name).exists()
            )

    def test_sequential_groups_send_each_ambiguous_pair_to_local_llm(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "job_4-4_(1-12345)_face"
            folder.mkdir()
            for name in (
                "1-alpha.tif",
                "1-beta.tif",
                "2-alpha.tif",
                "2-beta.tif",
            ):
                (folder / name).write_bytes(b"layout")

            proposals = [
                LlmProposal(
                    assignments=(
                        SideAssignment(
                            order_id="12345",
                            face="1-alpha.tif",
                            back="1-beta.tif",
                        ),
                    ),
                    confidence=0.9,
                    reason="первая пара",
                ),
                LlmProposal(
                    assignments=(
                        SideAssignment(
                            order_id="12346",
                            face="2-alpha.tif",
                            back="2-beta.tif",
                        ),
                    ),
                    confidence=0.9,
                    reason="вторая пара",
                ),
            ]
            with (
                patch("api_core.unpack_archives"),
                patch("api_core.fetch_suborders", return_value=["12346"]),
                patch(
                    "api_core.request_side_proposal", side_effect=proposals
                ) as request,
            ):
                summary = process_archives_with_api(
                    root,
                    api_key="test-key",
                    use_local_llm=True,
                    confirm_llm=lambda *_: True,
                )

            self.assertTrue(summary.ok)
            self.assertEqual(request.call_count, 2)
            self.assertTrue(
                (folder / "job_4-4_(1-12345)_face.tif").exists()
            )
            self.assertTrue(
                (folder / "job_4-4_(1-12346)_back.tif").exists()
            )

    def test_visual_layer_is_used_after_filename_llm_refuses(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "job_4-4_(1-12345)_face"
            folder.mkdir()
            first = folder / "alpha.tif"
            second = folder / "beta.tif"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            visual = LlmProposal(
                assignments=(
                    SideAssignment(
                        order_id="12345",
                        face="alpha.tif",
                        back="beta.tif",
                    ),
                ),
                confidence=0.95,
                reason="логотип и контакты",
            )

            with (
                patch("api_core.unpack_archives"),
                patch("api_core.fetch_suborders", return_value=[]),
                patch(
                    "api_core.request_side_proposal",
                    side_effect=LocalLlmError("имён недостаточно"),
                ),
                patch(
                    "api_core.request_vision_side_proposal",
                    return_value=visual,
                ) as vision,
            ):
                summary = process_archives_with_api(
                    root,
                    api_key="test-key",
                    use_local_llm=True,
                    confirm_llm=lambda *_: True,
                )

            self.assertTrue(summary.ok)
            vision.assert_called_once()
            self.assertTrue((folder / "job_4-4_(1-12345)_face.tif").exists())

    def test_interrupted_archive_move_is_resumed_without_second_rename(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "job_4-0_(1-12345)_face"
            folder.mkdir()
            (folder / "layout.pdf").write_bytes(pdf_bytes())

            with (
                patch("api_core.unpack_archives"),
                patch("api_core.fetch_suborders", return_value=[]) as fetch,
                patch("api_core._archive_source", side_effect=OSError("NAS offline")),
            ):
                first = process_archives_with_api(root, api_key="test-key")

            self.assertFalse(first.ok)
            self.assertFalse((folder / ".done").exists())
            self.assertTrue((folder / "job_4-0_(1-12345)_face_1.pdf").exists())

            with (
                patch("api_core.unpack_archives"),
                patch("api_core.fetch_suborders") as second_fetch,
            ):
                second = process_archives_with_api(root, api_key="test-key")

            self.assertTrue(second.ok)
            self.assertTrue((folder / ".done").exists())
            second_fetch.assert_not_called()
            fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
