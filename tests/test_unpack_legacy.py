import os
import sys
import unittest
import zipfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from unpack import (
    MAX_ARCHIVE_SIZE_BYTES,
    _extract_zip_safely,
    _preflight_rar,
    _validate_archive_member_names,
    cleanup_stale_extracting_dirs,
    decode_legacy_zip_name,
    unpack_archives,
)


class LegacyZipInfo(zipfile.ZipInfo):
    """Позволяет тесту записать байты имени без UTF-8-флага."""

    def _encodeFilenameFlags(self):
        return self.filename.encode("cp437"), self.flag_bits & ~0x800


class LegacyZipEncodingTests(unittest.TestCase):
    def test_cp866_filename_is_restored(self):
        expected = "лице макета.tif"
        mojibake = expected.encode("cp866").decode("cp437")
        self.assertEqual(decode_legacy_zip_name(mojibake, 0), expected)

    def test_cp1251_filename_is_restored(self):
        expected = "Привет мир.txt"
        mojibake = expected.encode("cp1251").decode("cp437")
        self.assertEqual(decode_legacy_zip_name(mojibake, 0), expected)

    def test_unflagged_utf8_filename_is_restored(self):
        expected = "NK0625 стікерпак 6х2cm (мікс 1).pdf"
        mojibake = expected.encode("utf-8").decode("cp437")
        self.assertEqual(decode_legacy_zip_name(mojibake, 0x8), expected)

    def test_utf8_filename_is_not_changed(self):
        expected = "макети/зворот.tif"
        self.assertEqual(decode_legacy_zip_name(expected, 0x800), expected)

    def test_legacy_zip_is_extracted_with_readable_cyrillic(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "legacy.zip"
            expected = "макеты/лице.tif"
            mojibake = expected.encode("cp866").decode("cp437")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(LegacyZipInfo(mojibake), b"layout")

            output = root / "output"
            output.mkdir()
            _extract_zip_safely(zip_path, output)

            self.assertEqual((output / expected).read_bytes(), b"layout")

    def test_unflagged_utf8_zip_is_extracted_with_readable_cyrillic(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "unflagged-utf8.zip"
            expected = "макети/стікерпак.pdf"
            mojibake = expected.encode("utf-8").decode("cp437")
            info = LegacyZipInfo(mojibake)
            info.flag_bits = 0x8
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(info, b"layout")

            output = root / "output"
            output.mkdir()
            _extract_zip_safely(zip_path, output)

            self.assertEqual((output / expected).read_bytes(), b"layout")

    def test_corrected_extractor_still_blocks_path_traversal(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "unsafe.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("../outside.txt", b"unsafe")

            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "небезопасный путь"):
                _extract_zip_safely(zip_path, output)
            self.assertFalse((root / "outside.txt").exists())

    def test_symbolic_link_member_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "symlink.zip"
            link = zipfile.ZipInfo("link.tif")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(link, "target.tif")

            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "символическую ссылку"):
                _extract_zip_safely(zip_path, output)


class ArchiveLimitTests(unittest.TestCase):
    def test_zip_bomb_metadata_is_rejected_before_writing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "bomb.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("large.txt", b"0" * 100_000)

            output = root / "output"
            output.mkdir()
            with patch("unpack.MAX_COMPRESSION_RATIO", 2):
                with self.assertRaisesRegex(RuntimeError, "Коэффициент распаковки"):
                    _extract_zip_safely(archive_path, output)

    def test_rar_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "небезопасный путь"):
            _validate_archive_member_names(["layouts/front.tif", "../outside.tif"], "RAR")

    def test_rar_is_listed_before_extraction(self):
        with TemporaryDirectory() as temp:
            archive = Path(temp) / "safe.rar"
            archive.write_bytes(b"rar")
            listing = (
                b"Name: layouts/front.tif\nType: File\nSize: 100\nPacked size: 50\n\n"
                b"Name: layouts/back.tif\nType: File\nSize: 100\nPacked size: 50\n\n"
            )
            completed = Mock(stdout=listing, stderr=b"")
            with patch("unpack.subprocess.run", return_value=completed) as run:
                _preflight_rar("unrar", archive)

            self.assertEqual(run.call_args.args[0][:4], ["unrar", "lt", "-c-", "-p-"])

    def test_old_extracting_folder_is_removed_but_fresh_one_is_kept(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / ".extracting_old_123"
            fresh = root / ".extracting_fresh_456"
            unrelated = root / ".cache"
            old.mkdir()
            fresh.mkdir()
            unrelated.mkdir()
            now = time.time()
            os.utime(old, (now - 100_000, now - 100_000))

            removed = cleanup_stale_extracting_dirs(root, now=now)

            self.assertEqual(removed, [old])
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(unrelated.exists())

    def test_pending_conflict_keeps_archive_retryable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "order_4-0.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("layout.tif", b"layout")
            folder = root / "order_4-0"
            folder.mkdir()
            (folder / ".conflict_pending").write_text("pending", encoding="utf-8")

            unpack_archives(str(root))

            self.assertTrue(archive.exists())
            self.assertFalse((root / "_TROUBLES_").exists())

    def test_archive_over_one_and_half_gb_moves_to_troubles(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "oversized.zip"
            with archive.open("wb") as stream:
                stream.truncate(MAX_ARCHIVE_SIZE_BYTES + 1)

            unpack_archives(str(root))

            moved = root / "_TROUBLES_" / archive.name
            problem = root / "_TROUBLES_" / "oversized_PROBLEM.txt"
            self.assertTrue(moved.exists())
            self.assertFalse(archive.exists())
            self.assertIn("превышает лимит 1,5 ГБ", problem.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
