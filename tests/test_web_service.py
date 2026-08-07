import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from web.service import WebPipelineService, WebServiceError


class WebPipelineServiceTests(unittest.TestCase):
    def wait_for_analysis(self, service, run_id):
        for _ in range(100):
            run = service.get(run_id)
            if run["status"] not in {"queued", "scanning", "analyzing"}:
                return run
            time.sleep(0.01)
        self.fail("analysis did not finish")

    def test_analysis_waits_for_approval_then_renames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "job_4-0_(1-12345)_face"
            folder.mkdir()
            source = folder / "layout.tif"
            source.write_bytes(b"layout")
            service = WebPipelineService(root / "state.json", (root,))
            with patch("web.service.unpack_archives"), patch("web.service.fetch_suborders", return_value=[]):
                created = service.start({"source_dir": str(root), "output_dir": str(root)})
                run = self.wait_for_analysis(service, created["id"])
            self.assertEqual(run["status"], "awaiting_decisions")
            self.assertTrue(source.exists())
            order = next(iter(run["orders"].values()))
            item = order["items"][0]
            service.approve(created["id"], order["order_id"], {item["file_id"]: item["proposed_name"]})
            self.assertFalse(source.exists())
            self.assertTrue((folder / item["proposed_name"]).exists())
            self.assertTrue((folder / ".done").exists())

    def test_custom_name_cannot_change_extension(self):
        with self.assertRaises(WebServiceError):
            WebPipelineService._valid_name("changed.pdf", "source.tif")

    def test_paths_outside_allowed_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            service = WebPipelineService(Path(allowed) / "state.json", (Path(allowed),))
            with self.assertRaises(WebServiceError):
                service.start({"source_dir": outside})

    def test_legacy_web_flags_are_not_shadowed_by_false_canonical_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = WebPipelineService(root / "state.json", (root,))
            with patch("web.service.threading.Thread.start"):
                created = service.start({
                    "source_dir": str(root),
                    "use_local_llm": False,
                    "use_llm": True,
                    "strict_order_ids": False,
                    "strict": True,
                })
            run = service.get(created["id"])
            self.assertTrue(run["use_local_llm"])
            self.assertTrue(run["strict_order_ids"])


if __name__ == "__main__":
    unittest.main()
