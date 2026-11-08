import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm import LlmProposal, LocalLlmError, SideAssignment
from web.service import WebPipelineService, WebServiceError


class WebPipelineServiceTests(unittest.TestCase):
    def wait_for_analysis(self, service, run_id):
        for _ in range(100):
            run = service.get(run_id)
            if run["status"] not in {"queued", "scanning", "analyzing"}:
                return run
            time.sleep(0.01)
        self.fail("analysis did not finish")

    def test_rule_based_analysis_renames_automatically(self):
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
            self.assertEqual(run["status"], "completed")
            self.assertFalse(source.exists())
            order = next(iter(run["orders"].values()))
            item = order["items"][0]
            self.assertEqual(order["status"], "completed")
            self.assertEqual(order["proposal_source"], "rules")
            self.assertTrue((folder / item["proposed_name"]).exists())
            self.assertTrue((folder / ".done").exists())

    def test_llm_proposal_still_waits_for_operator_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "job_4-4_(1-12345)_face"
            folder.mkdir()
            face = folder / "a.tif"
            back = folder / "b.tif"
            face.write_bytes(b"face")
            back.write_bytes(b"back")
            proposed = [
                (face.resolve(), folder.resolve() / "job_4-4_(1-12345)_face.tif"),
                (back.resolve(), folder.resolve() / "job_4-4_(1-12345)_back.tif"),
            ]
            service = WebPipelineService(root / "state.json", (root,))
            with (
                patch("web.service.unpack_archives"),
                patch("web.service.fetch_suborders", return_value=[]),
                patch.object(service, "_mapping_with_optional_llm", return_value=(proposed, True, "", 0.95)),
            ):
                created = service.start({"source_dir": str(root), "output_dir": str(root)})
                run = self.wait_for_analysis(service, created["id"])

            order = next(iter(run["orders"].values()))
            self.assertEqual(run["status"], "awaiting_decisions")
            self.assertEqual(order["status"], "awaiting_decision")
            self.assertEqual(order["proposal_source"], "llm")
            self.assertEqual(order["review_group"], "confident")
            self.assertTrue(face.exists())
            self.assertTrue(back.exists())

    def test_review_group_splits_confident_and_questionable(self):
        group = WebPipelineService._review_group
        items = [
            {"proposal_available": True, "proposed_name": "a_face.tif"},
            {"proposal_available": True, "proposed_name": "a_back.tif"},
        ]
        self.assertEqual(group("awaiting_decision", True, 0.95, items), "confident")
        self.assertEqual(group("awaiting_decision", True, 0.5, items), "questions")
        self.assertEqual(group("needs_attention", True, 0.99, items), "questions")
        blank = [items[0], {"proposal_available": False, "proposed_name": ""}]
        self.assertEqual(group("awaiting_decision", True, 0.99, blank), "questions")
        dup = [items[0], dict(items[0])]
        self.assertEqual(group("awaiting_decision", True, 0.99, dup), "questions")

    def test_partial_llm_plan_keeps_resolved_pair_and_leaves_failed_pair_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "job_4-4_(1-12345)_face"
            folder.mkdir()
            files = [folder / name for name in ("a.tif", "b.tif", "c.tif", "d.tif")]
            for path in files:
                path.write_bytes(b"layout")
            first = LlmProposal(
                assignments=(SideAssignment("12345", "a.tif", "b.tif"),),
                confidence=0.8,
                reason="первая пара распознана",
            )
            with (
                patch("web.service.request_side_proposal", side_effect=[first, LocalLlmError("имена неясны")]),
                patch("web.service.request_vision_side_proposal", side_effect=LocalLlmError("стороны не определены")),
            ):
                operations, used_llm, warning, _confidence = WebPipelineService._mapping_with_optional_llm(
                    folder.resolve(),
                    [path.resolve() for path in files],
                    ["12345", "12346"],
                    use_local_llm=True,
                    strict_order_ids=False,
                )

            self.assertTrue(used_llm)
            self.assertIn("12346", warning)
            self.assertEqual({source.name for source, _ in operations}, {"a.tif", "b.tif"})

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

    def test_runs_are_listed_by_creation_time_not_random_id_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = WebPipelineService(root / "state.json", (root,))
            service.state["runs"] = {
                "z-old": {"id": "z-old", "created_at": "2026-01-01T00:00:00", "orders": {}},
                "a-new": {"id": "a-new", "created_at": "2026-02-01T00:00:00", "orders": {}},
            }
            with patch.object(service, "public_run", side_effect=lambda run: run):
                self.assertEqual([run["id"] for run in service.list_runs()], ["a-new", "z-old"])


if __name__ == "__main__":
    unittest.main()
