import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from operational_state import (
    FileLock,
    InvalidStageTransition,
    LockHeldError,
    OperationalStateError,
    OrderStage,
    OrderStateStore,
    atomic_write_json,
    order_lock,
    run_lock,
    unique_path,
)


class OrderStateStoreTests(unittest.TestCase):
    def test_prepare_and_valid_transitions_persist_state(self):
        with TemporaryDirectory() as temp:
            store = OrderStateStore(Path(temp) / "state.json")
            prepared = store.prepare("12345", run_id="run-1", details={"archive": "a.zip"})
            self.assertEqual(prepared.stage, OrderStage.PREPARED)

            store.transition(OrderStage.RENAMED, details={"files": 2})
            store.transition(OrderStage.ARCHIVED)
            done = store.transition(OrderStage.DONE)

            self.assertEqual(done.stage, OrderStage.DONE)
            self.assertEqual(done.details, {"archive": "a.zip", "files": 2})
            self.assertEqual(store.load(), done)

    def test_invalid_transition_is_rejected_without_changing_file(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            store = OrderStateStore(path)
            store.prepare("12345")
            before = path.read_bytes()
            with self.assertRaises(InvalidStageTransition):
                store.transition(OrderStage.DONE)
            self.assertEqual(path.read_bytes(), before)

    def test_terminal_stage_is_idempotent_but_cannot_advance(self):
        with TemporaryDirectory() as temp:
            store = OrderStateStore(Path(temp) / "state.json")
            store.prepare("12345")
            failed = store.transition(OrderStage.FAILED)
            self.assertEqual(store.transition(OrderStage.FAILED), failed)
            with self.assertRaises(InvalidStageTransition):
                store.transition(OrderStage.PREPARED)

    def test_corrupt_json_is_reported(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(OperationalStateError):
                OrderStateStore(path).load()

    def test_atomic_write_replaces_complete_document_and_cleans_temp(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "state.json"
            path.write_text('{"old": true}', encoding="utf-8")
            atomic_write_json(path, {"new": "данные"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": "данные"})
            self.assertEqual(list(root.glob("state.json.*.tmp")), [])

    @patch("operational_state.os.replace", side_effect=OSError("disk error"))
    def test_failed_atomic_replace_preserves_old_file_and_cleans_temp(self, _replace):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "state.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with self.assertRaises(OSError):
                atomic_write_json(path, {"new": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(root.glob("state.json.*.tmp")), [])


class FileLockTests(unittest.TestCase):
    def test_exclusive_lock_and_context_release(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "run.lock"
            with FileLock(path) as held:
                self.assertEqual(held.owner.pid, os.getpid())
                with self.assertRaises(LockHeldError) as caught:
                    FileLock(path).acquire()
                self.assertFalse(caught.exception.stale)
            self.assertFalse(path.exists())

    def test_stale_lock_requires_explicit_permission_to_break(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "run.lock"
            old = {
                "token": "old",
                "pid": 1,
                "hostname": "host",
                "created_at": "2000-01-01T00:00:00+00:00",
                "created_timestamp": time.time() - 100,
            }
            path.write_text(json.dumps(old), encoding="utf-8")
            lock = FileLock(path, stale_after=10)
            with self.assertRaises(LockHeldError) as caught:
                lock.acquire()
            self.assertTrue(caught.exception.stale)

            lock.acquire(break_stale=True)
            self.assertNotEqual(lock.owner.token, "old")
            lock.release()

    def test_release_does_not_remove_replaced_lock(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "run.lock"
            lock = FileLock(path).acquire()
            replacement = {
                "token": "another",
                "pid": 2,
                "hostname": "host",
                "created_at": "now",
                "created_timestamp": time.time(),
            }
            path.write_text(json.dumps(replacement), encoding="utf-8")
            with self.assertRaises(OperationalStateError):
                lock.release()
            self.assertTrue(path.exists())

    def test_helpers_use_documented_lock_names(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(run_lock(root).path, root / ".unpack-run.lock")
            self.assertEqual(order_lock(root / "order").path, root / "order" / ".order.lock")


class UniquePathTests(unittest.TestCase):
    def test_unique_paths_are_distinct_and_keep_affixes(self):
        with TemporaryDirectory() as temp:
            first = unique_path(temp, prefix="work-", suffix=".tmp")
            second = unique_path(temp, prefix="work-", suffix=".tmp")
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("work-"))
            self.assertTrue(first.name.endswith(".tmp"))
            self.assertFalse(first.exists())


if __name__ == "__main__":
    unittest.main()
