"""Offline persistence contracts: temporary directories, no mouse or account."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bot_journal import RunJournal


class RunJournalTests(unittest.TestCase):
    def read(self, journal):
        return [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]

    def test_flushes_session_and_snapshots_nested_config(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = {"region": {"left": 42}, "use_ticket": False}
            with RunJournal("5sec-regime", True, cfg, folder) as journal:
                cfg["region"]["left"] = 99
                journal.record("test_signal", side="buy")
            records = self.read(journal)
            self.assertEqual([r["event"] for r in records], ["session_start", "test_signal", "session_end"])
            self.assertEqual(records[0]["config"]["region"]["left"], 42)
            self.assertEqual(len(records[0]["strategy_source_sha256"]), 64)
            self.assertFalse(records[0]["fills_and_account_position_verified"])
            self.assertTrue(records[0]["strategy_label_is_user_declared"])
            for record in records:
                self.assertEqual(record["mode"], "TEST")
                self.assertEqual(datetime.fromisoformat(record["timestamp_utc"]).utcoffset(), timezone.utc.utcoffset(None))
                self.assertGreater(record["epoch_ns"], 0)

    def test_two_runs_have_distinct_ids_and_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with RunJournal(None, False, {}, folder) as a:
                pass
            with RunJournal(None, False, {}, folder) as b:
                pass
            self.assertNotEqual(a.session_id, b.session_id)
            self.assertNotEqual(a.path, b.path)
            self.assertEqual(self.read(a)[0]["declared_strategy"], "unlabeled")
            self.assertEqual(self.read(a)[0]["mode"], "RUN")

    def test_exception_is_propagated_and_journal_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "simulation interrupted"):
                with RunJournal("5sec-reversed-regime", True, {}, folder) as journal:
                    raise RuntimeError("simulation interrupted")
            self.assertEqual(self.read(journal)[-1]["outcome"], "error")
            self.assertFalse(journal._thread.is_alive())
            with self.assertRaises(OSError):
                journal.record("after_close")
            journal.close()  # idempotent

    def test_invalid_labels_do_not_create_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for tag in ("../outside", "x/y", "a\\b", "x" * 81):
                with self.assertRaises(ValueError):
                    RunJournal(tag, True, {}, folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_keyboard_interrupt_is_logged_as_interruption_with_detail(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(KeyboardInterrupt):
                with RunJournal("1sec-after-3bars", True, {}, folder) as journal:
                    raise KeyboardInterrupt()
            records = self.read(journal)
            exception = next(record for record in records if record["event"] == "session_exception")
            self.assertEqual(exception["error_type"], "KeyboardInterrupt")
            self.assertIn("KeyboardInterrupt", exception["traceback"])
            self.assertEqual(records[-1]["outcome"], "interrupted")

    def test_existing_file_as_directory_blocks_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "occupied"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                RunJournal("5sec", True, {}, path)
            self.assertEqual(path.read_text(), "preserve")
