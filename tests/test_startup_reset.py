"""Startup reset against fake account observations and fake GUI only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock
from dataclasses import replace

from position_guard import AccountView, PositionGate
import test_ticket_guard as ticket_tests


class StartupResetTests(unittest.TestCase):
    pixel = ticket_tests.TicketGuardTests.pixel
    click = ticket_tests.TicketGuardTests.click

    def setUp(self):
        ticket_tests.TicketGuardTests.setUp(self)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/"position.json"
        self.path.write_text(json.dumps(dict(phase="pending", account="test", broker="Paper Trading")))
        self.now = 100.
        self.changes = {}
        self.locked = False
        self.gates = []
        self.monitor = Mock()
        self.monitor.fresh.side_effect = self.snapshot
        self.bot.AccountMonitor = MagicMock()
        self.bot.AccountMonitor.return_value.__enter__.return_value = self.monitor
        self.bot.BotInstanceLock = MagicMock()
        self.bot.BotInstanceLock.return_value.__enter__.side_effect = self.lock
        self.bot.BotInstanceLock.return_value.__exit__.side_effect = self.unlock
        self.bot.PositionGate = Mock(side_effect=self.gate)
        self.bot.position_session = self.actual_position_session
        self.cfg["use_ticket"] = False
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.mss.MSS = MagicMock()
        self.bot.keyboard.is_pressed.side_effect = self.stop_before_capture

    def stop_before_capture(self, key):
        self.assertTrue(self.locked, "OS lock must stay held through the start of the trading loop")
        return True

    def lock(self):
        self.locked = True

    def unlock(self, *args):
        self.locked = False
        return False

    def snapshot(self):
        self.assertTrue(self.locked, "Reset must hold the same single-instance lock as RUN")
        self.now += .15
        view = AccountView(self.now, "test", "Paper Trading", "flat", 1, 0,
                           True, True, "fixture", (10,20), (30,20), True,
                           entry_observed_at=self.now)
        return replace(view, **self.changes)

    def gate(self, cfg, monitor, dry):
        gate = PositionGate(cfg, monitor, self.path, dry=dry, clock=lambda: self.now)
        self.gates.append(gate)
        return gate

    def test_verified_run_reset_clears_only_local_reservation_before_loop(self):
        self.bot.run(reset_on_start=True)
        self.assertEqual(json.loads(self.path.read_text())["phase"], "flat")
        self.assertGreaterEqual(self.monitor.fresh.call_count, 3)
        self.assertFalse(self.locked)
        self.assertEqual(self.gates[-1].flat_reads, 0)  # reset alone grants no entry
        self.bot.pyautogui.click.assert_not_called()
        self.journal.record.assert_any_call("startup_position_reset", verified=True, simulated=False,
                                            account="test", broker="Paper Trading", phase="flat")

    def test_test_reset_is_simulated_and_preserves_saved_pending_bytes(self):
        before = self.path.read_bytes()
        self.bot.run(dry=True, reset_on_start=True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.gates[-1].phase, "flat")
        self.bot.BotInstanceLock.assert_called_once()
        self.bot.pyautogui.click.assert_not_called()

    def test_open_orders_unknown_wrong_account_and_bad_quantity_refuse_startup(self):
        before = self.path.read_bytes()
        for changes in ({"position": "open"}, {"working_orders": 1}, {"working_orders": None},
                        {"position": "unknown"}, {"quantity": 2}, {"quantity": None},
                        {"account": "other"}, {"page_ok": False}, {"entry_observed_at": 0}):
            with self.subTest(changes=changes):
                self.changes = changes
                with self.assertRaises(self.bot.StartupResetError):
                    self.bot.run(reset_on_start=True)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertFalse(self.locked)
                self.bot.pyautogui.click.assert_not_called()
                self.bot.mss.MSS.assert_not_called()

    def test_cli_refusal_has_nonzero_exit_and_does_not_start_loop(self):
        self.changes = dict(position="open")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.bot.main(["run", "--strategy", "1sec-after-3bars", "--reset-position-lock"])
        self.assertEqual(error.exception.code, 2)
        self.bot.pyautogui.click.assert_not_called()

    def test_corrupt_lock_is_not_deleted_or_overridden(self):
        self.path.write_text("corrupt")
        with self.assertRaises(self.bot.StartupResetError):
            self.bot.run(reset_on_start=True)
        self.assertEqual(self.path.read_text(), "corrupt")
        self.monitor.fresh.assert_not_called()

    def test_cli_forwards_explicit_flag_for_run_and_timed_test(self):
        self.bot.run = Mock()
        self.bot.main(["run", "--strategy", "1sec-after-3bars", "--reset-position-lock"])
        self.bot.run.assert_called_once_with(strategy_tag="1sec-after-3bars", reset_on_start=True)
        self.bot.run.reset_mock()
        self.bot.main(["test", "--strategy", "1sec-after-3bars", "--reset-position-lock", "--test-seconds", "3"])
        self.bot.run.assert_called_once_with(dry=True, strategy_tag="1sec-after-3bars", reset_on_start=True, test_seconds=3.)

    def test_flag_is_rejected_for_calibration_commands(self):
        self.bot.calibrate = Mock()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.bot.main(["calibrate", "--reset-position-lock"])
        self.bot.calibrate.assert_not_called()

    def test_clear_lock_binds_current_approved_account_before_startup_reset(self):
        self.path.write_text(json.dumps(dict(phase="flat")))
        cfg = dict(position_guard=dict(accounts=[dict(account="test", broker="Paper Trading")]))
        gate = PositionGate(cfg, self.monitor, self.path, clock=lambda: self.now)
        self.locked = True
        gate.reset()
        self.assertEqual(gate.account, "test")
        self.assertEqual(gate.phase, "flat")
