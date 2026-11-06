"""Tradovate lifecycle replay; no native UI, mouse or broker access."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from dataclasses import replace
from unittest.mock import Mock

from position_guard import AccountMonitor, AccountView, PositionGate, parse_view, EMPTY

CFG = dict(position_guard=dict(account="apex-test", broker="Tradovate"))


class TradovateActivityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.now = 100.
        self.monitor = AccountMonitor(CFG)
        self.monitor.fresh = Mock(side_effect=lambda: self.view())
        self.path = Path(self.folder.name)/"state.json"
        self.gate = PositionGate(CFG, self.monitor, self.path, clock=lambda: self.now)
        self.gate.observe(self.view())
        self.gate.observe(self.view())
        self.assertTrue(self.gate.claim())
        self.assertIsNone(self.gate.reservation_realized_pnl)

    def view(self, position="flat", **changes):
        self.now += .15
        view = AccountView(self.now, "apex-test", "Tradovate", position, 1, 0,
                           True, True, "fixture", (10,20), (30,20), True,
                           entry_observed_at=self.now)
        return replace(view, **changes)

    def close(self):
        first = self.view()
        self.monitor._publish(first)
        self.assertFalse(self.gate.observe(self.monitor.latest))
        self.monitor._publish(self.view())
        self.assertTrue(self.gate.observe(self.monitor.latest))
        self.assertEqual(self.gate.phase, "flat")
        self.assertEqual(json.loads(self.path.read_text())["phase"], "flat")

    def test_open_then_close_between_capture_frames_is_latched_even_at_breakeven(self):
        self.monitor._publish(self.view("open"))
        self.monitor._publish(self.view())  # latest now FLAT; main loop missed OPEN
        self.assertEqual(self.gate.phase, "pending")
        self.assertFalse(self.gate.observe(self.monitor.latest))
        self.assertEqual(self.gate.phase, "open")
        self.monitor._publish(self.view())
        self.assertTrue(self.gate.observe(self.monitor.latest))
        self.assertEqual(self.gate.phase, "flat")
        self.assertTrue(self.gate.claim())
        self.assertFalse(self.gate.claim())

    def test_order_activity_on_another_tab_then_closed_account_releases(self):
        activity = self.view("unknown", working_orders=2)
        self.monitor._publish(activity)
        self.close()
        self.assertEqual(self.gate.release_evidence, "working_orders_cleared_and_two_flat_reads")

    def test_positive_badge_is_open_even_with_orders_tab_selected(self):
        self.now += .15
        activity = parse_view(dict(account="apex-test", broker="Tradovate", page_ok=True,
                                   positions_selected=False, table_readable=True,
                                   position_count=1, empty_text=EMPTY, working_orders=0), self.now, CFG)
        self.assertEqual(activity.position, "open")
        self.monitor._publish(activity)
        self.close()

    def test_another_tab_without_badge_cannot_claim_flat(self):
        view = parse_view(dict(account="apex-test", broker="Tradovate", page_ok=True,
                               positions_selected=False, table_readable=True,
                               position_count=None, empty_text=EMPTY, working_orders=0), self.now, CFG)
        self.assertEqual(view.position, "unknown")

    def test_positive_badge_is_latched_while_position_rows_are_rendering(self):
        self.now += .15
        activity = parse_view(dict(account="apex-test", broker="Tradovate", page_ok=True,
                                   positions_selected=True, table_readable=False,
                                   position_count=1, working_orders=0), self.now, CFG)
        self.assertEqual(activity.position, "open")
        self.monitor._publish(activity)
        self.close()

    def test_tab_change_preserves_positive_position_but_never_flat_permission(self):
        payload = dict(account="apex-test", broker="Tradovate", page_ok=True,
                       positions_selected=True, table_readable=True, position_count=1,
                       position_rows=["NQZ6 Long 1 27350 Close"], empty_text="", working_orders=0,
                       tab_stable=False, entry_surface_clear=True)
        self.now += .15
        activity = parse_view(payload, self.now, CFG)
        self.assertEqual(activity.position, "open")
        self.assertFalse(activity.entry_surface_clear)
        self.monitor._publish(activity)
        self.close()
        payload.update(position_count=0, position_rows=[], empty_text=EMPTY)
        flat_during_switch = parse_view(payload, self.now, CFG)
        self.assertEqual(flat_during_switch.position, "unknown")
        self.assertFalse(self.gate.observe(flat_during_switch))

    def test_cached_flat_older_than_latched_activity_cannot_release(self):
        older = self.view()
        self.monitor._publish(self.view("open"))
        self.assertFalse(self.gate.observe(older))
        self.assertEqual(self.gate.flat_reads, 0)
        self.assertEqual(self.gate.phase, "open")
        self.close()

    def test_wrong_account_activity_does_not_acknowledge_reservation(self):
        self.monitor._publish(self.view("open", account="other"))
        for _ in range(3):
            self.assertFalse(self.gate.observe(self.view()))
        self.assertEqual(self.gate.phase, "pending")

    def test_pre_click_activity_does_not_acknowledge_current_reservation(self):
        self.monitor._publish(self.view("open", observed_at=self.gate.reserved_at-.01))
        self.assertFalse(self.gate.observe(self.view()))
        self.assertEqual(self.gate.phase, "pending")

    def test_duplicate_positive_reads_are_latched_only_once(self):
        self.monitor._publish(self.view("open", working_orders=2))
        self.monitor._publish(self.view("open", working_orders=2))
        self.assertEqual(len(self.monitor.drain_activity()), 1)

    def test_no_activity_still_cannot_expire_by_time(self):
        self.now += 10000
        for _ in range(3):
            self.monitor._publish(self.view())
            self.assertFalse(self.gate.observe(self.monitor.latest))
        self.assertEqual(self.gate.phase, "pending")

    def test_fresh_account_cannot_authorize_stale_entry_control_cache(self):
        self.monitor._publish(self.view("open"))
        self.close()
        view = self.view(entry_observed_at=self.now-10)
        self.assertFalse(self.gate.observe(view))
        self.assertIn("fresh check", self.gate.reason)
        self.assertTrue(self.gate.refresh_for_signal(view))


class FreshEntryGenerationTests(unittest.TestCase):
    def test_new_activity_between_full_read_and_claim_blocks_cached_flat(self):
        for position, orders in (("open", 0), ("unknown", 2)):
            with self.subTest(position=position):
                now = [100.]
                monitor = AccountMonitor(CFG)
                gate = PositionGate(CFG, monitor, None, clock=lambda: now[0])
                def view(position="flat", orders=0):
                    now[0] += .15
                    return AccountView(now[0], "apex-test", "Tradovate", position,
                                       1, orders, True, True, "fixture", (10,20), (30,20), True)
                gate.observe(view())
                gate.observe(view())
                cached_flat = view()
                monitor.fresh = Mock(return_value=cached_flat)
                monitor._publish(view(position, orders))
                self.assertFalse(gate.claim())
                self.assertIn(gate.phase, ("open", "working"))

    def test_fast_updates_cannot_satisfy_a_full_fresh_request(self):
        monitor = AccountMonitor(CFG)
        returned = []
        done = threading.Event()
        def request():
            returned.append(monitor.fresh(timeout=1))
            done.set()
        thread = threading.Thread(target=request)
        thread.start()
        self.assertTrue(monitor.requested.wait(.3))
        fast = AccountView(101, reason="fast account only")
        monitor._publish(fast)
        self.assertFalse(done.wait(.02))
        full = AccountView(102, reason="new full check")
        with monitor.condition:
            monitor._publish(full, monitor.request_generation)
            monitor._publish(AccountView(103, reason="later fast read"))
        self.assertTrue(done.wait(.3))
        thread.join()
        self.assertIs(returned[0], full)

    def test_full_request_timeout_returns_unknown_not_cached_permission(self):
        monitor = AccountMonitor(CFG)
        monitor._publish(AccountView(100, position="flat", quantity=1, entry_surface_clear=True), 0)
        view = monitor.fresh(timeout=.001)
        self.assertEqual(view.position, "unknown")
        self.assertIn("timed out", view.reason)
