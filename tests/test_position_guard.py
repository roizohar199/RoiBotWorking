"""Synthetic observations only: no browser, mouse, broker or account access."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from position_guard import AccountView, EMPTY, PositionGate, WindowsAccountReader, parse_view, configuration_errors, badge_count, BotInstanceLock, is_position_row, intersect_scan, region_probes, money_value
import test_ticket_guard as ticket_tests

CFG = {"position_guard": {"account": "test-account", "broker": "Paper Trading",
                          "window_title": "test", "page_url": "https://www.tradingview.com/chart/test/"}}


class PositionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "position.json"
        self.now = 100.0
        self.monitor = Mock()
        self.gate = PositionGate(CFG, self.monitor, self.path, clock=lambda: self.now)

    def view(self, position="flat", **changes):
        self.now += 0.15
        return replace(AccountView(self.now, "test-account", "Paper Trading", position,
                                   1, 0, True, True, "fixture", (10, 20), (30, 20)), **changes)

    def arm(self):
        self.assertFalse(self.gate.observe(self.view()))
        self.assertTrue(self.gate.observe(self.view()))

    def claim(self):
        self.monitor.fresh.side_effect = lambda: self.view()
        return self.gate.claim()

    def test_independent_flat_reads_required(self):
        view = self.view()
        for _ in range(20):
            self.assertFalse(self.gate.observe(view))
        self.assertEqual(self.gate.flat_reads, 1)
        self.assertTrue(self.gate.observe(self.view()))

    def test_open_at_start_is_persisted_and_blocks(self):
        self.assertFalse(self.gate.observe(self.view("open")))
        self.assertEqual(json.loads(self.path.read_text())["phase"], "open")

    def test_claim_persists_and_blocks_second_claim(self):
        self.arm()
        self.assertTrue(self.claim())
        self.assertEqual(json.loads(self.path.read_text())["phase"], "pending")
        self.assertFalse(self.claim())

    def test_pending_never_expires_by_time(self):
        self.arm()
        self.claim()
        self.now += 100000
        for _ in range(10):
            self.assertFalse(self.gate.observe(self.view()))
        self.assertEqual(self.gate.phase, "pending")

    def test_open_then_two_flat_reads_releases(self):
        self.arm()
        self.claim()
        self.assertFalse(self.gate.observe(self.view("open")))
        self.assertFalse(self.gate.observe(self.view()))
        self.assertTrue(self.gate.observe(self.view()))
        self.assertTrue(self.claim())

    def test_existing_manual_position_closes(self):
        self.gate.observe(self.view("open"))
        self.assertFalse(self.gate.observe(self.view()))
        self.assertTrue(self.gate.observe(self.view()))

    def test_restart_preserves_pending(self):
        self.arm()
        self.claim()
        other = PositionGate(CFG, self.monitor, self.path, clock=lambda: self.now)
        for _ in range(3):
            self.assertFalse(other.observe(self.view()))
        self.assertEqual(other.phase, "pending")

    def test_quantity_orders_account_page_foreground(self):
        for changes in ({"quantity": 2}, {"quantity": 0}, {"quantity": None},
                        {"working_orders": 1}, {"working_orders": None},
                        {"account": "other"}, {"broker": "other"},
                        {"page_ok": False}, {"foreground": False}):
            with self.subTest(changes=changes):
                self.gate = PositionGate(CFG, self.monitor, None, clock=lambda: self.now)
                self.arm()
                self.assertFalse(self.gate.observe(self.view(**changes)))
                if any(key in changes for key in ("working_orders", "account", "broker", "page_ok")):
                    self.assertEqual(self.gate.flat_reads, 0)

    def test_unknown_stale_and_future_block(self):
        self.arm()
        self.assertFalse(self.gate.observe(self.view("unknown")))
        self.assertFalse(self.gate.observe(self.view(observed_at=self.now - 5)))
        self.assertFalse(self.gate.observe(self.view(observed_at=self.now + 10)))

    def test_final_fresh_check_can_block_claim(self):
        self.arm()
        self.monitor.fresh.side_effect = lambda: self.view("open")
        self.assertFalse(self.gate.claim())

    def test_failed_write_cannot_reserve(self):
        self.arm()
        self.path.mkdir()
        self.monitor.fresh.side_effect = lambda: self.view()
        with self.assertRaises(OSError):
            self.gate.claim()
        self.assertEqual(self.gate.phase, "flat")

    def test_corrupt_or_wrong_account_state_blocks_start(self):
        self.path.write_text("not json")
        with self.assertRaises(RuntimeError):
            PositionGate(CFG, self.monitor, self.path)
        self.path.write_text(json.dumps(dict(phase="pending", account="other", broker="Paper Trading")))
        with self.assertRaises(RuntimeError):
            PositionGate(CFG, self.monitor, self.path)

    def test_test_mode_does_not_modify_state(self):
        self.arm()
        self.claim()
        before = self.path.read_bytes()
        dry = PositionGate(CFG, self.monitor, self.path, dry=True, clock=lambda: self.now)
        dry.observe(self.view("open"))
        dry.observe(self.view())
        dry.observe(self.view())
        self.assertEqual(self.path.read_bytes(), before)

    def test_confirmation_requires_still_flat(self):
        self.arm()
        self.claim()
        self.monitor.fresh.side_effect = lambda: self.view()
        self.assertTrue(self.gate.still_flat())
        self.monitor.fresh.side_effect = lambda: self.view("open")
        self.assertFalse(self.gate.still_flat())

    def test_explicit_reset_cannot_override_open_or_unknown(self):
        self.arm()
        self.claim()
        for state in ("open", "unknown"):
            self.monitor.fresh.side_effect = lambda state=state: self.view(state)
            with self.assertRaises(RuntimeError):
                self.gate.reset()
            self.assertEqual(self.gate.phase, "pending")

    def test_explicit_reset_requires_flat(self):
        self.arm()
        self.claim()
        self.monitor.fresh.side_effect = lambda: self.view()
        self.gate.reset()
        self.assertEqual(self.gate.phase, "flat")

    def test_open_observation_before_reservation_cannot_release_pending(self):
        self.arm()
        old = self.view("open")
        self.claim()
        self.assertFalse(self.gate.observe(old))
        self.assertEqual(self.gate.phase, "pending")
        self.assertFalse(self.gate.observe(self.view()))

    def test_unknown_quote_coordinates_and_ticket_mode_block_claim(self):
        self.arm()
        self.monitor.fresh.side_effect = lambda: self.view(buy_point=None)
        self.assertFalse(self.gate.claim())
        self.gate.cfg = dict(CFG, use_ticket=True)
        self.monitor.fresh.side_effect = lambda: self.view()
        self.assertFalse(self.gate.claim())

    def test_current_ui_coordinates_override_old_calibrated_buttons(self):
        self.arm()
        self.claim()
        original = dict(CFG, buy_btn=[999, 999], sell_btn=[888, 888])
        execution = self.gate.entry_config(original)
        self.assertEqual(execution["buy_btn"], (10, 20))
        self.assertEqual(execution["sell_btn"], (30, 20))
        self.assertEqual(original["buy_btn"], [999, 999])

    def test_both_accounts_bind_once_per_run_and_switching_blocks(self):
        multi = {"position_guard": {"accounts": [dict(account="test-account", broker="Paper Trading"),
                                                 dict(account="apex", broker="Tradovate")]}}
        gate = PositionGate(multi, self.monitor, None, clock=lambda: self.now)
        gate.observe(self.view())
        self.assertEqual((gate.account, gate.broker), ("test-account", "Paper Trading"))
        self.assertFalse(gate.observe(self.view(account="apex", broker="Tradovate")))
        new = PositionGate(multi, self.monitor, None, clock=lambda: self.now)
        new.observe(self.view(account="apex", broker="Tradovate"))
        self.assertEqual((new.account, new.broker), ("apex", "Tradovate"))

    def test_background_open_observation_still_locks_account(self):
        self.assertFalse(self.gate.observe(self.view("open", foreground=False)))
        self.assertEqual(self.gate.phase, "open")

    def test_two_run_instances_cannot_hold_same_os_lock(self):
        import sys
        if sys.platform != "win32":
            self.skipTest("Windows byte lock")
        path = Path(self.folder.name) / "instance.lock"
        with BotInstanceLock(path):
            with self.assertRaises(RuntimeError):
                with BotInstanceLock(path):
                    self.fail("Second RUN acquired the lock")
        with BotInstanceLock(path):
            pass

    def test_uncovered_window_can_enter_without_keyboard_focus(self):
        self.assertFalse(self.gate.observe(self.view(foreground=False, entry_surface_clear=True)))
        self.assertTrue(self.gate.observe(self.view(foreground=False, entry_surface_clear=True)))
        self.monitor.fresh.side_effect = lambda: self.view(foreground=False, entry_surface_clear=True)
        self.assertTrue(self.gate.claim())

    def test_keyboard_focus_cannot_override_a_covered_surface(self):
        self.arm()
        self.monitor.fresh.side_effect = lambda: self.view(foreground=True, entry_surface_clear=False)
        self.assertFalse(self.gate.claim())

    def test_explicit_background_reset_does_not_grant_entry_permission(self):
        self.arm()
        self.claim()
        self.monitor.fresh.side_effect = lambda: self.view(foreground=False, entry_surface_clear=False)
        self.gate.reset()
        self.assertEqual(self.gate.phase, "flat")
        self.assertEqual(self.gate.flat_reads, 0)
        self.assertFalse(self.gate.claim())

    def test_account_closure_releases_open_lock_while_buttons_are_covered(self):
        self.gate.observe(self.view("open", entry_surface_clear=False))
        self.assertFalse(self.gate.observe(self.view(entry_surface_clear=False)))
        self.assertFalse(self.gate.observe(self.view(entry_surface_clear=False)))
        self.assertEqual(self.gate.phase, "flat")
        self.assertEqual(json.loads(self.path.read_text())["phase"], "flat")
        self.monitor.fresh.side_effect = lambda: self.view(entry_surface_clear=False)
        self.assertFalse(self.gate.claim())
        self.assertTrue(self.gate.observe(self.view(entry_surface_clear=True)))

    def test_account_closure_does_not_depend_on_entry_quantity_widget(self):
        self.gate.observe(self.view("open", quantity=None))
        for _ in range(2):
            self.assertFalse(self.gate.observe(self.view(quantity=None)))
        self.assertEqual(self.gate.phase, "flat")
        self.assertIn("quantity", self.gate.reason)

    def test_orders_prevent_closure_even_with_empty_positions(self):
        self.gate.observe(self.view("open"))
        for _ in range(3):
            self.assertFalse(self.gate.observe(self.view(working_orders=2, entry_surface_clear=False)))
        self.assertEqual(self.gate.phase, "open")

    def test_covered_flat_account_cannot_clear_unobserved_pending(self):
        self.arm()
        self.claim()
        for _ in range(3):
            self.assertFalse(self.gate.observe(self.view(entry_surface_clear=False)))
        self.assertEqual(self.gate.phase, "pending")

    def claim_with_pnl(self, amount="100"):
        self.arm()
        self.monitor.fresh.side_effect = lambda: self.view(realized_pnl=amount)
        self.assertTrue(self.gate.claim())

    def test_fast_closed_trade_requires_two_changed_pnl_flat_reads(self):
        self.claim_with_pnl()
        changed = self.view(realized_pnl="120")
        self.assertFalse(self.gate.observe(changed))
        for _ in range(5):
            self.assertFalse(self.gate.observe(changed))
        self.assertTrue(self.gate.observe(self.view(realized_pnl="120")))
        self.assertEqual(self.gate.phase, "flat")
        self.assertEqual(self.gate.release_evidence, "realized_pnl_changed_and_two_flat_reads")
        self.assertEqual(json.loads(self.path.read_text())["phase"], "flat")
        self.monitor.fresh.side_effect = lambda: self.view(realized_pnl="120")
        self.assertTrue(self.gate.claim())
        self.assertEqual(self.gate.reservation_realized_pnl, "120")
        self.assertFalse(self.gate.claim())

    def test_loss_also_proves_closure_but_unchanged_pnl_does_not(self):
        self.claim_with_pnl()
        for _ in range(4):
            self.assertFalse(self.gate.observe(self.view(realized_pnl="100")))
        self.assertEqual(self.gate.flat_reads, 0)
        self.assertFalse(self.gate.observe(self.view(realized_pnl="-50")))
        self.assertTrue(self.gate.observe(self.view(realized_pnl="-50")))

    def test_pnl_closure_proof_cannot_override_open_position_or_orders(self):
        self.claim_with_pnl()
        self.assertFalse(self.gate.observe(self.view(working_orders=2, realized_pnl="120")))
        self.assertEqual(self.gate.phase, "working")
        self.assertFalse(self.gate.observe(self.view("open", realized_pnl="120")))
        self.assertEqual(self.gate.phase, "open")

    def test_pending_without_pnl_baseline_stays_locked(self):
        self.arm()
        self.claim()
        for _ in range(4):
            self.assertFalse(self.gate.observe(self.view(realized_pnl="120")))
        self.assertEqual(self.gate.phase, "pending")

    def test_changed_pnl_wrong_account_unknown_and_stale_cannot_release(self):
        self.claim_with_pnl()
        for changes in ({"account": "other"}, {"broker": "other"}, {"page_ok": False},
                        {"observed_at": self.now - 10}, {"position": "unknown"},
                        {"realized_pnl": None}):
            payload = dict(realized_pnl="120")
            payload.update(changes)
            self.assertFalse(self.gate.observe(self.view(**payload)))
            self.assertEqual(self.gate.phase, "pending")

    def test_old_pnl_change_from_before_click_cannot_release(self):
        self.arm()
        old = self.view(realized_pnl="120")
        self.monitor.fresh.side_effect = lambda: self.view(realized_pnl="100")
        self.assertTrue(self.gate.claim())
        self.assertFalse(self.gate.observe(old))
        self.assertEqual(self.gate.phase, "pending")

    def test_order_acknowledgement_then_flat_releases_if_open_was_missed(self):
        self.arm()
        self.claim()
        self.assertFalse(self.gate.observe(self.view(working_orders=2)))
        self.assertEqual(self.gate.phase, "working")
        self.assertFalse(self.gate.observe(self.view()))
        self.assertTrue(self.gate.observe(self.view()))
        self.assertEqual(self.gate.release_evidence, "working_orders_cleared_and_two_flat_reads")

    def test_slow_open_observation_still_acknowledges_but_never_allows_entry(self):
        self.arm()
        self.claim()
        opening = self.view("open")
        self.now += 2
        self.assertFalse(self.gate.observe(opening))
        self.assertEqual(self.gate.phase, "open")
        self.assertFalse(self.gate.observe(self.view()))
        self.assertTrue(self.gate.observe(self.view()))

    def test_restart_preserves_pnl_baseline_and_closes_after_confirmation(self):
        self.claim_with_pnl()
        self.assertEqual(json.loads(self.path.read_text())["reservation_realized_pnl"], "100")
        other = PositionGate(CFG, self.monitor, self.path, clock=lambda: self.now)
        self.assertFalse(other.observe(self.view(realized_pnl="120")))
        self.assertTrue(other.observe(self.view(realized_pnl="120")))

    def test_restart_preserves_acknowledged_orders_and_blocks_until_flat(self):
        self.arm()
        self.claim()
        self.gate.observe(self.view(working_orders=2))
        other = PositionGate(CFG, self.monitor, self.path, clock=lambda: self.now)
        self.assertFalse(other.observe(self.view(working_orders=1)))
        self.assertFalse(other.observe(self.view()))
        self.assertTrue(other.observe(self.view()))

    def test_pnl_closure_can_clear_lock_but_cannot_authorize_covered_click(self):
        self.claim_with_pnl()
        for _ in range(2):
            self.assertFalse(self.gate.observe(self.view(realized_pnl="120", entry_surface_clear=False)))
        self.assertEqual(self.gate.phase, "flat")
        self.monitor.fresh.side_effect = lambda: self.view(entry_surface_clear=False)
        self.assertFalse(self.gate.claim())

    def test_new_signal_refreshes_visual_failure_without_reserving(self):
        self.arm()
        cached = self.view(entry_surface_clear=False)
        self.assertFalse(self.gate.observe(cached))
        self.monitor.fresh.side_effect = [self.view(entry_surface_clear=False), self.view(entry_surface_clear=True)]
        self.assertTrue(self.gate.refresh_for_signal(cached))
        self.assertEqual(self.monitor.fresh.call_count, 2)
        self.assertEqual(self.gate.phase, "flat")
        self.assertFalse(self.path.exists())

    def test_claim_retries_transient_visual_failure_before_reserving(self):
        self.arm()
        self.monitor.fresh.side_effect = [self.view(entry_surface_clear=False), self.view(entry_surface_clear=True)]
        self.assertTrue(self.gate.claim())
        self.assertEqual(self.gate.phase, "pending")
        self.assertEqual(self.monitor.fresh.call_count, 2)

    def test_persistent_cover_is_not_overridden_by_signal_refresh(self):
        self.arm()
        cached = self.view(entry_surface_clear=False)
        self.gate.observe(cached)
        self.monitor.fresh.side_effect = lambda: self.view(entry_surface_clear=False)
        self.assertFalse(self.gate.refresh_for_signal(cached))
        self.assertEqual(self.monitor.fresh.call_count, 2)
        self.assertEqual(self.gate.phase, "flat")

    def test_open_pending_orders_unknown_and_stale_signals_are_never_retried(self):
        self.arm()
        for changes in ({"position": "open"}, {"working_orders": 1}, {"quantity": 2},
                        {"position": "unknown"}, {"account": "other"},
                        {"observed_at": self.now - 10}):
            self.monitor.fresh.reset_mock()
            payload = dict(entry_surface_clear=False)
            payload.update(changes)
            self.assertFalse(self.gate.refresh_for_signal(self.view(**payload)))
            self.monitor.fresh.assert_not_called()
        self.claim()
        self.monitor.fresh.reset_mock()
        self.assertFalse(self.gate.refresh_for_signal(self.view(entry_surface_clear=False)))
        self.monitor.fresh.assert_not_called()

    def test_position_opening_during_visual_refresh_blocks_both_directions(self):
        self.arm()
        cached = self.view(entry_surface_clear=False)
        self.gate.observe(cached)
        self.monitor.fresh.side_effect = lambda: self.view("open")
        self.assertFalse(self.gate.refresh_for_signal(cached))
        self.assertEqual(self.gate.phase, "open")


class IndependentAccountReaderTests(unittest.TestCase):
    def test_exact_money_parser(self):
        for source, expected in (("\u221211,300.00", "-11300"), ("+1,234.50", "1234.5"),
                                 ("0.00", "0"), ("-0.00", "0")):
            self.assertEqual(money_value(source), expected)
        for source in (None, "", "••••", "Unrealized PnL 12", "1,23.45", "NaN"):
            self.assertIsNone(money_value(source))

    def test_realized_label_reads_only_its_following_value_not_unrealized(self):
        from types import SimpleNamespace
        def node(name, children=()):
            return SimpleNamespace(Name=name, GetChildren=lambda: list(children),
                                   BoundingRectangle=SimpleNamespace(left=0, top=0, right=100, bottom=20))
        label = node("Realized PnL")
        group = node("", [node("\u221211,300.00")])
        parent = node("", [label, group, node("Unrealized PnL"), node("", [node("50.00")])])
        label.GetParentControl = lambda: parent
        reader = WindowsAccountReader(CFG)
        self.assertEqual(reader._realized_value(label), "-11300")

    def test_entry_hit_test_failure_preserves_quantity_and_blocks_clicks(self):
        reader = WindowsAccountReader(CFG)
        reader.chart = object()
        reader._quote_widget = Mock(return_value=(1, (10, 20), (30, 20)))
        reader._surface_clear = Mock(side_effect=RuntimeError("transient accessibility failure"))
        result = reader._entry_data()
        self.assertEqual(result["quantity"], 1)
        self.assertFalse(result["entry_surface_clear"])
        payload = dict(account="test-account", broker="Paper Trading", page_ok=True,
                       positions_selected=True, table_readable=True, empty_text=EMPTY,
                       position_count=0, position_rows=[], working_orders=0)
        payload.update(result)
        self.assertEqual(parse_view(payload, 100, CFG).position, "flat")

    def test_account_discovery_does_not_require_entry_chart(self):
        from types import SimpleNamespace
        def node(name, kind="GroupControl"):
            return SimpleNamespace(Name=name, ControlTypeName=kind, IsOffscreen=False,
                                   BoundingRectangle=SimpleNamespace(left=-1800, top=800, right=1900, bottom=1000))
        manager = node("Account manager")
        address = node("Address and search bar", "EditControl")
        broker = node("Paper Trading", "ButtonControl")
        window = SimpleNamespace(Name="test TradingView", ClassName="Chrome_WidgetWin_1")
        reader = WindowsAccountReader(CFG)
        reader.auto = SimpleNamespace(GetRootControl=lambda: SimpleNamespace(GetChildren=lambda: [window]))
        reader._walk = Mock(return_value=iter([manager, address, broker]))
        reader._bind()
        self.assertIs(reader.manager, manager)
        self.assertIsNone(reader.chart)

    def test_missing_quote_widget_does_not_change_open_account_evidence(self):
        reader = WindowsAccountReader(CFG)
        reader.chart = object()
        reader._quote_widget = Mock(side_effect=RuntimeError("header missing"))
        result = reader._entry_data()
        payload = dict(account="test-account", broker="Paper Trading", page_ok=True,
                       positions_selected=True, table_readable=True, empty_text="",
                       position_count=1, position_rows=["NQZ6 Long 1 27000 Close"], working_orders=2)
        payload.update(result)
        self.assertEqual(parse_view(payload, 100, CFG).position, "open")
        self.assertFalse(result["entry_surface_clear"])


class ViewParsingTests(unittest.TestCase):
    def test_native_paper_and_tradovate_row_examples_are_open(self):
        self.assertTrue(is_position_row("CME_MINI:NQ1! Long 1 27,109.25 0.00 Close"))
        self.assertTrue(is_position_row("NQZ6 Short 1 27044.5 3267315 Close"))
        self.assertTrue(is_position_row("ESZ6 Buy 2 6000.00 Close"))

    def test_empty_invitation_header_and_zero_quantity_are_not_positions(self):
        self.assertFalse(is_position_row("Make a trade to get started Buy or sell — choose your pill. Sell or Buy"))
        self.assertFalse(is_position_row("Symbol Side Qty Avg Fill Price Column setup"))
        self.assertFalse(is_position_row("NQZ6 Short 0 27044.5 Close"))

    def test_tab_badges_support_child_text_and_reject_conflicts(self):
        self.assertEqual(badge_count("Orders", ["Orders", "3"], "Orders", absent=0), 3)
        self.assertEqual(badge_count("Orders 3", ["3"], "Orders", absent=0), 3)
        self.assertEqual(badge_count("Orders", ["Orders"], "Orders", absent=0), 0)
        self.assertIsNone(badge_count("Orders 2", ["3"], "Orders", absent=0))

    def data(self, **changes):
        data = dict(account="test-account", broker="Paper Trading", foreground=True,
                    page_ok=True, quantity=1, working_orders=0, positions_selected=True,
                    table_readable=True, position_count=None, empty_text=EMPTY, position_rows=[])
        data.update(changes)
        return parse_view(data, 100, CFG)

    def test_positive_empty_is_flat(self):
        self.assertEqual(self.data().position, "flat")

    def test_hidden_missing_or_other_tab_is_unknown(self):
        for changes in ({"positions_selected": False}, {"table_readable": False}, {"empty_text": ""}):
            self.assertEqual(self.data(**changes).position, "unknown")

    def test_open_and_conflicting_evidence(self):
        self.assertEqual(self.data(empty_text="", position_count=1).position, "open")
        self.assertEqual(self.data(empty_text="", position_rows=["NQ Long 1 27000"]).position, "open")
        self.assertEqual(self.data(position_count=1).position, "unknown")

    def test_wrong_account_and_missing_binding(self):
        self.assertEqual(self.data(account="other").position, "unknown")
        self.assertTrue(configuration_errors({}))

    def test_empty_trade_invitation_is_not_an_open_position(self):
        self.assertEqual(self.data(empty_text="Make a trade to get started Buy or sell — choose your pill.").position, "flat")
        self.assertEqual(self.data(empty_text="Make a trade to get started Buy or sell — choose your pill. Sell or Buy").position, "flat")


class ClippedTableTests(unittest.TestCase):
    def node(self, name="", kind="DataGridControl", children=(), area=True):
        from types import SimpleNamespace
        return SimpleNamespace(Name=name, ControlTypeName=kind, IsOffscreen=True,
                               BoundingRectangle=SimpleNamespace(left=0, top=1200,
                                                                 right=800 if area else 0,
                                                                 bottom=1300 if area else 1200),
                               GetChildren=lambda: list(children))

    def view(self, table, selected=True):
        reader = WindowsAccountReader(CFG)
        data = dict(account="test-account", broker="Paper Trading", foreground=True,
                    page_ok=True, quantity=1, working_orders=0,
                    positions_selected=selected, position_count=None)
        data.update(reader._table_data(table))
        return parse_view(data, 100, CFG)

    def test_clipped_empty_grid_is_read_as_flat(self):
        table = self.node(children=[self.node(EMPTY, "TextControl")])
        self.assertEqual(self.view(table).position, "flat")

    def test_clipped_position_grid_is_read_as_open(self):
        table = self.node(children=[self.node("CME_MINI:NQ1! Short 1 27,100.00 Close", "DataItemControl")])
        self.assertEqual(self.view(table).position, "open")

    def test_css_hidden_grid_cannot_authorize_entry(self):
        table = self.node(children=[self.node(EMPTY, "TextControl")], area=False)
        self.assertEqual(self.view(table).position, "unknown")

    def test_other_selected_tab_cannot_authorize_entry(self):
        table = self.node(children=[self.node(EMPTY, "TextControl")])
        self.assertEqual(self.view(table, selected=False).position, "unknown")

    def test_hidden_empty_placeholder_cannot_conflict_with_real_position(self):
        table = self.node(children=[self.node(EMPTY, "TextControl", area=False),
                                    self.node("NQZ6 Long 1 27000 Close", "DataItemControl")])
        self.assertEqual(self.view(table).position, "open")

    def test_conflicting_grid_evidence_blocks(self):
        table = self.node(children=[self.node(EMPTY, "TextControl"),
                                    self.node("NQZ6 Long 1 27000 Close", "DataItemControl")])
        self.assertEqual(self.view(table).position, "unknown")


class DualMonitorSurfaceTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.handle = 66548
        self.chart = SimpleNamespace(BoundingRectangle=SimpleNamespace(left=-1900, top=200, right=1900, bottom=800))
        self.reader = WindowsAccountReader(dict(region=dict(left=1683, top=288, width=74, height=564)))
        self.reader.chart = self.chart
        self.reader.window = SimpleNamespace(NativeWindowHandle=self.handle)
        def control():
            return SimpleNamespace(IsEnabled=True, IsOffscreen=False,
                                   BoundingRectangle=SimpleNamespace(left=0, top=0, right=120, bottom=34))
        self.buy, self.sell, self.qty = control(), control(), control()
        self.reader._quote_surface = Mock(return_value=(True, "card verified"))
        self.reader.quote_controls = [((1064, 188), self.buy), ((904, 188), self.sell), ((984, 188), self.qty)]
        self.covered = set()
        self.modal_at = set()
        controls = dict(self.reader.quote_controls)
        self.reader.auto = SimpleNamespace(
            WindowFromPoint=lambda x, y: 99 if (x, y) in self.covered else 77,
            GetAncestor=lambda handle, flag: 999 if handle == 99 else self.handle,
            ControlFromPoint=lambda x, y: SimpleNamespace(GetParentControl=lambda: None) if (x, y) in self.modal_at else controls.get((x, y), self.chart),
            ControlsAreSame=lambda a, b: a is b)

    def test_spanning_two_monitors_is_supported_and_region_is_clipped(self):
        clear, bounds, detail = self.reader._surface_clear()
        self.assertTrue(clear)
        self.assertEqual(bounds[0], -1900)
        self.assertEqual(intersect_scan(self.reader.cfg["region"], bounds), (1683, 288, 1757, 800))

    def test_other_window_covering_an_order_button_blocks(self):
        self.covered.add((1064, 188))
        self.assertFalse(self.reader._surface_clear()[0])

    def test_other_window_covering_scan_area_blocks(self):
        self.covered.add((1683, 288))
        self.assertFalse(self.reader._surface_clear()[0])

    def test_modal_inside_same_chrome_window_still_blocks(self):
        self.reader._quote_surface.return_value = (False, "dimmed or covered quote pixels")
        self.assertFalse(self.reader._surface_clear()[0])

    def test_chrome_element_from_point_is_never_used(self):
        self.reader.auto.ControlFromPoint = Mock(side_effect=AssertionError("Unstable UIA point lookup"))
        self.assertTrue(self.reader._surface_clear()[0])
        self.reader.auto.ControlFromPoint.assert_not_called()

    def test_missing_positive_pixel_evidence_never_grants_entry(self):
        self.reader._quote_surface.return_value = (False, "unrecognized quote appearance")
        self.assertFalse(self.reader._surface_clear()[0])

    def test_window_covering_button_during_pixel_capture_blocks(self):
        def cover(control, point, side):
            self.covered.add(point)
            return True, "previous pixels clear"
        self.reader._quote_surface.side_effect = cover
        self.assertFalse(self.reader._surface_clear()[0])

    def test_disabled_quote_control_cannot_authorize_entry(self):
        self.buy.IsEnabled = False
        self.assertFalse(self.reader._surface_clear()[0])

    def test_missing_scan_overlap_and_negative_coordinates(self):
        self.reader.cfg["region"]["left"] = 3000
        self.assertFalse(self.reader._surface_clear()[0])
        points = region_probes((-1800, 200, -1700, 400))
        self.assertTrue(all(-1800 <= x < -1700 and 200 <= y < 400 for x, y in points))


class EntryIntegrationTests(unittest.TestCase):
    setUp = ticket_tests.TicketGuardTests.setUp
    prepare_run_frames = ticket_tests.TicketGuardTests.prepare_run_frames
    pixel = ticket_tests.TicketGuardTests.pixel
    click = ticket_tests.TicketGuardTests.click
    assert_no_confirmation = ticket_tests.TicketGuardTests.assert_no_confirmation

    def test_guard_required(self):
        self.assertFalse(self.bot.enter(self.cfg, "buy"))
        self.bot.pyautogui.click.assert_not_called()

    def test_guard_blocks_both_directions(self):
        self.guard.claim.return_value = False
        for side in ("buy", "sell"):
            self.assertIsNone(self.bot.enter(self.cfg, side, guard=self.guard))
        self.bot.pyautogui.click.assert_not_called()

    def test_reservation_before_click(self):
        self.cfg.update(use_ticket=False)
        events = []
        self.guard.claim.side_effect = lambda: events.append("reserve") or True
        self.bot.pyautogui.click.side_effect = lambda *args: events.append("click")
        self.assertTrue(self.bot.enter(self.cfg, "buy", guard=self.guard))
        self.assertEqual(events, ["reserve", "click"])

    def test_ticket_rechecks_position(self):
        self.guard.still_flat.return_value = False
        self.assertFalse(self.bot.enter(self.cfg, "buy", guard=self.guard))
        self.assert_no_confirmation()

    def test_run_discards_when_open(self):
        self.guard.observe.return_value = False
        self.guard.reason = "BLOCKED: open position"
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.enter = Mock()
        self.prepare_run_frames()
        self.bot.run()
        self.bot.enter.assert_not_called()
        self.bot.pyautogui.click.assert_not_called()

    def test_full_test_loop_with_real_gate_blocks_buy_and_sell_from_clipped_position(self):
        from contextlib import contextmanager
        from signal_tracker import Marker
        import numpy as np
        table = ClippedTableTests().node(children=[ClippedTableTests().node(
            "CME_MINI:NQ1! Short 1 27,100.00 Close", "DataItemControl")])
        table_data = WindowsAccountReader(self.cfg)._table_data(table)
        outer = self

        class Monitor:
            @property
            def latest(self):
                payload = dict(account="test", broker="Paper Trading", foreground=True,
                               page_ok=True, quantity=1, working_orders=0,
                               positions_selected=True, position_count=None)
                payload.update(table_data)
                return parse_view(payload, outer.bot.time.now, outer.cfg)

        gate = PositionGate(self.cfg, Monitor(), None, dry=True, clock=lambda: self.bot.time.now)

        @contextmanager
        def session(cfg, dry):
            yield gate

        self.bot.position_session = session
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.keyboard.is_pressed.side_effect = [False, False, False, True]
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        capture.return_value.grab = Mock(return_value=np.zeros((120, 80, 4), dtype=np.uint8))
        self.bot.find_markers = Mock(return_value=[])
        buy = Marker("buy", 20, 30, 15, 20, 26, 31, 60)
        sell = Marker("sell", 20, 30, 15, 20, 26, 31, 60)
        self.bot.choose_signal = Mock(side_effect=[(buy, []), (sell, []), (buy, [])])
        self.bot.run(dry=True)
        events = self.journal.record.call_args_list
        skipped = [call.kwargs["reason"] for call in events if call.args[0] == "signal_skipped"]
        self.assertEqual(len(skipped), 3)
        self.assertTrue(all("open position" in reason for reason in skipped))
        self.assertFalse(any(call.args[0] in ("signal_selected", "test_signal", "click_attempt") for call in events))
        self.assertEqual(gate.phase, "open")
        self.bot.pyautogui.click.assert_not_called()

    def test_full_test_loop_refreshes_negative_visual_snapshot_for_new_signal(self):
        from contextlib import contextmanager
        from signal_tracker import Marker
        import numpy as np
        outer = self
        class Monitor:
            calls = 0
            def view(self, clear):
                outer.bot.time.now += .15
                return AccountView(outer.bot.time.now, "test", "Paper Trading", "flat",
                                   1, 0, True, True, "fixture", (10, 20), (30, 20), clear)
            @property
            def latest(self):
                return self.view(False)
            def fresh(self):
                self.calls += 1
                return self.view(self.calls >= 2)
        monitor = Monitor()
        gate = PositionGate(self.cfg, monitor, None, dry=True, clock=lambda: self.bot.time.now)
        gate.observe(monitor.view(True))
        gate.observe(monitor.view(True))
        @contextmanager
        def session(cfg, dry):
            yield gate
        self.bot.position_session = session
        self.bot.load = Mock(return_value=self.cfg)
        self.cfg["use_ticket"] = False
        self.bot.keyboard.is_pressed.side_effect = [False, False, True]
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        capture.return_value.grab = Mock(return_value=np.zeros((120, 80, 4), dtype=np.uint8))
        buy = Marker("buy", 20, 30, 15, 20, 26, 31, 60)
        self.bot.find_markers = Mock(side_effect=[[], [], [buy], []])
        self.bot.choose_signal = Mock(side_effect=[(None, []), (buy, [])])
        self.bot.run(dry=True)
        events = self.journal.record.call_args_list
        self.assertTrue(any(c.args[0] == "test_signal" for c in events))
        rechecks = [c for c in events if c.args[0] == "signal_guard_recheck"]
        self.assertEqual(len(rechecks), 1)
        self.assertTrue(rechecks[0].kwargs["entry_allowed"])
        self.assertEqual(monitor.calls, 2)
        self.assertEqual(gate.phase, "flat")
        self.bot.pyautogui.click.assert_not_called()

    def test_timed_diagnostic_is_test_only(self):
        self.bot.run = Mock()
        self.bot.main(["test", "--test-seconds", "3", "--strategy", "1sec-after-3bars"])
        self.bot.run.assert_called_once_with(dry=True, strategy_tag="1sec-after-3bars", test_seconds=3.0)
        self.bot.run.reset_mock()
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.bot.main(["run", "--test-seconds", "3"])
        self.bot.run.assert_not_called()

    def test_timed_test_stops_without_clicks(self):
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.keyboard.is_pressed.return_value = False
        self.bot.enter = Mock()
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        import numpy as np
        capture.return_value.grab = Mock(return_value=np.zeros((120, 80, 4), dtype=np.uint8))
        self.bot.find_markers = Mock(return_value=[])
        self.bot.run(dry=True, test_seconds=0.05)
        self.bot.enter.assert_not_called()
        self.bot.pyautogui.click.assert_not_called()
