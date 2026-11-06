"""Replay the actual Pine queue; no UI clicks, Pine emulation, or P&L claims."""
from pathlib import Path
import unittest

from pine_scalar import ScalarPine


SOURCE = (Path(__file__).resolve().parents[1] / "אינדיקטוריםTRADING VIEW" /
          "instant_breakout_rsi_vwap_1sec_3bars.pine").read_text(encoding="utf-8")


class Queue:
    def __init__(self):
        self.pine = ScalarPine(SOURCE)
        block = SOURCE.split("// BEGIN THREE BAR QUEUE:", 1)[1].split("\n", 1)[1]
        block = block.split("// END THREE BAR QUEUE", 1)[0]
        self.nodes = self.pine.parse(block)
        self.env = dict(processed_bar=None, pending_side=0, pending_origin=None,
                        entry_side=0, delivered_origin=None, completed_bars=0,
                        WAIT_BARS=3)

    def update(self, index, raw=0):
        self.env.update(bar_index=index, raw_side=raw, fresh_buy=False, fresh_sell=False,
                        source_bar_ready=self.env["processed_bar"] != index)
        self.pine.execute(self.nodes, self.env)
        return self.env["fresh_buy"], self.env["fresh_sell"]


class ThreeBarEntryTests(unittest.TestCase):
    def test_exactly_three_following_closed_candles_both_directions(self):
        for side in (1, -1):
            with self.subTest(side=side):
                q = Queue()
                # Source on bar 10 is first known at bar 11 opening.
                self.assertEqual(q.update(11, side), (False, False))
                self.assertEqual(q.env["pending_origin"], 10)
                self.assertEqual(q.env["completed_bars"], 0)
                self.assertEqual(q.update(12), (False, False))
                self.assertEqual(q.env["completed_bars"], 1)
                self.assertEqual(q.update(13), (False, False))
                self.assertEqual(q.env["completed_bars"], 2)
                self.assertEqual(q.update(14), (side == 1, side == -1))
                self.assertEqual(q.env["completed_bars"], 3)
                self.assertEqual(q.env["delivered_origin"], 10)

    def test_tick_repeats_do_not_count_or_duplicate_entry(self):
        q = Queue()
        q.update(11, 1)
        for _ in range(20):
            self.assertEqual(q.update(11, 1), (False, False))
        self.assertEqual(q.env["completed_bars"], 0)
        q.update(12)
        q.update(13)
        self.assertEqual(q.update(14), (True, False))
        for _ in range(20):
            self.assertEqual(q.update(14), (False, False))
            self.assertEqual(q.env["entry_side"], 1)
        self.assertEqual(q.update(15), (False, False))
        self.assertEqual(q.env["entry_side"], 0)

    def test_same_direction_does_not_reset_wait(self):
        q = Queue()
        q.update(11, 1)
        q.update(12, 1)
        q.update(13, 1)
        self.assertEqual(q.update(14, 1), (True, False))
        self.assertEqual(q.env["delivered_origin"], 10)

    def test_opposite_signal_replaces_pending_and_restarts_count(self):
        q = Queue()
        q.update(11, 1)
        q.update(12)
        q.update(13, -1)
        self.assertEqual(q.env["pending_origin"], 12)
        self.assertEqual(q.env["completed_bars"], 0)
        self.assertEqual(q.update(14), (False, False))
        self.assertEqual(q.update(15), (False, False))
        self.assertEqual(q.update(16), (False, True))

    def test_opposite_at_delivery_boundary_cancels_old_entry(self):
        q = Queue()
        q.update(11, 1)
        q.update(12)
        q.update(13)
        self.assertEqual(q.update(14, -1), (False, False))
        self.assertEqual(q.env["pending_side"], -1)
        self.assertEqual(q.env["completed_bars"], 0)

    def test_elapsed_time_without_new_bars_cannot_advance_count(self):
        q = Queue()
        q.update(11, -1)
        # Arbitrarily many observations of the same bar, regardless of time.
        for _ in range(100):
            self.assertEqual(q.update(11), (False, False))
        self.assertEqual(q.env["completed_bars"], 0)
        self.assertEqual(q.update(12), (False, False))
        self.assertEqual(q.env["completed_bars"], 1)

    def test_missing_updates_cancel_old_entry_instead_of_late_delivery(self):
        q = Queue()
        q.update(11, 1)
        self.assertEqual(q.update(16), (False, False))
        self.assertEqual(q.env["pending_side"], 0)

    def test_new_source_signal_after_gap_can_start_fresh_count(self):
        q = Queue()
        q.update(11, 1)
        self.assertEqual(q.update(16, -1), (False, False))
        self.assertEqual(q.env["pending_origin"], 15)
        q.update(17)
        q.update(18)
        self.assertEqual(q.update(19), (False, True))

    def test_source_contract_and_only_delayed_triangles(self):
        self.assertIn("not timeframe.isseconds or timeframe.multiplier != 1", SOURCE)
        self.assertIn("not chart.is_standard", SOURCE)
        self.assertIn("bullish_breakout[1] ? 1 : bearish_breakout[1] ? -1 : 0", SOURCE)
        triangles = [line for line in SOURCE.splitlines() if line.startswith("plotshape(")]
        self.assertEqual(len(triangles), 2)
        self.assertTrue(all("entry_side" in line for line in triangles))
        self.assertNotIn("timenow", SOURCE)
        self.assertIn("calc_on_every_tick=true", SOURCE)


if __name__ == "__main__":
    unittest.main()
