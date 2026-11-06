"""Evaluate actual Pine boolean gates against synthetic market features.

This checks policy intent and generated contracts, not Pine compilation or P&L.
"""
import re
import tempfile
import unittest
from pathlib import Path

from build_regime_variants import ROOT, build


class RegimePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "pine_templates" / "regime_5sec.pine.in").read_text(encoding="utf-8")

    def gates(self, **updates):
        env = dict(efficiency=0.6, trend_min_er=0.35, range_max_er=0.25,
                   ema_fast=104.0, ema_slow=100.0, slope_atr=0.2, min_slope_atr=0.1,
                   range_gap_atr=0.5, atr_ref=10.0, atr_ratio=1.0,
                   min_atr_ratio=0.5, max_atr_ratio=2.0, high=108.0, low=101.0,
                   max_bar_atr=2.5, close=106.0, min_stretch_atr=0.5)
        env.update(updates)
        for name in ("trend_up", "trend_down", "range_market", "volatility_ok",
                     "trend_buy_ok", "trend_sell_ok", "range_buy_ok", "range_sell_ok"):
            expression = re.search(rf"^bool {name} = (.+)$", self.source, re.M).group(1)
            env[name] = eval(expression.replace("math.abs", "abs"), {"__builtins__": {}, "abs": abs}, env)
        return env

    def test_rising_trend_only_allows_trend_buy(self):
        result = self.gates()
        self.assertTrue(result["trend_buy_ok"] and result["volatility_ok"])
        self.assertFalse(result["trend_sell_ok"] or result["range_buy_ok"] or result["range_sell_ok"])

    def test_falling_trend_only_allows_trend_sell(self):
        result = self.gates(ema_fast=96.0, close=94.0, slope_atr=-0.2)
        self.assertTrue(result["trend_sell_ok"])
        self.assertFalse(result["trend_buy_ok"] or result["range_buy_ok"] or result["range_sell_ok"])

    def test_range_needs_stretch_in_correct_direction(self):
        for close, buy, sell in [(94.0, True, False), (100.0, False, False), (106.0, False, True)]:
            with self.subTest(close=close):
                result = self.gates(efficiency=0.2, slope_atr=0.0, ema_fast=100.0, close=close)
                self.assertEqual((result["range_buy_ok"], result["range_sell_ok"]), (buy, sell))
                self.assertFalse(result["trend_buy_ok"] or result["trend_sell_ok"])

    def test_transition_and_large_ema_gap_block_range(self):
        self.assertFalse(self.gates(efficiency=0.3)["trend_up"])
        self.assertFalse(self.gates(efficiency=0.3)["range_market"])
        self.assertFalse(self.gates(efficiency=0.1, ema_fast=110.0)["range_market"])

    def test_cold_market_shock_and_outsized_candle_block_entries(self):
        for values in ({"atr_ratio": 0.4}, {"atr_ratio": 2.1}, {"high": 150.0, "low": 100.0}):
            self.assertFalse(self.gates(**values)["volatility_ok"])

    def test_builds_separate_reproducible_variants_and_relative_exits(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = build(folder)
            self.assertEqual(len(paths), 2)
            for index, path in enumerate(paths):
                source = path.read_text(encoding="utf-8")
                self.assertEqual(source, (ROOT / "אינדיקטוריםTRADING VIEW" / path.name).read_text(encoding="utf-8"))
                self.assertNotIn("__", source)
                self.assertIn(f"const bool REVERSED = {'true' if index else 'false'}", source)
                self.assertEqual(source.count("profit=tp_ticks, loss=sl_ticks"), 2)
                self.assertEqual(source.count("qty=1"), 2)
                self.assertNotIn("strategy.close(", source)
                self.assertNotRegex(source, r"strategy.exit\([^\n]*(limit|stop)=")
                self.assertIn("strategy.position_size == 0", source)
                self.assertIn('input.bool(true, "Confirm Signals On Bar Close"', source)
                self.assertIn("timeframe.isseconds and timeframe.multiplier == 5", source)
                self.assertIn("REVERSED ? bearish_breakout : bullish_breakout", source)
                self.assertIn("REVERSED ? range_buy_ok : trend_buy_ok", source)
