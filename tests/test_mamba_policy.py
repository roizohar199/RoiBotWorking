"""Check the actual Pine entry gates against adversarial synthetic setups.

This is a logic/contract check, not a Pine compiler, tick replay or P&L test.
"""
import re
import unittest
from pathlib import Path
from pine_scalar import ScalarPine

SOURCE = (Path(__file__).resolve().parents[1] / "אינדיקטוריםTRADING VIEW" /
          "mamba_5m_1m_signals.pine").read_text(encoding="utf-8")
PINE = ScalarPine(SOURCE)


def pine_eval(expression, values):
    """Evaluate the scalar Pine gate/diagnostic expressions from the source."""
    depth = 0
    quote = False
    question = None
    nested = 0
    for index, char in enumerate(expression):
        if char == '"':
            quote = not quote
        if quote:
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif depth == 0 and char == '?':
            if question is None:
                question = index
            else:
                nested += 1
        elif depth == 0 and char == ':' and question is not None:
            if nested:
                nested -= 1
            else:
                condition = pine_eval(expression[:question].strip(), values)
                branch = expression[question + 1:index] if condition else expression[index + 1:]
                return pine_eval(branch.strip(), values)
    expression = expression.replace("math.min", "min").replace("math.max", "max")
    expression = expression.replace("na(", "is_na(").replace("close[1]", "previous_close")
    return eval(expression, {"__builtins__": {}, "min": min, "max": max,
                             "is_na": lambda v: v is None}, values)


def gate(name, values):
    expression = re.search(rf"^    (?:bool|int|string) {name} = (.+)$", SOURCE, re.M).group(1)
    return pine_eval(expression, values)


def setup(**changes):
    values = dict(prev_high=110.0, last_high=115.0, prev_low=100.0,
                  last_low=105.0, high_time=600000, low_time=720000,
                  prev_high_time=300000, prev_low_time=420000,
                  time=900000, setup_age=30, bias=1,
                  close=116.0, previous_close=113.0,
                  used_long=None, used_short=None,
                  entry_rule="Structure or trendline",
                  long_level=115.25, short_level=104.75)
    values.update(changes)
    for name in ("ready", "higher_low", "lower_high", "long_structure",
                 "short_structure", "falling_line", "rising_line",
                 "long_line", "short_line", "allow_structure", "allow_line",
                 "long_pattern", "short_pattern"):
        values[name] = gate(name, values)
    # Candidate gates only; source-code lifecycle replays are in test_mamba_tracker.
    values['long_block'] = PINE.call('f_candidate_block', values['ready'], values['long_pattern'],
                                    values['previous_close'], values['last_low'], values['long_level'],
                                    values['low_time'], values['used_long'], 1)
    values['short_block'] = PINE.call('f_candidate_block', values['ready'], values['short_pattern'],
                                     values['previous_close'], values['last_high'], values['short_level'],
                                     values['high_time'], values['used_short'], -1)
    values['buy'] = values['bias'] == 1 and values['long_block'] == 0 and values['close'] > values['long_level']
    values['sell'] = values['bias'] == -1 and values['short_block'] == 0 and values['close'] < values['short_level']
    return values


class MambaPolicyTests(unittest.TestCase):
    def test_long_requires_higher_high_then_higher_low_then_new_break(self):
        result = setup()
        self.assertTrue(result["long_structure"] and result["buy"])
        self.assertFalse(result["sell"])

    def test_mirrored_short(self):
        result = setup(prev_high=115, last_high=110, prev_low=105,
                       last_low=100, high_time=720000, low_time=600000,
                       bias=-1, close=99, previous_close=102,
                       long_level=110.25, short_level=99.75)
        self.assertTrue(result["short_structure"] and result["sell"])
        self.assertFalse(result["buy"])

    def test_wrong_or_neutral_5m_bias_blocks_buy(self):
        for bias in (-1, 0):
            self.assertFalse(setup(bias=bias)["buy"])

    def test_no_signal_before_or_exactly_at_buffer(self):
        for price in (114.0, 115.0, 115.25):
            self.assertFalse(setup(close=price)["buy"])

    def test_already_above_level_does_not_create_late_entry(self):
        self.assertFalse(setup(previous_close=116, close=117)["buy"])

    def test_consumed_pullback_cannot_fire_twice(self):
        self.assertFalse(setup(used_long=720000)["buy"])
        self.assertTrue(setup(used_long=420000)["buy"])

    def test_old_levels_and_incomplete_warmup_block_signal(self):
        self.assertFalse(setup(time=3000000)["buy"])
        self.assertFalse(setup(prev_high=None)["buy"])

    def test_pivot_order_matters(self):
        self.assertFalse(setup(high_time=800000, low_time=720000)["buy"])

    def test_trendline_path_is_explicit_and_optional(self):
        values = dict(prev_high=120, last_high=115,
                      long_level=113, previous_close=112, close=114)
        self.assertTrue(setup(**values)["long_line"])
        self.assertTrue(setup(**values)["buy"])
        self.assertFalse(setup(**values, entry_rule="Structure only")["buy"])

    def test_combined_line_and_level_does_not_accept_structure_only(self):
        self.assertFalse(setup(entry_rule="Trendline + horizontal level")["buy"])

    def test_touch_needs_return_outside_zone_and_spacing(self):
        values = dict(support=100, sw=1, low=100, high=103, open=102, close=102,
                      support_last_touch=10, bar_index=12, touch_gap=2)
        self.assertTrue(gate("support_rejection", values))
        for changes in ({"close": 100.5}, {"bar_index": 11}, {"low": 102}, {"open": 98}):
            self.assertFalse(gate("support_rejection", values | changes))

    def test_wick_sweep_below_support_and_recovery_is_allowed(self):
        values = dict(support=100, sw=1, low=97, high=104, open=102, close=103,
                      support_last_touch=None, bar_index=12, touch_gap=1)
        self.assertTrue(gate("support_rejection", values))
        for changes in ({"close": 98}, {"close": 101}, {"open": 98}, {"low": 102}):
            self.assertFalse(gate("support_rejection", values | changes))

    def test_wick_sweep_above_resistance_is_symmetric(self):
        values = dict(resistance=100, rw=1, high=103, low=96, open=98, close=97,
                      resistance_last_touch=None, bar_index=12, touch_gap=1)
        self.assertTrue(gate("resistance_rejection", values))
        for changes in ({"close": 102}, {"close": 99}, {"open": 102}, {"high": 98}):
            self.assertFalse(gate("resistance_rejection", values | changes))

    def test_first_confirmed_seed_can_activate_bias_without_second_retest(self):
        values = dict(support_rejection=False, support_seed_rejection=True,
                      support_touches=1, min_touches=1)
        self.assertTrue(gate("bullish_reaction", values))
        self.assertFalse(gate("bullish_reaction", values | {"min_touches": 2}))
        self.assertFalse(gate("bullish_reaction", values | {"support_seed_rejection": False}))
        mirrored = dict(resistance_rejection=False, resistance_seed_rejection=True,
                        resistance_touches=1, min_touches=1)
        self.assertTrue(gate("bearish_reaction", mirrored))

    def test_structural_diagnostic_is_independent_of_five_minute_bias(self):
        result = setup(bias=0)
        self.assertTrue(result["long_pattern"])
        self.assertEqual(result["long_block"], 0)
        self.assertFalse(result["buy"])

    def test_diagnostics_distinguish_missing_consumed_and_late_setups(self):
        self.assertEqual(setup(prev_high=None)["long_block"], 1)
        self.assertEqual(setup(high_time=800000)["long_block"], 2)
        self.assertEqual(setup(previous_close=104)["long_block"], 3)
        self.assertEqual(setup(used_long=720000)["long_block"], 4)
        self.assertEqual(setup(previous_close=116)["long_block"], 5)

    def test_default_rejection_count_and_confirmation_delay_are_explicit(self):
        self.assertRegex(SOURCE, r'int min_touches = input.int\(1, [^\n]+minval=1')
        self.assertRegex(SOURCE, r'int zone_right = input.int\(1,')
        self.assertIn('ta.pivotlow(low, zone_pivot, zone_right)', SOURCE)
        self.assertIn('support_seed_rejection := support_touches > 0 and close > support + sw', SOURCE)
        self.assertIn('resistance_seed_rejection := resistance_touches > 0 and close < resistance - rw', SOURCE)

    def test_five_minute_exports_are_all_shifted_and_pivots_not_backplotted(self):
        context = SOURCE.split("f_context_closed() =>", 1)[1].split("f_candidates() =>", 1)[0]
        exports = re.search(r"^    \[(.+)\]$", context, re.M).group(1).split(", ")
        self.assertTrue(all(value.endswith("[1]") for value in exports))
        packet = SOURCE.split('f_candidates() =>', 1)[1].split('f_setup_level(', 1)[0]
        self.assertIn('time, close[1], bias', packet)
        self.assertIn('float last_high = h1[1]', packet)
        self.assertIn('float last_low = l1[1]', packet)
        self.assertNotIn("offset=", SOURCE)
        self.assertNotIn("strategy.entry", SOURCE)


if __name__ == "__main__":
    unittest.main()
