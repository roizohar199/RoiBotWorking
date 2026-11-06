"""Adversarial replay of the actual Pine tracker control-flow block.

Fixtures provide the stable M1/M5 packet; this does not emulate Pine TA or ticks.
"""
import re
import unittest

from test_mamba_policy import SOURCE, PINE


TRACKER = PINE.parse(SOURCE.split('// BEGIN TRACKER:', 1)[1].split('\n', 1)[1].split('// END TRACKER', 1)[0])


class Replay:
    def __init__(self, **changes):
        self.state = dict(geometry_ready=True, candidate_long_pattern=True,
                          candidate_short_pattern=False, candidate_long_kind=1,
                          candidate_short_kind=1, candidate_high=115.0, candidate_low=105.0,
                          candidate_prev_high=110.0, candidate_prev_low=100.0,
                          candidate_high_time=600000, candidate_low_time=720000,
                          candidate_prev_high_time=300000, candidate_prev_low_time=420000,
                          setup_minute=900000, previous_close=113.0, close=114.0,
                          bias=1, gap=0.25, setup_age=30, timing='Confirmed 1M close')
        for match in re.finditer(r'^varip (?:int|float|bool) (\w+) = (.+)$', SOURCE, re.M):
            self.state[match[1]] = PINE.evaluate(match[2], self.state)
        self.state.update(changes)

    def step(self, **changes):
        e = self.state
        e.update(changes)
        e['fresh_buy'] = e['fresh_sell'] = False
        e['new_source_minute'] = PINE.evaluate(re.search(r'^bool new_source_minute = (.+)$', SOURCE, re.M)[1], e)
        e['candidate_long_level'] = PINE.call('f_setup_level', 1, e['candidate_long_kind'], e['candidate_high'],
                                            e['candidate_prev_high'], e['candidate_prev_high_time'],
                                            e['candidate_high'], e['candidate_high_time'], e['setup_minute'], e['gap'])
        e['candidate_short_level'] = PINE.call('f_setup_level', -1, e['candidate_short_kind'], e['candidate_low'],
                                             e['candidate_prev_low'], e['candidate_prev_low_time'],
                                             e['candidate_low'], e['candidate_low_time'], e['setup_minute'], e['gap'])
        PINE.execute(TRACKER, e)
        return dict(e)


class LockedSetupTests(unittest.TestCase):
    def test_horizontal_target_and_protective_swing_ignore_new_same_side_pivots(self):
        r = Replay()
        original = r.step()
        result = r.step(setup_minute=960000, previous_close=114,
                        candidate_high=120, candidate_low=108,
                        candidate_high_time=780000, candidate_low_time=840000)
        for field in ('tracked_id', 'tracked_origin', 'tracked_horizontal', 'tracked_protective',
                      'tracked_y0', 'tracked_y1', 'tracked_t0', 'tracked_t1'):
            self.assertEqual(result[field], original[field], field)
        self.assertEqual(result['tracked_level'], 115.25)

    def test_line_advances_with_time_using_frozen_anchors(self):
        r = Replay(candidate_long_kind=2, candidate_prev_high=120, previous_close=109, close=109)
        first = r.step()
        second = r.step(setup_minute=960000, previous_close=108.75,
                        candidate_prev_high=130, candidate_high=118,
                        candidate_high_time=780000, candidate_low_time=840000)
        self.assertEqual(first['tracked_level'], 110.25)
        self.assertEqual(second['tracked_level'], 109.25)
        self.assertEqual((second['tracked_y0'], second['tracked_y1']), (120, 115))

    def test_short_line_is_symmetric(self):
        r = Replay(candidate_long_pattern=False, candidate_short_pattern=True,
                   candidate_short_kind=2, candidate_high=110, candidate_low=100,
                   candidate_prev_low=95, candidate_high_time=720000, candidate_low_time=600000,
                   candidate_prev_low_time=300000, bias=-1, previous_close=106, close=106)
        first = r.step()
        second = r.step(setup_minute=960000, previous_close=106,
                        candidate_low=99, candidate_prev_low=90, candidate_low_time=840000)
        self.assertEqual(first['tracked_level'], 104.75)
        self.assertEqual(second['tracked_level'], 105.75)
        self.assertEqual((second['tracked_y0'], second['tracked_y1']), (95, 100))

    def test_confirmed_entry_uses_locked_price_not_latest_candidate(self):
        r = Replay()
        r.step()
        result = r.step(setup_minute=960000, previous_close=116,
                        candidate_high=120, candidate_low=108,
                        candidate_high_time=780000, candidate_low_time=840000)
        self.assertTrue(result['fresh_buy'])
        self.assertFalse(result['fresh_sell'])

    def test_confirmation_uses_bias_of_completed_minute_not_new_bias(self):
        r = Replay()
        r.step()
        self.assertTrue(r.step(setup_minute=960000, previous_close=116, bias=-1)['fresh_buy'])
        r = Replay(bias=0)
        r.step()
        self.assertFalse(r.step(setup_minute=960000, previous_close=116, bias=1)['fresh_buy'])

    def test_neutral_context_blocks_but_does_not_move_lock(self):
        r = Replay(bias=0)
        first = r.step()
        second = r.step(setup_minute=960000, previous_close=114, candidate_high=125, bias=-1)
        self.assertEqual(second['tracked_side'], 1)
        self.assertEqual(second['tracked_horizontal'], first['tracked_horizontal'])
        self.assertFalse(second['fresh_buy'])

    def test_invalidation_uses_frozen_pullback_and_prevents_same_id_rearm(self):
        r = Replay()
        r.step()
        result = r.step(setup_minute=960000, previous_close=105)
        self.assertEqual((result['tracked_side'], result['exit_code']), (0, 3))
        result = r.step(setup_minute=1020000, previous_close=113)
        self.assertEqual(result['tracked_side'], 0)

    def test_same_source_minute_does_not_resolve_twice_on_seconds_chart(self):
        r = Replay()
        r.step()
        self.assertTrue(r.step(setup_minute=960000, previous_close=116)['fresh_buy'])
        for _ in range(20):
            self.assertFalse(r.step()['fresh_buy'])

    def test_live_entry_releases_once_and_cannot_rearm_same_pullback(self):
        r = Replay(timing='Live breakout')
        r.step()
        result = r.step(close=116)
        self.assertTrue(result['fresh_buy'])
        self.assertEqual(result['tracked_side'], 0)
        self.assertFalse(r.step(close=117)['fresh_buy'])
        self.assertEqual(r.step(setup_minute=960000, previous_close=113)['tracked_side'], 0)

    def test_live_stop_before_break_cancels_setup(self):
        r = Replay(timing='Live breakout')
        r.step()
        self.assertEqual(r.step(close=104)['exit_code'], 3)
        self.assertFalse(r.step(close=116)['fresh_buy'])

    def test_live_wrong_bias_cannot_fire(self):
        r = Replay(timing='Live breakout', bias=0)
        r.step()
        self.assertFalse(r.step(close=116)['fresh_buy'])

    def test_newer_opposite_structure_replaces_lock_explicitly(self):
        r = Replay()
        r.step()
        result = r.step(setup_minute=960000, previous_close=110,
                        candidate_long_pattern=False, candidate_short_pattern=True,
                        candidate_low=100, candidate_high=112,
                        candidate_low_time=780000, candidate_high_time=840000)
        self.assertEqual(result['tracked_side'], -1)
        self.assertEqual(result['tracked_id'], 840000)
        self.assertEqual(result['tracked_horizontal'], 100)

    def test_expiry_uses_original_geometry_not_new_same_side_pivots(self):
        r = Replay()
        r.step()
        for minute in range(960000, 2460001, 60000):
            result = r.step(setup_minute=minute, previous_close=113,
                            candidate_high=120, candidate_high_time=minute - 120000,
                            candidate_low_time=minute - 60000,
                            candidate_long_pattern=False)
        self.assertEqual((result['tracked_side'], result['exit_code']), (0, 1))

    def test_source_gap_cannot_deliver_a_delayed_old_signal(self):
        r = Replay()
        r.step()
        result = r.step(setup_minute=1080000, previous_close=116)
        self.assertFalse(result['fresh_buy'])
        self.assertEqual(result['exit_code'], 7)

    def test_combined_mode_keeps_horizontal_barrier_and_original_slope(self):
        self.assertEqual(PINE.call('f_setup_level', 1, 3, 115, 120, 300000, 115, 600000, 900000, .25), 115.25)
        self.assertEqual(PINE.call('f_setup_level', -1, 3, 100, 95, 300000, 100, 600000, 900000, .25), 99.75)


if __name__ == '__main__':
    unittest.main()
