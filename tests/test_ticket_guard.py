"""Offline tests: screen capture, mouse, keyboard and time are all replaced."""
import copy
import json
import importlib.util
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import numpy as np


class Clock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def monotonic(self):
        self.now += 0.002
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def strftime(self, _):
        return "TEST"


class TicketGuardTests(unittest.TestCase):
    def setUp(self):
        fake_gui = types.ModuleType("pyautogui")
        fake_keys = types.ModuleType("keyboard")
        fake_mss = types.ModuleType("mss")
        fake_keys.is_pressed = Mock(return_value=False)
        fake_mss.MSS = Mock()
        for name in ("click", "hotkey", "write", "press"):
            setattr(fake_gui, name, Mock())
        spec = importlib.util.spec_from_file_location(
            "triangle_bot_under_test", Path(__file__).resolve().parents[1] / "nq_triangle_bot.py")
        self.bot = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"pyautogui": fake_gui, "keyboard": fake_keys, "mss": fake_mss}):
            spec.loader.exec_module(self.bot)
        self.bot.time = Clock()
        self.bot.log = Mock()
        self.actual_position_session = self.bot.position_session
        self.bot.position_session = MagicMock()
        self.guard = self.bot.position_session.return_value.__enter__.return_value
        self.guard.observe.return_value = True
        self.guard.refresh_for_signal.return_value = False
        self.guard.claim.return_value = True
        self.guard.still_flat.return_value = True
        self.guard.reason = "READY"
        self.guard.monitor.latest.chart_bounds = None
        self.bot.RunJournal = MagicMock()
        self.journal = self.bot.RunJournal.return_value.__enter__.return_value
        self.actual_save_test_observation = self.bot.save_test_observation
        self.bot.save_test_observation = Mock(return_value="offline-evidence.png")
        self.cfg = copy.deepcopy(self.bot.DEFAULTS)
        self.guard.entry_config.side_effect = lambda cfg: cfg
        self.cfg.update(position_guard=dict(account="test", broker="Paper Trading", window_title="test", page_url="https://www.tradingview.com/chart/test/"), region={"left": 0, "top": 0, "width": 30, "height": 600},
                        use_ticket=True, fill_sltp=True,
                        red_rgb=[255, 167, 38], green_rgb=[38, 198, 218],
                        buy_btn=[1, 1], sell_btn=[2, 2], confirm_btn=[100, 100],
                        tp_field=[40, 40], sl_field=[50, 50],
                        tp_toggle=[20, 20], sl_toggle=[30, 30],
                        tp_enabled_rgb=[255, 255, 255], sl_enabled_rgb=[255, 255, 255],
                        confirm_buy_rgb=[41, 98, 255], confirm_sell_rgb=[242, 54, 69],
                        tp_value="20", sl_value="20", ticket_timeout_ms=30)
        self.side = None
        self.tp_on = self.sl_on = True
        self.bot.px = Mock(side_effect=self.pixel)
        fake_gui.click.side_effect = self.click

    def pixel(self, x, y):
        point = [x, y]
        if self.side and point == self.cfg["confirm_btn"]:
            return self.cfg[f"confirm_{self.side}_rgb"]
        if self.side and point == self.cfg["tp_toggle"] and self.tp_on:
            return self.cfg["tp_enabled_rgb"]
        if self.side and point == self.cfg["sl_toggle"] and self.sl_on:
            return self.cfg["sl_enabled_rgb"]
        return [80, 80, 80]

    def click(self, x, y):
        if [x, y] == self.cfg["buy_btn"]:
            self.side = "buy"
        elif [x, y] == self.cfg["sell_btn"]:
            self.side = "sell"

    def assert_no_confirmation(self):
        self.assertNotIn(tuple(self.cfg["confirm_btn"]),
                         [call.args for call in self.bot.pyautogui.click.call_args_list])

    def prepare_run_frames(self):
        # Warm up on an empty chart, then show a persistent new SELL marker.
        from signal_tracker import Marker
        marker = Marker("sell", 20, 30, 15, 20, 26, 31, 60)
        self.bot.find_markers = Mock(side_effect=[[], []] * 100 + [[marker], []] * 2)
        self.bot.keyboard.is_pressed.side_effect = [False] * 102 + [True]
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        capture.return_value.grab = Mock(return_value=np.zeros((1, 1, 4), dtype=np.uint8))
        return capture.return_value

    def test_missing_calibration_never_clicks_chart(self):
        self.cfg["confirm_buy_rgb"] = None
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "sell"))
        self.bot.pyautogui.click.assert_not_called()

    def test_existing_open_ticket_is_not_submitted(self):
        self.side = "buy"
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "sell"))
        self.bot.pyautogui.click.assert_not_called()

    def test_timeout_never_fills_or_confirms(self):
        self.bot.pyautogui.click.side_effect = None
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "sell"))
        self.bot.pyautogui.write.assert_not_called()
        self.assert_no_confirmation()

    def test_disabled_exits_never_fill_or_confirm(self):
        for toggle in ("tp_on", "sl_on"):
            with self.subTest(toggle=toggle):
                self.side = None
                self.tp_on = self.sl_on = True
                setattr(self, toggle, False)
                self.bot.pyautogui.click.reset_mock()
                self.bot.pyautogui.write.reset_mock()
                self.assertFalse(self.bot._enter_unchecked(self.cfg, "sell"))
                self.bot.pyautogui.write.assert_not_called()
                self.assert_no_confirmation()

    def test_wrong_side_never_fills_or_confirms(self):
        self.bot.pyautogui.click.side_effect = lambda *args: setattr(self, "side", "buy")
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "sell"))
        self.bot.pyautogui.write.assert_not_called()
        self.assert_no_confirmation()

    def test_escape_before_opening_does_not_click(self):
        self.bot.keyboard.is_pressed.return_value = True
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "buy"))
        self.bot.pyautogui.click.assert_not_called()

    def test_lost_exit_after_filling_prevents_confirmation(self):
        self.bot.pyautogui.write.side_effect = lambda *args, **kwargs: setattr(self, "sl_on", False)
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "sell"))
        self.assert_no_confirmation()

    def test_valid_buy_and_sell_only_confirm_after_two_fills(self):
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                self.side = None
                events = []
                self.bot.pyautogui.click.side_effect = lambda x, y: (events.append(("click", [x, y])), self.click(x, y))
                self.bot.pyautogui.write.side_effect = lambda value, **kwargs: events.append(("write", value))
                self.assertTrue(self.bot._enter_unchecked(self.cfg, side))
                self.assertEqual(events[-1], ("click", self.cfg["confirm_btn"]))
                self.assertEqual([e for e in events if e[0] == "write"], [("write", "20"), ("write", "20")])

    def test_small_hover_shade_difference_is_accepted(self):
        self.side = "buy"
        original = self.pixel
        self.bot.px.side_effect = lambda x, y: [30, 83, 229] if [x, y] == self.cfg["confirm_btn"] else original(x, y)
        self.assertTrue(self.bot.ticket_matches(self.cfg, "buy"))
        self.assertFalse(self.bot.ticket_matches(self.cfg, "sell"))

    def test_check_ticket_never_clicks_or_writes(self):
        self.side = "sell"
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.check_ticket()
        self.bot.pyautogui.click.assert_not_called()
        self.bot.pyautogui.write.assert_not_called()

    def test_ticket_calibration_preserves_signal_settings(self):
        original = copy.deepcopy(self.cfg)
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.side = "buy"
        points = iter([self.cfg["tp_toggle"], self.cfg["sl_toggle"],
                       self.cfg["tp_field"], self.cfg["sl_field"],
                       self.cfg["confirm_btn"], self.cfg["confirm_btn"]])

        def manual_step(message):
            if "Select SELL" in message:
                self.side = "sell"
            point = next(points)
            return types.SimpleNamespace(x=point[0], y=point[1])

        self.bot.countdown = manual_step
        self.bot.calibrate_ticket()
        self.bot.save.assert_called_once()
        saved = self.bot.save.call_args.args[0]
        for key in ("region", "red_rgb", "green_rgb", "buy_btn", "sell_btn",
                    "tp_value", "sl_value", "cooldown_sec"):
            self.assertEqual(saved[key], original[key])
        self.bot.pyautogui.click.assert_not_called()

    def test_dark_switch_calibration_is_not_saved(self):
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.bot.countdown = Mock(return_value=types.SimpleNamespace(x=20, y=20))
        self.bot.calibrate_ticket()
        self.bot.save.assert_not_called()
        self.bot.pyautogui.click.assert_not_called()

    def test_run_stops_after_first_failed_attempt(self):
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.enter = Mock(return_value=False)
        capture = self.prepare_run_frames()
        self.bot.run()
        self.bot.enter.assert_called_once_with(self.cfg, "sell", signal_seen_at=ANY, guard=self.guard)
        self.assertEqual(capture.grab.call_count, 102)
        self.bot.pyautogui.click.assert_not_called()

    def test_one_click_sends_exactly_one_chart_click_without_ticket_access(self):
        self.cfg.update(use_ticket=False, fill_sltp=False)
        for key in self.bot.ticket_required(self.cfg):
            self.cfg[key] = None
        self.bot.px = Mock(side_effect=AssertionError("one-click must not sample ticket pixels"))
        self.bot.ticket_matches = Mock(side_effect=AssertionError("one-click must not wait for a ticket"))
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                self.bot.pyautogui.click.reset_mock()
                self.assertTrue(self.bot._enter_unchecked(self.cfg, side))
                self.bot.pyautogui.click.assert_called_once_with(*self.cfg[f"{side}_btn"])
        self.bot.pyautogui.write.assert_not_called()
        self.bot.pyautogui.hotkey.assert_not_called()
        self.bot.pyautogui.press.assert_not_called()
        self.bot.px.assert_not_called()

    def test_one_click_escape_prevents_chart_click(self):
        self.cfg.update(use_ticket=False, fill_sltp=False)
        self.bot.keyboard.is_pressed.return_value = True
        self.assertFalse(self.bot._enter_unchecked(self.cfg, "buy"))
        self.bot.pyautogui.click.assert_not_called()

    def test_button_only_calibration_preserves_other_settings(self):
        original = copy.deepcopy(self.cfg)
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.bot.countdown = Mock(side_effect=[types.SimpleNamespace(x=160, y=200),
                                              types.SimpleNamespace(x=45, y=200)])
        self.bot.calibrate_buttons()
        self.bot.save.assert_called_once()
        saved = self.bot.save.call_args.args[0]
        self.assertEqual(saved["buy_btn"], [160, 200])
        self.assertEqual(saved["sell_btn"], [45, 200])
        self.assertEqual({k: v for k, v in saved.items() if k not in ("buy_btn", "sell_btn")},
                         {k: v for k, v in original.items() if k not in ("buy_btn", "sell_btn")})
        self.bot.pyautogui.click.assert_not_called()

    def test_button_calibration_rejects_identical_points(self):
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.bot.countdown = Mock(return_value=types.SimpleNamespace(x=160, y=200))
        self.bot.calibrate_buttons()
        self.bot.save.assert_not_called()

    def test_full_one_click_calibration_skips_order_ticket_steps(self):
        self.cfg.update(use_ticket=False, fill_sltp=False)
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.bot.countdown = Mock(side_effect=[types.SimpleNamespace(x=x, y=y) for x, y in
                                              [(1700, 200), (1730, 800), (10, 10), (11, 11),
                                               (160, 200), (45, 200)]])
        self.bot.sample_signal_color = Mock(side_effect=[[255, 167, 38], [38, 198, 218]])
        self.bot.calibrate()
        self.assertEqual(self.bot.countdown.call_count, 6)
        self.bot.save.assert_called_once()
        self.bot.pyautogui.click.assert_not_called()

    def test_signal_colour_sample_ignores_grey_crosshair(self):
        point = types.SimpleNamespace(x=50, y=60)
        image = np.full((13, 13, 4), 156, dtype=np.uint8)
        image[3:10, 3:10, :3] = [38, 167, 255]  # orange in BGRA
        image[6, :, :3] = 156
        image[:, 6, :3] = 156
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        capture.return_value.grab = Mock(return_value=image)
        self.assertEqual(self.bot.sample_signal_color(point), [255, 167, 38])
        self.bot.pyautogui.click.assert_not_called()

    def test_grey_only_signal_sample_is_rejected(self):
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        capture.return_value.grab = Mock(return_value=np.full((13, 13, 4), 156, dtype=np.uint8))
        self.assertIsNone(self.bot.sample_signal_color(types.SimpleNamespace(x=50, y=60)))

    def test_grey_or_overlapping_colours_block_run_before_capture(self):
        for sell, buy in (([156, 156, 156], [156, 156, 156]),
                          ([156, 156, 156], [230, 230, 230]),
                          ([255, 167, 38], [235, 160, 48])):
            with self.subTest(sell=sell, buy=buy):
                self.cfg.update(use_ticket=False, red_rgb=sell, green_rgb=buy)
                self.bot.load = Mock(return_value=self.cfg)
                self.bot.run()
                self.bot.mss.MSS.assert_not_called()
                self.bot.pyautogui.click.assert_not_called()

    def test_current_cyan_orange_colours_pass_validation(self):
        self.assertEqual(self.bot.signal_color_errors(self.cfg), [])

    def test_bad_full_colour_calibration_does_not_overwrite_settings(self):
        original = copy.deepcopy(self.cfg)
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.bot.countdown = Mock(side_effect=[types.SimpleNamespace(x=x, y=y) for x, y in
                                              [(1592, 242), (1661, 776), (10, 10), (11, 11)]])
        self.bot.sample_signal_color = Mock(return_value=[156, 156, 156])
        self.bot.calibrate()
        self.bot.save.assert_not_called()
        self.assertEqual(self.cfg, original)
        self.bot.pyautogui.click.assert_not_called()

    def test_colour_only_calibration_preserves_area_buttons_and_timing(self):
        original = copy.deepcopy(self.cfg)
        self.bot.load = Mock(return_value=copy.deepcopy(self.cfg))
        self.bot.save = Mock()
        self.bot.countdown = Mock(return_value=types.SimpleNamespace(x=10, y=10))
        self.bot.sample_signal_color = Mock(side_effect=[[255, 180, 0], [0, 210, 225]])
        self.bot.calibrate_colors()
        self.bot.save.assert_called_once()
        saved = self.bot.save.call_args.args[0]
        self.assertEqual(saved["red_rgb"], [255, 180, 0])
        self.assertEqual(saved["green_rgb"], [0, 210, 225])
        self.assertEqual({key: value for key, value in saved.items() if key not in ("red_rgb", "green_rgb")},
                         {key: value for key, value in original.items() if key not in ("red_rgb", "green_rgb")})
        self.bot.pyautogui.click.assert_not_called()

    def test_one_click_run_accepts_absent_ticket_calibration(self):
        self.cfg.update(use_ticket=False, fill_sltp=False)
        for key in self.bot.ticket_required(self.cfg):
            self.cfg[key] = None
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.enter = Mock(return_value=True)
        self.prepare_run_frames()
        self.bot.run()
        self.bot.enter.assert_called_once_with(self.cfg, "sell", signal_seen_at=ANY, guard=self.guard)
        self.bot.save_test_observation.assert_not_called()
        events = [call.args[0] for call in self.journal.record.call_args_list]
        self.assertIn("click_attempt", events)
        self.assertIn("click_result", events)
        self.assertNotIn("test_signal", events)

    def test_test_mode_never_calls_entry(self):
        self.cfg.update(use_ticket=False, fill_sltp=False, buy_btn=None, sell_btn=None)
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.enter = Mock(side_effect=AssertionError("test must not enter an order"))
        self.prepare_run_frames()
        self.bot.run(dry=True)
        self.bot.enter.assert_not_called()
        self.bot.pyautogui.click.assert_not_called()
        self.assertTrue(any("SELL signal detected" in call.args[0]
                            for call in self.bot.log.call_args_list))
        self.assertEqual(self.bot.save_test_observation.call_count, 2)
        events = [call.args[0] for call in self.journal.record.call_args_list]
        self.assertIn("test_signal", events)
        self.assertNotIn("click_attempt", events)

    def test_cli_passes_explicit_strategy_without_changing_mode(self):
        self.bot.run = Mock()
        self.bot.main(["test", "--strategy", "5sec-reversed-regime"])
        self.bot.run.assert_called_once_with(dry=True, strategy_tag="5sec-reversed-regime")
        self.bot.run.reset_mock()
        self.bot.main(["run"])
        self.bot.run.assert_called_once_with(strategy_tag=None)

    def test_cli_rejects_bad_label_before_scanning(self):
        self.bot.run = Mock()
        with patch("sys.stderr"), self.assertRaises(SystemExit) as result:
            self.bot.main(["run", "--strategy", "../outside"])
        self.assertEqual(result.exception.code, 2)
        self.bot.run.assert_not_called()

    def test_click_interruption_is_recorded_and_not_retried(self):
        self.cfg.update(use_ticket=False, fill_sltp=False)
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.enter = Mock(side_effect=RuntimeError("simulated click failure"))
        self.prepare_run_frames()
        with self.assertRaisesRegex(RuntimeError, "simulated click failure"):
            self.bot.run(strategy_tag="5sec-regime")
        self.bot.enter.assert_called_once()
        self.assertIn("click_interrupted", [call.args[0] for call in self.journal.record.call_args_list])

    def test_evidence_contains_actual_pixels_and_selected_marker_metadata(self):
        from signal_tracker import Marker
        marker = Marker("buy", 10, 20, 5, 20, 16, 31, 60)
        image = np.zeros((3, 4, 4), dtype=np.uint8)
        image[:, :, :3] = [218, 198, 38]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(self.actual_save_test_observation(image, self.cfg, [marker], marker,
                                                        ["candidate"], 1, folder=folder))
            png = path.read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", png[16:24]), (4, 3))
            offset = 8
            while png[offset + 4:offset + 8] != b"IDAT":
                offset += struct.unpack(">I", png[offset:offset + 4])[0] + 12
            size = struct.unpack(">I", png[offset:offset + 4])[0]
            pixels = zlib.decompress(png[offset + 8:offset + 8 + size])
            self.assertEqual(pixels, (b"\x00" + bytes([38, 198, 218]) * 4) * 3)
            metadata = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(metadata["scan_region"], self.cfg["region"])
            self.assertEqual(metadata["selected"]["side"], "buy")
            self.assertEqual(metadata["messages"], ["candidate"])

    def test_dry_run_real_image_pipeline_never_clicks(self):
        self.cfg.update(use_ticket=False, fill_sltp=False, buy_btn=None, sell_btn=None)
        self.bot.load = Mock(return_value=self.cfg)
        self.bot.enter = Mock(side_effect=AssertionError("test must not enter an order"))
        empty = np.zeros((120, 80, 4), dtype=np.uint8)
        signal = empty.copy()
        for depth in range(12):
            signal[40 + depth, 50 - depth // 2:51 + depth // 2, :3] = self.cfg["green_rgb"][::-1]
        self.bot.keyboard.is_pressed.side_effect = [False] * 103 + [True]
        capture = self.bot.mss.MSS.return_value.__enter__ = Mock()
        self.bot.mss.MSS.return_value.__exit__ = Mock(return_value=False)
        capture.return_value.grab = Mock(side_effect=[empty] * 100 + [signal] * 3)
        self.bot.run(dry=True)
        self.bot.enter.assert_not_called()
        self.bot.pyautogui.click.assert_not_called()
        detected = [call.args[0] for call in self.bot.log.call_args_list if "BUY signal detected" in call.args[0]]
        self.assertEqual(len(detected), 1)

    def test_ready_click_precedes_console_logging_and_reports_timing(self):
        self.cfg.update(use_ticket=False, fill_sltp=False)
        self.bot.load = Mock(return_value=self.cfg)
        self.prepare_run_frames()
        events = []
        self.bot.log.side_effect = lambda message: events.append(("log", message))
        self.bot.pyautogui.click.side_effect = lambda *args: events.append(("click", args))
        self.bot.keyboard.is_pressed.side_effect = lambda key: any(event[0] == "click" for event in events)
        self.bot.run()
        click_index = next(index for index, event in enumerate(events) if event[0] == "click")
        self.assertEqual(events[click_index], ("click", tuple(self.cfg["sell_btn"])))
        self.assertTrue(all(event[0] == "log" and event[1].startswith("SCAN:")
                            for event in events[:click_index]))
        self.bot.pyautogui.click.assert_called_once_with(*self.cfg["sell_btn"])
        self.assertTrue(any("first observation to click" in event[1] for event in events if event[0] == "log"))

    def test_scan_period_includes_processing_time(self):
        self.cfg.update(use_ticket=False, poll_ms=10)
        self.bot.load = Mock(return_value=self.cfg)
        self.prepare_run_frames()
        self.bot.keyboard.is_pressed.side_effect = [False, True]
        self.bot.time.sleep = Mock(wraps=self.bot.time.sleep)
        self.bot.run(dry=True)
        self.bot.time.sleep.assert_called_once()
        # The fake clock advances 4ms during capture/processing.
        self.assertAlmostEqual(self.bot.time.sleep.call_args.args[0], 0.006)
        self.bot.pyautogui.click.assert_not_called()

    def test_slow_capture_does_not_add_an_extra_poll_delay(self):
        self.cfg.update(use_ticket=False, poll_ms=1)
        self.bot.load = Mock(return_value=self.cfg)
        self.prepare_run_frames()
        self.bot.keyboard.is_pressed.side_effect = [False, True]
        self.bot.time.sleep = Mock(wraps=self.bot.time.sleep)
        self.bot.run(dry=True)
        self.bot.time.sleep.assert_not_called()
        self.bot.pyautogui.click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
