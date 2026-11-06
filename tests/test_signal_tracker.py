"""Synthetic chart frames only: no real screen capture or mouse input."""
import unittest

import numpy as np

from signal_tracker import MarkerTracker, choose_signal, find_markers


BUY = [38, 198, 218]
SELL = [255, 167, 38]


def frame(*triangles):
    image = np.zeros((180, 100, 4), dtype=np.uint8)
    for side, x, y in triangles:
        colour = BUY if side == "buy" else SELL
        for depth in range(12):
            row = y + depth if side == "buy" else y - depth
            half_width = depth // 2
            image[row, max(0, x - half_width):min(100, x + half_width + 1), :3] = colour[::-1]
    return image


def detect(image):
    return (find_markers(image, BUY, 25, 12, "buy")
            + find_markers(image, SELL, 25, 12, "sell"))


class SignalTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = MarkerTracker()

    def update(self, now, *triangles):
        visible = detect(frame(*triangles))
        return self.tracker.update(visible, now), visible

    def test_separate_and_touching_same_colour_triangles(self):
        for spacing in (6, 20):
            with self.subTest(spacing=spacing):
                markers = detect(frame(("buy", 30, 50), ("buy", 30 + spacing, 50)))
                self.assertEqual([marker.x for marker in markers], [30, 30 + spacing])

    def test_sell_tips_are_not_confused_with_buy_tips(self):
        markers = detect(frame(("sell", 50, 100), ("sell", 56, 100), ("buy", 80, 120)))
        self.assertEqual(sorted((marker.side, marker.x, marker.y) for marker in markers),
                         [("buy", 80, 120), ("sell", 50, 100), ("sell", 56, 100)])

    def test_single_pixel_grid_line_does_not_split_one_triangle(self):
        for side, y in (("buy", 50), ("sell", 100)):
            with self.subTest(side=side):
                image = frame((side, 50, y))
                image[:, 50] = 0
                markers = detect(image)
                self.assertEqual(len(markers), 1)
                self.assertLessEqual(abs(markers[0].x - 50), 1)

    def test_scattered_pixels_and_thin_lines_do_not_form_a_marker(self):
        image = frame()
        image[10, 10:30, :3] = BUY[::-1]
        for i in range(15):
            image[40 + i * 3, 40, :3] = BUY[::-1]
        self.assertEqual(detect(image), [])

    def test_filled_coloured_rectangles_and_candles_are_not_triangles(self):
        for width, height in ((4, 4), (4, 8), (8, 12), (12, 12), (9, 70), (30, 8)):
            for colour in (BUY, SELL):
                with self.subTest(width=width, height=height, colour=colour):
                    image = frame()
                    image[30:30 + height, 30:30 + width, :3] = colour[::-1]
                    self.assertEqual(detect(image), [])

    def test_triangle_with_wrong_colour_direction_is_rejected(self):
        for side, y in (("buy", 40), ("sell", 100)):
            image = frame((side, 50, y))
            wrong_colour = SELL if side == "buy" else BUY
            image[np.any(image[:, :, :3] != 0, axis=2), :3] = wrong_colour[::-1]
            self.assertEqual(detect(image), [])

    def test_filled_circle_and_diamond_are_not_triangles(self):
        yy, xx = np.ogrid[:180, :100]
        for shape in (((xx - 50) ** 2 + (yy - 60) ** 2 <= 8 ** 2),
                      (np.abs(xx - 50) + np.abs(yy - 60) <= 8)):
            image = frame()
            image[shape, :3] = BUY[::-1]
            self.assertEqual(detect(image), [])

    def test_triangle_shaped_arrow_head_with_stem_is_rejected(self):
        image = frame(("buy", 50, 50))
        image[62:74, 50:52, :3] = BUY[::-1]
        self.assertEqual(detect(image), [])

    def test_hollow_text_like_triangle_is_not_a_filled_signal(self):
        image = frame()
        for depth in range(16):
            image[50 + depth, 50 - depth // 2, :3] = BUY[::-1]
            image[50 + depth, 50 + depth // 2, :3] = BUY[::-1]
        image[59, 46:55, :3] = BUY[::-1]
        self.assertEqual(detect(image), [])

    def test_existing_startup_marker_never_trades(self):
        for now in (0, 0.15, 0.35, 1, 5, 10):
            self.assertEqual(self.update(now, ("buy", 50, 50))[0], [])

    def test_new_same_colour_marker_while_previous_remains(self):
        old = ("buy", 30, 50)
        new = ("buy", 50, 70)
        self.update(0, old)
        self.update(0.4, old)
        self.assertEqual(self.update(1, old, new)[0], [])
        fresh, visible = self.update(1.02, old, new)
        self.assertEqual([marker.x for marker in fresh], [50])
        self.assertEqual(self.tracker.first_seen_at(fresh[0]), 1)
        selected, messages = choose_signal(fresh, visible, 0)
        self.assertEqual(selected.side, "buy")
        self.assertEqual(messages, [])
        self.assertEqual(self.update(1.04, old, new)[0], [])

    def test_new_opposite_colour_not_blocked_by_older_marker(self):
        old = ("buy", 30, 50)
        new = ("sell", 60, 100)
        self.update(0, old)
        self.update(0.4, old)
        self.update(1, old, new)
        fresh, visible = self.update(1.02, old, new)
        selected, messages = choose_signal(fresh, visible, 0)
        self.assertEqual(selected.side, "sell")
        self.assertEqual(messages, [])

    def test_scrolling_into_former_position_does_not_lose_new_marker(self):
        old = ("buy", 60, 50)
        self.update(0, old)
        self.update(0.4, old)
        self.update(1, ("buy", 54, 50), ("buy", 60, 50))
        fresh, _ = self.update(1.02, ("buy", 54, 50), ("buy", 60, 50))
        self.assertEqual([marker.x for marker in fresh], [60])

    def test_continuous_left_scroll_and_vertical_jitter_do_not_repeat(self):
        self.update(0, ("buy", 80, 70))
        for index in range(1, 10):
            fresh, _ = self.update(index * 0.2, ("buy", 80 - index * 6, 70 + index % 2))
            self.assertEqual(fresh, [])

    def test_vertical_auto_scale_does_not_repeat_a_marker(self):
        self.update(0, ("buy", 60, 30))
        self.update(0.4, ("buy", 60, 30))
        self.assertEqual(self.update(0.6, ("buy", 60, 110))[0], [])
        self.assertEqual(self.update(0.62, ("buy", 60, 110))[0], [])

    def test_simultaneous_scroll_and_price_scale_change_do_not_repeat(self):
        old = (("buy", 40, 30), ("buy", 70, 70))
        shifted = (("buy", 34, 60), ("buy", 64, 120))
        self.update(0, *old)
        self.update(0.4, *old)
        self.assertEqual(self.update(0.6, *shifted)[0], [])
        self.assertEqual(self.update(0.62, *shifted)[0], [])

    def test_brief_occlusion_does_not_repeat(self):
        self.update(0, ("buy", 60, 50))
        self.update(0.4, ("buy", 60, 50))
        self.update(0.5)
        self.assertEqual(self.update(0.7, ("buy", 60, 50))[0], [])
        self.assertEqual(self.update(0.72, ("buy", 60, 50))[0], [])

    def test_capture_stall_rebaselines_instead_of_repeating_old_marker(self):
        self.update(0, ("buy", 60, 50))
        self.update(0.4, ("buy", 60, 50))
        self.assertEqual(self.update(3, ("buy", 60, 50))[0], [])
        self.assertIn("capture gap", self.tracker.messages[0])
        self.assertEqual(self.update(3.02, ("buy", 60, 50))[0], [])
        self.assertEqual(self.update(3.4, ("buy", 60, 50))[0], [])

    def test_one_frame_flash_is_not_a_signal(self):
        self.update(0)
        self.update(0.4, ("buy", 60, 50))
        self.assertEqual(self.update(0.42)[0], [])

    def test_cooldown_skip_is_consumed_not_delayed(self):
        self.update(0)
        self.update(0.4, ("buy", 60, 50))
        fresh, visible = self.update(0.42, ("buy", 60, 50))
        selected, messages = choose_signal(fresh, visible, 2)
        self.assertIsNone(selected)
        self.assertTrue(any("cooldown" in message for message in messages))
        for now in (0.6, 1, 1.8, 2.5, 3):
            fresh, visible = self.update(now, ("buy", 60, 50))
            self.assertEqual(fresh, [])
            self.assertIsNone(choose_signal(fresh, visible, 0)[0])

    def test_same_column_opposite_colours_are_explicitly_ambiguous(self):
        visible = detect(frame(("buy", 60, 40), ("sell", 60, 100)))
        selected, messages = choose_signal(visible, visible, 0)
        self.assertIsNone(selected)
        self.assertTrue(all("same candle column" in message for message in messages))

    def test_newly_exposed_older_marker_is_skipped(self):
        visible = detect(frame(("buy", 30, 50), ("sell", 70, 100)))
        selected, messages = choose_signal([visible[0]], visible, 0)
        self.assertIsNone(selected)
        self.assertIn("older candle", messages[0])

    def test_clipped_marker_at_left_border_is_skipped(self):
        visible = detect(frame(("buy", 2, 50)))
        selected, messages = choose_signal(visible, visible, 0)
        self.assertIsNone(selected)
        self.assertIn("left edge", messages[0])


if __name__ == "__main__":
    unittest.main()
