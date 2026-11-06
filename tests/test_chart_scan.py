"""Captured chart fixture and synthetic new markers. No desktop/trading input."""
from pathlib import Path
import unittest
import numpy as np
from position_guard import signal_scan_region
from signal_tracker import find_markers, MarkerTracker, choose_signal, Marker


class BoundChartScanTests(unittest.TestCase):
    def test_scan_tracks_chart_geometry_and_excludes_axes(self):
        cfg = dict(scan_bound_chart=True, region=dict(left=1726, top=287, width=52, height=384))
        self.assertEqual(signal_scan_region(cfg, (835, 163, 1853, 711)),
                         dict(left=843, top=167, width=920, height=518))
        self.assertIsNone(signal_scan_region(cfg, None))
        self.assertIsNone(signal_scan_region(cfg, (0, 0, 100, 100)))
        moved = signal_scan_region(cfg, (-1800, 100, -800, 800))
        self.assertEqual(moved["left"], -1792)

    def test_manual_scan_is_preserved_when_explicitly_selected(self):
        cfg = dict(region=dict(left=1726, top=287, width=52, height=384))
        self.assertEqual(signal_scan_region(cfg, None), cfg["region"])

    def detect(self, image):
        return (find_markers(image, [38, 198, 218], 25, 12, "buy")
                + find_markers(image, [255, 167, 38], 25, 12, "sell"))

    def chart(self):
        with np.load(Path(__file__).parent / "fixtures/one_second_triangles.npz") as data:
            return data["image"].copy()

    def test_real_historical_triangles_are_detected_and_never_sent_at_startup(self):
        markers = self.detect(self.chart())
        self.assertEqual(len(markers), 6)
        tracker = MarkerTracker()
        self.assertEqual(tracker.update(markers, 0), [])
        self.assertEqual(tracker.update(markers, .4), [])
        self.assertIsNone(choose_signal([], markers, 0)[0])

    def test_new_triangle_above_old_calibrated_area_is_selected_once(self):
        image = self.chart()
        tracker = MarkerTracker()
        tracker.update(self.detect(image), 0)
        tracker.update(self.detect(image), .4)
        for depth in range(12):
            image[90+depth, 890-depth//2:891+depth//2, :3] = [218, 198, 38]
        markers = self.detect(image)
        self.assertEqual(tracker.update(markers, .5), [])
        fresh = tracker.update(markers, .51)
        selected, _ = choose_signal(fresh, markers, 0)
        self.assertIsNotNone(selected)
        self.assertEqual((selected.side, selected.x, selected.y), ("buy", 890, 90))
        self.assertLess(167+selected.y, 287)
        self.assertEqual(tracker.update(markers, .52), [])

    def test_collective_right_pan_is_baselined_as_history(self):
        markers = self.detect(self.chart())
        tracker = MarkerTracker()
        tracker.update(markers, 0)
        tracker.update(markers, .4)
        moved = [Marker(m.side,m.x+10,m.y,m.left+10,m.top,m.right+10,m.bottom,m.pixels) for m in markers]
        self.assertEqual(tracker.update(moved, .5), [])
        self.assertTrue(any("moved right" in message for message in tracker.messages))
        self.assertEqual(tracker.update(moved, .9), [])

    def test_newly_visible_historical_marker_outside_live_band_is_not_traded(self):
        marker = Marker("buy", 600, 100, 594, 100, 607, 112, 60)
        selected, messages = choose_signal([marker], [marker], 0, live_left=800)
        self.assertIsNone(selected)
        self.assertTrue(any("historical" in message for message in messages))

    def test_rightmost_new_marker_inside_live_band_is_eligible(self):
        marker = Marker("buy", 890, 100, 884, 100, 897, 112, 60)
        self.assertIs(choose_signal([marker], [marker], 0, live_left=800)[0], marker)
