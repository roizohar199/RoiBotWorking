"""Screen-marker detection and tracking. No screen, mouse or trading APIs."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Marker:
    side: str
    x: float
    y: float
    left: int
    top: int
    right: int
    bottom: int
    pixels: int


def _is_triangle(mask, tip_x, tip_y):
    """Check an UP-pointing filled triangle after SELL masks are flipped.

    A coloured component alone is insufficient: candles, labels and rectangles
    can have the same colour. Require a narrow tip and a widening filled body.
    Incomplete/occluded shapes can be rejected rather than guessed.
    """
    x, y = int(round(tip_x)), int(tip_y)
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or not mask[y, x]:
        return False
    column = mask[y:min(mask.shape[0], y + 41), x]
    gaps = np.flatnonzero(~column)
    height = int(gaps[0]) if len(gaps) else len(column)
    if not 6 <= height <= 40:
        return False

    # Width of the CONTIGUOUS top run, not the whole component: neighbouring
    # triangles may touch at their bases while retaining distinct narrow tips.
    def run_width(row):
        left = right = x
        while left > 0 and row[left - 1]:
            left -= 1
        while right + 1 < mask.shape[1] and row[right + 1]:
            right += 1
        return right - left + 1

    if run_width(mask[y]) > max(2, height * 0.4):
        return False
    # A circle, diamond or triangle-with-an-arrow-stem narrows again at its
    # bottom. A breakout triangle needs a broad filled base.
    if max(run_width(mask[y + height - 1]), run_width(mask[y + height - 2])) < max(4, height * 0.65):
        return False

    radius = (height - 1) // 2
    offsets = np.arange(-radius, radius + 1)
    rows = np.arange(height)
    expected = np.abs(offsets[None, :]) <= (rows[:, None] // 2)
    columns = x + offsets
    valid = (columns >= 0) & (columns < mask.shape[1])
    observed = np.zeros(expected.shape, dtype=bool)
    observed[:, valid] = mask[y:y + height, columns[valid]]
    if np.count_nonzero(observed & expected) < 0.82 * np.count_nonzero(expected):
        return False

    # A solid rectangle also contains a triangular core, so check its shoulders.
    # One clear shoulder is sufficient when another real triangle touches it.
    upper_rows = np.arange(max(3, (height * 2) // 3))
    outside = upper_rows // 2 + 2
    occupancies = []
    for sign in (-1, 1):
        columns = x + sign * outside
        valid = (columns >= 0) & (columns < mask.shape[1])
        occupancies.append(np.count_nonzero(mask[y + upper_rows[valid], columns[valid]])
                           / len(upper_rows))
    return min(occupancies) <= 0.35


def find_markers(img, rgb, tolerance, min_pixels, side):
    """Find separate colour components, splitting touching triangle tips.

    Coordinates are relative to the calibrated capture area. This is still
    visual evidence, not proof of a new candle or of a TradingView order.
    """
    # Compare byte planes directly: avoid copying/converting the whole RGB
    # image twice per capture. Clamp bounds to preserve uint8 edge behaviour.
    mask = np.ones(img.shape[:2], dtype=bool)
    for channel, target in enumerate(reversed(rgb)):
        low = max(0, int(target) - tolerance)
        high = min(255, int(target) + tolerance)
        plane = img[:, :, channel]
        mask &= (plane >= low) & (plane <= high)
    # A one-pixel chart grid line can cut ONE triangle into two components.
    # Bridge just that gap; separate tips are still split below.
    mask[:, 1:-1] |= mask[:, :-2] & mask[:, 2:]
    if side == "sell":
        mask = mask[::-1]  # Both directions can now be examined tip-first.
    height = mask.shape[0]
    runs, parents, previous = [], [], []
    previous_y = -2

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for y in np.flatnonzero(mask.any(axis=1)):
        if y != previous_y + 1:
            previous = []
        edges = np.diff(np.r_[False, mask[y], False].astype(np.int8))
        current = []
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            index = len(runs)
            runs.append((int(y), int(start), int(stop)))
            parents.append(index)
            for old_index in previous:
                _, old_start, old_stop = runs[old_index]
                if start <= old_stop and stop >= old_start:  # 8-connected pixels
                    parents[root(index)] = root(old_index)
            current.append(index)
        previous, previous_y = current, y

    groups = {}
    for index, run in enumerate(runs):
        groups.setdefault(root(index), []).append(run)
    markers = []
    for component in groups.values():
        if sum(stop - start for _, start, stop in component) < min_pixels:
            continue
        left = min(start for _, start, _ in component)
        right = max(stop for _, _, stop in component)
        top = min(y for y, _, _ in component)
        bottom = max(y for y, _, _ in component) + 1
        if right - left < 3 or bottom - top < 3:
            continue
        envelope = np.full(right - left, height, dtype=int)
        for y, start, stop in component:
            envelope[start - left:stop - left] = np.minimum(envelope[start - left:stop - left], y)
        # A flat local minimum is a triangle tip. Separate tips can survive
        # even when neighbouring triangles touch along their wider bases.
        tips = []
        start = 0
        while start < len(envelope):
            stop = start + 1
            while stop < len(envelope) and envelope[stop] == envelope[start]:
                stop += 1
            value = int(envelope[start])
            if (start > 0 and stop < len(envelope)
                    and envelope[start - 1] > value and envelope[stop] > value
                    and max(envelope[max(0, start - 3):start]) >= value + 2
                    and max(envelope[stop:min(len(envelope), stop + 3)]) >= value + 2):
                tips.append((left + (start + stop - 1) / 2, value))
            start = stop
        # Small jagged/antialiased edges are not distinct neighbouring tips.
        separated_tips = []
        for tip in tips:
            if separated_tips and tip[0] - separated_tips[-1][0] < 5:
                if tip[1] < separated_tips[-1][1]:
                    separated_tips[-1] = tip
            else:
                separated_tips.append(tip)
        tips = separated_tips
        if not tips:
            continue
        for index, (tip_x, tip_y) in enumerate(tips):
            if not _is_triangle(mask, tip_x, tip_y):
                continue
            segment_left = left if index == 0 else int((tips[index - 1][0] + tip_x) / 2) + 1
            segment_right = right if index == len(tips) - 1 else int((tip_x + tips[index + 1][0]) / 2) + 1
            pixels = sum(max(0, min(stop, segment_right) - max(start, segment_left))
                         for _, start, stop in component)
            if pixels < min_pixels:
                continue
            actual_y = tip_y if side == "buy" else height - 1 - tip_y
            actual_top = top if side == "buy" else height - bottom
            actual_bottom = bottom if side == "buy" else height - top
            markers.append(Marker(side, tip_x, actual_y, segment_left, actual_top,
                                  segment_right, actual_bottom, pixels))
    return markers


@dataclass
class Track:
    marker: Marker
    last_seen: float
    first_seen: float
    frames: int = 1
    handled: bool = False


class MarkerTracker:
    """Associate markers one-to-one, including leftward chart scrolling.

    Startup markers are baselined, brief disappearance is tolerated, and each
    new track is emitted once. A marker must survive two consecutive captures.
    """
    def __init__(self, warmup_sec=0.3, forget_sec=1.0, max_scroll_px=24, jitter_px=3):
        self.warmup_sec = warmup_sec
        self.forget_sec = forget_sec
        self.max_scroll_px = max_scroll_px
        self.jitter_px = jitter_px
        self.started = None
        self.last_update = None
        self.messages = []
        self.tracks = []

    def first_seen_at(self, marker):
        """Start of this marker's consecutive observations, for timing logs."""
        return next(track.first_seen for track in self.tracks if track.marker is marker)

    def update(self, markers, now):
        self.messages = []
        if self.last_update is not None and now - self.last_update > self.forget_sec:
            # A stalled capture must not make the same old screen tradable.
            self.tracks = []
            self.started = now
            self.messages.append("SKIP: capture gap; re-baselining visible markers for 0.3s")
        self.last_update = now
        if self.started is None:
            self.started = now
        warming_up = now - self.started < self.warmup_sec
        self.tracks = [track for track in self.tracks if now - track.last_seen <= self.forget_sec]
        # A collective rightward move is chart navigation, not new candles.
        # Re-baseline that scene rather than trading historical triangles.
        if len(markers) == len(self.tracks) and len(markers) >= 2:
            old = sorted(self.tracks, key=lambda item: (item.marker.side, item.marker.x))
            current = sorted(markers, key=lambda item: (item.side, item.x))
            shifts = [new.x - previous.marker.x for previous, new in zip(old, current)]
            if (all(previous.marker.side == new.side for previous, new in zip(old, current))
                    and min(shifts) > self.jitter_px and max(shifts)-min(shifts) <= 2):
                self.tracks = []
                self.started = now
                warming_up = True
                self.messages.append("SKIP: chart moved right; re-baselining historical markers")
        available = list(self.tracks)
        assigned = []
        unmatched = []
        # Old markers move LEFT as candles arrive. Matching from left to right
        # avoids assigning an old track to a new marker at its former position.
        for marker in sorted(markers, key=lambda item: item.x):
            candidates = [track for track in available if track.marker.side == marker.side
                          and -self.max_scroll_px <= marker.x - track.marker.x <= self.jitter_px
                          and abs(marker.y - track.marker.y) <= 12]
            if candidates:
                track = min(candidates, key=lambda item: (abs(marker.y - item.marker.y),
                                                         abs(marker.x - item.marker.x)))
                available.remove(track)
                assigned.append((track, marker))
            else:
                unmatched.append(marker)
        # If all unmatched columns retain their order, allow a vertical scale
        # change at the same time as leftward scrolling. Do not use this for
        # scenes with extra/new columns, whose identity would be ambiguous.
        if unmatched and len(unmatched) == len(available):
            old_order = sorted(available, key=lambda item: (item.marker.side, item.marker.x))
            new_order = sorted(unmatched, key=lambda item: (item.side, item.x))
            if all(track.marker.side == marker.side
                   and -self.max_scroll_px <= marker.x - track.marker.x <= self.jitter_px
                   for track, marker in zip(old_order, new_order)):
                assigned.extend(zip(old_order, new_order))
                unmatched, available = [], []
        # Price auto-scaling can also move a marker vertically while a new
        # column appears. Use only a unique same-column match in that case.
        still_unmatched = []
        for marker in unmatched:
            candidates = [track for track in available if track.marker.side == marker.side
                          and abs(marker.x - track.marker.x) <= self.jitter_px]
            if len(candidates) == 1 and sum(
                    other.side == marker.side and abs(other.x - candidates[0].marker.x) <= self.jitter_px
                    for other in unmatched) == 1:
                track = candidates[0]
                available.remove(track)
                assigned.append((track, marker))
            else:
                still_unmatched.append(marker)
        for track in available:
            track.frames = 0
        for marker in still_unmatched:
            track = Track(marker, now, first_seen=now, handled=warming_up)
            self.tracks.append(track)
            assigned.append((track, marker))
            track.frames = 0
        fresh = []
        for track, marker in assigned:
            if track.frames == 0:
                track.first_seen = now
            track.marker = marker
            track.last_seen = now
            track.frames += 1
            if warming_up:
                track.handled = True
            elif not track.handled and track.frames >= 2:
                track.handled = True
                fresh.append(marker)
        return fresh


def choose_signal(fresh, visible, cooldown_remaining, live_left=None):
    """Choose the rightmost NEW marker; explain consumed/skipped events.

    Never queue a cooldown signal for a delayed trade. Two colours at different
    candle columns no longer block one another. Same-column conflicts are
    ambiguous and are deliberately not traded.
    """
    messages = []
    eligible = []
    for marker in fresh:
        label = f"{marker.side.upper()} at scan x={marker.x:.1f}, y={marker.y:.1f}"
        if live_left is not None and marker.x < live_left:
            messages.append(f"SKIP {label}: outside the live candle area; historical marker")
        elif marker.left == 0:
            messages.append(f"SKIP {label}: clipped at left edge; possibly entering from history")
        elif visible and marker.x < max(item.x for item in visible) - 3:
            messages.append(f"SKIP {label}: an existing marker is farther right (older candle)")
        elif any(item.side != marker.side and abs(item.x - marker.x) <= 3 for item in visible):
            messages.append(f"SKIP {label}: BUY and SELL share the same candle column")
        else:
            eligible.append(marker)
    if not eligible:
        return None, messages
    selected = max(eligible, key=lambda item: item.x)
    for marker in eligible:
        if marker is not selected:
            messages.append(f"SKIP {marker.side.upper()}: newer marker selected farther right")
    if cooldown_remaining > 0:
        messages.append(f"SKIP {selected.side.upper()}: cooldown {cooldown_remaining:.2f}s remaining; not queued")
        return None, messages
    return selected, messages
