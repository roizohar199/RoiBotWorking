"""Positive screen evidence for TradingView's filled blue/red quote cards.

Pure pixel validation. This module never moves a mouse or submits an order.
It deliberately rejects unrecognized themes, dimmed cards and covered points.
"""
import numpy as np


def quote_card_visible(rgb, side, point):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or side not in ("buy", "sell"):
        return False, "Invalid quote image"
    height, width = image.shape[:2]
    if not (50 <= width <= 250 and 25 <= height <= 55):
        return False, "Quote card dimensions changed"
    x, y = point
    if not (2 <= x < width - 2 and 2 <= y < height - 2):
        return False, "Click point outside quote card interior"
    red, green, blue = np.moveaxis(image.astype(np.int16), -1, 0)
    if side == "buy":
        color = ((blue >= 170) & (red <= 120) & (green <= 180)
                 & (blue - red >= 90) & (blue - green >= 50))
    else:
        color = ((red >= 170) & (green <= 135) & (blue <= 155)
                 & (red - green >= 70) & (red - blue >= 60))
    white = ((np.minimum(np.minimum(red, green), blue) >= 190)
             & (np.maximum(np.maximum(red, green), blue)
                - np.minimum(np.minimum(red, green), blue) <= 35))
    # The actual click lies in the solid left margin, clear of changing prices.
    # A partly covering popup cannot pass just because most of a card is visible.
    click_clear = color[y - 2:y + 3, x - 2:x + 3].mean() >= .9
    colored = float(color.mean())
    text = float(white.mean())
    visible = bool(click_clear and colored >= .55 and .02 <= text <= .40)
    return visible, f"{side.upper()} card: color={colored:.2f}, text={text:.2f}, click_clear={bool(click_clear)}"


class DesktopQuoteReader:
    """Fresh desktop pixels, acquired in the account-reader thread only."""
    def __init__(self):
        import mss
        self.capture = mss.MSS()

    def check(self, bounds, side, point):
        left, top, right, bottom = bounds
        frame = self.capture.grab(dict(left=left, top=top, width=right-left, height=bottom-top))
        rgb = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(bottom-top, right-left, 3)
        return quote_card_visible(rgb, side, (point[0]-left, point[1]-top))

    def close(self):
        self.capture.close()
