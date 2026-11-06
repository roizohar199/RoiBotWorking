"""Synthetic quote pixels only; no mouse, browser or broker access."""
import unittest
import numpy as np
from entry_surface import quote_card_visible


class QuotePixelTests(unittest.TestCase):
    def card(self, side="buy"):
        image = np.full((34, 121, 3), (41, 98, 255) if side == "buy" else (246, 54, 69), dtype=np.uint8)
        image[5:10, 20:100] = 240
        image[24:28, 10:110] = 240
        return image

    def test_filled_quote_cards_have_positive_color_text_and_click_evidence(self):
        for side in ("buy", "sell"):
            self.assertTrue(quote_card_visible(self.card(side), side, (6, 17))[0])

    def test_swapped_buy_and_sell_cards_block(self):
        self.assertFalse(quote_card_visible(self.card("sell"), "buy", (6, 17))[0])
        self.assertFalse(quote_card_visible(self.card("buy"), "sell", (6, 17))[0])

    def test_another_application_or_blank_region_blocks(self):
        self.assertFalse(quote_card_visible(np.full((34, 121, 3), 28), "buy", (6, 17))[0])

    def test_dimmed_modal_backdrop_blocks(self):
        self.assertFalse(quote_card_visible(self.card() // 2, "buy", (6, 17))[0])

    def test_popup_covering_only_the_click_point_blocks(self):
        image = self.card()
        image[14:21, 3:10] = 20
        self.assertFalse(quote_card_visible(image, "buy", (6, 17))[0])

    def test_solid_blue_without_price_or_preset_text_is_not_a_quote(self):
        image = np.full((34, 121, 3), (41, 98, 255), dtype=np.uint8)
        self.assertFalse(quote_card_visible(image, "buy", (6, 17))[0])

    def test_unknown_style_dimensions_and_click_point_fail_closed(self):
        for point in ((0, 0), (122, 17), (-1, 17)):
            self.assertFalse(quote_card_visible(self.card(), "buy", point)[0])
        self.assertFalse(quote_card_visible(np.zeros((3, 8, 3)), "buy", (4, 1))[0])
        self.assertFalse(quote_card_visible(np.zeros((34, 121)), "buy", (6, 17))[0])

    def test_price_text_changes_leave_solid_click_margin_valid(self):
        image = self.card()
        image[5:10, 20:100] = (41, 98, 255)
        image[4:12, 30:90:2] = 255
        self.assertTrue(quote_card_visible(image, "buy", (6, 17))[0])
