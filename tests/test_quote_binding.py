"""Exact permitted chart descriptions; simulated accessibility controls."""
from types import SimpleNamespace
import unittest
from position_guard import WindowsAccountReader


class QuoteDescriptionTests(unittest.TestCase):
    def reader(self, description, quantity="1"):
        def node(name, kind, x, y, w, h):
            return SimpleNamespace(Name=name, ControlTypeName=kind, IsOffscreen=False,
                                   BoundingRectangle=SimpleNamespace(left=x, top=y, right=x+w, bottom=y+h))
        text = node(quantity, "TextControl", 976, 265, 10, 16)
        group = node("", "GroupControl", 971, 263, 25, 22)
        sell = node("", "GroupControl", 844, 251, 120, 34)
        buy = node("", "GroupControl", 1003, 251, 121, 34)
        parent = SimpleNamespace(GetChildren=lambda: [sell, group, buy])
        group.GetChildren = lambda: [text]
        group.GetParentControl = lambda: parent
        text.GetParentControl = lambda: group
        image = node(description, "ImageControl", 835, 219, 1018, 500)
        cfg = dict(position_guard=dict(chart_description="Chart for CME_MINI:NQ1!, 1 second",
                                       chart_descriptions=["Chart for CME_MINI:NQZ2026, 1 second"]))
        reader = WindowsAccountReader(cfg)
        reader.chart = object()
        reader._walk = lambda *args, **kwargs: iter([image, text])
        return reader

    def test_both_explicit_nq_symbols_are_recognized_on_one_second_chart(self):
        for symbol in ("NQ1!", "NQZ2026"):
            reader = self.reader(f"Chart for CME_MINI:{symbol}, 1 second")
            self.assertEqual(reader._quote_widget(), (1, (1009, 268), (850, 268)))

    def test_other_contract_or_timeframe_is_not_silently_accepted(self):
        for description in ("Chart for CME_MINI:MNQZ2026, 1 second", "Chart for CME_MINI:NQZ2026, 5 seconds"):
            reader = self.reader(description)
            self.assertEqual(reader._quote_widget(), (None, None, None))
            self.assertIn("Unexpected chart", reader.quantity_detail)

    def test_quantity_is_read_as_displayed_instead_of_coerced_to_one(self):
        reader = self.reader("Chart for CME_MINI:NQZ2026, 1 second", quantity="2")
        self.assertEqual(reader._quote_widget()[0], 2)
