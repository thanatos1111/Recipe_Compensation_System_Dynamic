import unittest

from ui.model_panel import ModelPanel


class TestBenchmarkUIPolishHelpers(unittest.TestCase):
    def test_dedupe_preserve_order(self) -> None:
        items = ["a", "b", "a", "c", "b", "d"]
        out = ModelPanel._dedupe_preserve_order(items)
        self.assertEqual(out, ["a", "b", "c", "d"])

    def test_dedupe_preserve_order_empty(self) -> None:
        out = ModelPanel._dedupe_preserve_order([])
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()

