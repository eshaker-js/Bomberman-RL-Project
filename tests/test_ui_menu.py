"""Headless checks for the menu model (ui/menu.py). Pygame is not required."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.menu import Menu  # noqa: E402


class TestMenu(unittest.TestCase):
    def test_starts_on_first_option(self):
        self.assertEqual(Menu(["A", "B", "C"]).selected, "A")

    def test_down_and_up(self):
        menu = Menu(["A", "B", "C"])
        menu.move(1)
        self.assertEqual(menu.selected, "B")
        menu.move(1)
        self.assertEqual(menu.selected, "C")
        menu.move(-1)
        self.assertEqual(menu.selected, "B")

    def test_wraps_around(self):
        menu = Menu(["A", "B", "C"])
        menu.move(-1)
        self.assertEqual(menu.selected, "C")
        menu.move(1)
        self.assertEqual(menu.selected, "A")

    def test_single_option(self):
        menu = Menu(["BACK"])
        menu.move(1)
        self.assertEqual(menu.selected, "BACK")

    def test_empty_menu_rejected(self):
        with self.assertRaises(ValueError):
            Menu([])


if __name__ == "__main__":
    unittest.main()
