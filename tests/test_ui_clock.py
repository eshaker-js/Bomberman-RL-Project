"""Headless checks for the UI tick clock (ui/clock.py). Pygame is not required."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.clock import TickClock  # noqa: E402
from ui.config import TICK_MS  # noqa: E402


class TestTickClock(unittest.TestCase):
    def test_default_tick_is_250_ms(self):
        self.assertEqual(TICK_MS, 250)
        self.assertEqual(TickClock().tick_ms, 250)

    def test_fires_exactly_when_a_tick_has_passed(self):
        clock = TickClock(250)
        self.assertFalse(clock.update(100))
        self.assertFalse(clock.update(149))
        self.assertTrue(clock.update(1))
        self.assertEqual(clock.elapsed, 0)

    def test_keeps_the_remainder_after_a_tick(self):
        clock = TickClock(250)
        self.assertTrue(clock.update(260))
        self.assertAlmostEqual(clock.progress, 10 / 250)
        self.assertFalse(clock.update(239))
        self.assertTrue(clock.update(1))

    def test_progress_goes_from_zero_to_one(self):
        clock = TickClock(200)
        self.assertEqual(clock.progress, 0.0)
        clock.update(50)
        self.assertAlmostEqual(clock.progress, 0.25)
        clock.update(100)
        self.assertAlmostEqual(clock.progress, 0.75)

    def test_at_most_one_tick_per_update_and_backlog_dropped(self):
        clock = TickClock(250)
        self.assertTrue(clock.update(2000))
        self.assertEqual(clock.elapsed, 0)
        self.assertFalse(clock.update(10))

    def test_paused_clock_does_not_advance(self):
        clock = TickClock(250)
        clock.update(100)
        clock.paused = True
        self.assertFalse(clock.update(1000))
        self.assertAlmostEqual(clock.progress, 0.4)
        clock.paused = False
        self.assertTrue(clock.update(150))

    def test_finish_holds_at_end_of_tick(self):
        clock = TickClock(250)
        clock.update(30)
        clock.finish()
        self.assertEqual(clock.progress, 1.0)
        self.assertFalse(clock.update(1000))
        self.assertEqual(clock.progress, 1.0)

    def test_reset(self):
        clock = TickClock(250)
        clock.finish()
        clock.reset()
        self.assertEqual(clock.progress, 0.0)
        self.assertFalse(clock.paused)

    def test_rejects_non_positive_tick(self):
        with self.assertRaises(ValueError):
            TickClock(0)


if __name__ == "__main__":
    unittest.main()
