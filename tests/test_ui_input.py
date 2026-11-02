"""Checks for keyboard input (ui/input.py). Needs pygame for its key codes only.

Skipped with a clear reason when pygame is not installed.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:
    pygame = None

from game.environment import Action  # noqa: E402

NO_PYGAME = "pygame is not installed (pip install -r requirements.txt); UI input tests skipped"


@unittest.skipIf(pygame is None, NO_PYGAME)
class TestKeyMap(unittest.TestCase):
    def setUp(self):
        from ui.input import KEY_TO_ACTION
        self.keys = KEY_TO_ACTION

    def test_arrows_and_wasd_move(self):
        expected = {
            pygame.K_UP: Action.UP, pygame.K_w: Action.UP,
            pygame.K_DOWN: Action.DOWN, pygame.K_s: Action.DOWN,
            pygame.K_LEFT: Action.LEFT, pygame.K_a: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT, pygame.K_d: Action.RIGHT,
        }
        for key, action in expected.items():
            self.assertIs(self.keys[key], action)

    def test_space_bombs(self):
        self.assertIs(self.keys[pygame.K_SPACE], Action.BOMB)

    def test_escape_and_enter_are_not_actions(self):
        for key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_n, pygame.K_r):
            self.assertNotIn(key, self.keys)


@unittest.skipIf(pygame is None, NO_PYGAME)
class TestInputBuffer(unittest.TestCase):
    def setUp(self):
        from ui.input import InputBuffer
        self.buffer = InputBuffer()

    def press(self, key):
        return self.buffer.key_down(key)

    def test_no_key_means_wait(self):
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_last_key_wins(self):
        self.press(pygame.K_LEFT)
        self.press(pygame.K_SPACE)
        self.press(pygame.K_UP)
        self.assertIs(self.buffer.consume(), Action.UP)

    def test_bomb_after_move_wins(self):
        self.press(pygame.K_RIGHT)
        self.buffer.key_up(pygame.K_RIGHT)
        self.press(pygame.K_SPACE)
        self.assertIs(self.buffer.consume(), Action.BOMB)

    def test_pressed_key_used_once(self):
        self.press(pygame.K_SPACE)
        self.buffer.key_up(pygame.K_SPACE)
        self.assertIs(self.buffer.consume(), Action.BOMB)
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_pressed_and_released_within_a_tick_still_counts(self):
        self.press(pygame.K_d)
        self.buffer.key_up(pygame.K_d)
        self.assertIs(self.buffer.consume(), Action.RIGHT)
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_held_move_key_repeats_every_tick(self):
        self.press(pygame.K_DOWN)
        self.assertIs(self.buffer.consume(), Action.DOWN)
        self.assertIs(self.buffer.consume(), Action.DOWN)
        self.buffer.key_up(pygame.K_DOWN)
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_held_space_does_not_repeat(self):
        self.press(pygame.K_SPACE)
        self.assertIs(self.buffer.consume(), Action.BOMB)
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_new_press_beats_held_key(self):
        self.press(pygame.K_LEFT)
        self.buffer.consume()
        self.press(pygame.K_SPACE)
        self.assertIs(self.buffer.consume(), Action.BOMB)
        self.assertIs(self.buffer.consume(), Action.LEFT)

    def test_most_recent_held_key_wins(self):
        self.press(pygame.K_LEFT)
        self.press(pygame.K_UP)
        self.buffer.consume()
        self.assertIs(self.buffer.consume(), Action.UP)
        self.buffer.key_up(pygame.K_UP)
        self.assertIs(self.buffer.consume(), Action.LEFT)

    def test_unmapped_key_ignored(self):
        self.assertFalse(self.press(pygame.K_ESCAPE))
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_release_all_after_focus_loss(self):
        self.press(pygame.K_RIGHT)
        self.buffer.consume()
        self.buffer.release_all()
        self.assertIs(self.buffer.consume(), Action.WAIT)

    def test_clear(self):
        self.press(pygame.K_RIGHT)
        self.buffer.clear()
        self.assertIs(self.buffer.consume(), Action.WAIT)


if __name__ == "__main__":
    unittest.main()
