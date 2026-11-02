"""Smoke tests: open and draw every screen with SDL's dummy video driver.

These only prove that screens build, react to keys and draw without errors;
they cannot check how anything looks. Skipped with a clear reason when
pygame is not installed.
"""

import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:
    pygame = None

from game.environment import AGENT_ID  # noqa: E402

NO_PYGAME = "pygame is not installed (pip install -r requirements.txt); UI smoke tests skipped"


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def key_up(k):
    return pygame.event.Event(pygame.KEYUP, key=k)


@unittest.skipIf(pygame is None, NO_PYGAME)
class UITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.display.init()
        from ui.config import WINDOW_SIZE
        cls.window = pygame.display.set_mode(WINDOW_SIZE)

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()

    def setUp(self):
        from ui.app import App
        self.app = App(self.window, random.Random(11))

    def frame(self, events=(), dt=16):
        self.app.frame(list(events), dt)

    def start_game(self, presses):
        """From the start screen, press keys, then skip READY/GO."""
        from ui.config import READY_MS
        self.frame([key(k) for k in presses])
        self.frame(dt=READY_MS)


class TestStartScreens(UITestCase):
    def test_start_screen_draws(self):
        from ui.screens import StartScreen
        self.frame()
        self.assertIsInstance(self.app.screen, StartScreen)
        self.assertTrue(self.app.running)

    def test_escape_on_start_screen_quits(self):
        self.frame([key(pygame.K_ESCAPE)])
        self.assertFalse(self.app.running)

    def test_window_close_quits(self):
        self.frame([pygame.event.Event(pygame.QUIT)])
        self.assertFalse(self.app.running)

    def test_not_ready_page_and_back(self):
        from ui.screens import NotReadyScreen, StartScreen
        self.frame([key(pygame.K_DOWN), key(pygame.K_DOWN), key(pygame.K_RETURN)])
        self.assertIsInstance(self.app.screen, NotReadyScreen)
        self.frame()
        self.frame([key(pygame.K_RETURN)])
        self.assertIsInstance(self.app.screen, StartScreen)
        self.frame([key(pygame.K_UP), key(pygame.K_RETURN)])
        self.assertIsInstance(self.app.screen, NotReadyScreen)
        self.frame([key(pygame.K_ESCAPE)])
        self.assertIsInstance(self.app.screen, StartScreen)


class TestGameScreen(UITestCase):
    def test_free_world_runs_and_draws(self):
        from ui.screens import GameScreen
        from ui.session import MODE_FREE
        self.frame([key(pygame.K_RETURN)])
        screen = self.app.screen
        self.assertIsInstance(screen, GameScreen)
        self.assertEqual(screen.session.mode, MODE_FREE)
        self.frame(dt=500)
        self.frame(dt=600)
        for _ in range(10):
            self.frame(dt=250)
        self.assertEqual(screen.session.cur["steps"], 10)

    def test_player_vs_bots_runs_and_draws(self):
        from ui.session import MODE_BOTS
        self.start_game([pygame.K_DOWN, pygame.K_RETURN])
        screen = self.app.screen
        self.assertEqual(screen.session.mode, MODE_BOTS)
        for _ in range(40):
            self.frame(dt=250)
        self.assertEqual(len(screen.session.cur["characters"]), 4)

    def test_no_tick_before_ready_go_ends(self):
        from ui.config import READY_MS, TICK_MS
        self.frame([key(pygame.K_RETURN)], dt=0)
        screen = self.app.screen
        self.frame(dt=READY_MS - 1)
        self.frame(dt=TICK_MS)
        self.assertEqual(screen.session.cur["steps"], 0)
        self.frame(dt=1)
        self.frame(dt=TICK_MS)
        self.assertEqual(screen.session.cur["steps"], 1)

    def test_one_tick_per_tick_ms(self):
        self.start_game([pygame.K_RETURN])
        screen = self.app.screen
        for _ in range(15):
            self.frame(dt=50)
        self.assertEqual(screen.session.cur["steps"], 3)

    def test_keys_reach_the_engine(self):
        self.start_game([pygame.K_RETURN])
        screen = self.app.screen
        start = screen.session.cur["player"]
        self.frame([key(pygame.K_SPACE)], dt=250)
        self.assertEqual(len(screen.session.cur["bombs"]), 1)
        self.assertEqual(screen.session.cur["player"], start)

    def test_held_key_keeps_walking(self):
        self.start_game([pygame.K_RETURN])
        session = self.app.screen.session
        session.env.grid[1][2] = session.env.grid[1][3] = session.env.grid[1][1]  # test-only: clear the path
        self.frame([key(pygame.K_RIGHT)], dt=250)
        self.frame(dt=250)
        self.assertEqual(session.cur["player"], (3, 1))
        self.frame([key_up(pygame.K_RIGHT)], dt=250)
        self.assertEqual(session.cur["player"], (3, 1))

    def test_escape_returns_to_start(self):
        from ui.screens import StartScreen
        self.start_game([pygame.K_RETURN])
        self.frame([key(pygame.K_ESCAPE)])
        self.assertIsInstance(self.app.screen, StartScreen)

    def test_game_stops_at_outcome_and_shows_it(self):
        self.start_game([pygame.K_RETURN])
        screen = self.app.screen
        self.frame([key(pygame.K_SPACE)], dt=250)
        for _ in range(10):
            self.frame(dt=250)
        self.assertEqual(screen.session.cur["outcome"], "loss")
        steps = screen.session.cur["steps"]
        self.frame(dt=250)
        self.assertEqual(screen.session.cur["steps"], steps)
        self.assertTrue(screen.session.cur["characters"][AGENT_ID]["dead"])

    def test_every_new_game_gets_a_new_seed(self):
        from ui.config import SEED_COUNT
        seeds = []
        for _ in range(3):
            self.frame([key(pygame.K_RETURN)])
            seeds.append(self.app.screen.session.seed)
            self.frame([key(pygame.K_ESCAPE)])
        self.assertEqual(len(set(seeds)), 3)
        self.assertTrue(all(0 <= s < SEED_COUNT for s in seeds))


if __name__ == "__main__":
    unittest.main()
