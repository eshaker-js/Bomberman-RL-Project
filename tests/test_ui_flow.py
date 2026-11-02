"""Game flow checks: opponent choice, pause, restart, result, seeds.

Runs with SDL's dummy video driver. Skipped with a clear reason when pygame
is not installed.
"""

import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

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

from agents import RandomBot, RuleBot  # noqa: E402
from game.environment import NPC_IDS  # noqa: E402

NO_PYGAME = "pygame is not installed (pip install -r requirements.txt); UI flow tests skipped"
UP, DOWN, ENTER, ESC, SPACE, RIGHT = (
    (pygame.K_UP, pygame.K_DOWN, pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RIGHT)
    if pygame else (None,) * 6
)


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


@unittest.skipIf(pygame is None, NO_PYGAME)
class FlowTestCase(unittest.TestCase):
    SEED_RNG = 21

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
        self.app = App(self.window, random.Random(self.SEED_RNG))

    def frame(self, keys=(), dt=16):
        self.app.frame([key(k) for k in keys], dt)

    def skip_ready(self):
        from ui.config import READY_MS
        self.frame(dt=READY_MS)

    def free_world(self):
        self.frame([ENTER], dt=0)
        self.skip_ready()
        return self.app.screen

    def lose(self, screen):
        """Drop a bomb and stand on it until the game is over and finished."""
        self.frame([SPACE], dt=250)
        for _ in range(12):
            self.frame(dt=250)
        self.assertTrue(screen.finished)


class TestOpponentScreen(FlowTestCase):
    def test_player_vs_bots_opens_opponent_choice(self):
        from ui.screens import OpponentScreen
        self.frame([DOWN, ENTER])
        self.assertIsInstance(self.app.screen, OpponentScreen)
        self.assertEqual(self.app.screen.menu.options, ["RANDOM", "RULE", "MIXED"])

    def test_each_choice_creates_its_bots(self):
        expected = {
            0: [RandomBot, RandomBot, RandomBot],
            1: [RuleBot, RuleBot, RuleBot],
            2: [RuleBot, RuleBot, RandomBot],
        }
        for index, classes in expected.items():
            with self.subTest(index=index):
                self.setUp()
                self.frame([DOWN, ENTER] + [DOWN] * index + [ENTER])
                bots = self.app.screen.session.bots
                self.assertEqual([type(bots[i]) for i in NPC_IDS], classes)
                self.assertTrue(self.app.screen.session.env.with_npcs)

    def test_escape_goes_back(self):
        from ui.screens import StartScreen
        self.frame([DOWN, ENTER])
        self.frame([ESC])
        self.assertIsInstance(self.app.screen, StartScreen)
        self.assertTrue(self.app.running)


class TestPause(FlowTestCase):
    def test_escape_pauses_and_freezes_the_game(self):
        screen = self.free_world()
        self.frame(dt=250)
        self.assertEqual(screen.session.cur["steps"], 1)
        self.frame([ESC])
        self.assertTrue(screen.paused)
        for _ in range(10):
            self.frame(dt=250)
        self.assertEqual(screen.session.cur["steps"], 1)
        self.frame([ESC])
        self.assertFalse(screen.paused)
        self.frame(dt=250)
        self.assertEqual(screen.session.cur["steps"], 2)

    def test_pause_during_ready_freezes_the_countdown(self):
        from ui.config import READY_MS
        self.frame([ENTER], dt=0)
        screen = self.app.screen
        self.frame([ESC])
        self.frame(dt=READY_MS * 3)
        self.assertGreater(screen.ready_ms, 0)
        self.frame([ENTER])  # RESUME
        self.assertFalse(screen.paused)

    def test_keys_in_pause_menu_do_not_reach_the_game(self):
        screen = self.free_world()
        self.frame([ESC, DOWN, UP, SPACE])
        self.frame([ESC])
        self.frame(dt=250)
        self.assertEqual(screen.session.cur["bombs"], [])

    def test_pause_restart_keeps_the_seed(self):
        from ui.session import GameSession, MODE_FREE
        screen = self.free_world()
        seed = screen.session.seed
        self.frame([SPACE], dt=250)
        self.frame(dt=250)
        self.frame([ESC, DOWN, ENTER])
        self.assertIs(self.app.screen, screen)
        self.assertFalse(screen.paused)
        self.assertEqual(screen.session.seed, seed)
        self.assertEqual(screen.session.cur, GameSession(MODE_FREE, seed).cur)
        self.assertGreater(screen.ready_ms, 0)

    def test_pause_menu_returns_to_start(self):
        from ui.screens import StartScreen
        self.free_world()
        self.frame([ESC, DOWN, DOWN, ENTER])
        self.assertIsInstance(self.app.screen, StartScreen)


class TestRestartReproduces(FlowTestCase):
    def play(self, screen, keys):
        states = []
        for k in keys:
            self.frame([k] if k else [], dt=250)
            states.append(screen.session.cur)
        return states

    def test_restart_replays_the_same_game_with_bots(self):
        self.frame([DOWN, ENTER, DOWN, DOWN, ENTER], dt=0)  # mixed bots
        screen = self.app.screen
        self.skip_ready()
        keys = [RIGHT, None, SPACE, None, None, None, None, None] * 5
        first = self.play(screen, keys)
        if screen.finished:
            self.frame([ENTER], dt=0)  # result dialog: RESTART
        else:
            self.frame([ESC, DOWN, ENTER], dt=0)  # pause menu: RESTART
        self.assertIs(self.app.screen, screen)
        self.assertEqual(screen.session.cur["steps"], 0)
        self.skip_ready()
        second = self.play(screen, keys)
        self.assertEqual(first, second)


class TestResult(FlowTestCase):
    def test_result_dialog_after_final_animation(self):
        screen = self.free_world()
        self.lose(screen)
        self.assertEqual(screen.session.cur["outcome"], "loss")
        self.assertEqual(screen.result_menu.options, ["RESTART", "MENU"])
        self.frame()

    def test_result_restart_same_seed(self):
        screen = self.free_world()
        seed = screen.session.seed
        self.lose(screen)
        self.frame([ENTER])
        self.assertIs(self.app.screen, screen)
        self.assertFalse(screen.finished)
        self.assertEqual(screen.session.seed, seed)
        self.assertEqual(screen.session.cur["steps"], 0)
        self.assertIsNone(screen.session.cur["outcome"])

    def test_result_menu_and_escape_go_back(self):
        from ui.screens import StartScreen
        for keys in ([DOWN, ENTER], [ESC]):
            with self.subTest(keys=keys):
                self.setUp()
                screen = self.free_world()
                self.lose(screen)
                self.frame(keys)
                self.assertIsInstance(self.app.screen, StartScreen)
                self.assertTrue(self.app.running)

    def test_every_outcome_has_a_result_title(self):
        from ui import render
        screen = self.free_world()
        self.lose(screen)
        for outcome, title in (("win", "WIN!"), ("loss", "LOSS"), ("draw", "DRAW")):
            with self.subTest(outcome=outcome):
                screen.session.cur = dict(screen.session.cur, outcome=outcome)
                with mock.patch.object(render, "draw_text_centered",
                                       wraps=render.draw_text_centered) as spy:
                    screen.draw(self.app.canvas)
                texts = [c.args[1] for c in spy.call_args_list]
                self.assertIn(title, texts)


class TestSeeds(FlowTestCase):
    def test_seeds_come_from_the_ui_rng(self):
        from ui.config import SEED_COUNT
        expected_rng = random.Random(self.SEED_RNG)
        expected = [expected_rng.randrange(SEED_COUNT) for _ in range(3)]
        got = []
        for _ in range(3):
            self.frame([ENTER])
            got.append(self.app.screen.session.seed)
            self.frame([ESC, DOWN, DOWN, ENTER])
        self.assertEqual(got, expected)

    def test_seed_shown_in_panel(self):
        from ui import render
        screen = self.free_world()
        with mock.patch.object(render, "draw_text", wraps=render.draw_text) as spy:
            screen.draw(self.app.canvas)
        texts = [c.args[1] for c in spy.call_args_list]
        self.assertIn(f"SEED {screen.session.seed}", texts)

    def test_global_random_state_untouched(self):
        state = random.getstate()
        self.frame([ENTER])
        self.skip_ready()
        self.frame(dt=250)
        self.assertEqual(random.getstate(), state)


class TestScreensDraw(FlowTestCase):
    def test_pause_result_and_opponent_screens_draw(self):
        self.frame([DOWN, ENTER])
        for _ in range(3):
            self.frame([DOWN])
        self.frame([ESC])
        screen = self.free_world()
        self.frame([ESC])
        self.frame()
        self.frame([ESC])
        self.lose(screen)
        self.frame()


if __name__ == "__main__":
    unittest.main()
