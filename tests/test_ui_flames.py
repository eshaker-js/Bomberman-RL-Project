"""Headless checks for the UI with the real game's timing (GAME_RULES).

Every mode plays with ``GAME_RULES``; flames are drawn from
``state["flames"]``, fading as ``ticks_left`` drops, and the screen still
matches the state exactly at the end of every tick, lingering flames
included. Pygame is only needed for the sprite-building test, which is
skipped with a clear reason without it.
"""

import functools
import os
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

from agents import RuleBot  # noqa: E402
from game.environment import GAME_RULES  # noqa: E402
from ui import art  # noqa: E402
from ui.scene import (  # noqa: E402
    LAYER_FLAME,
    flame_fade_level,
    initial_facings,
    scene,
    static_scene,
    update_facings,
)
from ui.session import MODE_BOTS, MODE_FREE, GameSession  # noqa: E402

NO_PYGAME = "pygame is not installed (pip install -r requirements.txt); UI sprite test skipped"
T_VALUES = (0.0, 0.1, 0.3, 0.5, 0.8, 0.99, 1.0)


@functools.lru_cache(maxsize=1)
def games():
    """Real GAME_RULES games: rule bot in the human seat, mixed and random NPCs."""
    result = []
    for seed in range(6):
        for opponents in ("mixed", "random"):
            session = GameSession(MODE_BOTS, seed, opponents)
            human = RuleBot(seed)
            pairs = []
            while not session.over and session.cur["steps"] < 200:
                prev = session.cur
                pairs.append((prev, session.advance(human.act(prev, 0))))
            result.append(tuple(pairs))
    return tuple(result)


def flame_items(items):
    return {(i.x // 16, i.y // 16): i.key for i in items if i.layer == LAYER_FLAME}


class TestModesUseGameRules(unittest.TestCase):
    def test_every_mode_plays_game_rules(self):
        expected = {"bomb_fuse": GAME_RULES["bomb_fuse"], "blast_range": 2, "flame_ticks": GAME_RULES["flame_ticks"]}
        for session in (GameSession(MODE_FREE, 1), GameSession(MODE_BOTS, 1, "random"),
                        GameSession(MODE_BOTS, 1, "rule"), GameSession(MODE_BOTS, 1, "mixed")):
            with self.subTest(mode=session.mode, opponents=session.opponents):
                self.assertEqual(session.env.bomb_fuse, GAME_RULES["bomb_fuse"])
                self.assertEqual(session.env.flame_ticks, GAME_RULES["flame_ticks"])
                self.assertEqual(session.cur["rules"], expected)


class TestFadeLevels(unittest.TestCase):
    def test_levels_follow_ticks_left(self):
        self.assertEqual([flame_fade_level(k, 4) for k in (4, 3, 2, 1)], [0, 1, 2, 3])
        self.assertEqual(flame_fade_level(1, 1), 0)
        self.assertEqual(flame_fade_level(1, 9), art.FLAME_FADE_LEVELS)

    def test_fade_sprites_are_recoloured_steady_flames(self):
        for level in range(1, art.FLAME_FADE_LEVELS + 1):
            faded = art.flame_fade(level)
            art.validate(f"fade{level}", faded, (16, 16))
            shape = [[ch == "." for ch in row] for row in faded]
            self.assertEqual(shape, [[ch == "." for ch in row] for row in art.FLAME_STEADY])
            self.assertNotEqual(faded, art.FLAME_STEADY)


class TestArt(unittest.TestCase):
    def test_floor_is_dark_grey_with_black_speckles(self):
        self.assertEqual(set("".join(art.FLOOR)), {"5", "0"})
        self.assertGreater("".join(art.FLOOR).count("5"), 200)

    def test_closed_gate_has_a_black_border(self):
        rows = art.GATE_CLOSED
        border = rows[0] + rows[-1] + "".join(r[0] + r[-1] for r in rows)
        self.assertEqual(set(border), {"0"})


class TestFlamesOnScreen(unittest.TestCase):
    def test_flames_drawn_from_state_with_fade_levels(self):
        levels_seen = set()
        for pairs in games():
            facings = initial_facings(pairs[0][0])
            for prev, cur in pairs:
                facings = update_facings(facings, prev, cur)
                drawn = flame_items(static_scene(cur, facings))
                expected = {}
                for f in cur["flames"]:
                    level = flame_fade_level(f["ticks_left"], 4)
                    levels_seen.add(level)
                    expected[(f["x"], f["y"])] = (
                        ("flame", "steady") if level == 0 else ("flame", "fade", level))
                self.assertEqual(drawn, expected)
        self.assertEqual(levels_seen, {0, 1, 2, 3})

    def test_lingering_flames_keep_their_level_all_tick_and_fresh_ones_grow(self):
        for pairs in games():
            facings = initial_facings(pairs[0][0])
            for prev, cur in pairs:
                fresh = set(cur["explosion"])
                for t in T_VALUES:
                    drawn = flame_items(scene(prev, cur, t, facings))
                    self.assertEqual(set(drawn), {(f["x"], f["y"]) for f in cur["flames"]})
                    for f in cur["flames"]:
                        cell = (f["x"], f["y"])
                        if cell not in fresh:
                            level = flame_fade_level(f["ticks_left"], 4)
                            self.assertEqual(drawn[cell], ("flame", "steady") if level == 0 else ("flame", "fade", level))
                        elif t == 0.0:
                            self.assertEqual(drawn[cell], ("flame", 0))

    def test_expired_flames_vanish_at_tick_start(self):
        found = 0
        for pairs in games():
            facings = initial_facings(pairs[0][0])
            for prev, cur in pairs:
                gone = ({(f["x"], f["y"]) for f in prev["flames"]}
                        - {(f["x"], f["y"]) for f in cur["flames"]})
                if gone:
                    found += 1
                    drawn = flame_items(scene(prev, cur, 0.0, facings))
                    self.assertFalse(gone & set(drawn))
        self.assertGreater(found, 10)

    def test_end_of_tick_matches_state_with_lingering_flames_and_deaths(self):
        lingering_ticks = lingering_deaths = 0
        for pairs in games():
            facings = initial_facings(pairs[0][0])
            for prev, cur in pairs:
                facings = update_facings(facings, prev, cur)
                self.assertEqual(scene(prev, cur, 1.0, facings), static_scene(cur, facings))
                lingering_ticks += any(f["ticks_left"] < 4 for f in cur["flames"])
                for event in cur["events"]:
                    if event["type"] == "death":
                        c = cur["characters"][event["character"]]
                        lingering_deaths += (c["x"], c["y"]) not in cur["explosion"]
        self.assertGreater(lingering_ticks, 100)
        self.assertGreater(lingering_deaths, 0)


@unittest.skipIf(pygame is None, NO_PYGAME)
class TestFlameSprites(unittest.TestCase):
    def test_every_flame_key_builds(self):
        pygame.display.init()
        try:
            pygame.display.set_mode((16, 16))
            from ui.sprites import Sprites
            sprites = Sprites()
            keys = [("flame", 0), ("flame", 1), ("flame", "steady"), ("tile", "floor"),
                    ("gate", "closed", 0)]
            keys += [("flame", "fade", level) for level in range(1, art.FLAME_FADE_LEVELS + 1)]
            for key in keys:
                self.assertEqual(sprites.get(key).get_size(), (16, 16))
        finally:
            pygame.display.quit()


if __name__ == "__main__":
    unittest.main()
