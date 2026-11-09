"""Buffs in the UI: floor badges, pop/sparkle/puff effects, ember flames, level pips (Task 7).

The scene tests are headless; the sprite and panel tests need pygame
(SDL dummy driver) and skip without it.
"""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

try:
    import pygame
except ImportError:  # pragma: no cover - depends on the installed packages
    pygame = None

from game.environment import GAME_RULES, Action, Bomb, BombermanEnv, BuffRules, BuffType
from ui import art
from ui.scene import (
    BUFF_POP,
    BUFF_PUFF_END,
    BUFF_SPARKLE_END,
    LAYER_BOMB,
    LAYER_BUFF,
    LAYER_EFFECT,
    LAYER_FLAME,
    bomb_pulses_per_tick,
    buff_frame,
    flame_item_key,
    initial_facings,
    scene,
    static_scene,
)

NO_PYGAME = "pygame is not installed"
FLAMES = GAME_RULES["flame_ticks"]
# Every destroyed crate drops a burn buff: deterministic drops for the tests.
ALWAYS_BURN = BuffRules((BuffType("range", 2, 5, 0.0), BuffType("bomb_count", 1, 3, 0.0),
                         BuffType("burn", 0, 2, 1.0)))


def env_with(**kwargs) -> BombermanEnv:
    rules = {**GAME_RULES, **kwargs}
    return BombermanEnv(with_npcs=True, **rules)


def keys(items, layer):
    return {(i.x // 16, i.y // 16): i.key for i in items if i.layer == layer}


def drop_transition():
    """(prev, cur) of a tick where the crate at (5, 1) drops a burn buff."""
    env = env_with(buffs=ALWAYS_BURN)
    env.player_x, env.player_y = 1, 3
    env.bomb = Bomb(3, 1, 1)
    prev = env.get_state()
    return env, prev, env.step(Action.WAIT)


class TestBuffScene(unittest.TestCase):
    def test_floor_buffs_are_drawn_where_the_state_has_them(self):
        env = env_with()
        env.floor_buffs.update({(2, 1): "range", (3, 1): "bomb_count", (1, 2): "burn"})
        state = env.get_state()
        drawn = keys(static_scene(state, initial_facings(state)), LAYER_BUFF)
        self.assertEqual({cell: key[:2] for cell, key in drawn.items()},
                         {(2, 1): ("buff", "range"), (3, 1): ("buff", "bomb_count"),
                          (1, 2): ("buff", "burn")})

    def test_bob_comes_from_the_state_only(self):
        env = env_with()
        env.floor_buffs[(3, 1)] = "range"
        state = env.get_state()
        facings = initial_facings(state)
        for t in (0.0, 0.3, 0.7, 1.0):  # the same all through a tick
            self.assertEqual(scene(state, state, t, facings), static_scene(state, facings))
        later = env.step(Action.WAIT)
        self.assertNotEqual(buff_frame(state["steps"], 3, 1), buff_frame(later["steps"], 3, 1))
        self.assertEqual(keys(static_scene(later, facings), LAYER_BUFF)[(3, 1)],
                         ("buff", "range", buff_frame(later["steps"], 3, 1)))

    def test_dropped_buff_pops_in(self):
        _, prev, cur = drop_transition()
        self.assertEqual(cur["buffs"], [{"x": 5, "y": 1, "type": "burn"}])
        facings = initial_facings(cur)
        start = scene(prev, cur, 0.0, facings)
        self.assertNotIn((5, 1), keys(start, LAYER_BUFF))
        self.assertEqual(keys(start, LAYER_EFFECT)[(5, 1)], ("buff_fx", "pop", 0))
        middle = scene(prev, cur, BUFF_POP[0], facings)
        self.assertIn((5, 1), keys(middle, LAYER_BUFF))
        self.assertEqual(keys(middle, LAYER_EFFECT)[(5, 1)], ("buff_fx", "pop", 1))
        after = scene(prev, cur, BUFF_POP[1], facings)
        self.assertNotIn(("buff_fx", "pop", 1), keys(after, LAYER_EFFECT).values())
        self.assertEqual(scene(prev, cur, 1.0, facings), static_scene(cur, facings))

    def test_pickup_sparkles(self):
        env = env_with()
        env.floor_buffs[(2, 1)] = "range"
        prev = env.get_state()
        cur = env.step(Action.RIGHT)
        self.assertEqual(cur["buffs"], [])
        facings = initial_facings(cur)
        self.assertEqual(keys(scene(prev, cur, 0.1, facings), LAYER_EFFECT)[(2, 1)],
                         ("buff_fx", "sparkle", 0))
        self.assertEqual(keys(scene(prev, cur, BUFF_SPARKLE_END * 0.75, facings),
                              LAYER_EFFECT)[(2, 1)], ("buff_fx", "sparkle", 1))
        self.assertNotIn((2, 1), keys(scene(prev, cur, BUFF_SPARKLE_END, facings), LAYER_EFFECT))
        self.assertNotIn((2, 1), keys(scene(prev, cur, 0.1, facings), LAYER_BUFF))
        self.assertEqual(scene(prev, cur, 1.0, facings), static_scene(cur, facings))

    def test_destroyed_buff_puffs(self):
        env = env_with()
        env.player_x, env.player_y = 1, 3
        env.floor_buffs[(2, 1)] = "bomb_count"
        env.bomb = Bomb(1, 1, 1)
        prev = env.get_state()
        cur = env.step(Action.WAIT)
        facings = initial_facings(cur)
        self.assertEqual(keys(scene(prev, cur, 0.0, facings), LAYER_EFFECT)[(2, 1)],
                         ("buff_fx", "puff", 0))
        self.assertNotIn((2, 1), keys(scene(prev, cur, BUFF_PUFF_END, facings), LAYER_EFFECT))
        self.assertNotIn((2, 1), keys(scene(prev, cur, 0.0, facings), LAYER_BUFF))

    def test_burn_cells_are_embers_until_their_extra_ticks_are_spent(self):
        env = env_with(start_buffs={0: {"burn": 1}})
        env.bomb = Bomb(3, 1, 1, burn=1)
        env.player_x, env.player_y = 1, 3
        prev = env.get_state()
        cur = env.step(Action.WAIT)
        facings = initial_facings(cur)
        start = keys(scene(prev, cur, 0.0, facings), LAYER_FLAME)
        self.assertEqual(start[(3, 1)], ("flame", "ember", 0))  # distance 0: 5 ticks
        self.assertEqual(start[(4, 1)], ("flame", "ember", 0))  # distance 1
        self.assertEqual(start[(5, 1)], ("flame", 0))  # distance 2: normal
        drawn = keys(static_scene(cur, facings), LAYER_FLAME)
        self.assertEqual(drawn[(3, 1)], ("flame", "ember", "steady"))
        later = env.step(Action.WAIT)  # 4 ticks left: like a fresh normal flame
        self.assertEqual(keys(static_scene(later, facings), LAYER_FLAME)[(3, 1)],
                         ("flame", "steady"))
        self.assertEqual(keys(static_scene(later, facings), LAYER_FLAME)[(5, 1)],
                         ("flame", "fade", 1))

    def test_flame_item_key(self):
        self.assertEqual(flame_item_key(6, 4), ("flame", "ember", "steady"))
        self.assertEqual(flame_item_key(5, 4, 1), ("flame", "ember", 1))
        self.assertEqual(flame_item_key(4, 4, 0), ("flame", 0))
        self.assertEqual(flame_item_key(2, 4), ("flame", "fade", 2))

    def test_several_bombs_of_one_character(self):
        env = env_with(start_buffs={0: {"bomb_count": 3}}, bomb_fuse=9)  # none explodes yet
        prev = env.get_state()
        for action in (Action.BOMB, Action.DOWN, Action.BOMB, Action.DOWN, Action.BOMB):
            prev, cur = env.get_state(), env.step(action)
        self.assertEqual(len(cur["bombs"]), 3)
        facings = initial_facings(cur)
        for items in (static_scene(cur, facings), scene(prev, cur, 0.4, facings)):
            self.assertEqual(sorted(keys(items, LAYER_BOMB)), [(1, 1), (1, 2), (1, 3)])

    def test_bomb_pulse_uses_the_game_fuse(self):
        self.assertEqual(bomb_pulses_per_tick(9, fuse=9), 1)
        self.assertEqual(bomb_pulses_per_tick(1, fuse=9), 9)
        self.assertEqual(bomb_pulses_per_tick(4, fuse=4), 1)


class TestBuffArt(unittest.TestCase):
    def test_three_different_icons_with_a_black_rim(self):
        icons = art.BUFF_ICONS
        self.assertEqual(set(icons), {"range", "bomb_count", "burn"})
        grids = list(icons.values())
        for i, a in enumerate(grids):
            for b in grids[i + 1:]:
                self.assertNotEqual(a, b)
        for name, rows in icons.items():
            with self.subTest(name=name):
                art.validate(name, rows, (16, 16))
                self.assertEqual(set(rows[1].replace(".", "")), {"0"})  # dark rim on top
                self.assertNotIn("5", "".join(rows))  # never the floor's grey

    def test_ember_differs_from_every_flame_and_fade(self):
        others = [art.FLAME_STEADY] + [art.flame_fade(level)
                                       for level in range(1, art.FLAME_FADE_LEVELS + 1)]
        for ember in art.EMBER_FRAMES:
            self.assertNotIn(ember, others)
        self.assertEqual([[ch == "." for ch in row] for row in art.EMBER_FRAMES[2]],
                         [[ch == "." for ch in row] for row in art.FLAME_STEADY])


@unittest.skipIf(pygame is None, NO_PYGAME)
class TestBuffSpritesAndPanel(unittest.TestCase):
    def setUp(self):
        pygame.display.init()
        pygame.display.set_mode((16, 16))
        from ui.sprites import Sprites
        self.sprites = Sprites()

    def tearDown(self):
        pygame.display.quit()

    def test_every_new_key_builds(self):
        keys_ = [("flame", "ember", frame) for frame in (0, 1, "steady")]
        keys_ += [("buff", kind, frame) for kind in art.BUFF_ICONS for frame in (0, 1)]
        keys_ += [("buff_fx", kind, frame) for kind in ("pop", "sparkle", "puff")
                  for frame in (0, 1)]
        for key in keys_:
            self.assertEqual(self.sprites.get(key).get_size(), (16, 16), key)
        for name in ("bomb", "range", "burn"):
            for lit in (True, False):
                self.assertEqual(self.sprites.get(("icon", name, lit)).get_size(), (7, 7))

    def test_panel_pips_show_levels(self):
        from ui import render
        from ui.config import CANVAS_SIZE, PANEL_X, PANEL_Y
        from ui.palette import rgb
        from ui.session import MODE_BOTS, GameSession
        session = GameSession(MODE_BOTS, 1, "random")
        session.env.reset(seed=1, start_buffs={0: {"range": 4, "burn": 2, "bomb_count": 3}})
        state = session.env.get_state()
        self.assertEqual(render.level_pips(state["characters"][0], state["buff_rules"], "range"),
                         [True, True, True, True, False])
        canvas = pygame.Surface(CANVAS_SIZE)
        render.draw_panel(canvas, self.sprites, session, state)
        tx, y = PANEL_X + 3 + 22, PANEL_Y + 11
        # Range: pips 1-4 lit blue, pip 5 grey. Burn: both lit orange.
        pip = lambda dx, level, dy: canvas.get_at(
            (tx + dx + 9 + (level - 1) * render.PIP_STEP + 1, y + dy + 3))[:3]
        self.assertEqual(pip(0, 4, 10), rgb(render.PIP_COLOR["range"]))
        self.assertEqual(pip(0, 5, 10), rgb(5))
        self.assertEqual(pip(31, 2, 10), rgb(render.PIP_COLOR["burn"]))
        self.assertEqual(pip(0, 3, 17), rgb(render.PIP_COLOR["bomb_count"]))


if __name__ == "__main__":
    unittest.main()
