"""Headless checks for scenes (ui/scene.py). Pygame is not required.

The scene is what the screen shows. These tests play real seeded games with
the baseline bots and check that the scene of a state shows exactly that
state: every object on its cell, dead characters gone.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RandomBot, RuleBot  # noqa: E402
from game.environment import Action, Cell  # noqa: E402
from ui.scene import (  # noqa: E402
    LAYER_BOMB,
    LAYER_CHARACTER,
    LAYER_FLAME,
    LAYER_GATE,
    LAYER_TILE,
    initial_facings,
    static_scene,
    update_facings,
)
from ui.session import MODE_BOTS, MODE_FREE, GameSession  # noqa: E402

TILE_NAMES = {Cell.EMPTY: "floor", Cell.WALL: "wall", Cell.BLOCK: "crate"}


def played_transitions(seed: int, opponents: str = "mixed", ticks: int = 150):
    """``(prev, cur)`` pairs of a real game; the human seat is a rule bot."""
    session = GameSession(MODE_BOTS, seed, opponents)
    human = RuleBot(seed)
    pairs = []
    for _ in range(ticks):
        if session.over:
            break
        prev = session.cur
        cur = session.advance(human.act(prev, 0))
        pairs.append((prev, cur))
    return pairs


def cells(items, layer):
    return sorted((item.x // 16, item.y // 16) for item in items if item.layer == layer)


class TestStaticScene(unittest.TestCase):
    def check_matches(self, state, facings):
        items = static_scene(state, facings)
        self.assertEqual([i.layer for i in items], sorted(i.layer for i in items))
        tiles = [i for i in items if i.layer == LAYER_TILE]
        self.assertEqual(len(tiles), 121)
        for item in tiles:
            self.assertEqual(item.key[1], TILE_NAMES[state["grid"][item.y // 16][item.x // 16]])
        self.assertEqual(cells(items, LAYER_BOMB), sorted((b["x"], b["y"]) for b in state["bombs"]))
        self.assertEqual(cells(items, LAYER_FLAME), sorted(state["explosion"]))
        alive = [c for c in state["characters"] if not c["dead"]]
        self.assertEqual(cells(items, LAYER_CHARACTER), sorted((c["x"], c["y"]) for c in alive))
        drawn_ids = [i.key[1] for i in items if i.layer == LAYER_CHARACTER]
        self.assertEqual(sorted(drawn_ids), sorted(c["id"] for c in alive))
        gates = [i for i in items if i.layer == LAYER_GATE]
        if state["gate"]["state"] == "hidden":
            self.assertEqual(gates, [])
        else:
            self.assertEqual(len(gates), 1)
            self.assertEqual(gates[0].key[1], state["gate"]["state"])

    def test_matches_state_over_real_games(self):
        seen_flames = seen_deaths = seen_gate = False
        for seed in range(6):
            for opponents in ("mixed", "random"):
                pairs = played_transitions(seed, opponents)
                facings = initial_facings(pairs[0][0])
                for prev, cur in pairs:
                    facings = update_facings(facings, prev, cur)
                    self.check_matches(cur, facings)
                    seen_flames |= bool(cur["explosion"])
                    seen_deaths |= any(c["dead"] for c in cur["characters"])
                    seen_gate |= cur["gate"]["state"] != "hidden"
        self.assertTrue(seen_flames and seen_deaths and seen_gate)

    def test_free_world_scene(self):
        session = GameSession(MODE_FREE, seed=9)
        self.check_matches(session.cur, initial_facings(session.cur))

    def test_human_drawn_last(self):
        session = GameSession(MODE_BOTS, seed=1)
        items = static_scene(session.cur, initial_facings(session.cur))
        ids = [i.key[1] for i in items if i.layer == LAYER_CHARACTER]
        self.assertEqual(ids, [3, 2, 1, 0])


class TestFacing(unittest.TestCase):
    def state(self, x, y):
        return {"characters": [{"id": 0, "x": x, "y": y, "dead": False}]}

    def test_initial_facing_is_down(self):
        self.assertEqual(initial_facings(self.state(1, 1)), {0: "down"})

    def test_facing_follows_movement(self):
        start = self.state(3, 3)
        for (x, y), facing in (((4, 3), "right"), ((2, 3), "left"), ((3, 2), "up"), ((3, 4), "down")):
            with self.subTest(facing=facing):
                self.assertEqual(update_facings({0: "down"}, start, self.state(x, y)), {0: facing})

    def test_facing_kept_when_standing_still(self):
        self.assertEqual(update_facings({0: "left"}, self.state(3, 3), self.state(3, 3)), {0: "left"})

    def test_facing_from_real_step(self):
        session = GameSession(MODE_FREE, seed=0)
        prev = session.cur
        cur = session.advance(Action.RIGHT)
        if cur["player"] != prev["player"]:
            self.assertEqual(update_facings({0: "down"}, prev, cur), {0: "right"})


if __name__ == "__main__":
    unittest.main()
