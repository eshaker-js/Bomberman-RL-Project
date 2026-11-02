"""Headless checks for scenes (ui/scene.py). Pygame is not required.

The scene is what the screen shows. These tests play real seeded games with
the baseline bots and check that the scene of a state shows exactly that
state: every object on its cell, dead characters gone.
"""

import functools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RandomBot, RuleBot  # noqa: E402
from game.environment import Action, Cell  # noqa: E402
from ui.scene import (  # noqa: E402
    CRATE_BREAK_END,
    GATE_BLINK_END,
    LAYER_BOMB,
    LAYER_CHARACTER,
    LAYER_EFFECT,
    LAYER_FLAME,
    LAYER_GATE,
    LAYER_TILE,
    bomb_frame,
    bomb_pulses_per_tick,
    broken_crates,
    death_frame,
    died_this_tick,
    flame_frame,
    gate_changed,
    initial_facings,
    lerp,
    scene,
    slide,
    static_scene,
    update_facings,
    walk_frame,
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



T_VALUES = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99)


@functools.lru_cache(maxsize=1)
def real_games():
    """Transitions of several seeded games with deaths, chains and gates (played once)."""
    return tuple(
        played_transitions(seed, opponents, ticks=200)
        for seed in range(8)
        for opponents in ("mixed", "random", "rule")
    )


class TestInterpolationMath(unittest.TestCase):
    def test_lerp(self):
        self.assertEqual(lerp(0, 16, 0.0), 0)
        self.assertEqual(lerp(0, 16, 0.5), 8)
        self.assertEqual(lerp(0, 16, 1.0), 16)
        self.assertEqual(lerp(32, 16, 0.25), 28)

    def test_slide_between_cells(self):
        self.assertEqual(slide((1, 1), (2, 1), 0.0), (16, 16))
        self.assertEqual(slide((1, 1), (2, 1), 0.5), (24, 16))
        self.assertEqual(slide((1, 1), (2, 1), 1.0), (32, 16))
        self.assertEqual(slide((3, 3), (3, 2), 0.25), (48, 44))
        self.assertEqual(slide((3, 3), (3, 3), 0.7), (48, 48))

    def test_walk_cycle(self):
        self.assertEqual(walk_frame(True, 0.0), 1)
        self.assertEqual(walk_frame(True, 0.49), 1)
        self.assertEqual(walk_frame(True, 0.5), 0)
        self.assertEqual(walk_frame(True, 1.0), 0)
        self.assertEqual(walk_frame(False, 0.2), 0)

    def test_bomb_pulses_faster_as_countdown_drops(self):
        rates = [bomb_pulses_per_tick(c) for c in (4, 3, 2, 1)]
        self.assertEqual(rates, sorted(rates))
        self.assertLess(rates[0], rates[-1])
        for countdown in (4, 3, 2, 1):
            self.assertEqual(bomb_frame(countdown, 0.0), 0)
            self.assertEqual(bomb_frame(countdown, 1.0), 0)
        self.assertEqual(bomb_frame(4, 0.6), 1)
        pulses = [bomb_frame(1, t / 100) for t in range(100)]
        switches = sum(1 for a, b in zip(pulses, pulses[1:]) if a != b)
        self.assertGreaterEqual(switches, 6)

    def test_flame_grows_then_stays(self):
        self.assertEqual(flame_frame(0.0), 0)
        self.assertEqual(flame_frame(0.2), 1)
        self.assertEqual(flame_frame(0.5), "steady")
        self.assertEqual(flame_frame(1.0), "steady")

    def test_death_frames(self):
        self.assertEqual([death_frame(t) for t in (0.0, 0.34, 0.67, 0.99)], [0, 1, 2, 2])


class TestDiffs(unittest.TestCase):
    def test_broken_crates_and_gate_changes_found(self):
        found_crate = found_gate = found_death = False
        for pairs in real_games():
            for prev, cur in pairs:
                for x, y in broken_crates(prev, cur):
                    found_crate = True
                    self.assertIs(prev["grid"][y][x], Cell.BLOCK)
                    self.assertIs(cur["grid"][y][x], Cell.EMPTY)
                    self.assertIn((x, y), cur["explosion"])
                    destroyed = [e["cell"] for e in cur["events"] if e["type"] == "block_destroyed"]
                    self.assertIn((x, y), destroyed)
                if gate_changed(prev, cur):
                    found_gate = True
                    types = {e["type"] for e in cur["events"]}
                    self.assertTrue({"gate_revealed", "gate_opened"} & types)
                for before, after in died_this_tick(prev, cur):
                    found_death = True
                    self.assertFalse(before["dead"])
                    self.assertTrue(after["dead"])
        self.assertTrue(found_crate and found_gate and found_death)

    def test_no_diff_between_equal_states(self):
        session = GameSession(MODE_FREE, seed=3)
        state = session.cur
        self.assertEqual(broken_crates(state, state), [])
        self.assertFalse(gate_changed(state, state))
        self.assertEqual(died_this_tick(state, state), [])


class TestAnimatedScene(unittest.TestCase):
    def test_end_of_every_tick_matches_the_state(self):
        """The core sync rule: at t = 1 the animation shows exactly ``cur``."""
        checked = 0
        for pairs in real_games():
            facings = initial_facings(pairs[0][0])
            for prev, cur in pairs:
                facings = update_facings(facings, prev, cur)
                self.assertEqual(scene(prev, cur, 1.0, facings), static_scene(cur, facings))
                checked += 1
        self.assertGreater(checked, 1000)

    def test_scene_of_unchanged_state_is_static(self):
        session = GameSession(MODE_BOTS, seed=4)
        state = session.cur
        facings = initial_facings(state)
        for t in T_VALUES:
            self.assertEqual(scene(state, state, t, facings), static_scene(state, facings))

    def test_layers_in_order_and_t_clamped(self):
        pairs = played_transitions(1)
        prev, cur = pairs[10]
        facings = initial_facings(cur)
        for t in T_VALUES:
            items = scene(prev, cur, t, facings)
            self.assertEqual([i.layer for i in items], sorted(i.layer for i in items))
        self.assertEqual(scene(prev, cur, 2.0, facings), scene(prev, cur, 1.0, facings))
        self.assertEqual(scene(prev, cur, -1.0, facings), scene(prev, cur, 0.0, facings))

    def test_characters_slide_from_prev_to_cur(self):
        for pairs in real_games():
            for prev, cur in pairs[:40]:
                facings = update_facings(initial_facings(prev), prev, cur)
                for t in T_VALUES:
                    items = scene(prev, cur, t, facings)
                    drawn = {i.key[1]: (i.x, i.y) for i in items if i.layer == LAYER_CHARACTER}
                    for before, after in zip(prev["characters"], cur["characters"]):
                        if before["dead"]:
                            self.assertNotIn(before["id"], drawn)
                            continue
                        expected = slide((before["x"], before["y"]), (after["x"], after["y"]), t)
                        self.assertEqual(drawn[after["id"]], expected)
            break

    def test_dying_character_animates_then_disappears(self):
        for pairs in real_games():
            for prev, cur in pairs:
                deaths = died_this_tick(prev, cur)
                if not deaths:
                    continue
                facings = initial_facings(cur)
                ids = {after["id"] for _, after in deaths}
                for t in T_VALUES:
                    keys = [i.key for i in scene(prev, cur, t, facings) if i.layer == LAYER_CHARACTER]
                    dying = {k[1] for k in keys if k[0] == "death"}
                    self.assertEqual(dying, ids)
                end = [i.key for i in scene(prev, cur, 1.0, facings) if i.layer == LAYER_CHARACTER]
                self.assertFalse(any(k[0] == "death" or k[1] in ids for k in end))
                return
        self.fail("no death found in the sample games")

    def test_crate_debris_only_early_in_the_tick(self):
        for pairs in real_games():
            for prev, cur in pairs:
                broken = broken_crates(prev, cur)
                if not broken:
                    continue
                facings = initial_facings(cur)
                early = scene(prev, cur, 0.0, facings)
                debris = sorted((i.x // 16, i.y // 16) for i in early
                                if i.layer == LAYER_EFFECT and i.key[0] == "debris")
                self.assertEqual(debris, sorted(broken))
                late = scene(prev, cur, CRATE_BREAK_END, facings)
                self.assertFalse(any(i.key[0] == "debris" for i in late))
                return
        self.fail("no broken crate found in the sample games")

    def test_gate_blinks_when_it_changes(self):
        for pairs in real_games():
            for prev, cur in pairs:
                if not gate_changed(prev, cur):
                    continue
                facings = initial_facings(cur)
                blink = [("gate", "flash", 0) in [i.key for i in scene(prev, cur, t, facings)]
                         for t in (0.0, 0.1, 0.2, 0.3, 0.4)]
                self.assertTrue(any(blink) and not all(blink))
                after = scene(prev, cur, GATE_BLINK_END, facings)
                self.assertNotIn(("gate", "flash", 0), [i.key for i in after])
                return
        self.fail("no gate change found in the sample games")

    def test_new_bomb_and_flames_appear_at_tick_start(self):
        for pairs in real_games():
            for prev, cur in pairs:
                if not cur["explosion"]:
                    continue
                items = scene(prev, cur, 0.0, initial_facings(cur))
                self.assertEqual(cells(items, LAYER_FLAME), sorted(cur["explosion"]))
                self.assertEqual(cells(items, LAYER_BOMB),
                                 sorted((b["x"], b["y"]) for b in cur["bombs"]))
                return


if __name__ == "__main__":
    unittest.main()
