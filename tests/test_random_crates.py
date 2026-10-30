"""Headless checks for seeded random crate layouts. Pygame is not required."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.environment import (  # noqa: E402
    CRATE_COUNT,
    GATE_POSITION,
    SAFE_CELLS,
    Action,
    BombermanEnv,
    Cell,
)

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB

SEEDS = range(1000)
FIXED_GRID = BombermanEnv().grid
# Both escape routes of every spawn run along the safe cells.
ESCAPES = {
    0: ((R, R, R), (D, D, D)),
    1: ((D, D, D), (L, L, L)),
    2: ((U, U, U), (R, R, R)),
    3: ((U, U, U), (L, L, L)),
}


def crates(grid) -> set[tuple[int, int]]:
    return {
        (x, y)
        for y, row in enumerate(grid)
        for x, cell in enumerate(row)
        if cell is Cell.BLOCK
    }


def layout(seed: int) -> frozenset[tuple[int, int]]:
    env = BombermanEnv(random_crates=True)
    env.reset(seed=seed)
    return frozenset(crates(env.grid))


class TestSeeding(unittest.TestCase):
    def test_same_seed_gives_the_same_layout(self):
        for seed in (0, 1, 12345):
            with self.subTest(seed=seed):
                first = BombermanEnv(random_crates=True, with_npcs=True)
                second = BombermanEnv(random_crates=True, with_npcs=True)
                self.assertEqual(first.reset(seed=seed), second.reset(seed=seed))
                # Reseeding the same env repeats the layout too.
                first.step(Action.RIGHT)
                self.assertEqual(first.reset(seed=seed), second.get_state())

    def test_different_seeds_give_different_layouts(self):
        self.assertEqual(len({layout(seed) for seed in SEEDS}), len(SEEDS))

    def test_reset_without_a_seed_continues_the_rng(self):
        env = BombermanEnv(random_crates=True)
        other = BombermanEnv(random_crates=True)
        env.reset(seed=7)
        other.reset(seed=7)
        follow_ups = [crates(env.reset()["grid"]) for _ in range(3)]
        self.assertEqual(follow_ups, [crates(other.reset()["grid"]) for _ in range(3)])
        self.assertNotEqual(follow_ups[0], set(layout(7)))
        self.assertEqual(len({frozenset(c) for c in follow_ups}), 3)


class TestLayoutRules(unittest.TestCase):
    def test_every_layout_follows_the_crate_rules(self):
        self.assertEqual(CRATE_COUNT, 18)
        self.assertEqual(len(SAFE_CELLS), 28)
        for seed in SEEDS:
            env = BombermanEnv(random_crates=True)
            env.reset(seed=seed)
            placed = crates(env.grid)
            with self.subTest(seed=seed):
                self.assertEqual(len(placed), CRATE_COUNT)
                self.assertIn(GATE_POSITION, placed)
                self.assertEqual(env.gate.state, "hidden")
                self.assertFalse(placed & SAFE_CELLS)
                for y, row in enumerate(env.grid):
                    for x, cell in enumerate(row):
                        fixed_wall = FIXED_GRID[y][x] is Cell.WALL
                        self.assertEqual(cell is Cell.WALL, fixed_wall)

    def test_every_spawn_escapes_its_own_bomb_in_every_layout(self):
        for seed in SEEDS:
            for character_id, routes in ESCAPES.items():
                for route in routes:
                    env = BombermanEnv(random_crates=True, with_npcs=True)
                    env.reset(seed=seed)
                    for action in (B, *route, W):
                        if character_id == 0:
                            env.step(action)
                        else:
                            env.step(W, {character_id: action})
                    with self.subTest(seed=seed, character=character_id, route=route):
                        self.assertEqual(env.bombs, {})
                        self.assertFalse(any(c.dead for c in env.characters))
                        self.assertIsNone(env.outcome)


class TestFixedMapDefault(unittest.TestCase):
    def test_fixed_map_ignores_seeds(self):
        env = BombermanEnv(with_npcs=True)
        self.assertFalse(env.random_crates)
        for seed in (None, 0, 99):
            with self.subTest(seed=seed):
                env.step(Action.RIGHT)
                state = env.reset(seed=seed)
                self.assertEqual(state["grid"], FIXED_GRID)
                self.assertEqual(state, BombermanEnv(with_npcs=True).get_state())

    def test_random_mode_also_works_without_npcs(self):
        env = BombermanEnv(random_crates=True)
        state = env.reset(seed=3)
        self.assertEqual(len(state["characters"]), 1)
        self.assertEqual(crates(state["grid"]), set(layout(3)))


if __name__ == "__main__":
    unittest.main()
