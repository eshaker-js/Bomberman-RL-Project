"""Headless checks for the Track A movement rules. Pygame is not required.

Characters never block each other, bombs block from the action after they
are placed, and each cell holds at most one bomb. NPC actions are supplied
directly by each test.
"""

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.environment import Action, Bomb, BombermanEnv, Cell  # noqa: E402

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB


def put(env: BombermanEnv, character_id: int, x: int, y: int) -> None:
    """Test-only setup: move a character of a fresh NPC env to a floor cell.

    Unlike the Task 1 helper, the cell may already hold another character.
    """
    assert env.with_npcs and env.steps == 0 and not env.bombs, "fresh NPC env only"
    assert env.grid[y][x] is Cell.EMPTY
    character = env.characters[character_id]
    character.x, character.y = x, y


def positions(env: BombermanEnv) -> list[tuple[int, int]]:
    return [(c.x, c.y) for c in env.characters]


class TestPassThrough(unittest.TestCase):
    def test_three_characters_can_share_one_cell(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 2, 1)
        put(env, 1, 4, 1)
        put(env, 2, 3, 2)
        env.step(R, {1: L, 2: U})
        self.assertEqual(positions(env)[:3], [(3, 1)] * 3)
        self.assertEqual(env.events, [])

        # Sharing a cell does not stop anyone from leaving it.
        env.step(L, {1: R, 2: D})
        self.assertEqual(positions(env)[:3], [(2, 1), (4, 1), (3, 2)])

    def test_a_line_of_characters_moves_together(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 1, 1)
        put(env, 1, 2, 1)
        put(env, 2, 3, 1)
        env.step(R, {1: R, 2: R})
        self.assertEqual(positions(env)[:3], [(2, 1), (3, 1), (4, 1)])


class TestFreshBombs(unittest.TestCase):
    def test_lower_id_places_and_higher_id_walks_onto_it_in_the_same_action(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 2, 1)
        put(env, 1, 3, 1)
        env.step(B, {1: L})
        self.assertEqual(positions(env)[:2], [(2, 1), (2, 1)])
        self.assertEqual(env.bombs, {0: Bomb(2, 1, 4)})

    def test_higher_id_places_and_lower_id_walks_onto_it_in_the_same_action(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 3, 1)
        put(env, 1, 2, 1)
        env.step(L, {1: B})
        self.assertEqual(positions(env)[:2], [(2, 1), (2, 1)])
        self.assertEqual(env.bombs, {1: Bomb(2, 1, 4)})

    def test_a_bomb_blocks_every_character_from_the_next_action(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 1, 1)
        put(env, 1, 2, 1)
        put(env, 2, 3, 1)
        env.step(W, {1: B})
        env.step(W, {1: R})
        self.assertEqual(positions(env)[:3], [(1, 1), (3, 1), (3, 1)])
        # The bomb at (2, 1) now blocks its owner and both neighbours.
        env.step(R, {1: L, 2: L})
        self.assertEqual(positions(env)[:3], [(1, 1), (3, 1), (3, 1)])
        self.assertEqual(env.steps, 3)
        self.assertEqual(env.bombs[1].countdown, 2)

    def test_anyone_on_a_bomb_may_step_off_but_not_back_on(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 3, 1)
        put(env, 1, 2, 1)
        # The agent enters NPC 1's cell as NPC 1 places its bomb there.
        env.step(L, {1: B})
        self.assertEqual(positions(env)[:2], [(2, 1), (2, 1)])

        env.step(R, {1: L})
        self.assertEqual(positions(env)[:2], [(3, 1), (1, 1)])

        env.step(L, {1: R})
        self.assertEqual(positions(env)[:2], [(3, 1), (1, 1)])
        self.assertEqual(env.bombs[1].countdown, 2)


class TestOneBombPerCell(unittest.TestCase):
    def test_lower_id_wins_a_shared_cell_and_the_other_turn_is_spent(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 3, 1)
        put(env, 1, 3, 1)
        state = env.step(B, {1: B})
        self.assertEqual(env.bombs, {0: Bomb(3, 1, 4)})
        self.assertEqual(state["steps"], 1)

        # Off the bomb, NPC 1 may place its own.
        env.step(W, {1: R})
        env.step(W, {1: B})
        self.assertEqual(env.bombs, {0: Bomb(3, 1, 2), 1: Bomb(4, 1, 4)})

    def test_tie_break_between_two_npcs_follows_id_order(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 3, 1)
        put(env, 2, 3, 1)
        env.step(W, {1: B, 2: B})
        self.assertEqual(env.bombs, {1: Bomb(3, 1, 4)})

    def test_a_bomb_from_an_earlier_action_also_refuses_a_second_one(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 2, 1)
        put(env, 1, 3, 1)
        env.step(B, {1: L})
        env.step(W, {1: B})
        self.assertEqual(env.bombs, {0: Bomb(2, 1, 3)})
        self.assertEqual(positions(env)[:2], [(2, 1), (2, 1)])


# Floor cells of the top-left area, where characters meet quickly.
_AREA = [
    (x, y)
    for y in range(1, 6)
    for x in range(1, 6)
    if BombermanEnv().grid[y][x] is Cell.EMPTY
]


def _scenario(seed: int):
    """Random start cells and three actions per character, without explosions."""
    rng = random.Random(seed)
    starts = [rng.choice(_AREA) for _ in range(4)]
    turns = [[rng.choice(list(Action)) for _ in range(4)] for _ in range(3)]
    return starts, turns


def _play(starts, turns):
    """Run a scenario. Return the env, or None when a same-cell bomb tie occurs."""
    env = BombermanEnv(with_npcs=True)
    for character_id, (x, y) in enumerate(starts):
        put(env, character_id, x, y)
    for actions in turns:
        bombers = [c for c in env.characters if actions[c.id] is B]
        cells = [(c.x, c.y) for c in bombers]
        if len(cells) != len(set(cells)):
            return None
        env.step(actions[0], {npc_id: actions[npc_id] for npc_id in (1, 2, 3)})
    return env


class TestOrderIndependence(unittest.TestCase):
    """Relabelling characters relabels the result, except for bomb ties.

    Three actions never finish a fuse, so nobody dies and the agent's
    special role does not matter.
    """

    SEEDS = range(200)
    RELABELLINGS = ((0, 3, 1, 2), (0, 2, 3, 1), (1, 0, 2, 3), (3, 2, 1, 0))

    def test_relabelling_characters_relabels_the_outcome(self):
        compared = 0
        for seed in self.SEEDS:
            starts, turns = _scenario(seed)
            original = _play(starts, turns)
            if original is None:
                continue
            for new_id in self.RELABELLINGS:
                with self.subTest(seed=seed, relabelling=new_id):
                    # Character i of the original plays as character new_id[i].
                    moved_starts = [None] * 4
                    moved_turns = [[None] * 4 for _ in turns]
                    for old, new in enumerate(new_id):
                        moved_starts[new] = starts[old]
                        for t, actions in enumerate(turns):
                            moved_turns[t][new] = actions[old]
                    relabelled = _play(moved_starts, moved_turns)
                    self.assertIsNotNone(relabelled)
                    for old, new in enumerate(new_id):
                        self.assertEqual(
                            positions(relabelled)[new], positions(original)[old]
                        )
                        self.assertEqual(
                            relabelled.bombs.get(new), original.bombs.get(old)
                        )
                    compared += 1
        # Guard against a generator that skips almost every scenario.
        self.assertGreater(compared, 500)


if __name__ == "__main__":
    unittest.main()
