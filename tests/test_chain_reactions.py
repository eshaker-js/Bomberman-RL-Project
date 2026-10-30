"""Headless checks for chain reactions. Pygame is not required.

A blast that reaches another bomb's cell burns it, stops there, and sets
that bomb off on the same action. NPC actions are supplied by each test.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.environment import GATE_POSITION, Action, BombermanEnv, Cell  # noqa: E402

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB


def put(env: BombermanEnv, character_id: int, x: int, y: int) -> None:
    """Test-only setup: move a character of a fresh NPC env to a floor cell."""
    assert env.with_npcs and env.steps == 0 and not env.bombs, "fresh NPC env only"
    assert env.grid[y][x] is Cell.EMPTY
    character = env.characters[character_id]
    character.x, character.y = x, y


def run(env: BombermanEnv, turns) -> BombermanEnv:
    """Each turn is ``(agent_action, npc_actions)``."""
    for action, npc_actions in turns:
        env.step(action, npc_actions)
    return env


def triggered(owner: int, *triggered_by: int) -> dict:
    return {"type": "bomb_triggered", "owner": owner, "triggered_by": triggered_by}


def death(character: int, *killed_by: int) -> dict:
    return {"type": "death", "character": character, "killed_by": killed_by}


def block(cell: tuple[int, int], *destroyed_by: int) -> dict:
    return {"type": "block_destroyed", "cell": cell, "destroyed_by": destroyed_by}


# The agent bombs its spawn on the first action and escapes down column 1,
# outside every blast used below. Its bomb explodes on action 5.
AGENT_ESCAPE = [B, D, D, D, W]


class TestSingleChain(unittest.TestCase):
    def test_blast_sets_off_a_bomb_and_credits_its_owner(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 3, 1)
        put(env, 2, 4, 1)
        put(env, 3, 2, 1)
        npc_turns = [None, {1: B}, {1: D}, {1: D}, {1: D}]
        run(env, list(zip(AGENT_ESCAPE[:4], npc_turns[:4])))
        self.assertEqual(env.bombs[1].countdown, 2)

        # Agent's bomb at (1, 1) reaches NPC 1's bomb at (3, 1). That bomb's
        # left ray stops at the agent's bomb cell although it already exploded.
        state = env.step(AGENT_ESCAPE[4], npc_turns[4])
        self.assertEqual(
            state["explosion"],
            [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (4, 1), (5, 1), (3, 2), (3, 3)],
        )
        self.assertEqual(
            state["events"],
            [triggered(1, 0), block((5, 1), 1), death(2, 1), death(3, 0, 1)],
        )
        self.assertEqual(env.bombs, {})
        self.assertEqual(env.player, (1, 4))
        self.assertEqual(env.characters[1].y, 4)
        self.assertFalse(env.dead)

        # Nothing is left to explode on the action NPC 1's fuse would have ended.
        state = env.step(W)
        self.assertEqual(state["explosion"], [])
        self.assertEqual(state["events"], [])

    def test_a_bomb_placed_on_this_action_can_be_set_off(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 3, 1)
        run(env, [(a, None) for a in AGENT_ESCAPE[:4]])
        state = env.step(AGENT_ESCAPE[4], {1: B})
        self.assertEqual(state["events"][0], triggered(1, 0))
        self.assertEqual(env.bombs, {})
        self.assertTrue(env.characters[1].dead)

    def test_a_dead_owners_bomb_can_be_set_off(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 3, 3)
        put(env, 2, 3, 4)
        npc_turns = [None, {2: B}, {1: B}, {1: U}, {1: U}]
        run(env, list(zip(AGENT_ESCAPE, npc_turns)))
        # The agent's blast killed NPC 1 at (3, 1); its bomb at (3, 3) remains.
        self.assertTrue(env.characters[1].dead)
        self.assertEqual(env.bombs[1].countdown, 2)

        state = env.step(W)
        self.assertEqual(
            state["events"],
            [
                triggered(1, 2),
                block((3, 5), 2),
                block((4, 3), 1),
                block((2, 3), 1),
                # NPC 1's bomb burns (3, 4), the cell of the bomb that set it off.
                death(2, 1, 2),
            ],
        )
        self.assertEqual(env.bombs, {})
        self.assertIn((3, 1), state["explosion"])


class TestWaves(unittest.TestCase):
    def test_three_waves_in_one_action(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 3, 1)
        put(env, 2, 3, 3)
        npc_turns = [None, {1: B}, {2: B}, None, None]
        state = run(env, list(zip(AGENT_ESCAPE[:4], npc_turns[:4]))).step(
            AGENT_ESCAPE[4], npc_turns[4]
        )
        self.assertEqual(
            state["explosion"],
            [
                (1, 1), (2, 1), (3, 1), (1, 2), (1, 3),  # agent, wave 1
                (4, 1), (5, 1), (3, 2), (3, 3),  # NPC 1, wave 2
                (4, 3), (2, 3), (3, 4), (3, 5),  # NPC 2, wave 3
            ],
        )
        self.assertEqual(
            state["events"],
            [
                triggered(1, 0),
                triggered(2, 1),
                block((5, 1), 1),
                block((4, 3), 2),
                block((2, 3), 2),
                block((3, 5), 2),
                death(1, 0, 1, 2),
                death(2, 1, 2),
            ],
        )
        self.assertEqual(env.bombs, {})
        self.assertFalse(env.dead)

    def test_two_bombs_in_one_wave_share_the_trigger(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 3, 1)
        put(env, 2, 3, 3)
        npc_turns = [{2: B}, {1: B}, None, None, None]
        state = run(env, list(zip(AGENT_ESCAPE[:4], npc_turns[:4]))).step(
            AGENT_ESCAPE[4], npc_turns[4]
        )
        self.assertEqual(state["events"][0], triggered(1, 0, 2))
        self.assertEqual(
            [e for e in state["events"] if e["type"] == "bomb_triggered"],
            [triggered(1, 0, 2)],
        )
        self.assertEqual(env.bombs, {})

    def test_later_waves_see_the_board_before_any_crate_broke(self):
        # NPC 1 at (1, 3) breaks the crate at (1, 5) in wave 1. NPC 3's bomb
        # at (1, 6), set off by NPC 2's, still stops at (1, 5) in wave 2, so
        # the agent at (1, 4) is killed by NPC 1 alone.
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 1, 4)
        put(env, 1, 1, 3)
        put(env, 2, 1, 8)
        put(env, 3, 1, 6)
        run(env, [(W, {1: B, 2: B}), (W, {3: B}), (W, None), (W, None)])
        state = env.step(W)
        self.assertEqual(
            state["events"],
            [
                triggered(3, 2),
                block((2, 3), 1),
                block((1, 5), 1, 3),
                death(0, 1),
                death(1, 1),
                death(2, 2, 3),
                death(3, 2, 3),
            ],
        )
        self.assertIs(env.grid[5][1], Cell.EMPTY)
        self.assertEqual(state["outcome"], "draw")


class TestEventOrder(unittest.TestCase):
    def test_all_event_types_in_one_action(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 5, 2)
        put(env, 2, 5, 4)
        put(env, 3, 5, 3)
        run(env, [(W, {1: B}), (W, {2: B}), (W, None), (W, None)])
        state = env.step(W)
        self.assertEqual(
            state["events"],
            [
                triggered(2, 1),
                block((5, 1), 1),
                block(GATE_POSITION, 2),
                {"type": "gate_revealed"},
                death(1, 1, 2),
                death(2, 1, 2),
                death(3, 1, 2),
                {"type": "gate_opened"},
            ],
        )
        self.assertEqual(state["gate"]["state"], "open")
        self.assertIsNone(state["outcome"])


if __name__ == "__main__":
    unittest.main()
