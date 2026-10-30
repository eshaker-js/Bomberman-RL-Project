"""Headless checks for the gate and the game outcome. Pygame is not required.

The gate hides under the crate at (5, 5). Its neighbours (4, 5) and (6, 5)
are dead ends between crates, so a bomb there reveals the gate and kills
whoever stands on it. The tests use that to reveal the gate and remove NPCs
on a known action. NPC actions are supplied directly by each test.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import GATE_POSITION, Action, BombermanEnv, Cell  # noqa: E402

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB

REVEALED = {"type": "gate_revealed"}
OPENED = {"type": "gate_opened"}


def put(env: BombermanEnv, character_id: int, x: int, y: int) -> None:
    """Test-only setup: move a character of a fresh NPC env to a floor cell.

    The cell may already hold another character.
    """
    assert env.with_npcs and env.steps == 0 and not env.bombs, "fresh NPC env only"
    assert env.grid[y][x] is Cell.EMPTY
    character = env.characters[character_id]
    character.x, character.y = x, y


def run(env: BombermanEnv, turns) -> BombermanEnv:
    """Each turn is ``(agent_action, npc_actions)``."""
    for action, npc_actions in turns:
        env.step(action, npc_actions)
    return env


def death(character: int, *killed_by: int) -> dict:
    return {"type": "death", "character": character, "killed_by": killed_by}


def block(cell: tuple[int, int], *destroyed_by: int) -> dict:
    return {"type": "block_destroyed", "cell": cell, "destroyed_by": destroyed_by}


def wait(count: int):
    return [(W, None)] * count


# NPC 1's bomb at (4, 5) burns (4, 5), the gate crate, and the crate at (3, 5).
NPC1_GATE_BLOCKS = [block(GATE_POSITION, 1), block((3, 5), 1)]


class GateTestCase(unittest.TestCase):
    def assert_frozen(self, env: BombermanEnv) -> None:
        before = env.get_state()
        npc_variants = (None, {1: D, 2: U, 3: L}) if env.with_npcs else (None,)
        for npc_actions in npc_variants:
            for action in (R, B, D):
                after = env.step(action, npc_actions)
                self.assertEqual(after, before)


class TestGateStates(GateTestCase):
    def test_hidden_then_closed_then_open(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 4, 5)
        put(env, 3, 1, 9)
        self.assertEqual(env.get_state()["gate"], {"x": 5, "y": 5, "state": "hidden"})

        run(env, [(W, {1: B})] + wait(2) + [(W, {2: B})])
        state = env.step(W)
        self.assertEqual(env.steps, 5)
        self.assertEqual(state["events"], NPC1_GATE_BLOCKS + [REVEALED, death(1, 1)])
        self.assertEqual(state["gate"]["state"], "closed")
        self.assertEqual(env.gate.state, "closed")
        self.assertIsNone(state["outcome"])

        state = run(env, wait(2)).step(W)
        self.assertEqual(env.steps, 8)
        self.assertEqual(
            state["events"], [block((3, 9), 2), death(2, 2), death(3, 2), OPENED]
        )
        self.assertEqual(state["gate"]["state"], "open")
        self.assertTrue(state["all_npcs_dead"])
        # The agent is not on the gate, so all NPCs dead is still no outcome.
        self.assertIsNone(state["outcome"])
        state = env.step(R)
        self.assertEqual(env.player, (2, 1))
        self.assertEqual(state["events"], [])

    def test_hidden_gate_opens_when_revealed_after_all_npcs_died(self):
        env = BombermanEnv(with_npcs=True)
        for npc_id in (1, 2, 3):
            put(env, npc_id, 4, 5)
        run(env, [(W, {1: B})] + wait(3))
        self.assertEqual(env.gate.state, "hidden")
        state = env.step(W)
        self.assertEqual(
            state["events"],
            NPC1_GATE_BLOCKS
            + [REVEALED, death(1, 1), death(2, 1), death(3, 1), OPENED],
        )
        self.assertEqual(state["gate"]["state"], "open")
        self.assertIsNone(state["outcome"])

    def test_single_player_gate_is_revealed_but_never_opens(self):
        env = BombermanEnv()
        # Test-only setup: clear (6, 3) so the agent can bomb (5, 3) and escape.
        env.grid[3][6] = Cell.EMPTY
        env.player_x, env.player_y = 5, 3
        for action in (B, R, R, U):
            env.step(action)
        state = env.step(W)
        self.assertEqual(env.player, (7, 2))
        self.assertEqual(
            state["events"],
            [block((4, 3), 0), block(GATE_POSITION, 0), block((5, 1), 0), REVEALED],
        )
        self.assertEqual(state["gate"]["state"], "closed")

        for action in (D, L, L, D, D):
            state = env.step(action)
        self.assertEqual(env.player, GATE_POSITION)
        self.assertEqual(state["gate"]["state"], "closed")
        self.assertIsNone(state["outcome"])


class TestGateIsFloor(GateTestCase):
    def test_crate_on_the_gate_blocks_moves_and_blasts_like_any_crate(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 5, 4)
        put(env, 1, 4, 5)
        env.step(D, {1: B})
        self.assertEqual(env.player, (5, 4))
        self.assertIs(env.grid[5][5], Cell.BLOCK)

        state = run(env, wait(3)).step(W)
        self.assertIn(GATE_POSITION, state["explosion"])
        # The crate ended the ray, so the floor behind it did not burn.
        self.assertNotIn((6, 5), state["explosion"])
        self.assertIs(env.grid[5][5], Cell.EMPTY)

    def test_gate_can_be_walked_on_bombed_and_crossed_by_blasts(self):
        env = BombermanEnv()
        env.grid[3][6] = Cell.EMPTY
        env.player_x, env.player_y = 5, 3
        for action in (B, R, R, U, W, D, L, L, D, D):
            env.step(action)
        self.assertEqual(env.player, GATE_POSITION)

        state = env.step(B)
        self.assertEqual(state["bomb"], {"x": 5, "y": 5, "countdown": 4})
        for action in (U, U, U):
            env.step(action)
        state = env.step(W)
        self.assertEqual(env.player, (5, 2))
        self.assertFalse(state["dead"])
        for cell in (GATE_POSITION, (5, 6), (5, 7), (5, 4), (5, 3), (4, 5), (6, 5)):
            self.assertIn(cell, state["explosion"])
        self.assertIs(env.grid[5][5], Cell.EMPTY)
        self.assertEqual(state["gate"]["state"], "closed")
        self.assertNotIn(REVEALED, state["events"])

    def test_a_blast_crosses_the_revealed_gate(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 5, 3)
        put(env, 1, 4, 5)
        run(env, [(W, {1: B})] + wait(4) + [(D, None), (B, None)])
        self.assertEqual(env.bomb.x, 5)
        self.assertEqual(env.bomb.y, 4)
        # Over the closed gate and down the column, out of range.
        for action in (D, D, D):
            state = env.step(action)
            self.assertIsNone(state["outcome"])
        state = env.step(W)
        self.assertEqual(env.player, (5, 7))
        self.assertFalse(env.dead)
        self.assertIn(GATE_POSITION, state["explosion"])
        self.assertIn((5, 6), state["explosion"])
        self.assertEqual(env.gate.state, "closed")


class TestOutcome(GateTestCase):
    def test_win_by_stepping_onto_the_open_gate(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 5, 3)
        for npc_id in (1, 2, 3):
            put(env, npc_id, 4, 5)
        run(env, [(W, {1: B})] + wait(4))
        self.assertEqual(env.gate.state, "open")

        state = env.step(D)
        self.assertIsNone(state["outcome"])
        state = env.step(D)
        self.assertEqual(env.player, GATE_POSITION)
        self.assertEqual(state["outcome"], "win")
        self.assertEqual(env.outcome, "win")
        self.assertEqual(state["events"], [])
        self.assert_frozen(env)

    def test_win_on_the_closed_gate_when_the_last_npcs_die(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 5, 3)
        put(env, 1, 4, 5)
        put(env, 3, 1, 9)
        run(env, [(W, {1: B})] + wait(2) + [(W, {2: B})] + wait(1))
        self.assertEqual(env.gate.state, "closed")

        run(env, [(D, None), (D, None)])
        self.assertEqual(env.player, GATE_POSITION)
        self.assertIsNone(env.outcome)

        state = env.step(W)
        self.assertEqual(
            state["events"], [block((3, 9), 2), death(2, 2), death(3, 2), OPENED]
        )
        self.assertEqual(state["outcome"], "win")
        self.assert_frozen(env)

    def test_stepping_onto_the_open_gate_as_a_blast_hits_it_is_a_loss(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 5, 3)
        for npc_id in (1, 2, 3):
            put(env, npc_id, 4, 5)
        run(env, [(W, {1: B})] + wait(4))
        self.assertEqual(env.gate.state, "open")

        run(env, [(B, None), (D, None), (W, None), (W, None)])
        self.assertEqual(env.player, (5, 4))
        self.assertIsNone(env.outcome)

        # Moves come before explosions, so the agent reaches the gate and dies.
        state = env.step(D)
        self.assertEqual(env.player, GATE_POSITION)
        self.assertTrue(state["dead"])
        self.assertEqual(
            state["events"],
            [block((6, 3), 0), block((4, 3), 0), block((5, 1), 0), death(0, 0)],
        )
        self.assertEqual(state["outcome"], "loss")
        self.assert_frozen(env)

    def test_loss_while_an_npc_is_alive(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 4, 5)
        put(env, 1, 4, 5)
        state = run(env, [(W, {1: B})] + wait(3)).step(W)
        self.assertEqual(
            state["events"], NPC1_GATE_BLOCKS + [REVEALED, death(0, 1), death(1, 1)]
        )
        self.assertEqual(state["outcome"], "loss")
        self.assertEqual(state["gate"]["state"], "closed")
        self.assert_frozen(env)

    def test_draw_when_the_agent_dies_with_all_npcs(self):
        env = BombermanEnv(with_npcs=True)
        for character_id in (0, 1, 2, 3):
            put(env, character_id, 4, 5)
        state = run(env, [(W, {1: B})] + wait(3)).step(W)
        self.assertEqual(
            state["events"],
            NPC1_GATE_BLOCKS
            + [REVEALED, death(0, 1), death(1, 1), death(2, 1), death(3, 1), OPENED],
        )
        self.assertEqual(state["outcome"], "draw")
        self.assert_frozen(env)

    def test_draw_when_the_agent_dies_with_the_last_living_npcs(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 4, 5)
        put(env, 0, 1, 9)
        put(env, 3, 1, 9)
        run(env, [(W, {1: B})] + wait(2) + [(W, {2: B})] + wait(1))
        self.assertTrue(env.characters[1].dead)
        self.assertIsNone(env.outcome)

        state = run(env, wait(2)).step(W)
        self.assertEqual(
            state["events"],
            [block((3, 9), 2), death(0, 2), death(2, 2), death(3, 2), OPENED],
        )
        self.assertEqual(state["outcome"], "draw")

    def test_single_player_death_is_a_loss(self):
        env = BombermanEnv()
        for action in (B, W, W, W):
            state = env.step(action)
            self.assertIsNone(state["outcome"])
        state = env.step(W)
        self.assertEqual(state["outcome"], "loss")
        self.assert_frozen(env)


class TestGateState(unittest.TestCase):
    def test_new_keys_are_copies_and_reset_restores_them(self):
        env = BombermanEnv(with_npcs=True)
        for character_id in (0, 1, 2, 3):
            put(env, character_id, 4, 5)
        run(env, [(W, {1: B})] + wait(4))
        self.assertEqual(env.outcome, "draw")

        state = env.get_state()
        state["gate"]["state"] = "hidden"
        self.assertEqual(env.gate.state, "open")

        state = env.reset()
        self.assertEqual(state["gate"], {"x": 5, "y": 5, "state": "hidden"})
        self.assertIsNone(state["outcome"])
        self.assertIsNone(env.outcome)
        self.assertIs(env.grid[5][5], Cell.BLOCK)
        env.step(R)
        self.assertEqual(env.player, (2, 1))


if __name__ == "__main__":
    unittest.main()
