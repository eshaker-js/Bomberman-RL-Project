"""Headless checks for the agent plus three NPCs. Pygame is not required.

NPC decisions are not tested here: each test supplies NPC actions directly.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import (  # noqa: E402
    NPC_STARTS,
    PLAYER_START,
    Action,
    Bomb,
    BombermanEnv,
    Cell,
)

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB

OLD_STATE_KEYS = {"grid", "player", "dead", "bomb", "explosion", "steps"}


def place(env: BombermanEnv, character_id: int, x: int, y: int) -> None:
    """Test-only setup: move a living character of a fresh NPC env to a free floor cell."""
    assert env.with_npcs and env.steps == 0 and not env.bombs, "fresh NPC env only"
    character = env.characters[character_id]
    assert not character.dead
    assert env.grid[y][x] is Cell.EMPTY
    assert all((c.x, c.y) != (x, y) for c in env.characters if c is not character)
    character.x, character.y = x, y


def run(env: BombermanEnv, turns) -> BombermanEnv:
    """Each turn is ``(agent_action, npc_actions)``."""
    for action, npc_actions in turns:
        env.step(action, npc_actions)
    return env


def positions(env: BombermanEnv) -> list[tuple[int, int]]:
    return [(c.x, c.y) for c in env.characters]


def death(character: int, *killed_by: int) -> dict:
    return {"type": "death", "character": character, "killed_by": killed_by}


def block(cell: tuple[int, int], *destroyed_by: int) -> dict:
    return {"type": "block_destroyed", "cell": cell, "destroyed_by": destroyed_by}


class TestSinglePlayerCompatibility(unittest.TestCase):
    def test_default_env_has_only_the_agent(self):
        env = BombermanEnv()
        state = env.get_state()
        self.assertEqual(len(env.characters), 1)
        self.assertTrue(OLD_STATE_KEYS <= set(state))
        self.assertEqual(state["characters"], [{"id": 0, "x": 1, "y": 1, "dead": False}])
        self.assertEqual(state["bombs"], [])
        self.assertEqual(state["events"], [])
        self.assertFalse(env.all_npcs_dead)
        self.assertFalse(state["all_npcs_dead"])

    def test_opening_blast_order_is_unchanged(self):
        env = BombermanEnv()
        for action in (B, R, R, R, W):
            env.step(action)
        self.assertEqual(env.explosion, [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)])
        self.assertEqual(env.events, [])

    def test_self_kill_reports_the_agent_as_killer(self):
        env = BombermanEnv()
        for action in (B, W, W, W, W):
            state = env.step(action)
        self.assertEqual(state["events"], [death(0, 0)])

    def test_block_destroyed_event(self):
        env = BombermanEnv()
        for action in (R, R, R, B, L, L, L, W):
            state = env.step(action)
        self.assertEqual(state["events"], [block((5, 1), 0)])

    def test_npc_actions_are_rejected_without_npcs(self):
        env = BombermanEnv()
        with self.assertRaises(ValueError):
            env.step(W, {1: W})
        with self.assertRaises(ValueError):
            env.step(W, {})
        self.assertEqual(env.steps, 0)

    def test_agent_attributes_stay_writable(self):
        env = BombermanEnv()
        env.player_x = 2
        env.player_y = 1
        self.assertEqual(env.player, (2, 1))
        self.assertEqual(positions(env), [(2, 1)])
        env.bomb = Bomb(2, 1, 3)
        self.assertEqual(env.bombs, {0: Bomb(2, 1, 3)})
        env.bomb = None
        self.assertEqual(env.bombs, {})
        env.dead = True
        self.assertTrue(env.characters[0].dead)


class TestSpawns(unittest.TestCase):
    def test_four_characters_on_distinct_floor_spawns(self):
        env = BombermanEnv(with_npcs=True)
        expected = [PLAYER_START, NPC_STARTS[1], NPC_STARTS[2], NPC_STARTS[3]]
        self.assertEqual(expected, [(1, 1), (9, 1), (1, 9), (9, 9)])
        self.assertEqual(positions(env), expected)
        self.assertEqual([c.id for c in env.characters], [0, 1, 2, 3])
        self.assertEqual(len(set(expected)), 4)
        for x, y in expected:
            self.assertIs(env.grid[y][x], Cell.EMPTY)
        self.assertFalse(any(c.dead for c in env.characters))

    def test_reset_restores_npcs_and_bombs(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(B, {1: B, 2: U}), (R, {1: D, 3: U})])
        self.assertTrue(env.bombs)

        state = env.reset()
        self.assertEqual(positions(env), [(1, 1), (9, 1), (1, 9), (9, 9)])
        self.assertEqual(env.bombs, {})
        self.assertEqual(state["bombs"], [])
        self.assertEqual(env.events, [])
        self.assertEqual(env.steps, 0)
        self.assertFalse(state["all_npcs_dead"])

    def test_every_spawn_escapes_its_own_bomb(self):
        routes = {0: (R, R, R), 1: (D, D, D), 2: (U, U, U), 3: (U, U, U)}
        ends = {0: (4, 1), 1: (9, 4), 2: (1, 6), 3: (9, 6)}
        for character_id, route in routes.items():
            with self.subTest(character=character_id):
                env = BombermanEnv(with_npcs=True)
                for action in (B, *route, W):
                    if character_id == 0:
                        env.step(action)
                    else:
                        env.step(W, {character_id: action})
                self.assertEqual(positions(env)[character_id], ends[character_id])
                self.assertFalse(any(c.dead for c in env.characters))
                self.assertEqual(env.bombs, {})
                self.assertTrue(env.explosion)
                self.assertFalse([e for e in env.events if e["type"] == "death"])

    def test_agent_also_escapes_downward(self):
        env = BombermanEnv(with_npcs=True)
        for action in (B, D, D, D, W):
            env.step(action)
        self.assertEqual(env.player, (1, 4))
        self.assertFalse(env.dead)


class TestNpcInput(unittest.TestCase):
    def test_missing_npc_actions_wait(self):
        env = BombermanEnv(with_npcs=True)
        env.step(R)
        self.assertEqual(positions(env), [(2, 1), (9, 1), (1, 9), (9, 9)])
        env.step(W, {2: U})
        self.assertEqual(positions(env), [(2, 1), (9, 1), (1, 8), (9, 9)])
        self.assertEqual(env.steps, 2)

    def test_bad_npc_input_raises_without_changing_state(self):
        env = BombermanEnv(with_npcs=True)
        before = env.get_state()
        for bad, error in (
            ({0: W}, ValueError),
            ({4: W}, ValueError),
            ({1: "up"}, TypeError),
            ([W, W, W], TypeError),
        ):
            with self.subTest(npc_actions=bad):
                with self.assertRaises(error):
                    env.step(R, bad)
        with self.assertRaises(TypeError):
            env.step("right", {1: W})
        self.assertEqual(env.get_state(), before)


class TestCollisions(unittest.TestCase):
    def setUp(self):
        self.env = BombermanEnv(with_npcs=True)

    def test_moving_into_a_character_shares_its_cell(self):
        place(self.env, 0, 2, 1)
        place(self.env, 1, 3, 1)
        self.env.step(B)
        self.env.step(R)
        self.assertEqual(positions(self.env)[:2], [(3, 1), (3, 1)])
        self.assertEqual(self.env.steps, 2)
        self.assertEqual(self.env.bomb.countdown, 3)
        self.assertFalse(any(c.dead for c in self.env.characters))

    def test_agent_can_follow_an_npc_that_moves_later(self):
        place(self.env, 0, 2, 1)
        place(self.env, 1, 3, 1)
        self.env.step(R, {1: R})
        self.assertEqual(positions(self.env)[:2], [(3, 1), (4, 1)])

    def test_npc_can_enter_a_cell_vacated_earlier_in_the_step(self):
        place(self.env, 0, 3, 1)
        place(self.env, 1, 2, 1)
        self.env.step(R, {1: R})
        self.assertEqual(positions(self.env)[:2], [(4, 1), (3, 1)])

    def test_neighbours_swap_through_each_other(self):
        place(self.env, 0, 2, 1)
        place(self.env, 1, 3, 1)
        self.env.step(R, {1: L})
        self.assertEqual(positions(self.env)[:2], [(3, 1), (2, 1)])

    def test_both_reach_a_shared_target_cell(self):
        place(self.env, 0, 2, 1)
        place(self.env, 1, 4, 1)
        self.env.step(R, {1: L})
        self.assertEqual(positions(self.env)[:2], [(3, 1), (3, 1)])
        self.assertFalse(any(c.dead for c in self.env.characters))
        self.assertEqual(self.env.events, [])


class TestBombOwnership(unittest.TestCase):
    def test_bombs_are_tracked_per_owner(self):
        env = BombermanEnv(with_npcs=True)
        state = env.step(W, {1: B})
        self.assertIsNone(env.bomb)
        self.assertIsNone(state["bomb"])
        self.assertEqual(state["bombs"], [{"owner": 1, "x": 9, "y": 1, "countdown": 4}])

        # NPC 1's second bomb is refused; the agent's new bomb keeps its full fuse.
        state = env.step(B, {1: B})
        self.assertEqual(state["bomb"], {"x": 1, "y": 1, "countdown": 4})
        self.assertEqual(
            state["bombs"],
            [
                {"owner": 0, "x": 1, "y": 1, "countdown": 4},
                {"owner": 1, "x": 9, "y": 1, "countdown": 3},
            ],
        )

    def test_npc_can_leave_its_bomb_but_not_reenter(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(W, {1: B}), (W, {1: D}), (W, {1: U})])
        self.assertEqual(positions(env)[1], (9, 2))
        self.assertEqual(env.bombs[1].countdown, 2)

    def test_an_npc_bomb_blocks_the_agent(self):
        env = BombermanEnv(with_npcs=True)
        place(env, 0, 2, 1)
        place(env, 1, 3, 1)
        run(env, [(W, {1: B}), (W, {1: R}), (R, None)])
        self.assertEqual(positions(env)[:2], [(2, 1), (4, 1)])
        self.assertEqual(env.bombs[1].countdown, 2)

    def test_timers_run_independently(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(B, None), (R, {2: B})])
        self.assertEqual(env.bombs[0].countdown, 3)
        self.assertEqual(env.bombs[2].countdown, 4)

        run(env, [(R, {2: U}), (R, {2: U}), (W, {2: U})])
        self.assertNotIn(0, env.bombs)
        self.assertEqual(env.bombs[2].countdown, 1)
        self.assertIn((1, 1), env.explosion)
        self.assertNotIn((1, 9), env.explosion)

        state = env.step(W)
        self.assertEqual(env.bombs, {})
        self.assertIn((1, 9), env.explosion)
        self.assertEqual(state["events"], [block((3, 9), 2)])
        self.assertEqual(positions(env)[:3], [(4, 1), (9, 1), (1, 6)])
        self.assertFalse(any(c.dead for c in env.characters))


class TestSimultaneousExplosions(unittest.TestCase):
    def test_two_blasts_share_one_block_event(self):
        env = BombermanEnv(with_npcs=True)
        place(env, 0, 4, 1)
        place(env, 1, 5, 2)
        run(
            env,
            [(B, {1: B}), (L, {1: D}), (L, {1: D}), (L, None), (W, None)],
        )
        self.assertEqual(
            env.explosion,
            [(4, 1), (5, 1), (3, 1), (2, 1), (5, 2), (5, 3), (5, 4)],
        )
        self.assertNotIn((6, 1), env.explosion)
        self.assertIs(env.grid[1][5], Cell.EMPTY)
        self.assertEqual(env.events, [block((5, 1), 0, 1), death(1, 1)])
        self.assertEqual(env.player, (1, 1))
        self.assertFalse(env.dead)

    def test_co_exploding_bombs_stop_at_each_other(self):
        # Bombs at (1, 1) and (2, 1). The agent's ray stops at (2, 1), so
        # (3, 1) is burned by NPC 1 alone.
        env = BombermanEnv(with_npcs=True)
        place(env, 1, 2, 1)
        run(
            env,
            [(B, {1: B}), (D, {1: R}), (D, None), (D, None), (W, None)],
        )
        self.assertEqual(
            env.explosion, [(1, 1), (2, 1), (1, 2), (1, 3), (3, 1), (4, 1)]
        )
        self.assertEqual(env.events, [death(1, 1)])
        self.assertEqual(env.player, (1, 4))
        self.assertFalse(env.dead)


class TestBlastMeetsBomb(unittest.TestCase):
    def test_blast_stops_at_a_live_bomb_which_keeps_counting_and_kills_later(self):
        env = BombermanEnv(with_npcs=True)
        place(env, 1, 1, 3)
        place(env, 2, 1, 4)
        run(env, [(B, None), (R, {1: B}), (R, None), (R, None)])
        self.assertEqual(env.bombs[1].countdown, 2)

        # The agent's blast burns (1, 3), kills NPC 1 standing on its own
        # bomb there, and stops before NPC 2 at (1, 4).
        state = env.step(W)
        self.assertEqual(env.explosion, [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)])
        self.assertEqual(state["events"], [death(1, 0)])
        self.assertEqual(env.bombs, {1: Bomb(1, 3, 1)})
        self.assertFalse(env.characters[2].dead)

        # The dead owner's bomb still blocks NPC 2 and then kills it.
        state = env.step(W, {1: R, 2: U})
        self.assertEqual(positions(env)[1:3], [(1, 3), (1, 4)])
        self.assertEqual(env.bombs, {})
        self.assertEqual(
            state["events"], [block((2, 3), 1), block((1, 5), 1), death(2, 1)]
        )
        self.assertFalse(env.dead)


class TestDeathAttribution(unittest.TestCase):
    def test_two_blasts_give_one_death_event_with_both_owners(self):
        env = BombermanEnv(with_npcs=True)
        place(env, 1, 3, 1)
        place(env, 2, 2, 1)
        run(
            env,
            [(B, {1: B}), (D, {1: D}), (D, {1: D}), (D, {1: D}), (W, None)],
        )
        self.assertEqual(env.events, [block((5, 1), 1), death(2, 0, 1)])
        self.assertFalse(env.dead)
        self.assertFalse(env.characters[1].dead)

    def test_victim_is_listed_when_its_own_bomb_contributed(self):
        env = BombermanEnv(with_npcs=True)
        place(env, 1, 3, 1)
        run(env, [(B, {1: B}), (D, {1: L}), (D, None), (D, None), (W, None)])
        self.assertEqual(positions(env)[1], (2, 1))
        self.assertEqual(env.events, [block((5, 1), 1), death(1, 0, 1)])

    def test_npc_self_kill(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(W, {3: B})] + [(W, None)] * 4)
        self.assertEqual(env.events, [block((7, 9), 3), death(3, 3)])


class TestDeadNpcs(unittest.TestCase):
    def setUp(self):
        self.env = BombermanEnv(with_npcs=True)
        place(self.env, 3, 9, 4)
        run(self.env, [(W, {1: B})] + [(W, None)] * 4)
        self.assertTrue(self.env.characters[1].dead)

    def test_dead_npc_ignores_actions(self):
        run(self.env, [(W, {1: D}), (W, {1: B})])
        self.assertEqual(positions(self.env)[1], (9, 1))
        self.assertEqual(self.env.bombs, {})

    def test_dead_npc_does_not_block(self):
        run(self.env, [(W, {3: U}), (W, {3: U}), (W, {3: U})])
        self.assertEqual(positions(self.env)[3], (9, 1))
        self.assertEqual(positions(self.env)[1], (9, 1))


class TestNpcElimination(unittest.TestCase):
    def test_all_npcs_dead_is_reported_but_not_terminal(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(W, {1: B}), (W, {2: B}), (W, {3: B}), (W, None)])
        self.assertFalse(env.all_npcs_dead)
        for dead_count in (1, 2):
            state = env.step(W)
            self.assertEqual(sum(c.dead for c in env.characters), dead_count)
            self.assertFalse(state["all_npcs_dead"])

        state = env.step(W)
        self.assertTrue(state["all_npcs_dead"])
        self.assertTrue(env.all_npcs_dead)
        self.assertFalse(env.dead)

        state = env.step(R)
        self.assertEqual(state["steps"], 8)
        self.assertEqual(env.player, (2, 1))


class TestAgentDeathWithNpcs(unittest.TestCase):
    def test_agent_death_freezes_everything_until_reset(self):
        env = BombermanEnv(with_npcs=True)
        run(
            env,
            [(B, None), (W, {1: B}), (W, {1: D}), (W, {1: D}), (W, {1: D})],
        )
        self.assertTrue(env.dead)
        before = env.get_state()
        self.assertEqual(before["events"], [death(0, 0)])
        self.assertEqual(before["bombs"], [{"owner": 1, "x": 9, "y": 1, "countdown": 1}])

        after = env.step(R, {1: D, 2: U, 3: U})
        self.assertEqual(after, before)

        env.reset()
        self.assertFalse(env.dead)
        self.assertEqual(env.bombs, {})


class TestSnapshots(unittest.TestCase):
    def test_new_state_keys_are_copies(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(W, {1: B})] + [(W, None)] * 4)
        state = env.get_state()
        state["events"].clear()
        state["characters"][2]["x"] = 5
        self.assertEqual(env.events, [block((7, 1), 1), death(1, 1)])
        self.assertEqual(positions(env)[2], (1, 9))

        env.reset()
        state = env.step(W, {1: B})
        state["bombs"][0]["countdown"] = 99
        self.assertEqual(env.bombs[1].countdown, 4)

    def test_events_clear_on_the_next_step_and_on_reset(self):
        env = BombermanEnv(with_npcs=True)
        run(env, [(W, {1: B})] + [(W, None)] * 4)
        self.assertTrue(env.events)
        env.step(W)
        self.assertEqual(env.events, [])

        run(env, [(W, {2: B})] + [(W, None)] * 4)
        self.assertTrue(env.events)
        env.reset()
        self.assertEqual(env.events, [])


if __name__ == "__main__":
    unittest.main()
