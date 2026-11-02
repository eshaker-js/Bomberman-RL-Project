"""Headless checks for configurable bomb fuse and lingering flames. Pygame is not required.

``BombermanEnv(bomb_fuse=..., flame_ticks=...)``: a bomb explodes
``bomb_fuse`` actions after the action it was placed on, and a burned cell
keeps burning (and killing) for ``flame_ticks`` actions, counting the action
of the explosion. The defaults (4 and 1) are the original rules, which the
other test files keep checking.

On the fixed map the agent bombs its spawn (1, 1) and walks right to (4, 1);
that blast burns (1, 1), (2, 1), (3, 1), (1, 2) and (1, 3). A few tests set
up a lingering flame or the gate directly on the env's public attributes,
like the other test files do, to reach one situation on a known action.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.environment import (  # noqa: E402
    BLAST_RANGE,
    BOMB_FUSE,
    FLAME_TICKS,
    GAME_RULES,
    GATE_POSITION,
    Action,
    BombermanEnv,
    Cell,
    Gate,
)

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB

ESCAPE_RIGHT = [B, R, R, R]
BURNED = {(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)}


def run(env: BombermanEnv, actions, npc_actions=None) -> dict:
    state = env.get_state()
    for action in actions:
        state = env.step(action, npc_actions)
    return state


def flames(state: dict) -> dict:
    return {(f["x"], f["y"]): (f["ticks_left"], f["owners"]) for f in state["flames"]}


def deaths(state: dict) -> list:
    return [e for e in state["events"] if e["type"] == "death"]


def lingering_env(**kwargs) -> BombermanEnv:
    """Fixed map, fuse 4, flames for 4 actions: the agent's blast happens on action 5."""
    return BombermanEnv(bomb_fuse=4, flame_ticks=4, **kwargs)


def only_npc_1_alive(env: BombermanEnv) -> None:
    """Test-only setup: NPCs 2 and 3 are already dead."""
    env.characters[2].dead = True
    env.characters[3].dead = True


class TestSettings(unittest.TestCase):
    def test_defaults_are_the_original_rules(self):
        env = BombermanEnv()
        self.assertEqual((env.bomb_fuse, env.flame_ticks), (BOMB_FUSE, FLAME_TICKS))
        self.assertEqual((BOMB_FUSE, FLAME_TICKS), (4, 1))
        self.assertEqual(
            env.get_state()["rules"],
            {"bomb_fuse": 4, "blast_range": BLAST_RANGE, "flame_ticks": 1},
        )
        self.assertEqual(env.get_state()["flames"], [])

    def test_game_rules_values_and_read_only(self):
        self.assertEqual(dict(GAME_RULES), {"bomb_fuse": 9, "flame_ticks": 4})
        with self.assertRaises(TypeError):
            GAME_RULES["bomb_fuse"] = 3  # type: ignore[index]

    def test_env_from_game_rules(self):
        state = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES).reset(seed=3)
        self.assertEqual(state["rules"], {"bomb_fuse": 9, "blast_range": 2, "flame_ticks": 4})

    def test_bad_settings_rejected(self):
        for name in ("bomb_fuse", "flame_ticks"):
            for value in (0, -1, 1.5, "4", None, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        BombermanEnv(**{name: value})

    def test_state_is_a_copy(self):
        env = lingering_env()
        state = run(env, ESCAPE_RIGHT + [W])
        state["flames"].clear()
        state["rules"]["flame_ticks"] = 99
        again = env.get_state()
        self.assertEqual(len(again["flames"]), len(BURNED))
        self.assertEqual(again["rules"]["flame_ticks"], 4)

    def test_reset_puts_out_flames(self):
        env = lingering_env()
        run(env, ESCAPE_RIGHT + [W])
        self.assertTrue(env.reset()["flames"] == [] and env.flames == {})


class TestFuse(unittest.TestCase):
    def test_fuse_9_explodes_9_actions_after_placing(self):
        env = BombermanEnv(**GAME_RULES)
        state = env.step(B)
        self.assertEqual(state["bombs"], [{"owner": 0, "x": 1, "y": 1, "countdown": 9}])
        state = run(env, [R, R, R] + [W] * 5)
        self.assertEqual(state["steps"], 9)
        self.assertEqual(state["explosion"], [])
        self.assertEqual(state["bombs"][0]["countdown"], 1)
        state = env.step(W)
        self.assertEqual(state["steps"], 10)
        self.assertEqual(set(state["explosion"]), BURNED)
        self.assertEqual(state["bombs"], [])
        self.assertFalse(state["dead"])

    def test_short_fuse(self):
        env = BombermanEnv(bomb_fuse=3)
        state = run(env, [B, D, D, D])
        self.assertEqual(state["steps"], 4)
        self.assertIn((1, 1), state["explosion"])
        self.assertFalse(state["dead"])


class TestFlameDuration(unittest.TestCase):
    def test_flames_burn_exactly_flame_ticks_actions(self):
        env = lingering_env()
        state = run(env, ESCAPE_RIGHT + [W])
        self.assertEqual(flames(state), {cell: (4, (0,)) for cell in BURNED})
        self.assertEqual(set(state["explosion"]), BURNED)
        for ticks_left in (3, 2, 1):
            state = env.step(W)
            self.assertEqual(flames(state), {cell: (ticks_left, (0,)) for cell in BURNED})
            self.assertEqual(state["explosion"], [])
        state = env.step(W)
        self.assertEqual(state["flames"], [])

    def test_one_tick_flames_are_the_original_rule(self):
        env = BombermanEnv()
        state = run(env, ESCAPE_RIGHT + [W])
        self.assertEqual(flames(state), {cell: (1, (0,)) for cell in BURNED})
        self.assertEqual(env.step(W)["flames"], [])

    def test_flames_match_explosion_by_default_over_random_games(self):
        import random
        for seed in range(30):
            rng = random.Random(seed)
            env = BombermanEnv(with_npcs=True, random_crates=True)
            state = env.reset(seed=seed)
            while state["outcome"] is None and state["steps"] < 200:
                npc = {i: rng.choice(tuple(Action)) for i in (1, 2, 3)}
                state = env.step(rng.choice(tuple(Action)), npc)
                self.assertEqual(sorted(flames(state)), sorted(state["explosion"]))

    def test_same_seed_same_game_with_game_rules(self):
        import random
        runs = []
        for _ in range(2):
            rng = random.Random(5)
            env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
            states = [env.reset(seed=5)]
            for _ in range(150):
                npc = {i: rng.choice(tuple(Action)) for i in (1, 2, 3)}
                states.append(env.step(rng.choice(tuple(Action)), npc))
            runs.append(states)
        self.assertEqual(runs[0], runs[1])


class TestLingeringDeaths(unittest.TestCase):
    def test_walking_into_a_lingering_flame_kills(self):
        env = lingering_env()
        run(env, ESCAPE_RIGHT + [W])
        state = env.step(L)
        self.assertEqual(state["player"], (3, 1))  # flames do not block
        self.assertTrue(state["dead"])
        self.assertEqual(deaths(state), [{"type": "death", "character": 0, "killed_by": (0,)}])
        self.assertEqual(state["outcome"], "loss")

    def test_last_burning_action_still_kills(self):
        env = lingering_env()
        state = run(env, ESCAPE_RIGHT + [W, W, W, L])
        self.assertEqual(state["steps"], 8)
        self.assertTrue(state["dead"])

    def test_cell_is_safe_once_the_flame_is_out(self):
        env = lingering_env()
        state = run(env, ESCAPE_RIGHT + [W, W, W, W, L, L, L])
        self.assertEqual(state["player"], (1, 1))
        self.assertFalse(state["dead"])
        self.assertIsNone(state["outcome"])

    def test_standing_on_a_burning_cell_kills(self):
        env = lingering_env(with_npcs=True)
        env.flames[(9, 1)] = (3, (2,))  # test-only: NPC 1 stands in a lingering flame
        state = env.step(W, {1: W})
        self.assertTrue(state["characters"][1]["dead"])
        self.assertEqual(deaths(state), [{"type": "death", "character": 1, "killed_by": (2,)}])

    def test_leaving_a_burning_cell_on_the_same_action_survives(self):
        env = lingering_env(with_npcs=True)
        env.flames[(9, 1)] = (3, (2,))  # test-only
        state = env.step(W, {1: L})
        self.assertFalse(state["characters"][1]["dead"])

    def test_deaths_in_id_order_with_cell_owners(self):
        env = lingering_env(with_npcs=True)
        only_npc_1_alive(env)
        env.flames[(2, 1)] = (3, (1,))  # test-only
        env.flames[(8, 1)] = (3, (0, 3))
        state = env.step(R, {1: L})
        self.assertEqual(
            deaths(state),
            [
                {"type": "death", "character": 0, "killed_by": (1,)},
                {"type": "death", "character": 1, "killed_by": (0, 3)},
            ],
        )


class TestFlamesDoNotInteract(unittest.TestCase):
    def test_rays_pass_burning_cells_and_fresh_blast_refreshes_them(self):
        env = lingering_env()
        run(env, [B, D, D, D])
        env.flames[(2, 1)] = (2, (1,))  # test-only: an older NPC blast, 2 actions left
        state = env.step(W)
        self.assertIn((3, 1), state["explosion"])  # the ray did not stop at (2, 1)
        self.assertEqual(flames(state)[(2, 1)], (4, (0,)))
        self.assertEqual(flames(state)[(3, 1)], (4, (0,)))
        self.assertFalse(any(e["type"] == "bomb_triggered" for e in state["events"]))

    def test_lingering_flames_do_not_set_off_bombs(self):
        env = lingering_env()
        run(env, [B, D])
        env.flames[(1, 1)] = (4, (1,))  # test-only: the bomb's cell is burning
        state = env.step(D)
        self.assertEqual(state["bombs"], [{"owner": 0, "x": 1, "y": 1, "countdown": 2}])
        self.assertEqual(state["explosion"], [])
        state = run(env, [D, W])
        self.assertEqual(state["steps"], 5)
        self.assertIn((1, 1), state["explosion"])  # on its own fuse, not earlier
        self.assertEqual(state["bombs"], [])
        self.assertFalse(any(e["type"] == "bomb_triggered" for e in state["events"]))


class TestOutcomes(unittest.TestCase):
    def test_agent_and_last_npc_die_in_lingering_flames_is_a_draw(self):
        env = lingering_env(with_npcs=True)
        only_npc_1_alive(env)
        env.flames[(2, 1)] = (3, (1,))  # test-only
        env.flames[(8, 1)] = (3, (0,))
        state = env.step(R, {1: L})
        self.assertEqual(state["outcome"], "draw")

    def test_last_npc_dying_in_a_lingering_flame_opens_the_gate(self):
        env = lingering_env(with_npcs=True)
        only_npc_1_alive(env)
        gx, gy = GATE_POSITION
        env.grid[gy][gx] = Cell.EMPTY  # test-only: the gate is already revealed
        env.gate = Gate(gx, gy, "closed")
        env.flames[(8, 1)] = (3, (0,))
        state = env.step(W, {1: L})
        self.assertEqual(state["explosion"], [])
        self.assertEqual(
            state["events"],
            [{"type": "death", "character": 1, "killed_by": (0,)}, {"type": "gate_opened"}],
        )
        self.assertEqual(state["gate"]["state"], "open")
        self.assertIsNone(state["outcome"])

    def open_gate_next_to_agent(self) -> BombermanEnv:
        env = lingering_env(with_npcs=True)
        for npc in env.characters[1:]:
            npc.dead = True  # test-only
        gx, gy = GATE_POSITION
        env.grid[gy][gx] = Cell.EMPTY
        env.gate = Gate(gx, gy, "open")
        env.characters[0].x, env.characters[0].y = gx - 1, gy
        return env

    def test_stepping_onto_a_burning_open_gate_is_a_loss(self):
        env = self.open_gate_next_to_agent()
        env.flames[GATE_POSITION] = (2, (1,))
        state = env.step(R)
        self.assertEqual(state["player"], GATE_POSITION)
        self.assertTrue(state["dead"])
        self.assertEqual(state["outcome"], "loss")

    def test_stepping_onto_the_open_gate_after_its_flame_is_out_wins(self):
        env = self.open_gate_next_to_agent()
        env.flames[GATE_POSITION] = (1, (1,))  # goes out at the start of the next action
        state = env.step(R)
        self.assertFalse(state["dead"])
        self.assertEqual(state["outcome"], "win")


if __name__ == "__main__":
    unittest.main()
