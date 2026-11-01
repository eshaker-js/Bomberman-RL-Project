"""Headless checks for the track B baseline bots. Pygame is not required.

The bots live in ``agents/`` and only read ``get_state()``; these tests never
change engine rules. A few tests edit a fresh env's public attributes (grid,
characters, gate) to build a known situation, like the other test files do.
"""

import copy
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RandomBot, RuleBot  # noqa: E402
from agents.evaluate import agent_seat, mixed_table  # noqa: E402
from agents.match import play_game  # noqa: E402
from agents.rule_bot import bombs_from_state, danger_times, plan_bombs  # noqa: E402
from game.environment import (  # noqa: E402
    GATE_POSITION,
    Action,
    BombermanEnv,
    Cell,
    Gate,
)

W = Action.WAIT


def put(env: BombermanEnv, character_id: int, x: int, y: int) -> None:
    """Test-only setup: move a character of a fresh env to a floor cell."""
    assert env.steps == 0 and not env.bombs, "fresh env only"
    assert env.grid[y][x] is Cell.EMPTY
    character = env.characters[character_id]
    character.x, character.y = x, y


def all_bots(kind):
    return {i: kind() for i in range(4)}


class TestRandomBot(unittest.TestCase):
    def test_same_seed_same_actions(self):
        state = BombermanEnv().reset()
        a, b = RandomBot(7), RandomBot(7)
        self.assertEqual([a.act(state, 0) for _ in range(50)],
                         [b.act(state, 0) for _ in range(50)])

    def test_reset_restarts_the_sequence(self):
        state = BombermanEnv().reset()
        bot = RandomBot()
        bot.reset(3)
        first = [bot.act(state, 0) for _ in range(20)]
        bot.reset(3)
        self.assertEqual([bot.act(state, 0) for _ in range(20)], first)

    def test_uses_every_action(self):
        state = BombermanEnv().reset()
        bot = RandomBot(0)
        seen = {bot.act(state, 0) for _ in range(300)}
        self.assertEqual(seen, set(Action))


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_game(self):
        for kind in (RandomBot, RuleBot):
            with self.subTest(kind=kind.__name__):
                first = play_game(all_bots(kind), seed=11, max_steps=150)
                second = play_game(all_bots(kind), seed=11, max_steps=150)
                self.assertEqual(first, second)

    def test_mixed_game_is_reproducible(self):
        def bots():
            return {0: RuleBot(), 1: RandomBot(), 2: RuleBot(), 3: RandomBot()}

        self.assertEqual(play_game(bots(), seed=5), play_game(bots(), seed=5))


class TestDangerPrediction(unittest.TestCase):
    def test_chain_reaction_moves_the_later_bomb_earlier(self):
        grid = BombermanEnv().reset()["grid"]
        # A at (1, 1) goes off next action and its ray reaches B at (3, 1).
        planned = plan_bombs(grid, [(1, 1, 1, 1), (3, 1, 4, 1)])
        self.assertEqual([b.explode_at for b in planned], [1, 1])
        danger = danger_times(planned)
        # (4, 1) is only in B's blast, so it burns on the next action too.
        self.assertEqual(danger[(4, 1)], {1})

    def test_prediction_matches_the_engine(self):
        """From many real game states, cells predicted to burn on the next
        action are exactly the cells the engine burns when everyone waits."""
        rng = random.Random(0)
        checked = 0
        for seed in range(15):
            env = BombermanEnv(with_npcs=True, random_crates=True)
            state = env.reset(seed=seed)
            bots = all_bots(RandomBot)
            for i, bot in bots.items():
                bot.reset(seed * 10 + i)
            for _ in range(60):
                if state["outcome"] is not None:
                    break
                if state["bombs"]:
                    planned = plan_bombs(state["grid"], bombs_from_state(state))
                    danger = danger_times(planned)
                    predicted = {cell for cell, ts in danger.items() if 1 in ts}
                    probe = copy.deepcopy(env)
                    probe.step(W, {1: W, 2: W, 3: W})
                    self.assertEqual(set(probe.explosion), predicted)
                    checked += 1
                actions = {c["id"]: bots[c["id"]].act(state, c["id"])
                           for c in state["characters"] if not c["dead"]}
                agent = actions.pop(0, W)
                if rng.random() < 0.5:
                    agent = W  # keep the agent alive longer for more states
                state = env.step(agent, actions)
        self.assertGreater(checked, 100)


class TestRuleBotBehaviour(unittest.TestCase):
    def test_flees_a_bomb_and_survives(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 1, 2, 1)
        state = env.step(W, {1: Action.BOMB})  # NPC 1 bombs (2, 1); agent at (1, 1)
        bot = RuleBot(0)
        first = bot.act(state, 0)
        self.assertEqual(first, Action.DOWN)  # the only way out of the row
        state = env.step(first)
        for _ in range(5):
            state = env.step(bot.act(state, 0))
        self.assertFalse(state["characters"][0]["dead"])
        self.assertTrue(state["characters"][1]["dead"])

    def test_bombs_a_crate_when_it_can_escape(self):
        env = BombermanEnv()
        env.grid[1][2] = Cell.BLOCK  # crate right next to the start
        self.assertEqual(RuleBot(0).act(env.get_state(), 0), Action.BOMB)

    def test_never_bombs_without_an_escape(self):
        env = BombermanEnv()
        env.grid[1][2] = Cell.BLOCK  # crate to the right
        env.grid[3][1] = Cell.BLOCK  # only (1, 2) is reachable: inside the blast
        bot = RuleBot(0)
        state = env.get_state()
        self.assertNotEqual(bot.act(state, 0), Action.BOMB)
        for _ in range(30):
            state = env.step(bot.act(state, 0))
        self.assertFalse(state["dead"])
        self.assertEqual(state["bombs"], [])

    def test_survives_its_own_bomb_next_to_a_ready_opponent(self):
        """It only bombs if it could still escape should the opponent next
        to it drop a bomb on the same action."""
        env = BombermanEnv(with_npcs=True)
        env.grid[1][2] = Cell.BLOCK
        put(env, 1, 1, 2)  # NPC 1 stands on the agent's only exit
        state = env.get_state()
        self.assertNotEqual(RuleBot(0).act(state, 0), Action.BOMB)

    def test_walks_onto_the_open_gate(self):
        env = BombermanEnv(with_npcs=True)
        put(env, 0, 5, 4)
        for npc in env.characters[1:]:
            npc.dead = True
        gx, gy = GATE_POSITION
        env.grid[gy][gx] = Cell.EMPTY
        env.gate = Gate(gx, gy, "open")
        state = env.step(RuleBot(0).act(env.get_state(), 0))
        self.assertEqual(state["outcome"], "win")

    def test_clears_the_map_alone_without_dying(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                env = BombermanEnv(random_crates=True)
                state = env.reset(seed=seed)
                bot = RuleBot(seed)
                destroyed = 0
                for _ in range(150):
                    state = env.step(bot.act(state, 0))
                    destroyed += sum(
                        e["type"] == "block_destroyed" for e in state["events"]
                    )
                self.assertFalse(state["dead"])
                self.assertGreaterEqual(destroyed, 15)

    def test_plays_from_any_seat(self):
        for seat in (1, 2, 3):
            with self.subTest(seat=seat):
                bots = all_bots(RandomBot)
                bots[seat] = RuleBot()
                result = play_game(bots, seed=seat)
                self.assertNotIn(seat, result.suicides)


class TestRuleBeatsRandom(unittest.TestCase):
    """The track B done-check and our first evaluation sanity check."""

    def test_agent_seat(self):
        rule = agent_seat("rule", games=25)
        rand = agent_seat("random", games=25)
        self.assertGreaterEqual(rule["win"], 23)
        self.assertEqual(rand["win"], 0)
        self.assertLess(rule["loss"], rand["loss"])

    def test_mixed_table(self):
        stats = mixed_table(games=24)
        self.assertLess(stats["rule"]["deaths"] * 5, stats["random"]["deaths"])
        self.assertGreater(stats["rule"]["crates"], stats["random"]["crates"])


if __name__ == "__main__":
    unittest.main()
