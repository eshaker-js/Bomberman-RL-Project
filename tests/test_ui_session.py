"""Headless checks for game sessions (ui/session.py). Pygame is not required.

A session must create the right environment and bots for each mode, and must
call the bots exactly as ``agents/match.py`` does, so a UI game is the same
game the evaluation scripts play.
"""

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RandomBot, RuleBot  # noqa: E402
from agents.match import bot_seed  # noqa: E402
from game.environment import AGENT_ID, NPC_IDS, Action, BombermanEnv  # noqa: E402
from ui.session import MODE_BOTS, MODE_FREE, GameSession  # noqa: E402


def reference_states(seed: int, bot_classes, human_actions):
    """States of a game played the way agents/match.py plays one."""
    env = BombermanEnv(with_npcs=True, random_crates=True)
    state = env.reset(seed=seed)
    bots = {npc_id: cls() for npc_id, cls in zip(NPC_IDS, bot_classes)}
    for npc_id, bot in bots.items():
        bot.reset(bot_seed(seed, npc_id))
    states = [state]
    for action in human_actions:
        actions = {
            c["id"]: bots[c["id"]].act(state, c["id"])
            for c in state["characters"]
            if c["id"] != AGENT_ID and not c["dead"]
        }
        state = env.step(action, actions)
        states.append(state)
    return states


class TestModes(unittest.TestCase):
    def test_free_world_is_single_player_with_random_crates(self):
        session = GameSession(MODE_FREE, seed=5)
        self.assertFalse(session.env.with_npcs)
        self.assertTrue(session.env.random_crates)
        self.assertEqual(session.bots, {})
        self.assertEqual(len(session.cur["characters"]), 1)
        self.assertEqual(session.cur, BombermanEnv(random_crates=True).reset(seed=5))

    def test_player_vs_bots_has_three_npcs_and_random_crates(self):
        session = GameSession(MODE_BOTS, seed=5)
        self.assertTrue(session.env.with_npcs)
        self.assertTrue(session.env.random_crates)
        self.assertEqual(sorted(session.bots), list(NPC_IDS))
        self.assertEqual(
            session.cur, BombermanEnv(with_npcs=True, random_crates=True).reset(seed=5)
        )

    def test_default_opponents_are_rule_bots(self):
        session = GameSession(MODE_BOTS, seed=1)
        self.assertEqual([type(session.bots[i]) for i in NPC_IDS], [RuleBot] * 3)

    def test_opponent_sets(self):
        expected = {
            "random": [RandomBot, RandomBot, RandomBot],
            "rule": [RuleBot, RuleBot, RuleBot],
            "mixed": [RuleBot, RuleBot, RandomBot],
        }
        for opponents, classes in expected.items():
            with self.subTest(opponents=opponents):
                session = GameSession(MODE_BOTS, seed=1, opponents=opponents)
                self.assertEqual([type(session.bots[i]) for i in NPC_IDS], classes)

    def test_bad_mode_or_opponents_rejected(self):
        with self.assertRaises(ValueError):
            GameSession("arcade", seed=1)
        with self.assertRaises(ValueError):
            GameSession(MODE_BOTS, seed=1, opponents="smart")

    def test_labels(self):
        session = GameSession(MODE_BOTS, seed=1, opponents="mixed")
        self.assertEqual(session.label(0), "YOU")
        self.assertEqual(session.label(1), "BOT 1 RULE")
        self.assertEqual(session.label(3), "BOT 3 RANDOM")


class TestAdvance(unittest.TestCase):
    def test_bots_called_exactly_like_match(self):
        rng = random.Random(3)
        actions = [rng.choice(tuple(Action)) for _ in range(120)]
        for opponents, classes in (
            ("rule", (RuleBot, RuleBot, RuleBot)),
            ("random", (RandomBot, RandomBot, RandomBot)),
            ("mixed", (RuleBot, RuleBot, RandomBot)),
        ):
            for seed in (0, 7, 123456):
                with self.subTest(opponents=opponents, seed=seed):
                    expected = reference_states(seed, classes, actions)
                    session = GameSession(MODE_BOTS, seed, opponents)
                    got = [session.cur]
                    for action in actions:
                        got.append(session.advance(action))
                    self.assertEqual(got, expected)

    def test_prev_and_cur_are_the_last_two_states(self):
        session = GameSession(MODE_BOTS, seed=2)
        first = session.cur
        second = session.advance(Action.WAIT)
        self.assertEqual(session.prev, first)
        self.assertEqual(session.cur, second)
        self.assertEqual(second["steps"], 1)

    def test_dead_bots_are_not_asked(self):
        class Spy:
            def __init__(self):
                self.calls = 0

            def reset(self, seed=None):
                pass

            def act(self, state, me):
                self.calls += 1
                return Action.WAIT

        session = GameSession(MODE_BOTS, seed=2)
        spies = {npc_id: Spy() for npc_id in NPC_IDS}
        session.bots = spies
        session.env.characters[2].dead = True  # test-only setup
        session.cur = session.env.get_state()
        session.advance(Action.WAIT)
        self.assertEqual([spies[i].calls for i in NPC_IDS], [1, 0, 1])

    def test_free_world_steps_without_npc_actions(self):
        session = GameSession(MODE_FREE, seed=4)
        state = session.advance(Action.RIGHT)
        self.assertEqual(state["steps"], 1)

    def test_over_after_agent_death(self):
        session = GameSession(MODE_FREE, seed=4)
        session.advance(Action.BOMB)
        for _ in range(4):
            session.advance(Action.WAIT)
        self.assertTrue(session.cur["dead"])
        self.assertEqual(session.cur["outcome"], "loss")
        self.assertTrue(session.over)


if __name__ == "__main__":
    unittest.main()
