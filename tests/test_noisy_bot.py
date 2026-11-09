"""The noisy rule bot and the opponent ladder (Task 6, stage 2).

Headless; pygame is not required. Full games are played with ``GAME_RULES``
so the bots see the real timing.
"""

import contextlib
import io
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import ladder_eval  # noqa: E402
from agents.match import bot_seed  # noqa: E402
from agents.noisy_bot import LADDER, NoisyBot, make_bot  # noqa: E402
from agents.random_bot import RandomBot  # noqa: E402
from agents.rule_bot import RuleBot  # noqa: E402
from game.environment import GAME_RULES, NPC_IDS, Action, BombermanEnv  # noqa: E402


def record_game(make_agent, seed: int, max_steps: int = 150) -> list:
    """Actions the agent (ID 0) picks in one seeded game vs 3 random bots."""
    env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
    state = env.reset(seed=seed)
    agent = make_agent()
    agent.reset(bot_seed(seed, 0))
    npcs = {i: RandomBot() for i in NPC_IDS}
    for i, bot in npcs.items():
        bot.reset(bot_seed(seed, i))
    actions = []
    for _ in range(max_steps):
        mine = agent.act(state, 0)
        actions.append(mine)
        others = {c["id"]: npcs[c["id"]].act(state, c["id"])
                  for c in state["characters"][1:] if not c["dead"]}
        state = env.step(mine, others)
        if state["outcome"] is not None:
            break
    return actions


class NoisyBotTest(unittest.TestCase):
    def test_same_seed_same_actions(self):
        for eps in (0.1, 0.5):
            with self.subTest(eps=eps):
                a = record_game(lambda: NoisyBot(eps), 4)
                b = record_game(lambda: NoisyBot(eps), 4)
                self.assertEqual(a, b)

    def test_reset_restarts_the_sequence(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        state = env.reset(seed=0)
        bot = NoisyBot(0.5)
        bot.reset(7)
        first = [bot.act(state, 0) for _ in range(30)]
        bot.reset(7)
        again = [bot.act(state, 0) for _ in range(30)]
        self.assertEqual(first, again)

    def test_different_seeds_differ(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        state = env.reset(seed=0)
        runs = []
        for seed in (1, 2):
            bot = NoisyBot(0.5)
            bot.reset(seed)
            runs.append([bot.act(state, 0) for _ in range(40)])
        self.assertNotEqual(runs[0], runs[1])

    def test_epsilon_zero_is_exactly_rule_bot(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                noisy = record_game(lambda: NoisyBot(0.0), seed, max_steps=400)
                rule = record_game(RuleBot, seed, max_steps=400)
                self.assertEqual(noisy, rule)

    def test_epsilon_zero_as_npc(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        state = env.reset(seed=2)
        noisy, rule = NoisyBot(0.0), RuleBot()
        noisy.reset(5)
        rule.reset(5)
        for me in NPC_IDS:
            self.assertEqual(noisy.act(state, me), rule.act(state, me))

    def test_epsilon_one_is_uniform_random(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        state = env.reset(seed=0)
        bot = NoisyBot(1.0)
        bot.reset(3)
        counts = Counter(bot.act(state, 0) for _ in range(6000))
        self.assertEqual(set(counts), set(Action))
        for action in Action:
            self.assertGreater(counts[action], 850)
            self.assertLess(counts[action], 1150)

    def test_noise_rate_is_about_epsilon(self):
        """In a state where RuleBot always waits, other actions are noise."""
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        state = env.reset(seed=0)
        state["outcome"] = "loss"  # RuleBot waits once the game is over
        bot = NoisyBot(0.5)
        bot.reset(11)
        non_wait = sum(bot.act(state, 0) is not Action.WAIT for _ in range(6000))
        # P(not WAIT) = 0.5 * 5/6 = 0.4167
        self.assertAlmostEqual(non_wait / 6000, 0.5 * 5 / 6, delta=0.03)

    def test_bad_epsilon_is_rejected(self):
        for bad in (-0.1, 1.5, "0.5", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    NoisyBot(bad)


class LadderTest(unittest.TestCase):
    def test_levels_weakest_first(self):
        self.assertEqual(list(LADDER), ["random", "noisy50", "noisy25", "noisy10", "rule"])

    def test_level_types_and_epsilon(self):
        self.assertIsInstance(make_bot("random"), RandomBot)
        self.assertIsInstance(make_bot("rule"), RuleBot)
        for name, eps in (("noisy50", 0.5), ("noisy25", 0.25), ("noisy10", 0.1)):
            bot = make_bot(name)
            self.assertIsInstance(bot, NoisyBot)
            self.assertEqual(bot.epsilon, eps)

    def test_factories_return_fresh_bots(self):
        for name in LADDER:
            self.assertIsNot(make_bot(name), make_bot(name))

    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            make_bot("expert")

    def test_ladder_eval_main_runs(self):
        out = io.StringIO()
        argv = ["ladder_eval", "--games", "1", "--max-steps", "30", "--game-rules"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
            ladder_eval.main()
        text = out.getvalue()
        self.assertIn(f"GAME_RULES (fuse {GAME_RULES['bomb_fuse']}, flames {GAME_RULES['flame_ticks']})", text)
        for name in LADDER:
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
