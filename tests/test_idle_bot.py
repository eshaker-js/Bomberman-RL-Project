"""Idle bots: a training-only opponent that always WAITs (Task 11, part 1).

- ``IdleBot`` returns ``WAIT`` from every state, for every character, with
  or without a seed;
- ``"idle"`` works as the training opponents of ``BombermanGymEnv`` and of
  a tiny PPO run, and the idle bots never move or bomb;
- idle bots die to the agent's blasts, so the gate opens and a scripted game
  (the rule bot playing the agent) ends in a win;
- ``"idle"`` stays out of the ladder and the evaluation: ``LADDER`` and
  ``LEVELS`` are unchanged, and ``eval_opponents``, the evaluation protocol
  and ``rl.evaluate_agent`` refuse it. A run against idle bots must name its
  ``eval_opponents``.

The engine tests need nothing beyond the standard library; the RL tests need
numpy, gymnasium, torch and stable-baselines3, like the other RL tests. Only
few-step runs, in temporary folders; every env and file is closed.
"""

import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.idle_bot import TRAINING_ONLY_OPPONENTS, IdleBot  # noqa: E402
from agents.noisy_bot import LADDER  # noqa: E402
from agents.random_bot import ACTIONS as BOT_ACTIONS  # noqa: E402
from agents.rule_bot import RuleBot  # noqa: E402
from game.environment import (  # noqa: E402
    AGENT_ID,
    GAME_RULES,
    NPC_IDS,
    Action,
    BombermanEnv,
)

try:
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover - depends on the machine
    gymnasium = None
try:
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING_GYM = "numpy/gymnasium not installed; test skipped"
MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if gymnasium is not None:
    from rl.bomberman_env import ACTIONS, TRAINING_OPPONENTS, BombermanGymEnv
    from rl.config import RewardConfig
if gymnasium is not None and torch is not None:
    from rl import eval_protocol, evaluate_agent, train_ppo
    from rl.config import PPOConfig, apply_overrides

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}
# The ladder exactly as it was before Task 11, weakest first.
LADDER_NAMES = ("random", "noisy50", "noisy25", "noisy10", "rule")
# A game the rule bot, playing the agent, wins against three idle bots (it
# wins every seed 0-39; this one is short). Found once, fixed here.
WIN_SEED = 6


class IdleBotTest(unittest.TestCase):
    def test_always_waits_from_every_state(self):
        # States from real games of the full rules, every character's view.
        rng = random.Random(0)
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        bots = [IdleBot(), IdleBot(7)]
        checked = 0
        for seed in range(25):
            state = env.reset(seed=seed)
            for bot in bots:
                bot.reset(seed)
            while state["outcome"] is None and state["steps"] < 150:
                for me in (AGENT_ID, *NPC_IDS):
                    for bot in bots:
                        self.assertIs(bot.act(state, me), Action.WAIT)
                        checked += 1
                state = env.step(rng.choice(BOT_ACTIONS),
                                 {npc: rng.choice(BOT_ACTIONS) for npc in NPC_IDS})
        self.assertGreater(checked, 1000)

    def test_has_the_bot_interface(self):
        bot = IdleBot()
        bot.reset()
        bot.reset(123)
        self.assertIs(bot.act({}, 1), Action.WAIT)
        self.assertIs(TRAINING_ONLY_OPPONENTS["idle"], IdleBot)
        self.assertIsInstance(TRAINING_ONLY_OPPONENTS["idle"](), IdleBot)


class LadderUnchangedTest(unittest.TestCase):
    def test_ladder_is_unchanged_and_has_no_idle(self):
        self.assertEqual(tuple(LADDER), LADDER_NAMES)
        self.assertNotIn("idle", LADDER)
        self.assertEqual(set(TRAINING_ONLY_OPPONENTS), {"idle"})

    @unittest.skipIf(gymnasium is None or torch is None, MISSING)
    def test_evaluation_levels_are_unchanged(self):
        self.assertEqual(eval_protocol.LEVELS, LADDER_NAMES)
        self.assertNotIn("idle", eval_protocol.LEVELS)

    @unittest.skipIf(gymnasium is None, MISSING_GYM)
    def test_training_accepts_the_ladder_then_idle(self):
        self.assertEqual(TRAINING_OPPONENTS, (*LADDER_NAMES, "idle"))


@unittest.skipIf(gymnasium is None, MISSING_GYM)
class IdleOpponentsEnvTest(unittest.TestCase):
    def test_three_idle_bots_that_never_move_or_bomb(self):
        env = BombermanGymEnv("idle")
        self.assertTrue(env.engine.with_npcs)
        self.assertEqual(sorted(env.controllers), list(NPC_IDS))
        self.assertTrue(all(isinstance(c, IdleBot) for c in env.controllers.values()))
        env.reset(seed=3)
        start = {c["id"]: (c["x"], c["y"]) for c in env.state["characters"] if c["id"] in NPC_IDS}
        rng = random.Random(1)
        for _ in range(60):
            moves = [a for a in range(len(ACTIONS)) if ACTIONS[a] is not Action.BOMB]
            _, _, terminated, truncated, _ = env.step(rng.choice(moves))
            for c in env.state["characters"]:
                if c["id"] in NPC_IDS:
                    self.assertEqual((c["x"], c["y"]), start[c["id"]])
            self.assertFalse([b for b in env.state["bombs"] if b["owner"] in NPC_IDS])
            if terminated or truncated:
                break
        env.close()

    def test_unknown_opponents_are_refused(self):
        with self.assertRaisesRegex(ValueError, "idle"):
            BombermanGymEnv("idle2")

    def test_scripted_game_against_idle_bots_ends_in_a_win(self):
        env = BombermanGymEnv("idle")
        env.reset(seed=WIN_SEED)
        player = RuleBot()
        player.reset(WIN_SEED)
        events, rewards, done = [], [], False
        while not done:
            action = ACTIONS.index(player.act(env.state, AGENT_ID))
            _, reward, terminated, truncated, info = env.step(action)
            events += info["events"]
            rewards.append((reward, info["reward_terms"]))
            done = terminated or truncated
        env.close()
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["outcome"], "win")
        deaths = [e for e in events if e["type"] == "death"]
        self.assertEqual(sorted(e["character"] for e in deaths), list(NPC_IDS))
        for death in deaths:
            self.assertIn(AGENT_ID, death["killed_by"])
        self.assertEqual(sum(e["type"] == "gate_opened" for e in events), 1)
        last_reward, last_terms = rewards[-1]
        self.assertEqual(last_terms["outcome"], RewardConfig().win)
        self.assertEqual(sum(t["kill"] for _, t in rewards), 3 * RewardConfig().kill)


@unittest.skipIf(gymnasium is None or torch is None, MISSING)
class IdleTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        config = cls.tmp / "idle.json"
        config.write_text(json.dumps({**TINY, "opponents": "idle",
                                      "eval_opponents": "random"}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.run_dir = train_ppo.main(["--seed", "0", "--runs-dir", str(cls.tmp / "runs"),
                                      "--config", str(config), "--timesteps", "128",
                                      "--eval-every", "128", "--eval-games", "2"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_tiny_run_against_idle_bots(self):
        saved = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["opponents"], saved["eval_opponents"]), ("idle", "random"))
        meta = json.loads((self.run_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(meta["finished_at"])
        self.assertEqual(meta["summary"]["timesteps"], 128)
        self.assertTrue((self.run_dir / "model.zip").exists())

    def test_training_envs_play_idle_bots(self):
        factory = train_ppo.EnvFactory("idle", PPOConfig().env)
        env = factory()
        self.assertTrue(all(isinstance(c, IdleBot) for c in env.unwrapped.controllers.values()))
        env.close()

    def test_eval_opponents_refuse_idle(self):
        for eval_every in (0, 64):
            cfg = apply_overrides(PPOConfig(), {**TINY, "eval_every": eval_every,
                                                "eval_opponents": "idle"})
            with self.assertRaisesRegex(ValueError, "eval_opponents"):
                train_ppo.check_eval_opponents(cfg)
            with self.assertRaisesRegex(ValueError, "eval_opponents"):
                train_ppo.make_model(cfg, 1)

    def test_idle_run_must_name_its_eval_opponents(self):
        for eval_every in (0, 64):
            cfg = apply_overrides(PPOConfig(), {**TINY, "eval_every": eval_every,
                                                "opponents": "idle"})
            with self.assertRaisesRegex(ValueError, "eval_opponents"):
                train_ppo.check_eval_opponents(cfg)
        # A ladder level with eval_opponents unset is exactly as before.
        train_ppo.check_eval_opponents(PPOConfig())
        self.assertEqual(train_ppo.eval_opponents(PPOConfig(opponents="noisy25")), "noisy25")
        train_ppo.check_eval_opponents(PPOConfig(opponents="idle", eval_opponents="rule"))

    def test_refused_before_the_run_folder_exists(self):
        config = self.tmp / "no_eval.json"
        config.write_text(json.dumps({**TINY, "opponents": "idle"}), encoding="utf-8")
        runs = self.tmp / "refused_runs"
        with self.assertRaisesRegex(ValueError, "eval_opponents"):
            train_ppo.main(["--seed", "0", "--runs-dir", str(runs), "--config", str(config),
                            "--timesteps", "128"])
        self.assertFalse(runs.exists() and any(runs.iterdir()))

    def test_evaluation_protocol_refuses_idle(self):
        with self.assertRaisesRegex(ValueError, "idle"):
            eval_protocol.run_protocol(self.run_dir, games=1, levels=("idle",),
                                       runs_dir=self.tmp / "eval", quiet=True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            eval_protocol.main(["--run", str(self.run_dir), "--levels", "idle"])

    def test_evaluate_agent_refuses_idle(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_agent.main(["--run", str(self.run_dir), "--opponents", "idle"])


if __name__ == "__main__":
    unittest.main()
