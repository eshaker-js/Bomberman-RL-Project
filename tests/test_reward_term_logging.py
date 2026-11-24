"""Reward terms in the training logs, and the evaluation opponents (Task 10, part 2).

- ``RewardTermLogger``: the per-rollout means in ``progress.csv`` equal a
  recount from the episodes' ``info["reward_terms"]``; a rollout without a
  finished episode leaves the term cells blank; logging never changes the
  trained model (``dummy`` and ``subproc``).
- ``PPOConfig.eval_opponents``: reaches the evaluation env and
  ``config.json``, is checked at the start of every run, and with the
  default (``None``) the evaluation is exactly as before.
- ``str | None`` config fields take a string or ``null``; the other fields
  still reject ``null``.

Needs numpy, gymnasium, torch and stable-baselines3, like the other RL
tests. Only few-step runs, in temporary folders; every env, logger and file
is closed.
"""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.logger import configure

    from agents.noisy_bot import NoisyBot
    from agents.random_bot import RandomBot
    from rl import train_ppo
    from rl.bomberman_env import REWARD_TERMS
    from rl.config import PPOConfig, apply_overrides, config_to_dict

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}
TERM_COLUMNS = [f"reward_terms/{t}" for t in (*REWARD_TERMS, "total")]
TIMING = ("eval_seconds", "wall_seconds")


def params(model) -> dict:
    return {k: v.clone() for k, v in model.policy.state_dict().items()}


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if torch is not None:
    class Recorder(BaseCallback):
        """Keeps every step's dones and reward terms, and the rollout ends."""

        def __init__(self) -> None:
            super().__init__()
            self.steps: list = []  # per step: [(done, terms) per env]
            self.returns: list = []  # Monitor's return of each finished episode, in order
            self.rollout_ends: list = []  # len(self.steps) at each rollout end

        def _on_step(self) -> bool:
            self.steps.append([(bool(done), dict(info["reward_terms"]))
                               for done, info in zip(self.locals["dones"],
                                                     self.locals["infos"])])
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.returns.append(info["episode"]["r"])
            return True

        def _on_rollout_end(self) -> None:
            self.rollout_ends.append(len(self.steps))


def recount(recorder: "Recorder", n_envs: int) -> list:
    """Per rollout: the finished episodes' term sums, recounted from the infos."""
    rollouts, start = [], 0
    episode = [[] for _ in range(n_envs)]  # per env: the terms of each step so far
    for end in recorder.rollout_ends:
        finished = []
        for step in recorder.steps[start:end]:
            for i, (done, terms) in enumerate(step):
                episode[i].append(terms)
                if done:
                    finished.append({t: sum(s[t] for s in episode[i]) for t in REWARD_TERMS})
                    episode[i] = []
        rollouts.append(finished)
        start = end
    return rollouts


@unittest.skipIf(torch is None, MISSING)
class LoggingTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def learned(self, cfg, callback=None, steps=256, seed=1):
        model = train_ppo.make_model(cfg, seed)
        try:
            model.learn(total_timesteps=steps, callback=callback)
            return params(model)
        finally:
            model.get_env().close()

    def assert_same(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)


class RewardTermLoggerTest(LoggingTestCase):
    def logged_run(self, cfg, folder):
        """A tiny run with the logger, a recorder and SB3's CSV logger."""
        model = train_ppo.make_model(cfg, 1)
        logger = configure(str(folder), ["csv"])
        recorder, terms = Recorder(), train_ppo.RewardTermLogger()
        counter = train_ppo.OutcomeCounter()
        try:
            model.set_logger(logger)
            model.learn(total_timesteps=384,
                        callback=CallbackList([counter, terms, recorder]))
            return params(model), terms, recorder, counter
        finally:
            logger.close()
            model.get_env().close()

    def test_logged_means_equal_a_recount_from_the_infos(self):
        cfg = apply_overrides(PPOConfig(), TINY)
        _, terms, recorder, counter = self.logged_run(cfg, self.tmp)
        rollouts = recount(recorder, cfg.n_envs)
        self.assertEqual(len(rollouts), 3)  # 384 steps / (2 envs x 64)
        self.assertEqual(len(terms.logged), 3)
        rows = read_csv(self.tmp / "progress.csv")
        self.assertEqual(len(rows), 3)
        for finished, logged, row in zip(rollouts, terms.logged, rows):
            self.assertGreater(len(finished), 0)  # max_steps 40: episodes do finish
            self.assertEqual(logged["reward_terms/episodes"], len(finished))
            self.assertEqual(int(row["reward_terms/episodes"]), len(finished))
            for term in REWARD_TERMS:
                expected = sum(e[term] for e in finished) / len(finished)
                self.assertAlmostEqual(logged[f"reward_terms/{term}"], expected, places=12)
                self.assertAlmostEqual(float(row[f"reward_terms/{term}"]), expected,
                                       places=12)
            total = sum(sum(e.values()) for e in finished) / len(finished)
            self.assertAlmostEqual(float(row["reward_terms/total"]), total, places=12)
        # Every finished episode is counted once, like the outcomes.
        self.assertEqual(sum(len(f) for f in rollouts), sum(counter.counts.values()))

    def test_term_sums_equal_the_episode_returns(self):
        # The terms of a step add up to its reward, so an episode's term
        # sums add up to its return (Monitor's "r", rounded to 6 digits).
        cfg = apply_overrides(PPOConfig(), TINY)
        _, _, recorder, _ = self.logged_run(cfg, self.tmp)
        finished = [e for rollout in recount(recorder, cfg.n_envs) for e in rollout]
        self.assertGreater(len(finished), 0)
        self.assertEqual(len(finished), len(recorder.returns))
        for episode, episode_return in zip(finished, recorder.returns):
            self.assertAlmostEqual(sum(episode.values()), episode_return, places=5)

    def test_rollout_without_finished_episodes_leaves_cells_blank(self):
        logger = configure(str(self.tmp), ["csv"])
        callback = train_ppo.RewardTermLogger()
        try:
            callback.model = type("FakeModel", (), {"logger": logger})()
            callback.finished = [dict.fromkeys(REWARD_TERMS, 0.0)]
            callback.finished[0]["crate"] = 0.04
            callback._on_rollout_end()
            logger.dump(1)
            callback._on_rollout_end()  # nothing finished in this rollout
            logger.dump(2)
        finally:
            logger.close()
        rows = read_csv(self.tmp / "progress.csv")
        self.assertEqual(rows[0]["reward_terms/episodes"], "1")
        self.assertEqual(float(rows[0]["reward_terms/crate"]), 0.04)
        self.assertEqual(float(rows[0]["reward_terms/total"]), 0.04)
        self.assertEqual(rows[1]["reward_terms/episodes"], "0")
        for column in TERM_COLUMNS:
            self.assertEqual(rows[1][column], "", column)
        self.assertEqual(callback.logged[1], {"reward_terms/episodes": 0})

    def test_logging_does_not_change_the_model(self):
        cfg = apply_overrides(PPOConfig(), TINY)
        plain = self.learned(cfg)
        logged = self.learned(cfg, callback=train_ppo.RewardTermLogger())
        self.assert_same(plain, logged)

    def test_logging_under_subproc_spawn(self):
        cfg = apply_overrides(PPOConfig(), TINY)
        dummy = train_ppo.RewardTermLogger()
        subproc = train_ppo.RewardTermLogger()
        a = self.learned(cfg, callback=dummy)
        b = self.learned(apply_overrides(cfg, {"vec_env": "subproc"}), callback=subproc)
        self.assert_same(a, b)
        self.assertEqual(dummy.logged, subproc.logged)

    def test_training_run_writes_the_columns(self):
        config = self.tmp / "tiny.json"
        config.write_text(json.dumps(TINY))
        with contextlib.redirect_stdout(io.StringIO()):
            run = train_ppo.main(["--seed", "1", "--runs-dir", str(self.tmp / "runs"),
                                  "--config", str(config), "--timesteps", "256"])
        rows = read_csv(run / "progress.csv")
        self.assertEqual(len(rows), 2)
        for column in ["reward_terms/episodes", *TERM_COLUMNS]:
            self.assertIn(column, rows[0])
        self.assertEqual(json.loads((run / "config.json").read_text())["eval_opponents"],
                         None)


class EvalOpponentsTest(LoggingTestCase):
    def evaluator(self, **extra):
        cfg = apply_overrides(PPOConfig(), {**TINY, "eval_every": 128, "eval_games": 3,
                                            **extra})
        folder = self.tmp / f"e{len(list(self.tmp.iterdir()))}"
        folder.mkdir()
        return cfg, folder, train_ppo.TrainingEvalCallback(cfg, folder)

    def test_default_is_none(self):
        self.assertIsNone(PPOConfig().eval_opponents)
        self.assertEqual(train_ppo.eval_opponents(PPOConfig()), "random")
        self.assertEqual(train_ppo.eval_opponents(PPOConfig(opponents="rule")), "rule")

    def test_eval_opponents_reach_the_evaluation_env(self):
        cfg, _, callback = self.evaluator(opponents="random", eval_opponents="noisy25")
        try:
            bots = list(callback.env.controllers.values())
            self.assertEqual(len(bots), 3)
            for bot in bots:
                self.assertIsInstance(bot, NoisyBot)
                self.assertEqual(bot.epsilon, 0.25)
        finally:
            callback.close()
        model = train_ppo.make_model(cfg, 1)
        try:  # training still plays random bots
            for env in model.get_env().envs:
                for bot in env.unwrapped.controllers.values():
                    self.assertIsInstance(bot, RandomBot)
        finally:
            model.get_env().close()

    def test_default_evaluation_is_unchanged(self):
        # None and the training level itself give the same curve and games
        # (apart from the timing columns), on the same trained model.
        _, folder_a, default = self.evaluator()
        _, folder_b, explicit = self.evaluator(eval_opponents="random")
        try:
            for bot in default.env.controllers.values():
                self.assertIsInstance(bot, RandomBot)
            self.learned(apply_overrides(PPOConfig(), TINY), callback=default)
            self.learned(apply_overrides(PPOConfig(), TINY), callback=explicit)
        finally:
            default.close()
            explicit.close()
        self.assertEqual(len(default.rows), 3)  # 0, 128, 256
        strip = lambda rows: [{k: v for k, v in r.items() if k not in TIMING} for r in rows]
        self.assertEqual(strip(default.rows), strip(explicit.rows))
        self.assertEqual(strip(read_csv(folder_a / "eval_curve.csv")),
                         strip(read_csv(folder_b / "eval_curve.csv")))
        self.assertEqual(read_csv(folder_a / "eval_games.csv"),
                         read_csv(folder_b / "eval_games.csv"))

    def test_unknown_level_stops_every_run(self):
        for eval_every in (0, 64):
            cfg = apply_overrides(PPOConfig(), {**TINY, "eval_every": eval_every,
                                                "eval_opponents": "noisy33"})
            with self.assertRaisesRegex(ValueError, "eval_opponents"):
                train_ppo.make_model(cfg, 1)

    def test_recorded_in_config_json(self):
        config = self.tmp / "c.json"
        config.write_text(json.dumps({**TINY, "eval_opponents": "noisy25"}))
        with contextlib.redirect_stdout(io.StringIO()):
            run = train_ppo.main(["--seed", "1", "--runs-dir", str(self.tmp / "runs"),
                                  "--config", str(config), "--timesteps", "128",
                                  "--eval-every", "128", "--eval-games", "2"])
        saved = json.loads((run / "config.json").read_text())
        self.assertEqual((saved["opponents"], saved["eval_opponents"]), ("random", "noisy25"))
        self.assertEqual(len(read_csv(run / "eval_curve.csv")), 2)


@unittest.skipIf(torch is None, MISSING)
class OptionalStrConfigTest(unittest.TestCase):
    def test_str_or_null(self):
        cfg = apply_overrides(PPOConfig(), {"eval_opponents": "random"})
        self.assertEqual(cfg.eval_opponents, "random")
        self.assertIsNone(apply_overrides(cfg, {"eval_opponents": None}).eval_opponents)
        self.assertEqual(config_to_dict(cfg)["eval_opponents"], "random")

    def test_str_or_null_field_rejects_other_types(self):
        for value in (1, 2.5, True, ["random"], {"a": 1}):
            with self.assertRaisesRegex(TypeError, "str or null"):
                apply_overrides(PPOConfig(), {"eval_opponents": value})

    def test_other_fields_still_reject_null(self):
        for key in ("opponents", "device", "vec_env", "ent_coef", "total_timesteps",
                    "torch_threads", "high_priority", "conv_channels"):
            with self.assertRaises(TypeError, msg=key):
                apply_overrides(PPOConfig(), {key: None})
        with self.assertRaises(TypeError):
            apply_overrides(PPOConfig(), {"env": {"reward": {"wait": None}}})


if __name__ == "__main__":
    unittest.main()
