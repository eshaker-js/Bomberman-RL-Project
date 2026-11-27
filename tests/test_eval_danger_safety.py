"""Evaluating runs with the danger map and the safety filter (Task 13, part 3).

- ``rl.evaluate_agent.run_settings``: read from ``config.json``; a run from
  before Task 13 (no keys, or no ``config.json``) gives off;
- ``rl.eval_protocol`` on a run from before Task 13 gives exactly the rows
  and games of the old code path (``PPO.load``, plain env, no mask);
- on a run with the danger map and the filter: played with both, recorded
  in ``eval.json``, every protocol action allowed by the run's filter;
- ``--action-mask``: ``off`` on a masked run, ``cautious`` on a plain run
  (only allowed actions; with nothing masked the same as ``predict``), and
  unknown modes refused;
- an inconsistent run folder (model and ``config.json`` disagree on the
  danger map) is refused;
- ``rl.evaluate_agent`` plays a danger and filter run with its settings;
- a tiny plan with the two ablation variants (map + filter, map only) runs
  and evaluates end to end with ``rl.run_ppo_seeds``.

Needs numpy, gymnasium, torch, stable-baselines3 and sb3-contrib. Only
few-step runs and few games, in temporary folders.
"""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gymnasium  # noqa: F401
    import numpy as np
    import sb3_contrib  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3/sb3-contrib not installed; test skipped"

if torch is not None:
    from stable_baselines3 import PPO

    from rl import eval_protocol, evaluate_agent, run_ppo_seeds, train_ppo
    from rl.bomberman_env import BombermanGymEnv
    from rl.config import EnvConfig
    from rl.evaluate_agent import ModelPolicy, PPOPolicy, run_settings
    from rl.safety import action_mask
    from rl.seeds import eval_seeds

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40},
        "total_timesteps": 128}
BOTH = {**TINY, "env": {"max_steps": 40, "danger_obs": True, "action_mask": "cautious"}}
STEPS = 60  # protocol games are cut at this many steps in these tests


def run_main(tmp: Path, overrides: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / "config.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return train_ppo.main(["--seed", "0", "--runs-dir", str(tmp / "runs"),
                               "--config", str(config)])


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol(run: Path, out: Path, **kwargs):
    return eval_protocol.run_protocol(run, games=2, levels=("random", "rule"),
                                      runs_dir=out, quiet=True, max_steps=STEPS, **kwargs)


class Spy:
    """Wraps a policy: records each state's mask and the action chosen."""

    def __init__(self, policy, mode: str) -> None:
        self.policy, self.mode, self.checked = policy, mode, 0

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs, state):
        action = self.policy.act(obs, state)
        assert action_mask(state, 0, self.mode)[action], (state["steps"], action)
        self.checked += 1
        return action


@unittest.skipIf(torch is None, MISSING)
class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.plain = run_main(cls.tmp / "plain", TINY)
        cls.both = run_main(cls.tmp / "both", BOTH)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_run_settings(self):
        self.assertEqual(run_settings(self.plain), {"danger_obs": False, "action_mask": "off"})
        self.assertEqual(run_settings(self.both), {"danger_obs": True, "action_mask": "cautious"})
        self.assertEqual(run_settings(self.tmp / "nowhere"),
                         {"danger_obs": False, "action_mask": "off"})

    def old_run_folder(self) -> Path:
        """A copy of the plain run whose config.json has no Task 13 keys."""
        old = self.tmp / "old_run"
        if not old.exists():
            old.mkdir()
            (old / "model.zip").write_bytes((self.plain / "model.zip").read_bytes())
            config = read_json(self.plain / "config.json")
            del config["env"]["danger_obs"], config["env"]["action_mask"]
            (old / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return old

    def test_old_run_is_evaluated_exactly_as_before(self):
        out, rows = protocol(self.old_run_folder(), self.tmp / "e_old")
        # The code path before Task 13: PPO.load, a plain env, no mask.
        model = PPO.load(self.plain / "model.zip", device="cpu")
        seeds = eval_seeds(2)
        reference, games = {}, []
        for level in ("random", "rule"):
            per_game = []
            env = BombermanGymEnv(level, EnvConfig(max_steps=STEPS))
            try:
                reference[level] = evaluate_agent.evaluate_seeds(
                    env, ModelPolicy(model), seeds, per_game)
            finally:
                env.close()
            games += [{"level": level, **g} for g in per_game]
        want = [eval_protocol.summary_row(level, reference[level]) for level in reference]
        for got, row in zip(rows, want):
            self.assertEqual(got, row)
        self.assertEqual(len(read_csv(out / "games.csv")), len(games))
        settings = read_json(out / "eval.json")["settings"]
        self.assertEqual((settings["danger_obs"], settings["action_mask"],
                          settings["trained_action_mask"], settings["algorithm"]),
                         (False, "off", "off", "PPO"))

    def test_run_with_danger_and_filter(self):
        spies = []
        real = eval_protocol.evaluate_full

        def spying(policy, *args, **kwargs):
            spy = Spy(policy, "cautious")
            spies.append(spy)
            self.assertTrue(kwargs["danger_obs"])
            return real(spy, *args, **kwargs)

        with mock.patch.object(eval_protocol, "evaluate_full", spying):
            out, rows = protocol(self.both, self.tmp / "e_both")
        self.assertEqual(len(rows), 3)
        self.assertGreater(sum(s.checked for s in spies), 50)
        settings = read_json(out / "eval.json")["settings"]
        self.assertEqual((settings["danger_obs"], settings["action_mask"],
                          settings["trained_action_mask"], settings["algorithm"]),
                         (True, "cautious", "cautious", "MaskablePPO"))

    def test_ablation_without_the_filter(self):
        out, _ = protocol(self.both, self.tmp / "e_off", action_mask="off")
        settings = read_json(out / "eval.json")["settings"]
        self.assertEqual((settings["action_mask"], settings["trained_action_mask"]),
                         ("off", "cautious"))
        policy = PPOPolicy(self.both / "model.zip", action_mask="off")
        self.assertIsNone(policy.action_mask)

    def test_filter_added_to_a_plain_agent(self):
        out, _ = protocol(self.plain, self.tmp / "e_add", action_mask="cautious")
        self.assertEqual(read_json(out / "eval.json")["settings"]["action_mask"], "cautious")
        masked = PPOPolicy(self.plain / "model.zip", action_mask="cautious")
        free = PPOPolicy(self.plain / "model.zip")
        sampler = PPOPolicy(self.plain / "model.zip", deterministic=False,
                            action_mask="board")
        env = BombermanGymEnv("rule", EnvConfig(max_steps=STEPS))
        torch.manual_seed(0)
        same_when_free = narrowed = 0
        try:
            for seed in range(4):
                obs, _ = env.reset(seed=seed)
                done = False
                while not done:
                    state = env.state
                    mask = action_mask(state, 0, "cautious")
                    action = masked.act(obs, state)
                    self.assertTrue(mask[action])
                    self.assertTrue(action_mask(state, 0, "board")[sampler.act(obs, state)])
                    if mask.all():
                        self.assertEqual(action, free.act(obs, state))
                        same_when_free += 1
                    else:
                        narrowed += 1
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
        finally:
            env.close()
        self.assertGreater(same_when_free, 10)
        self.assertGreater(narrowed, 0)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "unknown action_mask"):
            protocol(self.plain, self.tmp / "e_bad", action_mask="own")
        broken = self.tmp / "broken"
        broken.mkdir(exist_ok=True)
        (broken / "model.zip").write_bytes((self.both / "model.zip").read_bytes())
        config = read_json(self.both / "config.json")
        config["env"]["danger_obs"] = False
        (broken / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            protocol(broken, self.tmp / "e_broken")
        self.assertFalse((self.tmp / "e_broken").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            eval_protocol.main(["--run", str(self.plain), "--action-mask", "own"])

    def test_evaluate_agent_uses_the_run_settings(self):
        with contextlib.redirect_stdout(io.StringIO()):
            out, results = evaluate_agent.main(
                ["--run", str(self.both), "--games", "2", "--max-steps", str(STEPS),
                 "--runs-dir", str(self.tmp / "e_agent")])
        self.assertEqual(results["games"], 2)


@unittest.skipIf(torch is None, MISSING)
class AblationPlanTest(unittest.TestCase):
    def test_two_variants_end_to_end(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            plan_file = tmp / "plan.json"
            plan_file.write_text(json.dumps({
                "name": "ablation", "seeds": [0], "parallel": 1,
                "config": {**TINY, "eval_every": 64, "eval_games": 1,
                           "env": {"max_steps": 30}},
                "variants": {"danger_filter": {"env": {"max_steps": 30, "danger_obs": True,
                                                       "action_mask": "cautious"}},
                             "danger_only": {"env": {"max_steps": 30, "danger_obs": True}}},
                "evaluation": {"games": 1}}))
            with contextlib.redirect_stdout(io.StringIO()):
                plan_dir = run_ppo_seeds.main([str(plan_file), "--runs-dir",
                                               str(tmp / "runs"), "--quiet"])
            status = {r["variant"]: r for r in read_csv(plan_dir / "status.csv")}
            self.assertEqual({v: s["status"] for v, s in status.items()},
                             {"danger_filter": "done", "danger_only": "done"})
            for variant, mask, algorithm in (("danger_filter", "cautious", "MaskablePPO"),
                                             ("danger_only", "off", "PPO")):
                folder = plan_dir / status[variant]["eval_dir"]
                settings = read_json(folder / "eval.json")["settings"]
                self.assertEqual((settings["danger_obs"], settings["action_mask"],
                                  settings["algorithm"]), (True, mask, algorithm))
                self.assertEqual(len(read_csv(folder / "protocol.csv")), 6)


if __name__ == "__main__":
    unittest.main()
