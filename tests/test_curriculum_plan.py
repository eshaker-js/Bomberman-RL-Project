"""The curriculum plan file ``experiments/ppo/curriculum.json`` (Task 11, part 3).

- The committed file has ``null`` values (filled after the next benchmark
  run), including each stage's ``timesteps`` inside the ``stages`` list, so
  ``load_plan`` refuses it and names every one of them;
- a test copy with them filled in loads, has seeds 0 to 4 and one variant,
  and resolves to a valid ``PPOConfig`` with exactly the intended changes
  from v0 (E2's ``shaped_explore`` rewards and ``ent_coef``, the stages);
- ``_unset`` now looks inside lists: the existing plan files load exactly
  as with the old version (the same refusal, the same filled result).

Needs numpy, gymnasium, torch and stable-baselines3, like the other plan
tests. Nothing is trained.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

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
    from rl import run_ppo_seeds, train_ppo
    from rl.config import PPOConfig, Stage, apply_overrides, config_to_dict
    from rl.curriculum import check_stages
    from rl.run_ppo_seeds import load_plan

PLANS = ROOT / "experiments" / "ppo"
PLAN = PLANS / "curriculum.json"
LEVELS = ("idle", "noisy25", "noisy10", "rule")
UNSET = ("config.total_timesteps", "config.eval_every", "config.device", "config.vec_env",
         "config.torch_threads", "variants.curriculum.eval_opponents",
         *(f"variants.curriculum.stages.{k}.timesteps" for k in range(4)))
# Test values for the nulls (the real ones are set after the benchmark run).
STEPS = (2_048_000, 1_024_000, 1_024_000, 1_024_000)
FILLED = {"total_timesteps": sum(STEPS), "eval_every": 102_400, "device": "cpu",
          "vec_env": "dummy", "torch_threads": 0}
SHAPED = {"wait": -0.002, "bomb_survived": 0.05}


def old_unset(data, path="") -> list:
    """``rl.run_ppo_seeds._unset`` exactly as before Task 11 (dicts only)."""
    if isinstance(data, dict):
        return [p for k, v in data.items() for p in old_unset(v, f"{path}{k}.")]
    return [path.rstrip(".")] if data is None else []


def changed(cfg: dict, base: dict, path="") -> dict:
    """``{dotted key: value}`` of every value in ``cfg`` that differs from ``base``."""
    out = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            out.update(changed(value, base[key], f"{path}{key}."))
        elif value != base[key]:
            out[f"{path}{key}"] = value
    return out


def plan_result(text: str):
    """``load_plan``'s result, or its error as ``(type, message)``."""
    try:
        return load_plan(text)
    except (ValueError, KeyError, TypeError) as error:
        return type(error), str(error)


@unittest.skipIf(torch is None, MISSING)
class CurriculumPlanTest(unittest.TestCase):
    def setUp(self):
        self.text = PLAN.read_text(encoding="utf-8")
        self.data = json.loads(self.text)

    def filled_data(self) -> dict:
        data = json.loads(self.text)
        data["config"].update(FILLED)
        variant = data["variants"]["curriculum"]
        variant["eval_opponents"] = "random"
        for stage, steps in zip(variant["stages"], STEPS):
            stage["timesteps"] = steps
        return data

    def test_committed_file_waits_for_its_values(self):
        with self.assertRaisesRegex(ValueError, "null") as caught:
            load_plan(self.text)
        message = str(caught.exception)
        for path in UNSET:
            self.assertIn(f"'{path}'", message)
        self.assertEqual(run_ppo_seeds._unset(run_ppo_seeds.strip_comments(self.data)),
                         ["config.total_timesteps", "config.eval_every", "config.device",
                          "config.vec_env", "config.torch_threads",
                          "variants.curriculum.eval_opponents",
                          *(f"variants.curriculum.stages.{k}.timesteps" for k in range(4))])
        # Each null has a "_" comment saying it is still to be set.
        for comment in ("_total_timesteps", "_eval_every", "_device_vec_env_torch_threads"):
            self.assertIn("set", self.data["config"][comment].lower())
        for comment in ("_eval_opponents", "_stages"):
            self.assertIn("SET ME", self.data["variants"]["curriculum"][comment])

    def test_plan_settings(self):
        plan = load_plan(json.dumps(self.filled_data()))
        self.assertEqual(plan["name"], "curriculum")
        self.assertEqual(plan["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(plan["parallel"], 1)
        self.assertEqual(tuple(plan["variants"]), ("curriculum",))
        self.assertEqual(plan["evaluation"], {"games": 200, "stochastic": False})
        cfg = plan["variants"]["curriculum"]
        self.assertTrue(cfg["high_priority"])
        self.assertEqual((cfg["eval_games"], cfg["checkpoint_every"],
                          cfg["checkpoint_keep_every"]), (50, 1_000_000, 1_000_000))
        self.assertEqual([s["opponents"] for s in cfg["stages"]], list(LEVELS))

    def test_variant_is_a_valid_config_with_only_its_changes(self):
        plan = load_plan(json.dumps(self.filled_data()))
        cfg_dict = plan["variants"]["curriculum"]
        cfg = apply_overrides(PPOConfig(), cfg_dict)  # a valid PPOConfig
        self.assertEqual(config_to_dict(cfg), cfg_dict)
        base = config_to_dict(apply_overrides(PPOConfig(), {
            "eval_games": 50, "checkpoint_every": 1_000_000,
            "checkpoint_keep_every": 1_000_000, "high_priority": True, **FILLED}))
        expected_stages = tuple({"opponents": o, "timesteps": t} for o, t in zip(LEVELS, STEPS))
        self.assertEqual(changed(cfg_dict, base), {
            "ent_coef": 0.05, "opponents": "idle", "eval_opponents": "random",
            "stages": expected_stages,
            **{f"env.reward.{k}": v for k, v in SHAPED.items()}})
        self.assertEqual(cfg.stages, tuple(Stage(o, t) for o, t in zip(LEVELS, STEPS)))
        for seed in plan["seeds"]:
            check_stages(cfg, seed)
            train_ppo.check_eval_opponents(cfg)
        self.assertIn(cfg.vec_env, train_ppo.VEC_ENVS)
        # Everything else is v0: PPO values, network and the other reward terms.
        v0 = config_to_dict(PPOConfig())
        for key in ("n_envs", "n_steps", "batch_size", "n_epochs", "learning_rate", "gamma",
                    "gae_lambda", "clip_range", "vf_coef", "max_grad_norm", "conv_channels",
                    "features_dim", "net_arch_pi", "net_arch_vf"):
            self.assertEqual(cfg_dict[key], v0[key], key)
        reward = {k: v for k, v in cfg_dict["env"]["reward"].items() if k not in SHAPED}
        self.assertEqual(reward, {k: v for k, v in v0["env"]["reward"].items()
                                  if k not in SHAPED})
        self.assertEqual(cfg_dict["env"]["max_steps"], v0["env"]["max_steps"])

    def test_a_filled_copy_whose_stages_do_not_add_up_is_refused(self):
        data = self.filled_data()
        data["config"]["total_timesteps"] += 2048
        with self.assertRaisesRegex(ValueError, "add up"):
            load_plan(json.dumps(data))


@unittest.skipIf(torch is None, MISSING)
class ExistingPlansUnchangedTest(unittest.TestCase):
    """The existing plan files load exactly as with the old ``_unset``."""

    FILLS = {"first_runs.json": {"total_timesteps": 2_000_000, "eval_every": 100_000},
             "e2_waiting.json": {"total_timesteps": 2_000_000, "eval_every": 100_000,
                                 "device": "cpu", "vec_env": "dummy", "torch_threads": 0}}

    def test_same_refusal_and_same_filled_result(self):
        for name, fill in self.FILLS.items():
            text = (PLANS / name).read_text(encoding="utf-8")
            data = json.loads(text)
            data["config"].update(fill)
            filled = json.dumps(data)
            new = (plan_result(text), plan_result(filled))
            with mock.patch.object(run_ppo_seeds, "_unset", old_unset):
                old = (plan_result(text), plan_result(filled))
            with self.subTest(name):
                self.assertEqual(new, old)
                self.assertIs(new[0][0], ValueError)  # the committed file is refused
                self.assertIsInstance(new[1], dict)  # the filled copy loads

    def test_lists_without_nulls_change_nothing(self):
        data = {"config": {"conv_channels": [8, 8, 8], "x": None}, "seeds": [0, 1]}
        self.assertEqual(run_ppo_seeds._unset(data), old_unset(data))
        self.assertEqual(run_ppo_seeds._unset({"a": [1, {"b": None}, [None]]}),
                         ["a.1.b", "a.2.0"])


if __name__ == "__main__":
    unittest.main()
