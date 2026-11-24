"""The E2 plan file ``experiments/ppo/e2_waiting.json`` (Task 10, part 3).

The committed file has ``null`` values (set from the laptop benchmark), so
``load_plan`` refuses it; a test copy with them filled in loads, has seeds 0
to 4 and the five variants, and every variant resolves to a valid
``PPOConfig`` with exactly the intended changes from v0.

Needs numpy, gymnasium, torch and stable-baselines3, like the other plan
tests. Nothing is trained.
"""

import json
import sys
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
    from rl import train_ppo
    from rl.config import PPOConfig, apply_overrides, config_to_dict
    from rl.run_ppo_seeds import load_plan

PLAN = ROOT / "experiments" / "ppo" / "e2_waiting.json"
UNSET = ("total_timesteps", "eval_every", "device", "vec_env", "torch_threads")
# Test values for the nulls (the real ones come from the benchmark).
FILLED = {"total_timesteps": 2_000_000, "eval_every": 100_000, "device": "cpu",
          "vec_env": "dummy", "torch_threads": 0}
VARIANTS = ("v0", "shaped", "explore", "shaped_explore", "shaped_explore_noisy25")
SHAPED = {"wait": -0.002, "bomb_survived": 0.05}


def changed(cfg: dict, base: dict, path="") -> dict:
    """``{dotted key: value}`` of every value in ``cfg`` that differs from ``base``."""
    out = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            out.update(changed(value, base[key], f"{path}{key}."))
        elif value != base[key]:
            out[f"{path}{key}"] = value
    return out


@unittest.skipIf(torch is None, MISSING)
class E2PlanTest(unittest.TestCase):
    def setUp(self):
        self.text = PLAN.read_text(encoding="utf-8")
        self.data = json.loads(self.text)

    def filled_plan(self) -> dict:
        data = json.loads(self.text)
        data["config"].update(FILLED)
        return load_plan(json.dumps(data))

    def test_committed_file_waits_for_the_benchmark_values(self):
        self.assertEqual(sorted(k for k, v in self.data["config"].items() if v is None),
                         sorted(UNSET))
        # Each null has a "_" comment saying where its value comes from.
        for comment in ("_total_timesteps", "_eval_every", "_device_vec_env_torch_threads"):
            self.assertIn("benchmark", self.data["config"][comment].lower())
        with self.assertRaisesRegex(ValueError, "null"):
            load_plan(self.text)

    def test_plan_settings(self):
        plan = self.filled_plan()
        self.assertEqual(plan["name"], "e2_waiting")
        self.assertEqual(plan["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(plan["parallel"], 1)
        self.assertEqual(tuple(plan["variants"]), VARIANTS)
        self.assertEqual(plan["evaluation"], {"games": 200, "stochastic": False})
        for cfg in plan["variants"].values():
            self.assertTrue(cfg["high_priority"])
            self.assertEqual((cfg["eval_games"], cfg["checkpoint_every"]), (50, 1_000_000))

    def test_every_variant_is_a_valid_config_with_only_its_changes(self):
        plan = self.filled_plan()
        base = config_to_dict(apply_overrides(PPOConfig(), {"opponents": "random",
                                                            "eval_games": 50,
                                                            "checkpoint_every": 1_000_000,
                                                            "high_priority": True,
                                                            **FILLED}))
        expected = {
            "v0": {},
            "shaped": {f"env.reward.{k}": v for k, v in SHAPED.items()},
            "explore": {"ent_coef": 0.05},
            "shaped_explore": {"ent_coef": 0.05,
                               **{f"env.reward.{k}": v for k, v in SHAPED.items()}},
            "shaped_explore_noisy25": {"ent_coef": 0.05, "opponents": "noisy25",
                                       "eval_opponents": "random",
                                       **{f"env.reward.{k}": v for k, v in SHAPED.items()}},
        }
        for name, cfg_dict in plan["variants"].items():
            cfg = apply_overrides(PPOConfig(), cfg_dict)  # a valid PPOConfig
            self.assertEqual(config_to_dict(cfg), cfg_dict, name)
            self.assertEqual(changed(cfg_dict, base), expected[name], name)
            train_ppo.check_eval_opponents(cfg)
            self.assertIn(cfg.vec_env, train_ppo.VEC_ENVS)
        # v0 is exactly the v0 values of rl.config (reward and PPO).
        v0 = plan["variants"]["v0"]
        self.assertEqual(v0["env"], config_to_dict(PPOConfig())["env"])
        self.assertEqual((v0["ent_coef"], v0["eval_opponents"]), (0.01, None))


if __name__ == "__main__":
    unittest.main()
