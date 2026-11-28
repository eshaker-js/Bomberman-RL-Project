"""Starting a run from the other kind of model (Task 14, ``init_model_convert``).

- ``PPOConfig.init_model_convert``: default off, round trip through JSON;
- with it on, a MaskablePPO model (safety filter on) starts a plain PPO run
  and a plain PPO model starts a masked run; both train, are verified, and
  ``meta.json`` records ``source_algorithm``, ``run_algorithm`` and
  ``converted``; also with the danger map on, the case of the report run;
- with it off, crossing kinds is still refused before a run folder exists,
  and the message names the key;
- with it on and the same kind, nothing is converted; with it on and no
  ``init_model``, the run starts from fresh weights as before;
- a plan with ``start_from`` and the key resolves, ``plan.json`` records the
  conversion, and without the key the plan is refused and nothing starts.

Needs numpy, gymnasium, torch, stable-baselines3 and sb3-contrib. Only
few-step runs, in temporary folders; every env is closed.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import PPOConfig, apply_overrides, config_to_dict  # noqa: E402

try:
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
    import sb3_contrib  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3/sb3-contrib not installed; test skipped"

if torch is not None:
    from rl import run_ppo_seeds, train_ppo
    from rl.init_model import file_sha256

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40},
        "total_timesteps": 128}
MASKED = {**TINY, "env": {"max_steps": 40, "action_mask": "cautious"}}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args)


def run_main(tmp: Path, overrides: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / f"config_{len(list(tmp.glob('config_*.json')))}.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    return quiet(train_ppo.main, ["--seed", "0", "--runs-dir", str(tmp / "runs"),
                                  "--config", str(config)])


def with_danger(overrides: dict) -> dict:
    return {**overrides, "env": {**overrides["env"], "danger_obs": True}}


class ConvertKeyTest(unittest.TestCase):
    def test_default_off_and_round_trip(self):
        self.assertFalse(PPOConfig().init_model_convert)
        self.assertFalse(config_to_dict(PPOConfig())["init_model_convert"])
        cfg = apply_overrides(PPOConfig(), {"init_model_convert": True})
        saved = json.loads(json.dumps(config_to_dict(cfg)))
        self.assertTrue(apply_overrides(PPOConfig(), saved).init_model_convert)


@unittest.skipIf(torch is None, MISSING)
class ConvertRunsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        tmp = Path(cls._dir.name)
        cls.tmp = tmp
        cls.masked = run_main(tmp / "m", MASKED) / "model.zip"
        cls.plain = run_main(tmp / "p", TINY) / "model.zip"
        cls.masked_danger = run_main(tmp / "md", with_danger(MASKED)) / "model.zip"

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def check_record(self, run: Path, source: Path, have: str, want: str):
        meta = read_json(run / "meta.json")
        record = meta["init_model"]
        self.assertTrue(record["verified"])
        self.assertEqual(record["sha256"], file_sha256(source))
        self.assertEqual(record["algorithm"], have)
        self.assertEqual(record["source_algorithm"], have)
        self.assertEqual(record["run_algorithm"], want)
        self.assertEqual(record["converted"], have != want)
        self.assertEqual(meta["algorithm"], want)
        self.assertTrue(read_json(run / "config.json")["init_model_convert"])

    def test_masked_to_plain(self):
        run = run_main(self.tmp / "mp", {**TINY, "init_model": str(self.masked),
                                         "init_model_convert": True})
        self.check_record(run, self.masked, "MaskablePPO", "PPO")
        self.assertTrue((run / "model.zip").is_file())

    def test_masked_to_plain_with_the_danger_map(self):
        run = run_main(self.tmp / "mpd", {**with_danger(TINY),
                                          "init_model": str(self.masked_danger),
                                          "init_model_convert": True})
        self.check_record(run, self.masked_danger, "MaskablePPO", "PPO")

    def test_plain_to_masked(self):
        run = run_main(self.tmp / "pm", {**MASKED, "init_model": str(self.plain),
                                         "init_model_convert": True})
        self.check_record(run, self.plain, "PPO", "MaskablePPO")

    def test_same_kind_with_the_key_is_not_converted(self):
        run = run_main(self.tmp / "mm", {**MASKED, "init_model": str(self.masked),
                                         "init_model_convert": True})
        self.check_record(run, self.masked, "MaskablePPO", "MaskablePPO")

    def test_without_the_key_crossing_is_still_refused(self):
        for name, overrides in (("x1", {**MASKED, "init_model": str(self.plain)}),
                                ("x2", {**TINY, "init_model": str(self.masked)})):
            with self.subTest(name), \
                    self.assertRaisesRegex(ValueError, "same kind.*init_model_convert"):
                run_main(self.tmp / name, overrides)
            self.assertFalse((self.tmp / name / "runs").exists())

    def test_the_key_alone_changes_nothing_but_config(self):
        run = run_main(self.tmp / "k", {**TINY, "init_model_convert": True})
        self.assertIsNone(read_json(run / "meta.json").get("init_model"))
        self.assertEqual(read_json(run / "meta.json")["algorithm"], "PPO")

    def test_a_plan_records_the_conversion(self):
        tmp = self.tmp / "plans"
        tmp.mkdir()
        source_file = tmp / "a.json"
        source_file.write_text(json.dumps({"name": "a", "seeds": [0], "config": MASKED,
                                           "evaluation": {"games": 2}}), encoding="utf-8")
        plan_a = quiet(run_ppo_seeds.main, [str(source_file), "--runs-dir", str(tmp),
                                            "--quiet"])
        start = {"plan_dir": str(plan_a), "variant": "v0", "file": "model"}
        refused_file = tmp / "b0.json"
        refused_file.write_text(json.dumps({"name": "b0", "seeds": [0], "config": TINY,
                                            "start_from": start,
                                            "evaluation": {"games": 2}}), encoding="utf-8")
        before = sorted(tmp.iterdir())
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit):
            run_ppo_seeds.main([str(refused_file), "--runs-dir", str(tmp), "--quiet"])
        self.assertIn("init_model_convert", err.getvalue())
        self.assertEqual(sorted(tmp.iterdir()), before)

        plan_file = tmp / "b.json"
        plan_file.write_text(json.dumps({"name": "b", "seeds": [0],
                                         "config": {**TINY, "init_model_convert": True},
                                         "start_from": start,
                                         "evaluation": {"games": 2}}), encoding="utf-8")
        plan_b = quiet(run_ppo_seeds.main, [str(plan_file), "--runs-dir", str(tmp), "--quiet"])
        entry = read_json(plan_b / "plan.json")["init_models"]["v0"]["0"]
        self.assertEqual((entry["source_algorithm"], entry["run_algorithm"], entry["converted"]),
                         ("MaskablePPO", "PPO", True))
        run = run_ppo_seeds.finished_training(plan_b / "v0" / "seed0", 0)
        record = read_json(run / "meta.json")["init_model"]
        self.assertEqual(record["sha256"], entry["sha256"])
        self.assertTrue(record["converted"] and record["verified"])
        status = {r["seed"]: r["status"] for r in run_ppo_seeds._read_csv(plan_b / "status.csv")}
        self.assertEqual(status, {"0": "done"})


if __name__ == "__main__":
    unittest.main()
