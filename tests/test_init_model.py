"""Starting a PPO run from a saved model (Task 12, part 1).

- ``PPOConfig.init_model`` and ``PPOConfig.env_seed_offset``: defaults,
  JSON round trip through ``config.json``, refusals of wrong types;
- loading gives exactly the saved policy and optimizer state, and the new
  config's ``ent_coef``, ``learning_rate`` and rewards apply afterwards;
- every refusal (missing file, not a zip, not one of our PPO models, wrong
  network sizes, an inconsistent source folder, a bad offset) happens
  before any run folder exists;
- a tiny run continues from a checkpoint and from the ``model.zip`` of
  another tiny run, also with stages; its first evaluation equals the
  source's last one; ``meta.json`` records the source, its SHA-256 and step
  count, and a chain of sources; a model outside any run folder works too;
- the source folder is unchanged byte for byte afterwards;
- ``env_seed_offset`` shifts only the training envs' seeds, in every stage,
  and is checked against the evaluation seeds (also by the plan runner);
- with both options unset, a run trains the same model as today, bit for bit;
- a ``cuda``-saved model continues on ``cpu`` and back (skipped without CUDA).

The config tests need nothing beyond the standard library; the rest needs
numpy, gymnasium, torch and stable-baselines3, like the other RL tests.
Only few-step runs, in temporary folders; every env and file is closed.
"""

import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import PPOConfig, apply_overrides, config_to_dict  # noqa: E402

try:
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"
CUDA = torch is not None and torch.cuda.is_available()

if torch is not None:
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.save_util import load_from_zip_file

    from rl import run_ppo_seeds, train_ppo
    from rl.bomberman_env import BombermanGymEnv
    from rl.init_model import file_sha256, inspect_init_model, load_init_model, same
    from rl.networks import SmallGridCNN

# n_envs x n_steps = 128 steps per rollout.
TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}
SOURCE = {**TINY, "total_timesteps": 256, "checkpoint_every": 128,
          "checkpoint_keep_every": 128, "eval_every": 128, "eval_games": 2}
TIMING = ("eval_seconds", "wall_seconds")


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_main(tmp: Path, overrides: dict, *extra) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / f"config_{len(list(tmp.glob('config_*.json')))}.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return train_ppo.main(["--seed", "0", "--runs-dir", str(tmp / "runs"),
                               "--config", str(config), *extra])


def folder_hashes(folder: Path) -> dict:
    """``{relative path: SHA-256}`` of every file under ``folder``."""
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def assert_same_model(test, path_a: Path, path_b: Path) -> None:
    """Policy and optimizer state of two saved models are equal, tensor for tensor."""
    _, params_a, _ = load_from_zip_file(path_a, device="cpu", load_data=False)
    _, params_b, _ = load_from_zip_file(path_b, device="cpu", load_data=False)
    test.assertEqual(list(params_a["policy"]), list(params_b["policy"]))
    test.assertTrue(same(params_a["policy"], params_b["policy"]))
    test.assertTrue(same(params_a["policy.optimizer"], params_b["policy.optimizer"]))


class InitModelConfigTest(unittest.TestCase):
    """The two new fields of ``PPOConfig`` (``rl.config``), no RL packages needed."""

    def test_defaults_are_off(self):
        self.assertIsNone(PPOConfig().init_model)
        self.assertEqual(PPOConfig().env_seed_offset, 0)
        saved = config_to_dict(PPOConfig())
        self.assertIsNone(saved["init_model"])
        self.assertEqual(saved["env_seed_offset"], 0)

    def test_json_round_trip(self):
        cfg = apply_overrides(PPOConfig(), {"init_model": "runs/x/model.zip",
                                            "env_seed_offset": 1000})
        self.assertEqual((cfg.init_model, cfg.env_seed_offset), ("runs/x/model.zip", 1000))
        saved = json.loads(json.dumps(config_to_dict(cfg)))
        self.assertEqual(apply_overrides(PPOConfig(), saved), cfg)
        self.assertIsNone(apply_overrides(cfg, {"init_model": None}).init_model)

    def test_wrong_types_are_refused(self):
        for key, value in (("init_model", 5), ("init_model", ["a"]),
                           ("env_seed_offset", "1"), ("env_seed_offset", 1.5),
                           ("env_seed_offset", True), ("env_seed_offset", None)):
            with self.subTest(key=key, value=value), self.assertRaises(TypeError):
                apply_overrides(PPOConfig(), {key: value})

    def test_an_old_config_without_the_fields_still_loads(self):
        saved = config_to_dict(PPOConfig())
        del saved["init_model"], saved["env_seed_offset"]
        self.assertEqual(apply_overrides(PPOConfig(), saved), PPOConfig())


@unittest.skipIf(torch is None, MISSING)
class SourceRunMixin:
    """One tiny source run (256 steps, checkpoints at 128 and 256), made once per class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.source = run_main(cls.tmp / "source", SOURCE)
        cls.source_hashes = folder_hashes(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def cfg(self, **changes):
        return apply_overrides(PPOConfig(), {**TINY, **changes})


@unittest.skipIf(torch is None, MISSING)
class LoadTest(SourceRunMixin, unittest.TestCase):
    def test_loading_gives_exactly_the_saved_policy_and_optimizer(self):
        # Saved after the first update, so the optimizer has a state.
        path = self.source / "checkpoints" / "model_256.zip"
        cfg = self.cfg(init_model=str(path), ent_coef=0.05, learning_rate=1e-4)
        record = inspect_init_model(cfg)
        model = train_ppo.make_model(cfg, 3)
        try:
            # Before loading, the fresh weights differ from the saved ones.
            _, params, _ = load_from_zip_file(path, device="cpu", load_data=False)
            self.assertFalse(same(model.policy.state_dict(), params["policy"]))
            load_init_model(model, record)
            self.assertTrue(record["verified"])
            state = model.policy.state_dict()
            self.assertEqual(set(state), set(params["policy"]))
            for key in state:
                self.assertTrue(torch.equal(state[key], params["policy"][key]), key)
            saved_opt = params["policy.optimizer"]
            loaded_opt = model.policy.optimizer.state_dict()
            self.assertEqual(set(loaded_opt["state"]), set(saved_opt["state"]))
            self.assertGreater(len(saved_opt["state"]), 0)
            for index, values in saved_opt["state"].items():
                for key, value in values.items():
                    self.assertTrue(torch.equal(loaded_opt["state"][index][key], value))
            # Only the weights come from the file: the new config's values apply.
            self.assertEqual(model.ent_coef, 0.05)
            self.assertEqual(model.num_timesteps, 0)
            model.learn(128)
            self.assertEqual(model.policy.optimizer.param_groups[0]["lr"], 1e-4)
        finally:
            model.get_env().close()

    def test_record_describes_the_source(self):
        path = self.source / "checkpoints" / "model_128.zip"
        record = inspect_init_model(self.cfg(init_model=str(path)))
        self.assertEqual(record["path"], str(path))
        self.assertEqual(record["absolute_path"], str(path.resolve()))
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(record["num_timesteps"], 128)
        self.assertEqual(record["source_run_dir"], self.source.name)
        self.assertEqual(record["source_seed"], 0)
        self.assertIsNone(record["source_init_model"])
        self.assertEqual(record["sb3_version"], stable_baselines3.__version__)
        self.assertEqual(record["network"], {"conv_channels": [8, 8, 8], "features_dim": 32,
                                             "net_arch_pi": [64], "net_arch_vf": [64]})
        self.assertTrue(record["optimizer_loaded"])
        self.assertFalse(record["verified"])  # until load_init_model
        self.assertIsNone(inspect_init_model(self.cfg()))

    def test_a_changed_file_is_not_loaded(self):
        tmp = self.tmp / "changed"
        tmp.mkdir()
        path = tmp / "model.zip"
        shutil.copy(self.source / "model.zip", path)
        cfg = self.cfg(init_model=str(path))
        record = inspect_init_model(cfg)
        shutil.copy(self.source / "checkpoints" / "model_128.zip", path)
        model = train_ppo.make_model(cfg, 0)
        try:
            with self.assertRaisesRegex(RuntimeError, "changed after it was checked"):
                load_init_model(model, record)
        finally:
            model.get_env().close()


@unittest.skipIf(torch is None, MISSING)
class RefusalTest(SourceRunMixin, unittest.TestCase):
    """Every refusal happens before any run folder exists."""

    def refused(self, overrides: dict, pattern: str) -> None:
        folder = Path(tempfile.mkdtemp(dir=self.tmp))
        with self.assertRaisesRegex(ValueError, pattern):
            run_main(folder, {**TINY, "total_timesteps": 128, **overrides})
        runs = folder / "runs"
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a run folder was created")

    def test_missing_file(self):
        self.refused({"init_model": str(self.tmp / "nope.zip")}, "no such file")

    def test_not_a_zip(self):
        path = self.tmp / "text.zip"
        path.write_text("hello", encoding="utf-8")
        self.refused({"init_model": str(path)}, "not a saved PPO model")

    def test_a_zip_without_the_policy(self):
        path = self.tmp / "no_policy.zip"
        with zipfile.ZipFile(self.source / "model.zip") as src, \
                zipfile.ZipFile(path, "w") as dst:
            for name in src.namelist():
                if name != "policy.pth":
                    dst.writestr(name, src.read(name))
        self.refused({"init_model": str(path)}, "not a saved PPO model")

    def test_another_algorithm(self):
        env = BombermanGymEnv("random")
        try:
            kwargs = {"features_extractor_class": SmallGridCNN,
                      "features_extractor_kwargs": {"features_dim": 32,
                                                    "conv_channels": (8, 8, 8)},
                      "net_arch": {"pi": [64], "vf": [64]}, "normalize_images": False}
            model = A2C("MlpPolicy", env, policy_kwargs=kwargs, seed=0, device="cpu")
            path = self.tmp / "a2c.zip"
            model.save(path)
        finally:
            env.close()
        self.refused({"init_model": str(path)}, "not a PPO model")

    def test_wrong_network_sizes(self):
        path = str(self.source / "model.zip")
        for key, value in (("features_dim", 16), ("conv_channels", [8, 8]),
                           ("net_arch_pi", [32]), ("net_arch_vf", [64, 64])):
            with self.subTest(key=key):
                self.refused({"init_model": path, key: value}, f"network sizes differ.*{key}")

    def test_inconsistent_source_folder(self):
        copy = self.tmp / "inconsistent"
        shutil.copytree(self.source, copy)
        config = read_json(copy / "config.json")
        config["features_dim"] = 99
        (copy / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.refused({"init_model": str(copy / "model.zip")}, "source folder is inconsistent")

    def test_bad_env_seed_offset(self):
        self.refused({"env_seed_offset": -1}, "env_seed_offset must be")
        # Shifted onto the evaluation seeds (1,000,000 on).
        self.refused({"env_seed_offset": 999_999}, "reserved|evaluation")
        self.refused({"env_seed_offset": 999_997, "opponents": "idle",
                      "eval_opponents": "random", "total_timesteps": 256,
                      "stages": [{"opponents": "idle", "timesteps": 128},
                                 {"opponents": "random", "timesteps": 128}]},
                     "reserved|evaluation")

    def test_source_folder_is_untouched_by_the_refusals(self):
        self.refused({"init_model": str(self.source / "model.zip"), "features_dim": 16},
                     "network sizes")
        self.assertEqual(folder_hashes(self.source), self.source_hashes)


@unittest.skipIf(torch is None, MISSING)
class ContinueRunTest(SourceRunMixin, unittest.TestCase):
    def test_continue_from_a_checkpoint_with_new_settings(self):
        path = self.source / "checkpoints" / "model_128.zip"
        new = {**TINY, "total_timesteps": 256, "init_model": str(path), "ent_coef": 0.05,
               "learning_rate": 1e-4, "env": {"max_steps": 40, "reward": {"wait": -0.5}},
               "checkpoint_every": 128, "checkpoint_keep_every": 128}
        run = run_main(self.tmp / "continued", new)
        meta, config = read_json(run / "meta.json"), read_json(run / "config.json")
        self.assertEqual(config["init_model"], str(path))
        self.assertEqual(config["ent_coef"], 0.05)
        record = meta["init_model"]
        self.assertEqual(record["sha256"], file_sha256(path))
        self.assertEqual(record["num_timesteps"], 128)
        self.assertEqual(record["source_run_dir"], self.source.name)
        self.assertTrue(record["verified"])
        # The run counts its own steps from 0.
        self.assertEqual(meta["summary"]["timesteps"], 256)
        self.assertTrue((run / "checkpoints" / "model_128.zip").exists())
        rows = read_csv(run / "progress.csv")
        self.assertEqual([int(r["time/total_timesteps"]) for r in rows], [128, 256])
        # The new reward and learning rate really apply.
        self.assertTrue(all(float(r["train/learning_rate"]) == 1e-4
                            for r in rows if r.get("train/learning_rate")))
        waits = [float(r["reward_terms/wait"]) for r in rows if r["reward_terms/wait"]]
        self.assertTrue(waits and min(waits) < 0)
        source_waits = [float(r["reward_terms/wait"]) for r in
                        read_csv(self.source / "progress.csv") if r["reward_terms/wait"]]
        self.assertTrue(all(w == 0 for w in source_waits))
        saved = PPO.load(run / "model.zip", device="cpu")
        self.assertEqual(saved.ent_coef, 0.05)
        self.assertEqual(folder_hashes(self.source), self.source_hashes)

    def test_first_evaluation_equals_the_sources_last(self):
        new = {**TINY, "total_timesteps": 128, "init_model": str(self.source / "model.zip"),
               "eval_every": 128, "eval_games": 2}
        run = run_main(self.tmp / "eval", new)
        strip = lambda row: {k: v for k, v in row.items()  # noqa: E731
                             if k not in TIMING and k != "timesteps"}
        first = read_csv(run / "eval_curve.csv")[0]
        self.assertEqual(first["timesteps"], "0")
        self.assertEqual(strip(first), strip(read_csv(self.source / "eval_curve.csv")[-1]))
        self.assertEqual(folder_hashes(self.source), self.source_hashes)

    def test_continue_with_stages_and_an_offset(self):
        new = {**TINY, "init_model": str(self.source / "checkpoints" / "model_256.zip"),
               "opponents": "idle", "eval_opponents": "random", "total_timesteps": 256,
               "eval_every": 128, "eval_games": 2, "env_seed_offset": 1000,
               "stages": [{"opponents": "idle", "timesteps": 128},
                          {"opponents": "random", "timesteps": 128}]}
        run = run_main(self.tmp / "staged", new)
        meta = read_json(run / "meta.json")
        self.assertEqual(meta["seed"], 0)
        self.assertTrue(run.name.endswith("_ppo_seed0"))
        self.assertEqual(meta["init_model"]["num_timesteps"], 256)
        self.assertEqual([(s["start_step"], s["env_seed"]) for s in meta["stages"]],
                         [(0, 1000), (128, 1002)])
        self.assertEqual(meta["stages"][0]["checkpoint"], "checkpoints/stage0_idle_128.zip")
        self.assertTrue((run / "model.zip").exists())
        self.assertEqual(len(read_csv(run / "eval_curve.csv")), 3)
        self.assertEqual(folder_hashes(self.source), self.source_hashes)

    def test_a_failed_load_closes_the_envs(self):
        built = []
        real = train_ppo.make_model

        def make_model(cfg, seed):
            model = real(cfg, seed)
            built.append(mock.patch.object(model.get_env(), "close",
                                           wraps=model.get_env().close).start())
            return model

        try:
            with mock.patch.object(train_ppo, "make_model", make_model), \
                    mock.patch.object(train_ppo, "load_init_model",
                                      side_effect=RuntimeError("boom")), \
                    self.assertRaisesRegex(RuntimeError, "boom"):
                run_main(self.tmp / "failed", {**TINY, "total_timesteps": 128,
                                               "init_model": str(self.source / "model.zip")})
            self.assertEqual(len(built), 1)
            built[0].assert_called()
        finally:
            mock.patch.stopall()

    def test_a_chain_and_a_model_outside_any_run_folder(self):
        loose = self.tmp / "sent_by_a_teammate"
        loose.mkdir()
        shutil.copy(self.source / "model.zip", loose / "model.zip")
        run_b = run_main(self.tmp / "b", {**TINY, "total_timesteps": 128,
                                          "init_model": str(loose / "model.zip")})
        record_b = read_json(run_b / "meta.json")["init_model"]
        self.assertIsNone(record_b["source_run_dir"])
        self.assertIsNone(record_b["source_seed"])
        self.assertEqual(record_b["num_timesteps"], 256)
        run_c = run_main(self.tmp / "c", {**TINY, "total_timesteps": 128,
                                          "init_model": str(run_b / "model.zip")})
        record_c = read_json(run_c / "meta.json")["init_model"]
        self.assertEqual(record_c["source_run_dir"], run_b.name)
        self.assertEqual(record_c["source_init_model"], record_b)
        self.assertEqual(record_c["num_timesteps"], 128)


@unittest.skipIf(torch is None, MISSING)
class EnvSeedOffsetTest(unittest.TestCase):
    def cfg(self, **changes):
        return apply_overrides(PPOConfig(), {**TINY, **changes})

    def test_only_the_env_seeds_move(self):
        for offset, vec_env in ((0, "dummy"), (500, "dummy"), (500, "subproc")):
            with self.subTest(offset=offset, vec_env=vec_env):
                model = train_ppo.make_model(self.cfg(env_seed_offset=offset,
                                                      vec_env=vec_env), 7)
                try:
                    self.assertEqual(model.get_env()._seeds, [7 + offset, 8 + offset])
                    self.assertEqual(model.seed, 7)
                finally:
                    model.get_env().close()

    def test_plan_runner_checks_the_shifted_seeds(self):
        plan = {"name": "p", "seeds": [0], "config": {**TINY, "total_timesteps": 256,
                                                      "env_seed_offset": 999_997},
                "variants": {"c": {"opponents": "idle", "eval_opponents": "random",
                                   "stages": [{"opponents": "idle", "timesteps": 128},
                                              {"opponents": "random", "timesteps": 128}]}}}
        with self.assertRaisesRegex(ValueError, "variant c, seed 0"):
            run_ppo_seeds.load_plan(json.dumps(plan))
        plan["config"]["env_seed_offset"] = 0
        run_ppo_seeds.load_plan(json.dumps(plan))
        plan["config"]["env_seed_offset"] = -5
        with self.assertRaisesRegex(ValueError, "env_seed_offset"):
            run_ppo_seeds.load_plan(json.dumps(plan))


@unittest.skipIf(torch is None, MISSING)
class DefaultIsUnchangedTest(unittest.TestCase):
    def test_same_model_as_today_bit_for_bit(self):
        """``main`` without the options equals today's path: make_model, then learn."""
        overrides = {**TINY, "total_timesteps": 256}
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            run_a = run_main(tmp / "a", overrides)
            run_b = run_main(tmp / "b", {**overrides, "env_seed_offset": 0})
            cfg = apply_overrides(PPOConfig(), overrides)
            model = train_ppo.make_model(cfg, 0)
            try:
                model.learn(total_timesteps=256, callback=CallbackList(
                    [train_ppo.OutcomeCounter(), train_ppo.RewardTermLogger()]))
                model.save(tmp / "reference.zip")
            finally:
                model.get_env().close()
            assert_same_model(self, run_a / "model.zip", tmp / "reference.zip")
            assert_same_model(self, run_a / "model.zip", run_b / "model.zip")
            meta = read_json(run_a / "meta.json")
            self.assertNotIn("init_model", meta)
            config = read_json(run_a / "config.json")
            self.assertIsNone(config["init_model"])
            self.assertEqual(config["env_seed_offset"], 0)


@unittest.skipUnless(CUDA, "needs a CUDA GPU and a CUDA build of torch")
class CrossDeviceTest(unittest.TestCase):
    def test_cuda_saved_continues_on_cpu_and_back(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            gpu = run_main(tmp / "gpu", {**TINY, "total_timesteps": 128, "device": "cuda"})
            cpu = run_main(tmp / "cpu", {**TINY, "total_timesteps": 128, "device": "cpu",
                                         "init_model": str(gpu / "model.zip")})
            self.assertTrue(read_json(cpu / "meta.json")["init_model"]["verified"])
            back = run_main(tmp / "back", {**TINY, "total_timesteps": 128, "device": "cuda",
                                           "init_model": str(cpu / "model.zip")})
            meta = read_json(back / "meta.json")
            self.assertTrue(meta["init_model"]["verified"])
            self.assertTrue(meta["runtime"]["device"].startswith("cuda"))


if __name__ == "__main__":
    unittest.main()
