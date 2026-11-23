"""PPO training options (Task 8, part 2).

Each option is used and saved in ``config.json``; ``subproc`` (spawn) gives
the same model as ``dummy``; checkpoints and the evaluation during training
are written as described, and the learning-curve counts match an
independent re-evaluation of the saved checkpoints; the evaluation does not
change training; the defaults train exactly as before Task 8.

Needs numpy, gymnasium, torch and stable-baselines3, like the other RL
tests. Only few-step runs, in temporary folders.
"""

import builtins
import contextlib
import csv
import io
import json
import pickle
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
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

from rl import priority  # noqa: E402  (standard library only)

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl import train_ppo
    from rl.bomberman_env import BombermanGymEnv
    from rl.config import PPOConfig, apply_overrides
    from rl.evaluate_agent import ModelPolicy, PPOPolicy, evaluate_seeds
    from rl.networks import SmallGridCNN
    from rl.seeds import TRAIN_EVAL_SEEDS, TrainingSeedGuard

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}


def params(model) -> dict:
    return {k: v.clone() for k, v in model.policy.state_dict().items()}


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipIf(torch is None, MISSING)
class OptionsTestCase(unittest.TestCase):
    def setUp(self):
        self._threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self._threads)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def config_file(self, **extra) -> str:
        path = self.tmp / f"c{len(list(self.tmp.iterdir()))}.json"
        path.write_text(json.dumps({**TINY, **extra}))
        return str(path)

    def train(self, *args, **extra) -> Path:
        with contextlib.redirect_stdout(io.StringIO()):
            return train_ppo.main(["--seed", "1", "--runs-dir", str(self.tmp),
                                   "--config", self.config_file(**extra), *args])

    def learned(self, cfg, seed=1, steps=256, callback=None):
        model = train_ppo.make_model(cfg, seed)
        try:
            model.learn(total_timesteps=steps, callback=callback)
            return params(model)
        finally:
            model.get_env().close()


class ConfigTest(OptionsTestCase):
    def test_defaults_keep_todays_behaviour(self):
        c = PPOConfig()
        self.assertEqual((c.vec_env, c.torch_threads, c.high_priority), ("dummy", 0, False))
        self.assertEqual((c.checkpoint_every, c.eval_every), (0, 0))
        self.assertEqual((c.checkpoint_keep_every, c.eval_games), (1_000_000, 50))

    def test_each_option_is_used_and_saved(self):
        run = self.train("--timesteps", "128", "--torch-threads", "2", "--high-priority",
                         "--checkpoint-every", "64", "--eval-every", "64",
                         "--eval-games", "2")
        config = json.loads((run / "config.json").read_text())
        self.assertEqual((config["torch_threads"], config["high_priority"],
                          config["checkpoint_every"], config["eval_every"],
                          config["eval_games"], config["vec_env"]),
                         (2, True, 64, 64, 2, "dummy"))
        meta = json.loads((run / "meta.json").read_text())
        self.assertEqual(meta["runtime"]["torch_threads"], 2)
        if sys.platform != "win32":
            self.assertIn("nothing changed", meta["runtime"]["priority"])
        self.assertTrue((run / "checkpoints").is_dir())
        self.assertTrue((run / "eval_curve.csv").exists())

    def test_json_values_and_bad_values(self):
        run = self.train("--timesteps", "64", vec_env="dummy", eval_every=64, eval_games=1)
        self.assertEqual(json.loads((run / "config.json").read_text())["eval_games"], 1)
        with self.assertRaises(ValueError):
            train_ppo.make_model(apply_overrides(PPOConfig(), {**TINY, "vec_env": "x"}), 0)
        with self.assertRaises(ValueError):
            train_ppo.apply_runtime(apply_overrides(PPOConfig(), {"torch_threads": -1}))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_ppo.main(["--vec-env", "fork"])

    def test_default_run_has_no_new_files(self):
        run = self.train("--timesteps", "64")
        self.assertEqual(sorted(p.name for p in run.iterdir()),
                         ["config.json", "meta.json", "model.zip", "progress.csv"])


class SameModelTest(OptionsTestCase):
    def old_make_model(self, cfg, seed):
        """``rl.train_ppo.make_model`` exactly as before Task 8 (fe0bd9c)."""
        envs = make_vec_env(lambda: BombermanGymEnv(cfg.opponents, cfg.env),
                            n_envs=cfg.n_envs, seed=seed, vec_env_cls=DummyVecEnv)
        policy_kwargs = {
            "features_extractor_class": SmallGridCNN,
            "features_extractor_kwargs": {"features_dim": cfg.features_dim,
                                          "conv_channels": tuple(cfg.conv_channels)},
            "net_arch": {"pi": list(cfg.net_arch_pi), "vf": list(cfg.net_arch_vf)},
            "normalize_images": False,
        }
        return PPO("MlpPolicy", envs, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
                   batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
                   gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
                   ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
                   max_grad_norm=cfg.max_grad_norm, policy_kwargs=policy_kwargs,
                   seed=seed, device="cpu", verbose=0)

    def assert_same(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_defaults_reproduce_todays_model(self):
        cfg = apply_overrides(PPOConfig(), TINY)
        old = self.old_make_model(cfg, 1)
        try:
            old.learn(total_timesteps=256)
            before = params(old)
        finally:
            old.get_env().close()
        self.assert_same(before, self.learned(cfg))

    def test_subproc_spawn_gives_the_same_model_as_dummy(self):
        dummy = self.learned(apply_overrides(PPOConfig(), TINY))
        subproc = self.learned(apply_overrides(PPOConfig(), {**TINY, "vec_env": "subproc"}))
        self.assert_same(dummy, subproc)

    def test_evaluation_during_training_does_not_change_training(self):
        cfg = apply_overrides(PPOConfig(), TINY)
        plain = self.learned(cfg)
        evaluated = apply_overrides(cfg, {"eval_every": 64, "eval_games": 3})
        callback = train_ppo.TrainingEvalCallback(evaluated, self.tmp)
        try:
            with_eval = self.learned(evaluated, callback=callback)
        finally:
            callback.close()
        self.assertEqual(len(callback.rows), 5)  # 0, 64 .. 256
        self.assert_same(plain, with_eval)

    def test_env_factory_is_picklable(self):
        factory = train_ppo.EnvFactory("noisy25", PPOConfig().env, True)
        copy = pickle.loads(pickle.dumps(factory))
        self.assertEqual(copy, factory)
        env = copy()
        try:
            self.assertIsInstance(env, TrainingSeedGuard)
            self.assertEqual(env.reset(seed=3)[1]["episode_seed"], 3)
        finally:
            env.close()


class CheckpointTest(OptionsTestCase):
    def test_checkpoints_appear_and_spares_are_deleted(self):
        run = self.train("--timesteps", "512", "--checkpoint-every", "128",
                         checkpoint_keep_every=256)
        names = sorted(p.name for p in (run / "checkpoints").iterdir())
        self.assertEqual(names, ["model_256.zip", "model_512.zip"])
        self.assertTrue((run / "model.zip").exists())
        PPOPolicy(run / "checkpoints" / "model_256.zip")  # loads

    def test_keeper_keeps_the_latest_and_one_per_block_and_only_deletes_its_own(self):
        class FakeModel:
            def save(self, path):
                Path(path).write_text("m")

        folder = self.tmp / "checkpoints"
        keeper = train_ppo.CheckpointKeeper(folder, 1000)
        for steps in (100, 400, 800, 1200, 1600, 2000, 2400, 3100):
            keeper.save(FakeModel(), steps)
        names = sorted(p.name for p in folder.iterdir())
        # The first of each 1,000-step block (1200, 2000) and the latest (3100).
        self.assertEqual(names, ["model_1200.zip", "model_2000.zip", "model_3100.zip"])

        other = self.tmp / "other"
        other.mkdir()
        (other / "model_5.zip").write_text("not mine")
        keeper = train_ppo.CheckpointKeeper(other, 1000)
        for steps in (100, 200):
            keeper.save(FakeModel(), steps)
        self.assertEqual(sorted(p.name for p in other.iterdir()),
                         ["model_200.zip", "model_5.zip"])
        with self.assertRaises(FileExistsError):  # never overwrites a file
            keeper.save(FakeModel(), 5)


class TrainingEvalTest(OptionsTestCase):
    def test_curve_files_and_an_independent_recount(self):
        run = self.train("--timesteps", "256", "--eval-every", "128", "--eval-games", "4",
                         "--checkpoint-every", "128", checkpoint_keep_every=1)
        rows = read_csv(run / "eval_curve.csv")
        self.assertEqual(list(rows[0]), list(train_ppo.CURVE_FIELDS))
        self.assertEqual([int(r["timesteps"]) for r in rows], [0, 128, 256])
        games = read_csv(run / "eval_games.csv")
        self.assertEqual(list(games[0]), list(train_ppo.CURVE_GAME_FIELDS))
        self.assertEqual(len(games), 12)
        seeds = [int(g["seed"]) for g in games]
        self.assertEqual(seeds, [2_000_000, 2_000_001, 2_000_002, 2_000_003] * 3)
        self.assertTrue(all(s in TRAIN_EVAL_SEEDS for s in seeds))

        cfg = apply_overrides(PPOConfig(), TINY)
        for row in rows[1:]:
            steps = int(row["timesteps"])
            path = run / "checkpoints" / f"model_{steps}.zip"
            if steps == 256:
                self.assertTrue((run / "model.zip").exists())
            env = BombermanGymEnv(cfg.opponents, cfg.env)
            try:
                recount = evaluate_seeds(env, PPOPolicy(path), range(2_000_000, 2_000_004))
            finally:
                env.close()
            for key in ("win", "loss", "draw", "timeout", "self_kills", "kills", "crates",
                        "buffs_picked", "buffs_raised", "death_total", "death_own_fresh",
                        "death_opponent_fresh", "death_own_older", "death_opponent_older"):
                self.assertEqual(int(row[key]), recount[key], (steps, key))
            self.assertAlmostEqual(float(row["mean_length"]), recount["steps"] / 4)
            mine = [g for g in games if int(g["timesteps"]) == steps]
            self.assertEqual(sum(int(g["steps"]) for g in mine), recount["steps"])

        # The step-0 row is the untrained policy: recount it with a fresh model.
        model = train_ppo.make_model(cfg, 1)
        try:
            env = BombermanGymEnv(cfg.opponents, cfg.env)
            try:
                first = evaluate_seeds(env, ModelPolicy(model), range(2_000_000, 2_000_004))
            finally:
                env.close()
        finally:
            model.get_env().close()
        for key in ("games", "win", "loss", "draw", "timeout", "self_kills", "death_total"):
            self.assertEqual(int(rows[0][key]), first[key], key)
        self.assertAlmostEqual(float(rows[0]["mean_length"]), first["steps"] / 4)

    def test_files_closed_also_on_a_crash(self):
        real_open = builtins.open
        opened = []

        def recording_open(file, *args, **kwargs):
            handle = real_open(file, *args, **kwargs)
            if str(file).startswith(str(self.tmp)):
                opened.append(handle)
            return handle

        with mock.patch.object(builtins, "open", recording_open), \
                mock.patch.object(train_ppo.PPO, "save", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                self.train("--timesteps", "128", "--eval-every", "64", "--eval-games", "1")
        names = [Path(h.name).name for h in opened]
        self.assertIn("eval_curve.csv", names)
        self.assertEqual([n for n, h in zip(names, opened) if not h.closed], [])
        for handle in opened:
            handle.close()
        run = next(self.tmp.glob("*_ppo_seed1"))
        self.assertEqual(len(read_csv(run / "eval_curve.csv")), 3)  # 0, 64, 128 kept


class PriorityTest(unittest.TestCase):
    def test_no_op_off_windows(self):
        with mock.patch.object(priority.sys, "platform", "linux"):
            self.assertIn("nothing changed", priority.raise_priority())

    def test_windows_path_asks_for_high_never_realtime(self):
        kernel32 = mock.Mock()
        kernel32.GetCurrentProcess.return_value = 42
        kernel32.SetPriorityClass.return_value = 1
        with mock.patch.object(priority.sys, "platform", "win32"), \
                mock.patch.object(priority, "_kernel32", return_value=kernel32):
            self.assertEqual(priority.raise_priority(), "process priority raised to High")
        kernel32.SetPriorityClass.assert_called_once_with(42, 0x80)


if __name__ == "__main__":
    unittest.main()
