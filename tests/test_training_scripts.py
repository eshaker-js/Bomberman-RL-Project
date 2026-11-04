"""PPO training script, its CNN and agent evaluation (Task 6, stage 4).

Needs numpy, gymnasium, torch and stable-baselines3; every test here is
skipped with a clear reason if one is missing. Only a few-step PPO run, in a
temporary folder (never ``runs/``).
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

try:
    import gymnasium  # noqa: F401
    import numpy as np
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = ("numpy/gymnasium/torch/stable-baselines3 not installed "
           "(pip install -r requirements.txt); PPO test skipped")

if torch is not None:
    from rl.bomberman_env import BombermanGymEnv
    from rl.config import PPOConfig, apply_overrides, load_config
    from rl.evaluate_agent import main as evaluate_main
    from rl.networks import SmallGridCNN
    from rl.train_ppo import main as train_main

TINY_PPO = {"n_envs": 1, "n_steps": 64, "batch_size": 32, "n_epochs": 1,
            "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 50}}


@unittest.skipIf(torch is None, MISSING)
class NetworkTest(unittest.TestCase):
    def test_small_cnn_takes_the_11x11_observation(self):
        space = BombermanGymEnv(None).observation_space
        net = SmallGridCNN(space, features_dim=256)
        obs = torch.zeros((4, *space.shape))
        self.assertEqual(tuple(net(obs).shape), (4, 256))
        self.assertEqual(net.features_dim, 256)
        convs = [m for m in net.cnn if isinstance(m, torch.nn.Conv2d)]
        self.assertEqual([c.out_channels for c in convs], [32, 64, 64])
        self.assertTrue(all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs))


@unittest.skipIf(torch is None, MISSING)
class PPOConfigTest(unittest.TestCase):
    def test_defaults_are_the_planned_v0(self):
        c = PPOConfig()
        self.assertEqual((c.n_envs, c.n_steps, c.batch_size, c.n_epochs), (8, 256, 256, 10))
        self.assertEqual((c.learning_rate, c.gamma, c.gae_lambda, c.clip_range),
                         (3e-4, 0.99, 0.95, 0.2))
        self.assertEqual((c.ent_coef, c.vf_coef, c.max_grad_norm), (0.01, 0.5, 0.5))
        self.assertEqual(c.opponents, "random")
        self.assertEqual(c.env.max_steps, 800)

    def test_json_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(TINY_PPO))
            cfg = load_config(PPOConfig(), path)
        self.assertEqual(cfg.conv_channels, (8, 8, 8))
        self.assertEqual(cfg.env.max_steps, 50)
        self.assertEqual(cfg.env.reward.kill, 0.3)
        with self.assertRaises(KeyError):
            apply_overrides(PPOConfig(), {"n_env": 2})


@unittest.skipIf(torch is None, MISSING)
class PPOScriptTest(unittest.TestCase):
    def test_few_step_run_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            overrides = Path(tmp) / "c.json"
            overrides.write_text(json.dumps(TINY_PPO))
            with contextlib.redirect_stdout(io.StringIO()):
                run = train_main(["--timesteps", "128", "--seed", "1",
                                  "--config", str(overrides), "--runs-dir", tmp])
            for name in ("config.json", "meta.json", "progress.csv", "model.zip"):
                self.assertTrue((run / name).exists(), name)
            meta = json.loads((run / "meta.json").read_text())
            self.assertEqual(meta["kind"], "ppo")
            self.assertGreaterEqual(meta["summary"]["timesteps"], 128)
            self.assertIn("torch", meta["versions"])
            config = json.loads((run / "config.json").read_text())
            self.assertEqual(config["total_timesteps"], 128)

            with contextlib.redirect_stdout(io.StringIO()):
                out, results = evaluate_main(["--run", str(run), "--runs-dir", tmp,
                                              "--games", "2", "--max-steps", "30",
                                              "--opponents", "noisy50"])
            self.assertEqual(results["games"], 2)
            self.assertEqual(sum(results[k] for k in ("win", "loss", "draw", "timeout")), 2)
            saved = json.loads((out / "eval.json").read_text())
            self.assertEqual(saved["settings"]["stage"], "full")
            self.assertEqual(saved["settings"]["opponents"], "noisy50")

            with self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                evaluate_main(["--run", str(run), "--runs-dir", tmp, "--stage", "tiny"])

    def test_evaluation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            overrides = Path(tmp) / "c.json"
            overrides.write_text(json.dumps(TINY_PPO))
            with contextlib.redirect_stdout(io.StringIO()):
                run = train_main(["--timesteps", "64", "--seed", "0",
                                  "--config", str(overrides), "--runs-dir", tmp])
                args = ["--run", str(run), "--games", "2", "--max-steps", "25"]
                _, a = evaluate_main([*args, "--runs-dir", str(Path(tmp) / "a")])
                _, b = evaluate_main([*args, "--runs-dir", str(Path(tmp) / "b")])
        self.assertEqual(a, b)
        self.assertIsNotNone(np)


if __name__ == "__main__":
    unittest.main()
