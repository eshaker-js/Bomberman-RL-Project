"""The device option of PPO training (Task 9, part 1).

``PPOConfig.device`` defaults to ``"cpu"`` and is saved in ``config.json``;
``"cuda"`` without a usable GPU stops before any env, process or run folder
is created (in ``rl.train_ppo`` and in ``rl.run_ppo_seeds``), with a message
that says how to check the install; ``meta.json`` records the device
actually used; the runner passes the device through to training; a saved
model always loads and evaluates on the CPU; and the one SB3 warning that
is silenced on the GPU is matched exactly.

The real CUDA tests are skipped without a usable GPU and run on a computer
with one. Needs numpy, gymnasium, torch and stable-baselines3, like the
other RL tests. Only few-step runs, in temporary folders.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
import warnings
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
CUDA = torch is not None and torch.cuda.is_available()
NO_CUDA = "CUDA not available (no NVIDIA GPU or a CPU-only torch build); runs on the laptop"

if torch is not None:
    from rl import devices, run_ppo_seeds, train_ppo
    from rl.bomberman_env import BombermanGymEnv
    from rl.config import PPOConfig, apply_overrides
    from rl.evaluate_agent import PPOPolicy, evaluate_seeds
    from rl.seeds import TRAIN_EVAL_SEEDS

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}


def no_cuda():
    """Pretend this computer has no usable GPU, whatever it really has."""
    return mock.patch("torch.cuda.is_available", return_value=False)


@unittest.skipIf(torch is None, MISSING)
class DeviceTestCase(unittest.TestCase):
    def setUp(self):
        self._threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self._threads)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.runs = self.tmp / "runs"

    def config_file(self, **extra) -> str:
        path = self.tmp / f"c{len(list(self.tmp.glob('c*.json')))}.json"
        path.write_text(json.dumps({**TINY, **extra}))
        return str(path)

    def train(self, *args, **extra) -> Path:
        with contextlib.redirect_stdout(io.StringIO()):
            return train_ppo.main(["--seed", "1", "--runs-dir", str(self.runs),
                                   "--config", self.config_file(**extra), *args])

    def run_folders(self) -> list:
        return sorted(self.runs.iterdir()) if self.runs.exists() else []


class DeviceConfigTest(DeviceTestCase):
    def test_default_is_cpu_and_is_saved(self):
        self.assertEqual(PPOConfig().device, "cpu")
        run = self.train("--timesteps", "64")
        config = json.loads((run / "config.json").read_text())
        self.assertEqual(config["device"], "cpu")
        runtime = json.loads((run / "meta.json").read_text())["runtime"]
        self.assertEqual(runtime["device"], "cpu")
        self.assertNotIn("gpu", runtime)

    def test_command_line_and_json_values(self):
        run = self.train("--timesteps", "64", "--device", "cpu", device="cuda")
        self.assertEqual(json.loads((run / "config.json").read_text())["device"], "cpu")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train_ppo.main(["--device", "gpu"])
        with self.assertRaises(ValueError):
            train_ppo.make_model(apply_overrides(PPOConfig(), {**TINY, "device": "gpu"}), 0)
        with self.assertRaises(ValueError):
            self.train("--timesteps", "64", device="cuda:0")
        self.assertEqual(self.run_folders(), [run])

    def test_cuda_without_cuda_stops_before_anything_is_built(self):
        cfg = apply_overrides(PPOConfig(), {**TINY, "device": "cuda"})
        with no_cuda(), mock.patch.object(train_ppo, "make_vec_env") as vec, \
                self.assertRaises(RuntimeError) as caught:
            train_ppo.make_model(cfg, 0)
        vec.assert_not_called()
        message = str(caught.exception)
        for text in ("torch.cuda.is_available()", "torch.version.cuda", "nvidia-smi",
                     torch.__version__):
            self.assertIn(text, message)
        with no_cuda(), self.assertRaises(RuntimeError):
            self.train("--timesteps", "64", "--device", "cuda")
        self.assertEqual(self.run_folders(), [])  # no run folder either

    def test_gpu_info(self):
        with no_cuda():
            self.assertIsNone(devices.gpu_info())
        props = mock.Mock(total_memory=6 * 2**30)
        smi = mock.Mock(stdout="566.14\n")
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.get_device_properties", return_value=props), \
                mock.patch("torch.cuda.get_device_name", return_value="Test GPU"), \
                mock.patch("torch.cuda.device_count", return_value=1), \
                mock.patch.object(devices.subprocess, "run", return_value=smi):
            info = devices.gpu_info()
        self.assertEqual(info, {"name": "Test GPU", "memory_gb": 6.0,
                                "cuda": torch.version.cuda, "driver": "566.14", "count": 1})
        with mock.patch.object(devices.subprocess, "run", side_effect=OSError):
            self.assertIsNone(devices.driver_version())

    def test_device_record(self):
        cpu = mock.Mock(device=torch.device("cpu"))
        self.assertEqual(train_ppo.device_record(cpu), {"device": "cpu"})
        gpu = mock.Mock(device=torch.device("cuda"))
        with mock.patch.object(train_ppo, "gpu_info", return_value={"name": "x"}):
            record = train_ppo.device_record(gpu)
        self.assertEqual((record["device"], record["gpu"]), ("cuda", {"name": "x"}))
        self.assertIn("not bit-identical", record["reproducibility"])

    def test_only_sbs_gpu_warning_is_silenced(self):
        """The filter of ``make_model`` matches SB3's real warning, and only it."""
        model = train_ppo.make_model(apply_overrides(PPOConfig(), TINY), 0)
        try:
            model.device = torch.device("cuda")  # only to make SB3 emit its warning
            with warnings.catch_warnings(record=True) as plain:
                warnings.simplefilter("always")
                model._maybe_recommend_cpu()
            self.assertEqual(len(plain), 1)
            with warnings.catch_warnings(record=True) as filtered:
                warnings.simplefilter("always")
                warnings.filterwarnings("ignore", message=train_ppo.GPU_MLP_WARNING,
                                        category=UserWarning)
                model._maybe_recommend_cpu()
                warnings.warn("another warning", UserWarning)
            self.assertEqual([str(w.message) for w in filtered], ["another warning"])
        finally:
            model.get_env().close()


class CpuLoadingTest(DeviceTestCase):
    def test_saved_model_loads_and_evaluates_on_the_cpu(self):
        run = self.train("--timesteps", "64", device="cpu")
        policy = PPOPolicy(run / "model.zip")
        self.assertTrue(all(p.device.type == "cpu"
                            for p in policy.model.policy.parameters()))
        env = BombermanGymEnv("random", PPOConfig().env)
        try:
            totals = evaluate_seeds(env, policy, TRAIN_EVAL_SEEDS[:2])
        finally:
            env.close()
        self.assertEqual(totals["games"], 2)

    def test_ppo_policy_always_asks_for_the_cpu(self):
        with mock.patch("stable_baselines3.PPO.load") as load:
            PPOPolicy(Path("some/model.zip"))
        self.assertEqual(load.call_args.kwargs["device"], "cpu")


class RunnerDeviceTest(DeviceTestCase):
    PLAN = {"name": "tiny", "seeds": [0], "config": {**TINY, "total_timesteps": 64,
                                                     "opponents": "random"},
            "variants": {"v0": {}}, "evaluation": {"games": 1}}

    def plan_file(self, **config) -> Path:
        plan = {**self.PLAN, "config": {**self.PLAN["config"], **config}}
        path = self.tmp / "plan.json"
        path.write_text(json.dumps(plan))
        return path

    def test_device_passes_through_to_training(self):
        plan = run_ppo_seeds.load_plan(self.plan_file(device="cpu").read_text())
        job = run_ppo_seeds.Job(self.tmp / "plan", plan, "v0", 0)
        job.refresh()
        command = job.next_command()
        overrides = json.loads((job.dir / "config_overrides.json").read_text())
        self.assertEqual(overrides["device"], "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            run = train_ppo.main(command[command.index("--seed"):])
        self.assertEqual(json.loads((run / "config.json").read_text())["device"], "cpu")

    def test_cuda_plan_without_cuda_is_refused_before_any_folder(self):
        path = self.plan_file(device="cuda")
        with no_cuda(), self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            run_ppo_seeds.main([str(path), "--runs-dir", str(self.runs), "--quiet"])
        self.assertIn("variant v0", err.getvalue())
        self.assertIn("torch.cuda.is_available()", err.getvalue())
        self.assertEqual(self.run_folders(), [])


@unittest.skipIf(torch is None, MISSING)
@unittest.skipUnless(CUDA, NO_CUDA)
class RealCudaTest(DeviceTestCase):
    def test_cuda_run_records_the_gpu(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = self.train("--timesteps", "128", "--device", "cuda",
                             "--eval-every", "64", "--eval-games", "2")
        self.assertFalse([w for w in caught if "on the GPU" in str(w.message)])
        runtime = json.loads((run / "meta.json").read_text())["runtime"]
        self.assertTrue(runtime["device"].startswith("cuda"))
        for key in ("name", "memory_gb", "cuda", "driver"):
            self.assertIn(key, runtime["gpu"])
        self.assertIn("not bit-identical", runtime["reproducibility"])
        self.assertTrue((run / "eval_curve.csv").exists())

    def test_gpu_trained_model_loads_on_the_cpu_unchanged(self):
        model = train_ppo.make_model(apply_overrides(PPOConfig(), {**TINY, "device": "cuda"}), 1)
        try:
            model.learn(total_timesteps=128)
            trained = {k: v.detach().cpu().clone()
                       for k, v in model.policy.state_dict().items()}
            model.save(self.tmp / "model.zip")
        finally:
            model.get_env().close()
        policy = PPOPolicy(self.tmp / "model.zip")
        loaded = policy.model.policy.state_dict()
        self.assertEqual(trained.keys(), loaded.keys())
        for key, value in loaded.items():
            self.assertEqual(value.device.type, "cpu", key)
            self.assertTrue(torch.equal(trained[key], value), key)
        env = BombermanGymEnv("random", PPOConfig().env)
        try:
            self.assertEqual(evaluate_seeds(env, policy, TRAIN_EVAL_SEEDS[:2])["games"], 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
