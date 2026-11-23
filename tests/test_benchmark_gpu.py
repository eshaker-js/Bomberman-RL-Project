"""The GPU part of the PPO benchmark (Task 9, part 2).

Without CUDA the quick benchmark skips the GPU part, says so, and still
writes a complete ``benchmark.json`` whose conclusion is the CPU; the
conclusion picks between hand-made CPU and GPU rows correctly (the GPU only
when strictly better, unmeasured entries ignored); the GPU memory per run
and the runs that fit are computed as documented. The real quick GPU
benchmark is skipped without CUDA and runs on a computer with a GPU.

Needs numpy, gymnasium, torch and stable-baselines3, like the other RL
tests. Benchmark output goes to temporary folders.
"""

import contextlib
import dataclasses
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
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"
CUDA = torch is not None and torch.cuda.is_available()
NO_CUDA = "CUDA not available (no NVIDIA GPU or a CPU-only torch build); runs on the laptop"

if torch is not None:
    from rl import benchmark_ppo
    from rl.config import PPOConfig


def run_quick(tmp: str) -> dict:
    threads = torch.get_num_threads()
    try:
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            out, _ = benchmark_ppo.main(["--quick", "--runs-dir", tmp])
    finally:
        torch.set_num_threads(threads)
    saved = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    return saved, printed.getvalue()


@unittest.skipIf(torch is None, MISSING)
class QuickWithoutCudaTest(unittest.TestCase):
    def test_gpu_part_is_skipped_and_the_json_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("torch.cuda.is_available", return_value=False):
            saved, printed = run_quick(tmp)
        self.assertIn("CUDA not available", saved["gpu_skipped"])
        self.assertIn("GPU part skipped", printed)
        self.assertIsNone(saved["machine"]["gpu"])
        for key in ("training_gpu", "side_by_side_gpu", "eval_gpu"):
            self.assertNotIn(key, saved)
        for key in ("env_speed", "training", "side_by_side", "eval", "conclusion",
                    "finished_at"):
            self.assertIn(key, saved)
        self.assertFalse(saved["partial"])
        conclusion = saved["conclusion"]
        self.assertEqual(conclusion["device"], "cpu")
        self.assertEqual(list(conclusion["by_device"]), ["cpu"])
        self.assertIsNone(conclusion["gpu_memory"])
        self.assertTrue(all(p["device"] == "cpu" for p in conclusion["night_plans"]))


@unittest.skipIf(torch is None, MISSING)
class ConclusionTest(unittest.TestCase):
    """``conclude`` on hand-made rows (cycle 8 x 256 = 2,048 steps)."""

    def setUp(self):
        self.cpu = dataclasses.replace(PPOConfig(), torch_threads=4)
        self.gpu = dataclasses.replace(PPOConfig(), torch_threads=2, vec_env="subproc",
                                       device="cuda")

    def results(self, gpu_sps: float, gpu_sbs: list | None = None) -> dict:
        row = {"opponents": "random", "n_envs": 8, "vec_env": "dummy", "torch_threads": 4,
               "steps_per_second": 400.0}
        gpu_row = {**row, "vec_env": "subproc", "torch_threads": 2,
                   "steps_per_second": gpu_sps, "device": "cuda"}
        if gpu_sbs is None:
            gpu_sbs = [{"runs": 1, "torch_threads_each": 2, "measured": True,
                        "steps_per_second_each": [gpu_sps],
                        "total_steps_per_second": gpu_sps, "device": "cuda"}]
        return {"training": [row],
                "side_by_side": [{"runs": 1, "torch_threads_each": 4, "measured": True,
                                  "steps_per_second_each": [400.0],
                                  "total_steps_per_second": 400.0}],
                "training_gpu": [gpu_row], "side_by_side_gpu": gpu_sbs,
                "eval": {"smallest_eval_every_for_5_percent": 50_000},
                "eval_gpu": {"smallest_eval_every_for_5_percent": 70_000}}

    def test_gpu_wins_when_faster(self):
        c = benchmark_ppo.conclude(self.results(600.0), self.cpu, self.gpu)
        self.assertEqual((c["device"], c["vec_env"], c["torch_threads"]),
                         ("cuda", "subproc", 2))
        self.assertEqual(c["single_run_steps_per_second"], 600.0)
        self.assertEqual(c["smallest_eval_every_for_5_percent"], 70_000)
        # 8 h x 3600 x 600 x 0.95 / 5 rounds = 3,283,200 -> whole rollouts: 3,282,944
        self.assertEqual(c["steps_per_seed_per_night"], 3_282_944)
        self.assertEqual(c["by_device"]["cpu"]["best_steps_per_seed_per_night"], 2_187_264)
        self.assertEqual(c["hours_for_1M_steps"], benchmark_ppo.hours(1_000_000, 600.0))
        self.assertEqual(sorted(p["device"] for p in c["night_plans"]), ["cpu", "cuda"])

    def test_cpu_wins_when_faster_or_tied(self):
        for gpu_sps in (300.0, 400.0):
            c = benchmark_ppo.conclude(self.results(gpu_sps), self.cpu, self.gpu)
            self.assertEqual((c["device"], c["torch_threads"]), ("cpu", 4), gpu_sps)
            self.assertEqual(c["smallest_eval_every_for_5_percent"], 50_000)

    def test_unmeasured_gpu_entries_are_ignored(self):
        broken = [{"runs": 2, "torch_threads_each": 2, "measured": False,
                   "steps_per_second_each": [None, 900.0], "total_steps_per_second": None,
                   "device": "cuda"},
                  {"runs": 3, "torch_threads_each": 2, "device": "cuda",
                   "skipped": "oversubscribed"}]
        c = benchmark_ppo.conclude(self.results(600.0, broken), self.cpu, self.gpu)
        # Only the GPU grid speed is left: one run at a time at 600 steps/s.
        self.assertEqual((c["device"], c["runs_side_by_side"]), ("cuda", 1))
        self.assertEqual(c["steps_per_seed_per_night"], 3_282_944)

    def test_without_a_gpu_the_conclusion_is_the_cpu(self):
        results = self.results(600.0)
        for key in ("training_gpu", "side_by_side_gpu", "eval_gpu"):
            del results[key]
        c = benchmark_ppo.conclude(results, self.cpu)
        self.assertEqual((c["device"], list(c["by_device"])), ("cpu", ["cpu"]))
        self.assertIsNone(c["gpu_memory"])


@unittest.skipIf(torch is None, MISSING)
class GpuMemoryAndPlanTest(unittest.TestCase):
    def test_memory_per_run_and_runs_that_fit(self):
        entries = [
            {"runs": 1, "measured": True, "gpu_total_mb": 6144.0,
             "gpu_used_before_mb": 500.0, "gpu_used_peak_mb": 1100.0},
            {"runs": 2, "measured": True, "gpu_total_mb": 6144.0,
             "gpu_used_before_mb": 500.0, "gpu_used_peak_mb": 1700.0},
            {"runs": 3, "measured": False, "gpu_total_mb": 6144.0,
             "gpu_used_before_mb": 500.0, "gpu_used_peak_mb": 9999.0},
            {"runs": 4, "skipped": "oversubscribed"},
        ]
        memory = benchmark_ppo.gpu_memory(entries)
        # From the 2-run entry: (1700 - 500) / 2 = 600 MB per run; (6144 - 500) // 600 = 9.
        self.assertEqual((memory["per_run_mb"], memory["from_runs"]), (600.0, 2))
        self.assertEqual(memory["runs_that_fit"], 9)
        self.assertIsNone(benchmark_ppo.gpu_memory([]))
        self.assertIsNone(benchmark_ppo.gpu_memory(entries[2:]))

    def test_full_plan_settings(self):
        full = benchmark_ppo.FULL
        self.assertEqual((full["gpu_threads"], full["gpu_vec_envs"]),
                         ((1, 2, 4), ("dummy", "subproc")))
        self.assertEqual((full["gpu_info_n_envs"], full["gpu_side_by_side"]),
                         ((16, 32), (1, 2, 3)))
        self.assertEqual(full["gpu_window_seconds"], 30.0)
        # The CPU grid is cut only when the GPU is measured too.
        cuts = benchmark_ppo.FULL_CPU_CUTS_WITH_GPU
        self.assertEqual((cuts["threads"], cuts["rule_threads"]), ((4, 8, 12), 1))
        self.assertEqual((full["threads"], full["rule_threads"]), ((4, 8, 12, 16), 2))

    def test_bench_config_keeps_or_sets_the_device(self):
        base = PPOConfig()
        self.assertEqual(benchmark_ppo.bench_config(base, "random", 8, "dummy", 4).device, "cpu")
        self.assertEqual(benchmark_ppo.bench_config(base, "rule", 8, "dummy", 1,
                                                    "cuda").device, "cuda")


@unittest.skipIf(torch is None, MISSING)
@unittest.skipUnless(CUDA, NO_CUDA)
class QuickWithCudaTest(unittest.TestCase):
    def test_quick_gpu_part(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved, printed = run_quick(tmp)
        self.assertIsNone(saved["gpu_skipped"])
        self.assertIsNotNone(saved["machine"]["gpu"])
        self.assertTrue(saved["training_gpu"])
        self.assertTrue(all(r["device"] == "cuda" and r["steps_per_second"] > 0
                            and r["gpu_peak_reserved_mb"] > 0 for r in saved["training_gpu"]))
        self.assertTrue(all(r["outside_conclusion"] for r in saved["training_gpu_info"]))
        self.assertEqual([r["runs"] for r in saved["side_by_side_gpu"]], [1, 2])
        for entry in saved["side_by_side_gpu"]:
            self.assertTrue(entry["measured"])
            self.assertGreater(entry["gpu_used_peak_mb"], 0)
        self.assertIn("eval_gpu", saved)
        conclusion = saved["conclusion"]
        self.assertIn(conclusion["device"], ("cpu", "cuda"))
        self.assertEqual(sorted(conclusion["by_device"]), ["cpu", "cuda"])
        self.assertIsNotNone(conclusion["gpu_memory"])
        self.assertIn("on the GPU", printed)


if __name__ == "__main__":
    unittest.main()
