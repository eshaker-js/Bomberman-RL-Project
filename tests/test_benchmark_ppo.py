"""The laptop benchmark in quick mode (Task 8, part 3).

Runs ``rl.benchmark_ppo --quick`` (tiny network, a few hundred steps, two
side-by-side worker processes started with spawn) in a temporary folder and
checks that ``benchmark.json`` has every part. Needs numpy, gymnasium, torch
and stable-baselines3, like the other RL tests.
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
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from rl import benchmark_ppo


@unittest.skipIf(torch is None, MISSING)
class QuickBenchmarkTest(unittest.TestCase):
    def test_quick_mode_writes_its_json(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                out, results = benchmark_ppo.main(["--quick", "--runs-dir", tmp])
            saved = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
        self.assertTrue(out.name.endswith("_benchmark_seed0"))
        self.assertTrue(saved["quick"])
        self.assertFalse(saved["partial"])
        for key in ("cpu", "physical_cores", "logical_threads", "ram_gb", "python", "torch"):
            self.assertIn(key, saved["machine"])
        self.assertEqual([r["opponents"] for r in saved["env_speed"]], ["random"])
        self.assertEqual({r["vec_env"] for r in saved["training"]}, {"dummy", "subproc"})
        self.assertTrue(all(r["steps_per_second"] > 0 for r in saved["training"]))
        self.assertEqual([r["runs"] for r in saved["side_by_side"]], [1, 2])
        self.assertTrue(all(r["total_steps_per_second"] > 0 for r in saved["side_by_side"]))
        self.assertEqual(len(saved["side_by_side"][1]["steps_per_second_each"]), 2)
        conclusion = saved["conclusion"]
        for key in ("vec_env", "torch_threads", "single_run_steps_per_second",
                    "hours_for_1M_steps", "hours_for_10M_steps", "runs_side_by_side",
                    "steps_per_seed_per_night", "smallest_eval_every_for_5_percent"):
            self.assertIn(key, conclusion)
        self.assertEqual(conclusion["steps_per_seed_per_night"] % (2 * 32), 0)
        self.assertIn("Conclusion", printed.getvalue())
        self.assertEqual(results["conclusion"], conclusion)

    def test_night_plan_arithmetic(self):
        sbs = [{"runs": 1, "steps_per_second_each": [100.0], "total_steps_per_second": 100.0,
                "torch_threads_each": 8},
               {"runs": 2, "steps_per_second_each": [80.0, 90.0],
                "total_steps_per_second": 170.0, "torch_threads_each": 4}]
        plans = benchmark_ppo.night_plan(sbs, 2048)
        # One at a time: 8 h x 100 steps/s x 0.95 / 5 rounds = 547,200, rounded
        # down to whole rollouts of 2,048 steps. Two at once: 3 rounds at the
        # slower run's 80 steps/s = 729,600 -> 729,088.
        self.assertEqual(plans[0]["steps_per_seed_per_night"], 546_816)
        self.assertEqual((plans[1]["rounds"], plans[1]["steps_per_seed_per_night"]),
                         (3, 729_088))
        self.assertEqual(benchmark_ppo.hours(1_000_000, 100.0), round(1e6 / 100 / 3600 / 0.95, 2))


if __name__ == "__main__":
    unittest.main()
