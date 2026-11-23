"""The multi-seed PPO runner and its plots (Task 8, part 4).

A tiny plan (2 seeds, a few hundred steps) runs end to end; a crashed run
does not stop the others; ``--continue`` skips finished runs and refuses a
changed plan file; the summary's mean and 95% interval are checked on a
hand-made example; the plot script runs on the tiny output.

Needs numpy, gymnasium, torch and stable-baselines3 (and matplotlib for the
plots), like the other RL tests. Every run is a separate process started
with ``subprocess``, in temporary folders.
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

from rl import priority  # noqa: E402  (standard library only)

try:
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

try:
    import matplotlib  # noqa: F401
except ImportError:  # pragma: no cover - depends on the machine
    matplotlib = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from rl import run_ppo_seeds
    from rl.run_ppo_seeds import Job, load_plan, mean_ci, t_975

TINY = {"n_envs": 1, "n_steps": 64, "batch_size": 32, "n_epochs": 1, "features_dim": 32,
        "conv_channels": [8, 8, 8], "env": {"max_steps": 30}, "total_timesteps": 128,
        "eval_every": 64, "eval_games": 2, "opponents": "random"}


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipIf(torch is None, MISSING)
class StatisticsTest(unittest.TestCase):
    def test_known_t_values(self):
        self.assertEqual(t_975(1), 12.706)
        self.assertEqual(t_975(2), 4.303)
        self.assertEqual(t_975(4), 2.776)  # 5 seeds
        self.assertEqual(t_975(9), 2.262)
        self.assertEqual(t_975(30), 2.042)
        self.assertEqual(t_975(120), 2.042)  # above the table: the df=30 value
        with self.assertRaises(ValueError):
            t_975(0)

    def test_mean_ci_by_hand(self):
        # 5 seeds 0.1 .. 0.5: mean 0.3, sample sd 0.158114, half width
        # 2.776 * 0.158114 / sqrt(5) = 0.196293.
        self.assertEqual(mean_ci([0.1, 0.2, 0.3, 0.4, 0.5]), (0.3, 0.1037, 0.4963))
        # 2 seeds 10 and 20: mean 15, sd 7.0711, half 12.706 * 7.0711 / sqrt(2) = 63.53.
        self.assertEqual(mean_ci([10, 20]), (15.0, -48.53, 78.53))
        self.assertEqual(mean_ci([0.25]), (0.25, "", ""))
        self.assertEqual(mean_ci(["", 1.0, ""]), (1.0, "", ""))
        self.assertEqual(mean_ci([]), ("", "", ""))


@unittest.skipIf(torch is None, MISSING)
class PlanFileTest(unittest.TestCase):
    def test_comments_nulls_and_bad_plans(self):
        plan = load_plan(json.dumps({"_x": 1, "seeds": [3, 4],
                                     "config": {**TINY, "_note": "a comment"}}))
        self.assertEqual(plan["seeds"], [3, 4])
        self.assertEqual(list(plan["variants"]), ["v0"])
        self.assertEqual(plan["variants"]["v0"]["total_timesteps"], 128)
        self.assertEqual(plan["evaluation"], {"games": 200, "stochastic": False})
        with self.assertRaisesRegex(ValueError, "total_timesteps"):
            load_plan(json.dumps({"seeds": [0], "config": {"total_timesteps": None}}))
        for bad in ({"seeds": []}, {"seeds": [1, 1]}, {"seeds": [0], "parallel": 0},
                    {"seeds": [0], "extra": 1}, {"seeds": [0], "variants": {"a b": {}}},
                    {"seeds": [0], "evaluation": {"level": "rule"}}):
            with self.assertRaises(ValueError, msg=bad):
                load_plan(json.dumps(bad))
        with self.assertRaises(KeyError):
            load_plan(json.dumps({"seeds": [0], "config": {"n_env": 2}}))

    def test_first_runs_plan(self):
        text = (ROOT / "experiments" / "ppo" / "first_runs.json").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "total_timesteps.*eval_every"):
            load_plan(text)
        data = json.loads(text)
        data["config"].update(total_timesteps=2_000_000, eval_every=100_000)
        plan = load_plan(json.dumps(data))
        self.assertEqual(plan["seeds"], [0, 1, 2, 3, 4])
        v0 = plan["variants"]["v0"]
        self.assertEqual((v0["opponents"], v0["n_envs"], v0["eval_games"],
                          v0["checkpoint_every"], v0["high_priority"]),
                         ("random", 8, 50, 1_000_000, True))
        self.assertEqual(plan["evaluation"], {"games": 200, "stochastic": False})

    def test_a_job_with_finished_training_only_evaluates(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = load_plan(json.dumps({"seeds": [7], "config": TINY}))
            job = Job(Path(tmp), plan, "v0", 7)
            run = job.dir / "20260101-000000-000000_ppo_seed7"
            run.mkdir(parents=True)
            (run / "model.zip").write_text("m")
            (run / "meta.json").write_text(json.dumps({"finished_at": "x"}))
            crashed = job.dir / "20260101-000001-000000_ppo_seed7"
            crashed.mkdir()
            (crashed / "meta.json").write_text(json.dumps({"finished_at": None}))
            job.refresh()
            self.assertEqual(job.run_dir, run)
            command = job.next_command()
        self.assertEqual(command[1:3], ["-m", "rl.eval_protocol"])
        self.assertIn(str(run), command)


class KeepAwakeTest(unittest.TestCase):
    def test_no_op_off_windows(self):
        with mock.patch.object(priority.sys, "platform", "linux"):
            with priority.keep_awake() as message:
                self.assertIn("nothing changed", message)

    def test_windows_request_is_released_also_on_a_crash(self):
        kernel32 = mock.Mock()
        kernel32.SetThreadExecutionState.return_value = 1
        with mock.patch.object(priority.sys, "platform", "win32"), \
                mock.patch.object(priority, "_kernel32", return_value=kernel32):
            with self.assertRaises(RuntimeError):
                with priority.keep_awake():
                    raise RuntimeError("boom")
        self.assertEqual([c.args for c in kernel32.SetThreadExecutionState.call_args_list],
                         [(0x80000001,), (0x80000000,)])


@unittest.skipIf(torch is None, MISSING)
class RunnerTest(unittest.TestCase):
    """One tiny plan, run once in ``setUpClass``; the tests read its folder."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.plan_file = cls.tmp / "plan.json"
        cls.plan_file.write_text(json.dumps({
            "_about": "tiny test plan", "name": "tiny", "seeds": [0, 1], "parallel": 2,
            "config": TINY,
            # "broken" crashes when its envs are built (an unknown ladder level).
            "variants": {"v0": {}, "broken": {"opponents": "no-such-level"}},
            "evaluation": {"games": 1}}))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.plan_dir = run_ppo_seeds.main([str(cls.plan_file), "--runs-dir",
                                               str(cls.tmp / "runs"), "--quiet"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_finished_and_crashed_runs(self):
        status = {(r["variant"], r["seed"]): r for r in read_csv(self.plan_dir / "status.csv")}
        self.assertEqual(status[("v0", "0")]["status"], "done")
        self.assertEqual(status[("v0", "1")]["status"], "done")
        for seed in ("0", "1"):
            self.assertTrue(status[("broken", seed)]["status"].startswith("crashed (train"))
        log = self.plan_dir / status[("broken", "0")]["note"]
        self.assertIn("no-such-level", log.read_text(encoding="utf-8"))
        for seed in (0, 1):
            job_dir = self.plan_dir / "v0" / f"seed{seed}"
            runs = list(job_dir.glob(f"*_ppo_seed{seed}"))
            evals = list(job_dir.glob("*_eval-protocol_seed1000000"))
            self.assertEqual((len(runs), len(evals)), (1, 1))
            self.assertTrue((runs[0] / "eval_curve.csv").exists())
            protocol = read_csv(evals[0] / "protocol.csv")
            self.assertEqual(len(protocol), 6)

    def test_summary_and_aggregate(self):
        summary = read_csv(self.plan_dir / "summary.csv")
        self.assertEqual(len(summary), 2 * 6)  # 2 seeds x (5 levels + total)
        self.assertEqual({r["variant"] for r in summary}, {"v0"})
        aggregate = read_csv(self.plan_dir / "aggregate.csv")
        self.assertEqual([r["level"] for r in aggregate],
                         ["random", "noisy50", "noisy25", "noisy10", "rule", "total"])
        for entry in aggregate:
            members = [r for r in summary if r["level"] == entry["level"]]
            for metric in ("win_rate", "loss_rate", "mean_length", "death_own_fresh_rate"):
                mean, low, high = mean_ci([float(r[metric]) for r in members])
                self.assertEqual(float(entry[f"{metric}_mean"]), mean)
                self.assertEqual(entry[f"{metric}_ci_low"], str(low))
                self.assertEqual(entry[f"{metric}_ci_high"], str(high))
        saved = json.loads((self.plan_dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["plan"]["variants"]["v0"]["eval_every"], 64)

    def test_continue_skips_finished_runs(self):
        before = sorted(p.relative_to(self.plan_dir).as_posix()
                        for p in (self.plan_dir / "v0").rglob("*"))
        stamps = {p: p.stat().st_mtime_ns for p in (self.plan_dir / "v0").rglob("*")}
        with contextlib.redirect_stdout(io.StringIO()):
            run_ppo_seeds.main(["--continue", str(self.plan_dir), "--quiet"])
        after = sorted(p.relative_to(self.plan_dir).as_posix()
                       for p in (self.plan_dir / "v0").rglob("*"))
        self.assertEqual(before, after)
        self.assertEqual({p: p.stat().st_mtime_ns for p in stamps}, stamps)
        # The crashed runs were tried again, with new log files.
        logs = sorted(p.name for p in (self.plan_dir / "logs").iterdir())
        self.assertIn("broken_seed0_train_2.log", logs)
        self.assertIn("broken_seed0_train_1.log", logs)

    def test_continue_refuses_a_changed_plan_file(self):
        original = self.plan_file.read_text(encoding="utf-8")
        self.addCleanup(self.plan_file.write_text, original)
        self.plan_file.write_text(original.replace('"games": 1', '"games": 2'))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_ppo_seeds.main(["--continue", str(self.plan_dir), "--quiet"])

    @unittest.skipIf(matplotlib is None, "matplotlib not installed")
    def test_plots_from_the_plan_and_from_one_run(self):
        from rl.plot_ppo import find_curves, main as plot_main
        out = self.tmp / "plots"
        with contextlib.redirect_stdout(io.StringIO()):
            paths = plot_main([str(self.plan_dir), "--out", str(out)])
        self.assertEqual([p.name for p in paths], ["learning_curves.png", "death_causes.png"])
        self.assertTrue(all(p.stat().st_size > 0 for p in paths))
        groups = find_curves(self.plan_dir)
        self.assertEqual(list(groups), ["v0"])
        self.assertEqual(len(groups["v0"]), 2)
        run = next((self.plan_dir / "v0" / "seed0").glob("*_ppo_seed0"))
        with contextlib.redirect_stdout(io.StringIO()):
            single = plot_main([str(run), "--out", str(self.tmp / "one")])
        self.assertEqual(len(single), 2)


if __name__ == "__main__":
    unittest.main()
