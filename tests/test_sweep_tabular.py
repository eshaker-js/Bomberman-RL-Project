"""The tabular sweep, its training hook and its plots.

Needs numpy and gymnasium (skipped with a clear reason otherwise); the plot
test also needs matplotlib. Sweeps write into temporary folders, never into
``runs/``.
"""

import contextlib
import csv
import dataclasses
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
except ImportError:  # pragma: no cover - depends on the machine
    np = None

try:
    import matplotlib  # noqa: F401
except ImportError:  # pragma: no cover - depends on the machine
    matplotlib = None

MISSING = "numpy/gymnasium not installed (pip install -r requirements.txt); RL test skipped"
NO_PLOTS = "matplotlib not installed (pip install matplotlib); plot test skipped"

if np is not None:
    from rl import sweep_tabular as sw
    from rl.config import TabularConfig
    from rl.train_tabular import train

SMALL = {"episodes": 40, "max_steps": 60}


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def run_small_sweep(root: Path) -> Path:
    """Two alpha values x two seeds on a 40-episode stage, in ``root``."""
    plan = root / "plan.json"
    plan.write_text(json.dumps({"alpha": [0.1, 0.5]}), encoding="utf-8")
    config = root / "base.json"
    config.write_text(json.dumps(SMALL), encoding="utf-8")
    return quiet(sw.main, ["--plan", str(plan), "--config", str(config),
                           "--seeds", "2", "--eval-every", "20",
                           "--eval-games", "3", "--runs-dir", str(root / "runs")])


@unittest.skipIf(np is None, MISSING)
class TrainingHookTest(unittest.TestCase):
    def test_hook_does_not_change_training(self):
        cfg = TabularConfig(episodes=150)
        plain, plain_summary = train(cfg, 3)
        seen = []

        def hook(episode, table):
            seen.append(episode)
            sw.greedy_result(table, cfg, 1, 3)  # reads the table, plays a game

        hooked, hooked_summary = train(cfg, 3, on_episode_end=hook)
        self.assertEqual(seen, list(range(150)))
        self.assertEqual(plain_summary, hooked_summary)
        self.assertEqual(sorted(plain.values), sorted(hooked.values))
        for state, row in plain.values.items():
            np.testing.assert_array_equal(row, hooked.values[state])


@unittest.skipIf(np is None, MISSING)
class LearnedAtTest(unittest.TestCase):
    def test_cases(self):
        self.assertIsNone(sw.learned_at([]))
        self.assertIsNone(sw.learned_at([(50, False), (100, False)]))
        self.assertEqual(sw.learned_at([(50, True), (100, True)]), 50)
        self.assertEqual(sw.learned_at([(50, False), (100, True), (150, True)]), 100)
        # solved, forgot, solved again: only the last streak counts
        self.assertEqual(sw.learned_at([(50, True), (100, False), (150, True),
                                        (200, True)]), 150)
        self.assertIsNone(sw.learned_at([(50, True), (100, True), (150, False)]))


@unittest.skipIf(np is None, MISSING)
class ShortestSolutionTest(unittest.TestCase):
    def test_v0_tiny_stage_needs_seven_steps(self):
        self.assertEqual(sw.shortest_solution(TabularConfig()), 7)

    def test_no_solution_within_a_short_limit(self):
        self.assertIsNone(sw.shortest_solution(TabularConfig(max_steps=6)))

    def test_seven_steps_really_work(self):
        """Replays one 7-step game: 2 moves, a bomb, 4 ticks of fuse."""
        from game.environment import Action
        from rl.tabular_env import TinyStageEnv

        env = TinyStageEnv(TabularConfig())
        env.reset()
        names = ("DOWN", "DOWN", "BOMB", "UP", "UP", "WAIT", "RIGHT")
        actions = [list(Action).index(Action[name]) for name in names]
        for action in actions:
            *_, info = env.step(action)
        self.assertEqual(info["result"], "crate")
        self.assertEqual(info["steps"], 7)


@unittest.skipIf(np is None, MISSING)
class PlanTest(unittest.TestCase):
    def test_default_plan_shares_the_base_runs(self):
        variants = sw.expand_plan(TabularConfig(), sw.SWEEP_PLAN)
        values = sum(len(v) for v in sw.SWEEP_PLAN.values())
        self.assertEqual(len(variants), values - len(sw.SWEEP_PLAN) + 1)  # 13
        base = variants[0]
        self.assertEqual(base.name, sw.BASE_NAME)
        self.assertEqual(base.config, TabularConfig())
        self.assertEqual({p for p, _ in base.groups}, set(sw.SWEEP_PLAN))
        self.assertEqual(len({v.name for v in variants}), len(variants))

    def test_each_variant_changes_exactly_one_value(self):
        base = TabularConfig()
        for variant in sw.expand_plan(base, sw.SWEEP_PLAN)[1:]:
            (param, value), = variant.groups
            self.assertEqual(dataclasses.replace(base, **{param: value}),
                             variant.config)
            self.assertNotEqual(getattr(variant.config, param), getattr(base, param))

    def test_bad_plans_are_rejected(self):
        with self.assertRaises(KeyError):
            sw.expand_plan(TabularConfig(), {"alhpa": [0.5]})
        with self.assertRaises(TypeError):
            sw.expand_plan(TabularConfig(), {"episodes": [0.5]})

    def test_int_values_count_as_floats(self):
        variants = sw.expand_plan(TabularConfig(), {"alpha": [1]})
        self.assertEqual(variants[1].config.alpha, 1.0)
        self.assertIsInstance(variants[1].config.alpha, float)


@unittest.skipIf(np is None, MISSING)
class SweepRunTest(unittest.TestCase):
    def test_writes_every_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            sweep = run_small_sweep(Path(tmp))
            summary = read_csv(sweep / "summary.csv")
            self.assertEqual([(r["variant"], r["seed"]) for r in summary],
                             [("base", "0"), ("base", "1"),
                              ("alpha=0.5", "0"), ("alpha=0.5", "1")])
            checkpoints = read_csv(sweep / "checkpoints.csv")
            self.assertEqual(len(checkpoints), 4 * 2)  # after episodes 20 and 40
            self.assertEqual({c["episodes"] for c in checkpoints}, {"20", "40"})
            for row in summary:
                run = sweep / row["run_dir"]
                for name in ("config.json", "meta.json", "metrics.csv", "q_table.npz"):
                    self.assertTrue((run / name).exists(), run / name)
                meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
                self.assertIsNotNone(meta["finished_at"])
                self.assertEqual(len(read_csv(run / "metrics.csv")), SMALL["episodes"])
            plan = json.loads((sweep / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["seeds"], [0, 1])
            self.assertEqual(plan["base_config"]["episodes"], SMALL["episodes"])
            self.assertIsNotNone(plan["finished_at"])
            self.assertEqual({v["shortest_game"] for v in plan["variants"]}, {7})
            for row in summary:
                optimal = (row["eval_crate_rate"] == "1.0"
                           and float(row["eval_steps"]) == 7.0)
                self.assertEqual(row["optimal_route"], str(int(optimal)))
            aggregate = read_csv(sweep / "aggregate.csv")
            self.assertEqual([(a["param"], a["value"], a["seeds"]) for a in aggregate],
                             [("alpha", "0.1", "2"), ("alpha", "0.5", "2")])
            for a in aggregate:
                self.assertLessEqual(int(a["optimal"]), int(a["solved"]))

    def test_same_seeds_same_results(self):
        ignore = {"seconds", "run_dir"}
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                sweep = run_small_sweep(Path(tmp))
                summary = [{k: v for k, v in r.items() if k not in ignore}
                           for r in read_csv(sweep / "summary.csv")]
                results.append((summary, read_csv(sweep / "checkpoints.csv")))
        self.assertEqual(results[0], results[1])

    def test_matches_a_plain_training_run(self):
        """A sweep run trains exactly what rl.train_tabular would train."""
        with tempfile.TemporaryDirectory() as tmp:
            sweep = run_small_sweep(Path(tmp))
            row = read_csv(sweep / "summary.csv")[0]
        _, summary = train(TabularConfig(**SMALL), int(row["seed"]))
        self.assertEqual(int(row["train_crate"]), summary["crate"])
        self.assertEqual(int(row["train_death"]), summary["death"])
        self.assertEqual(int(row["states_visited"]), summary["states_visited"])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            sw.run_sweep(TabularConfig(**SMALL), {"alpha": [0.1]}, [0], Path("unused"),
                         eval_every=0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sw.main(["--seeds", "0"])


@unittest.skipIf(np is None, MISSING)
@unittest.skipIf(matplotlib is None, NO_PLOTS)
class PlotTest(unittest.TestCase):
    def test_one_figure_per_parameter(self):
        from rl import plot_tabular

        with tempfile.TemporaryDirectory() as tmp:
            sweep = run_small_sweep(Path(tmp))
            written = quiet(plot_tabular.main, [str(sweep), "--window", "10"])
            self.assertEqual([p.name for p in written], ["alpha.png"])
            self.assertGreater(written[0].stat().st_size, 10_000)

    def test_moving_success(self):
        from rl.plot_tabular import moving_success

        got = moving_success(["crate", "death", "crate", "crate"], 2)
        np.testing.assert_allclose(got, [1.0, 0.5, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
