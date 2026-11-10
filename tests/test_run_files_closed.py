"""Training and evaluation scripts close every file they open (Task 7, stage 6).

On Windows an open file cannot be deleted, so a run that leaves its log file
open breaks ``TemporaryDirectory`` cleanup (WinError 32). Linux allows the
delete, so these tests check directly: every file opened under the run
folder during a script is closed when the script returns, also when the run
fails part-way. Each opened file is recorded and kept referenced, so garbage
collection cannot close it behind the test's back.

Needs numpy, gymnasium, torch and stable-baselines3, like the other RL tests.
"""

import builtins
import contextlib
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

if torch is not None:
    from rl import train_ppo, train_tabular
    from rl.evaluate_agent import main as evaluate_main

TINY_PPO = {"n_envs": 1, "n_steps": 64, "batch_size": 32, "n_epochs": 1,
            "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 50}}


@contextlib.contextmanager
def record_opened(folder: str):
    """Record every file object opened under ``folder`` while the block runs."""
    real_open = builtins.open
    opened = []

    def recording_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if isinstance(file, (str, Path)) and str(Path(file).resolve()).startswith(folder):
            opened.append(handle)
        return handle

    with mock.patch.object(builtins, "open", recording_open):
        yield opened


def close_all(handles):
    for handle in handles:
        handle.close()


@unittest.skipIf(torch is None, MISSING)
class FilesClosedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = str(Path(self._tmp.name).resolve())
        self.addCleanup(self._tmp.cleanup)
        self.overrides = Path(self.tmp) / "c.json"
        self.overrides.write_text(json.dumps(TINY_PPO))

    def assert_all_closed(self, opened):
        self.addCleanup(close_all, opened)
        names = [Path(h.name).name for h in opened]
        self.assertTrue(opened)
        still_open = [Path(h.name).name for h in opened if not h.closed]
        self.assertEqual(still_open, [], f"files left open (opened: {names})")

    def ppo_run(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return train_ppo.main(["--timesteps", "64", "--seed", "0",
                                   "--config", str(self.overrides), "--runs-dir", self.tmp])

    def test_ppo_run_closes_progress_csv(self):
        with record_opened(self.tmp) as opened:
            run = self.ppo_run()
        self.assertIn("progress.csv", [Path(h.name).name for h in opened])
        self.assert_all_closed(opened)
        self.assertTrue((run / "progress.csv").read_text().strip())

    def test_ppo_run_closes_files_when_training_fails(self):
        with record_opened(self.tmp) as opened:
            with mock.patch.object(train_ppo.PPO, "save", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    self.ppo_run()
        self.assert_all_closed(opened)

    def test_ppo_run_closes_its_envs(self):
        closed = []
        real_make_model = train_ppo.make_model

        def spy(cfg, seed):
            model = real_make_model(cfg, seed)
            real_close = model.get_env().close
            model.get_env().close = lambda: (closed.append(True), real_close())
            return model

        with mock.patch.object(train_ppo, "make_model", spy):
            self.ppo_run()
        self.assertEqual(closed, [True])

    def test_tabular_run_closes_metrics_csv_also_on_failure(self):
        args = ["--episodes", "20", "--seed", "0", "--runs-dir", self.tmp]
        failing = ["--episodes", "20", "--seed", "0", "--runs-dir", str(Path(self.tmp) / "b")]
        with record_opened(self.tmp) as opened:
            with contextlib.redirect_stdout(io.StringIO()):
                train_tabular.main(args)
        self.assertIn("metrics.csv", [Path(h.name).name for h in opened])
        self.assert_all_closed(opened)

        with record_opened(self.tmp) as opened:
            with mock.patch.object(train_tabular.QTable, "save",
                                   side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError), \
                        contextlib.redirect_stdout(io.StringIO()):
                    train_tabular.main(failing)
        self.assertIn("metrics.csv", [Path(h.name).name for h in opened])
        self.assert_all_closed(opened)

    def test_evaluation_closes_its_files(self):
        run = self.ppo_run()
        with record_opened(self.tmp) as opened:
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_main(["--run", str(run), "--runs-dir", self.tmp,
                               "--games", "1", "--max-steps", "10"])
        self.assert_all_closed(opened)


if __name__ == "__main__":
    unittest.main()
