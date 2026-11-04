"""Tabular Q-learning, its tiny stage and its state encoding (Task 6, stage 4).

Needs numpy and gymnasium (skipped with a clear reason otherwise); torch is
not needed here. Runs write into temporary folders, never into ``runs/``.
"""

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
    import numpy as np
except ImportError:  # pragma: no cover - depends on the machine
    np = None

MISSING = "numpy/gymnasium not installed (pip install -r requirements.txt); RL test skipped"

from game.environment import Action, BombermanEnv, GAME_RULES  # noqa: E402

if np is not None:
    from rl import tabular_env as te
    from rl.config import TabularConfig, load_config
    from rl.evaluate_agent import main as evaluate_main
    from rl.train_tabular import QTable, epsilon_at, main as train_main, q_update, train

A = {a: i for i, a in enumerate(Action)}


@unittest.skipIf(np is None, MISSING)
class EncodingTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(len(te.CELLS), 65)  # 47 floor + 18 crate cells
        self.assertEqual(te.N_STATES, 65 * (65 * 9 + 1) * (65 * 4 + 1))

    def test_round_trip(self):
        cases = [((1, 1), None, None), ((3, 1), (3, 1, 9), None),
                 ((4, 1), None, (3, 1, 4)), ((9, 9), (8, 9, 1), (1, 1, 1)),
                 ((5, 5), (5, 5, 5), (5, 5, 2))]
        seen = set()
        for case in cases:
            index = te.encode_state(*case)
            self.assertEqual(te.decode_state(index), case)
            self.assertTrue(0 <= index < te.N_STATES)
            seen.add(index)
        self.assertEqual(len(seen), len(cases))

    def test_extremes(self):
        self.assertEqual(te.encode_state(te.CELLS[0]), 0)
        last = te.encode_state(te.CELLS[-1], (*te.CELLS[-1], 9), (*te.CELLS[-1], 4))
        self.assertEqual(last, te.N_STATES - 1)

    def test_tracker_follows_bomb_and_flames(self):
        env = BombermanEnv(random_crates=False, **GAME_RULES)
        tracker = te.TabularStateTracker(0)
        self.assertEqual(te.decode_state(tracker.observe(env.reset())),
                         ((1, 1), None, None))
        state = env.step(Action.BOMB)
        self.assertEqual(te.decode_state(tracker.observe(state)), ((1, 1), (1, 1, 9), None))
        for _ in range(3):
            state = env.step(Action.RIGHT)
            tracker.observe(state)
        self.assertEqual(te.decode_state(tracker.observe(state)), ((4, 1), (1, 1, 6), None))
        for _ in range(6):
            state = env.step(Action.WAIT)
            index = tracker.observe(state)
        self.assertEqual(state["steps"], 10)  # exploded on this step
        self.assertEqual(te.decode_state(index), ((4, 1), None, (1, 1, 4)))
        # A new bomb while the old flames burn: both fields at once.
        state = env.step(Action.BOMB)
        self.assertEqual(te.decode_state(tracker.observe(state)),
                         ((4, 1), (4, 1, 9), (1, 1, 3)))
        for _ in range(3):
            state = env.step(Action.WAIT)
        self.assertEqual(te.decode_state(tracker.observe(state)), ((4, 1), (4, 1, 6), None))


@unittest.skipIf(np is None, MISSING)
class TinyStageTest(unittest.TestCase):
    def play(self, actions, cfg=None):
        env = TinyStageEnv(cfg)
        env.reset(seed=0)
        out = None
        for a in actions:
            out = env.step(A[a])
            if out[2] or out[3]:
                break
        return env, out

    def test_reset_is_fixed_map_single_player(self):
        env = te.TinyStageEnv()
        s, info = env.reset(seed=5)
        self.assertEqual(te.decode_state(s), ((1, 1), None, None))
        self.assertEqual(len(env.state["characters"]), 1)
        self.assertEqual(env.state["grid"], BombermanEnv().reset()["grid"])
        self.assertEqual(env.state["rules"]["bomb_fuse"], 9)

    def test_crate_ends_with_plus_one(self):
        # Bomb at (3,1): the blast reaches the crate at (5,1). (3,2) is a wall,
        # so escape back left to (1,1) and down to (1,3), outside the blast.
        plan = [Action.RIGHT, Action.RIGHT, Action.BOMB, Action.LEFT, Action.LEFT,
                Action.DOWN, Action.DOWN] + [Action.WAIT] * 20
        env, (s, r, term, trunc, info) = self.play(plan)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info["result"], "crate")
        self.assertAlmostEqual(r, 1.0 - 0.001)

    def test_death_ends_with_minus_one(self):
        env, (s, r, term, trunc, info) = self.play([Action.BOMB] + [Action.WAIT] * 20)
        self.assertTrue(term)
        self.assertEqual(info["result"], "death")
        self.assertEqual(info["steps"], 10)
        self.assertAlmostEqual(r, -1.0 - 0.001)

    def test_crate_and_death_together_is_only_death(self):
        # Bomb at (3,1) next to the crate and stay: crate and agent burn together.
        env, (s, r, term, trunc, info) = self.play(
            [Action.RIGHT, Action.RIGHT, Action.BOMB] + [Action.WAIT] * 20)
        self.assertTrue(any(e["type"] == "block_destroyed" for e in info["events"]))
        self.assertEqual(info["result"], "death")
        self.assertAlmostEqual(r, -1.001)

    def test_truncated_at_step_limit(self):
        env, (s, r, term, trunc, info) = self.play([Action.WAIT] * 300)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info["steps"], 200)
        self.assertEqual(info["result"], "timeout")
        self.assertAlmostEqual(r, -0.001)

    def test_step_limit_from_config(self):
        env, (_, _, _, trunc, info) = self.play([Action.WAIT] * 20, TabularConfig(max_steps=7))
        self.assertTrue(trunc)
        self.assertEqual(info["steps"], 7)


def TinyStageEnv(cfg=None):  # noqa: N802 - small helper used above
    return te.TinyStageEnv(cfg)


@unittest.skipIf(np is None, MISSING)
class QLearningTest(unittest.TestCase):
    def test_update_bootstrapped(self):
        table = QTable()
        table.row(7)[:] = [0.0, 2.0, 0.5, 0, 0, 0]
        table.row(3)[1] = 1.0
        error = q_update(table, 3, 1, r=0.5, s2=7, terminated=False, alpha=0.1, gamma=0.9)
        # target = 0.5 + 0.9 * 2.0 = 2.3; error = 2.3 - 1.0 = 1.3
        self.assertAlmostEqual(error, 1.3)
        self.assertAlmostEqual(table.get(3)[1], 1.13)

    def test_update_terminal_does_not_bootstrap(self):
        table = QTable()
        table.row(7)[:] = 5.0
        q_update(table, 3, 2, r=-1.0, s2=7, terminated=True, alpha=0.5, gamma=0.99)
        self.assertAlmostEqual(table.get(3)[2], -0.5)

    def test_unvisited_state_is_zero_and_not_stored(self):
        table = QTable()
        self.assertTrue((table.get(42) == 0).all())
        self.assertEqual(len(table), 0)
        q_update(table, 1, 0, r=0.0, s2=42, terminated=False, alpha=0.1, gamma=0.9)
        self.assertEqual(len(table), 1)

    def test_greedy_and_ties(self):
        table = QTable()
        table.row(1)[:] = [0, 0, 3, 0, 0, 0]
        rng = np.random.default_rng(0)
        self.assertEqual(table.greedy(1, rng), 2)
        picks = {table.greedy(99, rng) for _ in range(200)}
        self.assertEqual(picks, set(range(6)))  # all tie on an unseen state

    def test_epsilon_schedule(self):
        cfg = TabularConfig(episodes=1000)
        self.assertEqual(epsilon_at(0, cfg), 1.0)
        self.assertAlmostEqual(epsilon_at(400, cfg), 1.0 - 0.95 * 0.5)
        self.assertAlmostEqual(epsilon_at(800, cfg), 0.05)
        self.assertAlmostEqual(epsilon_at(999, cfg), 0.05)
        self.assertGreater(epsilon_at(799, cfg), 0.05)

    def test_save_and_load(self):
        table = QTable()
        table.row(5)[:] = [1, 2, 3, 4, 5, 6]
        table.row(10**6)[3] = -0.25
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.npz"
            table.save(path)
            loaded = QTable.load(path)
        self.assertEqual(sorted(loaded.values), [5, 10**6])
        np.testing.assert_array_equal(loaded.get(5), table.get(5))
        self.assertEqual(loaded.get(10**6)[3], -0.25)

    def test_training_is_reproducible(self):
        cfg = TabularConfig(episodes=30)
        a, sa = train(cfg, seed=3)
        b, sb = train(cfg, seed=3)
        self.assertEqual(sa, sb)
        self.assertEqual(sorted(a.values), sorted(b.values))
        for s in a.values:
            np.testing.assert_array_equal(a.get(s), b.get(s))


@unittest.skipIf(np is None, MISSING)
class ScriptTest(unittest.TestCase):
    def test_config_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"alpha": 0.5, "max_steps": 50}))
            cfg = load_config(TabularConfig(), path)
        self.assertEqual((cfg.alpha, cfg.max_steps, cfg.gamma), (0.5, 50, 0.99))

    def test_one_episode_run_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            overrides = Path(tmp) / "c.json"
            overrides.write_text(json.dumps({"max_steps": 30}))
            with contextlib.redirect_stdout(io.StringIO()):
                run = train_main(["--episodes", "1", "--seed", "2",
                                  "--config", str(overrides), "--runs-dir", tmp])
            for name in ("config.json", "meta.json", "metrics.csv", "q_table.npz"):
                self.assertTrue((run / name).exists(), name)
            config = json.loads((run / "config.json").read_text())
            self.assertEqual((config["episodes"], config["max_steps"]), (1, 30))
            meta = json.loads((run / "meta.json").read_text())
            self.assertEqual(meta["seed"], 2)
            self.assertIn("commit", meta["git"])
            self.assertIn("numpy", meta["versions"])
            self.assertIsNotNone(meta["finished_at"])
            self.assertEqual(len((run / "metrics.csv").read_text().splitlines()), 2)

            for args in (["--games", "2"],
                         ["--stage", "full", "--opponents", "random", "--games", "2",
                          "--max-steps", "40"]):
                with self.subTest(args=args):
                    with contextlib.redirect_stdout(io.StringIO()):
                        out, results = evaluate_main(["--run", str(run),
                                                      "--runs-dir", tmp, *args])
                    self.assertEqual(results["games"], 2)
                    self.assertTrue((out / "eval.json").exists())
                    self.assertNotEqual(out, run)
            # The evaluated run folder is never changed.
            self.assertEqual(sorted(p.name for p in run.iterdir()),
                             ["config.json", "meta.json", "metrics.csv", "q_table.npz"])

    def test_run_dir_is_never_reused(self):
        from datetime import datetime, timezone

        from rl import run_logging

        class FrozenClock:
            @staticmethod
            def now(tz=None):
                return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_logging, "datetime", FrozenClock):
            first = run_logging.new_run_dir("x", 0, tmp)
            self.assertEqual(first.name, "20260102-030405_x_seed0")
            (first / "keep.txt").write_text("results")
            with self.assertRaises(FileExistsError):
                run_logging.new_run_dir("x", 0, tmp)
            self.assertEqual((first / "keep.txt").read_text(), "results")

if __name__ == "__main__":
    unittest.main()
