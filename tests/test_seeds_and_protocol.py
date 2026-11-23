"""Run folders, seed ranges, the evaluation protocol and its new metrics (Task 8, part 1).

- ``new_run_dir`` never fails or reuses a folder under rapid creation
- training never plays a seed of an evaluation range (``rl.seeds``)
- ``GameTally``: game length by outcome, buffs and the cause of each agent death, on
  scripted games of the engine (fixed map, real game rules; a few cells and
  positions are set directly on the env's public attributes, like the other
  test files do)
- ``rl.eval_protocol`` on a tiny saved model against all 5 levels
- ``rl.evaluate_agent``'s old results and command-line defaults are unchanged

The RL parts need numpy, gymnasium, torch and stable-baselines3, like the
other RL tests, and are skipped with a clear reason without them.
"""

import contextlib
import csv
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.environment import GAME_RULES, Action, BombermanEnv, Cell  # noqa: E402
from rl.run_logging import new_run_dir  # noqa: E402

try:
    import gymnasium  # noqa: F401
    import numpy as np
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from gymnasium.utils import seeding

    from agents.noisy_bot import LADDER
    from game.environment import AGENT_ID
    from rl import evaluate_agent
    from rl.bomberman_env import BombermanGymEnv
    from rl.eval_protocol import main as protocol_main
    from rl.eval_protocol import run_protocol
    from rl.evaluate_agent import (
        DEATH_CAUSES,
        NEW_FULL_KEYS,
        GameTally,
        _evaluate_full,
        classify_death,
    )
    from rl.seeds import (
        EVAL_SEEDS,
        TRAIN_EVAL_SEEDS,
        TrainingSeedGuard,
        check_training_seed,
        eval_seeds,
        is_reserved,
        train_eval_seeds,
    )
    from rl.train_ppo import main as train_main

TINY_PPO = {"n_envs": 1, "n_steps": 64, "batch_size": 32, "n_epochs": 1,
            "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 50}}

U, D, L, R = Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
W, B = Action.WAIT, Action.BOMB


class RunDirTest(unittest.TestCase):
    def test_rapid_creation_gives_unique_sorted_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = [new_run_dir("ppo", 0, tmp) for _ in range(200)]
            self.assertEqual(len({p.name for p in made}), 200)
            self.assertEqual([p.name for p in made], sorted(p.name for p in made))
            for path in made:
                self.assertTrue(path.is_dir())
                self.assertRegex(path.name, r"^\d{8}-\d{6}-\d{6}_ppo_seed0$")

    def test_threads_at_once_get_unique_folders(self):
        made, lock = [], threading.Lock()
        with tempfile.TemporaryDirectory() as tmp:
            def create():
                for _ in range(25):
                    path = new_run_dir("eval-ppo-full", 3, tmp)
                    with lock:
                        made.append(path.name)

            threads = [threading.Thread(target=create) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(set(made)), 200)
            self.assertEqual(len(list(Path(tmp).iterdir())), 200)

    def test_an_existing_folder_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = new_run_dir("ppo", 1, tmp)
            (first / "model.zip").write_text("x")
            with mock.patch("rl.run_logging.datetime") as clock:
                # A frozen clock: every call returns the first folder's time.
                from datetime import datetime, timezone
                stamp = datetime.strptime(first.name.split("_")[0] + "+0000",
                                          "%Y%m%d-%H%M%S-%f%z").astimezone(timezone.utc)
                clock.now.return_value = stamp
                second = new_run_dir("ppo", 1, tmp)
            self.assertNotEqual(first, second)
            self.assertGreater(second.name, first.name)
            self.assertEqual((first / "model.zip").read_text(), "x")


@unittest.skipIf(torch is None, MISSING)
class SeedRangeTest(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual((EVAL_SEEDS.start, EVAL_SEEDS.stop), (1_000_000, 2_000_000))
        self.assertEqual((TRAIN_EVAL_SEEDS.start, TRAIN_EVAL_SEEDS.stop),
                         (2_000_000, 3_000_000))
        self.assertEqual(list(eval_seeds(3)), [1_000_000, 1_000_001, 1_000_002])
        self.assertEqual(eval_seeds(200, 200).start, 1_000_200)
        self.assertEqual(train_eval_seeds(50), range(2_000_000, 2_000_050))
        with self.assertRaises(ValueError):
            eval_seeds(10, EVAL_SEEDS.stop - EVAL_SEEDS.start - 5)
        for seed in (999_999, 3_000_000, 0, 2**31 - 2):
            self.assertFalse(is_reserved(seed))
        for seed in (1_000_000, 1_999_999, 2_000_000, 2_999_999):
            self.assertTrue(is_reserved(seed))

    def test_run_seeds_reaching_a_reserved_range_are_refused(self):
        for seed in (0, 4, 999_992, 3_000_000):
            check_training_seed(seed, 8)
        for seed in (999_993, 1_500_000, 2_999_999):
            with self.assertRaises(ValueError):
                check_training_seed(seed, 8)
        env = TrainingSeedGuard(BombermanGymEnv("random"))
        try:
            with self.assertRaises(ValueError):
                env.reset(seed=1_000_000)
        finally:
            env.close()

    def test_a_drawn_reserved_seed_is_skipped(self):
        """Env seed 5 (rank 5 of a seed-0 run): its 17th drawn seed is reserved.

        Found by scanning numpy's PCG64 stream, which is specified bit for
        bit, so this holds on every machine.
        """
        rng, _ = seeding.np_random(5)
        drawn = [int(rng.integers(0, 2**31 - 1)) for _ in range(20)]
        self.assertEqual(drawn[16], 2_180_441)
        self.assertTrue(is_reserved(drawn[16]))
        self.assertFalse(any(is_reserved(s) for s in drawn[:16] + drawn[17:]))

        plain = BombermanGymEnv("random")
        guarded = TrainingSeedGuard(BombermanGymEnv("random"))
        try:
            seen_plain = [plain.reset(seed=5)[1]["episode_seed"]]
            seen_plain += [plain.reset()[1]["episode_seed"] for _ in range(19)]
            seen = [guarded.reset(seed=5)[1]["episode_seed"]]
            seen += [guarded.reset()[1]["episode_seed"] for _ in range(18)]
        finally:
            plain.close()
            guarded.close()
        self.assertEqual(seen_plain, [5, *drawn[:19]])  # unguarded: plays 2,180,441
        self.assertEqual(seen, [5, *drawn[:16], *drawn[17:19]])
        self.assertEqual(guarded.redraws, 1)

    def test_redraw_loop_on_a_scripted_env(self):
        class Scripted(BombermanGymEnv):
            script = [7, 1_000_000, 2_500_000, 42, 2_000_001, 9]

            def reset(self, *, seed=None, options=None):
                obs, info = super().reset(seed=seed, options=options)
                if seed is None:
                    info["episode_seed"] = self.script.pop(0)
                return obs, info

        env = TrainingSeedGuard(Scripted("random"))
        try:
            got = [env.reset()[1]["episode_seed"] for _ in range(3)]
        finally:
            env.close()
        self.assertEqual(got, [7, 42, 9])
        self.assertEqual(env.redraws, 3)

    def test_training_envs_never_draw_a_reserved_seed(self):
        """Every env of several training seeds, 400 episodes each."""
        for run_seed in (0, 1, 2, 3, 4, 17):
            for rank in range(8):
                env = TrainingSeedGuard(BombermanGymEnv(None))
                try:
                    seeds = [env.reset(seed=run_seed + rank)[1]["episode_seed"]]
                    seeds += [env.reset()[1]["episode_seed"] for _ in range(400)]
                finally:
                    env.close()
                self.assertEqual(seeds[0], run_seed + rank)
                self.assertFalse([s for s in seeds if is_reserved(s)],
                                 (run_seed, rank))

    def test_a_real_training_run_plays_no_reserved_seed(self):
        from stable_baselines3.common.callbacks import BaseCallback

        from rl import train_ppo
        from rl.config import PPOConfig, apply_overrides
        played = []

        class Record(BaseCallback):
            def _on_step(self):
                for rank, (done, info) in enumerate(zip(self.locals["dones"],
                                                        self.locals["infos"])):
                    if done:
                        played.append((rank, info["episode_seed"]))
                return True

        cfg = apply_overrides(PPOConfig(), {**TINY_PPO, "n_envs": 4,
                                            "env": {"max_steps": 20}})
        model = train_ppo.make_model(cfg, 3)
        try:
            model.learn(total_timesteps=512, callback=Record())
        finally:
            model.get_env().close()
        self.assertTrue(played)
        first = {}
        for rank, seed in played:
            first.setdefault(rank, seed)
        self.assertEqual(first, {0: 3, 1: 4, 2: 5, 3: 6})  # env rank r starts on 3 + r
        self.assertFalse([s for _, s in played if is_reserved(s)])


def scripted_engine(cleared=()) -> BombermanEnv:
    """The fixed map with the real game rules; ``cleared`` cells become floor."""
    env = BombermanEnv(with_npcs=True, random_crates=False, **GAME_RULES)
    env.reset(seed=0)
    for x, y in cleared:
        env.grid[y][x] = Cell.EMPTY
    return env


def place(env: BombermanEnv, character: int, x: int, y: int) -> None:
    env.characters[character].x = x
    env.characters[character].y = y


def play(env: BombermanEnv, agent_actions, npc_actions=None) -> "GameTally":
    """Play the agent's actions (NPC 1 follows ``npc_actions``) and tally them."""
    tally = GameTally(0)
    state = env.get_state()
    for i, action in enumerate(agent_actions):
        npc = {1: npc_actions[i]} if npc_actions and i < len(npc_actions) else None
        new = env.step(action, npc)
        tally.observe(state, new)
        state = new
        if new["outcome"] is not None:
            break
    return tally


# The agent's bomb on (1, 3) burns (2, 3) and (3, 3) to the right; NPC 1's
# bomb on (3, 1) burns (3, 2) and (3, 3) downwards. Both are placed on
# action 1 and explode together on action 5, so (3, 3) is burned by both,
# and (4, 3) is safe from both (3 tiles from the agent's bomb).
ROW_3 = ((2, 3), (4, 3))


@unittest.skipIf(torch is None, MISSING)
class DeathCauseTest(unittest.TestCase):
    def assert_death(self, tally, cause, step):
        self.assertEqual(tally.death_cause, cause)
        self.assertEqual(tally.died_at, step)
        self.assertEqual(tally.finish("loss", step)["steps"], step)

    def own_setup(self):
        env = scripted_engine(ROW_3)
        place(env, 0, 1, 3)
        return env

    def both_bomb_setup(self):
        env = self.own_setup()
        place(env, 1, 3, 1)
        return env

    def test_own_fresh(self):
        tally = play(self.own_setup(), [B, W, W, W, W])
        self.assert_death(tally, "own_fresh", 5)
        self.assertEqual(tally.self_kill, 1)

    def test_own_older(self):
        tally = play(self.own_setup(), [B, R, R, R, W, L])
        self.assert_death(tally, "own_older", 6)
        self.assertEqual(tally.self_kill, 1)

    def test_opponent_fresh(self):
        env = scripted_engine(ROW_3)
        place(env, 0, 3, 3)
        place(env, 1, 3, 1)
        tally = play(env, [W] * 5, [B])
        self.assert_death(tally, "opponent_fresh", 5)
        self.assertEqual(tally.self_kill, 0)

    def test_opponent_older(self):
        env = scripted_engine(ROW_3)
        place(env, 0, 4, 3)
        place(env, 1, 3, 1)
        tally = play(env, [W, W, W, W, W, L], [B])
        self.assert_death(tally, "opponent_older", 6)

    def test_shared_fresh(self):
        tally = play(self.both_bomb_setup(), [B, R, R, W, W], [B])
        self.assert_death(tally, "shared_fresh", 5)
        self.assertEqual(tally.self_kill, 1)

    def test_shared_older(self):
        tally = play(self.both_bomb_setup(), [B, R, R, R, W, L], [B])
        self.assert_death(tally, "shared_older", 6)

    def test_no_death_no_cause(self):
        env = scripted_engine()
        tally = play(env, [W] * 10)
        row = tally.finish(None, 10)
        self.assertEqual((row["outcome"], row["steps"], row["death_cause"]),
                         ("timeout", 10, ""))
        self.assertIsNone(tally.died_at)
        self.assertNotIn("survival", row)

    def test_classify_on_hand_built_events(self):
        state = {"characters": [{"x": 2, "y": 2}], "explosion": [(2, 2)]}
        older = {"characters": [{"x": 2, "y": 2}], "explosion": [(5, 5)]}
        self.assertEqual(classify_death({"killed_by": (0,)}, state), "own_fresh")
        self.assertEqual(classify_death({"killed_by": (1, 3)}, state), "opponent_fresh")
        self.assertEqual(classify_death({"killed_by": (0, 2)}, state), "shared_fresh")
        self.assertEqual(classify_death({"killed_by": (0,)}, older), "own_older")
        self.assertEqual(classify_death({"killed_by": (2,)}, older), "opponent_older")
        self.assertEqual(classify_death({"killed_by": (0, 1, 2)}, older), "shared_older")
        self.assertEqual(classify_death({"killed_by": ()}, state), "unknown")


@unittest.skipIf(torch is None, MISSING)
class BuffCountTest(unittest.TestCase):
    def test_picked_and_raised_per_type(self):
        env = BombermanEnv(with_npcs=True, random_crates=False, **GAME_RULES)
        env.reset(seed=0, start_buffs={0: {"range": 5}})  # range already at its max
        env.floor_buffs[(2, 1)] = "range"
        env.floor_buffs[(3, 1)] = "burn"
        env.floor_buffs[(4, 1)] = "bomb_count"
        tally = play(env, [R, R, R])
        row = tally.finish(None, 3)
        self.assertEqual((row["buffs_picked"], row["buffs_raised"]), (3, 2))
        self.assertEqual((row["buffs_picked_range"], row["buffs_raised_range"]), (1, 0))
        self.assertEqual((row["buffs_picked_burn"], row["buffs_raised_burn"]), (1, 1))
        self.assertEqual((row["buffs_picked_bomb_count"], row["buffs_raised_bomb_count"]),
                         (1, 1))

    def test_an_opponents_pickup_is_not_counted(self):
        env = scripted_engine()
        place(env, 1, 2, 1)
        env.floor_buffs[(3, 1)] = "burn"
        tally = play(env, [W], [R])
        self.assertEqual(tally.finish(None, 1)["buffs_picked"], 0)


def old_evaluate_full(env, policy, games, first_seed):
    """``rl.evaluate_agent._evaluate_full`` exactly as before Task 8 (fe0bd9c)."""
    totals = {"games": games, "win": 0, "loss": 0, "draw": 0, "timeout": 0,
              "self_kills": 0, "kills": 0, "crates": 0, "steps": 0}
    for seed in range(first_seed, first_seed + games):
        obs, _ = env.reset(seed=seed)
        policy.reset()
        done, info = False, {}
        while not done:
            obs, _, terminated, truncated, info = env.step(policy.act(obs, env.state))
            for event in info["events"]:
                if event["type"] == "death" and AGENT_ID in event["killed_by"]:
                    if event["character"] == AGENT_ID:
                        totals["self_kills"] += 1
                    else:
                        totals["kills"] += 1
                elif event["type"] == "block_destroyed" and AGENT_ID in event["destroyed_by"]:
                    totals["crates"] += 1
            done = terminated or truncated
        totals[info["outcome"] or "timeout"] += 1
        totals["steps"] += info["steps"]
    return totals


class SeededRandomPolicy:
    """A reproducible stand-in agent: random actions from a per-game RNG."""

    def __init__(self, seed):
        self.seed = seed
        self.games = 0

    def reset(self):
        self.rng = np.random.default_rng(self.seed + self.games)
        self.games += 1

    def act(self, obs, state):
        return int(self.rng.integers(6))


@unittest.skipIf(torch is None, MISSING)
class EvaluateAgentUnchangedTest(unittest.TestCase):
    def test_old_keys_and_values_unchanged(self):
        from rl.config import EnvConfig
        for level in ("random", "noisy50", "rule"):
            results = {}
            for name, fn in (("old", old_evaluate_full), ("new", _evaluate_full)):
                env = BombermanGymEnv(level, EnvConfig(max_steps=120))
                try:
                    results[name] = fn(env, SeededRandomPolicy(11), 12, 40)
                finally:
                    env.close()
            old, new = results["old"], results["new"]
            self.assertEqual(list(new)[:len(old)], list(old))
            self.assertEqual({k: new[k] for k in old}, old)
            self.assertEqual(list(new)[len(old):], list(NEW_FULL_KEYS))

    def test_new_totals_add_up(self):
        from rl.config import EnvConfig
        env = BombermanGymEnv("noisy50", EnvConfig(max_steps=200))
        rows = []
        try:
            totals = _evaluate_full(env, SeededRandomPolicy(3), 15, 0, rows)
        finally:
            env.close()
        self.assertEqual([r["seed"] for r in rows], list(range(15)))
        self.assertEqual(totals["death_total"], totals["loss"] + totals["draw"])
        self.assertEqual(totals["death_total"],
                         sum(totals[f"death_{c}"] for c in DEATH_CAUSES))
        self.assertEqual(totals["death_unknown"], 0)
        self.assertEqual(totals["steps"], sum(r["steps"] for r in rows))
        self.assertEqual(totals["steps_death"] + totals["steps_win"] + totals["steps_timeout"],
                         totals["steps"])
        self.assertEqual(totals["steps_death"],
                         sum(r["steps"] for r in rows if r["outcome"] in ("loss", "draw")))
        self.assertEqual(totals["steps_timeout"],
                         sum(r["steps"] for r in rows if r["outcome"] == "timeout"))
        self.assertEqual(totals["buffs_picked"], sum(
            totals[f"buffs_picked_{n}"] for n in ("range", "bomb_count", "burn")))

    def test_command_line_defaults_unchanged(self):
        captured = {}

        def fake_full(policy, opponents, games, first_seed, max_steps):
            captured.update(opponents=opponents, games=games, first_seed=first_seed,
                            max_steps=max_steps)
            return {"games": games, "win": 0, "loss": 0, "draw": 0, "timeout": games,
                    "self_kills": 0, "kills": 0, "crates": 0, "steps": 0}

        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                run = train_main(["--timesteps", "64", "--seed", "0", "--runs-dir", tmp,
                                  "--config", self._tiny(tmp)])
                with mock.patch.object(evaluate_agent, "evaluate_full", fake_full):
                    evaluate_agent.main(["--run", str(run), "--runs-dir", tmp])
        self.assertEqual(captured, {"opponents": "random", "games": 100, "first_seed": 0,
                                    "max_steps": 800})

    @staticmethod
    def _tiny(tmp):
        path = Path(tmp) / "c.json"
        path.write_text(json.dumps(TINY_PPO))
        return str(path)


@unittest.skipIf(torch is None, MISSING)
class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        overrides = cls.tmp / "c.json"
        overrides.write_text(json.dumps(TINY_PPO))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.run_dir = train_main(["--timesteps", "64", "--seed", "0",
                                  "--config", str(overrides), "--runs-dir", str(cls.tmp)])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def protocol(self, sub, **kwargs):
        return run_protocol(self.run_dir, games=3, runs_dir=self.tmp / sub, quiet=True,
                            max_steps=40, **kwargs)

    def read(self, path):
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_all_levels_and_total(self):
        out, rows = self.protocol("a")
        self.assertTrue(out.name.endswith("_eval-protocol_seed1000000"))
        table = self.read(out / "protocol.csv")
        self.assertEqual([r["level"] for r in table], [*LADDER, "total"])
        games = self.read(out / "games.csv")
        self.assertEqual(len(games), 15)
        self.assertEqual({int(g["seed"]) for g in games}, {1_000_000, 1_000_001, 1_000_002})
        for level in LADDER:
            self.assertEqual(sorted(int(g["seed"]) for g in games if g["level"] == level),
                             [1_000_000, 1_000_001, 1_000_002])
        total = table[-1]
        for key in ("games", "win", "loss", "draw", "timeout", "self_kills", "steps",
                    "steps_death", "steps_win", "steps_timeout", "death_total",
                    "buffs_picked", "games_death"):
            self.assertEqual(int(total[key]), sum(int(r[key]) for r in table[:-1]), key)
        self.assertEqual(int(total["games"]), 15)
        self.assertEqual(int(total["death_unknown"]), 0)
        self.assertEqual(int(total["steps"]), sum(int(g["steps"]) for g in games))
        for row in table:
            self.assertAlmostEqual(float(row["mean_length"]),
                                   int(row["steps"]) / int(row["games"]), places=3)
            self.assertNotIn("mean_survival", row)
            for outcomes, n, mean in ((("loss", "draw"), "games_death", "mean_length_death"),
                                      (("win",), "games_win", "mean_length_win"),
                                      (("timeout",), "games_timeout", "mean_length_timeout")):
                lengths = [int(g["steps"]) for g in games if g["outcome"] in outcomes
                           and (row["level"] == "total" or g["level"] == row["level"])]
                self.assertEqual(int(row[n]), len(lengths))
                if lengths:
                    self.assertAlmostEqual(float(row[mean]), sum(lengths) / len(lengths),
                                           places=3)
                else:
                    self.assertEqual(row[mean], "")
        saved = json.loads((out / "eval.json").read_text())
        self.assertEqual(saved["settings"]["policy"], "deterministic")
        self.assertEqual(saved["settings"]["first_seed"], 1_000_000)
        self.assertEqual(rows[-1]["level"], "total")
        # The evaluated run folder is unchanged.
        self.assertEqual(sorted(p.name for p in self.run_dir.iterdir()),
                         ["config.json", "meta.json", "model.zip", "progress.csv"])

    def test_deterministic_and_stochastic_are_repeatable(self):
        for stochastic in (False, True):
            a, _ = self.protocol(f"r{stochastic}1", deterministic=not stochastic)
            b, _ = self.protocol(f"r{stochastic}2", deterministic=not stochastic)
            self.assertEqual((a / "games.csv").read_text(), (b / "games.csv").read_text())
        saved = json.loads((b / "eval.json").read_text())
        self.assertEqual(saved["settings"]["policy"], "stochastic")

    def test_seed_offset_and_bad_arguments(self):
        out, _ = self.protocol("o", seed_offset=200, levels=("random",))
        games = self.read(out / "games.csv")
        self.assertEqual([int(g["seed"]) for g in games], [1_000_200, 1_000_201, 1_000_202])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            protocol_main(["--run", str(self.run_dir), "--seed-offset", "-1",
                           "--runs-dir", str(self.tmp / "bad")])
        with self.assertRaises(FileNotFoundError):
            run_protocol(self.tmp, games=1, runs_dir=self.tmp / "bad")


if __name__ == "__main__":
    unittest.main()
