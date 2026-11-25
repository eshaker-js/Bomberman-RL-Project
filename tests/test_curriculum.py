"""Curriculum stages in one PPO run (Task 11, part 2).

- ``PPOConfig.stages``: the JSON form, its round trip through
  ``config.json``, and every refusal of ``rl.curriculum.check_stages``
  (before any run folder exists, and in the plan runner);
- a three-stage tiny run: the opponents change at the right step, the step
  count, evaluation during training, checkpoints and reward term logging
  run on across the switches, ``progress.csv`` has the stage column,
  ``meta.json`` the stages, and the end-of-stage checkpoints survive the
  checkpoint keeper and load and evaluate on the CPU;
- no training env plays an evaluation seed in any stage, each stage's envs
  start on ``stage_seed``, and every env set is closed (also on a crash);
- exactness: one stage trains the same model, bit for bit, as the same run
  without stages; a run without stages has no stage column or entry;
- a two-stage tiny run under ``subproc`` (spawn).

The config tests need nothing beyond the standard library; the rest needs
numpy, gymnasium, torch and stable-baselines3, like the other RL tests.
Only few-step runs, in temporary folders; every env and file is closed.
"""

import builtins
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

from rl.config import PPOConfig, Stage, apply_overrides, config_to_dict  # noqa: E402

try:
    import gymnasium  # noqa: F401
    import numpy  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from agents.idle_bot import IdleBot
    from rl import run_ppo_seeds, train_ppo
    from rl.curriculum import check_stages, stage_seed, stage_starts
    from rl.evaluate_agent import PPOPolicy, evaluate_full
    from rl.seeds import is_reserved

# n_envs x n_steps = 128 steps per rollout.
TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}
THREE = [{"opponents": "idle", "timesteps": 128}, {"opponents": "random", "timesteps": 256},
         {"opponents": "noisy25", "timesteps": 128}]
CURRICULUM = {**TINY, "opponents": "idle", "eval_opponents": "random",
              "total_timesteps": 512, "stages": THREE}
TIMING = ("eval_seconds", "wall_seconds")


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_main(tmp: Path, overrides: dict, *extra) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / f"config_{len(list(tmp.glob('config_*.json')))}.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return train_ppo.main(["--seed", "0", "--runs-dir", str(tmp / "runs"),
                               "--config", str(config), *extra])


class StagesConfigTest(unittest.TestCase):
    """The form of ``PPOConfig.stages`` (``rl.config``), no RL packages needed."""

    def test_default_is_no_stages(self):
        self.assertEqual(PPOConfig().stages, ())
        self.assertEqual(config_to_dict(PPOConfig())["stages"], ())

    def test_json_list_becomes_stages_and_round_trips(self):
        cfg = apply_overrides(PPOConfig(), {"stages": THREE})
        self.assertEqual(cfg.stages, (Stage("idle", 128), Stage("random", 256),
                                      Stage("noisy25", 128)))
        saved = json.loads(json.dumps(config_to_dict(cfg)))
        self.assertEqual(saved["stages"], THREE)
        self.assertEqual(apply_overrides(PPOConfig(), saved), cfg)
        self.assertEqual(apply_overrides(cfg, {"stages": []}).stages, ())
        self.assertEqual(apply_overrides(PPOConfig(), {"stages": list(cfg.stages)}), cfg)

    def test_malformed_stages_are_refused(self):
        bad_types = ("idle", 5, None, [["idle", 128]], [{"opponents": "idle", "timesteps": None}],
                     [{"opponents": "idle", "timesteps": True}],
                     [{"opponents": "idle", "timesteps": 128.0}],
                     [{"opponents": None, "timesteps": 128}])
        for value in bad_types:
            with self.subTest(value=value), self.assertRaises(TypeError):
                apply_overrides(PPOConfig(), {"stages": value})
        for value in ([{"opponents": "idle"}], [{"opponents": "idle", "timesteps": 128, "x": 1}]):
            with self.subTest(value=value), self.assertRaisesRegex(KeyError, "timesteps"):
                apply_overrides(PPOConfig(), {"stages": value})


@unittest.skipIf(torch is None, MISSING)
class CheckStagesTest(unittest.TestCase):
    def cfg(self, **changes):
        return apply_overrides(PPOConfig(), {**CURRICULUM, **changes})

    def test_a_valid_curriculum_passes(self):
        check_stages(self.cfg(), seed=0)
        check_stages(PPOConfig(), seed=0)  # nothing to check without stages
        self.assertEqual(stage_starts(self.cfg()), [0, 128, 384])
        self.assertEqual([stage_seed(7, k, 8) for k in range(3)], [7, 15, 23])

    def test_refusals(self):
        cases = {
            "add up": {"total_timesteps": 640},
            "whole number of rollouts": {"stages": [{"opponents": "idle", "timesteps": 192},
                                                    {"opponents": "rule", "timesteps": 320}]},
            "positive": {"stages": [{"opponents": "idle", "timesteps": 0},
                                    {"opponents": "rule", "timesteps": 512}]},
            "unknown opponents": {"stages": [{"opponents": "idle", "timesteps": 256},
                                             {"opponents": "noisy33", "timesteps": 256}]},
            "first stage": {"opponents": "random"},
            "eval_opponents": {"eval_opponents": None},
        }
        for message, changes in cases.items():
            with self.subTest(message):
                cfg = self.cfg(**changes)
                with self.assertRaisesRegex(ValueError, message):
                    check_stages(cfg, seed=0)
                with self.assertRaisesRegex(ValueError, message):
                    train_ppo.make_model(cfg, 0)

    def test_every_stage_seed_is_checked(self):
        # Stage 0 (999,996-999,997) and 1 are free; stage 2 starts at 1,000,000.
        check_stages(self.cfg(), seed=999_990)
        with self.assertRaisesRegex(ValueError, "reserved"):
            check_stages(self.cfg(), seed=999_996)

    def test_refused_before_the_run_folder_exists(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            for changes in ({"total_timesteps": 640}, {"eval_opponents": None},
                            {"opponents": "rule"}):
                with self.subTest(changes), self.assertRaises(ValueError):
                    run_main(tmp, {**CURRICULUM, **changes})
            self.assertFalse((tmp / "runs").exists())

    def test_plan_runner_refuses_a_bad_curriculum(self):
        plan = {"name": "p", "seeds": [0, 999_996], "config": TINY,
                "variants": {"c": {k: v for k, v in CURRICULUM.items() if k not in TINY}}}
        with self.assertRaisesRegex(ValueError, "variant c, seed 999996: .*reserved"):
            run_ppo_seeds.load_plan(json.dumps(plan))
        plan["seeds"] = [0]
        loaded = run_ppo_seeds.load_plan(json.dumps(plan))
        self.assertEqual(json.loads(json.dumps(loaded["variants"]["c"]))["stages"], THREE)
        plan["variants"]["c"]["total_timesteps"] = 256
        with self.assertRaisesRegex(ValueError, "add up"):
            run_ppo_seeds.load_plan(json.dumps(plan))


if torch is not None:
    class Probe(BaseCallback):
        """Records, at each rollout start, the step, the opponents' types and the
        episode seeds of the training envs, and every episode seed played."""

        def __init__(self) -> None:
            super().__init__()
            self.rollouts: list = []
            self.seeds: set = set()

        def _on_rollout_start(self) -> None:
            env = self.model.get_env()
            types = {type(c).__name__ for cs in env.get_attr("controllers") for c in cs.values()}
            self.rollouts.append((int(self.model.num_timesteps), types,
                                  list(env.get_attr("episode_seed"))))

        def _on_step(self) -> bool:
            self.seeds.update(self.model.get_env().get_attr("episode_seed"))
            return True


@unittest.skipIf(torch is None, MISSING)
class StageSwitchTest(unittest.TestCase):
    """``run_stages`` with a probe, on the model of ``make_model``."""

    def train(self, overrides: dict, seed: int = 3):
        cfg = apply_overrides(PPOConfig(), overrides)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run_dir = Path(tmp.name)
        model = train_ppo.make_model(cfg, seed)
        logger = configure(str(run_dir), ["csv"])
        model.set_logger(logger)
        probe, counter, terms = Probe(), train_ppo.OutcomeCounter(), train_ppo.RewardTermLogger()
        keeper = train_ppo.CheckpointKeeper(run_dir / "checkpoints", 128)
        callbacks = train_ppo.StagedCallbacks(
            [counter, terms, train_ppo.CheckpointCallback(keeper, 128), probe])
        meta: dict = {}
        try:
            train_ppo.run_stages(model, cfg, seed, callbacks, run_dir, meta)
        finally:
            logger.close()
            model.get_env().close()
        return model, run_dir, probe, counter, terms, meta

    def test_opponents_seeds_and_counters_across_stages(self):
        model, run_dir, probe, counter, terms, meta = self.train(CURRICULUM)
        self.assertEqual(model.num_timesteps, 512)
        self.assertEqual([(step, types) for step, types, _ in probe.rollouts],
                         [(0, {"IdleBot"}), (128, {"RandomBot"}), (256, {"RandomBot"}),
                          (384, {"NoisyBot"})])
        # Each stage's envs start on its stage seed; nothing played is reserved.
        for (step, _, seeds), first in ((probe.rollouts[0], 3), (probe.rollouts[1], 5),
                                        (probe.rollouts[3], 7)):
            self.assertEqual(seeds, [first, first + 1], step)
        self.assertTrue(probe.seeds)
        self.assertFalse([s for s in probe.seeds if is_reserved(s)])
        self.assertEqual([e["env_seed"] for e in meta["stages"]], [3, 5, 7])
        # Counters run on: one reward-term row per rollout, the same finished
        # episodes as the outcome counter.
        self.assertEqual(len(terms.logged), 4)
        self.assertEqual(sum(row["reward_terms/episodes"] for row in terms.logged),
                         sum(counter.counts.values()))
        # Checkpoints keep their cumulative step numbers.
        names = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
        self.assertEqual(names, ["model_128.zip", "model_256.zip", "model_384.zip",
                                 "model_512.zip", "stage0_idle_128.zip",
                                 "stage1_random_384.zip"])
        rows = read_csv(run_dir / "progress.csv")
        self.assertEqual([(r["curriculum/stage"], r["time/total_timesteps"]) for r in rows],
                         [("0", "128"), ("1", "256"), ("1", "384"), ("2", "512")])

    def test_running_sums_restart_with_the_new_envs(self):
        terms = train_ppo.RewardTermLogger()
        terms.running = [{"outcome": 5.0}]
        model = train_ppo.make_model(apply_overrides(PPOConfig(), CURRICULUM), 3)
        try:
            terms.init_callback(model)
            terms.on_new_stage()
            self.assertEqual(len(terms.running), 2)
            self.assertTrue(all(v == 0.0 for sums in terms.running for v in sums.values()))
        finally:
            model.get_env().close()

    def test_two_stages_under_subproc(self):
        model, run_dir, probe, *_ , meta = self.train(
            {**CURRICULUM, "vec_env": "subproc", "total_timesteps": 256,
             "stages": [{"opponents": "idle", "timesteps": 128},
                        {"opponents": "rule", "timesteps": 128}]})
        self.assertEqual([(step, types) for step, types, _ in probe.rollouts],
                         [(0, {"IdleBot"}), (128, {"RuleBot"})])
        self.assertTrue(model.get_env().closed)
        self.assertEqual([e["start_step"] for e in meta["stages"]], [0, 128])
        self.assertTrue((run_dir / "checkpoints" / "stage0_idle_128.zip").exists())


@contextlib.contextmanager
def record_opened(folder: Path):
    """Record every file object opened under ``folder`` while the block runs."""
    real_open = builtins.open
    opened = []
    root = str(folder.resolve())

    def recording_open(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        if isinstance(file, (str, Path)) and str(Path(file).resolve()).startswith(root):
            opened.append(handle)
        return handle

    with mock.patch.object(builtins, "open", recording_open):
        yield opened


@unittest.skipIf(torch is None, MISSING)
class CurriculumRunTest(unittest.TestCase):
    """A three-stage tiny run through ``main``, with every env set recorded."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.env_sets = []
        real = train_ppo.make_envs

        def spy(cfg, opponents, seed):
            # A new set is only built once every earlier set is closed.
            cls.closed_before_build = all(closed for _, closed in cls.env_sets) \
                and getattr(cls, "closed_before_build", True)
            envs = real(cfg, opponents, seed)
            record = [(opponents, seed), False]
            real_close = envs.close

            def close():
                record[1] = True
                real_close()
            envs.close = close
            cls.env_sets.append(record)
            return envs

        with mock.patch.object(train_ppo, "make_envs", spy):
            cls.run_dir = run_main(cls.tmp, {**CURRICULUM, "eval_every": 128, "eval_games": 2,
                                             "checkpoint_every": 128})

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_env_set_is_closed(self):
        self.assertEqual([opp_seed for opp_seed, _ in self.env_sets],
                         [("idle", 0), ("random", 2), ("noisy25", 4)])
        self.assertTrue(all(closed for _, closed in self.env_sets))
        self.assertTrue(self.closed_before_build)

    def test_meta_config_and_progress(self):
        meta = json.loads((self.run_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(meta["finished_at"])
        self.assertEqual(meta["summary"]["timesteps"], 512)
        self.assertEqual(meta["stages"], [
            {"index": 0, "opponents": "idle", "timesteps": 128, "start_step": 0,
             "env_seed": 0, "checkpoint": "checkpoints/stage0_idle_128.zip"},
            {"index": 1, "opponents": "random", "timesteps": 256, "start_step": 128,
             "env_seed": 2, "checkpoint": "checkpoints/stage1_random_384.zip"},
            {"index": 2, "opponents": "noisy25", "timesteps": 128, "start_step": 384,
             "env_seed": 4, "checkpoint": None}])
        config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["stages"], config["opponents"], config["eval_opponents"]),
                         (THREE, "idle", "random"))
        rows = read_csv(self.run_dir / "progress.csv")
        self.assertEqual([r["curriculum/stage"] for r in rows], ["0", "1", "1", "2"])
        self.assertEqual([r["time/total_timesteps"] for r in rows], ["128", "256", "384", "512"])
        losses = [int(r["outcomes/loss"]) + int(r["outcomes/draw"]) for r in rows]
        self.assertEqual(losses, sorted(losses))  # cumulative across stages
        self.assertTrue(all(r["reward_terms/episodes"] != "" for r in rows))

    def test_evaluation_runs_on_across_stages(self):
        curve = read_csv(self.run_dir / "eval_curve.csv")
        self.assertEqual([r["timesteps"] for r in curve], ["0", "128", "256", "384", "512"])
        games = read_csv(self.run_dir / "eval_games.csv")
        # The same two evaluation seeds every time.
        self.assertEqual({r["seed"] for r in games}, {"2000000", "2000001"})
        self.assertEqual(len(games), 10)

    def test_stage_checkpoints_survive_the_keeper_and_evaluate(self):
        names = sorted(p.name for p in (self.run_dir / "checkpoints").iterdir())
        # checkpoint_keep_every is 1,000,000: the keeper kept only its latest.
        self.assertEqual(names, ["model_512.zip", "stage0_idle_128.zip",
                                 "stage1_random_384.zip"])
        for name in ("stage0_idle_128.zip", "stage1_random_384.zip"):
            policy = PPOPolicy(self.run_dir / "checkpoints" / name)
            self.assertEqual(policy.model.device.type, "cpu")
            totals = evaluate_full(policy, "random", 1, 2_000_000, 40)
            self.assertEqual(totals["games"], 1)

    def test_crash_in_a_later_stage_closes_everything(self):
        tmp = self.tmp / "crash"
        tmp.mkdir()
        built = []
        real = train_ppo.make_envs

        def failing(cfg, opponents, seed):
            if built:
                raise RuntimeError("boom")
            envs = real(cfg, opponents, seed)
            real_close = envs.close
            envs.close = lambda: (built.append("closed"), real_close())
            built.append("built")
            return envs

        with record_opened(tmp) as opened, mock.patch.object(train_ppo, "make_envs", failing):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_main(tmp, {**CURRICULUM, "eval_every": 128, "eval_games": 1})
        self.addCleanup(lambda: [h.close() for h in opened])
        self.assertEqual(built[:2], ["built", "closed"])
        self.assertTrue(opened)
        self.assertEqual([Path(h.name).name for h in opened if not h.closed], [])
        meta = json.loads(next((tmp / "runs").glob("*/meta.json")).read_text(encoding="utf-8"))
        self.assertIsNone(meta["finished_at"])
        self.assertEqual(len(meta["stages"]), 1)


@unittest.skipIf(torch is None, MISSING)
class ExactnessTest(unittest.TestCase):
    def test_one_stage_trains_the_same_model_as_no_stages(self):
        plain = {**TINY, "opponents": "noisy25", "eval_opponents": "random",
                 "total_timesteps": 256, "eval_every": 128, "eval_games": 2}
        staged = {**plain, "stages": [{"opponents": "noisy25", "timesteps": 256}]}
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            run_a = run_main(tmp / "a", plain)
            run_b = run_main(tmp / "b", staged)
            model_a = PPO.load(run_a / "model.zip", device="cpu")
            model_b = PPO.load(run_b / "model.zip", device="cpu")
            params_a, params_b = model_a.policy.state_dict(), model_b.policy.state_dict()
            self.assertEqual(list(params_a), list(params_b))
            for key in params_a:
                self.assertTrue(torch.equal(params_a[key], params_b[key]), key)
            opt_a = model_a.policy.optimizer.state_dict()["state"]
            opt_b = model_b.policy.optimizer.state_dict()["state"]
            for index in opt_a:
                for key in opt_a[index]:
                    self.assertTrue(torch.equal(opt_a[index][key], opt_b[index][key]))
            strip = lambda rows: [{k: v for k, v in r.items()  # noqa: E731
                                   if k not in TIMING and not k.startswith("time/")
                                   and k != "curriculum/stage"} for r in rows]
            self.assertEqual(strip(read_csv(run_a / "progress.csv")),
                             strip(read_csv(run_b / "progress.csv")))
            self.assertEqual(strip(read_csv(run_a / "eval_curve.csv")),
                             strip(read_csv(run_b / "eval_curve.csv")))
            self.assertEqual(read_csv(run_a / "eval_games.csv"),
                             read_csv(run_b / "eval_games.csv"))
            # The run without stages has no stage column and no stage entry.
            self.assertNotIn("curriculum/stage", read_csv(run_a / "progress.csv")[0])
            self.assertEqual([r["curriculum/stage"] for r in read_csv(run_b / "progress.csv")],
                             ["0", "0"])
            meta_a = json.loads((run_a / "meta.json").read_text(encoding="utf-8"))
            self.assertNotIn("stages", meta_a)
            config_a = json.loads((run_a / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config_a["stages"], [])
            self.assertFalse((run_b / "checkpoints").exists())  # one stage = no stage checkpoint


if __name__ == "__main__":
    unittest.main()
