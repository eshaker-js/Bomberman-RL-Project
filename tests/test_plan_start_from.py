"""Plans that start from the saved models of an earlier plan (Task 12, part 2).

- the ``start_from`` plan key: its forms, the refusals of bad forms, a
  variant's own ``start_from`` over the plan's, and ``init_model`` together
  with ``start_from`` refused;
- resolution of ``model``, ``latest_checkpoint``, ``{"stage": k}`` and
  ``{"step": N}`` for every seed; the finished run first, else the newest
  attempt; a missing file names the other attempts that have it and none
  is used; several finished runs are ambiguous;
- every problem of a plan is listed at once and nothing starts, no plan
  folder is made;
- ``plan.json`` records the resolved files, each seed's
  ``config_overrides.json`` and run ``meta.json`` the same file;
- ``--continue`` restarts a run from the same file and refuses a changed one;
- a two-plan chain (plan A, then plan B from A's models) under ``subproc``
  (spawn).

Needs numpy, gymnasium, torch and stable-baselines3, like the other plan
runner tests. Only few-step runs, in temporary folders; every env and file
is closed.
"""

import contextlib
import io
import json
import os
import shutil
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
    from rl import run_ppo_seeds, train_ppo
    from rl.init_model import file_sha256
    from rl.run_ppo_seeds import find_source_file, load_plan, resolve_init_models

# n_envs x n_steps = 128 steps per rollout.
TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 1,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 30}}
# Two stages, checkpoints at 128 and 256: model_128.zip, stage0_idle_128.zip,
# model_256.zip and model.zip.
STAGED = {**TINY, "total_timesteps": 256, "opponents": "idle", "eval_opponents": "random",
          "checkpoint_every": 128, "checkpoint_keep_every": 128,
          "stages": [{"opponents": "idle", "timesteps": 128},
                     {"opponents": "random", "timesteps": 128}]}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args)


def refused(test, argv: list) -> str:
    """Run the plan runner, expect it to refuse; return what it printed to stderr."""
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), \
            test.assertRaises(SystemExit):
        run_ppo_seeds.main(argv)
    return err.getvalue()


def train_into(job_dir: Path, seed: int, overrides: dict) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    config = job_dir / "overrides.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    return quiet(train_ppo.main, ["--seed", str(seed), "--runs-dir", str(job_dir),
                                  "--config", str(config)])


def unfinish(run: Path) -> None:
    """Make a copied run look stopped part-way: no model.zip, no finish time."""
    (run / "model.zip").unlink()
    meta = read_json(run / "meta.json")
    meta["finished_at"] = None
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def spec(plan_dir: Path, which="model", variant="v") -> dict:
    return {"plan_dir": str(plan_dir), "variant": variant, "file": which}


def plan_for(start, seeds=(0, 1), **variants) -> dict:
    return {"name": "b", "seeds": list(seeds), "config": {**TINY, "total_timesteps": 128},
            "start_from": start, "variants": variants or {"v": {}},
            "evaluation": {"games": 2}}


@unittest.skipIf(torch is None, MISSING)
class PlanShapeTest(unittest.TestCase):
    def test_forms(self):
        for which in ("model", "latest_checkpoint", {"stage": 0}, {"step": 128}):
            with self.subTest(which=which):
                plan = load_plan(json.dumps(plan_for(spec(Path("runs/a"), which))))
                self.assertEqual(plan["start_from"]["v"]["file"], which)

    def test_bad_forms_are_refused(self):
        bad = ("x", {"plan_dir": "a", "variant": "v"},
               {**spec(Path("a")), "extra": 1}, spec(Path("a"), "final"),
               spec(Path("a"), {"stage": -1}), spec(Path("a"), {"step": 0}),
               spec(Path("a"), {"step": True}), spec(Path("a"), {"stage": 1, "step": 2}),
               {**spec(Path("a")), "plan_dir": ""}, {**spec(Path("a")), "variant": 3})
        for value in bad:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "start_from"):
                load_plan(json.dumps(plan_for(value)))
        with self.assertRaisesRegex(ValueError, "start_from.plan_dir"):
            load_plan(json.dumps(plan_for({**spec(Path("a")), "plan_dir": None})))

    def test_a_variants_own_start_from_wins(self):
        plan = load_plan(json.dumps(plan_for(
            spec(Path("a")), x={}, y={"start_from": spec(Path("b"), "latest_checkpoint")})))
        self.assertEqual(plan["start_from"]["x"]["plan_dir"], "a")
        self.assertEqual(plan["start_from"]["y"]["plan_dir"], "b")
        self.assertNotIn("start_from", plan["variants"]["y"])  # not a PPOConfig field

    def test_init_model_and_start_from_together_are_refused(self):
        with self.assertRaisesRegex(ValueError, "either init_model or start_from"):
            load_plan(json.dumps(plan_for(spec(Path("a")), v={"init_model": "m.zip"})))

    def test_a_plan_without_start_from_is_as_before(self):
        plan = load_plan(json.dumps({"seeds": [0], "config": {**TINY}}))
        self.assertNotIn("start_from", plan)
        self.assertEqual(resolve_init_models(plan), {})


@unittest.skipIf(torch is None, MISSING)
class ResolveTest(unittest.TestCase):
    """Resolution against an earlier plan folder with two staged seeds (made once)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.plan_a = cls.tmp / "plan_a"
        cls.plan_a.mkdir()
        (cls.plan_a / "plan.json").write_text("{}", encoding="utf-8")
        cls.runs = {s: train_into(cls.plan_a / "v" / f"seed{s}", s, STAGED) for s in (0, 1)}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def copy_plan(self, name: str) -> Path:
        copy = self.tmp / name
        shutil.copytree(self.plan_a, copy)
        return copy

    def test_every_form_for_every_seed(self):
        cases = {"model": ("model.zip", 256), "latest_checkpoint": ("checkpoints/model_256.zip", 256),
                 "stage": ("checkpoints/stage0_idle_128.zip", 128),
                 "step": ("checkpoints/model_128.zip", 128)}
        forms = {"model": "model", "latest_checkpoint": "latest_checkpoint",
                 "stage": {"stage": 0}, "step": {"step": 128}}
        for key, (relative, steps) in cases.items():
            with self.subTest(file=key):
                plan = load_plan(json.dumps(plan_for(spec(self.plan_a, forms[key]))))
                resolved = resolve_init_models(plan)["v"]
                self.assertEqual(sorted(resolved), ["0", "1"])
                for seed in (0, 1):
                    entry = resolved[str(seed)]
                    path = self.runs[seed] / relative
                    self.assertEqual(entry["path"], str(path.resolve()))
                    self.assertEqual(entry["sha256"], file_sha256(path))
                    self.assertEqual(entry["num_timesteps"], steps)
                    self.assertEqual(entry["source_seed"], seed)
                    self.assertEqual(entry["source_run_dir"], str(self.runs[seed].resolve()))
                    self.assertEqual(entry["start_from"]["file"], forms[key])

    def test_relative_paths_are_taken_from_the_repository_root(self):
        relative = os.path.relpath(self.plan_a, ROOT)
        plan = load_plan(json.dumps(plan_for(spec(Path(relative)))))
        resolved = resolve_init_models(plan)["v"]
        self.assertEqual(resolved["0"]["path"], str((self.runs[0] / "model.zip").resolve()))

    def test_finished_run_first_else_newest_attempt(self):
        plan_dir = self.copy_plan("newest")
        job = plan_dir / "v" / "seed0"
        finished = job / self.runs[0].name
        newer = job / "29991231-000000-000000_ppo_seed0"
        shutil.copytree(finished, newer)
        unfinish(newer)
        path, problem = find_source_file(plan_dir, "v", 0, "latest_checkpoint")
        self.assertIsNone(problem)
        self.assertEqual(path, finished / "checkpoints" / "model_256.zip")
        # Without a finished run, the newest attempt is the source.
        unfinish(finished)
        path, problem = find_source_file(plan_dir, "v", 0, "latest_checkpoint")
        self.assertEqual(path, newer / "checkpoints" / "model_256.zip")
        _, problem = find_source_file(plan_dir, "v", 0, "model")
        self.assertIn("has no model.zip", problem)

    def test_a_missing_file_names_the_other_attempts_and_none_is_used(self):
        plan_dir = self.copy_plan("older")
        job = plan_dir / "v" / "seed0"
        older = job / self.runs[0].name
        unfinish(older)
        newer = job / "29991231-000000-000000_ppo_seed0"
        shutil.copytree(older, newer)
        shutil.rmtree(newer / "checkpoints")
        path, problem = find_source_file(plan_dir, "v", 0, "latest_checkpoint")
        self.assertIsNone(path)
        self.assertIn(newer.name, problem)
        self.assertIn(f"other attempts that have it (not used): ['{older.name}']", problem)

    def test_several_finished_runs_are_ambiguous(self):
        plan_dir = self.copy_plan("ambiguous")
        job = plan_dir / "v" / "seed1"
        shutil.copytree(job / self.runs[1].name, job / "29991231-000000-000000_ppo_seed1")
        _, problem = find_source_file(plan_dir, "v", 1, "model")
        self.assertIn("ambiguous: 2 finished training runs", problem)

    def test_stage_problems(self):
        _, problem = find_source_file(self.plan_a, "v", 0, {"stage": 1})
        self.assertIn('last stage', problem)
        self.assertIn('"file": "model"', problem)
        _, problem = find_source_file(self.plan_a, "v", 0, {"stage": 5})
        self.assertIn("has only 2 stages", problem)

    def test_every_problem_is_listed_and_nothing_starts(self):
        plan_dir = self.copy_plan("problems")
        shutil.copytree(plan_dir / "v" / "seed1" / self.runs[1].name,
                        plan_dir / "v" / "seed1" / "29991231-000000-000000_ppo_seed1")
        plan = plan_for(spec(plan_dir), seeds=(0, 1, 2), v={},
                        small={"features_dim": 16},
                        step={"start_from": spec(plan_dir, {"step": 999})},
                        nowhere={"start_from": spec(self.tmp / "no_plan_here")})
        path = self.tmp / "plan_b.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        runs_dir = self.tmp / "runs_b"
        message = refused(self, [str(path), "--runs-dir", str(runs_dir), "--quiet"])
        self.assertIn("nothing was started", message)
        self.assertIn("variant v, seed 1: ambiguous", message)
        self.assertIn("variant v, seed 2: no training run", message)
        self.assertIn("variant small, seed 0: init_model", message)
        self.assertIn("features_dim: 32 vs 16", message)
        self.assertIn("variant step, seed 0: ", message)
        self.assertIn("no checkpoints/model_999.zip", message)
        self.assertIn("variant nowhere, seed 0: ", message)
        self.assertIn("is not a plan folder", message)
        # 4 variants x 3 seeds, all refused but seed 0 of v.
        self.assertEqual(message.count("\n- "), 11)
        self.assertFalse(runs_dir.exists() and any(runs_dir.iterdir()))

    def test_init_model_set_directly_is_checked_and_recorded(self):
        source = self.runs[0] / "model.zip"
        plan = load_plan(json.dumps({"seeds": [0, 1], "config": {**TINY},
                                     "variants": {"v": {"init_model": str(source)}}}))
        resolved = resolve_init_models(plan)["v"]
        self.assertEqual([resolved[s]["path"] for s in ("0", "1")],
                         [str(source.resolve())] * 2)
        self.assertEqual(resolved["0"]["init_model"], str(source))
        plan["variants"]["v"]["init_model"] = str(self.tmp / "missing.zip")
        with self.assertRaisesRegex(ValueError, "seed 1: init_model .* no such file"):
            resolve_init_models(plan)


@unittest.skipIf(torch is None, MISSING)
class ChainTest(unittest.TestCase):
    """Plan A trains under subproc (spawn); plan B starts from A's models; --continue."""

    def test_two_plan_chain_and_continue(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            config = {**TINY, "total_timesteps": 128, "vec_env": "subproc"}
            file_a = tmp / "a.json"
            file_a.write_text(json.dumps({"name": "a", "seeds": [0, 1], "config": config,
                                          "evaluation": {"games": 2}}), encoding="utf-8")
            plan_a = quiet(run_ppo_seeds.main, [str(file_a), "--runs-dir", str(tmp), "--quiet"])
            file_b = tmp / "b.json"
            file_b.write_text(json.dumps({
                "name": "b", "seeds": [0, 1], "config": {**config, "env_seed_offset": 1000},
                "start_from": {"plan_dir": str(plan_a), "variant": "v0", "file": "model"},
                "evaluation": {"games": 2}}), encoding="utf-8")
            plan_b = quiet(run_ppo_seeds.main, [str(file_b), "--runs-dir", str(tmp), "--quiet"])

            saved = read_json(plan_b / "plan.json")
            self.assertNotIn("init_models", read_json(plan_a / "plan.json"))
            records = saved["init_models"]["v0"]
            for seed in (0, 1):
                source = run_ppo_seeds.finished_training(plan_a / "v0" / f"seed{seed}", seed)
                entry = records[str(seed)]
                self.assertEqual(entry["path"], str((source / "model.zip").resolve()))
                self.assertEqual(entry["sha256"], file_sha256(source / "model.zip"))
                job_dir = plan_b / "v0" / f"seed{seed}"
                overrides = read_json(job_dir / "config_overrides.json")
                self.assertEqual(overrides["init_model"], entry["path"])
                run = run_ppo_seeds.finished_training(job_dir, seed)
                meta = read_json(run / "meta.json")
                self.assertEqual(meta["init_model"]["sha256"], entry["sha256"])
                self.assertEqual(meta["init_model"]["source_seed"], seed)
                self.assertTrue(meta["init_model"]["verified"])
            status = {r["seed"]: r["status"] for r in run_ppo_seeds._read_csv(
                plan_b / "status.csv")}
            self.assertEqual(status, {"0": "done", "1": "done"})

            # Seed 1 of plan B is "lost": --continue must train it again from the same file.
            for folder in (plan_b / "v0" / "seed1").iterdir():
                if folder.is_dir():
                    shutil.rmtree(folder)
            source = Path(records["1"]["path"])
            backup = tmp / "backup.zip"
            shutil.copy(source, backup)
            source.write_bytes(b"changed")
            message = refused(self, ["--continue", str(plan_b), "--quiet"])
            self.assertIn("variant v0, seed 1:", message)
            self.assertIn("changed since the plan started", message)
            self.assertNotIn("seed 0", message)  # its training already finished
            shutil.copy(backup, source)
            quiet(run_ppo_seeds.main, ["--continue", str(plan_b), "--quiet"])
            run = run_ppo_seeds.finished_training(plan_b / "v0" / "seed1", 1)
            self.assertEqual(read_json(run / "meta.json")["init_model"]["sha256"],
                             records["1"]["sha256"])
            # Every run finished: --continue needs no source file any more.
            source.unlink()
            quiet(run_ppo_seeds.main, ["--continue", str(plan_b), "--quiet"])


if __name__ == "__main__":
    unittest.main()
