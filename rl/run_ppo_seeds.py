"""Run a plan of PPO runs (several seeds, optionally a few config variants).

Run from the repository root:

    python -m rl.run_ppo_seeds experiments/ppo/first_runs.json
    python -m rl.run_ppo_seeds --continue runs/<time>_ppo-plan_seed0   # after a stop
    python -m rl.plot_ppo runs/<time>_ppo-plan_seed0                 # learning curves

A plan file (JSON; keys starting with ``_`` are comments and ignored)::

    {"name": "first_runs",
     "seeds": [0, 1, 2, 3, 4],
     "parallel": 1,                      # runs at the same time (from the benchmark)
     "config": {"opponents": "random", "total_timesteps": 2000000, ...},
     "variants": {"v0": {}},             # name -> extra PPOConfig overrides
     "evaluation": {"games": 200, "stochastic": false}}

``config`` holds ``PPOConfig`` overrides for every run, including
``torch_threads``, ``vec_env`` and ``high_priority``; each variant adds its
own on top. A value of ``null`` means "not set yet", and the plan refuses
to start until it is set.

Pattern after ``rl.sweep_tabular`` (Joelle): one folder for the whole plan,
``runs/<time>_ppo-plan_seed<first seed>/``, with ``plan.json`` (the plan,
its SHA-256, each variant's resolved config), ``status.csv``,
``summary.csv`` (final evaluation per variant, seed and level),
``aggregate.csv`` (mean and 95% confidence interval over seeds, per
variant and level) and ``logs/``. Each run lives in
``<variant>/seed<seed>/``: the normal PPO run folder made by
``rl.train_ppo``, then the protocol folder made by ``rl.eval_protocol``
(all 5 levels on the evaluation seeds).

Safe to leave running overnight:

- every training run and every evaluation is its own process: a crash is
  recorded in ``status.csv`` and its log, and the other runs go on;
- ``--continue`` skips every run whose training and evaluation both
  finished, only evaluates a run whose training finished, and starts a
  crashed or interrupted run again **from zero** in a new folder (the old
  folder is left as it was). It refuses a plan whose file has changed;
- with ``high_priority`` the processes run at Windows "High" priority, and
  on Windows the runner keeps the computer from sleeping while it runs
  (``rl.priority``).

The summary and status files are rewritten from the run folders on every
call; the run folders themselves are never changed.

Confidence interval: ``mean +- t(0.975, n - 1) * s / sqrt(n)`` over the
``n`` training seeds, with the sample standard deviation ``s`` and the
Student-t quantiles of ``T_975`` (standard t table, e.g. NIST/SEMATECH
e-Handbook of Statistical Methods, table 1.3.6.7.2).
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rl.config import PPOConfig, apply_overrides, config_to_dict
from rl.curriculum import check_stages
from rl.devices import check_device
from rl.eval_protocol import DEFAULT_GAMES, LEVELS
from rl.evaluate_agent import DEATH_CAUSES
from rl.priority import keep_awake
from rl.run_logging import ROOT, dependency_versions, git_info, new_run_dir, utc_now, write_json

# t(0.975, df) for df = 1 .. 30; above 30 the df=30 value is used (a little wide).
T_975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)

METRICS = ("win_rate", "loss_rate", "draw_rate", "timeout_rate", "self_kill_rate",
           "kills_per_game", "crates_per_game", "mean_length", "mean_length_death",
           "mean_length_win", "mean_length_timeout", "buffs_picked_per_game",
           "buffs_raised_per_game", *(f"death_{c}_rate" for c in DEATH_CAUSES))
SUMMARY_FIELDS = ("variant", "seed", "level", "games", *METRICS, "run_dir", "eval_dir")
AGGREGATE_FIELDS = ("variant", "level", "seeds",
                    *(f"{m}_{part}" for m in METRICS for part in ("mean", "ci_low", "ci_high")))
STATUS_FIELDS = ("variant", "seed", "status", "run_dir", "eval_dir", "note")
POLL_SECONDS = 1.0


# ------------------------------------------------------------------ statistics

def t_975(df: int) -> float:
    if df < 1:
        raise ValueError("a confidence interval needs at least 2 values")
    return T_975[min(df, len(T_975)) - 1]


def mean_ci(values: list) -> tuple:
    """``(mean, low, high)`` of the 95% t interval; ``("", "", "")`` without values.

    One value gives its mean and an empty interval.
    """
    values = [float(v) for v in values if v != "" and v is not None]
    if not values:
        return "", "", ""
    mean = statistics.fmean(values)
    if len(values) == 1:
        return round(mean, 4), "", ""
    half = t_975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return round(mean, 4), round(mean - half, 4), round(mean + half, 4)


# ------------------------------------------------------------------ the plan

def strip_comments(data):
    """``data`` without the keys that start with ``_`` (at every level)."""
    if isinstance(data, dict):
        return {k: strip_comments(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, list):
        return [strip_comments(v) for v in data]
    return data


def _unset(data, path="") -> list:
    if isinstance(data, dict):
        return [p for k, v in data.items() for p in _unset(v, f"{path}{k}.")]
    return [path.rstrip(".")] if data is None else []


def load_plan(text: str) -> dict:
    """The checked plan from the plan file's text (``ValueError`` if it is not usable)."""
    plan = strip_comments(json.loads(text))
    if not isinstance(plan, dict):
        raise ValueError("a plan file holds one JSON object")
    unknown = set(plan) - {"name", "seeds", "parallel", "config", "variants", "evaluation"}
    if unknown:
        raise ValueError(f"unknown plan keys {sorted(unknown)}")
    seeds = plan.get("seeds")
    if not (isinstance(seeds, list) and seeds and all(isinstance(s, int) for s in seeds)
            and len(set(seeds)) == len(seeds)):
        raise ValueError("seeds must be a non-empty list of different integers")
    parallel = plan.get("parallel", 1)
    if not isinstance(parallel, int) or parallel < 1:
        raise ValueError("parallel must be an integer of at least 1")
    variants = plan.get("variants", {"v0": {}})
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must map names to config overrides")
    for name in variants:
        if not name or any(c in name for c in '/\\:*?"<>| '):
            raise ValueError(f"variant name {name!r} cannot be a folder name")
    missing = _unset(plan.get("config", {}), "config.") + [
        p for name, extra in variants.items() for p in _unset(extra, f"variants.{name}.")]
    if missing:
        raise ValueError(f"set these plan values first (they are null): {missing}")
    evaluation = {"games": DEFAULT_GAMES, "stochastic": False, **plan.get("evaluation", {})}
    if set(evaluation) - {"games", "stochastic"}:
        raise ValueError("evaluation takes only games and stochastic")
    configs = {}
    for name, extra in variants.items():
        cfg = apply_overrides(apply_overrides(PPOConfig(), plan.get("config", {})), extra)
        # Task 11: curriculum stages are checked now, for every seed, so a bad
        # plan never starts (nothing to check for a variant without stages).
        for seed in seeds:
            try:
                check_stages(cfg, seed)
            except ValueError as error:
                raise ValueError(f"variant {name}, seed {seed}: {error}") from None
        configs[name] = config_to_dict(cfg)
    return {"name": plan.get("name", "plan"), "seeds": seeds, "parallel": parallel,
            "variants": configs, "evaluation": evaluation}


def plan_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ run folders

def _finished(folder: Path, needed: str) -> bool:
    meta = folder / "meta.json"
    if not (meta.exists() and (folder / needed).exists()):
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("finished_at") is not None
    except (OSError, json.JSONDecodeError):
        return False


def finished_training(job_dir: Path, seed: int) -> Path | None:
    done = [p for p in sorted(job_dir.glob(f"*_ppo_seed{seed}")) if _finished(p, "model.zip")]
    return done[-1] if done else None


def finished_eval(job_dir: Path, run_dir: Path) -> Path | None:
    for folder in sorted(job_dir.glob("*_eval-protocol_seed*"), reverse=True):
        if not _finished(folder, "protocol.csv"):
            continue
        settings = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if Path(settings["run"]).resolve() == run_dir.resolve():
            return folder
    return None


# ------------------------------------------------------------------ running

class Job:
    """One (variant, seed): train, then evaluate, each in its own process."""

    def __init__(self, plan_dir: Path, plan: dict, variant: str, seed: int) -> None:
        self.plan_dir = plan_dir
        self.variant = variant
        self.seed = seed
        self.config = plan["variants"][variant]
        self.evaluation = plan["evaluation"]
        self.dir = plan_dir / variant / f"seed{seed}"
        self.status = "pending"
        self.note = ""
        self.run_dir: Path | None = None
        self.eval_dir: Path | None = None
        self.proc = None
        self.log = None
        self.stage = None
        self.started = 0.0

    def refresh(self) -> None:
        """Read what already finished on disk."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = finished_training(self.dir, self.seed)
        self.eval_dir = finished_eval(self.dir, self.run_dir) if self.run_dir else None
        if self.eval_dir:
            self.status = "done"

    def next_command(self) -> list | None:
        if self.run_dir is None:
            overrides = self.dir / "config_overrides.json"
            if not overrides.exists():
                write_json(overrides, self.config)
            self.stage = "train"
            return [sys.executable, "-m", "rl.train_ppo", "--seed", str(self.seed),
                    "--config", str(overrides), "--runs-dir", str(self.dir)]
        if self.eval_dir is None:
            self.stage = "eval"
            command = [sys.executable, "-m", "rl.eval_protocol", "--run", str(self.run_dir),
                       "--games", str(self.evaluation["games"]), "--runs-dir", str(self.dir)]
            return command + (["--stochastic"] if self.evaluation["stochastic"] else [])
        return None

    def start(self) -> bool:
        """Start the next process; ``False`` when there is nothing left to do."""
        command = self.next_command()
        if command is None:
            self.status = "done"
            return False
        logs = self.plan_dir / "logs"
        logs.mkdir(exist_ok=True)
        attempt = 1
        while (logs / f"{self.variant}_seed{self.seed}_{self.stage}_{attempt}.log").exists():
            attempt += 1
        path = logs / f"{self.variant}_seed{self.seed}_{self.stage}_{attempt}.log"
        self.log = open(path, "w", encoding="utf-8")
        flags = 0
        if self.config.get("high_priority") and sys.platform == "win32":
            flags = subprocess.HIGH_PRIORITY_CLASS
        self.proc = subprocess.Popen(command, cwd=ROOT, stdout=self.log,
                                     stderr=subprocess.STDOUT, creationflags=flags)
        self.status = "training" if self.stage == "train" else "evaluating"
        self.note = path.relative_to(self.plan_dir).as_posix()
        self.started = time.perf_counter()
        return True

    def poll(self) -> str | None:
        """``None`` while running; otherwise how the process ended."""
        code = self.proc.poll()
        if code is None:
            return None
        self.close()
        minutes = (time.perf_counter() - self.started) / 60
        if code != 0:
            self.status = f"crashed ({self.stage}, exit code {code})"
            return "crashed"
        self.refresh()
        if self.stage == "train" and self.run_dir is None:
            self.status = "crashed (train finished without a model)"
            return "crashed"
        if self.stage == "eval" and self.eval_dir is None:
            self.status = "crashed (evaluation finished without results)"
            return "crashed"
        self.note = f"{self.stage} took {minutes:.1f} min"
        return "ok"

    def close(self) -> None:
        if self.log is not None:
            self.log.close()
            self.log = None

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.close()


def write_csv(path: Path, fields: tuple, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_status(plan_dir: Path, jobs: list) -> None:
    rows = [{"variant": j.variant, "seed": j.seed, "status": j.status,
             "run_dir": j.run_dir.relative_to(plan_dir).as_posix() if j.run_dir else "",
             "eval_dir": j.eval_dir.relative_to(plan_dir).as_posix() if j.eval_dir else "",
             "note": j.note} for j in jobs]
    write_csv(plan_dir / "status.csv", STATUS_FIELDS, rows)


def run_jobs(plan_dir: Path, plan: dict, quiet: bool = False) -> list:
    """Run every unfinished job, ``plan["parallel"]`` processes at a time."""
    jobs = [Job(plan_dir, plan, v, s) for v in plan["variants"] for s in plan["seeds"]]
    for job in jobs:
        job.refresh()
    queue = [j for j in jobs if j.status != "done"]
    running: list = []

    def say(text: str) -> None:
        if not quiet:
            print(f"[{utc_now()}] {text}", flush=True)

    say(f"{len(jobs) - len(queue)} of {len(jobs)} runs already done")
    write_status(plan_dir, jobs)
    try:
        while queue or running:
            while queue and len(running) < plan["parallel"]:
                job = queue.pop(0)
                if job.start():
                    running.append(job)
                    say(f"{job.variant} seed {job.seed}: {job.stage} started ({job.note})")
            write_status(plan_dir, jobs)
            time.sleep(POLL_SECONDS)
            for job in list(running):
                ended = job.poll()
                if ended is None:
                    continue
                running.remove(job)
                if ended == "crashed":
                    say(f"{job.variant} seed {job.seed}: {job.status}; the other runs go on")
                elif job.start():  # the evaluation after the training
                    running.append(job)
                    say(f"{job.variant} seed {job.seed}: {job.stage} started")
                else:
                    say(f"{job.variant} seed {job.seed}: done")
                write_status(plan_dir, jobs)
    finally:
        for job in running:
            job.stop()
            job.status = "interrupted (start it again with --continue)"
        write_status(plan_dir, jobs)
    return jobs


# ------------------------------------------------------------------ summary

def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(plan_dir: Path, jobs: list) -> tuple[list, list]:
    """Rows of ``summary.csv`` and ``aggregate.csv`` from the finished jobs."""
    rows = []
    for job in jobs:
        if job.status != "done":
            continue
        for row in _read_csv(job.eval_dir / "protocol.csv"):
            games = int(row["games"])
            entry = {"variant": job.variant, "seed": job.seed, "level": row["level"],
                     "games": games,
                     "run_dir": job.run_dir.relative_to(plan_dir).as_posix(),
                     "eval_dir": job.eval_dir.relative_to(plan_dir).as_posix()}
            for metric in METRICS:
                if metric.startswith("death_") and metric.endswith("_rate"):
                    count = int(row[metric[:-len("_rate")]])
                    entry[metric] = round(count / games, 4) if games else ""
                else:
                    entry[metric] = "" if row[metric] == "" else float(row[metric])
            rows.append(entry)
    aggregate = []
    for variant in dict.fromkeys(j.variant for j in jobs):
        for level in (*LEVELS, "total"):
            members = [r for r in rows if r["variant"] == variant and r["level"] == level]
            if not members:
                continue
            entry = {"variant": variant, "level": level, "seeds": len(members)}
            for metric in METRICS:
                mean, low, high = mean_ci([r[metric] for r in members])
                entry.update({f"{metric}_mean": mean, f"{metric}_ci_low": low,
                              f"{metric}_ci_high": high})
            aggregate.append(entry)
    write_csv(plan_dir / "summary.csv", SUMMARY_FIELDS, rows)
    write_csv(plan_dir / "aggregate.csv", AGGREGATE_FIELDS, aggregate)
    return rows, aggregate


def _ci(entry: dict, metric: str, scale: float = 1.0, digits: int = 1) -> str:
    mean, low, high = (entry[f"{metric}_{p}"] for p in ("mean", "ci_low", "ci_high"))
    if mean == "":
        return "n/a"
    if low == "":
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"


def print_aggregate(aggregate: list) -> None:
    print(f"{'variant':10} {'level':8} {'seeds':>5}  {'win % [95% CI]':>22}  "
          f"{'self-kill % [95% CI]':>22}  {'length [95% CI]':>24}")
    for e in aggregate:
        print(f"{e['variant']:10} {e['level']:8} {e['seeds']:>5}  "
              f"{_ci(e, 'win_rate', 100):>22}  {_ci(e, 'self_kill_rate', 100):>22}  "
              f"{_ci(e, 'mean_length'):>24}")


def _check_devices(parser, plan: dict) -> None:
    """Refuse a plan asking for a device this computer lacks (Task 9).

    Without this every seed would start and fail on its own; the check runs
    before any folder is created.
    """
    for name, config in plan["variants"].items():
        try:
            check_device(config["device"])
        except (ValueError, RuntimeError) as error:
            parser.error(f"variant {name}: {error}")


def main(argv: list | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", nargs="?", help="plan file (JSON)")
    parser.add_argument("--continue", dest="resume", default=None,
                        help="an existing plan folder: finish what is left")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if (args.plan is None) == (args.resume is None):
        parser.error("give a plan file, or --continue <plan folder> (not both)")

    if args.resume:
        plan_dir = Path(args.resume)
        saved = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
        text = saved["plan_text"]
        source = Path(saved["plan_file"])
        if source.exists() and plan_hash(source.read_text(encoding="utf-8")) != saved["sha256"]:
            parser.error(f"{source} changed since this plan started; start a new plan "
                         f"instead (the saved copy is in {plan_dir / 'plan.json'})")
        plan = load_plan(text)
        _check_devices(parser, plan)
    else:
        text = Path(args.plan).read_text(encoding="utf-8")
        try:
            plan = load_plan(text)
        except (ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
        _check_devices(parser, plan)
        plan_dir = new_run_dir("ppo-plan", plan["seeds"][0], args.runs_dir)
        write_json(plan_dir / "plan.json", {
            "kind": "ppo-plan", "plan_file": str(Path(args.plan).resolve()),
            "sha256": plan_hash(text), "plan_text": text, "plan": plan,
            "command": [sys.executable, "-m", "rl.run_ppo_seeds",
                        *(sys.argv[1:] if argv is None else argv)],
            "git": git_info(), "versions": dependency_versions(), "created_at": utc_now()})

    with keep_awake() as awake:
        if not args.quiet:
            print(f"plan folder: {plan_dir}\n{awake}")
        jobs = run_jobs(plan_dir, plan, quiet=args.quiet)
    _, aggregate = summarize(plan_dir, jobs)
    if not args.quiet:
        print()
        print_aggregate(aggregate)
        failed = [j for j in jobs if j.status != "done"]
        for job in failed:
            print(f"NOT DONE: {job.variant} seed {job.seed}: {job.status} ({job.note})")
        print(f"\nplan folder: {plan_dir}")
    return plan_dir


if __name__ == "__main__":
    main()
