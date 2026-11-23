"""How fast PPO trains on this computer, to plan the long runs.

Run from the repository root (takes about 15 minutes; the long runs should
use the same priority, so pass ``--high-priority`` on Windows):

    python -m rl.benchmark_ppo --high-priority
    python -m rl.benchmark_ppo --quick          # under a minute, for tests

Measures, each after a warm-up and only for a few thousand steps:

1. raw env speed: steps per second with random actions against ``random``,
   ``noisy25`` and ``rule`` (one env, no learning)
2. PPO training speed (collecting rollouts and the updates, the v0 config
   with only the measured settings changed):
   - against ``random``: every combination of ``torch_threads`` in
     {4, 8, 12, 16}, ``n_envs`` in {4, 8, 12} and ``vec_env`` in
     {dummy, subproc} (24)
   - against ``rule``: the two best thread counts of the ``random`` grid
     with every ``n_envs`` and ``vec_env`` (12). Threads only speed up the
     network update, which does not depend on the opponent; the opponent
     changes the env's cost, which mostly moves the best ``n_envs`` and
     ``vec_env``
   One measurement = one warm-up rollout and update, then enough timed
   rollouts and updates for at least 2,048 steps.
3. runs side by side: 1, 2 or 3 separate training processes at once (the
   best setting with ``n_envs = 8``), each with ``best threads // k`` and
   each with the full best thread count; total steps per second over the
   same 60-second window
4. one evaluation during training: the cost of an evaluation step, the
   worst case of ``eval_games`` games of 800 steps, and the smallest
   ``eval_every`` that keeps that worst case under 5% of training time

``n_envs`` is a PPO hyperparameter (it changes the rollout size), so the
conclusion only chooses threads, ``vec_env`` and how many runs side by side,
always at the v0 ``n_envs = 8``; the other values are for information.

Prints a table and a conclusion (the fastest setting, hours for 1M and 10M
steps, and the steps per seed when 5 seeds share one 8-hour night) and
writes ``runs/<time>_benchmark_seed0/benchmark.json`` with every number and
the machine (CPU, cores, threads, RAM, Python, torch). If the default
benchmark passes ``--budget-minutes`` (20) it stops the training grid early
and marks the results ``partial``.
"""

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from rl.bomberman_env import BombermanGymEnv
from rl.config import EnvConfig, PPOConfig, apply_overrides, config_to_dict
from rl.evaluate_agent import ModelPolicy, evaluate_seeds
from rl.run_logging import ROOT, dependency_versions, git_info, new_run_dir, utc_now, write_json
from rl.seeds import train_eval_seeds
from rl.train_ppo import apply_runtime, make_model

BENCH_SEED = 0  # training seed of every benchmark run
ENV_SEEDS = 5_000_000  # first seed of the raw env measurement (no evaluation range)
NIGHT_HOURS = 8
NIGHT_SEEDS = 5
EVAL_SHARE = 0.05  # the in-training evaluation may cost this share of training time
V0_N_ENVS = PPOConfig().n_envs

FULL = {
    "env_opponents": ("random", "noisy25", "rule"),
    "env_steps": 3000,
    "threads": (4, 8, 12, 16),
    "n_envs": (4, 8, 12),
    "vec_envs": ("dummy", "subproc"),
    "rule_threads": 2,
    "min_timed_steps": 2048,
    "side_by_side": (1, 2, 3),
    "window_seconds": 60.0,
    "eval_games": PPOConfig().eval_games,
    "eval_probe_games": 10,
    "overrides": {},
}
QUICK = {
    "env_opponents": ("random",),
    "env_steps": 200,
    "threads": (1,),
    "n_envs": (2,),
    "vec_envs": ("dummy", "subproc"),
    "rule_threads": 0,
    "min_timed_steps": 64,
    "side_by_side": (1, 2),
    "window_seconds": 3.0,
    "eval_games": 2,
    "eval_probe_games": 2,
    "overrides": {"n_steps": 32, "batch_size": 32, "n_epochs": 1, "features_dim": 32,
                  "conv_channels": [8, 8, 8], "env": {"max_steps": 40}},
}


# --------------------------------------------------------------- machine info

def _cpu_name() -> str | None:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _physical_cores() -> int | None:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property "
                 "NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=30, check=True).stdout
            return int(out.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    try:
        cores = set()
        physical = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("physical id"):
                    physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip() and core is not None:
                    cores.add((physical, core))
                    physical = core = None
        if core is not None:
            cores.add((physical, core))
        return len(cores) or None
    except OSError:
        return None


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64),
                        ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64),
                        ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 2**30, 1)
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return None


def machine_info() -> dict:
    return {"cpu": _cpu_name(), "physical_cores": _physical_cores(),
            "logical_threads": os.cpu_count(), "ram_gb": _ram_gb(),
            "system": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "torch_default_threads": torch.get_num_threads()}


# --------------------------------------------------------------- measurements

def env_speed(opponents: str, steps: int) -> dict:
    """Steps per second of one env with uniformly random agent actions."""
    env = BombermanGymEnv(opponents, EnvConfig())
    rng = np.random.default_rng(0)
    try:
        seed, done = ENV_SEEDS, True
        began = time.perf_counter()
        for _ in range(steps):
            if done:
                env.reset(seed=seed)
                seed += 1
            _, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
            done = terminated or truncated
        seconds = time.perf_counter() - began
    finally:
        env.close()
    return {"opponents": opponents, "steps": steps, "seconds": round(seconds, 3),
            "steps_per_second": round(steps / seconds, 1)}


class CycleClock(BaseCallback):
    """Times each rollout+update cycle; optionally stops training at a deadline."""

    def __init__(self, deadline: float | None = None) -> None:
        super().__init__()
        self.starts: list[float] = []
        self.deadline = deadline

    def _on_rollout_start(self) -> None:
        self.starts.append(time.perf_counter())

    def _on_step(self) -> bool:
        return self.deadline is None or time.perf_counter() < self.deadline


def bench_config(base: PPOConfig, opponents: str, n_envs: int, vec_env: str,
                 threads: int) -> PPOConfig:
    return dataclasses.replace(base, opponents=opponents, n_envs=n_envs, vec_env=vec_env,
                               torch_threads=threads, checkpoint_every=0, eval_every=0)


def training_speed(cfg: PPOConfig, min_timed_steps: int) -> dict:
    """Steps per second of PPO training with ``cfg``, after one warm-up cycle."""
    apply_runtime(cfg)
    cycle = cfg.n_envs * cfg.n_steps
    timed = max(1, math.ceil(min_timed_steps / cycle))
    model = make_model(cfg, BENCH_SEED)
    try:
        clock = CycleClock()
        model.learn(total_timesteps=(1 + timed) * cycle, callback=clock)
        end = time.perf_counter()
    finally:
        model.get_env().close()
    seconds = end - clock.starts[1]
    return {"opponents": cfg.opponents, "torch_threads": cfg.torch_threads,
            "n_envs": cfg.n_envs, "vec_env": cfg.vec_env, "timed_steps": timed * cycle,
            "seconds": round(seconds, 3), "steps_per_second": round(timed * cycle / seconds, 1)}


def worker(spec_path: str) -> None:
    """One side-by-side training process: warm up, wait for "go", train for the window."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    cfg = apply_overrides(PPOConfig(), spec["config"])
    apply_runtime(cfg)
    ready, go, result = Path(spec["ready"]), Path(spec["go"]), Path(spec["result"])
    cycle = cfg.n_envs * cfg.n_steps
    model = make_model(cfg, BENCH_SEED + spec["index"])
    try:
        model.learn(total_timesteps=cycle)  # warm-up
        ready.write_text("ready", encoding="utf-8")
        waited = time.perf_counter()
        while not go.exists():
            if time.perf_counter() - waited > 600:
                raise TimeoutError("no go signal")
            time.sleep(0.05)
        clock = CycleClock(deadline=time.perf_counter() + spec["window_seconds"])
        model.learn(total_timesteps=10**9, callback=clock, reset_num_timesteps=False)
        # Every full cycle starts at one rollout start and ends at the next.
        starts = clock.starts
        cycles = len(starts) - 1
        seconds = starts[-1] - starts[0] if cycles else 0.0
    finally:
        model.get_env().close()
    result.write_text(json.dumps({"cycles": cycles, "steps": cycles * cycle,
                                  "seconds": seconds}), encoding="utf-8")


def side_by_side(base: PPOConfig, k: int, threads: int, window: float, folder: Path) -> dict:
    """Total steps per second of ``k`` training processes at once."""
    folder.mkdir(parents=True)
    go = folder / "go"
    procs, specs, logs = [], [], []
    try:
        for index in range(k):
            spec = {"config": {**config_to_dict(base), "torch_threads": threads},
                    "index": index, "window_seconds": window,
                    "ready": str(folder / f"ready{index}"), "go": str(go),
                    "result": str(folder / f"result{index}.json")}
            path = folder / f"spec{index}.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            specs.append(spec)
            log = open(folder / f"worker{index}.log", "w", encoding="utf-8")
            logs.append(log)
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "rl.benchmark_ppo", "--worker", str(path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        began = time.perf_counter()
        while not all(Path(s["ready"]).exists() for s in specs):
            if any(p.poll() not in (None, 0) for p in procs):
                raise RuntimeError(f"a side-by-side worker failed; see {folder}")
            if time.perf_counter() - began > 900:
                raise TimeoutError("side-by-side workers never got ready")
            time.sleep(0.05)
        go.write_text("go", encoding="utf-8")
        for proc in procs:
            proc.wait(timeout=window + 900)
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError(f"a side-by-side worker failed; see {folder}")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()
    results = [json.loads(Path(s["result"]).read_text(encoding="utf-8")) for s in specs]
    per_run = [r["steps"] / r["seconds"] if r["seconds"] else 0.0 for r in results]
    return {"runs": k, "torch_threads_each": threads, "n_envs": base.n_envs,
            "vec_env": base.vec_env,
            "steps_per_second_each": [round(x, 1) for x in per_run],
            "total_steps_per_second": round(sum(per_run), 1)}


def eval_cost(cfg: PPOConfig, probe_games: int, games: int, train_sps: float) -> dict:
    """Time of in-training evaluation steps with an untrained policy, and the smallest
    ``eval_every`` that keeps the worst case (``games`` games of 800 steps) under
    ``EVAL_SHARE`` of training time at ``train_sps``."""
    apply_runtime(cfg)
    model = make_model(cfg, BENCH_SEED)
    env = BombermanGymEnv(cfg.opponents, EnvConfig())
    try:
        began = time.perf_counter()
        totals = evaluate_seeds(env, ModelPolicy(model), train_eval_seeds(probe_games))
        seconds = time.perf_counter() - began
    finally:
        env.close()
        model.get_env().close()
    per_step = seconds / max(1, totals["steps"])
    worst = games * EnvConfig().max_steps * per_step
    smallest = worst * train_sps / EVAL_SHARE
    return {"probe_games": probe_games, "probe_steps": totals["steps"],
            "seconds_per_eval_step": round(per_step, 6), "eval_games": games,
            "worst_case_seconds": round(worst, 1),
            "smallest_eval_every_for_5_percent": int(math.ceil(smallest / 10_000) * 10_000)}


# --------------------------------------------------------------- the benchmark

def best(rows: list, **match) -> dict | None:
    chosen = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return max(chosen, key=lambda r: r["steps_per_second"]) if chosen else None


def night_plan(sbs: list, cycle: int) -> list:
    """Steps per seed when ``NIGHT_SEEDS`` seeds share one night, for each k.

    ``k`` runs at once need ``ceil(5 / k)`` rounds; each run gets the night's
    share of its own speed, less the evaluation's share. Rounded down to
    whole rollouts. (The last round may run fewer than ``k`` processes,
    which is faster, so this is on the safe side.)
    """
    plans = []
    for entry in sbs:
        k = entry["runs"]
        each = min(entry["steps_per_second_each"])
        rounds = math.ceil(NIGHT_SEEDS / k)
        steps = NIGHT_HOURS * 3600 * each * (1 - EVAL_SHARE) / rounds
        plans.append({**entry, "rounds": rounds,
                      "steps_per_seed_per_night": int(steps // cycle * cycle)})
    return plans


def hours(steps: int, sps: float) -> float:
    return round(steps / sps / 3600 / (1 - EVAL_SHARE), 2)


def run_benchmark(quick: bool = False, high_priority: bool = False,
                  runs_dir: str | Path | None = None, budget_minutes: float = 20.0,
                  argv: list | None = None) -> tuple[Path, dict]:
    plan = QUICK if quick else FULL
    base = apply_overrides(PPOConfig(), {**plan["overrides"], "high_priority": high_priority})
    out = new_run_dir("benchmark", BENCH_SEED, runs_dir)
    started = time.perf_counter()
    default_threads = torch.get_num_threads()
    results = {"kind": "benchmark", "quick": quick, "started_at": utc_now(),
               "command": [sys.executable, *(sys.argv if argv is None else argv)],
               "git": git_info(), "versions": dependency_versions(),
               "machine": machine_info(), "plan": {k: v for k, v in plan.items()},
               "base_config": config_to_dict(base), "partial": False}
    write_json(out / "benchmark.json", results)

    def over_budget() -> bool:
        return not quick and time.perf_counter() - started > budget_minutes * 60

    def say(text: str) -> None:
        print(f"[{(time.perf_counter() - started) / 60:5.1f} min] {text}", flush=True)

    try:
        results["env_speed"] = []
        for opponents in plan["env_opponents"]:
            row = env_speed(opponents, plan["env_steps"])
            results["env_speed"].append(row)
            say(f"env vs {opponents}: {row['steps_per_second']} steps/s")

        grid = [("random", t, n, v) for t in plan["threads"] for n in plan["n_envs"]
                for v in plan["vec_envs"]]
        results["training"] = []
        done_random = False
        for phase in ("random", "rule"):
            if phase == "rule":
                if not plan["rule_threads"]:
                    break
                by_threads = {}
                for r in results["training"]:
                    by_threads[r["torch_threads"]] = max(by_threads.get(r["torch_threads"], 0),
                                                         r["steps_per_second"])
                top = sorted(by_threads, key=by_threads.get, reverse=True)[:plan["rule_threads"]]
                grid = [("rule", t, n, v) for t in sorted(top) for n in plan["n_envs"]
                        for v in plan["vec_envs"]]
            for index, (opponents, threads, n_envs, vec) in enumerate(grid):
                if over_budget():
                    results["partial"] = True
                    say("over the time budget: the training grid stops here (partial)")
                    break
                row = training_speed(bench_config(base, opponents, n_envs, vec, threads),
                                     plan["min_timed_steps"])
                results["training"].append(row)
                say(f"train vs {opponents} [{index + 1}/{len(grid)}] threads {threads}, "
                    f"n_envs {n_envs}, {vec}: {row['steps_per_second']} steps/s")
                write_json(out / "benchmark.json", results)
            done_random = True
            if results["partial"]:
                break
        torch.set_num_threads(default_threads)

        top = best(results["training"], opponents="random", n_envs=V0_N_ENVS) \
            or best(results["training"], opponents="random")
        if top is None or not done_random:
            raise RuntimeError("no training measurement finished")
        chosen = bench_config(base, "random", V0_N_ENVS if not quick else top["n_envs"],
                              top["vec_env"], top["torch_threads"])
        results["side_by_side"] = []
        for k in plan["side_by_side"]:
            variants = [top["torch_threads"]]
            if k > 1:
                variants = sorted({max(1, top["torch_threads"] // k), top["torch_threads"]})
            for threads in variants:
                entry = side_by_side(chosen, k, threads, plan["window_seconds"],
                                     out / f"side_by_side_k{k}_threads{threads}")
                entry["threads_split"] = threads != top["torch_threads"]
                results["side_by_side"].append(entry)
                say(f"{k} run(s) at once, {threads} threads each: "
                    f"{entry['total_steps_per_second']} steps/s in total")
                write_json(out / "benchmark.json", results)

        results["eval"] = eval_cost(chosen, plan["eval_probe_games"], plan["eval_games"],
                                    top["steps_per_second"])
        torch.set_num_threads(default_threads)
        results["conclusion"] = conclude(results, chosen)
    finally:
        results["finished_at"] = utc_now()
        results["minutes"] = round((time.perf_counter() - started) / 60, 2)
        write_json(out / "benchmark.json", results)
    print_report(results)
    print(f"results: {out / 'benchmark.json'}")
    return out, results


def conclude(results: dict, chosen: PPOConfig) -> dict:
    cycle = chosen.n_envs * chosen.n_steps
    single = best(results["training"], opponents="random", n_envs=chosen.n_envs,
                  vec_env=chosen.vec_env, torch_threads=chosen.torch_threads)
    plans = night_plan(results["side_by_side"], cycle)
    plan = max(plans, key=lambda p: p["steps_per_seed_per_night"])
    sps = single["steps_per_second"]
    return {
        "opponents": "random",
        "n_envs": chosen.n_envs,
        "vec_env": chosen.vec_env,
        "torch_threads": chosen.torch_threads,
        "single_run_steps_per_second": sps,
        "hours_for_1M_steps": hours(1_000_000, sps),
        "hours_for_10M_steps": hours(10_000_000, sps),
        "runs_side_by_side": plan["runs"],
        "torch_threads_each": plan["torch_threads_each"],
        "total_steps_per_second": plan["total_steps_per_second"],
        "steps_per_seed_per_night": plan["steps_per_seed_per_night"],
        "night": f"{NIGHT_SEEDS} seeds in {NIGHT_HOURS} hours, {plan['rounds']} round(s) "
                 f"of {plan['runs']} run(s), {int(EVAL_SHARE * 100)}% kept for evaluation",
        "night_plans": plans,
        "smallest_eval_every_for_5_percent":
            results["eval"]["smallest_eval_every_for_5_percent"],
        "note": "hours include 5% for the evaluation during training; "
                "n_envs stays at the v0 value (a PPO hyperparameter)",
    }


def print_report(results: dict) -> None:
    print("\nRaw env speed (random agent actions, one env)")
    for r in results.get("env_speed", []):
        print(f"  vs {r['opponents']:8} {r['steps_per_second']:10.1f} steps/s")
    print("\nPPO training speed (steps/s)")
    print(f"  {'opponents':9} {'threads':>7} {'n_envs':>6} {'vec_env':>8} {'steps/s':>9}")
    for r in results.get("training", []):
        print(f"  {r['opponents']:9} {r['torch_threads']:>7} {r['n_envs']:>6} "
              f"{r['vec_env']:>8} {r['steps_per_second']:>9.1f}")
    if results.get("side_by_side"):
        print("\nRuns side by side")
        print(f"  {'runs':>4} {'threads each':>12} {'each (steps/s)':>24} {'total':>8}")
        for r in results["side_by_side"]:
            each = ", ".join(f"{x:.0f}" for x in r["steps_per_second_each"])
            print(f"  {r['runs']:>4} {r['torch_threads_each']:>12} {each:>24} "
                  f"{r['total_steps_per_second']:>8.1f}")
    if results.get("eval"):
        e = results["eval"]
        print(f"\nEvaluation during training: {e['seconds_per_eval_step'] * 1000:.2f} ms per "
              f"step, worst case {e['eval_games']} games x 800 steps = "
              f"{e['worst_case_seconds']} s")
    c = results.get("conclusion")
    if c:
        print("\nConclusion" + (" (PARTIAL: the grid stopped early)" if results["partial"]
                                else ""))
        v0 = " = v0" if c["n_envs"] == V0_N_ENVS else ", not v0: quick mode"
        print(f"  fastest setting (n_envs {c['n_envs']}{v0}): vec_env {c['vec_env']}, "
              f"torch_threads {c['torch_threads']}: "
              f"{c['single_run_steps_per_second']} steps/s for one run")
        print(f"  1M steps: {c['hours_for_1M_steps']} h, 10M steps: "
              f"{c['hours_for_10M_steps']} h (one run, with the evaluation)")
        print(f"  best for 5 seeds: {c['runs_side_by_side']} run(s) at once with "
              f"{c['torch_threads_each']} threads each, {c['total_steps_per_second']} steps/s "
              f"in total")
        print(f"  one 8-hour night: {c['steps_per_seed_per_night']:,} steps per seed "
              f"({c['night']})")
        print(f"  smallest eval_every for <= 5% evaluation cost: "
              f"{c['smallest_eval_every_for_5_percent']:,} steps")


def main(argv: list | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny version for tests")
    parser.add_argument("--high-priority", action="store_true",
                        help="Windows High priority, like the real runs")
    parser.add_argument("--budget-minutes", type=float, default=20.0,
                        help="stop the training grid after this long (default 20)")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    return run_benchmark(quick=args.quick, high_priority=args.high_priority,
                         runs_dir=args.runs_dir, budget_minutes=args.budget_minutes,
                         argv=["-m", "rl.benchmark_ppo", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    main()
