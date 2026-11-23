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
   same 60-second window. Each worker keeps going until it has timed at
   least 2 full rollouts and updates (at most ``WORKER_CAP_SECONDS``); a
   worker that still has none is "not measured" (never 0). A variant whose
   ``runs x threads`` exceeds the computer's logical threads is skipped as
   oversubscribed (it cannot win: the runs only slow each other down)
4. one evaluation during training: the cost of an evaluation step, the
   worst case of ``eval_games`` games of 800 steps, and the smallest
   ``eval_every`` that keeps that worst case under 5% of training time

5. the GPU (Task 9), only when ``torch.cuda.is_available()``; otherwise
   it is skipped, says so (``gpu_skipped``) and everything above is exactly
   as before Task 9:
   - training on the GPU against ``random`` and ``rule`` at ``n_envs = 8``:
     ``torch_threads`` in {1, 2, 4} x ``vec_env`` in {dummy, subproc}
   - for information only, outside the conclusion: the best GPU setting at
     ``n_envs`` 16 and 32 (do more game copies help on the GPU?)
   - 1, 2 and 3 runs at once sharing the GPU at the best GPU setting, one
     variant each, same rules as above but a 30-second window; the main
     process samples the GPU memory meanwhile, so the memory per run
     includes each process's CUDA context
   - the evaluation probe again with the model on the GPU
   To stay near 20 minutes, the CPU grid then drops the 16-thread rows and
   uses one thread count (not two) for ``rule``.

``n_envs`` is a PPO hyperparameter (it changes the rollout size), so the
conclusion only chooses the device, threads, ``vec_env`` and how many runs
side by side, always at the v0 ``n_envs = 8``; the other values are for
information. CPU and GPU are compared by steps per seed per night (a tie
goes to the CPU, the default); with a GPU the conclusion also gives the GPU
memory per run and how many runs fit.

Prints a table and a conclusion (the fastest setting, hours for 1M and 10M
steps, and the steps per seed when 5 seeds share one 8-hour night, and the
device) and writes ``runs/<time>_benchmark_seed0/benchmark.json`` with every number and
the machine (CPU, cores, threads, RAM, Python, torch, GPU). If the default
benchmark passes ``--budget-minutes`` (20) it starts no further training or
side-by-side measurement (the short evaluation probe still runs, the
conclusion needs it) and marks the results ``partial``.
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
from rl.devices import gpu_info
from rl.evaluate_agent import ModelPolicy, evaluate_seeds
from rl.run_logging import ROOT, dependency_versions, git_info, new_run_dir, utc_now, write_json
from rl.seeds import train_eval_seeds
from rl.train_ppo import apply_runtime, make_model

BENCH_SEED = 0  # training seed of every benchmark run
ENV_SEEDS = 5_000_000  # first seed of the raw env measurement (no evaluation range)
NIGHT_HOURS = 8
NIGHT_SEEDS = 5
EVAL_SHARE = 0.05  # the in-training evaluation may cost this share of training time
MIN_TIMED_CYCLES = 2  # full rollouts+updates each side-by-side worker must time
WORKER_CAP_SECONDS = 240.0  # but no worker trains longer than this after "go"
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
    # Task 9: the GPU part (only with CUDA)
    "gpu_threads": (1, 2, 4),
    "gpu_vec_envs": ("dummy", "subproc"),
    "gpu_info_n_envs": (16, 32),
    "gpu_side_by_side": (1, 2, 3),
    "gpu_window_seconds": 30.0,
}
# With CUDA the full CPU grid is cut so CPU and GPU fit in about 20 minutes:
# no 16-thread rows, and one thread count (not two) for the rule phase.
FULL_CPU_CUTS_WITH_GPU = {"threads": (4, 8, 12), "rule_threads": 1}
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
    "gpu_threads": (1,),
    "gpu_vec_envs": ("dummy", "subproc"),
    "gpu_info_n_envs": (4,),
    "gpu_side_by_side": (1, 2),
    "gpu_window_seconds": 3.0,
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
            "torch": torch.__version__, "torch_default_threads": torch.get_num_threads(),
            "gpu": gpu_info()}


def _mb(n_bytes: int) -> float:
    return round(n_bytes / 2**20, 1)


def gpu_used_mb() -> float:
    """Memory in use on GPU 0 by every process (CUDA contexts included)."""
    free, total = torch.cuda.mem_get_info(0)
    return _mb(total - free)


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
    """Times each rollout+update cycle; optionally stops training.

    With a ``deadline`` training stops once it has passed and at least
    ``min_cycles`` full cycles were timed, or at the ``cap`` in any case.
    A full cycle runs from one rollout start to the next.
    """

    def __init__(self, deadline: float | None = None, min_cycles: int = 0,
                 cap: float | None = None) -> None:
        super().__init__()
        self.starts: list[float] = []
        self.deadline = deadline
        self.min_cycles = min_cycles
        self.cap = cap

    def _on_rollout_start(self) -> None:
        self.starts.append(time.perf_counter())

    def keep_going(self, now: float) -> bool:
        if self.deadline is None:
            return True
        if self.cap is not None and now >= self.cap:
            return False
        return now < self.deadline or len(self.starts) - 1 < self.min_cycles

    def _on_step(self) -> bool:
        return self.keep_going(time.perf_counter())


def bench_config(base: PPOConfig, opponents: str, n_envs: int, vec_env: str,
                 threads: int, device: str | None = None) -> PPOConfig:
    """``base`` with the measured settings; ``device`` ``None`` keeps ``base.device``."""
    return dataclasses.replace(base, opponents=opponents, n_envs=n_envs, vec_env=vec_env,
                               torch_threads=threads, checkpoint_every=0, eval_every=0,
                               device=device or base.device)


def training_speed(cfg: PPOConfig, min_timed_steps: int) -> dict:
    """Steps per second of PPO training with ``cfg``, after one warm-up cycle."""
    apply_runtime(cfg)
    cycle = cfg.n_envs * cfg.n_steps
    timed = max(1, math.ceil(min_timed_steps / cycle))
    gpu = cfg.device == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    model = make_model(cfg, BENCH_SEED)
    try:
        clock = CycleClock()
        model.learn(total_timesteps=(1 + timed) * cycle, callback=clock)
        end = time.perf_counter()
    finally:
        model.get_env().close()
    seconds = end - clock.starts[1]
    row = {"opponents": cfg.opponents, "torch_threads": cfg.torch_threads,
           "n_envs": cfg.n_envs, "vec_env": cfg.vec_env, "timed_steps": timed * cycle,
           "seconds": round(seconds, 3), "steps_per_second": round(timed * cycle / seconds, 1)}
    if gpu:
        # Only GPU rows carry the device, so the CPU rows stay as before Task 9.
        row["device"] = "cuda"
        row["gpu_peak_reserved_mb"] = _mb(torch.cuda.max_memory_reserved())
        del model
        torch.cuda.empty_cache()  # the next row starts from an empty GPU cache
    return row


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
        now = time.perf_counter()
        clock = CycleClock(deadline=now + spec["window_seconds"],
                           min_cycles=MIN_TIMED_CYCLES,
                           cap=now + max(WORKER_CAP_SECONDS, spec["window_seconds"]))
        model.learn(total_timesteps=10**9, callback=clock, reset_num_timesteps=False)
        # Every full cycle starts at one rollout start and ends at the next.
        starts = clock.starts
        cycles = len(starts) - 1
        seconds = starts[-1] - starts[0] if cycles else 0.0
    finally:
        model.get_env().close()
    measured = {"cycles": cycles, "steps": cycles * cycle, "seconds": seconds}
    if cfg.device == "cuda":
        measured["gpu_peak_reserved_mb"] = _mb(torch.cuda.max_memory_reserved())
    result.write_text(json.dumps(measured), encoding="utf-8")


def side_by_side(base: PPOConfig, k: int, threads: int, window: float, folder: Path) -> dict:
    """Total steps per second of ``k`` training processes at once.

    On the GPU this process also samples the GPU memory in use, before the
    workers start and while they train, for the memory per run.
    """
    folder.mkdir(parents=True)
    go = folder / "go"
    procs, specs, logs = [], [], []
    gpu = base.device == "cuda"
    memory = {"before": gpu_used_mb(), "peak": 0.0} if gpu else None
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
        if gpu:
            _wait_sampling_gpu(procs, window + 900, memory)
        else:
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
    entry = side_by_side_entry(k, threads, base, results)
    if gpu:
        _, total = torch.cuda.mem_get_info(0)
        entry.update(device="cuda", gpu_total_mb=_mb(total),
                     gpu_used_before_mb=memory["before"], gpu_used_peak_mb=memory["peak"],
                     gpu_peak_reserved_mb_each=[r.get("gpu_peak_reserved_mb")
                                                for r in results])
    return entry


def _wait_sampling_gpu(procs: list, timeout: float, memory: dict) -> None:
    """Wait for every worker, recording the highest GPU memory in use."""
    began = time.perf_counter()
    while any(p.poll() is None for p in procs):
        memory["peak"] = max(memory["peak"], gpu_used_mb())
        if time.perf_counter() - began > timeout:
            raise subprocess.TimeoutExpired("side-by-side workers", timeout)
        time.sleep(0.25)


def side_by_side_entry(k: int, threads: int, base: PPOConfig, results: list) -> dict:
    """The result of ``k`` workers; a worker without a full cycle is not measured.

    The total is only given when every worker was measured: a sum over some
    of them would look like a slower setting, not a failed measurement.
    """
    per_run = [round(r["steps"] / r["seconds"], 1) if r["cycles"] and r["seconds"] else None
               for r in results]
    measured = all(x is not None for x in per_run)
    return {"runs": k, "torch_threads_each": threads, "n_envs": base.n_envs,
            "vec_env": base.vec_env, "measured": measured,
            "cycles_each": [r["cycles"] for r in results],
            "steps_per_second_each": per_run,
            "total_steps_per_second": round(sum(per_run), 1) if measured else None}


def side_by_side_variants(k: int, best_threads: int, logical: int | None) -> tuple:
    """``(threads to measure, threads skipped as oversubscribed)`` for ``k`` runs.

    ``k`` runs try ``best // k`` threads each and the full ``best``; any
    choice with ``k x threads`` above the logical threads is skipped. One
    run alone is always measured.
    """
    wanted = [best_threads] if k == 1 else sorted({max(1, best_threads // k), best_threads})
    if k == 1 or not logical:
        return wanted, []
    keep = [t for t in wanted if k * t <= logical]
    return keep, [t for t in wanted if k * t > logical]


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
        if not entry.get("measured", True) or entry.get("skipped"):
            continue
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
    cuda = torch.cuda.is_available()
    plan = QUICK if quick else FULL
    if cuda and not quick:
        plan = {**plan, **FULL_CPU_CUTS_WITH_GPU}
    base = apply_overrides(PPOConfig(), {**plan["overrides"], "high_priority": high_priority})
    out = new_run_dir("benchmark", BENCH_SEED, runs_dir)
    started = time.perf_counter()
    default_threads = torch.get_num_threads()
    results = {"kind": "benchmark", "quick": quick, "started_at": utc_now(),
               "command": [sys.executable, *(sys.argv if argv is None else argv)],
               "git": git_info(), "versions": dependency_versions(),
               "machine": machine_info(), "plan": {k: v for k, v in plan.items()},
               "base_config": config_to_dict(base), "partial": False,
               "gpu_skipped": None if cuda else
               "CUDA not available (torch.cuda.is_available() is False): GPU part skipped"}
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

        top_gpu = chosen_gpu = None
        if cuda:
            top_gpu = gpu_training(results, base, plan, quick, over_budget, say, out)
            torch.set_num_threads(default_threads)
            if top_gpu is not None:
                chosen_gpu = bench_config(base, "random",
                                          V0_N_ENVS if not quick else top_gpu["n_envs"],
                                          top_gpu["vec_env"], top_gpu["torch_threads"], "cuda")

        top = best(results["training"], opponents="random", n_envs=V0_N_ENVS) \
            or best(results["training"], opponents="random")
        if top is None or not done_random:
            raise RuntimeError("no training measurement finished")
        chosen = bench_config(base, "random", V0_N_ENVS if not quick else top["n_envs"],
                              top["vec_env"], top["torch_threads"])
        results["side_by_side"] = []
        logical = os.cpu_count()
        for k in plan["side_by_side"]:
            variants, skipped = side_by_side_variants(k, top["torch_threads"], logical)
            for threads in skipped:
                results["side_by_side"].append(
                    {"runs": k, "torch_threads_each": threads, "skipped":
                     f"oversubscribed: {k} x {threads} threads > {logical} logical threads"})
                say(f"{k} run(s) at once, {threads} threads each: skipped (oversubscribed)")
            for threads in variants:
                if over_budget() and k > 1:
                    results["partial"] = True
                    say("over the time budget: no more side-by-side measurements (partial)")
                    break
                entry = side_by_side(chosen, k, threads, plan["window_seconds"],
                                     out / f"side_by_side_k{k}_threads{threads}")
                entry["threads_split"] = threads != top["torch_threads"]
                results["side_by_side"].append(entry)
                total = entry["total_steps_per_second"]
                say(f"{k} run(s) at once, {threads} threads each: "
                    + (f"{total} steps/s in total" if entry["measured"]
                       else f"NOT MEASURED (rollouts timed: {entry['cycles_each']})"))
                write_json(out / "benchmark.json", results)

        if chosen_gpu is not None:
            gpu_side_by_side(results, chosen_gpu, plan, over_budget, say, out)

        results["eval"] = eval_cost(chosen, plan["eval_probe_games"], plan["eval_games"],
                                    top["steps_per_second"])
        if chosen_gpu is not None:
            results["eval_gpu"] = eval_cost(chosen_gpu, plan["eval_probe_games"],
                                            plan["eval_games"], top_gpu["steps_per_second"])
        torch.set_num_threads(default_threads)
        results["conclusion"] = conclude(results, chosen, chosen_gpu)
    finally:
        results["finished_at"] = utc_now()
        results["minutes"] = round((time.perf_counter() - started) / 60, 2)
        write_json(out / "benchmark.json", results)
    print_report(results)
    print(f"results: {out / 'benchmark.json'}")
    return out, results


def gpu_training(results: dict, base: PPOConfig, plan: dict, quick: bool,
                 over_budget, say, out: Path) -> dict | None:
    """The GPU grid (``training_gpu``) and the information rows; the best row.

    The grid is ``random`` then ``rule`` at the plan's ``n_envs`` (8 in the
    full benchmark). The information rows (``training_gpu_info``) take the
    best ``random`` setting to more game copies; they are never used by the
    conclusion, because ``n_envs`` is a PPO hyperparameter.
    """
    n_envs = V0_N_ENVS if not quick else plan["n_envs"][0]
    results["training_gpu"] = []
    for opponents in ("random", "rule"):
        if opponents == "rule" and not plan["rule_threads"]:
            break
        grid = [(t, v) for t in plan["gpu_threads"] for v in plan["gpu_vec_envs"]]
        for index, (threads, vec) in enumerate(grid):
            if over_budget():
                results["partial"] = True
                say("over the time budget: the GPU grid stops here (partial)")
                break
            row = training_speed(bench_config(base, opponents, n_envs, vec, threads, "cuda"),
                                 plan["min_timed_steps"])
            results["training_gpu"].append(row)
            say(f"GPU train vs {opponents} [{index + 1}/{len(grid)}] threads {threads}, "
                f"n_envs {n_envs}, {vec}: {row['steps_per_second']} steps/s")
            write_json(out / "benchmark.json", results)
    top = best(results["training_gpu"], opponents="random")
    if top is None:
        results["gpu_skipped"] = "over the time budget before any GPU measurement"
        return None
    results["training_gpu_info"] = []
    for more in plan["gpu_info_n_envs"]:
        if over_budget():
            results["partial"] = True
            break
        row = training_speed(bench_config(base, "random", more, top["vec_env"],
                                          top["torch_threads"], "cuda"),
                             plan["min_timed_steps"])
        row["outside_conclusion"] = True
        results["training_gpu_info"].append(row)
        say(f"GPU info (outside the conclusion): n_envs {more}: "
            f"{row['steps_per_second']} steps/s")
        write_json(out / "benchmark.json", results)
    return top


def gpu_side_by_side(results: dict, chosen_gpu: PPOConfig, plan: dict, over_budget,
                     say, out: Path) -> None:
    """1, 2, 3 runs sharing the GPU at the best GPU setting (``side_by_side_gpu``).

    One variant per run count (the best GPU thread count each); the rules of
    the CPU side by side hold: at least 2 full cycles per worker, "not
    measured" instead of 0, oversubscribed variants skipped.
    """
    results["side_by_side_gpu"] = []
    threads = chosen_gpu.torch_threads
    logical = os.cpu_count()
    for k in plan["gpu_side_by_side"]:
        if k > 1 and logical and k * threads > logical:
            results["side_by_side_gpu"].append(
                {"runs": k, "torch_threads_each": threads, "device": "cuda", "skipped":
                 f"oversubscribed: {k} x {threads} threads > {logical} logical threads"})
            continue
        if over_budget() and k > 1:
            results["partial"] = True
            say("over the time budget: no more GPU side-by-side measurements (partial)")
            break
        entry = side_by_side(chosen_gpu, k, threads, plan["gpu_window_seconds"],
                             out / f"side_by_side_gpu_k{k}_threads{threads}")
        results["side_by_side_gpu"].append(entry)
        total = entry["total_steps_per_second"]
        say(f"GPU: {k} run(s) at once, {threads} threads each: "
            + (f"{total} steps/s in total, GPU memory in use {entry['gpu_used_peak_mb']} MB"
               if entry["measured"] else
               f"NOT MEASURED (rollouts timed: {entry['cycles_each']})"))
        write_json(out / "benchmark.json", results)


def gpu_memory(side_by_side_gpu: list) -> dict | None:
    """GPU memory per run and how many runs fit, from the GPU side by side.

    Uses the measured entry with the most runs: per run = (peak in use -
    in use before the workers) / runs, CUDA contexts included. "Before"
    includes this benchmark process and the desktop, so the fit is on the
    safe side.
    """
    usable = [e for e in side_by_side_gpu if e.get("measured") and not e.get("skipped")
              and e.get("gpu_used_peak_mb") is not None]
    if not usable:
        return None
    entry = max(usable, key=lambda e: e["runs"])
    per_run = (entry["gpu_used_peak_mb"] - entry["gpu_used_before_mb"]) / entry["runs"]
    free = entry["gpu_total_mb"] - entry["gpu_used_before_mb"]
    return {"per_run_mb": round(per_run, 1), "from_runs": entry["runs"],
            "total_mb": entry["gpu_total_mb"], "used_before_mb": entry["gpu_used_before_mb"],
            "runs_that_fit": int(free // per_run) if per_run > 0 else None,
            "note": "per run includes its CUDA context; 'used before' includes the "
                    "benchmark process and the desktop"}


def device_plans(device: str, training: list, sbs: list, chosen: PPOConfig) -> dict:
    """One device's single-run speed and its best night plan."""
    cycle = chosen.n_envs * chosen.n_steps
    single = best(training, opponents="random", n_envs=chosen.n_envs,
                  vec_env=chosen.vec_env, torch_threads=chosen.torch_threads)
    sps = single["steps_per_second"]
    plans = night_plan(sbs, cycle)
    if not plans:  # nothing side by side was measured: one run at a time, grid speed
        plans = night_plan([{"runs": 1, "torch_threads_each": chosen.torch_threads,
                             "steps_per_second_each": [sps],
                             "total_steps_per_second": sps}], cycle)
    plans = [{**p, "device": device} for p in plans]
    return {"device": device, "chosen": chosen, "sps": sps, "plans": plans,
            "plan": max(plans, key=lambda p: p["steps_per_seed_per_night"])}


def conclude(results: dict, chosen: PPOConfig, chosen_gpu: PPOConfig | None = None) -> dict:
    """The fastest setting overall. Without ``chosen_gpu``: the CPU, as before Task 9.

    CPU and GPU are compared by steps per seed per night; the GPU must be
    strictly better (a tie keeps the CPU, the default).
    """
    options = [device_plans("cpu", results["training"], results["side_by_side"], chosen)]
    if chosen_gpu is not None:
        options.append(device_plans("cuda", results["training_gpu"],
                                    results.get("side_by_side_gpu", []), chosen_gpu))
    win = options[0]
    for option in options[1:]:
        if option["plan"]["steps_per_seed_per_night"] > win["plan"]["steps_per_seed_per_night"]:
            win = option
    chosen, sps, plan = win["chosen"], win["sps"], win["plan"]
    plans = [p for option in options for p in option["plans"]]
    evaluation = results["eval_gpu"] if win["device"] == "cuda" else results["eval"]
    return {
        "device": win["device"],
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
            evaluation["smallest_eval_every_for_5_percent"],
        "by_device": {o["device"]: {
            "single_run_steps_per_second": o["sps"],
            "vec_env": o["chosen"].vec_env, "torch_threads": o["chosen"].torch_threads,
            "best_runs_side_by_side": o["plan"]["runs"],
            "best_steps_per_seed_per_night": o["plan"]["steps_per_seed_per_night"]}
            for o in options},
        "gpu_memory": gpu_memory(results.get("side_by_side_gpu", [])),
        "gpu_skipped": results.get("gpu_skipped"),
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
            if r.get("skipped"):
                print(f"  {r['runs']:>4} {r['torch_threads_each']:>12}   skipped: {r['skipped']}")
                continue
            each = ", ".join("not measured" if x is None else f"{x:.0f}"
                             for x in r["steps_per_second_each"])
            total = r["total_steps_per_second"]
            print(f"  {r['runs']:>4} {r['torch_threads_each']:>12} {each:>24} "
                  f"{'n/a' if total is None else f'{total:.1f}':>8}")
    if results.get("training_gpu"):
        print("\nPPO training speed on the GPU (steps/s)")
        print(f"  {'opponents':9} {'threads':>7} {'n_envs':>6} {'vec_env':>8} {'steps/s':>9}"
              f" {'GPU MB':>8}")
        rows = results["training_gpu"] + results.get("training_gpu_info", [])
        for r in rows:
            label = "  (information only, outside the conclusion)" \
                if r.get("outside_conclusion") else ""
            print(f"  {r['opponents']:9} {r['torch_threads']:>7} {r['n_envs']:>6} "
                  f"{r['vec_env']:>8} {r['steps_per_second']:>9.1f} "
                  f"{r['gpu_peak_reserved_mb']:>8.0f}{label}")
    if results.get("side_by_side_gpu"):
        print("\nRuns side by side on the GPU")
        print(f"  {'runs':>4} {'threads each':>12} {'each (steps/s)':>24} {'total':>8}"
              f" {'GPU MB in use':>14}")
        for r in results["side_by_side_gpu"]:
            if r.get("skipped"):
                print(f"  {r['runs']:>4} {r['torch_threads_each']:>12}   skipped: {r['skipped']}")
                continue
            each = ", ".join("not measured" if x is None else f"{x:.0f}"
                             for x in r["steps_per_second_each"])
            total = r["total_steps_per_second"]
            print(f"  {r['runs']:>4} {r['torch_threads_each']:>12} {each:>24} "
                  f"{'n/a' if total is None else f'{total:.1f}':>8} "
                  f"{r['gpu_used_peak_mb']:>14.0f}")
    if results.get("gpu_skipped"):
        print(f"\nGPU: {results['gpu_skipped']}")
    if results.get("eval"):
        e = results["eval"]
        print(f"\nEvaluation during training: {e['seconds_per_eval_step'] * 1000:.2f} ms per "
              f"step, worst case {e['eval_games']} games x 800 steps = "
              f"{e['worst_case_seconds']} s")
    if results.get("eval_gpu"):
        e = results["eval_gpu"]
        print(f"Evaluation during training on the GPU: {e['seconds_per_eval_step'] * 1000:.2f}"
              f" ms per step, worst case {e['worst_case_seconds']} s")
    c = results.get("conclusion")
    if c:
        print("\nConclusion" + (" (PARTIAL: stopped early at the time budget)"
                                if results["partial"]
                                else ""))
        v0 = " = v0" if c["n_envs"] == V0_N_ENVS else ", not v0: quick mode"
        if len(c["by_device"]) > 1:
            for name, d in c["by_device"].items():
                print(f"  {name:4}: {d['single_run_steps_per_second']} steps/s for one run, "
                      f"best night {d['best_steps_per_seed_per_night']:,} steps per seed "
                      f"({d['best_runs_side_by_side']} at once)")
        print(f"  device: {c['device']}")
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
        if c.get("gpu_memory"):
            m = c["gpu_memory"]
            free = f"{m['total_mb'] - m['used_before_mb']:.0f} MB free of {m['total_mb']:.0f} MB"
            if m["runs_that_fit"] is None:
                print(f"  GPU memory: no change seen while the runs trained ({free})")
            else:
                print(f"  GPU memory: about {m['per_run_mb']:.0f} MB per run; "
                      f"{m['runs_that_fit']} run(s) fit in the {free}")


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
