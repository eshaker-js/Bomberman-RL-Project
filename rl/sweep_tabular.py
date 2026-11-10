"""Hyperparameter sweep for tabular Q-learning on the tiny stage.

Run from the repository root:

    python -m rl.sweep_tabular                       # the default plan, 5 seeds
    python -m rl.sweep_tabular --seeds 3 --plan my_plan.json
    python -m rl.plot_tabular runs/<the sweep folder>  # learning curves

Design: one parameter at a time. Every run starts from the v0
``TabularConfig`` (or ``--config`` overrides) and changes exactly one value
from the plan, so each difference in the results comes from that one value.
The base config itself is run once per seed and shared by every group (its
value appears in every list of the default plan).

What is measured, per run:

- ``learned_at``: the greedy policy (no exploration, the same policy
  ``rl.evaluate_agent`` uses) plays one tiny-stage game every
  ``--eval-every`` episodes during training. ``learned_at`` is the first
  checkpoint from which every later checkpoint destroys a crate and survives
  (``None`` if the last one fails). This measures the learned policy, not
  the epsilon-greedy one: with ``epsilon_end = 0.05`` training success stays
  well below 100% even after the policy is perfect.
- ``eval_*``: the final greedy policy over ``--eval-games`` games. The stage
  is deterministic; only greedy ties (unvisited states) are random.
- ``eval_steps``: mean game length, and ``optimal_route``: 1 if every
  evaluation game took the fewest steps possible. That minimum is not
  assumed: ``shortest_solution`` finds it by searching every action
  sequence on the tiny stage (7 steps with the v0 settings).
- training counts (crate / death / timeout) and visited states.

Every run is a normal tabular run folder (``config.json``, ``meta.json``,
``metrics.csv``, ``q_table.npz``) inside one sweep folder
``runs/<time>_sweep-tabular_seed<first seed>/``, which also holds:

- ``plan.json``: the plan, the base config, the seeds and the sweep settings
- ``summary.csv``: one row per run
- ``checkpoints.csv``: one row per greedy checkpoint
- ``aggregate.csv``: one row per (parameter, value) over all seeds
"""

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rl.config import TabularConfig, apply_overrides, config_to_dict, load_config
from rl.evaluate_agent import TabularPolicy, evaluate_tiny
from rl.run_logging import (
    RunLog,
    dependency_versions,
    git_info,
    new_run_dir,
    utc_now,
    write_json,
)
from rl.tabular_env import ACTIONS, TinyStageEnv
from rl.train_tabular import train

# One list of values per parameter. Each list includes the v0 value, so every
# group is compared against the same base runs.
SWEEP_PLAN = {
    "alpha": (0.01, 0.05, 0.1, 0.3, 0.5, 1.0),
    "gamma": (0.5, 0.9, 0.95, 0.99),
    "epsilon_decay_fraction": (0.2, 0.5, 0.8),
    "episodes": (300, 1000, 3000),
}
DEFAULT_SEEDS = 5
DEFAULT_EVAL_EVERY = 50
DEFAULT_EVAL_GAMES = 20
BASE_NAME = "base"

SUMMARY_FIELDS = (
    "variant", "seed", "learned_at",
    "eval_crate_rate", "eval_death_rate", "eval_timeout_rate", "eval_steps",
    "optimal_route",
    "train_crate", "train_death", "train_timeout", "last100_crate_rate",
    "states_visited", "seconds", "run_dir",
)
CHECKPOINT_FIELDS = ("variant", "seed", "episodes", "result", "steps")
AGGREGATE_FIELDS = (
    "param", "value", "variant", "seeds", "solved", "optimal", "learned",
    "learned_at_mean", "learned_at_std", "learned_at_max",
    "eval_steps_mean", "train_crate_rate_mean",
)


@dataclass(frozen=True)
class Variant:
    """One config of the sweep and the (parameter, value) groups it belongs to."""

    name: str
    config: TabularConfig
    groups: tuple  # ((param, value), ...)


def expand_plan(base: TabularConfig, plan: dict) -> list:
    """The base config plus one variant per plan value that differs from it.

    Values are type-checked like ``--config`` overrides (unknown parameters
    raise ``KeyError``, wrong types ``TypeError``).
    """
    base_groups = []
    others = []
    for param, values in plan.items():
        for value in values:
            config = apply_overrides(base, {param: value})
            value = getattr(config, param)  # the checked value, e.g. 1 -> 1.0
            if config == base:
                base_groups.append((param, value))
            else:
                others.append(Variant(f"{param}={value}", config, ((param, value),)))
    return [Variant(BASE_NAME, base, tuple(base_groups)), *others]


def learned_at(checkpoints: list):
    """First checkpoint from which every later one succeeded, or ``None``.

    ``checkpoints`` holds ``(episodes done, succeeded)`` pairs in order.
    """
    first = None
    for episodes, succeeded in checkpoints:
        if not succeeded:
            first = None
        elif first is None:
            first = episodes
    return first


def _situation(state: dict) -> tuple:
    """Everything that can change on the tiny stage before its first crate goes."""
    me = state["characters"][0]
    bombs = tuple(sorted((b["x"], b["y"], b["countdown"]) for b in state["bombs"]))
    flames = tuple(sorted((f["x"], f["y"], f["ticks_left"]) for f in state["flames"]))
    return me["x"], me["y"], bombs, flames


def shortest_solution(config: TabularConfig) -> int | None:
    """Fewest steps that destroy a crate without dying on the tiny stage.

    Breadth-first search over every action sequence, merging sequences that
    reach the same situation (``_situation``). It reads the engine directly,
    not the agent's state encoding, so it is an independent check of what
    the agent could achieve. ``None`` if nothing works within ``max_steps``.
    """
    env = TinyStageEnv(config)
    try:
        env.reset()
        frontier, seen = [env], {_situation(env.state)}
        for depth in range(1, config.max_steps + 1):
            following = []
            for node in frontier:
                for action in range(len(ACTIONS)):
                    child = copy.deepcopy(node)
                    _, _, terminated, truncated, info = child.step(action)
                    if info["result"] == "crate":
                        return depth
                    situation = _situation(child.state)
                    if terminated or truncated or situation in seen:
                        continue
                    seen.add(situation)
                    following.append(child)
            if not following:
                return None
            frontier = following
        return None
    finally:
        env.close()


def greedy_result(table, config: TabularConfig, games: int, seed: int) -> dict:
    """The greedy policy of ``table`` on the tiny stage (ties broken by ``seed``)."""
    return evaluate_tiny(TabularPolicy(table, seed), games, config.max_steps)


def run_one(variant: Variant, seed: int, sweep_dir: Path, eval_every: int,
            eval_games: int, argv: list, shortest: int | None = None) -> tuple:
    """Train and evaluate one (variant, seed). Returns (summary row, checkpoint rows)."""
    config = variant.config
    run_dir = new_run_dir("tabular", seed, sweep_dir / variant.name)
    checkpoints = []

    def checkpoint(episode: int, table) -> None:
        done = episode + 1
        if done % eval_every == 0 or done == config.episodes:
            result = greedy_result(table, config, 1, seed)
            outcome = next(k for k in ("crate", "death", "timeout") if result[k])
            checkpoints.append({"variant": variant.name, "seed": seed,
                                "episodes": done, "result": outcome,
                                "steps": result["steps"]})

    started = time.perf_counter()
    with RunLog(run_dir, "tabular", seed, config_to_dict(config), argv=argv) as log:
        table, summary = train(config, seed, log, on_episode_end=checkpoint)
        table.save(run_dir / "q_table.npz")
        final = greedy_result(table, config, eval_games, seed)
        first = learned_at([(c["episodes"], c["result"] == "crate")
                            for c in checkpoints])
        summary.update({"learned_at": first, "final_eval": final})
        log.finish(summary)
    row = {
        "variant": variant.name,
        "seed": seed,
        "learned_at": "" if first is None else first,
        "eval_crate_rate": final["crate"] / eval_games,
        "eval_death_rate": final["death"] / eval_games,
        "eval_timeout_rate": final["timeout"] / eval_games,
        "eval_steps": final["steps"] / eval_games,
        "optimal_route": int(shortest is not None and final["crate"] == eval_games
                             and final["steps"] == shortest * eval_games),
        "train_crate": summary["crate"],
        "train_death": summary["death"],
        "train_timeout": summary["timeout"],
        "last100_crate_rate": summary["last100_crate_rate"],
        "states_visited": summary["states_visited"],
        "seconds": round(time.perf_counter() - started, 3),
        "run_dir": run_dir.relative_to(sweep_dir).as_posix(),
    }
    return row, checkpoints


def aggregate(variants: list, rows: list) -> list:
    """One row per (parameter, value): how its seeds did together."""
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    groups = []
    for param in dict.fromkeys(p for v in variants for p, _ in v.groups):
        members = [(value, v) for v in variants for p, value in v.groups if p == param]
        for value, variant in sorted(members, key=lambda m: m[0]):
            runs = by_variant.get(variant.name, [])
            learned = [r["learned_at"] for r in runs if r["learned_at"] != ""]
            groups.append({
                "param": param,
                "value": value,
                "variant": variant.name,
                "seeds": len(runs),
                "solved": sum(r["eval_crate_rate"] == 1.0 for r in runs),
                "optimal": sum(r["optimal_route"] for r in runs),
                "learned": len(learned),
                "learned_at_mean": _round(statistics.fmean(learned)) if learned else "",
                "learned_at_std": _round(statistics.pstdev(learned)) if learned else "",
                "learned_at_max": max(learned) if learned else "",
                "eval_steps_mean": _round(statistics.fmean(r["eval_steps"] for r in runs))
                if runs else "",
                "train_crate_rate_mean": _round(statistics.fmean(
                    r["train_crate"] / (r["train_crate"] + r["train_death"]
                                        + r["train_timeout"]) for r in runs))
                if runs else "",
            })
    return groups


def _round(x: float) -> float:
    return round(x, 3)


def write_csv(path: Path, fields: tuple, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def print_aggregate(groups: list) -> None:
    print(f"{'param':24} {'value':>7} {'solved':>7} {'optimal':>8} {'learned':>8} "
          f"{'learned_at':>14} {'eval steps':>11}")
    for g in groups:
        n = g["seeds"]
        at = (f"{g['learned_at_mean']:.0f} +- {g['learned_at_std']:.0f}"
              if g["learned"] else "n/a")
        print(f"{g['param']:24} {g['value']!s:>7} {g['solved']:>3}/{n:<3} "
              f"{g['optimal']:>4}/{n:<3} {g['learned']:>4}/{n:<3} {at:>14} "
              f"{g['eval_steps_mean']:>11}")


def run_sweep(base: TabularConfig, plan: dict, seeds: list, sweep_dir: Path, *,
              eval_every: int = DEFAULT_EVAL_EVERY,
              eval_games: int = DEFAULT_EVAL_GAMES,
              argv: list | None = None, quiet: bool = False) -> list:
    """Run every (variant, seed), write the sweep files and return the aggregate."""
    if eval_every < 1 or eval_games < 1:
        raise ValueError("eval_every and eval_games must be at least 1")
    argv = list(sys.argv if argv is None else argv)
    variants = expand_plan(base, plan)
    by_limit = {}  # the shortest game only depends on the step limit
    shortest = {v.name: by_limit.setdefault(v.config.max_steps,
                                            shortest_solution(v.config))
                for v in variants}
    meta = {
        "kind": "sweep-tabular",
        "command": [sys.executable, *argv],
        "git": git_info(),
        "versions": dependency_versions(),
        "started_at": utc_now(),
        "finished_at": None,
        "base_config": config_to_dict(base),
        "plan": {param: list(values) for param, values in plan.items()},
        "seeds": list(seeds),
        "eval_every": eval_every,
        "eval_games": eval_games,
        "variants": [{"name": v.name, "groups": [list(g) for g in v.groups],
                      "shortest_game": shortest[v.name]}
                     for v in variants],
    }
    write_json(sweep_dir / "plan.json", meta)

    rows, checkpoints = [], []
    total = len(variants) * len(seeds)
    for variant in variants:
        for seed in seeds:
            row, points = run_one(variant, seed, sweep_dir, eval_every, eval_games,
                                  argv, shortest[variant.name])
            rows.append(row)
            checkpoints.extend(points)
            if not quiet:
                print(f"[{len(rows)}/{total}] {variant.name} seed {seed}: "
                      f"learned_at={row['learned_at'] or 'n/a'} "
                      f"eval_crate={row['eval_crate_rate']:.2f} "
                      f"steps={row['eval_steps']:.1f} ({row['seconds']}s)")

    groups = aggregate(variants, rows)
    write_csv(sweep_dir / "summary.csv", SUMMARY_FIELDS, rows)
    write_csv(sweep_dir / "checkpoints.csv", CHECKPOINT_FIELDS, checkpoints)
    write_csv(sweep_dir / "aggregate.csv", AGGREGATE_FIELDS, groups)
    meta["finished_at"] = utc_now()
    write_json(sweep_dir / "plan.json", meta)
    return groups


def load_plan(path: str | None) -> dict:
    """``SWEEP_PLAN``, or the JSON file ``{"param": [values, ...], ...}``."""
    if path is None:
        return dict(SWEEP_PLAN)
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict) or not all(isinstance(v, list) and v
                                             for v in plan.values()):
        raise ValueError("a plan file must map each parameter to a non-empty list")
    return plan


def main(argv: list | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="training seeds per config (default 5)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--plan", default=None,
                        help='JSON file {"param": [values]} (default: SWEEP_PLAN)')
    parser.add_argument("--config", default=None,
                        help="JSON overrides of the base TabularConfig")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY,
                        help="episodes between greedy checkpoints (default 50)")
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES,
                        help="games in the final greedy evaluation (default 20)")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    base = load_config(TabularConfig(), args.config)
    plan = load_plan(args.plan)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    sweep_dir = new_run_dir("sweep-tabular", args.first_seed, args.runs_dir)
    command = sys.argv if argv is None else ["-m", "rl.sweep_tabular", *argv]
    groups = run_sweep(base, plan, seeds, sweep_dir, eval_every=args.eval_every,
                       eval_games=args.eval_games, argv=command)
    print()
    print_aggregate(groups)
    print(f"\nsweep folder: {sweep_dir}")
    return sweep_dir


if __name__ == "__main__":
    main()
