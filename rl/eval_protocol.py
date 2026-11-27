"""The evaluation protocol of every agent: all 5 ladder levels, fixed seeds.

Run from the repository root:

    python -m rl.eval_protocol --run runs/<ppo run>
    python -m rl.eval_protocol --run runs/<ppo run> --stochastic
    python -m rl.eval_protocol --run runs/<ppo run> --seed-offset 200   # fresh seeds

The saved PPO agent plays ``--games`` games (default 200) against three bots
of each ladder level (``random < noisy50 < noisy25 < noisy10 < rule``), on
the evaluation seeds of ``rl.seeds.EVAL_SEEDS`` (1,000,000 ..; training never
plays them). Every level uses the same seeds, so levels and training seeds
are compared on the same maps. Games stop at ``EnvConfig().max_steps`` (800,
the game's limit), whatever the run trained with.

The policy is deterministic by default. ``--stochastic`` samples actions;
torch's global RNG is then seeded once per level (first seed + level index),
so a stochastic evaluation is repeatable too (on the same torch thread count).

Writes a new folder ``runs/<time>_eval-protocol_seed<first seed>/``:

- ``protocol.csv``: one row per level and a ``total`` row (counts, rates
  and means; see ``PROTOCOL_FIELDS``)
- ``games.csv``: one row per game (level, seed, outcome, length, kills,
  crates, buffs, death cause)
- ``eval.json``, ``config.json``, ``meta.json``

The evaluated run folder is never changed.

Task 13: the agent plays with the danger map and the safety filter its run
trained with (``env.danger_obs`` and ``env.action_mask`` in the run's
``config.json``; a run from before Task 13 has neither, so exactly as
before). ``--action-mask`` overrides the filter for an ablation:
``off`` plays a filter-trained agent without its filter, ``board`` or
``cautious`` add a filter at test time, also to a plain PPO agent (its
action probabilities without the masked actions). The default ``run`` is
what it trained with. ``eval.json`` and ``config.json`` record the danger
map, the filter used, the filter it trained with and the algorithm.
"""

import argparse
import csv
import sys
from pathlib import Path

from agents.noisy_bot import LADDER
from rl.config import EnvConfig
from rl.evaluate_agent import (
    BUFF_NAMES,
    DEATH_CAUSES,
    NEW_FULL_KEYS,
    PPOPolicy,
    check_observation,
    detect_kind,
    evaluate_full,
    is_maskable,
    run_settings,
)
from rl.safety import MASK_MODES, check_mask_mode
from rl.run_logging import RunLog, new_run_dir, write_json
from rl.seeds import eval_seeds

LEVELS = tuple(LADDER)
# ``--action-mask``: "run" = the filter the run trained with (Task 13).
ACTION_MASK_CHOICES = ("run", *MASK_MODES)
DEFAULT_GAMES = 200

COUNT_FIELDS = ("games", "win", "loss", "draw", "timeout", "self_kills", "kills",
                "crates", "steps", *NEW_FULL_KEYS)
# Game length by outcome: the mean over the games that ended that way, and
# their count (death = loss or draw). A mean over no games is left empty.
LENGTH_FIELDS = ("games_death", "mean_length_death", "games_win", "mean_length_win",
                 "games_timeout", "mean_length_timeout")
RATE_FIELDS = ("win_rate", "loss_rate", "draw_rate", "timeout_rate", "self_kill_rate",
               "kills_per_game", "crates_per_game", "mean_length", *LENGTH_FIELDS,
               "buffs_picked_per_game", "buffs_raised_per_game")
PROTOCOL_FIELDS = ("level", *RATE_FIELDS, *COUNT_FIELDS)
GAME_FIELDS = ("level", "seed", "outcome", "steps", "self_kill", "kills",
               "crates", "buffs_picked", "buffs_raised",
               *(f"buffs_picked_{n}" for n in BUFF_NAMES),
               *(f"buffs_raised_{n}" for n in BUFF_NAMES), "death_cause")


def mean_or_empty(total: float, count: int):
    """``total / count`` rounded, or ``""`` when there were no such games."""
    return round(total / count, 4) if count else ""


def length_by_outcome(counts: dict) -> dict:
    """The ``LENGTH_FIELDS`` of ``evaluate_full``-style totals."""
    deaths = counts["loss"] + counts["draw"]
    return {"games_death": deaths,
            "mean_length_death": mean_or_empty(counts["steps_death"], deaths),
            "games_win": counts["win"],
            "mean_length_win": mean_or_empty(counts["steps_win"], counts["win"]),
            "games_timeout": counts["timeout"],
            "mean_length_timeout": mean_or_empty(counts["steps_timeout"],
                                                 counts["timeout"])}


def summary_row(level: str, counts: dict) -> dict:
    """The ``protocol.csv`` row of ``counts`` (``evaluate_full`` totals)."""
    games = counts["games"]

    def per(key: str) -> float:
        return round(counts[key] / games, 4) if games else 0.0

    row = {"level": level,
           "win_rate": per("win"), "loss_rate": per("loss"), "draw_rate": per("draw"),
           "timeout_rate": per("timeout"), "self_kill_rate": per("self_kills"),
           "kills_per_game": per("kills"), "crates_per_game": per("crates"),
           "mean_length": per("steps"), **length_by_outcome(counts),
           "buffs_picked_per_game": per("buffs_picked"),
           "buffs_raised_per_game": per("buffs_raised")}
    row.update({key: counts[key] for key in COUNT_FIELDS})
    return row


def total_counts(per_level: dict) -> dict:
    """The counts of every level added together."""
    return {key: sum(counts[key] for counts in per_level.values()) for key in COUNT_FIELDS}


def write_csv(path: Path, fields: tuple, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_levels(policy, seeds: range, levels=LEVELS, max_steps: int | None = None,
                    stochastic_seed: bool = False,
                    danger_obs: bool = False) -> tuple[dict, list]:
    """``({level: totals}, game rows)`` of ``policy`` on ``seeds`` against each level."""
    max_steps = max_steps or EnvConfig().max_steps
    per_level, games = {}, []
    for index, level in enumerate(levels):
        if stochastic_seed:
            import torch  # only needed for a sampling policy
            torch.manual_seed(seeds.start + index)
        rows = []
        per_level[level] = evaluate_full(policy, level, len(seeds), seeds.start,
                                         max_steps, per_game=rows, danger_obs=danger_obs)
        games += [{"level": level, **row} for row in rows]
    return per_level, games


def run_protocol(run: str | Path, *, games: int = DEFAULT_GAMES, seed_offset: int = 0,
                 deterministic: bool = True, levels=LEVELS,
                 runs_dir: str | Path | None = None, argv: list | None = None,
                 quiet: bool = False, max_steps: int | None = None,
                 action_mask: str = "run") -> tuple[Path, list]:
    """Evaluate the PPO run folder ``run``; return the results folder and its rows.

    ``action_mask`` (Task 13): ``"run"`` (default) plays with the filter the
    run trained with, or one of ``rl.safety.MASK_MODES``.

    ``max_steps`` is only for tests (short games); the protocol itself, and
    its command line, always use ``EnvConfig().max_steps``.
    """
    run = Path(run)
    if detect_kind(run) != "ppo":
        raise ValueError(f"{run} is not a PPO run (no model.zip)")
    unknown = [level for level in levels if level not in LADDER]
    if unknown:
        raise ValueError(f"unknown ladder levels {unknown}; choose from {list(LADDER)}")
    trained = run_settings(run)
    if action_mask != "run":
        check_mask_mode(action_mask)
    mask = trained["action_mask"] if action_mask == "run" else action_mask
    check_mask_mode(mask)
    seeds = eval_seeds(games, seed_offset)
    max_steps = max_steps or EnvConfig().max_steps
    policy = PPOPolicy(run / "model.zip", deterministic=deterministic, action_mask=mask)
    check_observation(policy.model, trained["danger_obs"], run / "model.zip")
    settings = {"run": str(run), "kind": "ppo", "protocol": "all ladder levels",
                "levels": list(levels), "games_per_level": games,
                "first_seed": seeds.start, "last_seed": seeds.stop - 1,
                "seed_offset": seed_offset, "max_steps": max_steps,
                "policy": "deterministic" if deterministic else "stochastic",
                "torch_seed_per_level": None if deterministic else
                {level: seeds.start + i for i, level in enumerate(levels)},
                "danger_obs": trained["danger_obs"], "action_mask": mask,
                "trained_action_mask": trained["action_mask"],
                "algorithm": "MaskablePPO" if is_maskable(policy.model) else "PPO"}
    out = new_run_dir("eval-protocol", seeds.start, runs_dir)
    with RunLog(out, "eval-protocol", seeds.start, settings,
                argv=sys.argv if argv is None else ["-m", "rl.eval_protocol", *argv]) as log:
        per_level, game_rows = evaluate_levels(policy, seeds, levels, max_steps,
                                               stochastic_seed=not deterministic,
                                               danger_obs=trained["danger_obs"])
        rows = [summary_row(level, per_level[level]) for level in levels]
        rows.append(summary_row("total", total_counts(per_level)))
        write_csv(out / "protocol.csv", PROTOCOL_FIELDS, rows)
        write_csv(out / "games.csv", GAME_FIELDS, game_rows)
        write_json(out / "eval.json", {"settings": settings, "results": per_level,
                                       "rows": rows})
        log.finish({"rows": rows})
    if not quiet:
        print(f"Protocol for {run}: {settings['policy']} policy, {games} games per level, "
              f"seeds {seeds.start:,} to {seeds.stop - 1:,}, max {max_steps} steps, "
              f"danger map {'on' if trained['danger_obs'] else 'off'}, "
              f"safety filter {mask} (trained with {trained['action_mask']})")
        print_rows(rows)
        print(f"results folder: {out}")
    return out, rows


def print_rows(rows: list) -> None:
    print(f"{'level':>8} {'win':>6} {'draw':>6} {'loss':>6} {'timeout':>8} "
          f"{'selfkill':>9} {'kills/g':>8} {'crates/g':>9} {'steps/g':>8} "
          f"{'death@':>7} {'win@':>6} {'buffs/g':>8}")
    for r in rows:
        print(f"{r['level']:>8} {100 * r['win_rate']:5.1f}% {100 * r['draw_rate']:5.1f}% "
              f"{100 * r['loss_rate']:5.1f}% {100 * r['timeout_rate']:7.1f}% "
              f"{100 * r['self_kill_rate']:8.1f}% {r['kills_per_game']:8.2f} "
              f"{r['crates_per_game']:9.2f} {r['mean_length']:8.1f} "
              f"{_maybe(r['mean_length_death']):>7} {_maybe(r['mean_length_win']):>6} "
              f"{r['buffs_picked_per_game']:8.2f}")
    print("death@ / win@: mean length of the games that ended in the agent's death "
          "(loss or draw) / in a win")
    print("death causes (all games): " + ", ".join(
        f"{cause} {rows[-1][f'death_{cause}']}" for cause in DEATH_CAUSES)
        + f", unknown {rows[-1]['death_unknown']}")


def _maybe(value) -> str:
    return "n/a" if value == "" else f"{value:.1f}"


def main(argv: list | None = None) -> tuple[Path, list]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="PPO run folder (with model.zip)")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="games per level (default 200)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="start this many seeds into the evaluation range "
                             "(default 0: seeds 1,000,000 ..)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of the deterministic policy")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LADDER),
                        help="ladder levels (default: all 5)")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    parser.add_argument("--action-mask", choices=ACTION_MASK_CHOICES, default="run",
                        help="safety filter to play with (default run: what it trained "
                             "with; off / board / cautious for the ablation)")
    args = parser.parse_args(argv)
    try:
        return run_protocol(args.run, games=args.games, seed_offset=args.seed_offset,
                            deterministic=not args.stochastic, levels=tuple(args.levels),
                            runs_dir=args.runs_dir, argv=argv,
                            action_mask=args.action_mask)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
