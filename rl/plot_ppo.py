"""Learning-curve figures of PPO runs, from their evaluation during training.

Run from the repository root (needs matplotlib):

    python -m rl.plot_ppo runs/<time>_ppo-plan_seed0     # a plan of rl.run_ppo_seeds
    python -m rl.plot_ppo runs/<time>_ppo_seed0          # one run

Reads every ``eval_curve.csv`` under the folder (one per run; in a plan
folder they are grouped by variant, the first folder level) and writes into
``<folder>/plots/`` (or ``--out``):

- ``learning_curves.png``: win rate, self-kill rate, mean game length, and
  the mean length of the games that ended in the agent's death (loss or
  draw) and in a win, against training steps. One line per variant: the
  mean over seeds with its 95% confidence interval as a band (the same
  Student-t interval as the plan summary; no band with one seed)
- ``death_causes.png``: what the evaluation games ended in, over training,
  as stacked shares of all games, averaged over seeds: each of the six
  death causes, wins and timeouts (they add up to 1). One panel per variant

Only steps at which every seed of a variant was evaluated are drawn. Only
reads the run folders.
"""

import argparse
import csv
from pathlib import Path

from rl.evaluate_agent import DEATH_CAUSES
from rl.plot_tabular import GRID, SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from rl.run_ppo_seeds import mean_ci

PANELS = (
    ("win rate", lambda r: _rate(r, "win")),
    ("self-kill rate", lambda r: _rate(r, "self_kills")),
    ("mean game length (steps)", lambda r: float(r["mean_length"])),
    ("mean length to death (steps)", lambda r: _number(r["mean_length_death"])),
)
ENDINGS = (*(f"death_{c}" for c in DEATH_CAUSES), "death_unknown", "win", "timeout")
ENDING_LABELS = {f"death_{c}": c.replace("_", " ") for c in DEATH_CAUSES}
ENDING_LABELS.update({"death_unknown": "unknown", "win": "win", "timeout": "timeout"})
# Categorical slots 1-7 in fixed order (validated palette of rl.plot_tabular),
# timeouts in neutral gray, the always-zero "unknown" in black.
ENDING_COLORS = dict(zip(ENDINGS, (*SERIES_COLORS[:6], "#000000", SERIES_COLORS[6],
                                   "#bfbfbf")))


def _rate(row: dict, key: str) -> float:
    return int(row[key]) / int(row["games"])


def _number(value: str):
    return "" if value == "" else float(value)


def read_curve(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_curves(folder: Path) -> dict:
    """``{variant: [rows of one run, ...]}`` for every ``eval_curve.csv`` under ``folder``."""
    groups: dict = {}
    for path in sorted(folder.rglob("eval_curve.csv")):
        rows = read_curve(path)
        if not rows:
            continue
        parts = path.relative_to(folder).parts
        variant = parts[0] if len(parts) > 3 else "run"
        groups.setdefault(variant, []).append(rows)
    return groups


def common_steps(runs: list) -> list:
    steps = None
    for rows in runs:
        mine = {int(r["timesteps"]) for r in rows}
        steps = mine if steps is None else steps & mine
    return sorted(steps or ())


def series(runs: list, value) -> tuple:
    """``(steps, means, lows, highs)`` of ``value(row)`` over the runs."""
    steps = common_steps(runs)
    by_step = [{int(r["timesteps"]): r for r in rows} for rows in runs]
    means, lows, highs = [], [], []
    for step in steps:
        mean, low, high = mean_ci([value(rows[step]) for rows in by_step])
        means.append(None if mean == "" else mean)
        lows.append(None if low == "" else low)
        highs.append(None if high == "" else high)
    return steps, means, lows, highs


def ending_shares(runs: list) -> tuple:
    """``(steps, {ending: mean share of games})``; the shares add up to 1."""
    steps = common_steps(runs)
    by_step = [{int(r["timesteps"]): r for r in rows} for rows in runs]
    shares = {ending: [] for ending in ENDINGS}
    for step in steps:
        for ending in ENDINGS:
            values = [_rate(rows[step], ending) for rows in by_step]
            shares[ending].append(sum(values) / len(values))
    return steps, shares


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("training steps", color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot(folder: Path, out: Path | None = None) -> list:
    """Write the two figures; return their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = find_curves(folder)
    if not groups:
        raise FileNotFoundError(f"no eval_curve.csv under {folder} (was eval_every set?)")
    out = Path(out) if out is not None else folder / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, (title, value) in zip(axes.flat, PANELS):
        for color, (variant, runs) in zip(SERIES_COLORS, groups.items()):
            steps, means, lows, highs = series(runs, value)
            points = [(s, m) for s, m in zip(steps, means) if m is not None]
            if not points:
                continue
            ax.plot(*zip(*points), color=color, linewidth=2,
                    label=f"{variant} ({len(runs)} seed{'s' if len(runs) > 1 else ''})")
            band = [(s, lo, hi) for s, lo, hi in zip(steps, lows, highs) if lo is not None]
            if band:
                xs, los, his = zip(*band)
                ax.fill_between(xs, los, his, color=color, alpha=0.18, linewidth=0)
        _style(ax, title)
    axes.flat[0].legend(frameon=False, labelcolor=TEXT_SECONDARY)
    fig.suptitle("Evaluation during training (deterministic policy, mean and 95% CI over seeds)",
                 color=TEXT_PRIMARY)
    fig.tight_layout()
    curves = out / "learning_curves.png"
    fig.savefig(curves, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 4.5),
                             facecolor=SURFACE, squeeze=False)
    for ax, (variant, runs) in zip(axes.flat, groups.items()):
        steps, shares = ending_shares(runs)
        present = [e for e in ENDINGS if any(shares[e])]
        ax.stackplot(steps, *(shares[e] for e in present),
                     labels=[ENDING_LABELS[e] for e in present],
                     colors=[ENDING_COLORS[e] for e in present],
                     edgecolor=SURFACE, linewidth=1)  # a thin gap between the layers
        ax.set_ylim(0, 1)
        _style(ax, f"{variant}: how the games ended ({len(runs)} seed"
                   f"{'s' if len(runs) > 1 else ''})")
        ax.set_ylabel("share of evaluation games", color=TEXT_SECONDARY)
        ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    causes = out / "death_causes.png"
    fig.savefig(causes, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return [curves, causes]


def main(argv: list | None = None) -> list:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="a plan folder or one PPO run folder")
    parser.add_argument("--out", default=None, help="output folder (default <folder>/plots)")
    args = parser.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        parser.error("matplotlib is not installed: pip install matplotlib")
    paths = plot(Path(args.folder), args.out)
    for path in paths:
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    main()
