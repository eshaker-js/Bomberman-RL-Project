"""Learning-curve figures for a tabular sweep (``rl.sweep_tabular``).

Run from the repository root (needs matplotlib, ``pip install matplotlib``):

    python -m rl.plot_tabular runs/<time>_sweep-tabular_seed0

Writes ``plots/<param>.png`` into the sweep folder, one figure per swept
parameter, each with two panels over training episodes:

- left: the share of seeds whose greedy policy (no exploration) solves the
  tiny stage at each checkpoint, from ``checkpoints.csv``
- right: the training success rate (epsilon-greedy, so it includes the
  exploration's mistakes), a moving average over ``--window`` episodes from
  each run's ``metrics.csv``, mean over seeds with a band of one standard
  deviation

One line per value, in the order of the plan. Only reads the sweep folder.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_WINDOW = 100

# Categorical slots 1-8 of the reference data-viz palette (light mode), used
# in this fixed order, never cycled.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sweep(sweep_dir: Path) -> dict:
    """The sweep's plan, runs and checkpoints."""
    with open(sweep_dir / "plan.json", encoding="utf-8") as handle:
        plan = json.load(handle)
    return {
        "plan": plan,
        "summary": read_csv(sweep_dir / "summary.csv"),
        "checkpoints": read_csv(sweep_dir / "checkpoints.csv"),
    }


def groups_of(plan: dict) -> dict:
    """``param -> [(value, variant name), ...]`` in the order of the plan."""
    names = {}
    for variant in plan["variants"]:
        for param, value in variant["groups"]:
            names[(param, value)] = variant["name"]
    return {param: [(value, names[(param, value)]) for value in values
                    if (param, value) in names]
            for param, values in plan["plan"].items()}


def greedy_curve(checkpoints: list, variant: str) -> tuple:
    """Episodes and share of seeds whose greedy policy solved the stage there."""
    by_episode = {}
    for row in checkpoints:
        if row["variant"] == variant:
            by_episode.setdefault(int(row["episodes"]), []).append(
                row["result"] == "crate")
    episodes = sorted(by_episode)
    share = [sum(by_episode[e]) / len(by_episode[e]) for e in episodes]
    return np.array(episodes), np.array(share)


def moving_success(results: list, window: int) -> np.ndarray:
    """Moving average of crate (1) vs anything else (0) over ``window`` episodes.

    The first ``window - 1`` episodes average over what exists so far.
    """
    hits = np.array([r == "crate" for r in results], dtype=float)
    sums = np.cumsum(hits)
    counts = np.minimum(np.arange(1, len(hits) + 1), window)
    shifted = np.concatenate([np.zeros(window), sums])[:len(hits)]
    return (sums - shifted) / counts


def training_curve(sweep_dir: Path, summary: list, variant: str, window: int) -> tuple:
    """Mean and std over seeds of the moving training success rate."""
    curves = []
    for row in summary:
        if row["variant"] == variant:
            metrics = read_csv(sweep_dir / row["run_dir"] / "metrics.csv")
            curves.append(moving_success([m["result"] for m in metrics], window))
    if not curves:
        return np.array([]), np.array([]), np.array([])
    stacked = np.vstack(curves)  # every seed of one variant has the same length
    episodes = np.arange(1, stacked.shape[1] + 1)
    return episodes, stacked.mean(axis=0), stacked.std(axis=0)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("training episodes", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_param(sweep_dir: Path, data: dict, param: str, members: list,
               window: int, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(members) > len(SERIES_COLORS):
        raise ValueError(f"{param}: at most {len(SERIES_COLORS)} values per figure")
    seeds = len(data["plan"]["seeds"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for color, (value, variant) in zip(SERIES_COLORS, members):
        label = f"{param} = {value}"
        x, share = greedy_curve(data["checkpoints"], variant)
        left.step(x, share, where="post", color=color, linewidth=2, label=label)
        x, mean, std = training_curve(sweep_dir, data["summary"], variant, window)
        right.plot(x, mean, color=color, linewidth=2, label=label)
        right.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                           color=color, alpha=0.12, linewidth=0)
    _style(left, f"Greedy policy solves the stage ({seeds} seeds)",
           "share of seeds")
    _style(right, f"Training success, moving average of {window} episodes",
           "success rate (mean ± 1 std)")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(members),
               frameon=False, labelcolor=TEXT_PRIMARY)
    fig.suptitle(f"Tabular Q-learning, tiny stage: effect of {param}",
                 color=TEXT_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: list | None = None) -> list:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep_dir", help="a runs/<time>_sweep-tabular_seed<n> folder")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="moving-average window in episodes (default 100)")
    args = parser.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        parser.error("matplotlib is not installed: pip install matplotlib")
    if args.window < 1:
        parser.error("--window must be at least 1")

    sweep_dir = Path(args.sweep_dir)
    data = load_sweep(sweep_dir)
    written = []
    for param, members in groups_of(data["plan"]).items():
        out = sweep_dir / "plots" / f"{param}.png"
        written.append(plot_param(sweep_dir, data, param, members, args.window, out))
        print(f"wrote {out}")
    return written


if __name__ == "__main__":
    main()
