"""Tabular Q-learning on the tiny stage (our own implementation, no RL library).

Run from the repository root:

    python -m rl.train_tabular --episodes 3000 --seed 0
    python -m rl.train_tabular --config my_overrides.json --seed 1

Values come from ``rl.config.TabularConfig`` (all v0), optionally overridden
by a JSON file; ``--episodes`` overrides the episode count. Each run writes
a new folder ``runs/<time>_tabular_seed<seed>/`` with ``config.json``,
``meta.json``, ``metrics.csv`` (one row per episode) and ``q_table.npz``.
Evaluate it with ``python -m rl.evaluate_agent --run <that folder>``.

Q-learning (Watkins and Dayan, 1992; Sutton and Barto, 2018, section 6.5),
per step::

    Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])

with no bootstrap (target ``r``) when the step terminated the episode, and a
normal bootstrap when it was only truncated by the step limit. Behaviour is
epsilon-greedy; epsilon falls linearly from ``epsilon_start`` to
``epsilon_end`` over the first ``epsilon_decay_fraction`` of the episodes,
then stays at ``epsilon_end``. Greedy ties are broken by the seeded RNG.
"""

import argparse
import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rl.config import TabularConfig, config_to_dict, load_config
from rl.run_logging import RunLog, new_run_dir
from rl.tabular_env import ACTIONS, TinyStageEnv

N_ACTIONS = len(ACTIONS)


class QTable:
    """Q-values of visited states only: ``state index -> array of N_ACTIONS``.

    An unvisited state has all-zero Q-values.
    """

    def __init__(self) -> None:
        self.values: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.values)

    def get(self, state: int) -> np.ndarray:
        """The Q-values of ``state`` (zeros if never visited). Do not mutate."""
        found = self.values.get(state)
        return found if found is not None else np.zeros(N_ACTIONS)

    def row(self, state: int) -> np.ndarray:
        """The mutable Q-value row of ``state``, created on first use."""
        if state not in self.values:
            self.values[state] = np.zeros(N_ACTIONS)
        return self.values[state]

    def greedy(self, state: int, rng: np.random.Generator) -> int:
        """An action with the highest Q-value; ties broken uniformly by ``rng``."""
        q = self.get(state)
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else rng.choice(best))

    def save(self, path: Path) -> None:
        states = np.array(sorted(self.values), dtype=np.int64)
        values = (np.stack([self.values[s] for s in states]) if len(states)
                  else np.zeros((0, N_ACTIONS)))
        np.savez(path, states=states, values=values)

    @classmethod
    def load(cls, path: Path) -> "QTable":
        table = cls()
        with np.load(path) as data:
            for state, row in zip(data["states"], data["values"]):
                table.values[int(state)] = row.astype(np.float64)
        return table


def q_update(table: QTable, s: int, a: int, r: float, s2: int, terminated: bool,
             alpha: float, gamma: float) -> float:
    """Apply one Q-learning update in place and return the TD error."""
    target = r if terminated else r + gamma * float(table.get(s2).max())
    row = table.row(s)
    error = target - row[a]
    row[a] += alpha * error
    return float(error)


def epsilon_at(episode: int, cfg: TabularConfig) -> float:
    """Linear decay over the first ``epsilon_decay_fraction`` of episodes, then constant."""
    decay = cfg.epsilon_decay_fraction * cfg.episodes
    if decay <= 0 or episode >= decay:
        return cfg.epsilon_end
    return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * episode / decay


# Called after every episode with (episode index, Q-table so far). It must
# not change the table; it may read it, e.g. to evaluate the greedy policy.
EpisodeHook = Callable[[int, "QTable"], None]


def train(cfg: TabularConfig, seed: int, log: RunLog | None = None,
          on_episode_end: EpisodeHook | None = None) -> tuple[QTable, dict]:
    """Train for ``cfg.episodes`` episodes. Returns the Q-table and a summary.

    ``on_episode_end`` (optional) is called after each episode with the
    episode index and the table. It does not touch the training RNG, so a
    run with a hook learns exactly the same table as a run without one.
    """
    rng = np.random.default_rng(seed)
    env = TinyStageEnv(cfg)
    try:
        return _train(env, cfg, seed, rng, log, on_episode_end)
    finally:
        env.close()


def _train(env: TinyStageEnv, cfg: TabularConfig, seed: int, rng: np.random.Generator,
           log: RunLog | None,
           on_episode_end: EpisodeHook | None = None) -> tuple[QTable, dict]:
    """The training loop of ``train`` on an env it owns."""
    env.reset(seed=seed)
    table = QTable()
    counts = {"crate": 0, "death": 0, "timeout": 0}
    last = []  # results of the most recent episodes, for the summary
    for episode in range(cfg.episodes):
        epsilon = epsilon_at(episode, cfg)
        s, _ = env.reset()
        total, done, info = 0.0, False, {}
        while not done:
            if rng.random() < epsilon:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = table.greedy(s, rng)
            s2, r, terminated, truncated, info = env.step(a)
            q_update(table, s, a, r, s2, terminated, cfg.alpha, cfg.gamma)
            total += r
            s = s2
            done = terminated or truncated
        counts[info["result"]] += 1
        last = (last + [info["result"]])[-100:]
        if log is not None:
            log.metric({"episode": episode, "return": round(total, 6),
                        "length": info["steps"], "result": info["result"],
                        "epsilon": round(epsilon, 6), "states": len(table)})
        if on_episode_end is not None:
            on_episode_end(episode, table)
    summary = {"episodes": cfg.episodes, "states_visited": len(table), **counts,
               "last100_crate_rate": last.count("crate") / max(len(last), 1)}
    return table, summary


def main(argv: list | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None,
                        help="override TabularConfig.episodes")
    parser.add_argument("--config", default=None, help="JSON file of overrides")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    args = parser.parse_args(argv)

    cfg = load_config(TabularConfig(), args.config)
    if args.episodes is not None:
        cfg = dataclasses.replace(cfg, episodes=args.episodes)
    run_dir = new_run_dir("tabular", args.seed, args.runs_dir)
    with RunLog(run_dir, "tabular", args.seed, config_to_dict(cfg),
                argv=sys.argv if argv is None else ["-m", "rl.train_tabular", *argv]) as log:
        table, summary = train(cfg, args.seed, log)
        table.save(run_dir / "q_table.npz")
        log.finish(summary)
    print(f"run folder: {run_dir}")
    print("summary:", summary)
    return run_dir


if __name__ == "__main__":
    main()
