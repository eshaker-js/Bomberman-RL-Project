"""Evaluate a saved agent (tabular or PPO) on fixed seeds.

Run from the repository root:

    python -m rl.evaluate_agent --run runs/<tabular run>               # tiny stage
    python -m rl.evaluate_agent --run runs/<tabular run> --stage full --opponents random
    python -m rl.evaluate_agent --run runs/<ppo run> --opponents noisy25 --games 200

The agent kind is read from the run folder (``q_table.npz`` or
``model.zip``). Both play greedily: the tabular agent takes its best
Q-value (ties broken by a seeded RNG; unseen states have all-zero values),
PPO uses ``deterministic=True``.

Stages:

- ``tiny`` (tabular only, the tabular default): the training stage, single
  player on the fixed map; reports crate / death / timeout rates.
- ``full`` (the PPO default): ``BombermanGymEnv`` against three bots of the
  ladder level ``--opponents``, on seeds ``first-seed .. first-seed+games-1``
  (the same seed gives the same crates and bot seeds as
  ``agents.evaluate``). Reports win / loss / draw / timeout rates, self-kills,
  kills and crates of the agent.

Results are printed and written to a new folder
``runs/<time>_eval-<kind>-<stage>_seed<first-seed>/`` (``eval.json``,
``config.json``, ``meta.json``); the evaluated run folder is never changed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from agents.noisy_bot import LADDER
from game.environment import AGENT_ID
from rl.bomberman_env import BombermanGymEnv
from rl.config import EnvConfig, TabularConfig
from rl.run_logging import RunLog, new_run_dir, write_json
from rl.tabular_env import TabularStateTracker, TinyStageEnv
from rl.train_tabular import QTable


def detect_kind(run_dir: Path) -> str:
    if (run_dir / "q_table.npz").exists():
        return "tabular"
    if (run_dir / "model.zip").exists():
        return "ppo"
    raise FileNotFoundError(f"no q_table.npz or model.zip in {run_dir}")


class TabularPolicy:
    """Greedy policy of a saved Q-table, from the engine state."""

    def __init__(self, table: QTable, seed: int) -> None:
        self.table = table
        self.tracker = TabularStateTracker(AGENT_ID)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.tracker.reset()

    def act(self, obs, state: dict) -> int:
        return self.table.greedy(self.tracker.observe(state), self.rng)


class PPOPolicy:
    """Deterministic policy of a saved SB3 PPO model, from the observation."""

    def __init__(self, path: Path) -> None:
        from stable_baselines3 import PPO  # only needed for PPO runs
        self.model = PPO.load(path, device="cpu")

    def reset(self) -> None:
        pass

    def act(self, obs, state: dict) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def evaluate_tiny(policy: TabularPolicy, games: int, max_steps: int) -> dict:
    env = TinyStageEnv(TabularConfig(max_steps=max_steps))
    try:
        return _evaluate_tiny(env, policy, games)
    finally:
        env.close()


def _evaluate_tiny(env: TinyStageEnv, policy: TabularPolicy, games: int) -> dict:
    totals = {"games": games, "crate": 0, "death": 0, "timeout": 0, "steps": 0}
    for _ in range(games):
        env.reset()
        policy.reset()
        done, info = False, {}
        while not done:
            _, _, terminated, truncated, info = env.step(policy.act(None, env.state))
            done = terminated or truncated
        totals[info["result"]] += 1
        totals["steps"] += info["steps"]
    return totals


def evaluate_full(policy, opponents: str, games: int, first_seed: int,
                  max_steps: int) -> dict:
    env = BombermanGymEnv(opponents, EnvConfig(max_steps=max_steps))
    try:
        return _evaluate_full(env, policy, games, first_seed)
    finally:
        env.close()


def _evaluate_full(env: BombermanGymEnv, policy, games: int, first_seed: int) -> dict:
    totals = {"games": games, "win": 0, "loss": 0, "draw": 0, "timeout": 0,
              "self_kills": 0, "kills": 0, "crates": 0, "steps": 0}
    for seed in range(first_seed, first_seed + games):
        obs, _ = env.reset(seed=seed)
        policy.reset()
        done, info = False, {}
        while not done:
            obs, _, terminated, truncated, info = env.step(policy.act(obs, env.state))
            for event in info["events"]:
                if event["type"] == "death" and AGENT_ID in event["killed_by"]:
                    if event["character"] == AGENT_ID:
                        totals["self_kills"] += 1
                    else:
                        totals["kills"] += 1
                elif event["type"] == "block_destroyed" and AGENT_ID in event["destroyed_by"]:
                    totals["crates"] += 1
            done = terminated or truncated
        totals[info["outcome"] or "timeout"] += 1
        totals["steps"] += info["steps"]
    return totals


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "   n/a"


def print_results(results: dict, stage: str) -> None:
    g = results["games"]
    if stage == "tiny":
        print(f"{'crate':>7} {'death':>7} {'timeout':>8} {'steps/g':>8}")
        print(f"{_pct(results['crate'], g):>7} {_pct(results['death'], g):>7} "
              f"{_pct(results['timeout'], g):>8} {results['steps'] / g:8.1f}")
    else:
        print(f"{'win':>7} {'draw':>7} {'loss':>7} {'timeout':>8} {'selfkill':>9} "
              f"{'kills/g':>8} {'crates/g':>9} {'steps/g':>8}")
        print(f"{_pct(results['win'], g):>7} {_pct(results['draw'], g):>7} "
              f"{_pct(results['loss'], g):>7} {_pct(results['timeout'], g):>8} "
              f"{_pct(results['self_kills'], g):>9} {results['kills'] / g:8.2f} "
              f"{results['crates'] / g:9.2f} {results['steps'] / g:8.1f}")


def main(argv: list | None = None) -> tuple[Path, dict]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run folder with the saved agent")
    parser.add_argument("--stage", choices=("tiny", "full"), default=None,
                        help="default: tiny for tabular, full for PPO")
    parser.add_argument("--opponents", choices=list(LADDER), default="random",
                        help="ladder level of the three NPCs (full stage)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0,
                        help="first game seed (full stage) and tie-break seed")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="default: TabularConfig/EnvConfig max_steps (200 / 800)")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    args = parser.parse_args(argv)

    run = Path(args.run)
    kind = detect_kind(run)
    stage = args.stage or ("tiny" if kind == "tabular" else "full")
    if stage == "tiny" and kind != "tabular":
        parser.error("the tiny stage only fits tabular agents")
    max_steps = args.max_steps or (TabularConfig().max_steps if stage == "tiny"
                                   else EnvConfig().max_steps)
    if kind == "tabular":
        policy = TabularPolicy(QTable.load(run / "q_table.npz"), args.first_seed)
    else:
        policy = PPOPolicy(run / "model.zip")

    settings = {"run": str(run), "kind": kind, "stage": stage, "games": args.games,
                "first_seed": args.first_seed, "max_steps": max_steps,
                "opponents": args.opponents if stage == "full" else None}
    out = new_run_dir(f"eval-{kind}-{stage}", args.first_seed, args.runs_dir)
    with RunLog(out, f"eval-{kind}-{stage}", args.first_seed, settings,
                argv=sys.argv if argv is None else ["-m", "rl.evaluate_agent", *argv]) as log:
        if stage == "tiny":
            results = evaluate_tiny(policy, args.games, max_steps)
        else:
            results = evaluate_full(policy, args.opponents, args.games, args.first_seed,
                                    max_steps)
        write_json(out / "eval.json", {"settings": settings, "results": results})
        log.finish(results)

    where = f"vs 3 x {args.opponents}" if stage == "full" else "single-player, fixed map"
    print(f"Evaluated {kind} agent from {run} on the {stage} stage ({where}), "
          f"{args.games} games, max {max_steps} steps")
    print_results(results, stage)
    print(f"results folder: {out}")
    return out, results


if __name__ == "__main__":
    main()
