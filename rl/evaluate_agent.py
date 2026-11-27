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
  kills and crates of the agent, and (added in Task 8) the game length by
  outcome, buffs collected and the cause of every agent death (``GameTally``).

Results are printed and written to a new folder
``runs/<time>_eval-<kind>-<stage>_seed<first-seed>/`` (``eval.json``,
``config.json``, ``meta.json``); the evaluated run folder is never changed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from agents.noisy_bot import LADDER
from game.environment import AGENT_ID, BUFF_RULES
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


class ModelPolicy:
    """Policy of an SB3 model in memory, from the observation.

    ``deterministic=True`` (the default) takes the most likely action;
    ``False`` samples from the policy, using torch's global RNG.

    ``action_mask`` (Task 13): a safety filter mode of ``rl.safety`` for a
    ``MaskablePPO`` model; the mask is computed from ``state`` and passed to
    ``predict``. ``None`` or ``"off"`` (default): exactly the call before.
    """

    def __init__(self, model, deterministic: bool = True,
                 action_mask: str | None = None) -> None:
        self.model = model
        self.deterministic = deterministic
        self.action_mask = None if action_mask == "off" else action_mask

    def reset(self) -> None:
        pass

    def act(self, obs, state: dict) -> int:
        if self.action_mask is None:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        else:
            from rl.safety import action_mask
            masks = action_mask(state, AGENT_ID, self.action_mask)
            action, _ = self.model.predict(obs, deterministic=self.deterministic,
                                           action_masks=masks)
        return int(action)


class PPOPolicy(ModelPolicy):
    """Policy of a saved SB3 PPO model (deterministic unless asked otherwise)."""

    def __init__(self, path: Path, deterministic: bool = True) -> None:
        from stable_baselines3 import PPO  # only needed for PPO runs
        super().__init__(PPO.load(path, device="cpu"), deterministic)


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


BUFF_NAMES = tuple(buff.name for buff in BUFF_RULES.types)  # range, bomb_count, burn

# Cause of an agent death. "fresh": the agent's cell was burned on that very
# step (it is in ``state["explosion"]``); "older": a flame of an earlier
# step still burning. The owners are the death event's ``killed_by``, the
# owners of the blast that last burned the cell (a fresh blast replaces the
# owners of an older flame). "own": only the agent; "opponent": the agent is
# not among them; "shared": the agent and at least one opponent. A bomb set
# off in a chain reaction credits only its own owner (the engine's rule), so
# an opponent's blast that sets off my bomb, which then kills me, is "own".
DEATH_CAUSES = ("own_fresh", "opponent_fresh", "shared_fresh",
                "own_older", "opponent_older", "shared_older")
# Game length by outcome: summed steps of the games that ended in the agent's
# death (loss or draw: time to death), in a win (time to win) and in a
# timeout. (The agent's death ends the game, so "steps survived" would always
# equal the game length; these split it instead.)
LENGTH_KEYS = {"loss": "steps_death", "draw": "steps_death",
               "win": "steps_win", "timeout": "steps_timeout"}
# Keys added to ``evaluate_full``'s results in Task 8 (old keys unchanged).
NEW_FULL_KEYS = (
    "steps_death", "steps_win", "steps_timeout", "buffs_picked", "buffs_raised",
    *(f"buffs_picked_{name}" for name in BUFF_NAMES),
    *(f"buffs_raised_{name}" for name in BUFF_NAMES),
    *(f"death_{cause}" for cause in DEATH_CAUSES), "death_unknown", "death_total",
)


def classify_death(event: dict, state: dict, me: int = AGENT_ID) -> str:
    """The ``DEATH_CAUSES`` entry of ``me``'s death ``event`` (``"unknown"`` if none fits).

    ``state`` is the state right after the step of the death (its
    ``explosion`` and ``me``'s position, which a death never changes).
    """
    owners = tuple(event["killed_by"])
    character = state["characters"][me]
    fresh = (character["x"], character["y"]) in {tuple(c) for c in state["explosion"]}
    if not owners:
        return "unknown"
    if owners == (me,):
        who = "own"
    elif me in owners:
        who = "shared"
    else:
        who = "opponent"
    return f"{who}_{'fresh' if fresh else 'older'}"


class GameTally:
    """What one character did in one game, counted step by step.

    Call ``observe(prev, state)`` after every step with the state before and
    after it, then ``finish(outcome, steps)``. ``died_at`` is the step on
    which the character died (``None`` if it did not). A buff pickup "raised" a level
    when the level after it is higher than before (a pickup at the max is
    consumed and raises nothing).
    """

    def __init__(self, me: int = AGENT_ID) -> None:
        self.me = me
        self.self_kill = 0
        self.kills = 0
        self.crates = 0
        self.picked = dict.fromkeys(BUFF_NAMES, 0)
        self.raised = dict.fromkeys(BUFF_NAMES, 0)
        self.died_at: int | None = None
        self.death_cause: str | None = None

    def observe(self, prev: dict, state: dict) -> None:
        me = self.me
        for event in state["events"]:
            kind = event["type"]
            if kind == "death":
                if me in event["killed_by"]:
                    if event["character"] == me:
                        self.self_kill += 1
                    else:
                        self.kills += 1
                if event["character"] == me:
                    self.died_at = state["steps"]
                    self.death_cause = classify_death(event, state, me)
            elif kind == "block_destroyed" and me in event["destroyed_by"]:
                self.crates += 1
            elif kind == "buff_picked" and event["character"] == me:
                name = event["buff"]
                self.picked[name] = self.picked.get(name, 0) + 1
                before = prev["characters"][me].get(name, event["level"])
                if event["level"] > before:
                    self.raised[name] = self.raised.get(name, 0) + 1

    def finish(self, outcome: str | None, steps: int) -> dict:
        """One row for this game (``outcome`` ``None`` counts as ``"timeout"``)."""
        row = {
            "outcome": outcome or "timeout",
            "steps": steps,
            "self_kill": self.self_kill,
            "kills": self.kills,
            "crates": self.crates,
            "buffs_picked": sum(self.picked.values()),
            "buffs_raised": sum(self.raised.values()),
        }
        for name in BUFF_NAMES:
            row[f"buffs_picked_{name}"] = self.picked[name]
            row[f"buffs_raised_{name}"] = self.raised[name]
        row["death_cause"] = self.death_cause or ""
        return row


def evaluate_full(policy, opponents: str, games: int, first_seed: int,
                  max_steps: int, per_game: list | None = None) -> dict:
    env = BombermanGymEnv(opponents, EnvConfig(max_steps=max_steps))
    try:
        return _evaluate_full(env, policy, games, first_seed, per_game)
    finally:
        env.close()


def _evaluate_full(env: BombermanGymEnv, policy, games: int, first_seed: int,
                   per_game: list | None = None) -> dict:
    """Totals over the games on seeds ``first_seed ..``; ``per_game`` gets one row per game."""
    return evaluate_seeds(env, policy, range(first_seed, first_seed + games), per_game)


def evaluate_seeds(env: BombermanGymEnv, policy, seeds, per_game: list | None = None) -> dict:
    """Play one game per seed in ``seeds`` and return the totals.

    The first keys are those ``evaluate_full`` always had (same values);
    ``NEW_FULL_KEYS`` follow. ``per_game`` (a list) gets one ``GameTally``
    row per game, with its ``seed``.
    """
    seeds = list(seeds)
    totals = {"games": len(seeds), "win": 0, "loss": 0, "draw": 0, "timeout": 0,
              "self_kills": 0, "kills": 0, "crates": 0, "steps": 0}
    totals.update(dict.fromkeys(NEW_FULL_KEYS, 0))
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        tally = GameTally(AGENT_ID)
        done, info = False, {}
        while not done:
            prev = env.state
            obs, _, terminated, truncated, info = env.step(policy.act(obs, prev))
            tally.observe(prev, env.state)
            done = terminated or truncated
        row = tally.finish(info["outcome"], info["steps"])
        add_game(totals, row)
        if per_game is not None:
            per_game.append({"seed": seed, **row})
    return totals


def add_game(totals: dict, row: dict) -> None:
    """Add one ``GameTally`` row to ``evaluate_seeds``-style totals."""
    totals[row["outcome"]] += 1
    totals["self_kills"] += row["self_kill"]
    totals["kills"] += row["kills"]
    totals["crates"] += row["crates"]
    totals["steps"] += row["steps"]
    totals[LENGTH_KEYS[row["outcome"]]] += row["steps"]
    for key in ("buffs_picked", "buffs_raised",
                *(f"buffs_picked_{n}" for n in BUFF_NAMES),
                *(f"buffs_raised_{n}" for n in BUFF_NAMES)):
        totals[key] += row[key]
    cause = row["death_cause"]
    if cause:
        key = f"death_{cause}"
        totals[key if key in totals else "death_unknown"] += 1
        totals["death_total"] += 1


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
