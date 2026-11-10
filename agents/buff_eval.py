"""Measure what each buff is worth to the rule bot.

Run from the repository root:

    python -m agents.buff_eval --games 200
    python -m agents.buff_eval --games 200 --opponents noisy25

Every row plays ``--games`` seeded games with ``GAME_RULES`` (random crates,
buffs on): the rule bot as character 0 **starts with one buff at one level**
(``start_buffs``) against three bots of the ``--opponents`` ladder level that
start without buffs. The first row is the baseline without a start buff.
Every character can still pick up buffs during the game.

Reported per row (agent side): win, loss, draw and timeout rates, survival
(alive at the end of the game, so timeouts count as survived), kills and
self-kills per game, crates and steps per game.

Seeds are ``first-seed .. first-seed+games-1`` in every row, so all rows play
the same crate layouts and bot seeds; the same command always prints the
same numbers.
"""

import argparse

from agents.match import DEFAULT_MAX_STEPS, play_game
from agents.noisy_bot import LADDER, make_bot
from game.environment import AGENT_ID, GAME_RULES, NPC_IDS

# (buff name, start level) of every row; None is the baseline. The levels
# above each buff's start level, up to its max in GAME_RULES.
ROWS = (None,) + tuple(
    (buff.name, level)
    for buff in GAME_RULES["buffs"].types
    for level in range(buff.start + 1, buff.max + 1)
)
DEFAULT_OPPONENTS = ("rule", "noisy25")


def row_name(row) -> str:
    return "no buff" if row is None else f"{row[0]} {row[1]}"


def evaluate_row(row, opponents: str, games: int, first_seed: int = 0,
                 max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Agent-side totals for the rule bot with the start buff ``row``."""
    rules = dict(GAME_RULES)
    if row is not None:
        rules["start_buffs"] = {AGENT_ID: {row[0]: row[1]}}
    totals = {"games": games, "win": 0, "loss": 0, "draw": 0, "timeout": 0,
              "survived": 0, "kills": 0, "self_kills": 0, "crates": 0, "steps": 0}
    for seed in range(first_seed, first_seed + games):
        bots = {AGENT_ID: make_bot("rule")}
        bots.update({i: make_bot(opponents) for i in NPC_IDS})
        result = play_game(bots, seed, max_steps=max_steps, rules=rules)
        totals[result.outcome] += 1
        totals["survived"] += AGENT_ID not in result.death_step
        totals["kills"] += result.kills.get(AGENT_ID, 0)
        totals["self_kills"] += result.suicides.get(AGENT_ID, 0)
        totals["crates"] += result.crates.get(AGENT_ID, 0)
        totals["steps"] += result.steps
    return totals


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "   n/a"


def print_table(opponents: str, rows: dict, span: str) -> None:
    print(f"Rule bot as agent with a start buff vs 3 {opponents} bots ({span})")
    print(f"{'start buff':11} {'win':>7} {'loss':>7} {'draw':>7} {'timeout':>8} "
          f"{'survived':>9} {'kills/g':>8} {'selfkill':>9} {'crates/g':>9} {'steps/g':>8}")
    for name, t in rows.items():
        g = t["games"]
        print(f"{name:11} {_pct(t['win'], g):>7} {_pct(t['loss'], g):>7} "
              f"{_pct(t['draw'], g):>7} {_pct(t['timeout'], g):>8} "
              f"{_pct(t['survived'], g):>9} {t['kills'] / g:8.2f} "
              f"{_pct(t['self_kills'], g):>9} {t['crates'] / g:9.2f} {t['steps'] / g:8.1f}")


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS),
                        choices=list(LADDER), help="ladder levels of the three NPCs")
    args = parser.parse_args(argv)

    rules = GAME_RULES
    print(f"Rules: GAME_RULES (fuse {rules['bomb_fuse']}, flames {rules['flame_ticks']}, "
          f"buffs on), max {args.max_steps} steps")
    span = (f"{args.games} games per row, seeds {args.first_seed}.."
            f"{args.first_seed + args.games - 1}")
    for n, opponents in enumerate(args.opponents):
        if n:
            print()
        rows = {row_name(row): evaluate_row(row, opponents, args.games, args.first_seed,
                                            args.max_steps)
                for row in ROWS}
        print_table(opponents, rows, span)


if __name__ == "__main__":
    main()
