"""Compare the baseline bots over many seeded games.

Run from the repository root:

    python -m agents.evaluate
    python -m agents.evaluate --games 500 --first-seed 0

Experiment A, "agent seat": one bot type plays the agent against three
random NPCs. Reports the agent's win/draw/loss/timeout rates and what it did.

Experiment B, "mixed table": two rule bots and two random bots share a game,
with the seats rotated through all six arrangements. Reports per bot type how
often it died, how often it killed itself, its kills and crates.

Every game uses a fixed seed, so the same command always prints the same numbers.
"""

import argparse
from itertools import combinations

from agents.match import DEFAULT_MAX_STEPS, play_game
from agents.random_bot import RandomBot
from agents.rule_bot import RuleBot

BOT_TYPES = {"random": RandomBot, "rule": RuleBot}
SEATS = (0, 1, 2, 3)


def agent_seat(agent_type: str, games: int, first_seed: int = 0,
               max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Experiment A: ``agent_type`` as the agent against three random NPCs."""
    totals = {"win": 0, "draw": 0, "loss": 0, "timeout": 0,
              "suicides": 0, "kills": 0, "crates": 0, "steps": 0}
    for seed in range(first_seed, first_seed + games):
        bots = {0: BOT_TYPES[agent_type]()}
        bots.update({i: RandomBot() for i in (1, 2, 3)})
        result = play_game(bots, seed, max_steps=max_steps)
        totals[result.outcome] += 1
        totals["suicides"] += result.suicides.get(0, 0)
        totals["kills"] += result.kills.get(0, 0)
        totals["crates"] += result.crates.get(0, 0)
        totals["steps"] += result.steps
    totals["games"] = games
    return totals


def mixed_table(games: int, first_seed: int = 0,
                max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Experiment B: two rule bots and two random bots, seats rotated."""
    arrangements = list(combinations(SEATS, 2))  # seats that get a rule bot
    stats = {name: {"seats": 0, "deaths": 0, "suicides": 0, "kills": 0, "crates": 0}
             for name in BOT_TYPES}
    for n, seed in enumerate(range(first_seed, first_seed + games)):
        rule_seats = arrangements[n % len(arrangements)]
        kind = {seat: "rule" if seat in rule_seats else "random" for seat in SEATS}
        bots = {seat: BOT_TYPES[kind[seat]]() for seat in SEATS}
        result = play_game(bots, seed, max_steps=max_steps)
        for seat in SEATS:
            s = stats[kind[seat]]
            s["seats"] += 1
            s["deaths"] += seat in result.death_step
            s["suicides"] += result.suicides.get(seat, 0)
            s["kills"] += result.kills.get(seat, 0)
            s["crates"] += result.crates.get(seat, 0)
    return stats


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "   n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    last_seed = args.first_seed + args.games - 1
    print(f"Experiment A: agent seat vs 3 random NPCs "
          f"({args.games} games, seeds {args.first_seed}..{last_seed})")
    print(f"{'agent':8} {'win':>7} {'draw':>7} {'loss':>7} {'timeout':>8} "
          f"{'suicide':>8} {'kills/g':>8} {'crates/g':>9} {'steps/g':>8}")
    for name in BOT_TYPES:
        t = agent_seat(name, args.games, args.first_seed, args.max_steps)
        g = t["games"]
        print(f"{name:8} {_pct(t['win'], g):>7} {_pct(t['draw'], g):>7} "
              f"{_pct(t['loss'], g):>7} {_pct(t['timeout'], g):>8} "
              f"{_pct(t['suicides'], g):>8} {t['kills'] / g:8.2f} "
              f"{t['crates'] / g:9.2f} {t['steps'] / g:8.1f}")

    print()
    print(f"Experiment B: 2 rule + 2 random per game, seats rotated "
          f"({args.games} games)")
    print(f"{'bot':8} {'died':>7} {'suicide':>8} {'kills/g':>8} {'crates/g':>9}")
    for name, s in mixed_table(args.games, args.first_seed, args.max_steps).items():
        n = s["seats"]
        print(f"{name:8} {_pct(s['deaths'], n):>7} {_pct(s['suicides'], n):>8} "
              f"{s['kills'] / n:8.2f} {s['crates'] / n:9.2f}")


if __name__ == "__main__":
    main()
