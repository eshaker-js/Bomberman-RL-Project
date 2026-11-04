"""Check that the opponent ladder really gets harder, level by level.

Run from the repository root:

    python -m agents.ladder_eval --games 200 --game-rules

Table 1, "level as agent": each ladder level plays the agent (ID 0) against
three random bots. A stronger level should win more and kill itself less.

Table 2, "rule bot vs level": the rule bot plays the agent against three
bots of each level. Stronger opponents should make the rule bot win less
(or take longer).

Both tables use the same fixed seeds, so the same command always prints the
same numbers. Without ``--game-rules`` the engine defaults are used.
"""

import argparse

from agents.match import DEFAULT_MAX_STEPS, play_game
from agents.noisy_bot import LADDER, make_bot
from game.environment import GAME_RULES, NPC_IDS


def _totals(agent_level: str, npc_level: str, games: int, first_seed: int,
            max_steps: int, rules) -> dict:
    """Agent-side totals for ``agent_level`` against three ``npc_level`` bots."""
    totals = {"win": 0, "draw": 0, "loss": 0, "timeout": 0,
              "suicides": 0, "kills": 0, "crates": 0, "steps": 0, "games": games}
    for seed in range(first_seed, first_seed + games):
        bots = {0: make_bot(agent_level)}
        bots.update({i: make_bot(npc_level) for i in NPC_IDS})
        result = play_game(bots, seed, max_steps=max_steps, rules=rules)
        totals[result.outcome] += 1
        totals["suicides"] += result.suicides.get(0, 0)
        totals["kills"] += result.kills.get(0, 0)
        totals["crates"] += result.crates.get(0, 0)
        totals["steps"] += result.steps
    return totals


def level_as_agent(games: int, first_seed: int = 0,
                   max_steps: int = DEFAULT_MAX_STEPS, rules=None) -> dict:
    """Table 1: every ladder level as the agent against three random bots."""
    return {level: _totals(level, "random", games, first_seed, max_steps, rules)
            for level in LADDER}


def rule_vs_level(games: int, first_seed: int = 0,
                  max_steps: int = DEFAULT_MAX_STEPS, rules=None) -> dict:
    """Table 2: the rule bot as the agent against three bots of every level."""
    return {level: _totals("rule", level, games, first_seed, max_steps, rules)
            for level in LADDER}


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "   n/a"


def _print_table(title: str, column: str, rows: dict) -> None:
    print(title)
    print(f"{column:9} {'win':>7} {'draw':>7} {'loss':>7} {'timeout':>8} "
          f"{'suicide':>8} {'kills/g':>8} {'crates/g':>9} {'steps/g':>8}")
    for name, t in rows.items():
        g = t["games"]
        print(f"{name:9} {_pct(t['win'], g):>7} {_pct(t['draw'], g):>7} "
              f"{_pct(t['loss'], g):>7} {_pct(t['timeout'], g):>8} "
              f"{_pct(t['suicides'], g):>8} {t['kills'] / g:8.2f} "
              f"{t['crates'] / g:9.2f} {t['steps'] / g:8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--game-rules", action="store_true",
                        help="play the real game's timing (GAME_RULES)")
    args = parser.parse_args()
    rules = dict(GAME_RULES) if args.game_rules else None

    last_seed = args.first_seed + args.games - 1
    if rules is None:
        print("Rules: engine defaults (fuse 4, flames 1)")
    else:
        print(f"Rules: GAME_RULES (fuse {rules['bomb_fuse']}, "
              f"flames {rules['flame_ticks']})")
    span = f"{args.games} games, seeds {args.first_seed}..{last_seed}"
    _print_table(f"Table 1: ladder level as agent vs 3 random bots ({span})", "agent",
                 level_as_agent(args.games, args.first_seed, args.max_steps, rules))
    print()
    _print_table(f"Table 2: rule bot as agent vs 3 bots of each level ({span})",
                 "opponents",
                 rule_vs_level(args.games, args.first_seed, args.max_steps, rules))


if __name__ == "__main__":
    main()
