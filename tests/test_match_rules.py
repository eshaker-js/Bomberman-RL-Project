"""Evaluation can play the real game's timing (Task 6, stage 1).

``play_game(..., rules=None)`` must behave exactly as before, and
``rules=GAME_RULES`` must reach the engine. Pygame is not required.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import evaluate  # noqa: E402
from agents.evaluate import agent_seat, mixed_table  # noqa: E402
from agents.match import play_game  # noqa: E402
from agents.random_bot import RandomBot  # noqa: E402
from agents.rule_bot import RuleBot  # noqa: E402
from game.environment import BLAST_RANGE, GAME_RULES, Action  # noqa: E402

DEFAULT_RULES = {"bomb_fuse": 4, "blast_range": BLAST_RANGE, "flame_ticks": 1}
REAL_RULES = {"bomb_fuse": GAME_RULES["bomb_fuse"], "blast_range": BLAST_RANGE, "flame_ticks": GAME_RULES["flame_ticks"]}


class SpyBot:
    """Waits every turn and remembers the ``rules`` it saw in the state."""

    def __init__(self) -> None:
        self.seen = []

    def reset(self, seed=None) -> None:
        self.seen = []

    def act(self, state: dict, me: int) -> Action:
        self.seen.append(dict(state["rules"]))
        return Action.WAIT


def _table(seed: int) -> dict:
    return {0: RuleBot(), 1: RandomBot(), 2: RuleBot(), 3: RandomBot()}


class PlayGameRulesTest(unittest.TestCase):
    def test_none_is_the_old_default(self):
        for seed in range(6):
            old = play_game(_table(seed), seed)
            new = play_game(_table(seed), seed, rules=None)
            self.assertEqual(old, new, f"seed {seed}")

    def test_default_rules_reach_the_bots(self):
        spy = SpyBot()
        bots = {0: spy, 1: RandomBot(), 2: RandomBot(), 3: RandomBot()}
        play_game(bots, 3, max_steps=5)
        self.assertTrue(spy.seen)
        for rules in spy.seen:
            self.assertEqual(rules, DEFAULT_RULES)

    def test_game_rules_reach_the_engine(self):
        spy = SpyBot()
        bots = {0: spy, 1: RandomBot(), 2: RandomBot(), 3: RandomBot()}
        play_game(bots, 3, max_steps=5, rules=GAME_RULES)
        self.assertTrue(spy.seen)
        for rules in spy.seen:
            self.assertEqual(rules, REAL_RULES)

    def test_game_rules_change_some_games(self):
        """The longer fuse must actually change play on some seed."""
        differs = any(
            play_game(_table(s), s) != play_game(_table(s), s, rules=GAME_RULES)
            for s in range(10)
        )
        self.assertTrue(differs)

    def test_game_rules_are_reproducible(self):
        a = play_game(_table(5), 5, rules=GAME_RULES)
        b = play_game(_table(5), 5, rules=GAME_RULES)
        self.assertEqual(a, b)

    def test_game_rules_mapping_is_not_changed(self):
        play_game(_table(1), 1, max_steps=3, rules=GAME_RULES)
        self.assertEqual(dict(GAME_RULES), {"bomb_fuse": 4, "flame_ticks": 4, "buffs": GAME_RULES["buffs"]})


class EvaluateRulesTest(unittest.TestCase):
    def test_agent_seat_default_is_unchanged(self):
        self.assertEqual(agent_seat("rule", 4), agent_seat("rule", 4, rules=None))

    def test_mixed_table_default_is_unchanged(self):
        self.assertEqual(mixed_table(4), mixed_table(4, rules=None))

    def test_agent_seat_passes_rules(self):
        with mock.patch.object(evaluate, "play_game", wraps=play_game) as spy:
            agent_seat("random", 2, rules=GAME_RULES)
        self.assertEqual(spy.call_count, 2)
        for call in spy.call_args_list:
            self.assertIs(call.kwargs["rules"], GAME_RULES)

    def test_mixed_table_passes_rules(self):
        with mock.patch.object(evaluate, "play_game", wraps=play_game) as spy:
            mixed_table(2, rules=GAME_RULES)
        self.assertEqual(spy.call_count, 2)
        for call in spy.call_args_list:
            self.assertIs(call.kwargs["rules"], GAME_RULES)

    def _run_main(self, *args) -> tuple[str, list]:
        out = io.StringIO()
        argv = ["evaluate", "--games", "2", *args]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(evaluate, "play_game", wraps=play_game) as spy, \
                contextlib.redirect_stdout(out):
            evaluate.main()
        return out.getvalue(), [c.kwargs["rules"] for c in spy.call_args_list]

    def test_main_without_flag_uses_defaults(self):
        text, rules = self._run_main()
        self.assertIn("engine defaults", text)
        self.assertTrue(rules)
        self.assertTrue(all(r is None for r in rules))

    def test_main_with_flag_uses_game_rules(self):
        text, rules = self._run_main("--game-rules")
        self.assertIn(f"GAME_RULES (fuse {GAME_RULES['bomb_fuse']}, flames {GAME_RULES['flame_ticks']})", text)
        self.assertIn("Experiment A", text)
        self.assertIn("Experiment B", text)
        self.assertTrue(rules)
        self.assertTrue(all(r == dict(GAME_RULES) for r in rules))


if __name__ == "__main__":
    unittest.main()
