"""Headless checks that the rule bot plays by the env's own timing rules. Pygame is not required.

The rule bot reads ``bomb_fuse``, ``blast_range`` and ``flame_ticks`` from
``state["rules"]`` and treats burning cells (``state["flames"]``) as deadly
for as long as they burn, so it never moves into or stays on a cell that
will still be burning. The original bot tests in ``test_bots.py`` cover the
default rules and stay unchanged.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import RandomBot, RuleBot  # noqa: E402
from agents.match import bot_seed  # noqa: E402
from agents.rule_bot import (  # noqa: E402
    MOVES,
    blast_cells,
    danger_times,
    plan_bombs,
    steps_to_safety,
)
from game.environment import GAME_RULES, Action, BombermanEnv  # noqa: E402

W, B = Action.WAIT, Action.BOMB


def burning(state: dict) -> dict:
    return {(f["x"], f["y"]): f["ticks_left"] for f in state["flames"]}


def next_cell(state: dict, me: int, action: Action) -> tuple[int, int]:
    """Where ``action`` would take character ``me`` if the cell were free."""
    c = state["characters"][me]
    dx, dy = MOVES.get(action, (0, 0))
    return (c["x"] + dx, c["y"] + dy)


class TestHelpers(unittest.TestCase):
    def test_danger_lasts_flame_ticks_actions(self):
        grid = BombermanEnv().get_state()["grid"]
        planned = plan_bombs(grid, [(1, 1, 3, 1)])
        self.assertEqual(danger_times(planned)[(2, 1)], {3})
        self.assertEqual(danger_times(planned, flame_ticks=4)[(2, 1)], {3, 4, 5, 6})

    def test_burning_cells_are_deadly_until_they_go_out(self):
        danger = danger_times([], flame_ticks=4, burning={(2, 1): 3, (3, 1): 1})
        self.assertEqual(danger, {(2, 1): {1, 2}})  # (3, 1) is out by the next action

    def test_blast_range_parameter(self):
        grid = BombermanEnv().get_state()["grid"]
        self.assertNotIn((4, 1), blast_cells(grid, 1, 1, set()))
        self.assertIn((4, 1), blast_cells(grid, 1, 1, set(), blast_range=3))
        planned = plan_bombs(grid, [(1, 1, 2, 1)], blast_range=3)
        self.assertIn((4, 1), planned[0].blast)

    def test_no_safe_way_through_a_burning_cell(self):
        grid = BombermanEnv().get_state()["grid"]
        # Standing at (2, 1); a bomb at (4, 1) burns (2, 1)..(4, 1) on action 2.
        # Left (1, 1) is still burning for 3 more actions, so moving there kills.
        planned = plan_bombs(grid, [(4, 1, 2, 1)])
        burning_cells = {(1, 1): 4}
        self.assertIsNone(
            steps_to_safety(grid, (2, 1), planned, Action.LEFT, 4, burning_cells))
        self.assertIsNotNone(steps_to_safety(grid, (2, 1), planned, Action.LEFT))


class TestReadsRules(unittest.TestCase):
    def test_possible_bombs_use_the_env_fuse(self):
        for rules, fuse in (({}, 4), (dict(GAME_RULES), 9)):
            state = BombermanEnv(with_npcs=True, **rules).get_state()
            possible = RuleBot()._possible_bombs(state, 0)
            self.assertEqual({b[2] for b in possible}, {1 + fuse})

    def test_act_uses_the_blast_range_in_state(self):
        """At the fixed spawn (1, 1) the nearest crate is (5, 1): out of reach with
        range 2, in reach with range 4, and the bot can still escape via (3, 2)."""
        state = BombermanEnv().get_state()
        self.assertNotEqual(RuleBot(0).act(state, 0), B)
        state["rules"]["blast_range"] = 4  # the state alone says the range is 4
        self.assertEqual(RuleBot(0).act(state, 0), B)


class TestAvoidsLingeringFlames(unittest.TestCase):
    def test_never_steps_into_or_stays_on_a_burning_cell(self):
        """Over real games with GAME_RULES, no rule bot's chosen action ends on a cell
        that is still burning on the next action, whenever any safe action exists."""
        checked = 0
        for seed in range(12):
            env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
            state = env.reset(seed=seed)
            bots = {i: RuleBot() for i in range(4)}
            for i, bot in bots.items():
                bot.reset(bot_seed(seed, i))
            for _ in range(120):
                actions = {}
                hot = {cell for cell, left in burning(state).items() if left > 1}
                for c in state["characters"]:
                    if c["dead"]:
                        continue
                    action = bots[c["id"]].act(state, c["id"])
                    actions[c["id"]] = action
                    if hot and action in MOVES:
                        self.assertNotIn(next_cell(state, c["id"], action), hot)
                        checked += 1
                    if hot and action in (W, B):
                        self.assertNotIn(next_cell(state, c["id"], W), hot)
                state = env.step(actions.pop(0), actions)
                if state["outcome"]:
                    break
        self.assertGreater(checked, 100)

    def test_rule_bot_agent_survives_random_bots_with_game_rules(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
                state = env.reset(seed=seed)
                bots = {0: RuleBot(), 1: RandomBot(), 2: RandomBot(), 3: RandomBot()}
                for i, bot in bots.items():
                    bot.reset(bot_seed(seed, i))
                for _ in range(400):
                    actions = {c["id"]: bots[c["id"]].act(state, c["id"])
                               for c in state["characters"] if not c["dead"]}
                    state = env.step(actions.pop(0), actions)
                    if state["outcome"]:
                        break
                self.assertEqual(state["outcome"], "win")

    def test_no_rule_bot_dies_in_a_lingering_flame(self):
        """Four rule bots with GAME_RULES: nobody walks into a flame from an earlier tick."""
        for seed in range(8):
            env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
            state = env.reset(seed=seed)
            bots = {i: RuleBot() for i in range(4)}
            for i, bot in bots.items():
                bot.reset(bot_seed(seed, i))
            for _ in range(150):
                actions = {c["id"]: bots[c["id"]].act(state, c["id"])
                           for c in state["characters"] if not c["dead"]}
                state = env.step(actions.pop(0), actions)
                for event in state["events"]:
                    if event["type"] == "death":
                        c = state["characters"][event["character"]]
                        self.assertIn((c["x"], c["y"]), state["explosion"],
                                      f"seed {seed}: lingering death {event}")
                if state["outcome"]:
                    break


if __name__ == "__main__":
    unittest.main()
