"""The buff evaluation tool and the 800-step evaluation limit (Task 7). Headless."""

import contextlib
import io
import unittest
from unittest import mock

from agents import buff_eval
from agents.match import DEFAULT_MAX_STEPS, play_game
from game.environment import GAME_RULES


class BuffEvalTest(unittest.TestCase):
    def test_evaluation_limit_matches_training(self):
        self.assertEqual(DEFAULT_MAX_STEPS, 800)

    def test_rows(self):
        self.assertEqual([buff_eval.row_name(r) for r in buff_eval.ROWS],
                         ["no buff", "range 3", "range 4", "range 5", "bomb_count 2",
                          "bomb_count 3", "burn 1", "burn 2"])

    def test_start_buff_reaches_the_engine(self):
        with mock.patch.object(buff_eval, "play_game", wraps=play_game) as spy:
            buff_eval.evaluate_row(("range", 4), "random", 2, max_steps=5)
            buff_eval.evaluate_row(None, "random", 1, max_steps=5)
        rules = [c.kwargs["rules"] for c in spy.call_args_list]
        self.assertEqual(rules[0], {**GAME_RULES, "start_buffs": {0: {"range": 4}}})
        self.assertEqual(rules[2], dict(GAME_RULES))
        self.assertNotIn("start_buffs", GAME_RULES)

    def test_totals_add_up_and_repeat(self):
        a = buff_eval.evaluate_row(("burn", 2), "noisy25", 3, max_steps=60)
        b = buff_eval.evaluate_row(("burn", 2), "noisy25", 3, max_steps=60)
        self.assertEqual(a, b)
        self.assertEqual(a["win"] + a["loss"] + a["draw"] + a["timeout"], 3)
        self.assertLessEqual(a["survived"], 3)
        self.assertGreaterEqual(a["survived"], a["win"] + a["timeout"])

    def test_main_prints_every_row(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            buff_eval.main(["--games", "1", "--max-steps", "20", "--opponents", "random"])
        text = out.getvalue()
        self.assertIn("buffs on", text)
        self.assertIn("survived", text)
        for row in buff_eval.ROWS:
            self.assertIn(buff_eval.row_name(row), text)


if __name__ == "__main__":
    unittest.main()
