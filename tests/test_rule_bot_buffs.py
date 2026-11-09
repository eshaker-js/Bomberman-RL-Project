"""The rule bot with buffs: per-bomb range and burn, several bombs (Task 7).

Headless; pygame is not required. Fixed map row 1 is ``#P...B.B..#``.
"""

import copy
import random
import unittest

from agents.match import bot_seed
from agents.random_bot import RandomBot
from agents.rule_bot import (
    RuleBot,
    bombs_from_state,
    danger_times,
    levels_of,
    plan_bombs,
)
from game.environment import (
    BUFF_RULES,
    GAME_RULES,
    Action,
    Bomb,
    BombermanEnv,
    Cell,
)

W, B = Action.WAIT, Action.BOMB
TIMING = {"bomb_fuse": GAME_RULES["bomb_fuse"], "flame_ticks": GAME_RULES["flame_ticks"]}


def single(start=None) -> BombermanEnv:
    """Single player on the fixed map with buffs."""
    return BombermanEnv(buffs=BUFF_RULES, start_buffs=start, **TIMING)


def moved_to(env, cell):
    env.player_x, env.player_y = cell
    return env.get_state()


class TestPlanning(unittest.TestCase):
    def test_bombs_from_state_reads_range_and_burn(self):
        env = single()
        env.bomb = Bomb(1, 1, 3, range=4, burn=1)
        state = env.get_state()
        self.assertEqual(bombs_from_state(state), [(1, 1, 3, 1, 4, 1)])
        planned = plan_bombs(state["grid"], bombs_from_state(state))
        self.assertIn((1, 4), planned[0].blast)  # range 4 down to the crate at (1, 5)
        danger = danger_times(planned, GAME_RULES["flame_ticks"])
        self.assertEqual(danger[(1, 2)], {3, 4, 5, 6, 7})  # distance 1: 4 + 1 actions
        self.assertEqual(danger[(1, 3)], {3, 4, 5, 6})

    def test_without_buffs_the_rules_apply(self):
        env = BombermanEnv(with_npcs=True)
        state = env.step(B)
        self.assertEqual(bombs_from_state(state), [(1, 1, 4, 1, 2, 0)])
        self.assertEqual(levels_of(state, state["characters"][0]), (2, 0, 0))
        self.assertEqual(levels_of(state, state["characters"][1]), (2, 0, 1))

    def test_prediction_matches_the_engine_with_buffs(self):
        """Cells predicted to burn on the next action are exactly the engine's,
        and every such cell is predicted dangerous for as long as it burns."""
        rng = random.Random(0)
        checked = burn_cells = 0
        for seed in range(30):
            start = {i: {"range": rng.randint(2, 5), "bomb_count": rng.randint(1, 3),
                         "burn": rng.randint(0, 2)} for i in range(4)}
            env = BombermanEnv(with_npcs=True, random_crates=True, start_buffs=start,
                               **GAME_RULES)
            state = env.reset(seed=seed)
            bots = {i: RandomBot() for i in range(4)}
            for i, bot in bots.items():
                bot.reset(seed * 10 + i)
            for _ in range(60):
                if state["outcome"] is not None:
                    break
                if state["bombs"]:
                    planned = plan_bombs(state["grid"], bombs_from_state(state))
                    danger = danger_times(planned, state["rules"]["flame_ticks"])
                    predicted = {cell for cell, ts in danger.items() if 1 in ts}
                    probe = copy.deepcopy(env)
                    after = probe.step(W, {1: W, 2: W, 3: W})
                    self.assertEqual(set(probe.explosion), predicted)
                    for f in after["flames"]:
                        cell = (f["x"], f["y"])
                        if cell in predicted:
                            self.assertLessEqual(set(range(1, f["ticks_left"] + 1)),
                                                 danger[cell])
                            burn_cells += f["ticks_left"] > state["rules"]["flame_ticks"]
                    checked += 1
                actions = {c["id"]: bots[c["id"]].act(state, c["id"])
                           for c in state["characters"] if not c["dead"]}
                agent = actions.pop(0, W)
                if rng.random() < 0.5:
                    agent = W
                state = env.step(agent, actions)
        self.assertGreater(checked, 100)
        self.assertGreater(burn_cells, 0)

    def test_possible_bombs_use_opponent_levels(self):
        env = BombermanEnv(with_npcs=True, start_buffs={1: {"range": 4, "burn": 2,
                                                              "bomb_count": 2}},
                           **GAME_RULES)
        env._bombs[(9, 1)] = (1, Bomb(9, 1, 3))  # NPC 1 still has a bomb left
        env._bombs[(1, 9)] = (2, Bomb(1, 9, 3))  # NPC 2 has none left
        env.characters[1].x, env.characters[1].y = 9, 2
        possible = RuleBot()._possible_bombs(env.get_state(), 0)
        fuse = GAME_RULES["bomb_fuse"]
        self.assertEqual(possible, [(9, 2, 1 + fuse, 2, 4, 2), (9, 9, 1 + fuse, 2, 2, 0)])


class TestSecondBomb(unittest.TestCase):
    def test_places_a_second_bomb_when_it_has_one(self):
        for count, expected in ((2, B), (1, None)):
            with self.subTest(bomb_count=count):
                env = single({0: {"bomb_count": count}})
                env._bombs[(1, 9)] = (0, Bomb(1, 9, 4))  # far away, harmless here
                state = moved_to(env, (3, 1))  # the crate (5, 1) is in reach
                action = RuleBot(0).act(state, 0)
                if expected is B:
                    self.assertEqual(action, B)
                else:
                    self.assertNotEqual(action, B)

    def test_no_second_bomb_without_an_escape_from_every_bomb(self):
        """(2, 1) is walled, so the only way out of a bomb at (3, 1) is down
        column 3, past (3, 3). The bot's first bomb at (3, 4) blocks that way."""
        for first_bomb, expected in ((False, B), (True, None)):
            with self.subTest(first_bomb=first_bomb):
                env = single({0: {"bomb_count": 2}})
                env.grid[1][2] = Cell.WALL
                if first_bomb:
                    env._bombs[(3, 4)] = (0, Bomb(3, 4, 4, range=1))
                state = moved_to(env, (3, 1))
                action = RuleBot(0).act(state, 0)
                if expected is B:
                    self.assertEqual(action, B)
                else:
                    self.assertNotEqual(action, B)

    def test_never_more_bombs_than_its_count(self):
        env = BombermanEnv(with_npcs=True, random_crates=True,
                           start_buffs={i: {"bomb_count": 3} for i in range(4)}, **GAME_RULES)
        seen_multi = False
        for seed in range(6):
            state = env.reset(seed=seed)
            bots = {i: RuleBot() for i in range(4)}
            for i, bot in bots.items():
                bot.reset(bot_seed(seed, i))
            for _ in range(200):
                actions = {c["id"]: bots[c["id"]].act(state, c["id"])
                           for c in state["characters"] if not c["dead"]}
                state = env.step(actions.pop(0), actions)
                for c in state["characters"]:
                    owned = sum(1 for b in state["bombs"] if b["owner"] == c["id"])
                    self.assertLessEqual(owned, c["bomb_count"])
                    seen_multi |= owned >= 2
                if state["outcome"]:
                    break
        self.assertTrue(seen_multi)


class TestSafetyWithBuffs(unittest.TestCase):
    def test_no_rule_bot_dies_in_a_lingering_burn_flame(self):
        """Four rule bots with long ranges and burn 2: nobody walks into a flame
        from an earlier tick, including the 5- and 6-tick burn cells."""
        long_burns = 0
        for seed in range(8):
            start = {i: {"burn": 2, "range": 3} for i in range(4)}
            env = BombermanEnv(with_npcs=True, random_crates=True, start_buffs=start,
                               **GAME_RULES)
            state = env.reset(seed=seed)
            bots = {i: RuleBot() for i in range(4)}
            for i, bot in bots.items():
                bot.reset(bot_seed(seed, i))
            for _ in range(150):
                actions = {c["id"]: bots[c["id"]].act(state, c["id"])
                           for c in state["characters"] if not c["dead"]}
                state = env.step(actions.pop(0), actions)
                long_burns += sum(f["ticks_left"] > GAME_RULES["flame_ticks"]
                                  for f in state["flames"])
                for event in state["events"]:
                    if event["type"] == "death":
                        c = state["characters"][event["character"]]
                        self.assertIn((c["x"], c["y"]), state["explosion"],
                                      f"seed {seed}: lingering death {event}")
                if state["outcome"]:
                    break
        self.assertGreater(long_burns, 0)

    def test_rule_bot_with_extra_bombs_still_beats_random_bots(self):
        wins = 0
        for seed in range(20):
            env = BombermanEnv(with_npcs=True, random_crates=True,
                               start_buffs={0: {"bomb_count": 3}}, **GAME_RULES)
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
            wins += state["outcome"] == "win"
        self.assertGreaterEqual(wins, 18)


if __name__ == "__main__":
    unittest.main()
