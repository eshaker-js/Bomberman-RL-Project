"""The Task 10 shaping terms ``wait`` and ``bomb_survived`` (``step_reward``).

Most tests use hand-made ``prev``/``state`` dicts with only the keys
``step_reward`` reads; a few play the real engine (``GAME_RULES``, the fixed
map, a crate put in place by hand). The last class checks that with both
terms at 0.0 (the default) every reward and every ``info`` value of a few
hundred random games equals today's, against a frozen copy of the v0
``step_reward``.

Needs numpy and gymnasium; every test is skipped with a clear reason if they
are not installed.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gymnasium  # noqa: F401
    import numpy as np
except ImportError:  # pragma: no cover - depends on the machine
    np = None

MISSING = "numpy/gymnasium not installed (pip install -r requirements.txt); RL test skipped"

from game.environment import GAME_RULES, Action, BombermanEnv, Cell  # noqa: E402

if np is not None:
    from rl.bomberman_env import (
        ACTIONS,
        MOVE_ACTIONS,
        REWARD_TERMS,
        BombermanGymEnv,
        bomb_refused,
        step_reward,
    )
    from rl.config import EnvConfig, RewardConfig

W = -0.002  # values of the E2 plan; any non-zero values would do
B = 0.05
SHAPED = None if np is None else RewardConfig(wait=W, bomb_survived=B)


def character(i, x, y, dead=False, bombs_left=1):
    return {"id": i, "x": x, "y": y, "dead": dead, "bombs_left": bombs_left}


def bomb(owner, x, y, countdown=1):
    return {"owner": owner, "x": x, "y": y, "countdown": countdown}


def state(characters=None, bombs=(), events=(), outcome=None):
    """A hand-made state: agent at (3, 3) and NPC 1 at (7, 7) unless given."""
    if characters is None:
        characters = [character(0, 3, 3), character(1, 7, 7)]
    return {"characters": characters, "bombs": list(bombs),
            "events": list(events), "outcome": outcome}


def crate(*owners, cell=(5, 5)):
    return {"type": "block_destroyed", "cell": cell, "destroyed_by": tuple(owners)}


def terms(prev, after, action=Action.WAIT, me=0, cfg=None):
    return step_reward(prev, after, action, me, cfg or SHAPED)[1]


@unittest.skipIf(np is None, MISSING)
class DefaultsTest(unittest.TestCase):
    def test_both_terms_are_off_by_default(self):
        cfg = RewardConfig()
        self.assertEqual(cfg.wait, 0.0)
        self.assertEqual(cfg.bomb_survived, 0.0)

    def test_terms_are_listed_last(self):
        self.assertEqual(REWARD_TERMS, ("outcome", "kill", "crate", "per_step", "bump",
                                        "buff", "wait", "bomb_survived"))


@unittest.skipIf(np is None, MISSING)
class WaitTermTest(unittest.TestCase):
    def test_wait_pays_wait(self):
        t = terms(state(), state(), Action.WAIT)
        self.assertEqual(t["wait"], W)

    def test_moves_do_not_pay_wait(self):
        for action in MOVE_ACTIONS:  # blocked or not, a move is never a wait
            self.assertEqual(terms(state(), state(), action)["wait"], 0.0, action)

    def test_placed_bomb_does_not_pay_wait(self):
        after = state(bombs=[bomb(0, 3, 3, countdown=4)])
        self.assertFalse(bomb_refused(state(), 0))
        self.assertEqual(terms(state(), after, Action.BOMB)["wait"], 0.0)

    def test_refused_bomb_without_bombs_left_pays_wait(self):
        prev = state([character(0, 3, 3, bombs_left=0), character(1, 7, 7)],
                     bombs=[bomb(0, 1, 1, countdown=3)])
        self.assertTrue(bomb_refused(prev, 0))
        self.assertEqual(terms(prev, prev, Action.BOMB)["wait"], W)

    def test_refused_bomb_on_an_occupied_cell_pays_wait(self):
        # Bombs left, but an NPC's bomb already lies on the agent's cell.
        prev = state(bombs=[bomb(1, 3, 3, countdown=2)])
        self.assertTrue(bomb_refused(prev, 0))
        self.assertEqual(terms(prev, prev, Action.BOMB)["wait"], W)

    def test_refused_bomb_without_buff_keys(self):
        # Without buffs there is no bombs_left: one bomb per character.
        plain = [{"id": 0, "x": 3, "y": 3, "dead": False}]
        self.assertFalse(bomb_refused(state(plain), 0))
        self.assertTrue(bomb_refused(state(plain, bombs=[bomb(0, 1, 1)]), 0))

    def test_real_engine_refused_bomb_pays_wait(self):
        env = BombermanEnv(**GAME_RULES)
        s0 = env.get_state()
        s1 = env.step(Action.BOMB)
        self.assertEqual(step_reward(s0, s1, Action.BOMB, 0, SHAPED)[1]["wait"], 0.0)
        s2 = env.step(Action.BOMB)  # one bomb only: refused, the agent stays
        self.assertEqual(s2["bombs"], [dict(s1["bombs"][0], countdown=3)])
        self.assertEqual(step_reward(s1, s2, Action.BOMB, 0, SHAPED)[1]["wait"], W)


@unittest.skipIf(np is None, MISSING)
class BombSurvivedTest(unittest.TestCase):
    def test_one_bomb_one_crate_alive_pays_once(self):
        prev = state(bombs=[bomb(0, 5, 3)])
        t = terms(prev, state(events=[crate(0)]))
        self.assertEqual(t["bomb_survived"], B)

    def test_bomb_breaking_no_crate_pays_nothing(self):
        prev = state(bombs=[bomb(0, 5, 3)])
        self.assertEqual(terms(prev, state())["bomb_survived"], 0.0)

    def test_bomb_breaking_two_crates_pays_once(self):
        prev = state(bombs=[bomb(0, 5, 3)])
        after = state(events=[crate(0, cell=(5, 5)), crate(0, cell=(7, 3))])
        self.assertEqual(terms(prev, after)["bomb_survived"], B)

    def test_two_bombs_two_crates_pay_twice(self):
        prev = state(bombs=[bomb(0, 5, 3), bomb(0, 1, 5)])
        after = state(events=[crate(0, cell=(5, 5)), crate(0, cell=(1, 7))])
        self.assertEqual(terms(prev, after)["bomb_survived"], 2 * B)

    def test_two_bombs_one_crate_pay_once(self):
        prev = state(bombs=[bomb(0, 5, 3), bomb(0, 1, 5)])
        self.assertEqual(terms(prev, state(events=[crate(0)]))["bomb_survived"], B)

    def test_known_approximation_one_bomb_two_crates_other_none_pays_twice(self):
        # Bomb (5, 3) breaks two crates, bomb (1, 5) none: exactly one bomb
        # broke a crate, but min(2 bombs, 2 crates) pays twice (documented).
        prev = state(bombs=[bomb(0, 5, 3), bomb(0, 1, 5)])
        after = state(events=[crate(0, cell=(5, 5)), crate(0, cell=(7, 3))])
        self.assertEqual(terms(prev, after)["bomb_survived"], 2 * B)

    def test_known_approximation_two_bombs_same_crate_pay_once(self):
        # Both bombs reach the crate at (5, 5): one event, me named once.
        # Two bombs broke a crate, but min(2 bombs, 1 crate) pays once.
        prev = state(bombs=[bomb(0, 5, 3), bomb(0, 3, 5)])
        self.assertEqual(terms(prev, state(events=[crate(0)]))["bomb_survived"], B)

    def test_dead_agent_gets_nothing(self):
        prev = state(bombs=[bomb(0, 5, 3)])
        after = state([character(0, 3, 3, dead=True), character(1, 7, 7)],
                      events=[crate(0), {"type": "death", "character": 0,
                                         "killed_by": (0,)}], outcome="loss")
        self.assertEqual(terms(prev, after)["bomb_survived"], 0.0)

    def test_opponent_bomb_pays_nothing(self):
        prev = state(bombs=[bomb(1, 5, 3)])
        # Even with a crate credited to me by another blast of this step.
        self.assertEqual(terms(prev, state(events=[crate(1), crate(0, cell=(1, 7))]))
                         ["bomb_survived"], 0.0)
        self.assertEqual(terms(prev, state(events=[crate(1)]), me=1)["bomb_survived"], B)

    def test_chain_triggered_own_bomb_counts(self):
        prev = state(bombs=[bomb(0, 5, 3, countdown=3), bomb(1, 5, 5)])
        after = state(events=[{"type": "bomb_triggered", "owner": 0, "triggered_by": (1,)},
                              crate(0, cell=(7, 3))])
        self.assertEqual(terms(prev, after)["bomb_survived"], B)

    def test_chain_counts_only_crates_credited_to_me(self):
        # My bomb exploded and set off NPC 1's bomb; only NPC 1's blast
        # reached the crate, so it is credited to NPC 1 alone.
        prev = state(bombs=[bomb(0, 5, 3), bomb(1, 5, 5, countdown=3)])
        after = state(events=[{"type": "bomb_triggered", "owner": 1, "triggered_by": (0,)},
                              crate(1, cell=(7, 5))])
        self.assertEqual(terms(prev, after)["bomb_survived"], 0.0)

    def test_new_bomb_while_old_explodes_pays_once(self):
        prev = state([character(0, 3, 3, bombs_left=1), character(1, 7, 7)],
                     bombs=[bomb(0, 5, 3)])
        after = state(bombs=[bomb(0, 3, 3, countdown=4)], events=[crate(0)])
        self.assertEqual(terms(prev, after, Action.BOMB)["bomb_survived"], B)

    def test_ticking_bomb_pays_nothing(self):
        prev = state(bombs=[bomb(0, 5, 3, countdown=3)])
        after = state(bombs=[bomb(0, 5, 3, countdown=2)], events=[crate(0)])
        self.assertEqual(terms(prev, after)["bomb_survived"], 0.0)

    def test_win_step_pays(self):
        prev = state(bombs=[bomb(0, 5, 3)])
        self.assertEqual(terms(prev, state(events=[crate(0)], outcome="win"))
                         ["bomb_survived"], B)

    def test_total_is_the_sum_of_the_terms(self):
        prev = state(bombs=[bomb(0, 5, 3)])
        reward, t = step_reward(prev, state(events=[crate(0)]), Action.WAIT, 0, SHAPED)
        self.assertEqual(reward, sum(t.values()))
        self.assertAlmostEqual(reward, 0.02 - 0.001 + W + B)


@unittest.skipIf(np is None, MISSING)
class RealEngineTest(unittest.TestCase):
    """Bomb, escape, survive on the real engine (fixed map, one crate added)."""

    PLAN = [Action.BOMB, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.WAIT]

    def play(self, with_crate):
        env = BombermanEnv(**GAME_RULES)
        env.grid[3][1] = Cell.BLOCK if with_crate else Cell.EMPTY  # in the bomb's down ray
        prev = env.get_state()
        paid = []
        for action in self.PLAN:
            after = env.step(action)
            paid.append(step_reward(prev, after, action, 0, SHAPED)[1])
            prev = after
        self.assertFalse(prev["characters"][0]["dead"])
        self.assertEqual(prev["bombs"], [])  # exploded on the last step
        return paid

    def test_bomb_that_breaks_a_crate_pays_on_its_explosion_step(self):
        paid = self.play(with_crate=True)
        self.assertEqual([t["bomb_survived"] for t in paid], [0.0, 0.0, 0.0, 0.0, B])
        self.assertEqual(paid[-1]["crate"], 0.02)
        self.assertEqual([t["wait"] for t in paid], [0.0, 0.0, 0.0, 0.0, W])

    def test_bomb_that_breaks_nothing_pays_nothing(self):
        paid = self.play(with_crate=False)
        self.assertEqual([t["bomb_survived"] for t in paid], [0.0] * 5)

    def test_bomb_placed_and_set_off_on_the_same_step_kills_its_placer(self):
        # NPC 1's bomb, due now, reaches the agent's cell as the agent places
        # a bomb there: the new bomb is not in prev, and the agent dies.
        env = BombermanEnv(with_npcs=True, **GAME_RULES)
        npc = env.characters[1]
        npc.x, npc.y = 3, 1
        env.step(Action.WAIT, {1: Action.BOMB})
        for _ in range(GAME_RULES["bomb_fuse"] - 1):
            env.step(Action.WAIT, {1: Action.RIGHT})
        prev = env.get_state()
        self.assertEqual([b["countdown"] for b in prev["bombs"]], [1])
        after = env.step(Action.BOMB)
        self.assertTrue(after["characters"][0]["dead"])
        self.assertIn({"type": "bomb_triggered", "owner": 0, "triggered_by": (1,),
                       "x": 1, "y": 1}, after["events"])
        t = step_reward(prev, after, Action.BOMB, 0, SHAPED)[1]
        self.assertEqual(t["bomb_survived"], 0.0)


def v0_step_reward(prev, state, action, me, cfg):
    """Frozen copy of ``step_reward`` before Task 10 (the reference for bit-identity)."""
    old_terms = ("outcome", "kill", "crate", "per_step", "bump", "buff")
    terms = dict.fromkeys(old_terms, 0.0)
    outcome = state["outcome"]
    if outcome is not None:
        terms["outcome"] = {"win": cfg.win, "loss": cfg.loss, "draw": cfg.draw}[outcome]
    for event in state["events"]:
        if event["type"] == "death":
            if event["character"] != me and me in event["killed_by"]:
                terms["kill"] += cfg.kill
        elif event["type"] == "block_destroyed":
            if me in event["destroyed_by"]:
                terms["crate"] += cfg.crate
        elif event["type"] == "buff_picked" and event["character"] == me:
            before = prev["characters"][me].get(event["buff"], event["level"])
            if event["level"] > before:
                terms["buff"] += cfg.buff
    terms["per_step"] = cfg.per_step
    if action in MOVE_ACTIONS:
        before = prev["characters"][me]
        after = state["characters"][me]
        if (before["x"], before["y"]) == (after["x"], after["y"]):
            terms["bump"] = cfg.bump
    return float(sum(terms.values())), terms


@unittest.skipIf(np is None, MISSING)
class DefaultsAreBitIdenticalTest(unittest.TestCase):
    GAMES = 300

    def play_all(self, opponents, cfg):
        env = BombermanGymEnv(opponents, EnvConfig(reward=cfg))
        rng = np.random.default_rng(10)
        steps = 0
        try:
            for seed in range(self.GAMES):
                env.reset(seed=1000 + seed)
                done = False
                while not done:
                    prev = env.state
                    action = int(rng.integers(len(ACTIONS)))
                    _, reward, term, trunc, info = env.step(action)
                    yield prev, env.state, ACTIONS[action], reward, info
                    steps += 1
                    done = term or trunc
        finally:
            env.close()
        self.assertGreater(steps, self.GAMES)

    def test_rewards_and_info_equal_v0(self):
        for opponents in ("random", "noisy25"):
            for prev, after, action, reward, info in self.play_all(opponents, RewardConfig()):
                old_reward, old_terms = v0_step_reward(prev, after, action, 0, RewardConfig())
                self.assertEqual(reward, old_reward)  # exactly, not almost
                new_terms = dict(info["reward_terms"])
                self.assertEqual(new_terms.pop("wait"), 0.0)
                self.assertEqual(new_terms.pop("bomb_survived"), 0.0)
                self.assertEqual(list(new_terms.items()), list(old_terms.items()))
                self.assertEqual(set(info), {"outcome", "steps", "events",
                                             "episode_seed", "reward_terms"})
                self.assertEqual(info["outcome"], after["outcome"])
                self.assertEqual(info["steps"], after["steps"])
                self.assertEqual(info["events"], after["events"])

    def test_shaped_total_is_v0_plus_the_new_terms(self):
        paid_wait = paid_bomb = 0
        for prev, after, action, reward, info in self.play_all("random", SHAPED):
            old_reward, _ = v0_step_reward(prev, after, action, 0, SHAPED)
            t = info["reward_terms"]
            self.assertAlmostEqual(reward, old_reward + t["wait"] + t["bomb_survived"])
            paid_wait += t["wait"] != 0.0
            paid_bomb += t["bomb_survived"] != 0.0
        self.assertGreater(paid_wait, 0)
        self.assertGreater(paid_bomb, 0)


if __name__ == "__main__":
    unittest.main()
