"""Observation v1 (buffs and levels), the buff reward term, and the tiny stage (Task 7).

Skipped without numpy and gymnasium, like the other RL tests.
"""

import random
import unittest

try:
    import numpy as np
    from gymnasium.utils.env_checker import check_env

    from rl import observation as ob
    from rl.bomberman_env import REWARD_TERMS, BombermanGymEnv, step_reward
    from rl.config import EnvConfig, RewardConfig
    from rl.tabular_env import TinyStageEnv
except ImportError:  # pragma: no cover - depends on the installed packages
    np = None

from game.environment import (
    BUFF_RULES,
    GAME_RULES,
    Action,
    Bomb,
    BombermanEnv,
    BuffRules,
    BuffType,
)

MISSING = "numpy and gymnasium are needed for the RL tests"
A = {a: i for i, a in enumerate(Action)}
MAX = {buff.name: buff.max for buff in BUFF_RULES.types}
LONGEST = GAME_RULES["flame_ticks"] + MAX["burn"]


def game_env(**kwargs) -> BombermanEnv:
    return BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES, **kwargs)


@unittest.skipIf(np is None, MISSING)
class ObservationV1Test(unittest.TestCase):
    def test_channels(self):
        self.assertEqual(ob.N_CHANNELS, 20)
        self.assertEqual(len(ob.CHANNEL_NAMES), ob.N_CHANNELS)
        self.assertEqual(ob.OBS_SHAPE, (20, 11, 11))
        indices = [getattr(ob, name.upper()) for name in ob.CHANNEL_NAMES]
        self.assertEqual(indices, list(range(20)))

    def test_start_levels(self):
        obs = ob.encode_observation(game_env().reset(seed=0), 0)
        self.assertTrue(np.allclose(obs[ob.MY_BOMBS_LEFT], 1 / MAX["bomb_count"]))
        self.assertTrue(np.allclose(obs[ob.MY_RANGE], BUFF_RULES.get("range").start / MAX["range"]))
        self.assertEqual(obs[ob.MY_BURN].sum(), 0)
        for x, y in ((9, 1), (1, 9), (9, 9)):
            self.assertAlmostEqual(obs[ob.OPP_RANGE, y, x], 2 / 5)
            self.assertAlmostEqual(obs[ob.OPP_BOMBS_LEFT, y, x], 1 / 3)
            self.assertEqual(obs[ob.OPP_BURN, y, x], 0)
        self.assertEqual(np.count_nonzero(obs[ob.OPP_RANGE]), 3)

    def test_max_levels_are_one(self):
        top = {"range": 5, "bomb_count": 3, "burn": 2}
        env = game_env(start_buffs={i: top for i in range(4)})
        obs = ob.encode_observation(env.reset(seed=1), 0)
        for channel in (ob.MY_BOMBS_LEFT, ob.MY_RANGE, ob.MY_BURN):
            self.assertTrue((obs[channel] == 1).all())
        for channel in (ob.OPP_RANGE, ob.OPP_BOMBS_LEFT, ob.OPP_BURN):
            self.assertEqual(obs[channel].max(), 1)

    def test_bombs_flames_and_floor_buffs(self):
        env = game_env(start_buffs={0: {"range": 4, "burn": 2, "bomb_count": 2}})
        env.reset(seed=0)
        env.floor_buffs.update({(2, 1): "range", (3, 1): "bomb_count", (1, 2): "burn"})
        state = env.step(Action.BOMB)
        obs = ob.encode_observation(state, 0)
        self.assertEqual(obs[ob.BOMBS, 1, 1], 1.0)
        self.assertAlmostEqual(obs[ob.BOMB_RANGE, 1, 1], 4 / 5)
        self.assertEqual(obs[ob.BOMB_BURN, 1, 1], 1.0)
        self.assertAlmostEqual(obs[ob.MY_BOMBS_LEFT, 0, 0], 1 / 3)
        for channel, cell in ((ob.BUFF_RANGE, (2, 1)), (ob.BUFF_BOMB, (3, 1)),
                              (ob.BUFF_BURN, (1, 2))):
            self.assertEqual(obs[channel, cell[1], cell[0]], 1)
            self.assertEqual(obs[channel].sum(), 1)
        # Burn 1, range 2: distance 0-1 burns flame_ticks + 1, distance 2 burns
        # flame_ticks; both divided by the longest possible burn.
        env2 = BombermanEnv(**GAME_RULES, start_buffs={0: {"burn": 1}})
        state = env2.step(Action.BOMB)
        for action in (Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.WAIT):
            state = env2.step(action)
        self.assertTrue(state["explosion"])
        obs = ob.encode_observation(state, 0)
        self.assertAlmostEqual(obs[ob.FLAMES, 1, 1], (GAME_RULES["flame_ticks"] + 1) / LONGEST)
        self.assertAlmostEqual(obs[ob.FLAMES, 3, 1], GAME_RULES["flame_ticks"] / LONGEST)

    def test_every_plane_from_npc_2s_point_of_view(self):
        env = game_env(start_buffs={2: {"range": 5, "burn": 1}, 0: {"bomb_count": 3}})
        env.reset(seed=4)
        state = env.step(Action.WAIT, {2: Action.BOMB})  # NPC 2 at (1, 9) bombs
        as0 = ob.encode_observation(state, 0)
        as2 = ob.encode_observation(state, 2)
        self.assertEqual(as2[ob.ME, 9, 1], 1)
        self.assertEqual(as2[ob.MY_BOMBS_LEFT].max(), 0)  # its only bomb is placed
        self.assertTrue((as2[ob.MY_RANGE] == 1).all())
        self.assertTrue(np.allclose(as2[ob.MY_BURN], 1 / 2))
        self.assertEqual(as2[ob.OPP_BOMBS_LEFT, 1, 1], 1)  # the agent: 3 of 3 left
        self.assertAlmostEqual(as2[ob.OPP_RANGE, 1, 1], 2 / 5)
        self.assertEqual(as2[ob.OPP_RANGE, 9, 1], 0)  # not itself
        self.assertEqual(as0[ob.OPP_RANGE, 9, 1], 1)  # NPC 2 seen by the agent
        self.assertAlmostEqual(as0[ob.OPP_BURN, 9, 1], 1 / 2)
        self.assertEqual(as0[ob.OPP_BOMBS_LEFT, 9, 1], 0)
        self.assertTrue((as0[ob.MY_BOMBS_LEFT] == 1).all())
        shared = [ob.WALLS, ob.CRATES, ob.BOMBS, ob.BOMB_RANGE, ob.BOMB_BURN, ob.FLAMES,
                  ob.GATE_HIDDEN, ob.GATE_CLOSED, ob.GATE_OPEN,
                  ob.BUFF_RANGE, ob.BUFF_BOMB, ob.BUFF_BURN]
        np.testing.assert_array_equal(as0[shared], as2[shared])

    def test_opponents_on_one_cell_keep_the_highest_value(self):
        env = game_env(start_buffs={1: {"range": 3}, 2: {"range": 5}})
        env.reset(seed=0)
        env.characters[2].x, env.characters[2].y = 9, 1  # same cell as NPC 1
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertEqual(obs[ob.OPP_RANGE, 1, 9], 1)
        self.assertEqual(obs[ob.OPPONENTS, 1, 9], 1)

    def test_dead_me_has_empty_level_planes(self):
        env = game_env()
        env.reset(seed=0)
        env.characters[0].dead = True
        obs = ob.encode_observation(env.get_state(), 0)
        for channel in (ob.ME, ob.MY_BOMBS_LEFT, ob.MY_RANGE, ob.MY_BURN):
            self.assertEqual(obs[channel].sum(), 0)

    def test_state_without_buffs_falls_back_to_base_levels(self):
        env = BombermanEnv(with_npcs=True)
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertTrue((obs[ob.MY_BOMBS_LEFT] == 1).all())  # the old "bomb ready"
        self.assertTrue((obs[ob.MY_RANGE] == 1).all())
        self.assertEqual(obs[ob.MY_BURN].sum(), 0)  # max burn 0: guarded, never NaN
        obs = ob.encode_observation(env.step(Action.BOMB), 0)
        self.assertEqual(obs[ob.MY_BOMBS_LEFT].sum(), 0)
        self.assertEqual(obs[ob.BOMB_RANGE, 1, 1], 1)
        self.assertEqual(obs[ob.FLAMES].sum(), 0)
        self.assertTrue(np.isfinite(obs).all())

    def test_max_of_zero_is_guarded(self):
        rules = BuffRules((BuffType("range", 2, 2, 0.5), BuffType("burn", 0, 0, 0.1)))
        env = BombermanEnv(with_npcs=True, buffs=rules)
        env.bomb = Bomb(1, 1, 3)
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(obs[ob.MY_BURN].sum(), 0)
        self.assertEqual(obs[ob.BOMB_BURN].sum(), 0)

    def test_range_over_real_games_with_buffs(self):
        """Every value stays in [0, 1] from every point of view, buffs at any level."""
        rng = random.Random(3)
        planes_seen = set()
        for seed in range(25):
            start = {i: {"range": rng.randint(2, 5), "bomb_count": rng.randint(1, 3),
                         "burn": rng.randint(0, 2)} for i in range(4)}
            env = game_env(start_buffs=start)
            state = env.reset(seed=seed)
            for _ in range(120):
                actions = {c["id"]: rng.choice(tuple(Action))
                           for c in state["characters"] if not c["dead"]}
                state = env.step(actions.pop(0, Action.WAIT), actions)
                for me in range(4):
                    obs = ob.encode_observation(state, me)
                    self.assertEqual(obs.dtype, np.float32)
                    self.assertGreaterEqual(obs.min(), 0.0)
                    self.assertLessEqual(obs.max(), 1.0)
                    planes_seen.update(np.flatnonzero(obs.reshape(20, -1).any(axis=1)))
                if state["outcome"]:
                    break
        # The gate rarely opens among random bots; every other plane shows up.
        self.assertEqual(planes_seen | {ob.GATE_CLOSED, ob.GATE_OPEN}, set(range(20)))


@unittest.skipIf(np is None, MISSING)
class BuffRewardTest(unittest.TestCase):
    def pickup(self, cfg, start=None):
        env = BombermanEnv(with_npcs=True, **GAME_RULES, start_buffs=start)
        env.floor_buffs[(2, 1)] = "range"
        prev = env.get_state()
        state = env.step(Action.RIGHT)
        return step_reward(prev, state, Action.RIGHT, 0, cfg)

    def test_buff_term_is_off_by_default(self):
        self.assertEqual(RewardConfig().buff, 0.0)
        self.assertIn("buff", REWARD_TERMS)
        _, terms = self.pickup(RewardConfig())
        self.assertEqual(terms["buff"], 0.0)

    def test_buff_term_counts_real_increases_only(self):
        cfg = RewardConfig(buff=0.5, per_step=0.0)
        reward, terms = self.pickup(cfg)
        self.assertEqual(terms["buff"], 0.5)
        self.assertEqual(reward, 0.5)
        _, terms = self.pickup(cfg, start={0: {"range": 5}})  # already at the max
        self.assertEqual(terms["buff"], 0.0)

    def test_another_characters_pickup_earns_nothing(self):
        env = BombermanEnv(with_npcs=True, **GAME_RULES)
        env.floor_buffs[(9, 2)] = "burn"
        prev = env.get_state()
        state = env.step(Action.WAIT, {1: Action.DOWN})
        self.assertEqual(state["characters"][1]["burn"], 1)
        _, terms = step_reward(prev, state, Action.WAIT, 0, RewardConfig(buff=1.0))
        self.assertEqual(terms["buff"], 0.0)
        _, terms = step_reward(prev, state, Action.DOWN, 1, RewardConfig(buff=1.0))
        self.assertEqual(terms["buff"], 1.0)

    def test_reward_terms_in_info(self):
        env = BombermanGymEnv("random", EnvConfig(reward=RewardConfig(buff=0.25)))
        env.reset(seed=0)
        env.engine.floor_buffs[(2, 1)] = "bomb_count"
        env._state = env.engine.get_state()
        _, reward, _, _, info = env.step(A[Action.RIGHT])
        self.assertEqual(set(info["reward_terms"]), set(REWARD_TERMS))
        self.assertEqual(info["reward_terms"]["buff"], 0.25)
        self.assertAlmostEqual(sum(info["reward_terms"].values()), reward)


@unittest.skipIf(np is None, MISSING)
class WrapperTest(unittest.TestCase):
    def test_wrapper_plays_buffs_and_passes_check_env(self):
        env = BombermanGymEnv("random")
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (20, 11, 11))
        self.assertIsNotNone(env.state["buff_rules"])
        self.assertTrue(env.observation_space.contains(obs))

    def test_tiny_stage_has_no_buffs(self):
        env = TinyStageEnv()
        env.reset(seed=0)
        self.assertIsNone(env.state["buff_rules"])
        self.assertEqual(env.state["rules"]["bomb_fuse"], GAME_RULES["bomb_fuse"])
        self.assertEqual(env.state["rules"]["flame_ticks"], GAME_RULES["flame_ticks"])


if __name__ == "__main__":
    unittest.main()
