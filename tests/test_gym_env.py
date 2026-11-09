"""The Gymnasium wrapper, observation and reward (Task 6, stage 3).

Needs numpy and gymnasium; every test here is skipped with a clear reason if
they are not installed. Pygame is never required. A few tests edit the
wrapper's engine after ``reset`` (public attributes, as the other test files
do) to build a known situation.
"""

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
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

from agents.match import bot_seed  # noqa: E402
from agents.noisy_bot import NoisyBot  # noqa: E402
from agents.random_bot import RandomBot  # noqa: E402
from agents.rule_bot import RuleBot  # noqa: E402
from game.environment import (  # noqa: E402
    GAME_RULES,
    GATE_POSITION,
    Action,
    BombermanEnv,
    Cell,
    Gate,
)

if np is not None:
    from gymnasium.utils.env_checker import check_env

    from rl import observation as ob
    from rl.bomberman_env import ACTIONS, BombermanGymEnv, step_reward
    from rl.config import (
        EnvConfig,
        RewardConfig,
        apply_overrides,
        config_to_dict,
        load_config,
    )

A = {a: i for i, a in enumerate(Action)}  # Action -> index


class WaitBot:
    def reset(self, seed=None):
        pass

    def act(self, state, me):
        return Action.WAIT


def chars(*positions, dead=()):
    """Minimal ``characters`` list for synthetic reward tests."""
    return [{"id": i, "x": x, "y": y, "dead": i in dead}
            for i, (x, y) in enumerate(positions)]


def synthetic(positions, events=(), outcome=None, dead=()):
    return {"characters": chars(*positions, dead=dead),
            "events": list(events), "outcome": outcome}


@unittest.skipIf(np is None, MISSING)
class ActionAndSpacesTest(unittest.TestCase):
    def test_action_mapping_follows_the_enum(self):
        self.assertEqual(ACTIONS, (Action.UP, Action.DOWN, Action.LEFT,
                                   Action.RIGHT, Action.WAIT, Action.BOMB))
        env = BombermanGymEnv(None)
        self.assertEqual(env.action_space.n, 6)

    def test_each_index_drives_its_action(self):
        # From (1,1): DOWN -> (1,2), RIGHT -> (2,1), UP/LEFT hit walls,
        # WAIT stays, BOMB places a bomb. (1,2) and (2,1) are safe cells.
        expected = {0: (1, 1), 1: (1, 2), 2: (1, 1), 3: (2, 1), 4: (1, 1), 5: (1, 1)}
        for index, cell in expected.items():
            env = BombermanGymEnv(None)
            env.reset(seed=0)
            env.step(index)
            c = env.state["characters"][0]
            self.assertEqual((c["x"], c["y"]), cell, f"action {index}")
            self.assertEqual(bool(env.state["bombs"]), index == 5)

    def test_invalid_action_raises(self):
        env = BombermanGymEnv(None)
        env.reset(seed=0)
        for bad in (6, -1):
            with self.assertRaises(ValueError):
                env.step(bad)

    def test_numpy_integer_actions_work(self):
        env = BombermanGymEnv(None)
        env.reset(seed=0)
        env.step(np.int64(3))
        self.assertEqual(env.state["characters"][0]["x"], 2)

    def test_rules_and_crates_are_the_real_game(self):
        env = BombermanGymEnv("random")
        env.reset(seed=1)
        self.assertEqual(env.state["rules"]["bomb_fuse"], GAME_RULES["bomb_fuse"])
        self.assertEqual(env.state["rules"]["flame_ticks"], GAME_RULES["flame_ticks"])
        self.assertTrue(env.engine.random_crates)


@unittest.skipIf(np is None, MISSING)
class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.env = BombermanGymEnv("random")
        self.obs, _ = self.env.reset(seed=3)

    def test_shape_dtype_range(self):
        self.assertEqual(self.obs.shape, (10, 11, 11))
        self.assertEqual(self.obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(self.obs))
        for _ in range(60):
            obs, _, term, trunc, _ = self.env.step(self.env.action_space.sample())
            self.assertGreaterEqual(obs.min(), 0.0)
            self.assertLessEqual(obs.max(), 1.0)
            if term or trunc:
                break

    def test_walls_and_crates(self):
        grid = self.env.state["grid"]
        for y in range(11):
            for x in range(11):
                self.assertEqual(self.obs[ob.WALLS, y, x], grid[y][x] is Cell.WALL)
                self.assertEqual(self.obs[ob.CRATES, y, x], grid[y][x] is Cell.BLOCK)

    def test_me_opponents_gate_ready_at_start(self):
        self.assertEqual(self.obs[ob.ME].sum(), 1)
        self.assertEqual(self.obs[ob.ME, 1, 1], 1)
        self.assertEqual(self.obs[ob.OPPONENTS].sum(), 3)
        for x, y in ((9, 1), (1, 9), (9, 9)):
            self.assertEqual(self.obs[ob.OPPONENTS, y, x], 1)
        gx, gy = GATE_POSITION
        self.assertEqual(self.obs[ob.GATE_HIDDEN, gy, gx], 1)
        self.assertEqual(self.obs[ob.GATE_HIDDEN].sum(), 1)
        self.assertEqual(self.obs[ob.GATE_CLOSED].sum() + self.obs[ob.GATE_OPEN].sum(), 0)
        self.assertTrue((self.obs[ob.BOMB_READY] == 1).all())

    def test_bomb_countdown_and_ready_plane(self):
        obs, *_ = self.env.step(A[Action.BOMB])
        self.assertAlmostEqual(obs[ob.BOMBS, 1, 1], 1.0)  # fresh bomb: countdown = fuse
        self.assertTrue((obs[ob.BOMB_READY] == 0).all())
        obs, *_ = self.env.step(A[Action.RIGHT])
        self.assertAlmostEqual(obs[ob.BOMBS, 1, 1], (GAME_RULES["bomb_fuse"] - 1) / GAME_RULES["bomb_fuse"], places=6)

    def test_flames_fraction(self):
        env = BombermanGymEnv(None)
        env.reset(seed=0)
        env.step(A[Action.BOMB])
        for _ in range(3):
            env.step(A[Action.RIGHT])  # to (4,1), outside the blast
        obs = None
        for _ in range(GAME_RULES["bomb_fuse"] - 3):  # bomb explodes on step 1 + fuse
            obs, *_ = env.step(A[Action.WAIT])
        self.assertTrue(env.state["flames"])
        for flame in env.state["flames"]:
            self.assertAlmostEqual(obs[ob.FLAMES, flame["y"], flame["x"]],
                                   flame["ticks_left"] / 4)
        self.assertAlmostEqual(obs[ob.FLAMES, 1, 1], 1.0)  # tick of the explosion
        self.assertEqual(obs[ob.BOMBS].sum(), 0)
        self.assertTrue((obs[ob.BOMB_READY] == 1).all())
        obs, *_ = env.step(A[Action.WAIT])
        self.assertAlmostEqual(obs[ob.FLAMES, 1, 1], 0.75)

    def test_gate_closed_and_open(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        env.reset(seed=0)
        gx, gy = GATE_POSITION
        env.gate = Gate(gx, gy, "closed")
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertEqual(obs[ob.GATE_CLOSED, gy, gx], 1)
        self.assertEqual(obs[ob.GATE_HIDDEN].sum() + obs[ob.GATE_OPEN].sum(), 0)
        env.gate = Gate(gx, gy, "open")
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertEqual(obs[ob.GATE_OPEN, gy, gx], 1)
        self.assertEqual(obs[ob.GATE_HIDDEN].sum() + obs[ob.GATE_CLOSED].sum(), 0)

    def test_dead_me_and_dead_opponent(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        env.reset(seed=0)
        env.characters[0].dead = True
        env.characters[2].dead = True
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertEqual(obs[ob.ME].sum(), 0)
        self.assertEqual(obs[ob.OPPONENTS].sum(), 2)
        self.assertEqual(obs[ob.OPPONENTS, 9, 1], 0)
        self.assertTrue((obs[ob.BOMB_READY] == 0).all())

    def test_two_opponents_on_one_cell_stay_binary(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        env.reset(seed=0)
        env.characters[2].x, env.characters[2].y = 9, 1  # same cell as NPC 1
        obs = ob.encode_observation(env.get_state(), 0)
        self.assertEqual(obs[ob.OPPONENTS, 1, 9], 1)
        self.assertEqual(obs[ob.OPPONENTS].sum(), 2)

    def test_point_of_view_of_another_character(self):
        env = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        state = env.reset(seed=4)
        state = env.step(Action.WAIT, {2: Action.BOMB})  # NPC 2 at (1,9) bombs
        as0 = ob.encode_observation(state, 0)
        as2 = ob.encode_observation(state, 2)
        self.assertEqual(as2[ob.ME, 9, 1], 1)
        self.assertEqual(as2[ob.ME].sum(), 1)
        self.assertEqual(as2[ob.OPPONENTS, 1, 1], 1)  # the agent is an opponent
        self.assertEqual(as2[ob.OPPONENTS, 9, 1], 0)
        self.assertEqual(as2[ob.OPPONENTS].sum(), 3)
        self.assertTrue((as2[ob.BOMB_READY] == 0).all())  # NPC 2's bomb is active
        self.assertTrue((as0[ob.BOMB_READY] == 1).all())
        shared = [ob.WALLS, ob.CRATES, ob.BOMBS, ob.FLAMES,
                  ob.GATE_HIDDEN, ob.GATE_CLOSED, ob.GATE_OPEN]
        np.testing.assert_array_equal(as0[shared], as2[shared])

    def test_unknown_character_rejected(self):
        state = BombermanEnv().reset(seed=0)  # single-player: only ID 0
        with self.assertRaises(ValueError):
            ob.encode_observation(state, 1)


@unittest.skipIf(np is None, MISSING)
class RewardTermsTest(unittest.TestCase):
    """Each term of ``step_reward`` on its own, with synthetic states."""

    cfg = None

    def setUp(self):
        self.cfg = RewardConfig()
        self.prev = synthetic([(3, 3), (5, 5)])

    def reward(self, state, action=Action.WAIT, me=0):
        return step_reward(self.prev, state, action, me, self.cfg)

    def test_per_step_only(self):
        r, terms = self.reward(synthetic([(3, 3), (5, 5)]))
        self.assertAlmostEqual(r, -0.001)
        self.assertEqual(terms["per_step"], -0.001)
        self.assertEqual(sum(v != 0 for v in terms.values()), 1)

    def test_outcomes(self):
        for outcome, value in (("win", 1.0), ("loss", -1.0), ("draw", -0.5)):
            r, terms = self.reward(synthetic([(3, 3), (5, 5)], outcome=outcome))
            self.assertEqual(terms["outcome"], value)
            self.assertAlmostEqual(r, value - 0.001)

    def test_kill(self):
        state = synthetic([(3, 3), (5, 5)], dead={1},
                          events=[{"type": "death", "character": 1, "killed_by": (0,)}])
        r, terms = self.reward(state)
        self.assertAlmostEqual(terms["kill"], 0.3)
        self.assertAlmostEqual(r, 0.3 - 0.001)

    def test_shared_kill_gives_full_credit(self):
        state = synthetic([(3, 3), (5, 5)], dead={1},
                          events=[{"type": "death", "character": 1, "killed_by": (0, 2)}])
        self.assertAlmostEqual(self.reward(state)[1]["kill"], 0.3)

    def test_kill_by_someone_else_is_not_mine(self):
        state = synthetic([(3, 3), (5, 5)], dead={1},
                          events=[{"type": "death", "character": 1, "killed_by": (2,)}])
        self.assertEqual(self.reward(state)[1]["kill"], 0.0)

    def test_self_kill_is_only_a_loss(self):
        state = synthetic([(3, 3), (5, 5)], dead={0}, outcome="loss",
                          events=[{"type": "death", "character": 0, "killed_by": (0,)}])
        r, terms = self.reward(state)
        self.assertEqual(terms["kill"], 0.0)
        self.assertAlmostEqual(r, -1.0 - 0.001)

    def test_two_kills_and_crates(self):
        events = [
            {"type": "block_destroyed", "cell": (4, 3), "destroyed_by": (0,)},
            {"type": "block_destroyed", "cell": (3, 4), "destroyed_by": (0, 1)},
            {"type": "block_destroyed", "cell": (7, 7), "destroyed_by": (1,)},
            {"type": "death", "character": 1, "killed_by": (0,)},
            {"type": "death", "character": 2, "killed_by": (0, 1)},
        ]
        state = synthetic([(3, 3), (5, 5), (6, 6)], events=events, dead={1, 2})
        self.prev = synthetic([(3, 3), (5, 5), (6, 6)])
        r, terms = self.reward(state)
        self.assertAlmostEqual(terms["crate"], 0.04)
        self.assertAlmostEqual(terms["kill"], 0.6)
        self.assertAlmostEqual(r, 0.64 - 0.001)

    def test_bump_on_blocked_move(self):
        for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            r, terms = self.reward(synthetic([(3, 3), (5, 5)]), action)
            self.assertEqual(terms["bump"], -0.01)
            self.assertAlmostEqual(r, -0.011)

    def test_no_bump_when_moved_or_not_a_move(self):
        r, terms = self.reward(synthetic([(4, 3), (5, 5)]), Action.RIGHT)
        self.assertEqual(terms["bump"], 0.0)
        for action in (Action.WAIT, Action.BOMB):
            self.assertEqual(self.reward(synthetic([(3, 3), (5, 5)]), action)[1]["bump"], 0.0)

    def test_point_of_view_of_npc(self):
        state = synthetic([(3, 3), (5, 5)], dead={0},
                          events=[{"type": "death", "character": 0, "killed_by": (1,)}])
        self.prev = synthetic([(3, 3), (5, 5)])
        r, terms = self.reward(state, Action.UP, me=1)
        self.assertAlmostEqual(terms["kill"], 0.3)
        self.assertEqual(terms["bump"], -0.01)

    def test_values_come_from_the_config(self):
        self.cfg = RewardConfig(kill=2.0, per_step=0.0)
        state = synthetic([(3, 3), (5, 5)], dead={1},
                          events=[{"type": "death", "character": 1, "killed_by": (0,)}])
        self.assertAlmostEqual(self.reward(state)[0], 2.0)


@unittest.skipIf(np is None, MISSING)
class EpisodeTest(unittest.TestCase):
    def test_bump_into_wall_in_the_env(self):
        env = BombermanGymEnv(None)
        env.reset(seed=0)
        _, r, _, _, info = env.step(A[Action.UP])  # (1,0) is a wall
        self.assertAlmostEqual(r, -0.011)
        self.assertEqual(info["reward_terms"]["bump"], -0.01)
        self.assertAlmostEqual(sum(info["reward_terms"].values()), r)

    def test_kill_in_the_env(self):
        env = BombermanGymEnv([WaitBot()])
        env.reset(seed=0)
        npc = env.engine.characters[1]
        npc.x, npc.y = 1, 2  # next to the agent, inside its blast
        env._state = env.engine.get_state()
        plan = [A[Action.BOMB]] + [A[Action.RIGHT]] * 3 + [A[Action.WAIT]] * (GAME_RULES["bomb_fuse"] - 3)
        total, info = 0.0, None
        for a in plan:
            _, r, term, trunc, info = env.step(a)
            total += r
        self.assertEqual(info["steps"], 1 + GAME_RULES["bomb_fuse"])
        self.assertIn({"type": "death", "character": 1, "killed_by": (0,)}, info["events"])
        self.assertAlmostEqual(info["reward_terms"]["kill"], 0.3)
        self.assertFalse(term)  # NPCs 2 and 3 still live
        self.assertIsNone(info["outcome"])

    def test_terminated_on_death(self):
        env = BombermanGymEnv(None)
        env.reset(seed=0)
        env.step(A[Action.BOMB])
        for step in range(2, 2 + GAME_RULES["bomb_fuse"]):
            _, r, term, trunc, info = env.step(A[Action.WAIT])
            if step < 1 + GAME_RULES["bomb_fuse"]:
                self.assertFalse(term)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info["outcome"], "loss")
        self.assertAlmostEqual(r, -1.001)

    def test_truncated_at_max_steps(self):
        env = BombermanGymEnv(None, EnvConfig(max_steps=5))
        env.reset(seed=0)
        for step in range(1, 6):
            _, _, term, trunc, info = env.step(A[Action.WAIT])
            self.assertFalse(term)
            self.assertEqual(trunc, step == 5)
        self.assertEqual(info["steps"], 5)
        self.assertIsNone(info["outcome"])

    def test_terminated_wins_over_truncated_on_the_same_step(self):
        env = BombermanGymEnv(None, EnvConfig(max_steps=1 + GAME_RULES["bomb_fuse"]))
        env.reset(seed=0)
        env.step(A[Action.BOMB])
        for _ in range(GAME_RULES["bomb_fuse"]):
            _, _, term, trunc, info = env.step(A[Action.WAIT])
        self.assertEqual(info["steps"], 1 + GAME_RULES["bomb_fuse"])
        self.assertTrue(term)
        self.assertFalse(trunc)

    def test_default_max_steps_is_800(self):
        self.assertEqual(EnvConfig().max_steps, 800)
        env = BombermanGymEnv(None)
        env.reset(seed=0)
        trunc = False
        for step in range(1, 801):
            _, _, term, trunc, _ = env.step(A[Action.WAIT])
            self.assertFalse(term)
            if step < 800:
                self.assertFalse(trunc)
        self.assertTrue(trunc)

    def test_info_contents(self):
        env = BombermanGymEnv("random")
        _, info = env.reset(seed=2)
        self.assertEqual(info, {"outcome": None, "steps": 0, "events": [],
                                "episode_seed": 2})
        _, _, _, _, info = env.step(A[Action.WAIT])
        self.assertEqual(set(info), {"outcome", "steps", "events",
                                     "episode_seed", "reward_terms"})
        self.assertEqual(info["steps"], 1)
        info["events"].append({"type": "fake"})  # a copy: the env is unaffected
        self.assertNotIn({"type": "fake"}, env.state["events"])


@unittest.skipIf(np is None, MISSING)
class OpponentsTest(unittest.TestCase):
    def test_ladder_name(self):
        env = BombermanGymEnv("noisy25")
        self.assertEqual(sorted(env.controllers), [1, 2, 3])
        for bot in env.controllers.values():
            self.assertIsInstance(bot, NoisyBot)
            self.assertEqual(bot.epsilon, 0.25)
        self.assertIsInstance(BombermanGymEnv("rule").controllers[3], RuleBot)

    def test_single_player(self):
        for opponents in (None, []):
            env = BombermanGymEnv(opponents)
            env.reset(seed=0)
            self.assertEqual(len(env.state["characters"]), 1)

    def test_bad_opponents(self):
        with self.assertRaises(ValueError):
            BombermanGymEnv("expert")
        with self.assertRaises(ValueError):
            BombermanGymEnv([WaitBot()] * 4)
        with self.assertRaises(TypeError):
            BombermanGymEnv([object()])

    def test_npcs_without_controller_always_wait(self):
        env = BombermanGymEnv([RandomBot()])
        env.reset(seed=1)
        self.assertEqual(len(env.state["characters"]), 4)
        start = [(c["x"], c["y"]) for c in env.state["characters"][2:]]
        for _ in range(40):
            _, _, term, trunc, _ = env.step(A[Action.WAIT])
            if term or trunc:
                break
            now = [(c["x"], c["y"]) for c in env.state["characters"][2:]]
            self.assertEqual(now, start)
            self.assertFalse([b for b in env.state["bombs"] if b["owner"] in (2, 3)])

    def test_controllers_get_their_seeds_and_the_previous_state(self):
        calls = []

        class Recorder(WaitBot):
            def reset(self, seed=None):
                calls.append(("reset", seed))

            def act(self, state, me):
                calls.append(("act", me, state["steps"]))
                return Action.WAIT

        env = BombermanGymEnv([Recorder(), Recorder()])
        env.reset(seed=7)
        env.step(A[Action.WAIT])
        env.step(A[Action.WAIT])
        self.assertEqual(calls[:2], [("reset", bot_seed(7, 1)), ("reset", bot_seed(7, 2))])
        self.assertEqual(calls[2:], [("act", 1, 0), ("act", 2, 0),
                                     ("act", 1, 1), ("act", 2, 1)])


@unittest.skipIf(np is None, MISSING)
class SeedingTest(unittest.TestCase):
    def rollout(self, seed, n=80, opponents="noisy50"):
        env = BombermanGymEnv(opponents)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(99)
        out = [obs]
        for _ in range(n):
            obs, r, term, trunc, info = env.step(int(rng.integers(6)))
            out.append((obs, r, term, trunc, info["events"]))
            if term or trunc:
                obs, _ = env.reset()
                out.append(obs)
        return out

    def test_same_seed_same_trajectory(self):
        a, b = self.rollout(5), self.rollout(5)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            if isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
            else:
                np.testing.assert_array_equal(x[0], y[0])
                self.assertEqual(x[1:], y[1:])

    def test_seed_matches_the_engine_layout(self):
        env = BombermanGymEnv("random")
        env.reset(seed=12)
        engine = BombermanEnv(with_npcs=True, random_crates=True, **GAME_RULES)
        self.assertEqual(env.state["grid"], engine.reset(seed=12)["grid"])

    def test_unseeded_resets_follow_the_last_seed(self):
        def seeds(first):
            env = BombermanGymEnv(None)
            env.reset(seed=first)
            return [env.reset()[1]["episode_seed"] for _ in range(4)]
        self.assertEqual(seeds(3), seeds(3))
        self.assertNotEqual(seeds(3), seeds(4))
        self.assertEqual(len(set(seeds(3))), 4)


@unittest.skipIf(np is None, MISSING)
class CheckEnvTest(unittest.TestCase):
    def test_check_env(self):
        for opponents in ("random", "rule", None):
            with self.subTest(opponents=opponents), warnings.catch_warnings():
                # Only warns that render modes cannot be tested without a
                # registered spec; the wrapper has no render modes.
                warnings.simplefilter("ignore", UserWarning)
                check_env(BombermanGymEnv(opponents))


@unittest.skipIf(np is None, MISSING)
class ConfigTest(unittest.TestCase):
    def test_defaults_are_v0(self):
        r = RewardConfig()
        self.assertEqual((r.win, r.loss, r.draw, r.kill, r.crate, r.per_step, r.bump),
                         (1.0, -1.0, -0.5, 0.3, 0.02, -0.001, -0.01))

    def test_overrides_nested_and_json(self):
        cfg = apply_overrides(EnvConfig(), {"max_steps": 600, "reward": {"kill": 1}})
        self.assertEqual(cfg.max_steps, 600)
        self.assertEqual(cfg.reward.kill, 1.0)
        self.assertEqual(cfg.reward.win, 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"reward": {"bump": -0.05}}))
            self.assertEqual(load_config(EnvConfig(), path).reward.bump, -0.05)
        self.assertEqual(load_config(EnvConfig(), None), EnvConfig())
        self.assertEqual(config_to_dict(cfg)["reward"]["kill"], 1.0)

    def test_bad_overrides(self):
        with self.assertRaises(KeyError):
            apply_overrides(EnvConfig(), {"max_step": 5})
        with self.assertRaises(KeyError):
            apply_overrides(EnvConfig(), {"reward": {"kil": 1}})
        with self.assertRaises(TypeError):
            apply_overrides(EnvConfig(), {"max_steps": 5.5})
        with self.assertRaises(TypeError):
            apply_overrides(EnvConfig(), {"reward": 3})


@unittest.skipIf(np is None, MISSING)
class IsolationTest(unittest.TestCase):
    def test_rl_does_not_import_pygame(self):
        code = ("import sys; import rl.bomberman_env, rl.observation, rl.config; "
                "print('pygame' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")

    def test_game_rules_unchanged_after_use(self):
        env = BombermanGymEnv("random")
        env.reset(seed=0)
        env.step(0)
        self.assertEqual(dict(GAME_RULES), {"bomb_fuse": 4, "flame_ticks": 4, "buffs": GAME_RULES["buffs"]})


if __name__ == "__main__":
    unittest.main()
