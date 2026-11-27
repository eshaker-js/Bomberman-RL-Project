"""The danger map in the observation (Task 13, part 1).

- ``rl.danger.danger_planes`` on hand-made states: one bomb in the open, a
  wall and a crate stopping a ray, a chain reaction that sets a later bomb
  off early, a burn-level bomb, lingering flames, a dead agent, and the
  ``bomb_escape`` plane (free corner, boxed in, no bomb left, bomb already
  on my cell, an opponent who could bomb next to me);
- ``danger_soon`` against the engine itself: on states from real games,
  copies of the engine are stepped with every character waiting, and every
  cell the plane marks really burns first on the predicted action;
- ``bomb_escape`` against the rule bot's own bombing check, on the same
  states (the bot bombs only where the plane says 1);
- ``EnvConfig.danger_obs``: off by default, the observation and its space
  unchanged when off (equal to ``encode_observation`` on every step of a
  game), 22 channels in [0, 1] when on, the first 20 equal to the v1 ones;
- a tiny ``rl.train_ppo`` run with the danger map, its ``config.json``,
  and ``init_model`` from a 20-channel model refused before a run folder
  exists.

Needs numpy, gymnasium, torch and stable-baselines3, like the other RL
tests. Only few-step runs, in temporary folders; every env is closed.
"""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.config import EnvConfig, PPOConfig, apply_overrides, config_to_dict  # noqa: E402

try:
    import gymnasium  # noqa: F401
    import numpy as np
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3 not installed; test skipped"

if torch is not None:
    from agents.rule_bot import RuleBot
    from game.environment import Action, Cell
    from rl import train_ppo
    from rl.bomberman_env import BombermanGymEnv, observation_shape
    from rl.danger import (
        BOMB_ESCAPE,
        DANGER_CHANNEL_NAMES,
        DANGER_SOON,
        N_DANGER_CHANNELS,
        danger_planes,
        first_burn,
    )
    from rl.observation import N_CHANNELS, OBS_SHAPE, encode_observation

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}


def run_main(tmp: Path, overrides: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / f"config_{len(list(tmp.glob('config_*.json')))}.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return train_ppo.main(["--seed", "0", "--runs-dir", str(tmp / "runs"),
                               "--config", str(config)])


def blank_state(me_at=(9, 9)) -> dict:
    """A real game state with every crate removed, no bombs or flames, NPCs dead.

    The agent (0) stands at ``me_at``. Walls are the fixed ones: the border
    and every cell with both coordinates even.
    """
    env = BombermanGymEnv("idle")
    try:
        env.reset(seed=5)
        state = copy.deepcopy(env.state)
    finally:
        env.close()
    state["grid"] = [[Cell.EMPTY if cell is Cell.BLOCK else cell for cell in row]
                     for row in state["grid"]]
    state["bombs"], state["flames"] = [], []
    for character in state["characters"][1:]:
        character["dead"] = True
    state["characters"][0]["x"], state["characters"][0]["y"] = me_at
    return state


def bomb(x, y, countdown, owner=1, blast_range=2, burn=0) -> dict:
    return {"owner": owner, "x": x, "y": y, "countdown": countdown,
            "range": blast_range, "burn": burn}


def soon(planes, x, y) -> float:
    return float(planes[DANGER_SOON, y, x])


@unittest.skipIf(torch is None, MISSING)
class DangerSoonTest(unittest.TestCase):
    """Plane 0 on hand-made states (agent far away at (9, 9), unless said)."""

    def test_no_bombs_no_danger(self):
        planes = danger_planes(blank_state())
        self.assertEqual(planes.shape, (N_DANGER_CHANNELS, 11, 11))
        self.assertEqual(planes.dtype, np.float32)
        self.assertEqual(float(planes[DANGER_SOON].sum()), 0.0)

    def test_one_bomb_in_the_open(self):
        state = blank_state()
        state["bombs"] = [bomb(5, 1, countdown=3)]
        planes = danger_planes(state)
        burned = {(5, 1), (4, 1), (3, 1), (6, 1), (7, 1), (5, 2), (5, 3)}
        for y in range(11):
            for x in range(11):
                want = 0.5 if (x, y) in burned else 0.0  # (4 + 1 - 3) / 4
                self.assertAlmostEqual(soon(planes, x, y), want, msg=(x, y))

    def test_values_by_countdown(self):
        for countdown, want in ((1, 1.0), (2, 0.75), (3, 0.5), (4, 0.25)):
            state = blank_state()
            state["bombs"] = [bomb(5, 1, countdown)]
            with self.subTest(countdown=countdown):
                self.assertAlmostEqual(soon(danger_planes(state), 5, 1), want)

    def test_wall_stops_the_ray(self):
        state = blank_state()
        state["bombs"] = [bomb(3, 1, countdown=2)]
        planes = danger_planes(state)
        self.assertGreater(soon(planes, 3, 3), 0.0)  # (3, 2) is floor
        state["bombs"] = [bomb(4, 1, countdown=2)]  # (4, 2) is a wall
        planes = danger_planes(state)
        self.assertEqual(soon(planes, 4, 2), 0.0)
        self.assertEqual(soon(planes, 4, 3), 0.0)

    def test_crate_stops_the_ray_and_burns(self):
        state = blank_state()
        state["grid"][1][6] = Cell.BLOCK
        state["bombs"] = [bomb(5, 1, countdown=1)]
        planes = danger_planes(state)
        self.assertEqual(soon(planes, 6, 1), 1.0)
        self.assertEqual(soon(planes, 7, 1), 0.0)

    def test_chain_reaction_sets_the_later_bomb_off_early(self):
        state = blank_state()
        state["bombs"] = [bomb(5, 1, countdown=1), bomb(7, 1, countdown=4)]
        planes = danger_planes(state)
        for cell in ((7, 1), (8, 1), (9, 1), (7, 2), (7, 3)):
            self.assertEqual(soon(planes, *cell), 1.0, cell)

    def test_burn_level_keeps_the_first_time(self):
        state = blank_state()
        state["bombs"] = [bomb(5, 1, countdown=3, burn=2)]
        self.assertAlmostEqual(soon(danger_planes(state), 6, 1), 0.5)

    def test_lingering_flames(self):
        state = blank_state()
        state["flames"] = [{"x": 3, "y": 9, "ticks_left": 3, "owners": [1]},
                           {"x": 5, "y": 9, "ticks_left": 1, "owners": [1]}]
        planes = danger_planes(state)
        self.assertEqual(soon(planes, 3, 9), 1.0)
        self.assertEqual(soon(planes, 5, 9), 0.0)  # out by the next action

    def test_dead_agent_sees_nothing(self):
        state = blank_state()
        state["bombs"] = [bomb(5, 1, countdown=1)]
        state["characters"][0]["dead"] = True
        self.assertEqual(float(danger_planes(state).sum()), 0.0)


@unittest.skipIf(torch is None, MISSING)
class BombEscapeTest(unittest.TestCase):
    """Plane 1: a full plane, 1 when bombing now still leaves a way out."""

    def escape(self, state) -> np.ndarray:
        return danger_planes(state)[BOMB_ESCAPE]

    def test_open_corner_can_escape(self):
        plane = self.escape(blank_state(me_at=(1, 1)))
        self.assertTrue(np.all(plane == 1.0))

    def test_boxed_in_cannot(self):
        state = blank_state(me_at=(1, 1))
        state["grid"][1][2] = Cell.BLOCK
        state["grid"][2][1] = Cell.BLOCK
        self.assertTrue(np.all(self.escape(state) == 0.0))

    def test_dead_end_too_short_cannot(self):
        # A dead-end corridor (1, 1)..(3, 1): my range 2 reaches its end.
        state = blank_state(me_at=(1, 1))
        state["grid"][2][1] = Cell.BLOCK
        state["grid"][2][3] = Cell.BLOCK
        state["grid"][1][4] = Cell.BLOCK
        self.assertTrue(np.all(self.escape(state) == 0.0))
        state["characters"][0]["range"] = 1  # now (3, 1) is out of reach
        self.assertTrue(np.all(self.escape(state) == 1.0))

    def test_no_bomb_left_or_bomb_on_my_cell(self):
        state = blank_state(me_at=(1, 1))
        state["characters"][0]["bombs_left"] = 0
        self.assertTrue(np.all(self.escape(state) == 0.0))
        state = blank_state(me_at=(1, 1))
        state["bombs"] = [bomb(1, 1, countdown=3, owner=1)]
        self.assertTrue(np.all(self.escape(state) == 0.0))

    def test_an_opponent_who_could_bomb_counts(self):
        # Corridor (1,1)..(3,1); my only way out is down column 3, where an
        # opponent stands with a bomb left. Board-only there is a way out.
        state = blank_state(me_at=(1, 1))
        state["grid"][2][1] = Cell.BLOCK
        state["grid"][1][4] = Cell.BLOCK
        state["characters"][0]["range"] = 1
        self.assertTrue(np.all(self.escape(state) == 1.0))
        opponent = state["characters"][1]
        opponent["dead"], opponent["x"], opponent["y"] = False, 3, 3
        self.assertTrue(np.all(self.escape(state) == 0.0))
        opponent["bombs_left"] = 0
        self.assertTrue(np.all(self.escape(state) == 1.0))


def game_states(n_games: int, max_steps: int = 120) -> list:
    """``(engine copy, state)`` pairs from real games: a rule bot agent vs 3 rule bots."""
    pairs = []
    env = BombermanGymEnv("rule", EnvConfig(max_steps=max_steps))
    agent = RuleBot(seed=0)
    try:
        for seed in range(n_games):
            env.reset(seed=10_000 + seed)
            agent.reset(seed)
            done = False
            while not done:
                if env.state["bombs"] or env.state["flames"]:
                    pairs.append((copy.deepcopy(env.engine), copy.deepcopy(env.state)))
                action = list(Action).index(agent.act(env.state, 0))
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
    finally:
        env.close()
    return pairs


@unittest.skipIf(torch is None, MISSING)
class AgainstTheEngineTest(unittest.TestCase):
    """The planes on states from real games, checked against the engine and the rule bot."""

    @classmethod
    def setUpClass(cls):
        cls.pairs = game_states(12)

    def test_enough_states(self):
        self.assertGreater(len(self.pairs), 150)

    def test_marked_cells_burn_first_on_the_predicted_action(self):
        """Step a copy of the engine with everyone waiting and watch the flames.

        A character on a cell dies on action ``t`` exactly when the cell is
        in ``flames`` after step ``t``. The rule bot treats crates as
        permanent, so states where a crate breaks before the horizon are
        skipped (a ray can then reach further than planned); the check stops
        at the agent's death (the game ends there).
        """
        checked = 0
        for engine, state in self.pairs:
            if state["characters"][0]["dead"] or state["outcome"] is not None:
                continue
            predicted = first_burn(state)
            horizon = state["rules"]["bomb_fuse"]
            seen, broke = {}, False
            sim = copy.deepcopy(engine)
            for t in range(1, horizon + 1):
                after = sim.step(Action.WAIT, {})
                broke |= any(e["type"] == "block_destroyed" for e in after["events"])
                for flame in after["flames"]:
                    seen.setdefault((flame["x"], flame["y"]), t)
                if after["outcome"] is not None:
                    break
            if broke:
                continue
            last = t
            for cell, when in predicted.items():
                if when <= last:
                    self.assertEqual(seen.get(cell), when, (cell, state["steps"]))
            for cell, when in seen.items():
                self.assertEqual(predicted.get(cell), when, (cell, state["steps"]))
            planes = danger_planes(state)
            fuse = state["rules"]["bomb_fuse"]
            for (x, y), when in predicted.items():
                self.assertAlmostEqual(float(planes[DANGER_SOON, y, x]),
                                       (fuse + 1 - when) / fuse)
            checked += 1
        self.assertGreater(checked, 50)

    def test_the_rule_bot_bombs_only_where_escape_is_one(self):
        bombed = 0
        for _, state in self.pairs:
            if state["characters"][0]["dead"] or state["outcome"] is not None:
                continue
            planes = danger_planes(state)
            self.assertTrue(np.all((planes >= 0.0) & (planes <= 1.0)))
            escape = planes[BOMB_ESCAPE]
            self.assertTrue(np.all(escape == escape[0, 0]))  # a full plane
            if RuleBot(seed=1).act(state, 0) is Action.BOMB:
                bombed += 1
                self.assertEqual(float(escape[0, 0]), 1.0, state["steps"])
        self.assertGreater(bombed, 5)


@unittest.skipIf(torch is None, MISSING)
class EnvSwitchTest(unittest.TestCase):
    def test_config_default_and_round_trip(self):
        self.assertFalse(EnvConfig().danger_obs)
        self.assertFalse(config_to_dict(PPOConfig())["env"]["danger_obs"])
        cfg = apply_overrides(PPOConfig(), {"env": {"danger_obs": True}})
        self.assertTrue(cfg.env.danger_obs)
        self.assertEqual(apply_overrides(PPOConfig(), json.loads(json.dumps(
            config_to_dict(cfg)))), cfg)
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                apply_overrides(EnvConfig(), {"danger_obs": value})

    def test_off_is_the_v1_observation_on_every_step(self):
        env = BombermanGymEnv("noisy25", EnvConfig(max_steps=150))
        try:
            self.assertEqual(env.observation_space.shape, OBS_SHAPE)
            self.assertEqual(observation_shape(EnvConfig()), OBS_SHAPE)
            obs, _ = env.reset(seed=3)
            rng = np.random.default_rng(0)
            done = False
            while True:
                self.assertTrue(np.array_equal(obs, encode_observation(env.state, 0)))
                if done:
                    break
                obs, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
                done = terminated or truncated
        finally:
            env.close()

    def test_on_adds_two_planes_after_the_v1_ones(self):
        self.assertEqual(N_CHANNELS + N_DANGER_CHANNELS, 22)
        self.assertEqual(DANGER_CHANNEL_NAMES, ("danger_soon", "bomb_escape"))
        env = BombermanGymEnv("noisy25", EnvConfig(max_steps=150, danger_obs=True))
        try:
            self.assertEqual(env.observation_space.shape, (22, 11, 11))
            obs, _ = env.reset(seed=3)
            rng = np.random.default_rng(0)
            danger_seen = False
            done = False
            while True:
                self.assertTrue(env.observation_space.contains(obs))
                self.assertTrue(np.array_equal(obs[:N_CHANNELS],
                                               encode_observation(env.state, 0)))
                self.assertTrue(np.array_equal(obs[N_CHANNELS:], danger_planes(env.state, 0)))
                danger_seen |= bool(obs[N_CHANNELS + DANGER_SOON].any())
                if done:
                    break
                obs, _, terminated, truncated, _ = env.step(int(rng.integers(6)))
                done = terminated or truncated
            self.assertTrue(danger_seen)
        finally:
            env.close()


@unittest.skipIf(torch is None, MISSING)
class TrainingWithDangerMapTest(unittest.TestCase):
    def test_tiny_run_and_init_model_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            on = {**TINY, "total_timesteps": 128, "eval_every": 128, "eval_games": 2,
                  "env": {"max_steps": 40, "danger_obs": True}}
            run = run_main(tmp / "on", on)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["env"]["danger_obs"])
            self.assertTrue((run / "model.zip").exists())
            plain = run_main(tmp / "plain", {**TINY, "total_timesteps": 128})
            runs = tmp / "refused" / "runs"
            with self.assertRaisesRegex(ValueError, "observation and action spaces"):
                run_main(tmp / "refused", {**on, "init_model": str(plain / "model.zip")})
            self.assertFalse(runs.exists() and any(runs.iterdir()))
            back = {**TINY, "total_timesteps": 128, "init_model": str(run / "model.zip")}
            with self.assertRaisesRegex(ValueError, "observation and action spaces"):
                run_main(tmp / "refused", back)
            continued = run_main(tmp / "continued", {**on, "init_model": str(run / "model.zip")})
            meta = json.loads((continued / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["init_model"]["verified"])


if __name__ == "__main__":
    unittest.main()
