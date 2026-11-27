"""The safety filter (Task 13, part 2).

- ``rl.safety.action_mask`` on hand-made states: nothing to fear, one way
  out of a blast, trapped (everything allowed), a refused BOMB acting as
  WAIT, an opponent who could bomb (``cautious`` vs ``board``, and the
  rule bot's fallback), a dead agent, ``off``;
- on states from real games: the rule bot's own choice is always allowed
  by ``cautious``, every mask has at least one action, ``board`` allows
  every action ``cautious`` allows;
- a random agent that only picks allowed actions (``board``) never dies
  against idle bots, which never bomb;
- ``EnvConfig.action_mask``: default ``"off"``, round trip, unknown modes
  refused (also before a run folder exists); ``action_masks()`` reaches
  through the training wrappers;
- training: a tiny masked run is a ``MaskablePPO`` run, never takes a
  masked action in its rollouts, evaluates with the mask, works with stages
  and under ``subproc`` (spawn); a default run is still ``PPO``;
- ``ModelPolicy`` with a mask only picks allowed actions;
- ``init_model``: masked to masked continues, crossing kinds is refused
  before a run folder exists.

Needs numpy, gymnasium, torch, stable-baselines3 and sb3-contrib. Only
few-step runs, in temporary folders; every env is closed.
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
    import sb3_contrib  # noqa: F401
    import stable_baselines3  # noqa: F401
    import torch
except ImportError:  # pragma: no cover - depends on the machine
    torch = None

MISSING = "numpy/gymnasium/torch/stable-baselines3/sb3-contrib not installed; test skipped"

if torch is not None:
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    from agents.rule_bot import RuleBot
    from game.environment import Action, Cell
    from rl import train_ppo
    from rl.bomberman_env import BombermanGymEnv
    from rl.evaluate_agent import ModelPolicy
    from rl.safety import MASK_MODES, action_mask

TINY = {"n_envs": 2, "n_steps": 64, "batch_size": 32, "n_epochs": 2,
        "features_dim": 32, "conv_channels": [8, 8, 8], "env": {"max_steps": 40}}
MASKED = {**TINY, "env": {"max_steps": 40, "action_mask": "cautious"}}
UP, DOWN, LEFT, RIGHT, WAIT, BOMB = range(6)


def run_main(tmp: Path, overrides: dict) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    config = tmp / f"config_{len(list(tmp.glob('config_*.json')))}.json"
    config.write_text(json.dumps(overrides), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return train_ppo.main(["--seed", "0", "--runs-dir", str(tmp / "runs"),
                               "--config", str(config)])


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def blank_state(me_at=(1, 1)) -> dict:
    """A real game state with every crate removed, no bombs or flames, NPCs dead."""
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


def allowed(mask) -> list:
    return [i for i, ok in enumerate(mask) if ok]


@unittest.skipIf(torch is None, MISSING)
class HandMadeMaskTest(unittest.TestCase):
    def test_off_and_nothing_to_fear(self):
        state = blank_state()
        for mode in MASK_MODES:
            with self.subTest(mode=mode):
                mask = action_mask(state, 0, mode)
                self.assertEqual(mask.dtype, bool)
                self.assertEqual(allowed(mask), [UP, DOWN, LEFT, RIGHT, WAIT, BOMB])

    def test_one_way_out_of_a_blast(self):
        # A bomb two cells right of me goes off on the next action: only
        # stepping down into column 1 leaves its cross.
        state = blank_state()
        state["bombs"] = [bomb(3, 1, countdown=1)]
        for mode in ("board", "cautious"):
            with self.subTest(mode=mode):
                self.assertEqual(allowed(action_mask(state, 0, mode)), [DOWN])
        self.assertEqual(len(allowed(action_mask(state, 0, "off"))), 6)

    def test_trapped_allows_everything(self):
        state = blank_state()
        state["grid"][1][2] = Cell.BLOCK
        state["grid"][2][1] = Cell.BLOCK
        state["bombs"] = [bomb(1, 1, countdown=2, owner=0)]
        state["characters"][0]["bombs_left"] = 0
        for mode in ("board", "cautious"):
            with self.subTest(mode=mode):
                self.assertTrue(action_mask(state, 0, mode).all())

    def test_refused_bomb_acts_as_wait(self):
        state = blank_state()
        state["characters"][0]["bombs_left"] = 0
        self.assertTrue(action_mask(state, 0, "board")[BOMB])
        state["bombs"] = [bomb(3, 1, countdown=1)]
        mask = action_mask(state, 0, "board")
        self.assertFalse(mask[WAIT])
        self.assertFalse(mask[BOMB])

    def test_own_bomb_in_a_dead_end(self):
        # Corridor (1, 1)..(3, 1), closed on every side: with range 2 my
        # bomb reaches its end, with range 1 the far end is safe.
        state = blank_state()
        for x, y in ((1, 2), (3, 2), (4, 1)):
            state["grid"][y][x] = Cell.BLOCK
        self.assertFalse(action_mask(state, 0, "board")[BOMB])
        state["characters"][0]["range"] = 1
        self.assertTrue(action_mask(state, 0, "board")[BOMB])

    def test_cautious_counts_an_opponent_who_could_bomb(self):
        # Same corridor (range 1); an opponent with a bomb left stands at its
        # far end. Its possible bomb would cover the whole corridor.
        state = blank_state()
        for x, y in ((1, 2), (3, 2), (4, 1)):
            state["grid"][y][x] = Cell.BLOCK
        state["characters"][0]["range"] = 1
        opponent = state["characters"][1]
        opponent["dead"], opponent["x"], opponent["y"] = False, 3, 1
        self.assertTrue(action_mask(state, 0, "board").all())
        mask = action_mask(state, 0, "cautious")
        # No move survives the worst case, so the moves fall back to the
        # board's bombs (none): all allowed. My own bomb is not.
        self.assertEqual(allowed(mask), [UP, DOWN, LEFT, RIGHT, WAIT])
        opponent["bombs_left"] = 0
        self.assertTrue(action_mask(state, 0, "cautious").all())

    def test_dead_or_game_over_allows_everything(self):
        state = blank_state()
        state["bombs"] = [bomb(3, 1, countdown=1)]
        state["characters"][0]["dead"] = True
        self.assertTrue(action_mask(state, 0, "cautious").all())
        state = blank_state()
        state["bombs"] = [bomb(3, 1, countdown=1)]
        state["outcome"] = "loss"
        self.assertTrue(action_mask(state, 0, "cautious").all())

    def test_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown action_mask"):
            action_mask(blank_state(), 0, "own")


@unittest.skipIf(torch is None, MISSING)
class RealGamesTest(unittest.TestCase):
    def test_rule_bot_choice_is_always_allowed(self):
        env = BombermanGymEnv("rule", EnvConfig(max_steps=120))
        bot = RuleBot(seed=0)
        checked = narrowed = 0
        try:
            for seed in range(10):
                env.reset(seed=20_000 + seed)
                bot.reset(seed)
                done = False
                while not done:
                    state = env.state
                    choice = bot.act(state, 0)
                    cautious = action_mask(state, 0, "cautious")
                    board = action_mask(state, 0, "board")
                    self.assertTrue(cautious.any() and board.any())
                    self.assertTrue(np.all(board | ~cautious) or cautious.all())
                    self.assertTrue(cautious[list(Action).index(choice)], state["steps"])
                    checked += 1
                    narrowed += int(not cautious.all())
                    _, _, terminated, truncated, _ = env.step(list(Action).index(choice))
                    done = terminated or truncated
        finally:
            env.close()
        self.assertGreater(checked, 300)
        self.assertGreater(narrowed, 50)

    def test_random_masked_agent_never_dies_against_idle_bots(self):
        env = BombermanGymEnv("idle", EnvConfig(max_steps=150, action_mask="board"))
        rng = np.random.default_rng(0)
        bombs = deaths = 0
        try:
            for seed in range(15):
                env.reset(seed=30_000 + seed)
                done = False
                while not done:
                    choices = allowed(env.action_masks())
                    action = int(rng.choice(choices))
                    bombs += action == BOMB and not env.state["bombs"]
                    _, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                deaths += info["outcome"] in ("loss", "draw")
        finally:
            env.close()
        self.assertGreater(bombs, 20)
        self.assertEqual(deaths, 0)


class RecordMasks(BaseCallback if torch is not None else object):
    """Checks every rollout step: the chosen action is allowed by its mask."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.narrowed = 0

    def _on_step(self) -> bool:
        masks, actions = self.locals["action_masks"], self.locals["actions"]
        for mask, action in zip(masks, actions):
            assert mask[int(action)], (mask, action)
            self.narrowed += int(not mask.all())
        self.steps += len(actions)
        return True


@unittest.skipIf(torch is None, MISSING)
class ConfigTest(unittest.TestCase):
    def test_default_round_trip_and_refusals(self):
        self.assertEqual(EnvConfig().action_mask, "off")
        self.assertEqual(config_to_dict(PPOConfig())["env"]["action_mask"], "off")
        cfg = apply_overrides(PPOConfig(), {"env": {"action_mask": "board"}})
        self.assertEqual(apply_overrides(PPOConfig(), json.loads(json.dumps(
            config_to_dict(cfg)))), cfg)
        with self.assertRaises(TypeError):
            apply_overrides(EnvConfig(), {"action_mask": True})
        with self.assertRaisesRegex(ValueError, "unknown action_mask"):
            BombermanGymEnv("random", EnvConfig(action_mask="Cautious"))

    def test_unknown_mode_refused_before_a_run_folder(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            with self.assertRaisesRegex(ValueError, "unknown action_mask"):
                run_main(tmp, {**TINY, "total_timesteps": 128,
                               "env": {"action_mask": "own"}})
            self.assertFalse((tmp / "runs").exists())

    def test_masks_reach_through_the_training_wrappers(self):
        cfg = apply_overrides(PPOConfig(), MASKED)
        envs = train_ppo.make_envs(cfg, "rule", 0)
        try:
            envs.reset()
            masks = np.stack(envs.env_method("action_masks"))
            self.assertEqual(masks.shape, (2, 6))
        finally:
            envs.close()


@unittest.skipIf(torch is None, MISSING)
class MaskedTrainingTest(unittest.TestCase):
    def test_masked_model_never_takes_a_masked_action(self):
        cfg = apply_overrides(PPOConfig(), {**MASKED, "opponents": "rule"})
        model = train_ppo.make_model(cfg, 0)
        try:
            self.assertIsInstance(model, MaskablePPO)
            recorder = RecordMasks()
            model.learn(total_timesteps=512, callback=recorder)
        finally:
            model.get_env().close()
        self.assertEqual(recorder.steps, 512)
        self.assertGreater(recorder.narrowed, 0)

    def test_default_is_still_ppo(self):
        cfg = apply_overrides(PPOConfig(), TINY)
        model = train_ppo.make_model(cfg, 0)
        try:
            self.assertIs(type(model), PPO)
        finally:
            model.get_env().close()

    def test_tiny_runs_with_eval_stages_and_subproc(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            run = run_main(tmp / "a", {**MASKED, "total_timesteps": 256, "eval_every": 128,
                                       "eval_games": 2, "opponents": "idle",
                                       "eval_opponents": "random",
                                       "stages": [{"opponents": "idle", "timesteps": 128},
                                                  {"opponents": "noisy25", "timesteps": 128}]})
            meta = read_json(run / "meta.json")
            self.assertEqual(meta["algorithm"], "MaskablePPO")
            self.assertEqual(read_json(run / "config.json")["env"]["action_mask"], "cautious")
            self.assertEqual(len(meta["stages"]), 2)
            self.assertTrue((run / "eval_curve.csv").exists())
            loaded = MaskablePPO.load(run / "model.zip", device="cpu")
            self.assertEqual(type(loaded).__name__, "MaskablePPO")
            sub = run_main(tmp / "b", {**MASKED, "total_timesteps": 128, "vec_env": "subproc"})
            self.assertEqual(read_json(sub / "meta.json")["algorithm"], "MaskablePPO")
            plain = run_main(tmp / "c", {**TINY, "total_timesteps": 128})
            self.assertEqual(read_json(plain / "meta.json")["algorithm"], "PPO")

    def test_model_policy_with_mask_picks_allowed_actions(self):
        cfg = apply_overrides(PPOConfig(), MASKED)
        model = train_ppo.make_model(cfg, 0)
        try:
            state = blank_state()
            state["bombs"] = [bomb(3, 1, countdown=1)]
            env = BombermanGymEnv("idle", cfg.env)
            obs = env._observe(state)
            env.close()
            for deterministic in (True, False):
                policy = ModelPolicy(model, deterministic, action_mask="cautious")
                for _ in range(20):
                    self.assertEqual(policy.act(obs, state), DOWN)
            self.assertIsNone(ModelPolicy(model, action_mask="off").action_mask)
        finally:
            model.get_env().close()


@unittest.skipIf(torch is None, MISSING)
class InitModelKindsTest(unittest.TestCase):
    def test_masked_continues_and_kinds_do_not_cross(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            masked = run_main(tmp / "m", {**MASKED, "total_timesteps": 128})
            plain = run_main(tmp / "p", {**TINY, "total_timesteps": 128})
            go_on = run_main(tmp / "go", {**MASKED, "total_timesteps": 128,
                                          "init_model": str(masked / "model.zip")})
            record = read_json(go_on / "meta.json")["init_model"]
            self.assertTrue(record["verified"])
            self.assertEqual(record["algorithm"], "MaskablePPO")
            board = run_main(tmp / "board", {**TINY, "total_timesteps": 128,
                                             "env": {"max_steps": 40, "action_mask": "board"},
                                             "init_model": str(masked / "model.zip")})
            self.assertTrue(read_json(board / "meta.json")["init_model"]["verified"])
            for name_, overrides in (
                    ("x1", {**MASKED, "init_model": str(plain / "model.zip")}),
                    ("x2", {**TINY, "init_model": str(masked / "model.zip")})):
                with self.subTest(name_), self.assertRaisesRegex(ValueError, "same kind"):
                    run_main(tmp / name_, {**overrides, "total_timesteps": 128})
                self.assertFalse((tmp / name_ / "runs").exists())


if __name__ == "__main__":
    unittest.main()
