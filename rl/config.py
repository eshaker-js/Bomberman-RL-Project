"""Every tunable value of the RL code, in one place.

All values here are **v0 / provisional**: the team reviews them before real
training starts. Change them here, or without touching code by passing a JSON
file of overrides to a script (``--config my_run.json``)::

    {"max_steps": 600, "reward": {"kill": 0.5}}

``load_config`` applies such a file on top of the defaults, field by field,
and rejects unknown keys so a typo can never be silently ignored. Every run
saves the resolved config next to its results (``config_to_dict``).

The game rules themselves are not here: the wrapper always plays
``GAME_RULES`` with random crates, like the real game.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RewardConfig:
    """Reward terms of the full-game wrapper (v0). See ``rl.bomberman_env.step_reward``."""

    win: float = 1.0  # v0: the agent stands on the open gate
    loss: float = -1.0  # v0: the agent died (including by its own bomb)
    draw: float = -0.5  # v0: the agent died together with the last NPCs
    kill: float = 0.3  # v0: per opponent death the agent's blast caused
    crate: float = 0.02  # v0: per crate the agent's blast destroyed
    per_step: float = -0.001  # v0: every step
    bump: float = -0.01  # v0: a move action that left the position unchanged
    buff: float = 0.0  # v0 (off): per buff picked up that raised one of my levels


@dataclass(frozen=True)
class EnvConfig:
    """The full-game Gymnasium wrapper (v0)."""

    max_steps: int = 800  # v0: truncation limit, in engine steps
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(frozen=True)
class TabularConfig:
    """Tabular Q-learning on the tiny stage (v0). See ``rl.tabular_env``."""

    episodes: int = 3000  # v0
    max_steps: int = 200  # v0: tiny-stage truncation limit
    alpha: float = 0.1  # v0: learning rate
    gamma: float = 0.99  # v0: discount
    epsilon_start: float = 1.0  # v0
    epsilon_end: float = 0.05  # v0
    epsilon_decay_fraction: float = 0.8  # v0: linear decay over this share of episodes
    reward_crate: float = 1.0  # v0: a crate destroyed while alive (ends the episode)
    reward_death: float = -1.0  # v0: the agent died (ends the episode)
    reward_step: float = -0.001  # v0: every step


@dataclass(frozen=True)
class PPOConfig:
    """PPO (Stable-Baselines3) on the full-game wrapper (v0). See ``rl.train_ppo``."""

    total_timesteps: int = 10_000  # v0: smoke-run size; real runs override it
    opponents: str = "random"  # v0: ladder level of the three NPCs
    n_envs: int = 8  # v0: parallel envs (DummyVecEnv)
    n_steps: int = 256  # v0: steps per env per rollout (8 x 256 = 2048)
    batch_size: int = 256  # v0
    n_epochs: int = 10  # SB3 default
    learning_rate: float = 3e-4  # SB3 default
    gamma: float = 0.99  # SB3 default
    gae_lambda: float = 0.95  # SB3 default
    clip_range: float = 0.2  # SB3 default
    ent_coef: float = 0.01  # v0 (SB3 default is 0.0)
    vf_coef: float = 0.5  # SB3 default
    max_grad_norm: float = 0.5  # SB3 default
    conv_channels: tuple = (32, 64, 64)  # v0: three 3x3 conv layers, padding 1
    features_dim: int = 256  # v0: linear layer after the convolutions
    net_arch_pi: tuple = (64,)  # v0: policy head hidden layers
    net_arch_vf: tuple = (64,)  # v0: value head hidden layers
    env: EnvConfig = field(default_factory=EnvConfig)


def config_to_dict(config) -> dict:
    """The config as plain JSON-ready data (nested dataclasses become dicts)."""
    return dataclasses.asdict(config)


def apply_overrides(config, overrides: dict):
    """A copy of the dataclass ``config`` with ``overrides`` applied, recursively.

    Nested dataclasses take nested dicts. Unknown keys raise ``KeyError``;
    a value of the wrong kind (for example a dict for a number) raises
    ``TypeError``. Ints are accepted where a float is expected.
    """
    if not isinstance(overrides, dict):
        raise TypeError(f"overrides for {type(config).__name__} must be an object")
    fields = {f.name: f for f in dataclasses.fields(config)}
    changes = {}
    for key, value in overrides.items():
        if key not in fields:
            raise KeyError(f"unknown config key {key!r} for {type(config).__name__}; "
                           f"known keys: {sorted(fields)}")
        current = getattr(config, key)
        if dataclasses.is_dataclass(current):
            changes[key] = apply_overrides(current, value)
        else:
            changes[key] = _checked(key, current, value)
    return dataclasses.replace(config, **changes)


def _checked(key: str, current, value):
    """``value`` converted to the type of ``current``, or ``TypeError``."""
    if isinstance(current, bool) or isinstance(value, bool):
        if isinstance(current, bool) and isinstance(value, bool):
            return value
    elif isinstance(current, float) and isinstance(value, (int, float)):
        return float(value)
    elif isinstance(current, int) and isinstance(value, int):
        return value
    elif isinstance(current, str) and isinstance(value, str):
        return value
    elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(f"config key {key!r} expects {type(current).__name__}, "
                    f"got {value!r}")


def load_config(default, path: str | Path | None = None):
    """``default`` with the JSON overrides in ``path`` applied (``None``: unchanged)."""
    if path is None:
        return default
    with open(path, encoding="utf-8") as handle:
        return apply_overrides(default, json.load(handle))
