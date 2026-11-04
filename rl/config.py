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


@dataclass(frozen=True)
class EnvConfig:
    """The full-game Gymnasium wrapper (v0)."""

    max_steps: int = 800  # v0: truncation limit, in engine steps
    reward: RewardConfig = field(default_factory=RewardConfig)


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
    elif current is None or isinstance(current, (list, tuple)):
        return value
    raise TypeError(f"config key {key!r} expects {type(current).__name__}, "
                    f"got {value!r}")


def load_config(default, path: str | Path | None = None):
    """``default`` with the JSON overrides in ``path`` applied (``None``: unchanged)."""
    if path is None:
        return default
    with open(path, encoding="utf-8") as handle:
        return apply_overrides(default, json.load(handle))
