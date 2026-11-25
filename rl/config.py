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
    # Task 10 shaping terms, both off (0.0) by default so training is exactly v0.
    wait: float = 0.0  # per step on which the agent chose WAIT (or a refused BOMB)
    # Per own bomb that exploded and broke at least one crate, while the agent
    # is alive (counted as min(bombs exploded, crates broken); see step_reward).
    bomb_survived: float = 0.0


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
class Stage:
    """One curriculum stage (Task 11): train against ``opponents`` for ``timesteps`` steps.

    ``opponents`` is a ladder level or ``"idle"``; ``timesteps`` is a whole
    number of rollouts (``n_envs x n_steps``). See ``rl.curriculum``.
    """

    opponents: str
    timesteps: int


@dataclass(frozen=True)
class PPOConfig:
    """PPO (Stable-Baselines3) on the full-game wrapper (v0). See ``rl.train_ppo``."""

    total_timesteps: int = 10_000  # v0: smoke-run size; real runs override it
    opponents: str = "random"  # v0: ladder level of the three NPCs, or "idle" (Task 11, training only)
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
    # How training runs (Task 8). None of these changes what PPO learns; the
    # defaults are exactly the behaviour before they existed.
    vec_env: str = "dummy"  # "dummy": envs in one process; "subproc": one process each (spawn)
    torch_threads: int = 0  # CPU threads of torch; 0 = torch's own default
    high_priority: bool = False  # Windows: raise the training processes to "High"
    checkpoint_every: int = 0  # save checkpoints/model_<steps>.zip every N steps; 0 = off
    checkpoint_keep_every: int = 1_000_000  # keep one checkpoint per N steps, plus the latest
    eval_every: int = 0  # evaluate during training every N steps; 0 = off (N set after the benchmark)
    eval_games: int = 50  # games per evaluation during training, the same seeds every time
    # Task 9: where the network trains. "cpu" (default) is exactly the
    # behaviour before; "cuda" trains on the GPU. Evaluation stays on the CPU.
    device: str = "cpu"
    # Task 10: the opponents of the evaluation during training (a ladder
    # level). None = the same as ``opponents``, exactly the behaviour before.
    # Lets a run that trains against noisy25 still draw its learning curve
    # against random bots, comparable with the other runs.
    eval_opponents: str | None = None
    # Task 11: curriculum stages, trained in order by the same model in one
    # run. Empty (default) = no stages, exactly the behaviour before. When
    # set, the stages add up to ``total_timesteps``, the first stage's
    # opponents equal ``opponents``, and ``eval_opponents`` must be set (all
    # checked by ``rl.curriculum.check_stages``). In JSON: a list of
    # ``{"opponents": ..., "timesteps": ...}`` objects.
    stages: tuple[Stage, ...] = ()


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
            changes[key] = _checked(key, current, value, fields[key].type)
    return dataclasses.replace(config, **changes)


# A field declared ``str | None`` (Task 10: ``PPOConfig.eval_opponents``).
OPTIONAL_STR = str | None
# A field declared as a tuple of stages (Task 11: ``PPOConfig.stages``).
STAGES = tuple[Stage, ...]
STAGE_KEYS = ("opponents", "timesteps")


def _stages(key: str, value) -> tuple:
    """``value`` (a list of stage objects, or ``Stage`` instances) as a tuple of ``Stage``.

    Only the form is checked here (keys and types); whether the stages fit
    the run (rollouts, total, opponents) is ``rl.curriculum.check_stages``.
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"config key {key!r} expects a list of stages, got {value!r}")
    stages = []
    for index, stage in enumerate(value):
        if isinstance(stage, Stage):
            stage = dataclasses.asdict(stage)
        if not isinstance(stage, dict):
            raise TypeError(f"{key}[{index}] must be an object with keys "
                            f"{list(STAGE_KEYS)}, got {stage!r}")
        if set(stage) != set(STAGE_KEYS):
            raise KeyError(f"{key}[{index}] needs exactly the keys {list(STAGE_KEYS)}, "
                           f"got {sorted(stage)}")
        opponents, timesteps = stage["opponents"], stage["timesteps"]
        if not isinstance(opponents, str):
            raise TypeError(f"{key}[{index}].opponents expects str, got {opponents!r}")
        if isinstance(timesteps, bool) or not isinstance(timesteps, int):
            raise TypeError(f"{key}[{index}].timesteps expects int, got {timesteps!r}")
        stages.append(Stage(opponents, timesteps))
    return tuple(stages)


def _checked(key: str, current, value, declared=None):
    """``value`` converted to the type of ``current``, or ``TypeError``.

    A field declared ``str | None`` (``declared``) takes a string or ``None``
    (JSON ``null``), whatever its current value; a field declared as a tuple
    of ``Stage`` takes a list of stages (``_stages``); every other field is
    checked against the type of its current value, as before.
    """
    if declared == STAGES:
        return _stages(key, value)
    if declared == OPTIONAL_STR:
        if value is None or isinstance(value, str):
            return value
    elif isinstance(current, bool) or isinstance(value, bool):
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
    expected = "str or null" if declared == OPTIONAL_STR else type(current).__name__
    raise TypeError(f"config key {key!r} expects {expected}, "
                    f"got {value!r}")


def load_config(default, path: str | Path | None = None):
    """``default`` with the JSON overrides in ``path`` applied (``None``: unchanged)."""
    if path is None:
        return default
    with open(path, encoding="utf-8") as handle:
        return apply_overrides(default, json.load(handle))
