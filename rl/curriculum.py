"""Curriculum stages of one PPO run: the checks and the seeds (Task 11).

``PPOConfig.stages`` lists the stages a run trains in order, each against
its own opponents for a fixed number of steps, the same for every seed.
The same model goes on from stage to stage; only the training envs (and so
the opponents) change. The training loop itself is in ``rl.train_ppo``.

This module needs no torch or Stable-Baselines3, so the plan runner
(``rl.run_ppo_seeds``) can refuse a bad plan before it starts.
"""

from rl.bomberman_env import TRAINING_OPPONENTS
from rl.seeds import check_training_seed


def stage_seed(seed: int, index: int, n_envs: int) -> int:
    """The seed of stage ``index``'s training envs: ``seed + index * n_envs``.

    Env ``r`` of stage ``k`` first resets with ``seed + k * n_envs + r``
    (SB3's ``VecEnv.seed``), so no two envs of a run share a first seed and
    no stage replays another stage's map sequence. Stage 0 uses the run
    seed itself, exactly like a run without stages.
    """
    return int(seed) + int(index) * int(n_envs)


def stage_starts(cfg) -> list:
    """The cumulative step at which each stage starts (the first is 0)."""
    starts, step = [], 0
    for stage in cfg.stages:
        starts.append(step)
        step += stage.timesteps
    return starts


def check_stages(cfg, seed: int | None = None) -> None:
    """``ValueError`` unless ``cfg.stages`` fits the run. Nothing to check without stages.

    - every stage's ``opponents`` is a training opponent (a ladder level or
      ``"idle"``) and its ``timesteps`` a positive whole number of rollouts
      (``n_envs x n_steps``), so each stage ends exactly on a rollout;
    - the stages add up to ``total_timesteps``;
    - ``opponents`` equals the first stage's opponents, so ``config.json``
      never names opponents the run did not train against;
    - ``eval_opponents`` is set (a ladder level, checked by
      ``rl.train_ppo.check_eval_opponents``): a curriculum run never lets
      the evaluation during training fall back to ``opponents``;
    - with ``seed``: no stage's envs would play a reserved evaluation seed
      (``rl.seeds.check_training_seed`` on each ``stage_seed``).
    """
    if not cfg.stages:
        return
    rollout = cfg.n_envs * cfg.n_steps
    for index, stage in enumerate(cfg.stages):
        if stage.opponents not in TRAINING_OPPONENTS:
            raise ValueError(f"stages[{index}]: unknown opponents {stage.opponents!r}; "
                             f"choose from {list(TRAINING_OPPONENTS)}")
        if stage.timesteps < 1 or stage.timesteps % rollout:
            raise ValueError(f"stages[{index}]: timesteps {stage.timesteps} is not a positive "
                             f"whole number of rollouts (n_envs x n_steps = {rollout})")
    total = sum(stage.timesteps for stage in cfg.stages)
    if total != cfg.total_timesteps:
        raise ValueError(f"the stages add up to {total} timesteps, but total_timesteps is "
                         f"{cfg.total_timesteps}; they must be equal")
    if cfg.opponents != cfg.stages[0].opponents:
        raise ValueError(f"with stages set, opponents must equal the first stage's opponents "
                         f"({cfg.stages[0].opponents!r}), not {cfg.opponents!r}")
    if cfg.eval_opponents is None:
        raise ValueError("a run with stages must set eval_opponents (a ladder level) for "
                         "the evaluation during training")
    if seed is not None:
        for index in range(len(cfg.stages)):
            check_training_seed(stage_seed(seed, index, cfg.n_envs), cfg.n_envs)
