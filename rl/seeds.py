"""Game seed ranges: which seeds evaluation uses, and which training never plays.

A game seed fixes the crate layout, the buff drops and the bot seeds (see
``rl.bomberman_env``). An agent evaluated on a seed it already trained on
would be measured on a map it may have memorised, so two ranges are
reserved for evaluation and training never plays a game on them:

- ``EVAL_SEEDS`` = ``[1,000,000, 2,000,000)``: the final evaluation protocol
  (``rl.eval_protocol``). Every level and every training seed uses the same
  first 200 seeds, so comparisons are paired; ``--seed-offset`` moves to
  fresh seeds of the range for later re-runs.
- ``TRAIN_EVAL_SEEDS`` = ``[2,000,000, 3,000,000)``: the evaluation during
  training (the learning curves, ``rl.train_ppo``).

Training seeds come from two places, and both are guarded:

1. the first reset of each of the ``n_envs`` training envs uses the run seed
   plus the env's rank (SB3's ``VecEnv.seed``): ``check_training_seed``
   refuses a run seed whose ``seed .. seed + n_envs - 1`` touches a reserved
   seed, before anything is built;
2. every later episode seed is drawn by the wrapper from its ``np_random``
   in ``[0, 2**31 - 1)``: ``TrainingSeedGuard`` resets again (the wrapper
   draws the next seed) until the drawn seed is not reserved. The rejected
   seed never plays a single step. About 0.09% of draws are rejected.

Redrawing (instead of a separate training range) keeps every training game
exactly as before this guard existed, unless a reserved seed is drawn.
"""

import gymnasium as gym

EVAL_SEEDS = range(1_000_000, 2_000_000)
TRAIN_EVAL_SEEDS = range(2_000_000, 3_000_000)
RESERVED_RANGES = (EVAL_SEEDS, TRAIN_EVAL_SEEDS)


def is_reserved(seed: int) -> bool:
    """True if ``seed`` belongs to an evaluation range."""
    return any(int(seed) in reserved for reserved in RESERVED_RANGES)


def check_training_seed(seed: int, n_envs: int) -> None:
    """Refuse a run seed whose first resets (``seed .. seed+n_envs-1``) are reserved."""
    first = range(int(seed), int(seed) + int(n_envs))
    for reserved in RESERVED_RANGES:
        if first.start < reserved.stop and reserved.start < first.stop:
            raise ValueError(
                f"training seed {seed} with {n_envs} envs would play reserved "
                f"evaluation seeds ({reserved.start:,} to {reserved.stop - 1:,}); "
                f"choose a seed outside the evaluation ranges")


def _seeds(within: range, games: int, offset: int) -> range:
    if games < 1 or offset < 0:
        raise ValueError("games must be at least 1 and offset at least 0")
    seeds = range(within.start + offset, within.start + offset + games)
    if seeds.stop > within.stop:
        raise ValueError(f"{games} games from offset {offset} leave the range "
                         f"{within.start:,} to {within.stop - 1:,}")
    return seeds


def eval_seeds(games: int, offset: int = 0) -> range:
    """The game seeds of the final evaluation: ``games`` seeds of ``EVAL_SEEDS``."""
    return _seeds(EVAL_SEEDS, games, offset)


def train_eval_seeds(games: int) -> range:
    """The game seeds of the evaluation during training (the same every time)."""
    return _seeds(TRAIN_EVAL_SEEDS, games, 0)


class TrainingSeedGuard(gym.Wrapper):
    """Makes a training env skip every reserved episode seed.

    Wraps a ``BombermanGymEnv`` (anything whose reset info has
    ``"episode_seed"``). An explicit reserved seed raises ``ValueError``; a
    drawn reserved seed is replaced by the next drawn seed. ``redraws``
    counts the replaced seeds.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.redraws = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None and is_reserved(seed):
            raise ValueError(f"training may not play the reserved evaluation seed {seed}")
        obs, info = self.env.reset(seed=seed, options=options)
        while is_reserved(info["episode_seed"]):
            # Only a drawn seed can get here: draw the next one.
            self.redraws += 1
            obs, info = self.env.reset(options=options)
        return obs, info
