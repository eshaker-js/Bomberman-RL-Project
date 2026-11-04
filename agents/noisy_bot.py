"""A rule bot that sometimes plays randomly, and the opponent difficulty ladder.

``NoisyBot(epsilon)`` plays like ``RuleBot``, except that on each turn, with
probability ``epsilon``, it takes a uniformly random action instead. Varying
``epsilon`` gives opponents between the random bot (weakest) and the rule bot
(strongest), so training can move up a ladder of opponents step by step.

Reproducibility: the bot owns two seeded RNGs. The noise RNG decides "random
or rule" and picks the random action; the inner ``RuleBot`` keeps its own RNG
for its tie-breaks. The noise RNG is only used when ``0 < epsilon``, so with
``epsilon=0`` the bot makes exactly the same choices as ``RuleBot`` with the
same seed.
"""

import random
from functools import partial

from agents.random_bot import ACTIONS, RandomBot
from agents.rule_bot import RuleBot
from game.environment import Action


def _noise_rng(seed: int | None) -> random.Random:
    """The noise RNG, seeded differently from the inner rule bot's RNG.

    A string seed is hashed with SHA-512 by ``random.Random``, so it does
    not depend on ``PYTHONHASHSEED`` and is the same on every run.
    """
    return random.Random(None if seed is None else f"noise:{seed}")


class NoisyBot:
    """``RuleBot`` that takes a uniformly random action with probability ``epsilon``."""

    def __init__(self, epsilon: float, seed: int | None = None) -> None:
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) \
                or not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be a number in [0, 1], not {epsilon!r}")
        self.epsilon = float(epsilon)
        self._rule = RuleBot(seed)
        self._rng = _noise_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rule.reset(seed)
        self._rng = _noise_rng(seed)

    def act(self, state: dict, me: int) -> Action:
        # epsilon == 0 never touches the noise RNG (exactly RuleBot);
        # epsilon == 1 never asks the rule bot.
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        return self._rule.act(state, me)


# Opponent difficulty ladder, weakest first. Every value is a factory that
# takes no arguments and returns a fresh, unseeded bot; seed it with
# ``reset(seed)`` before a game. The epsilon values are v0 (provisional).
LADDER = {
    "random": RandomBot,
    "noisy50": partial(NoisyBot, 0.5),
    "noisy25": partial(NoisyBot, 0.25),
    "noisy10": partial(NoisyBot, 0.1),
    "rule": RuleBot,
}


def make_bot(level: str):
    """A fresh bot of the ladder level ``level`` (a key of ``LADDER``)."""
    if level not in LADDER:
        raise ValueError(f"unknown ladder level {level!r}; choose from {sorted(LADDER)}")
    return LADDER[level]()
