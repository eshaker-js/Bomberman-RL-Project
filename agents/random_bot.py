"""The weakest baseline: a uniformly random action every turn."""

import random

from game.environment import Action

# Fixed order, so a seed always maps to the same action sequence.
ACTIONS = tuple(Action)


class RandomBot:
    """Picks one of the six actions uniformly at random, ignoring the state."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, state: dict, me: int) -> Action:
        return self._rng.choice(ACTIONS)
