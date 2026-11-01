"""Scripted baseline bots (track B).

Every bot has the same two methods:

- ``reset(seed)`` starts a new game with its own seeded RNG, so a bot given
  the same seed and the same states always picks the same actions.
- ``act(state, me)`` takes ``BombermanEnv.get_state()`` and the ID of the
  character it controls, and returns one ``Action``.

Bots only read the state and return actions. They never touch the engine,
so the same bot can play as the agent (ID 0) or as any NPC (IDs 1-3).
"""

from agents.random_bot import RandomBot
from agents.rule_bot import RuleBot

__all__ = ["RandomBot", "RuleBot"]
