"""A harmless training opponent: a bot that always WAITs (Task 11).

Three idle bots give the agent the real game (the same map, gate rule and
observation) against opponents that never bomb and never move. The agent can
learn to break crates, kill and win by the gate without being hunted: an
idle bot dies to blasts like any character, so once the agent's bombs have
killed all three, the gate opens. (A game without NPCs can't be used for
this: its gate never opens.)

Idle bots are a **training opponent only**. They are not a ladder level:
``agents.noisy_bot.LADDER``, and with it the evaluation protocol
(``rl.eval_protocol.LEVELS``), never contains them. Training code accepts
them through ``TRAINING_ONLY_OPPONENTS`` (see ``rl.bomberman_env``).
"""

from game.environment import Action


class IdleBot:
    """Always returns ``WAIT``, whatever the state. Same interface as the other bots."""

    def __init__(self, seed: int | None = None) -> None:
        # No randomness: the seed is accepted only for the common interface.
        pass

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, state: dict, me: int) -> Action:
        return Action.WAIT


# Training-only opponents, next to (and never inside) the ladder. Like the
# ladder, every value is a factory that takes no arguments and returns a
# fresh bot.
TRAINING_ONLY_OPPONENTS = {
    "idle": IdleBot,
}
