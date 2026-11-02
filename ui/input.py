"""Keyboard to ``Action`` for the human player.

During a tick, the last mapped key pressed wins. If no mapped key was
pressed during the tick but a movement key is still held, the most
recently pressed held movement key is the action (held Space does not
repeat). With neither, the action is ``WAIT``.
"""

import pygame

from game.environment import Action

KEY_TO_ACTION = {
    pygame.K_UP: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
    pygame.K_SPACE: Action.BOMB,
}
MOVES = frozenset({Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT})


class InputBuffer:
    """Collects key events between ticks and turns them into one action."""

    def __init__(self) -> None:
        self._pressed: Action | None = None
        self._held: list[int] = []  # movement keys still down, oldest first

    def key_down(self, key: int) -> bool:
        """Record a key press. Return ``True`` if the key maps to an action."""
        action = KEY_TO_ACTION.get(key)
        if action is None:
            return False
        self._pressed = action
        if action in MOVES:
            if key in self._held:
                self._held.remove(key)
            self._held.append(key)
        return True

    def key_up(self, key: int) -> None:
        if key in self._held:
            self._held.remove(key)

    def release_all(self) -> None:
        """Forget held keys, e.g. when the window loses focus and misses key-ups."""
        self._held.clear()

    def clear(self) -> None:
        self._pressed = None
        self._held.clear()

    def consume(self) -> Action:
        """The action for the tick that is due now. Clears the pressed key."""
        if self._pressed is not None:
            action = self._pressed
        elif self._held:
            action = KEY_TO_ACTION[self._held[-1]]
        else:
            action = Action.WAIT
        self._pressed = None
        return action
