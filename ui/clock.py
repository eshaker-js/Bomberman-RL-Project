"""The human play clock. Pure: no pygame.

The engine is turn-based. Only for a human at the keyboard does the UI call
``env.step`` on its own every ``TICK_MS``. The clock never changes what a
tick does, only when the UI asks for one.
"""

from ui.config import TICK_MS


class TickClock:
    """Adds up frame time and says when the next tick is due.

    ``update(dt)`` returns ``True`` at most once per call, so after a long
    stall (for example while the window is dragged) the backlog is dropped
    instead of being played as a burst of ticks. ``progress`` is how far the
    current tick is along, from 0.0 right after a tick to 1.0 when the next
    one is due; the renderer interpolates with it.
    """

    def __init__(self, tick_ms: int = TICK_MS) -> None:
        if tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        self.tick_ms = tick_ms
        self.elapsed = 0.0
        self.paused = False

    @property
    def progress(self) -> float:
        return min(self.elapsed / self.tick_ms, 1.0)

    def update(self, dt_ms: float) -> bool:
        """Advance by ``dt_ms``. Return ``True`` if a tick is due now."""
        if self.paused:
            return False
        self.elapsed += dt_ms
        if self.elapsed < self.tick_ms:
            return False
        self.elapsed -= self.tick_ms
        if self.elapsed >= self.tick_ms:
            # More than one tick behind: drop the backlog.
            self.elapsed = 0.0
        return True

    def finish(self) -> None:
        """Stop at the end of the current tick (``progress`` 1.0), for good."""
        self.elapsed = float(self.tick_ms)
        self.paused = True

    def reset(self) -> None:
        self.elapsed = 0.0
        self.paused = False
