"""Retro game window for the Bomberman environment. Start it with ``python -m ui``.

The UI is display only: it reads ``BombermanEnv.get_state()`` and sends
actions through ``BombermanEnv.step``. It never decides movement,
explosions, deaths, outcomes or timing. Nothing outside ``ui/`` imports this
package, so ``game/`` and ``agents/`` keep working without pygame.

Modules without "pygame" in their imports (``config``, ``palette``, ``art``,
``font``, ``clock``, ``menu``, ``session``, ``scene``) hold the logic and
the data, so they can be tested headlessly. Only ``input``, ``sprites``,
``render``, ``screens`` and ``app`` import pygame.
"""
