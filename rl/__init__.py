"""Reinforcement-learning code (track C). Headless: never imports pygame.

- ``rl.config``: every tunable value (all v0 / provisional) and JSON overrides
- ``rl.observation``: the numeric observation from one character's view
- ``rl.bomberman_env``: ``BombermanGymEnv``, the Gymnasium wrapper of the
  full game (``GAME_RULES``, random crates) against a chosen opponent level

Needs ``numpy`` and ``gymnasium`` (``pip install -r requirements.txt``).
Quick check from the repository root::

    python -c "from rl.bomberman_env import BombermanGymEnv; \
from gymnasium.utils.env_checker import check_env; check_env(BombermanGymEnv())"
"""
