"""Reinforcement-learning code (track C). Headless: never imports pygame.

- ``rl.config``: every tunable value (all v0 / provisional) and JSON overrides
- ``rl.observation``: the numeric observation from one character's view
- ``rl.bomberman_env``: ``BombermanGymEnv``, the Gymnasium wrapper of the
  full game (``GAME_RULES``, random crates) against a chosen opponent level
- ``rl.tabular_env``: the tiny stage and the tabular state encoding
- ``rl.train_tabular``: tabular Q-learning (our own code)
- ``rl.networks`` / ``rl.train_ppo``: PPO from Stable-Baselines3 with a small CNN
- ``rl.evaluate_agent``: evaluates a saved tabular or PPO agent
- ``rl.run_logging``: one new ``runs/`` folder per run (config, seed, commit,
  versions, metrics, model)

Install (CPU-only torch first, then the rest)::

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

Run from the repository root::

    python -m rl.train_tabular --episodes 3000 --seed 0
    python -m rl.train_ppo --timesteps 10000 --seed 0
    python -m rl.evaluate_agent --run runs/<run folder>
    python -m rl.evaluate_agent --run runs/<run folder> --stage full --opponents noisy25

Any script takes ``--config overrides.json`` to change v0 values without
touching code (see ``rl.config``).
"""
