"""Reinforcement-learning code (track C). Headless: never imports pygame.

- ``rl.config``: every tunable value (all v0 / provisional) and JSON overrides
- ``rl.observation``: the numeric observation from one character's view
- ``rl.bomberman_env``: ``BombermanGymEnv``, the Gymnasium wrapper of the
  full game (``GAME_RULES``, random crates) against a chosen opponent level
- ``rl.tabular_env``: the tiny stage and the tabular state encoding
- ``rl.train_tabular``: tabular Q-learning (our own code)
- ``rl.sweep_tabular`` / ``rl.plot_tabular``: hyperparameter sweep of tabular
  Q-learning over several seeds, and its learning-curve figures (matplotlib)
- ``rl.networks`` / ``rl.train_ppo``: PPO from Stable-Baselines3 with a small CNN
- ``rl.evaluate_agent``: evaluates a saved tabular or PPO agent
- ``rl.eval_protocol``: the evaluation protocol (all 5 ladder levels, 200 games
  each, on the evaluation seeds)
- ``rl.seeds``: the evaluation seed ranges, which training never plays
- ``rl.benchmark_ppo``: how fast PPO trains on this computer
- ``rl.run_ppo_seeds`` / ``rl.plot_ppo``: a plan of PPO runs over several seeds,
  its summary with 95% confidence intervals, and its learning-curve figures
- ``rl.priority``: Windows high priority and keeping Windows awake
- ``rl.run_logging``: one new ``runs/`` folder per run (config, seed, commit,
  versions, metrics, model)

Install (CPU-only torch first, then the rest)::

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

Run from the repository root::

    python -m rl.train_tabular --episodes 3000 --seed 0
    python -m rl.sweep_tabular --seeds 5
    python -m rl.plot_tabular runs/<sweep folder>
    python -m rl.train_ppo --timesteps 10000 --seed 0
    python -m rl.evaluate_agent --run runs/<run folder>
    python -m rl.evaluate_agent --run runs/<run folder> --stage full --opponents noisy25
    python -m rl.benchmark_ppo --high-priority
    python -m rl.run_ppo_seeds experiments/ppo/first_runs.json
    python -m rl.run_ppo_seeds --continue runs/<plan folder>
    python -m rl.eval_protocol --run runs/<ppo run folder>
    python -m rl.plot_ppo runs/<plan folder>

Any script takes ``--config overrides.json`` to change v0 values without
touching code (see ``rl.config``).
"""
