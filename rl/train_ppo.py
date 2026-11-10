"""PPO (Stable-Baselines3) on the full game against a ladder level.

Run from the repository root:

    python -m rl.train_ppo --timesteps 10000 --seed 0
    python -m rl.train_ppo --config my_overrides.json --seed 1

Values come from ``rl.config.PPOConfig`` (all v0), optionally overridden by
a JSON file; ``--timesteps`` overrides ``total_timesteps``. Training uses
``n_envs`` copies of ``BombermanGymEnv`` (``GAME_RULES``, random crates,
three NPCs of ``PPOConfig.opponents``) in a ``DummyVecEnv`` on the CPU, and
the small CNN of ``rl.networks``. Each run writes a new folder
``runs/<time>_ppo_seed<seed>/`` with ``config.json``, ``meta.json``,
``progress.csv`` (SB3's logger, one row per rollout, including the
cumulative episode outcomes) and ``model.zip``. Evaluate it with
``python -m rl.evaluate_agent --run <that folder>``.

PPO: J. Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
Implementation: Stable-Baselines3, A. Raffin et al., JMLR 22(268), 2021
(https://github.com/DLR-RM/stable-baselines3), used as a library.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.bomberman_env import BombermanGymEnv
from rl.config import PPOConfig, config_to_dict, load_config
from rl.networks import SmallGridCNN
from rl.run_logging import RunLog, new_run_dir

OUTCOMES = ("win", "loss", "draw", "timeout")


class OutcomeCounter(BaseCallback):
    """Counts finished episodes by outcome and logs the totals each rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.counts = dict.fromkeys(OUTCOMES, 0)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.counts[info.get("outcome") or "timeout"] += 1
        return True

    def _on_rollout_end(self) -> None:
        for name, count in self.counts.items():
            self.logger.record(f"outcomes/{name}", count)


def make_model(cfg: PPOConfig, seed: int) -> PPO:
    """The PPO model and its vectorised envs, seeded with ``seed``."""
    envs = make_vec_env(lambda: BombermanGymEnv(cfg.opponents, cfg.env),
                        n_envs=cfg.n_envs, seed=seed, vec_env_cls=DummyVecEnv)
    policy_kwargs = {
        "features_extractor_class": SmallGridCNN,
        "features_extractor_kwargs": {"features_dim": cfg.features_dim,
                                      "conv_channels": tuple(cfg.conv_channels)},
        "net_arch": {"pi": list(cfg.net_arch_pi), "vf": list(cfg.net_arch_vf)},
        "normalize_images": False,
    }
    # "MlpPolicy" only names SB3's actor-critic policy here: the features
    # come from SmallGridCNN, which avoids CnnPolicy's uint8 image checks.
    return PPO("MlpPolicy", envs, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
               batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
               gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
               ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
               max_grad_norm=cfg.max_grad_norm, policy_kwargs=policy_kwargs,
               seed=seed, device="cpu", verbose=0)


def main(argv: list | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override PPOConfig.total_timesteps")
    parser.add_argument("--config", default=None, help="JSON file of overrides")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    args = parser.parse_args(argv)

    cfg = load_config(PPOConfig(), args.config)
    if args.timesteps is not None:
        cfg = dataclasses.replace(cfg, total_timesteps=args.timesteps)
    run_dir = new_run_dir("ppo", args.seed, args.runs_dir)
    with RunLog(run_dir, "ppo", args.seed, config_to_dict(cfg),
                argv=sys.argv if argv is None else ["-m", "rl.train_ppo", *argv]) as log:
        model = make_model(cfg, args.seed)
        logger = configure(str(run_dir), ["csv"])
        model.set_logger(logger)
        try:
            counter = OutcomeCounter()
            model.learn(total_timesteps=cfg.total_timesteps, callback=counter)
            model.save(run_dir / "model.zip")
            summary = {"timesteps": int(model.num_timesteps), "episodes": counter.counts}
            log.finish(summary)
        finally:
            # Close progress.csv and the training envs even if training fails:
            # an open file cannot be deleted on Windows.
            logger.close()
            model.get_env().close()
    print(f"run folder: {run_dir}")
    print("summary:", summary)
    return run_dir


if __name__ == "__main__":
    main()
