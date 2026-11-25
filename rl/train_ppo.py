"""PPO (Stable-Baselines3) on the full game against a ladder level.

Run from the repository root:

    python -m rl.train_ppo --timesteps 10000 --seed 0
    python -m rl.train_ppo --config my_overrides.json --seed 1
    python -m rl.train_ppo --timesteps 1000000 --eval-every 50000 \\
        --checkpoint-every 250000 --vec-env subproc --torch-threads 8 --high-priority

Values come from ``rl.config.PPOConfig`` (all v0), optionally overridden by
a JSON file; the command-line options override single values on top
(``--timesteps`` is ``total_timesteps``). Training uses ``n_envs`` copies of
``BombermanGymEnv`` (``GAME_RULES``, random crates, three NPCs of
``PPOConfig.opponents``: a ladder level, or ``"idle"``, three bots that always
WAIT, Task 11) on the CPU, and the small CNN of ``rl.networks``
(on ``PPOConfig.device``, the CPU by default).
Training never plays an evaluation seed (``rl.seeds``).

Options (all off or unchanged by default, so a plain run trains exactly as
before they existed):

- ``vec_env``: ``"dummy"`` (all envs in this process) or ``"subproc"`` (one
  process per env, always started with ``spawn``, the only start method on
  Windows, so it behaves the same on every system)
- ``torch_threads``: torch's CPU threads (0: torch's default). A run is only
  bit-reproducible with the same thread count; the count used is in ``meta.json``
- ``high_priority``: Windows "High" priority for this process and its env
  workers (``rl.priority``); elsewhere it does nothing and says so
- ``checkpoint_every``: save ``checkpoints/model_<steps>.zip`` every N
  steps. Only the latest checkpoint and the first one of every
  ``checkpoint_keep_every`` steps are kept (the run deletes only its own
  older checkpoints)
- ``eval_every``: every N steps (and at step 0 and at the end), the
  deterministic policy plays ``eval_games`` games against ``eval_opponents``
  (Task 10; ``None``, the default, means the run's training opponents;
  a run against ``"idle"`` must set it to a ladder level) on
  the in-training evaluation seeds (``rl.seeds.TRAIN_EVAL_SEEDS``,
  the same seeds every time), with the run's ``env.max_steps``. One row per
  evaluation goes to ``eval_curve.csv`` and one row per game to
  ``eval_games.csv``, written and flushed at once, so a stopped run keeps
  every finished evaluation. The evaluation uses its own env and no training
  RNG, so it does not change what the run learns.
- ``device`` (Task 9): ``"cpu"`` (default) or ``"cuda"``, where the network
  trains. ``"cuda"`` without a usable GPU stops before anything is built
  (``rl.devices``); SB3 itself would silently train on the CPU. The device
  actually used, and for CUDA the GPU, are in ``meta.json``. The evaluation
  during training uses the live model on its device; saved models always
  load and evaluate on the CPU (``rl.evaluate_agent.PPOPolicy``). GPU runs
  are not bit-identical to CPU runs, nor guaranteed between GPU reruns.

Curriculum stages (Task 11, ``PPOConfig.stages``, empty by default): the
same model trains against each stage's opponents in order, each for its
fixed ``timesteps``, in one run folder (see ``run_stages`` and
``rl.curriculum``). Before anything is built, every stage is checked: whole
rollouts, adding up to ``total_timesteps``, known opponents, ``opponents``
equal to the first stage's, ``eval_opponents`` set, and no stage's envs on a
reserved seed (stage ``k``'s envs are seeded with ``seed + k * n_envs``).
The evaluation during training, the checkpoints and the outcome and reward
term counts run on across stages. ``progress.csv`` gets a
``curriculum/stage`` column, ``meta.json`` a ``stages`` list (start step,
opponents, env seed and end-of-stage checkpoint of each stage), and every
stage but the last ends with ``checkpoints/stage<k>_<opponents>_<step>.zip``.
SB3's own ``time/iterations``, ``time/fps`` and ``time/time_elapsed``
restart at each stage; ``time/total_timesteps`` does not. Without stages
the run trains exactly as before; one stage trains exactly like the same
run without stages.

Each run writes a new folder ``runs/<time>_ppo_seed<seed>/`` with
``config.json``, ``meta.json``, ``progress.csv`` (SB3's logger, one row per
rollout, including the cumulative episode outcomes), ``model.zip`` and the
files of the options above.

Reward terms in ``progress.csv`` (Task 10, always on): each finished
episode's reward is split into its terms (``info["reward_terms"]``, see
``rl.bomberman_env.step_reward``), summed over the episode. Each rollout row
has ``reward_terms/<term>`` (one column per term of ``REWARD_TERMS``) and
``reward_terms/total``, the means over the episodes that finished in that
rollout, and ``reward_terms/episodes``, how many there were. In a rollout
where no episode finished, ``episodes`` is 0 and the other cells are blank.
Logging only reads the step infos, so it never changes what the run learns. Evaluate it with
``python -m rl.eval_protocol --run <that folder>``.

PPO: J. Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
Implementation: Stable-Baselines3, A. Raffin et al., JMLR 22(268), 2021
(https://github.com/DLR-RM/stable-baselines3), used as a library.
"""

import argparse
import csv
import dataclasses
import multiprocessing
import sys
import time
import warnings
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.idle_bot import TRAINING_ONLY_OPPONENTS
from agents.noisy_bot import LADDER
from rl.bomberman_env import REWARD_TERMS, BombermanGymEnv
from rl.config import EnvConfig, PPOConfig, config_to_dict, load_config
from rl.curriculum import check_stages, stage_seed
from rl.devices import DEVICES, GPU_REPRODUCIBILITY, check_device, gpu_info
from rl.eval_protocol import length_by_outcome
from rl.evaluate_agent import (
    BUFF_NAMES,
    DEATH_CAUSES,
    ModelPolicy,
    evaluate_seeds,
)
from rl.networks import SmallGridCNN
from rl.priority import raise_priority
from rl.run_logging import RunLog, new_run_dir, write_json
from rl.seeds import TrainingSeedGuard, check_training_seed, train_eval_seeds

OUTCOMES = ("win", "loss", "draw", "timeout")
VEC_ENVS = ("dummy", "subproc")


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


class RewardTermLogger(BaseCallback):
    """Logs what each finished episode was paid for, per reward term (Task 10).

    Adds every step's ``info["reward_terms"]`` to a running sum per env.
    When an env's episode ends, SB3 hands over the info of its final step
    (the env is reset afterwards), so that step is still counted, and the
    sum moves to this rollout's finished episodes. At the end of each
    rollout it records the mean of each term (``reward_terms/<term>``), of
    their total (``reward_terms/total``) and the number of finished episodes
    (``reward_terms/episodes``). With no finished episode only ``episodes``
    (0) is recorded, so the other cells of that row stay blank. An episode
    counts in the rollout in which it finishes; an episode still running at
    the end of training is not logged.

    It only reads ``self.locals`` and draws no random numbers, so it never
    changes what the run learns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.running: list = []
        self.finished: list = []
        self.logged: list = []  # one dict per rollout, as recorded (for tests)

    def _on_training_start(self) -> None:
        self.running = [dict.fromkeys(REWARD_TERMS, 0.0)
                        for _ in range(self.training_env.num_envs)]

    def on_new_stage(self) -> None:
        """A curriculum stage starts with new envs (Task 11): drop the running sums.

        The old envs' unfinished episodes were cut off when they closed, so
        they never finished and are not logged. Finished ones are kept.
        """
        self._on_training_start()

    def _on_step(self) -> bool:
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            sums = self.running[i]
            for term, value in info["reward_terms"].items():
                sums[term] += value
            if done:
                self.finished.append(sums)
                self.running[i] = dict.fromkeys(REWARD_TERMS, 0.0)
        return True

    def _on_rollout_end(self) -> None:
        episodes = len(self.finished)
        row = {"reward_terms/episodes": episodes}
        if episodes:
            for term in REWARD_TERMS:
                row[f"reward_terms/{term}"] = sum(e[term] for e in self.finished) / episodes
            row["reward_terms/total"] = sum(sum(e.values()) for e in self.finished) / episodes
        for key, value in row.items():
            self.logger.record(key, value)
        self.logged.append(row)
        self.finished = []


@dataclasses.dataclass(frozen=True)
class EnvFactory:
    """Builds one training env. A plain picklable object, so ``spawn`` can send it.

    With ``subproc`` each env lives in its own spawned process, which imports
    this module and calls the factory there; with ``high_priority`` the
    worker raises its own priority first (Windows children don't inherit it).
    """

    opponents: str
    env: EnvConfig
    high_priority: bool = False

    def __call__(self) -> TrainingSeedGuard:
        if self.high_priority and multiprocessing.parent_process() is not None:
            raise_priority()
        return TrainingSeedGuard(BombermanGymEnv(self.opponents, self.env))


def eval_opponents(cfg: PPOConfig) -> str:
    """The ladder level of the evaluation during training (Task 10)."""
    return cfg.opponents if cfg.eval_opponents is None else cfg.eval_opponents


def check_eval_opponents(cfg: PPOConfig) -> None:
    """``ValueError`` unless the evaluation during training plays a ladder level.

    ``eval_opponents`` must be ``None`` or a ladder level. ``None`` falls back
    to ``opponents``, which is refused when ``opponents`` is a training-only
    opponent (Task 11: ``"idle"``): the evaluation never plays idle bots, so
    such a run must name its ``eval_opponents``.

    Checked at the start of every run (before its folder exists), even with
    ``eval_every = 0``, so a typo is never silently ignored.
    """
    if cfg.eval_opponents is not None and cfg.eval_opponents not in LADDER:
        raise ValueError(f"unknown eval_opponents {cfg.eval_opponents!r}; "
                         f"choose from {list(LADDER)} or null (= opponents)")
    if cfg.eval_opponents is None and cfg.opponents in TRAINING_ONLY_OPPONENTS:
        raise ValueError(f"opponents {cfg.opponents!r} is a training-only opponent and "
                         f"the evaluation during training never plays it: set "
                         f"eval_opponents to a ladder level {list(LADDER)}")


# The start of SB3's warning in OnPolicyAlgorithm._maybe_recommend_cpu.
GPU_MLP_WARNING = r"You are trying to run PPO on the GPU"


def make_envs(cfg: PPOConfig, opponents: str, seed: int):
    """``n_envs`` training envs against ``opponents``, seeded with ``seed``.

    ``seed`` must have passed ``check_training_seed`` (``make_model`` and
    ``check_stages`` check it); every env skips drawn reserved seeds.
    """
    factory = EnvFactory(opponents, cfg.env, cfg.high_priority)
    if cfg.vec_env == "subproc":
        vec = {"vec_env_cls": SubprocVecEnv, "vec_env_kwargs": {"start_method": "spawn"}}
    else:
        vec = {"vec_env_cls": DummyVecEnv}
    return make_vec_env(factory, n_envs=cfg.n_envs, seed=seed, **vec)


def make_model(cfg: PPOConfig, seed: int) -> PPO:
    """The PPO model and its vectorised envs, seeded with ``seed``.

    Training never plays a game on an evaluation seed (``rl.seeds``): the run
    seed is checked first, and every env skips drawn reserved seeds. The
    device is checked before any env is built (``rl.devices.check_device``).
    With stages (Task 11) the envs play the first stage's opponents, and
    every stage is checked first (``rl.curriculum.check_stages``).
    """
    if cfg.vec_env not in VEC_ENVS:
        raise ValueError(f"vec_env must be one of {VEC_ENVS}, not {cfg.vec_env!r}")
    check_device(cfg.device)
    check_eval_opponents(cfg)
    check_stages(cfg, seed)
    check_training_seed(seed, cfg.n_envs)
    # With stages, opponents equals the first stage's (check_stages).
    envs = make_envs(cfg, cfg.opponents, seed)
    policy_kwargs = {
        "features_extractor_class": SmallGridCNN,
        "features_extractor_kwargs": {"features_dim": cfg.features_dim,
                                      "conv_channels": tuple(cfg.conv_channels)},
        "net_arch": {"pi": list(cfg.net_arch_pi), "vf": list(cfg.net_arch_vf)},
        "normalize_images": False,
    }
    # "MlpPolicy" only names SB3's actor-critic policy here: the features
    # come from SmallGridCNN, which avoids CnnPolicy's uint8 image checks.
    def build():
        return PPO("MlpPolicy", envs, learning_rate=cfg.learning_rate, n_steps=cfg.n_steps,
                   batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, gamma=cfg.gamma,
                   gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
                   ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef,
                   max_grad_norm=cfg.max_grad_norm, policy_kwargs=policy_kwargs,
                   seed=seed, device=cfg.device, verbose=0)

    if cfg.device != "cuda":
        return build()
    # On the GPU, SB3 warns that PPO with "MlpPolicy" is meant for the CPU
    # because it assumes an MLP feature extractor. It does not apply here:
    # our features come from a CNN (SmallGridCNN). Only that one warning is
    # ignored, and only while the model is built.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=GPU_MLP_WARNING, category=UserWarning)
        return build()



def apply_runtime(cfg: PPOConfig) -> dict:
    """Set torch's threads and the priority of this process; return what was set."""
    if cfg.torch_threads < 0:
        raise ValueError("torch_threads must be 0 (torch's default) or more")
    check_device(cfg.device)
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
    priority = raise_priority() if cfg.high_priority else "normal (high_priority off)"
    return {"torch_threads": torch.get_num_threads(), "priority": priority,
            "vec_env": cfg.vec_env, "device": cfg.device}


def device_record(model) -> dict:
    """The device the model actually uses, for ``meta.json`` (with the GPU on CUDA)."""
    record = {"device": str(model.device)}
    if model.device.type == "cuda":
        record["gpu"] = gpu_info()
        record["reproducibility"] = GPU_REPRODUCIBILITY
    return record


class CheckpointKeeper:
    """Saves ``checkpoints/model_<steps>.zip`` and deletes this run's spare ones.

    Kept: the latest checkpoint, and the first checkpoint saved in every
    block of ``keep_every`` steps (at or after 1M, 2M, ... for 1,000,000).
    Only files this keeper saved itself are ever deleted.
    """

    def __init__(self, folder: Path, keep_every: int) -> None:
        if keep_every < 1:
            raise ValueError("checkpoint_keep_every must be at least 1")
        self.folder = Path(folder)
        self.keep_every = keep_every
        self.last_kept_block = 0
        self.spare: Path | None = None  # the latest one, unless it is a kept one

    def save(self, model, timesteps: int) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"model_{timesteps}.zip"
        if path.exists():
            raise FileExistsError(f"{path} already exists; a checkpoint never overwrites a file")
        model.save(path)
        if self.spare is not None:
            self.spare.unlink(missing_ok=True)
            self.spare = None
        block = timesteps // self.keep_every
        if block > self.last_kept_block:
            self.last_kept_block = block
        else:
            self.spare = path
        return path


class CheckpointCallback(BaseCallback):
    """Saves a checkpoint each time training passes another ``every`` steps."""

    def __init__(self, keeper: CheckpointKeeper, every: int) -> None:
        super().__init__()
        if every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        self.keeper = keeper
        self.every = every
        self.next_at = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.keeper.save(self.model, self.num_timesteps)
            while self.next_at <= self.num_timesteps:
                self.next_at += self.every
        return True


CURVE_FIELDS = (
    "timesteps", "games", "win", "loss", "draw", "timeout", "self_kills", "kills",
    "crates", "mean_length", "games_death", "mean_length_death", "games_win",
    "mean_length_win", "games_timeout", "mean_length_timeout",
    "buffs_picked", "buffs_raised",
    *(f"buffs_picked_{n}" for n in BUFF_NAMES), *(f"buffs_raised_{n}" for n in BUFF_NAMES),
    *(f"death_{c}" for c in DEATH_CAUSES), "death_unknown", "death_total",
    "eval_seconds", "wall_seconds",
)
CURVE_GAME_FIELDS = (
    "timesteps", "seed", "outcome", "steps", "self_kill", "kills", "crates",
    "buffs_picked", "buffs_raised",
    *(f"buffs_picked_{n}" for n in BUFF_NAMES), *(f"buffs_raised_{n}" for n in BUFF_NAMES),
    "death_cause",
)


def curve_row(timesteps: int, totals: dict, eval_seconds: float, wall_seconds: float) -> dict:
    """One ``eval_curve.csv`` row from ``evaluate_seeds`` totals."""
    games = totals["games"]
    row = {key: totals[key] for key in CURVE_FIELDS if key in totals}
    row.update(length_by_outcome(totals))
    row["timesteps"] = timesteps
    row["mean_length"] = round(totals["steps"] / games, 4) if games else ""
    row["eval_seconds"] = round(eval_seconds, 3)
    row["wall_seconds"] = round(wall_seconds, 3)
    return {key: row[key] for key in CURVE_FIELDS}


class TrainingEvalCallback(BaseCallback):
    """Evaluation during training: the learning curves (``eval_curve.csv``).

    Evaluates at step 0 (before any update), every time training passes
    another ``every`` steps, and once more at the end unless the last
    evaluation was already at the final step. ``close()`` closes its files
    and env; it runs at the end of training and again (harmlessly) in the
    training script's ``finally``, so nothing stays open on a crash.
    """

    def __init__(self, cfg: PPOConfig, run_dir: Path) -> None:
        super().__init__()
        if cfg.eval_every < 1 or cfg.eval_games < 1:
            raise ValueError("eval_every and eval_games must be at least 1")
        self.every = cfg.eval_every
        self.seeds = train_eval_seeds(cfg.eval_games)
        self.env = BombermanGymEnv(eval_opponents(cfg), cfg.env)
        self.next_at = cfg.eval_every
        self.last_at: int | None = None
        self.started = time.perf_counter()
        self._curve = open(run_dir / "eval_curve.csv", "w", newline="", encoding="utf-8")
        self._games = open(run_dir / "eval_games.csv", "w", newline="", encoding="utf-8")
        self._curve_writer = csv.DictWriter(self._curve, fieldnames=list(CURVE_FIELDS))
        self._games_writer = csv.DictWriter(self._games, fieldnames=list(CURVE_GAME_FIELDS))
        self._curve_writer.writeheader()
        self._games_writer.writeheader()
        self.rows: list = []

    def evaluate(self) -> dict:
        timesteps = int(self.model.num_timesteps)
        began = time.perf_counter()
        per_game: list = []
        totals = evaluate_seeds(self.env, ModelPolicy(self.model, deterministic=True),
                                self.seeds, per_game)
        now = time.perf_counter()
        row = curve_row(timesteps, totals, now - began, now - self.started)
        self._curve_writer.writerow(row)
        self._games_writer.writerows({"timesteps": timesteps, **game} for game in per_game)
        self._curve.flush()
        self._games.flush()
        self.rows.append(row)
        self.last_at = timesteps
        return row

    def _on_training_start(self) -> None:
        self.started = time.perf_counter()
        self.evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.evaluate()
            while self.next_at <= self.num_timesteps:
                self.next_at += self.every
        return True

    def _on_training_end(self) -> None:
        if self.last_at != self.model.num_timesteps:
            self.evaluate()
        self.close()

    def close(self) -> None:
        """Close the CSV files and the evaluation env. Safe to call more than once."""
        self._curve.close()
        self._games.close()
        self.env.close()


class StagedCallbacks(CallbackList):
    """The run's callbacks, kept going across curriculum stages (Task 11).

    Each stage is one ``model.learn(..., reset_num_timesteps=False)`` call,
    and SB3 starts and ends the callbacks on every call. This list passes
    ``on_training_start`` to its callbacks only in the first stage (so the
    step-0 evaluation runs once) and ``on_training_end`` only after the
    last stage (so the evaluator closes its files only at the end). At a
    later stage start each callback gets the new call's locals and the
    current step count, and ``on_new_stage()`` if it has one (the reward
    term logger drops the cut-off episodes of the old envs). Every other
    event goes to every callback, in order, as with a ``CallbackList``.
    Counters, next evaluation and checkpoint steps carry on unchanged.

    Each rollout row of ``progress.csv`` gets ``curriculum/stage``, the
    index of the stage it belongs to. Nothing here draws random numbers.
    """

    def __init__(self, callbacks: list) -> None:
        super().__init__(callbacks)
        self.stage = 0
        self.last_stage = True
        self._started = False

    def _on_training_start(self) -> None:
        if not self._started:
            self._started = True
            super()._on_training_start()
            return
        for callback in self.callbacks:
            callback.locals = self.locals
            callback.globals = self.globals
            callback.num_timesteps = self.model.num_timesteps
            if hasattr(callback, "on_new_stage"):
                callback.on_new_stage()

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/stage", self.stage)
        super()._on_rollout_end()

    def _on_training_end(self) -> None:
        if self.last_stage:
            super()._on_training_end()


def stage_checkpoint_name(index: int, opponents: str, timesteps: int) -> str:
    """``stage<index>_<opponents>_<cumulative steps>.zip``, e.g. ``stage0_idle_2000000.zip``."""
    return f"stage{index}_{opponents}_{timesteps}.zip"


def run_stages(model: PPO, cfg: PPOConfig, seed: int, callback: StagedCallbacks,
               run_dir: Path, meta: dict) -> None:
    """Train ``model`` through ``cfg.stages`` in order (Task 11).

    Stage 0 trains on the envs ``make_model`` built. Before each later
    stage the old envs are closed first (so at most one set of env workers
    exists), then new envs against the stage's opponents are built with
    ``stage_seed`` and set on the model; SB3 then resets them with their
    own seeds. Each stage trains exactly its ``timesteps`` (whole rollouts),
    and the step count goes on across stages. At the end of every stage but
    the last the model is saved as ``checkpoints/stage<k>_<opponents>_<step>.zip``
    (never deleted by ``CheckpointKeeper``); the last stage ends in
    ``model.zip``. ``meta["stages"]`` gets one entry per stage as it
    starts, and ``meta.json`` is rewritten each time.
    """
    meta["stages"] = []
    for index, stage in enumerate(cfg.stages):
        env_seed = stage_seed(seed, index, cfg.n_envs)
        if index > 0:
            model.get_env().close()
            model.set_env(make_envs(cfg, stage.opponents, env_seed))
        entry = {"index": index, "opponents": stage.opponents, "timesteps": stage.timesteps,
                 "start_step": int(model.num_timesteps), "env_seed": env_seed,
                 "checkpoint": None}
        meta["stages"].append(entry)
        write_json(run_dir / "meta.json", meta)
        callback.stage = index
        callback.last_stage = index == len(cfg.stages) - 1
        model.learn(total_timesteps=stage.timesteps, callback=callback,
                    reset_num_timesteps=False)
        if not callback.last_stage:
            folder = run_dir / "checkpoints"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / stage_checkpoint_name(index, stage.opponents,
                                                  int(model.num_timesteps))
            if path.exists():
                raise FileExistsError(f"{path} already exists; a checkpoint never "
                                      f"overwrites a file")
            model.save(path)
            entry["checkpoint"] = path.relative_to(run_dir).as_posix()
            write_json(run_dir / "meta.json", meta)


def parse_args(argv: list | None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override PPOConfig.total_timesteps")
    parser.add_argument("--config", default=None, help="JSON file of overrides")
    parser.add_argument("--runs-dir", default=None, help="parent folder (default runs/)")
    parser.add_argument("--vec-env", choices=VEC_ENVS, default=None,
                        help="override PPOConfig.vec_env")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="override PPOConfig.torch_threads (0: torch's default)")
    parser.add_argument("--high-priority", action="store_true", default=None,
                        help="set PPOConfig.high_priority (Windows only)")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="override PPOConfig.checkpoint_every (0: off)")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override PPOConfig.eval_every (0: off)")
    parser.add_argument("--eval-games", type=int, default=None,
                        help="override PPOConfig.eval_games")
    parser.add_argument("--device", choices=DEVICES, default=None,
                        help="override PPOConfig.device (cuda needs a CUDA build of torch)")
    return parser.parse_args(argv)


def resolve_config(args) -> PPOConfig:
    cfg = load_config(PPOConfig(), args.config)
    overrides = {"total_timesteps": args.timesteps, "vec_env": args.vec_env,
                 "torch_threads": args.torch_threads, "high_priority": args.high_priority,
                 "checkpoint_every": args.checkpoint_every, "eval_every": args.eval_every,
                 "eval_games": args.eval_games, "device": args.device}
    return dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def main(argv: list | None = None) -> Path:
    args = parse_args(argv)
    cfg = resolve_config(args)
    runtime = apply_runtime(cfg)
    # Before the run folder exists (both again in make_model).
    check_eval_opponents(cfg)
    check_stages(cfg, args.seed)
    run_dir = new_run_dir("ppo", args.seed, args.runs_dir)
    with RunLog(run_dir, "ppo", args.seed, config_to_dict(cfg),
                argv=sys.argv if argv is None else ["-m", "rl.train_ppo", *argv]) as log:
        log.meta["runtime"] = runtime
        write_json(run_dir / "meta.json", log.meta)
        model = make_model(cfg, args.seed)
        log.meta["runtime"].update(device_record(model))
        write_json(run_dir / "meta.json", log.meta)
        logger = configure(str(run_dir), ["csv"])
        model.set_logger(logger)
        evaluator = None
        try:
            counter = OutcomeCounter()
            callbacks = [counter, RewardTermLogger()]
            if cfg.checkpoint_every > 0:
                keeper = CheckpointKeeper(run_dir / "checkpoints", cfg.checkpoint_keep_every)
                callbacks.append(CheckpointCallback(keeper, cfg.checkpoint_every))
            if cfg.eval_every > 0:
                evaluator = TrainingEvalCallback(cfg, run_dir)
                callbacks.append(evaluator)
            if not cfg.stages:
                model.learn(total_timesteps=cfg.total_timesteps,
                            callback=callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks))
            else:
                run_stages(model, cfg, args.seed, StagedCallbacks(callbacks), run_dir, log.meta)
            model.save(run_dir / "model.zip")
            summary = {"timesteps": int(model.num_timesteps), "episodes": counter.counts}
            if evaluator is not None:
                summary["last_eval"] = evaluator.rows[-1]
            log.finish(summary)
        finally:
            # Close every file and env even if training fails: an open file
            # cannot be deleted on Windows.
            if evaluator is not None:
                evaluator.close()
            logger.close()
            model.get_env().close()
    print(f"run folder: {run_dir}")
    print("summary:", summary)
    return run_dir


if __name__ == "__main__":
    main()
