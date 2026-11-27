"""Start a PPO run from a saved model (Task 12, ``PPOConfig.init_model``).

A new run can start from the weights and the optimizer state of a saved PPO
model of this project: a run's ``model.zip`` or any of its checkpoints
(``checkpoints/model_<steps>.zip``, ``checkpoints/stage<k>_<opponents>_<step>.zip``).
Only the starting weights come from the file. Every setting of the new run
(rewards, ``ent_coef``, ``learning_rate``, opponents, stages, steps,
device, ...) comes from the new run's config, and the new run writes its own
new folder: the source file and its folder are only read, never changed.

Two steps, used by ``rl.train_ppo.main``:

1. ``inspect_init_model(cfg)``, before the run folder exists: the file
   exists; it is a PPO model of this project (an SB3 zip with exactly the
   ``policy`` and ``policy.optimizer`` parameters, SB3's actor-critic policy,
   PPO's own settings and ``rl.networks.SmallGridCNN``); its network sizes
   (``conv_channels``, ``features_dim``, ``net_arch_pi``, ``net_arch_vf``,
   read from the zip's own saved policy settings, which describe the weights
   actually in the file) equal the new config's; if the file lies in a PPO
   run folder, that run's ``config.json`` agrees with the zip; and its
   observation and action spaces equal the new run's. Returns the record for
   ``meta.json``.
2. ``load_init_model(model, record)``, after ``make_model`` built the model
   from the new config: SB3's ``set_parameters`` loads the saved policy and
   optimizer state onto the model's device (so a model trained on ``cuda``
   continues on ``cpu`` and the other way round), and every policy and
   optimizer tensor is compared with the file's at once.

Task 13: a run with the safety filter (``env.action_mask`` on) is a
``MaskablePPO`` run, and its saved models use sb3-contrib's maskable
actor-critic policy. Such a model only continues in another masked run, and
a plain PPO model only in an unmasked one: crossing kinds is refused before
anything is built. (The danger map changes the observation space, which the
space check already refuses.)

The optimizer state is always loaded, so Adam's running moments carry on.
The new ``learning_rate`` still applies: PPO sets it in the optimizer at the
start of every update (SB3's ``_update_learning_rate``). The step count of
the new run starts at 0; ``meta.json`` records where the source stopped.
Loading draws no random numbers.

**Only load model files from your own runs** (or from a teammate you
trust): reading a saved SB3 model unpickles its settings, which can run code.

Stable-Baselines3 (A. Raffin et al., JMLR 22(268), 2021), used as a library:
``load_from_zip_file`` and ``BaseAlgorithm.set_parameters``.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file

from rl.bomberman_env import BombermanGymEnv
from rl.networks import SmallGridCNN

MODEL_PARAMS = {"policy", "policy.optimizer"}
PPO_KEYS = ("clip_range", "n_epochs", "gae_lambda", "n_steps")
NETWORK_FIELDS = ("conv_channels", "features_dim", "net_arch_pi", "net_arch_vf")


def file_sha256(path) -> str:
    """SHA-256 of the file at ``path``, read in blocks (the file is closed afterwards)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def network_sizes(values) -> dict:
    """The four network sizes of a config (a ``PPOConfig`` or its ``config.json`` dict)."""
    get = values.get if isinstance(values, dict) else lambda name: getattr(values, name)
    return {"conv_channels": [int(c) for c in get("conv_channels")],
            "features_dim": int(get("features_dim")),
            "net_arch_pi": [int(n) for n in get("net_arch_pi")],
            "net_arch_vf": [int(n) for n in get("net_arch_vf")]}


def _not_ours(path: Path, why: str) -> ValueError:
    return ValueError(f"init_model {path} is not a saved PPO model of this project: {why}")


def _zip_network(path: Path, data: dict) -> dict:
    """The network sizes saved in the zip's policy settings (``ValueError`` if not ours)."""
    kwargs = data.get("policy_kwargs")
    if not isinstance(kwargs, dict) or kwargs.get("features_extractor_class") is not SmallGridCNN:
        raise _not_ours(path, "its network does not use rl.networks.SmallGridCNN")
    features = kwargs.get("features_extractor_kwargs") or {}
    arch = kwargs.get("net_arch")
    if ("features_dim" not in features or "conv_channels" not in features
            or not isinstance(arch, dict) or set(arch) != {"pi", "vf"}):
        raise _not_ours(path, "its saved network sizes are incomplete")
    return network_sizes({"conv_channels": features["conv_channels"],
                          "features_dim": features["features_dim"],
                          "net_arch_pi": arch["pi"], "net_arch_vf": arch["vf"]})


def _mismatches(have: dict, want: dict) -> list:
    return [f"{name}: {have[name]} vs {want[name]}" for name in NETWORK_FIELDS
            if have[name] != want[name]]


def source_run_info(path) -> dict:
    """The PPO run folder the file belongs to, if any.

    A run's ``model.zip`` lies in its run folder, a checkpoint in
    ``<run folder>/checkpoints/``. The folder counts as a PPO run folder only
    if its ``meta.json`` says ``"kind": "ppo"`` and it has a ``config.json``;
    otherwise (for example a model a teammate sent) everything is ``None``.
    """
    path = Path(path).resolve()
    folder = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    unknown = {"source_run_dir": None, "source_run_path": None, "source_seed": None,
               "source_init_model": None, "config": None}
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return unknown
    if not isinstance(meta, dict) or meta.get("kind") != "ppo" or not isinstance(config, dict):
        return unknown
    return {"source_run_dir": folder.name, "source_run_path": str(folder),
            "source_seed": meta.get("seed"), "source_init_model": meta.get("init_model"),
            "config": config}


def _is_maskable_policy(policy_class) -> bool:
    """Whether ``policy_class`` is sb3-contrib's maskable actor-critic policy (Task 13).

    Without sb3-contrib installed no saved model can have it (such a zip
    could not even be read), so it is ``False``.
    """
    try:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    except ImportError:
        return False
    return policy_class is MaskableActorCriticPolicy


def _sb3_version(path: Path) -> str | None:
    with zipfile.ZipFile(path) as archive:
        try:
            return archive.read("_stable_baselines3_version").decode("utf-8").strip()
        except KeyError:
            return None


def inspect_init_model(cfg) -> dict | None:
    """Check ``cfg.init_model`` before anything is built; ``None`` when it is unset.

    Raises ``ValueError`` with a clear message (see the module docstring for
    every check). Returns the record that goes to ``meta.json["init_model"]``.
    """
    if cfg.init_model is None:
        return None
    path = Path(cfg.init_model)
    if not path.is_file():
        raise ValueError(f"init_model {path}: no such file")
    try:
        data, params, _ = load_from_zip_file(path, device="cpu", load_data=True)
    except Exception as error:  # not a zip, a broken zip, or settings that cannot be read
        raise _not_ours(path, f"it cannot be read as an SB3 model ({error})") from error
    if not isinstance(data, dict) or set(params or {}) != MODEL_PARAMS:
        raise _not_ours(path, f"it must hold exactly the parameters {sorted(MODEL_PARAMS)}")
    policy_class = data.get("policy_class")
    is_masked = _is_maskable_policy(policy_class)
    if (policy_class is not ActorCriticPolicy and not is_masked) \
            or any(k not in data for k in PPO_KEYS):
        raise _not_ours(path, "it is not a PPO model with SB3's actor-critic policy")
    run_masked = cfg.env.action_mask != "off"
    if is_masked != run_masked:
        kinds = {True: "MaskablePPO (safety filter on)", False: "PPO (safety filter off)"}
        raise ValueError(f"init_model {path}: the saved model is {kinds[is_masked]} but "
                         f"this run is {kinds[run_masked]} (env.action_mask "
                         f"{cfg.env.action_mask!r}); a model only continues with the same kind")

    saved = _zip_network(path, data)
    wrong = _mismatches(saved, network_sizes(cfg))
    if wrong:
        raise ValueError(f"init_model {path}: its network sizes differ from this run's "
                         f"config (saved model vs this run): {'; '.join(wrong)}. Set them "
                         f"to the saved model's values; the network's shape cannot change")
    source = source_run_info(path)
    if source["config"] is not None:
        try:
            recorded = network_sizes(source["config"])
        except (KeyError, TypeError, ValueError):
            recorded = None
        if recorded is None or _mismatches(saved, recorded):
            raise ValueError(f"init_model {path}: the network sizes in "
                             f"{source['source_run_path']}/config.json do not match the "
                             f"saved model; the source folder is inconsistent")

    env = BombermanGymEnv(cfg.opponents, cfg.env)
    try:
        spaces_match = (data.get("observation_space") == env.observation_space
                        and data.get("action_space") == env.action_space)
        spaces = (env.observation_space, env.action_space)
    finally:
        env.close()
    if not spaces_match:
        raise ValueError(f"init_model {path}: its observation and action spaces "
                         f"({data.get('observation_space')}, {data.get('action_space')}) "
                         f"differ from this run's {spaces}")

    return {"path": cfg.init_model, "absolute_path": str(path.resolve()),
            "sha256": file_sha256(path), "num_timesteps": int(data["num_timesteps"]),
            "source_run_dir": source["source_run_dir"],
            "source_run_path": source["source_run_path"],
            "source_seed": source["source_seed"],
            "source_init_model": source["source_init_model"],
            "sb3_version": _sb3_version(path), "network": saved,
            "algorithm": "MaskablePPO" if is_masked else "PPO",
            "optimizer_loaded": True, "verified": False}


def same(a, b) -> bool:
    """``a`` equals ``b`` exactly, through dicts, lists and tensors (compared on the CPU)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def load_init_model(model, record: dict) -> None:
    """Load the checked file's policy and optimizer state into ``model``, then verify.

    The file must still have the SHA-256 ``inspect_init_model`` recorded.
    After loading, every policy tensor and the whole optimizer state are
    compared with the file's; ``record["verified"]`` becomes ``True``, or a
    ``RuntimeError`` is raised.
    """
    path = Path(record["absolute_path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"init_model {path} changed after it was checked")
    model.set_parameters(str(path), exact_match=True, device=model.device)
    _, params, _ = load_from_zip_file(path, device="cpu", load_data=False)
    if not same(model.policy.state_dict(), params["policy"]):
        raise RuntimeError(f"the policy loaded from {path} differs from the file")
    if not same(model.policy.optimizer.state_dict(), params["policy.optimizer"]):
        raise RuntimeError(f"the optimizer state loaded from {path} differs from the file")
    record["verified"] = True
