"""Which device PPO trains on, checked before anything is built (Task 9).

``PPOConfig.device`` is ``"cpu"`` (the default, as before Task 9) or
``"cuda"``. Stable-Baselines3 silently trains on the CPU when it is asked for
``"cuda"`` and CUDA is not available (``stable_baselines3.common.utils.
get_device``), so a GPU run that is really a CPU run would be recorded as a
GPU run. ``check_device`` stops such a run first, with the checks to make.

Only training uses this: evaluation and saved-model loading stay on the CPU
(``rl.evaluate_agent.PPOPolicy``). torch is imported inside the functions, so
importing this module (for example from ``rl.run_ppo_seeds``) stays cheap.
"""

import subprocess

DEVICES = ("cpu", "cuda")

# Written into meta.json of every CUDA run and repeated in the README.
GPU_REPRODUCIBILITY = (
    "GPU runs are not bit-identical to CPU runs with the same seed, and not "
    "guaranteed bit-identical between reruns on the GPU (SB3 sets cuDNN to "
    "deterministic mode, but some CUDA operations stay nondeterministic)")


def cuda_help() -> str:
    """How to check the CUDA install, for the error message."""
    import torch
    return (f"installed torch: {torch.__version__}, torch.version.cuda = "
            f"{torch.version.cuda} (None means a CPU-only build of torch). Check: "
            "(1) `nvidia-smi` shows the GPU and its driver; "
            "(2) `python -c \"import torch; print(torch.version.cuda, "
            "torch.cuda.is_available())\"` prints a CUDA version and True; "
            "(3) if not, install a CUDA build of torch (README, 'Training on the GPU')")


def check_device(device: str) -> str:
    """``device`` if it can be used; otherwise ``ValueError``/``RuntimeError``.

    Called before any env, process or run folder is created.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' was asked for, but torch.cuda.is_available() "
                               "is False, so nothing was started. " + cuda_help())
    return device


def driver_version() -> str | None:
    """The NVIDIA driver version from ``nvidia-smi``, or ``None`` if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[0] if lines else None


def gpu_info() -> dict | None:
    """GPU 0 (name, memory, CUDA version, driver), or ``None`` without CUDA."""
    import torch
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {"name": torch.cuda.get_device_name(0),
            "memory_gb": round(props.total_memory / 2**30, 2),
            "cuda": torch.version.cuda,
            "driver": driver_version(),
            "count": torch.cuda.device_count()}
