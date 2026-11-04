"""One new folder per run, with everything needed to reproduce it.

Every training or evaluation run calls ``new_run_dir``, which creates
``<root>/<UTC time>_<kind>_seed<seed>/`` (``root`` defaults to ``runs/``,
which git ignores) and refuses a folder that already exists, so a run can
never overwrite another run's results. Inside it a run writes:

- ``config.json``: the resolved config (defaults plus JSON overrides)
- ``meta.json``: kind, seed, command line, git commit and whether the
  working tree had uncommitted changes, Python/platform and dependency
  versions, start and finish time, and a short result summary
- ``metrics.csv``: one row per episode (tabular; PPO uses SB3's ``progress.csv``)
- the saved model or evaluation results
"""

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "runs"
DEPENDENCIES = ("numpy", "gymnasium", "torch", "stable-baselines3", "pygame")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_dir(kind: str, seed: int, root: str | Path | None = None) -> Path:
    """Create and return a new, empty run folder. Fails if it already exists."""
    root = Path(root) if root is not None else DEFAULT_RUNS_DIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}_{kind}_seed{seed}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_info() -> dict:
    """Current commit and whether the working tree is dirty (``None`` if unknown)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def dependency_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


class RunLog:
    """Writes ``config.json``, ``meta.json`` and ``metrics.csv`` of one run folder."""

    def __init__(self, run_dir: Path, kind: str, seed: int, config: dict,
                 argv: list | None = None) -> None:
        self.dir = Path(run_dir)
        self.meta = {
            "kind": kind,
            "seed": seed,
            "command": [sys.executable, *(sys.argv if argv is None else argv)],
            "git": git_info(),
            "versions": dependency_versions(),
            "started_at": utc_now(),
            "finished_at": None,
            "summary": None,
        }
        write_json(self.dir / "config.json", config)
        write_json(self.dir / "meta.json", self.meta)
        self._csv = None
        self._writer = None

    def metric(self, row: dict) -> None:
        """Append one row to ``metrics.csv`` (the first row fixes the columns)."""
        if self._writer is None:
            self._csv = open(self.dir / "metrics.csv", "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def finish(self, summary: dict) -> None:
        """Close ``metrics.csv`` and record the finish time and summary in ``meta.json``."""
        if self._csv is not None:
            self._csv.close()
        self.meta["finished_at"] = utc_now()
        self.meta["summary"] = summary
        write_json(self.dir / "meta.json", self.meta)
