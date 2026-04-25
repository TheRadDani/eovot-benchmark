"""Reproducibility snapshot for EOVOT experiments.

Captures the environment state at the time an experiment is run so that
results can be reproduced or compared with confidence:

- Git commit hash (short) + dirty-working-tree flag
- Python version and platform string
- Physical CPU count
- Key package versions (numpy, cv2, torch, onnxruntime, psutil, pandas)
- UTC timestamp
- Random seed used for the run

Typical usage::

    from eovot.experiment.snapshot import ReproducibilitySnapshot

    snap = ReproducibilitySnapshot.capture(seed=42)
    print(snap.git_commit)        # e.g. "5113070"
    print(snap.package_versions)  # {"numpy": "1.26.4", "cv2": "4.9.0", ...}

    # Persist alongside results
    import json
    with open("results/metadata.json", "w") as f:
        json.dump(snap.to_dict(), f, indent=2)
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class ReproducibilitySnapshot:
    """Immutable environment snapshot captured at experiment start.

    Attributes:
        timestamp:        UTC ISO-8601 timestamp of when the snapshot was taken.
        python_version:   Full Python version string (``sys.version``).
        platform_info:    OS/platform string (``platform.platform()``).
        cpu_count:        Number of physical CPU cores.
        package_versions: Mapping of package name → version string for key
                          packages present in the environment.
        git_commit:       Short git commit hash (7–12 hex chars), or ``None``
                          if not in a git repository.
        git_dirty:        ``True`` if there are uncommitted changes in the
                          working tree.
        random_seed:      The seed passed at capture time, or ``None``.
    """

    timestamp: str
    python_version: str
    platform_info: str
    cpu_count: int
    package_versions: Dict[str, str]
    git_commit: Optional[str] = None
    git_dirty: bool = False
    random_seed: Optional[int] = None

    @classmethod
    def capture(cls, seed: Optional[int] = None) -> "ReproducibilitySnapshot":
        """Capture the current environment state.

        Args:
            seed: The random seed used for this experiment run, if any.

        Returns:
            A populated :class:`ReproducibilitySnapshot`.
        """
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version,
            platform_info=platform.platform(),
            cpu_count=_cpu_count(),
            package_versions=_package_versions(),
            git_commit=_git_commit(),
            git_dirty=_git_dirty(),
            random_seed=seed,
        )

    def to_dict(self) -> Dict:
        """Serialize to a JSON-safe dict."""
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform_info,
            "cpu_count": self.cpu_count,
            "packages": self.package_versions,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "seed": self.random_seed,
        }

    def __str__(self) -> str:
        dirty = " (dirty)" if self.git_dirty else ""
        commit = f"  git={self.git_commit}{dirty}" if self.git_commit else ""
        return (
            f"ReproducibilitySnapshot @ {self.timestamp}"
            f"{commit}  py={self.python_version.split()[0]}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cpu_count() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1
    except ImportError:
        import os
        return os.cpu_count() or 1


def _package_versions() -> Dict[str, str]:
    """Query installed versions for packages relevant to EOVOT."""
    candidates = {
        "numpy": "numpy",
        "cv2": "cv2",
        "torch": "torch",
        "onnxruntime": "onnxruntime",
        "psutil": "psutil",
        "pandas": "pandas",
        "yaml": "yaml",
    }
    versions: Dict[str, str] = {}
    for display_name, module_name in candidates.items():
        try:
            mod = importlib.import_module(module_name)
            versions[display_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _git_commit() -> Optional[str]:
    """Return the short git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False
