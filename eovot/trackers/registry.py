"""Centralized tracker registry for EOVOT.

Before this module existed, the mapping from a tracker's string name (as
used in YAML configs and the CLI) to its concrete class was copy-pasted in
three places — :mod:`eovot.trackers`, :mod:`eovot.experiment.runner`, and
``scripts/run_benchmark.py`` — and the copies had already drifted: the CLI
script could not build an ``MIL`` tracker even though ``MILTracker`` has
existed since early in the project.

:data:`TRACKER_REGISTRY` and :func:`build_tracker` are now the single
source of truth; every entry point should import from here instead of
maintaining its own dict.

Wrapper trackers
----------------
Some trackers *wrap* another tracker to add frame-skipping, resolution
downscaling, or FPS-budget enforcement.  These cannot be expressed as
``name → class`` because they require a pre-built inner tracker.

:func:`build_from_config` handles both flat and nested configs:

    # Flat tracker (simple case):
    build_from_config({"name": "KCF", "params": {"learning_rate": 0.125}})

    # Nested wrapper:
    build_from_config({
        "name": "FrameSkip",
        "params": {"skip_rate": 3, "mode": "linear"},
        "inner": {"name": "MOSSE", "params": {}},
    })

Supported wrapper names: ``"FrameSkip"``, ``"ResolutionScaled"``.

Example::

    from eovot.trackers.registry import build_tracker, build_from_config

    tracker = build_tracker("KCF", learning_rate=0.075)
    wrapped = build_from_config({
        "name": "FrameSkip",
        "params": {"skip_rate": 2},
        "inner": {"name": "MOSSE"},
    })
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseTracker
from .camshift import CamShiftTracker
from .csrt import CSRTTracker
from .kcf import KCFTracker
from .lk_optical_flow import LKOpticalFlowTracker
from .median_flow import MedianFlowTracker
from .mil import MILTracker
from .mosse import MOSSETracker

# Trackers that ship with plain opencv-python and require no external model
# files — always safe to register eagerly.
TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "CSRT": CSRTTracker,
    "MIL": MILTracker,
    "MedianFlow": MedianFlowTracker,
    "CamShift": CamShiftTracker,
    "LKOpticalFlow": LKOpticalFlowTracker,
}

# DaSiamRPN / NanoTracker need pre-trained ONNX files the user supplies
# explicitly (see eovot.trackers.opencv_dl docstrings), but the registry
# should still be able to name and construct them when params are given.
try:
    from .opencv_dl import DaSiamRPNTracker, NanoTracker

    TRACKER_REGISTRY["DaSiamRPN"] = DaSiamRPNTracker
    TRACKER_REGISTRY["NanoTrack"] = NanoTracker
except ImportError:  # pragma: no cover - only triggered on exotic builds
    pass


# ---------------------------------------------------------------------------
# Wrapper tracker registry
# ---------------------------------------------------------------------------

def _build_wrapper(name: str, inner: BaseTracker, **params: Any) -> BaseTracker:
    """Instantiate a wrapper tracker by name.

    Args:
        name:   Wrapper name (e.g. ``"FrameSkip"``).
        inner:  Already-constructed inner tracker.
        params: Keyword arguments forwarded to the wrapper's constructor.

    Returns:
        A wrapper :class:`~.base.BaseTracker` instance.

    Raises:
        ValueError: If *name* is not a known wrapper.
    """
    if name == "FrameSkip":
        from .frame_skip import FrameSkipTracker
        return FrameSkipTracker(inner, **params)
    if name == "ResolutionScaled":
        from .scale import ResolutionScaledTracker
        return ResolutionScaledTracker(inner, **params)
    raise ValueError(
        f"Unknown wrapper tracker '{name}'. "
        f"Known wrappers: FrameSkip, ResolutionScaled. "
        f"Standalone trackers: {available_trackers()}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available_trackers() -> list:
    """Return the sorted list of standalone tracker names known to the registry."""
    return sorted(TRACKER_REGISTRY)


def build_tracker(name: str, **params: Any) -> BaseTracker:
    """Instantiate a registered standalone tracker by name.

    Args:
        name:   Registry key, e.g. ``"KCF"`` or ``"MIL"``.
        params: Keyword arguments forwarded to the tracker's constructor.

    Returns:
        A new :class:`~eovot.trackers.base.BaseTracker` instance.

    Raises:
        ValueError: If *name* is not a registered tracker.
    """
    cls = TRACKER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tracker '{name}'. Available: {available_trackers()}"
        )
    return cls(**params)


def build_from_config(cfg: Dict[str, Any]) -> BaseTracker:
    """Build a tracker (possibly a wrapper) from a config dict.

    This is the recommended factory for experiment configs and CLI tools
    because it handles both standalone and nested wrapper trackers.

    Config schema for a **standalone tracker**::

        {"name": "KCF", "params": {"learning_rate": 0.125}}

    Config schema for a **wrapper tracker** (``"inner"`` key triggers nesting)::

        {
            "name": "FrameSkip",
            "params": {"skip_rate": 3, "mode": "linear"},
            "inner": {
                "name": "MOSSE",
                "params": {}
            }
        }

    Nesting is recursive, so a wrapper can wrap another wrapper::

        {
            "name": "FrameSkip",
            "params": {"skip_rate": 2},
            "inner": {
                "name": "ResolutionScaled",
                "params": {"scale_factor": 0.5},
                "inner": {"name": "KCF", "params": {}}
            }
        }

    Args:
        cfg: Config dict with at least a ``"name"`` key.  ``"params"`` is
            optional (defaults to ``{}``).  ``"inner"`` is required for
            wrapper trackers and absent for standalone trackers.

    Returns:
        A :class:`~eovot.trackers.base.BaseTracker` instance.

    Raises:
        ValueError: If any tracker name in the config is unknown.
        KeyError:   If a wrapper config is missing the required ``"inner"`` key.
    """
    name: str = cfg["name"]
    params: Dict[str, Any] = dict(cfg.get("params") or {})
    inner_cfg = cfg.get("inner")

    if inner_cfg is not None:
        inner = build_from_config(inner_cfg)
        return _build_wrapper(name, inner, **params)

    return build_tracker(name, **params)
