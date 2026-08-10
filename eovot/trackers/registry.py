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

Example::

    from eovot.trackers.registry import build_tracker

    tracker = build_tracker("KCF", learning_rate=0.075)
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BaseTracker
from .camshift import CamShiftTracker
from .csrt import CSRTTracker
from .kcf import KCFTracker
from .median_flow import MedianFlowTracker
from .mil import MILTracker
from .mosse import MOSSETracker
from .adaptive import AdaptiveTracker

# Trackers that ship with plain opencv-python and require no external model
# files — always safe to register eagerly.
TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "CSRT": CSRTTracker,
    "MIL": MILTracker,
    "MedianFlow": MedianFlowTracker,
    "CamShift": CamShiftTracker,
    # AdaptiveTracker wraps another tracker; the registry builds the inner
    # tracker from an "inner" param key.  Example config:
    #   name: Adaptive-KCF
    #   params: {inner: KCF, confidence_threshold: 0.80, max_consecutive_skips: 3}
    "Adaptive": AdaptiveTracker,
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


def available_trackers() -> list:
    """Return the sorted list of tracker names known to the registry."""
    return sorted(TRACKER_REGISTRY)


def build_tracker(name: str, **params: Any) -> BaseTracker:
    """Instantiate a registered tracker by name.

    For the ``"Adaptive"`` tracker, the ``inner`` keyword selects the
    underlying tracker by name (looked up recursively in this registry).
    All remaining ``params`` are forwarded to :class:`~eovot.trackers.adaptive.AdaptiveTracker`.

    Example::

        tracker = build_tracker("Adaptive", inner="KCF",
                                confidence_threshold=0.80,
                                max_consecutive_skips=3)

    Args:
        name:   Registry key, e.g. ``"KCF"`` or ``"Adaptive"``.
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
    if cls is AdaptiveTracker:
        inner_name = params.pop("inner", "MOSSE")
        inner_tracker = build_tracker(inner_name)
        return AdaptiveTracker(tracker=inner_tracker, **params)
    return cls(**params)
