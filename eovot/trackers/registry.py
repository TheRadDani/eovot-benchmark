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

# Ensemble is built dynamically — it wraps other registered trackers rather
# than being a class with a fixed constructor, so it gets special handling
# in build_tracker rather than a plain entry in TRACKER_REGISTRY.
_ENSEMBLE_KEY = "Ensemble"
_DEFAULT_ENSEMBLE_MEMBERS = ["MOSSE", "KCF", "LKOpticalFlow"]

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
    """Return the sorted list of tracker names known to the registry.

    Includes ``"Ensemble"`` as a special entry that is handled by
    :func:`build_tracker` rather than stored as a plain class mapping.
    """
    return sorted(list(TRACKER_REGISTRY) + [_ENSEMBLE_KEY])


def build_tracker(name: str, **params: Any) -> BaseTracker:
    """Instantiate a registered tracker by name.

    Supports all entries in :data:`TRACKER_REGISTRY` plus the special key
    ``"Ensemble"``, which builds a consensus-based ensemble wrapper.

    Args:
        name:   Registry key, e.g. ``"KCF"``, ``"MIL"``, or ``"Ensemble"``.
        params: Keyword arguments forwarded to the tracker's constructor.
                For ``"Ensemble"``, the optional parameter ``member_names``
                (list of str) controls which sub-trackers are included.
                Default members: ``["MOSSE", "KCF", "MedianFlow"]``.

    Returns:
        A new :class:`~eovot.trackers.base.BaseTracker` instance.

    Raises:
        ValueError: If *name* is not a registered tracker.

    Example::

        # Default three-member ensemble
        tracker = build_tracker("Ensemble")

        # Custom members
        tracker = build_tracker("Ensemble", member_names=["MOSSE", "KCF", "CSRT"])
    """
    if name == _ENSEMBLE_KEY:
        from .ensemble import EnsembleTracker

        member_names: list = params.pop("member_names", _DEFAULT_ENSEMBLE_MEMBERS)
        ensemble_name: str = params.pop("name", None)
        return EnsembleTracker.from_names(
            member_names=member_names,
            name=ensemble_name,
        )

    cls = TRACKER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tracker '{name}'. Available: {available_trackers()}"
        )
    return cls(**params)
