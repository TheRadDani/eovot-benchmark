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

``build_tracker`` also handles the composite ``"AdaptiveRes"`` prefix for
:class:`~eovot.trackers.adaptive_resolution.AdaptiveResolutionTracker`.
Use the name format ``"AdaptiveRes(<inner>)"`` (e.g. ``"AdaptiveRes(KCF)"``)
to wrap any registered tracker with the adaptive resolution controller:

    tracker = build_tracker("AdaptiveRes(KCF)", target_fps=30.0)

Example::

    from eovot.trackers.registry import build_tracker

    tracker = build_tracker("KCF", learning_rate=0.075)
    adaptive = build_tracker("AdaptiveRes(MOSSE)", target_fps=60.0)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type

from .base import BaseTracker
from .camshift import CamShiftTracker
from .csrt import CSRTTracker
from .kcf import KCFTracker
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

_ADAPTIVE_PREFIX = "AdaptiveRes("
_ADAPTIVE_SUFFIX = ")"


def available_trackers() -> list:
    """Return the sorted list of tracker names known to the registry.

    Includes plain tracker names and the ``"AdaptiveRes(<inner>)"`` prefix
    pattern for the adaptive resolution wrapper.
    """
    plain = sorted(TRACKER_REGISTRY)
    adaptive = [f"AdaptiveRes({n})" for n in plain]
    return plain + sorted(adaptive)


def build_tracker(name: str, **params: Any) -> BaseTracker:
    """Instantiate a registered tracker by name.

    Supports two forms:

    * **Plain**: ``"KCF"`` — instantiates :class:`~eovot.trackers.kcf.KCFTracker`
      with ``params`` forwarded to its constructor.
    * **Adaptive wrapper**: ``"AdaptiveRes(KCF)"`` — wraps the inner tracker
      with :class:`~eovot.trackers.adaptive_resolution.AdaptiveResolutionTracker`.
      Supported *params* for the wrapper:
      ``target_fps``, ``scale_pyramid``, ``ema_alpha``, ``warmup_frames``.
      Any remaining params are forwarded to the inner tracker constructor.

    Args:
        name:   Registry key or ``"AdaptiveRes(<inner>)"`` composite name.
        params: Keyword arguments forwarded to the tracker constructor.

    Returns:
        A new :class:`~eovot.trackers.base.BaseTracker` instance.

    Raises:
        ValueError: If *name* is not a registered tracker.
    """
    if name.startswith(_ADAPTIVE_PREFIX) and name.endswith(_ADAPTIVE_SUFFIX):
        return _build_adaptive(name, **params)

    cls = TRACKER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tracker '{name}'. Available: {sorted(TRACKER_REGISTRY)}"
        )
    return cls(**params)


def _build_adaptive(name: str, **params: Any) -> "AdaptiveResolutionTracker":  # noqa: F821
    """Construct an :class:`AdaptiveResolutionTracker` from a composite name."""
    from .adaptive_resolution import AdaptiveResolutionTracker

    inner_name = name[len(_ADAPTIVE_PREFIX): -len(_ADAPTIVE_SUFFIX)]
    if not inner_name:
        raise ValueError(
            f"AdaptiveRes name '{name}' is missing the inner tracker name."
        )

    wrapper_keys = {"target_fps", "scale_pyramid", "ema_alpha", "warmup_frames"}
    wrapper_params = {k: params.pop(k) for k in wrapper_keys & set(params)}

    inner = build_tracker(inner_name, **params)
    return AdaptiveResolutionTracker(tracker=inner, **wrapper_params)
