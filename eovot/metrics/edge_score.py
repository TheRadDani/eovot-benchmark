"""Edge deployment scoring for EOVOT tracker benchmarks.

Computes a single composite *Edge Score* that balances tracking accuracy,
inference speed, memory footprint, and energy efficiency.  The score is
designed to rank trackers for **resource-constrained edge deployments**
(Raspberry Pi, Jetson Nano, etc.) where all four dimensions matter
simultaneously.

Score formula
-------------
Each metric is normalised to ``[0, 1]`` relative to a configurable reference
baseline, then combined as a weighted geometric mean::

    EdgeScore = exp(
        (w_iou * log(iou_norm) + w_fps * log(fps_norm)
         + w_mem * log(mem_norm) [+ w_e * log(e_norm)])
        / W
    )

where:

* ``iou_norm``  = ``clip(mean_iou, 0, 1)``
* ``fps_norm``  = ``clip(fps / fps_ref, 0, 1)``   (faster → higher)
* ``mem_norm``  = ``clip(1 - memory_mb / mem_ref_mb, 0, 1)``  (less → higher)
* ``e_norm``    = ``clip(1 - energy_mj / energy_ref_mj, 0, 1)``  (less → higher)
* ``W``         = sum of active weights

The geometric mean prevents a single inflated dimension from masking poor
performance elsewhere.

References
----------
- Zhu et al., “Distractor-aware Siamese Networks for Visual Object Tracking”,
  ECCV 2018.  (Motivates speed–accuracy trade-offs in tracking)
- Luo et al., “MobileTrack: Towards Real-Time Visual Tracking on Mobile
  Devices”, ICCV Workshop 2019.  (Edge tracking motivation)
- Patterson et al., “Carbon Emissions and Large Neural Network Training”,
  arXiv:2104.10350 (2021).  (Motivates energy-aware ML benchmarking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EdgeScoreWeights:
    """Configurable weighting for each metric dimension.

    Higher weight means that dimension has more influence on the final score.
    All weights must be non-negative; the total need not sum to 1.

    Args:
        iou:    Importance of tracking accuracy (mean IoU).  Default ``4.0``.
        fps:    Importance of throughput (frames per second).  Default ``2.0``.
        memory: Importance of memory efficiency.  Default ``2.0``.
        energy: Importance of energy efficiency.  Default ``2.0``.

    Example::

        # Accuracy-first profile (for surveillance cameras with stable power)
        weights = EdgeScoreWeights(iou=8.0, fps=1.0, memory=1.0, energy=0.0)

        # Balanced edge profile (default)
        weights = EdgeScoreWeights()
    """

    iou: float = 4.0
    fps: float = 2.0
    memory: float = 2.0
    energy: float = 2.0

    def __post_init__(self) -> None:
        for attr in ("iou", "fps", "memory", "energy"):
            val = getattr(self, attr)
            if val < 0:
                raise ValueError(
                    f"EdgeScoreWeights.{attr} must be non-negative, got {val}"
                )


# Default reference baselines (calibrated for real-time tracking on OTB-100
# with classical trackers on a laptop-class CPU).
_DEFAULT_FPS_REF: float = 30.0       # 30 FPS — live-video real-time threshold
_DEFAULT_MEM_REF_MB: float = 256.0   # 256 MB — conservative edge device budget
_DEFAULT_ENERGY_REF_MJ: float = 5.0  # 5 mJ/frame — estimated KCF on Raspberry Pi 4


def compute_edge_score(
    mean_iou: float,
    fps: float,
    memory_mb: float,
    energy_mj_per_frame: Optional[float] = None,
    *,
    weights: Optional[EdgeScoreWeights] = None,
    fps_ref: float = _DEFAULT_FPS_REF,
    mem_ref_mb: float = _DEFAULT_MEM_REF_MB,
    energy_ref_mj: float = _DEFAULT_ENERGY_REF_MJ,
) -> float:
    """Compute the composite edge deployment score for a single tracker run.

    Args:
        mean_iou:            Mean Intersection-over-Union across evaluated
                             frames, expected in ``[0, 1]``.
        fps:                 Tracker throughput in frames per second (> 0).
        memory_mb:           Peak process memory usage in megabytes (> 0).
        energy_mj_per_frame: Mean energy consumption per inference frame in
                             milli-Joules, or ``None`` to omit the energy
                             dimension from the score.
        weights:             Per-dimension weighting.  Defaults to
                             :class:`EdgeScoreWeights` default values
                             ``(iou=4, fps=2, memory=2, energy=2)``.
        fps_ref:             Reference FPS for normalisation (default: 30.0).
        mem_ref_mb:          Reference memory budget in MB (default: 256.0).
        energy_ref_mj:       Reference per-frame energy in mJ (default: 5.0).

    Returns:
        A float in ``[0, 1]`` representing edge deployment fitness.
        Higher is better.  Returns ``0.0`` when any active dimension is zero.

    Raises:
        ValueError: If ``fps`` or ``memory_mb`` are not strictly positive.

    Example::

        from eovot.metrics.edge_score import compute_edge_score

        score = compute_edge_score(
            mean_iou=0.62,
            fps=120.0,
            memory_mb=45.0,
            energy_mj_per_frame=2.1,
        )
        print(f"Edge Score: {score:.4f}")
    """
    if fps <= 0:
        raise ValueError(f"fps must be strictly positive, got {fps}")
    if memory_mb <= 0:
        raise ValueError(f"memory_mb must be strictly positive, got {memory_mb}")

    if weights is None:
        weights = EdgeScoreWeights()

    # ------------------------------------------------------------------
    # Normalise each dimension to [0, 1]
    # ------------------------------------------------------------------
    iou_norm = float(np.clip(mean_iou, 0.0, 1.0))
    fps_norm = float(np.clip(fps / fps_ref, 0.0, 1.0))
    mem_norm = float(np.clip(1.0 - memory_mb / mem_ref_mb, 0.0, 1.0))

    dims = [iou_norm, fps_norm, mem_norm]
    w_list = [weights.iou, weights.fps, weights.memory]

    if energy_mj_per_frame is not None:
        e_norm = float(np.clip(1.0 - energy_mj_per_frame / energy_ref_mj, 0.0, 1.0))
        dims.append(e_norm)
        w_list.append(weights.energy)

    # ------------------------------------------------------------------
    # Weighted geometric mean
    # Geometric mean collapses to 0 if any dimension is 0, which prevents
    # a tracker from scoring well by excelling on only one axis.
    # ------------------------------------------------------------------
    w_total = sum(w_list)
    if w_total == 0.0:
        return 0.0

    log_score = sum(
        w * float(np.log(max(d, 1e-9))) for d, w in zip(dims, w_list)
    ) / w_total
    return float(np.clip(np.exp(log_score), 0.0, 1.0))


def rank_by_edge_score(
    results: dict,
    weights: Optional[EdgeScoreWeights] = None,
    fps_ref: float = _DEFAULT_FPS_REF,
    mem_ref_mb: float = _DEFAULT_MEM_REF_MB,
    energy_ref_mj: float = _DEFAULT_ENERGY_REF_MJ,
) -> list:
    """Rank multiple tracker results by their Edge Score.

    Args:
        results: ``{tracker_name: summary_dict}`` where each ``summary_dict``
                 contains at least ``mean_iou``, ``mean_fps``,
                 ``peak_memory_mb`` and optionally
                 ``mean_energy_per_frame_mj``.
        weights: Scoring weights (default :class:`EdgeScoreWeights`).
        fps_ref:        Reference FPS (default 30.0).
        mem_ref_mb:     Reference memory in MB (default 256.0).
        energy_ref_mj:  Reference energy in mJ/frame (default 5.0).

    Returns:
        List of ``(tracker_name, edge_score)`` tuples sorted descending.

    Example::

        from eovot.metrics.edge_score import rank_by_edge_score

        ranking = rank_by_edge_score({
            "MOSSE": {"mean_iou": 0.52, "mean_fps": 520, "peak_memory_mb": 35},
            "KCF":   {"mean_iou": 0.58, "mean_fps": 220, "peak_memory_mb": 42},
            "CSRT":  {"mean_iou": 0.68, "mean_fps": 45,  "peak_memory_mb": 65},
        })
        for name, score in ranking:
            print(f"{name}: {score:.4f}")
    """
    scored = []
    for name, s in results.items():
        score = compute_edge_score(
            mean_iou=float(s["mean_iou"]),
            fps=float(s["mean_fps"]),
            memory_mb=float(s["peak_memory_mb"]),
            energy_mj_per_frame=s.get("mean_energy_per_frame_mj"),
            weights=weights,
            fps_ref=fps_ref,
            mem_ref_mb=mem_ref_mb,
            energy_ref_mj=energy_ref_mj,
        )
        scored.append((name, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)
