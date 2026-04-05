"""Hardware-aware tracker recommendation for EOVOT.

Given a :class:`~eovot.selection.hardware_profile.HardwareProfile` and a set
of deployment constraints, :class:`TrackerSelector` ranks all known trackers
and returns an ordered list of :class:`TrackerRecommendation` objects.

Tracker performance database
-----------------------------
The built-in database stores *baseline* characteristics measured on a
reference laptop (Intel Core i7, 16 GB RAM).  For a target device with
performance factor ``f`` (see :class:`~eovot.selection.hardware_profile.DeviceClass`),
the expected FPS is scaled as ``baseline_fps * f``.

Accuracy scores are normalised OTB-100 success AUC values from the
literature or reproduced benchmarks (0–1 scale).

Memory estimates reflect peak RSS during tracking of a 640×480 frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hardware_profile import DeviceClass, HardwareProfile


# ---------------------------------------------------------------------------
# Tracker performance database (baseline: laptop-class CPU, single core)
# ---------------------------------------------------------------------------

@dataclass
class _TrackerSpec:
    """Internal baseline performance spec for one tracker."""
    name: str
    baseline_fps: float       # FPS on reference laptop CPU
    accuracy_auc: float       # OTB-100 success AUC (0–1), higher is better
    peak_memory_mb: float     # Typical peak memory usage (MB)
    requires_gpu: bool = False
    notes: str = ""


# Reference baselines measured / reported on Intel Core i7-10750H (6 cores, 16 GB)
_TRACKER_DB: Dict[str, _TrackerSpec] = {
    "MOSSE": _TrackerSpec(
        name="MOSSE",
        baseline_fps=450.0,
        accuracy_auc=0.338,
        peak_memory_mb=35.0,
        notes="Fastest classical tracker; very low accuracy.",
    ),
    "KCF": _TrackerSpec(
        name="KCF",
        baseline_fps=220.0,
        accuracy_auc=0.477,
        peak_memory_mb=45.0,
        notes="Good speed/accuracy trade-off; robust to scale change.",
    ),
    "MedianFlow": _TrackerSpec(
        name="MedianFlow",
        baseline_fps=140.0,
        accuracy_auc=0.459,
        peak_memory_mb=50.0,
        notes="Reliable failure detection via forward-backward error.",
    ),
    "MIL": _TrackerSpec(
        name="MIL",
        baseline_fps=55.0,
        accuracy_auc=0.495,
        peak_memory_mb=80.0,
        notes="Multiple-instance learning; good against partial occlusion.",
    ),
    "CSRT": _TrackerSpec(
        name="CSRT",
        baseline_fps=45.0,
        accuracy_auc=0.620,
        peak_memory_mb=95.0,
        notes="Highest classical accuracy; requires opencv-contrib.",
    ),
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class TrackerConstraints:
    """Deployment constraints used to filter and rank trackers.

    Attributes:
        target_fps: Minimum acceptable frames-per-second on the target device.
            Trackers estimated below this threshold are excluded.  Default: 1.0
            (no filter).
        max_memory_mb: Maximum acceptable peak memory usage in MB.  Default:
            ``None`` (no limit).
        accuracy_weight: How much to weight accuracy vs. speed in the composite
            score used for ranking (0 = pure speed, 1 = pure accuracy).
            Default: 0.5.
        allow_gpu: Whether GPU-requiring trackers are eligible.  Default: False.
    """

    target_fps: float = 1.0
    max_memory_mb: Optional[float] = None
    accuracy_weight: float = 0.5
    allow_gpu: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.accuracy_weight <= 1.0):
            raise ValueError(
                f"accuracy_weight must be in [0, 1], got {self.accuracy_weight}"
            )
        if self.target_fps < 0:
            raise ValueError(f"target_fps must be non-negative, got {self.target_fps}")


@dataclass
class TrackerRecommendation:
    """A single tracker recommendation with estimated performance metrics.

    Attributes:
        rank: Position in the recommendation list (1 = best).
        tracker_name: Name of the tracker (matches TRACKER_REGISTRY keys).
        estimated_fps: Projected FPS on the target device.
        accuracy_auc: Normalised success AUC (0–1).
        peak_memory_mb: Estimated peak memory usage (MB).
        score: Composite score used for ranking (higher is better, 0–1).
        notes: Human-readable notes about this tracker.
    """

    rank: int
    tracker_name: str
    estimated_fps: float
    accuracy_auc: float
    peak_memory_mb: float
    score: float
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"#{self.rank:>2}  {self.tracker_name:<12s}  "
            f"FPS≈{self.estimated_fps:>7.1f}  "
            f"AUC={self.accuracy_auc:.3f}  "
            f"mem≈{self.peak_memory_mb:.0f}MB  "
            f"score={self.score:.3f}"
        )


class TrackerSelector:
    """Rank trackers for a given hardware profile and deployment constraints.

    Args:
        tracker_db: Optional override for the built-in performance database.
            Keys must match TRACKER_REGISTRY names in run_benchmark.py.

    Example::

        profile = HardwareProfile.detect()
        selector = TrackerSelector()
        recs = selector.rank(
            profile,
            TrackerConstraints(target_fps=30.0, accuracy_weight=0.7),
        )
        for r in recs:
            print(r)
    """

    def __init__(self, tracker_db: Optional[Dict[str, _TrackerSpec]] = None) -> None:
        self._db = tracker_db if tracker_db is not None else _TRACKER_DB

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        profile: HardwareProfile,
        constraints: Optional[TrackerConstraints] = None,
    ) -> List[TrackerRecommendation]:
        """Return all eligible trackers ranked by composite score.

        Args:
            profile: Hardware profile of the target device.
            constraints: Deployment constraints.  Uses default
                :class:`TrackerConstraints` if ``None``.

        Returns:
            List of :class:`TrackerRecommendation` sorted best-first.
            Returns an empty list if no tracker satisfies the constraints.
        """
        if constraints is None:
            constraints = TrackerConstraints()

        perf_factor = profile.device_class.performance_factor
        candidates = []

        for spec in self._db.values():
            if spec.requires_gpu and not constraints.allow_gpu:
                continue

            est_fps = spec.baseline_fps * perf_factor
            if est_fps < constraints.target_fps:
                continue

            if constraints.max_memory_mb is not None:
                if spec.peak_memory_mb > constraints.max_memory_mb:
                    continue

            candidates.append((spec, est_fps))

        if not candidates:
            return []

        # Normalise FPS and AUC within the candidate set for fair scoring.
        fps_values = [fps for _, fps in candidates]
        auc_values = [spec.accuracy_auc for spec, _ in candidates]
        fps_min, fps_max = min(fps_values), max(fps_values)
        auc_min, auc_max = min(auc_values), max(auc_values)

        def _norm(val: float, lo: float, hi: float) -> float:
            return (val - lo) / (hi - lo) if hi > lo else 1.0

        w_acc = constraints.accuracy_weight
        w_spd = 1.0 - w_acc

        scored = []
        for spec, est_fps in candidates:
            fps_score = _norm(est_fps, fps_min, fps_max)
            auc_score = _norm(spec.accuracy_auc, auc_min, auc_max)
            composite = w_spd * fps_score + w_acc * auc_score
            scored.append((composite, spec, est_fps))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            TrackerRecommendation(
                rank=i + 1,
                tracker_name=spec.name,
                estimated_fps=round(est_fps, 1),
                accuracy_auc=spec.accuracy_auc,
                peak_memory_mb=spec.peak_memory_mb,
                score=round(composite, 4),
                notes=spec.notes,
            )
            for i, (composite, spec, est_fps) in enumerate(scored)
        ]

    def recommend(
        self,
        profile: HardwareProfile,
        target_fps: float = 1.0,
        max_memory_mb: Optional[float] = None,
        accuracy_weight: float = 0.5,
    ) -> Optional[TrackerRecommendation]:
        """Return the single best tracker for the given constraints.

        Convenience wrapper around :meth:`rank`.

        Args:
            profile: Hardware profile of the target device.
            target_fps: Minimum acceptable FPS.
            max_memory_mb: Maximum acceptable peak memory (MB), or ``None``.
            accuracy_weight: Weight of accuracy vs. speed (0–1).

        Returns:
            The top-ranked :class:`TrackerRecommendation`, or ``None`` if no
            eligible tracker was found.
        """
        constraints = TrackerConstraints(
            target_fps=target_fps,
            max_memory_mb=max_memory_mb,
            accuracy_weight=accuracy_weight,
        )
        results = self.rank(profile, constraints)
        return results[0] if results else None

    def summary_table(
        self,
        profile: HardwareProfile,
        constraints: Optional[TrackerConstraints] = None,
    ) -> str:
        """Return a Markdown table of all ranked trackers.

        Args:
            profile: Hardware profile of the target device.
            constraints: Optional deployment constraints.

        Returns:
            Markdown-formatted string with one row per eligible tracker.
        """
        recs = self.rank(profile, constraints)
        if not recs:
            return "_No trackers satisfy the given constraints._"

        header = (
            "| Rank | Tracker | Est. FPS | AUC | Peak Mem (MB) | Score |\n"
            "|------|---------|----------|-----|---------------|-------|\n"
        )
        rows = "".join(
            f"| {r.rank} | {r.tracker_name} | {r.estimated_fps:.1f} "
            f"| {r.accuracy_auc:.3f} | {r.peak_memory_mb:.0f} | {r.score:.3f} |\n"
            for r in recs
        )
        return header + rows
