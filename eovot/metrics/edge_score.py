"""Edge Deployment Score (EDS) — composite deployability metric for VOT trackers.

The EDS ranks trackers for resource-constrained edge hardware by combining:
  - Accuracy   (mean IoU) — higher is better
  - Throughput (FPS)      — higher is better
  - Memory     (peak MB)  — lower is better
  - Energy     (mJ/frame) — lower is better

Each axis is min-max normalised across the compared tracker set, then a
weighted sum produces the scalar EDS ∈ [0, 1].  Pre-defined hardware profiles
(Raspberry Pi 4, Jetson Nano, laptop CPU, desktop GPU) ship with tuned weights
and hard deployment constraints (minimum FPS, maximum memory).

A Pareto-frontier analysis identifies trackers that are not dominated on any
two objectives simultaneously (default: accuracy vs. speed).

Typical usage::

    from eovot.metrics.edge_score import EdgeDeploymentScorer, TrackerMetrics

    trackers = [
        TrackerMetrics("MOSSE", mean_iou=0.42, fps=450.0, peak_memory_mb=45.0),
        TrackerMetrics("KCF",   mean_iou=0.51, fps=180.0, peak_memory_mb=52.0),
        TrackerMetrics("CSRT",  mean_iou=0.65, fps=40.0,  peak_memory_mb=68.0),
    ]

    scorer = EdgeDeploymentScorer.from_hardware_profile("raspberry_pi_4")
    scores = scorer.score(trackers)
    print(scorer.format_leaderboard(scores))

    pareto = scorer.pareto_frontier(trackers)
    optimal = [p.tracker_name for p in pareto if p.is_pareto_optimal]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hardware profiles
# ---------------------------------------------------------------------------

HARDWARE_PROFILES: Dict[str, Dict] = {
    "raspberry_pi_4": {
        "description": "Raspberry Pi 4 (Cortex-A72 quad-core, up to 15 W TDP)",
        "target_fps": 25.0,
        "max_memory_mb": 512.0,
        "tdp_watts": 15.0,
        "weights": {"accuracy": 0.35, "fps": 0.35, "memory": 0.20, "energy": 0.10},
    },
    "jetson_nano": {
        "description": "NVIDIA Jetson Nano (Cortex-A57 + Maxwell GPU, 10 W mode)",
        "target_fps": 30.0,
        "max_memory_mb": 1024.0,
        "tdp_watts": 10.0,
        "weights": {"accuracy": 0.40, "fps": 0.30, "memory": 0.15, "energy": 0.15},
    },
    "laptop_cpu": {
        "description": "Mid-range laptop CPU (15–45 W TDP, no discrete GPU)",
        "target_fps": 60.0,
        "max_memory_mb": 4096.0,
        "tdp_watts": 45.0,
        "weights": {"accuracy": 0.50, "fps": 0.25, "memory": 0.15, "energy": 0.10},
    },
    "desktop_gpu": {
        "description": "Desktop GPU workstation (high compute, 250 W TDP)",
        "target_fps": 120.0,
        "max_memory_mb": 8192.0,
        "tdp_watts": 250.0,
        "weights": {"accuracy": 0.70, "fps": 0.15, "memory": 0.10, "energy": 0.05},
    },
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TrackerMetrics:
    """Raw hardware metrics for one tracker obtained from a benchmark run.

    Args:
        name: Tracker identifier (e.g. ``"MOSSE"``).
        mean_iou: Mean IoU across all evaluated frames.
        fps: Mean frames-per-second throughput.
        peak_memory_mb: Peak RSS memory in megabytes.
        energy_per_frame_mj: Mean per-frame energy in milli-Joules.
            Defaults to ``0.0`` when energy profiling was not enabled.
    """

    name: str
    mean_iou: float
    fps: float
    peak_memory_mb: float
    energy_per_frame_mj: float = 0.0

    @classmethod
    def from_benchmark_result(cls, result: object) -> "TrackerMetrics":
        """Construct from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` instance.

        Returns:
            :class:`TrackerMetrics` populated from *result*'s aggregate
            properties.
        """
        energy_mj = getattr(result, "mean_energy_per_frame_mj", None) or 0.0
        return cls(
            name=result.tracker_name,  # type: ignore[attr-defined]
            mean_iou=result.mean_iou,  # type: ignore[attr-defined]
            fps=result.mean_fps,  # type: ignore[attr-defined]
            peak_memory_mb=result.peak_memory_mb,  # type: ignore[attr-defined]
            energy_per_frame_mj=float(energy_mj),
        )


@dataclass
class EdgeScore:
    """Composite Edge Deployment Score for one tracker.

    Attributes:
        tracker_name: Tracker identifier.
        raw: Original :class:`TrackerMetrics` before normalisation.
        normalized: Per-axis normalised scores in ``[0, 1]``.
        weights: Weights used for each axis.
        score: Final weighted EDS in ``[0, 1]`` (higher = more deployable).
        meets_fps_target: Whether *fps* ≥ ``target_fps`` constraint.
        meets_memory_target: Whether *peak_memory_mb* ≤ ``max_memory_mb``.
    """

    tracker_name: str
    raw: TrackerMetrics
    normalized: Dict[str, float]
    weights: Dict[str, float]
    score: float
    meets_fps_target: bool
    meets_memory_target: bool

    @property
    def is_deployable(self) -> bool:
        """``True`` if both FPS and memory hard constraints are met."""
        return self.meets_fps_target and self.meets_memory_target

    def to_dict(self) -> Dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "tracker": self.tracker_name,
            "eds": round(self.score, 4),
            "mean_iou": round(self.raw.mean_iou, 4),
            "fps": round(self.raw.fps, 2),
            "peak_memory_mb": round(self.raw.peak_memory_mb, 2),
            "energy_per_frame_mj": round(self.raw.energy_per_frame_mj, 4),
            "is_deployable": self.is_deployable,
            "normalized": {k: round(v, 4) for k, v in self.normalized.items()},
        }


@dataclass
class ParetoPoint:
    """One point on the accuracy–speed Pareto frontier.

    Attributes:
        tracker_name: Tracker identifier.
        mean_iou: Accuracy axis value.
        fps: Speed axis value.
        is_pareto_optimal: ``True`` if no other tracker strictly dominates
            this point on *both* objectives.
    """

    tracker_name: str
    mean_iou: float
    fps: float
    is_pareto_optimal: bool


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class EdgeDeploymentScorer:
    """Score and rank trackers for edge hardware deployment.

    Normalises each metric axis min-max across the tracker cohort, then
    computes a weighted sum to produce the Edge Deployment Score (EDS).
    Hard constraints (``target_fps``, ``max_memory_mb``) flag trackers that
    cannot meet deployment requirements regardless of their EDS.

    Args:
        weights: Dict with keys ``"accuracy"``, ``"fps"``, ``"memory"``,
            ``"energy"`` that must sum to ``1.0``.
            Defaults to equal 0.25 weighting on each axis.
        target_fps: Minimum FPS required for deployment.
        max_memory_mb: Maximum peak memory allowed (MB).

    Raises:
        ValueError: If *weights* keys are missing or do not sum to 1.0.
    """

    _REQUIRED_WEIGHT_KEYS = frozenset({"accuracy", "fps", "memory", "energy"})

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        target_fps: float = 25.0,
        max_memory_mb: float = 512.0,
    ) -> None:
        if weights is None:
            weights = {k: 0.25 for k in self._REQUIRED_WEIGHT_KEYS}
        self._validate_weights(weights)
        self.weights = weights
        self.target_fps = target_fps
        self.max_memory_mb = max_memory_mb

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_hardware_profile(cls, profile_name: str) -> "EdgeDeploymentScorer":
        """Instantiate a scorer tuned for a named hardware target.

        Args:
            profile_name: One of the keys in :data:`HARDWARE_PROFILES`
                (e.g. ``"raspberry_pi_4"``).

        Returns:
            Pre-configured :class:`EdgeDeploymentScorer`.

        Raises:
            KeyError: If *profile_name* is not in :data:`HARDWARE_PROFILES`.
        """
        if profile_name not in HARDWARE_PROFILES:
            available = list(HARDWARE_PROFILES)
            raise KeyError(
                f"Unknown hardware profile {profile_name!r}. "
                f"Available: {available}"
            )
        profile = HARDWARE_PROFILES[profile_name]
        return cls(
            weights=dict(profile["weights"]),
            target_fps=profile["target_fps"],
            max_memory_mb=profile["max_memory_mb"],
        )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(self, trackers: List[TrackerMetrics]) -> List[EdgeScore]:
        """Compute EDS for each tracker.

        Args:
            trackers: All trackers to compare.  Normalisation is relative
                to this cohort, so all trackers intended for comparison
                should be passed together.

        Returns:
            List of :class:`EdgeScore` sorted by ``score`` descending
            (best-for-deployment first).
        """
        if not trackers:
            return []

        norm = self._normalize(trackers)

        results: List[EdgeScore] = []
        for i, tracker in enumerate(trackers):
            eds = (
                self.weights["accuracy"] * norm["accuracy"][i]
                + self.weights["fps"] * norm["fps"][i]
                + self.weights["memory"] * norm["memory"][i]
                + self.weights["energy"] * norm["energy"][i]
            )
            results.append(
                EdgeScore(
                    tracker_name=tracker.name,
                    raw=tracker,
                    normalized={k: float(norm[k][i]) for k in norm},
                    weights=dict(self.weights),
                    score=float(eds),
                    meets_fps_target=tracker.fps >= self.target_fps,
                    meets_memory_target=tracker.peak_memory_mb <= self.max_memory_mb,
                )
            )

        return sorted(results, key=lambda s: s.score, reverse=True)

    def pareto_frontier(
        self,
        trackers: List[TrackerMetrics],
        objective_x: str = "fps",
        objective_y: str = "mean_iou",
    ) -> List[ParetoPoint]:
        """Identify Pareto-optimal trackers on a 2D objective plane.

        A tracker is Pareto-optimal if no other tracker is strictly better
        on **both** objectives simultaneously.

        Args:
            trackers: Tracker cohort.
            objective_x: Attribute name for the x-axis (higher-is-better).
                One of ``"fps"``, ``"mean_iou"``.
            objective_y: Attribute name for the y-axis (higher-is-better).

        Returns:
            List of :class:`ParetoPoint` with ``is_pareto_optimal`` set.
        """
        if not trackers:
            return []

        x_vals = np.array([getattr(t, objective_x) for t in trackers], dtype=float)
        y_vals = np.array([getattr(t, objective_y) for t in trackers], dtype=float)

        points: List[ParetoPoint] = []
        for i, t in enumerate(trackers):
            dominated = any(
                (x_vals[j] >= x_vals[i] and y_vals[j] >= y_vals[i])
                and (x_vals[j] > x_vals[i] or y_vals[j] > y_vals[i])
                for j in range(len(trackers))
                if j != i
            )
            points.append(
                ParetoPoint(
                    tracker_name=t.name,
                    mean_iou=t.mean_iou,
                    fps=t.fps,
                    is_pareto_optimal=not dominated,
                )
            )
        return points

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def format_leaderboard(self, scores: List[EdgeScore]) -> str:
        """Render a Markdown leaderboard table sorted by EDS.

        Args:
            scores: Output of :meth:`score` (already sorted).

        Returns:
            Markdown-formatted table string.
        """
        header = (
            "| Rank | Tracker | EDS | IoU | FPS | Memory (MB) | "
            "Energy (mJ/fr) | Deployable |"
        )
        sep = "|------|---------|-----|-----|-----|-------------|---------------|------------|"
        rows = [header, sep]
        for rank, s in enumerate(scores, 1):
            dep = "Yes" if s.is_deployable else "No"
            rows.append(
                f"| {rank} | {s.tracker_name} | {s.score:.3f} "
                f"| {s.raw.mean_iou:.3f} | {s.raw.fps:.1f} "
                f"| {s.raw.peak_memory_mb:.1f} "
                f"| {s.raw.energy_per_frame_mj:.3f} | {dep} |"
            )
        return "\n".join(rows)

    def suitability_report(
        self,
        scores: List[EdgeScore],
        profile_name: Optional[str] = None,
    ) -> str:
        """Generate a plain-text deployment suitability report.

        Args:
            scores: Output of :meth:`score`.
            profile_name: Optional hardware profile name for the header.

        Returns:
            Multi-line report string.
        """
        hw_line = (
            f"Hardware Profile : {profile_name} — "
            f"{HARDWARE_PROFILES[profile_name]['description']}"
            if profile_name and profile_name in HARDWARE_PROFILES
            else "Custom Hardware Profile"
        )
        deployable = [s for s in scores if s.is_deployable]
        not_deployable = [s for s in scores if not s.is_deployable]

        lines = [
            "Edge Deployment Suitability Report",
            "=" * 55,
            hw_line,
            f"  Target FPS      : >= {self.target_fps:.0f}",
            f"  Max Memory      : <= {self.max_memory_mb:.0f} MB",
            f"  Weights         : {self.weights}",
            "",
            f"Deployable Trackers ({len(deployable)}/{len(scores)}):",
        ]
        for s in deployable:
            lines.append(
                f"  [{s.score:.3f}] {s.tracker_name:<20s}  "
                f"IoU={s.raw.mean_iou:.3f}  FPS={s.raw.fps:.1f}  "
                f"Mem={s.raw.peak_memory_mb:.0f} MB"
            )
        if not_deployable:
            lines += ["", "Not Meeting Constraints:"]
            for s in not_deployable:
                reasons = []
                if not s.meets_fps_target:
                    reasons.append(f"FPS={s.raw.fps:.1f} < {self.target_fps:.0f}")
                if not s.meets_memory_target:
                    reasons.append(
                        f"Mem={s.raw.peak_memory_mb:.0f} MB > {self.max_memory_mb:.0f} MB"
                    )
                lines.append(
                    f"  [{s.score:.3f}] {s.tracker_name:<20s}  ({', '.join(reasons)})"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_weights(cls, weights: Dict[str, float]) -> None:
        missing = cls._REQUIRED_WEIGHT_KEYS - set(weights)
        if missing:
            raise ValueError(f"Missing weight keys: {sorted(missing)}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    @staticmethod
    def _normalize(trackers: List[TrackerMetrics]) -> Dict[str, List[float]]:
        """Min-max normalise each axis.  Memory and energy are inverted."""

        def minmax(vals: np.ndarray, invert: bool = False) -> np.ndarray:
            lo, hi = vals.min(), vals.max()
            if hi == lo:
                # All trackers equal on this axis → give everyone 1.0
                return np.ones_like(vals, dtype=float)
            normed = (vals - lo) / (hi - lo)
            return 1.0 - normed if invert else normed

        accuracy = minmax(np.array([t.mean_iou for t in trackers]))
        fps = minmax(np.array([t.fps for t in trackers]))
        memory = minmax(np.array([t.peak_memory_mb for t in trackers]), invert=True)
        energy = minmax(
            np.array([t.energy_per_frame_mj for t in trackers]), invert=True
        )

        return {
            "accuracy": accuracy.tolist(),
            "fps": fps.tolist(),
            "memory": memory.tolist(),
            "energy": energy.tolist(),
        }
