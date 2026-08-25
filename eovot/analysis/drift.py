"""Tracker drift analysis for edge deployment reliability assessment.

*Drift* is the gradual degradation of tracking accuracy over time within a
sequence — distinct from the discrete *failure* events counted by
:class:`~eovot.metrics.robustness.RobustnessAnalyzer`.  A tracker may never
trigger a hard failure (IoU never drops below the 0.1 threshold) and yet
steadily lose the target over hundreds of frames, making it unreliable for
long-running edge deployments.

This module quantifies drift with four complementary metrics:

**Stable duration**
    Consecutive frames at the start of the sequence where IoU stays above
    ``stable_threshold``.  Answers: *"How long can I trust this tracker
    without intervention?"*

**Decay rate**
    Slope of a linear regression of IoU vs. frame index (units: IoU/frame).
    Negative values indicate drift; zero or positive indicate stable tracking.

**Stability ratio**
    Mean IoU in the second half of the sequence divided by the mean IoU in
    the first half.  Values below 1.0 confirm drift; values above 1.0
    indicate improving accuracy (e.g. re-lock after initial jitter).

**Half-life**
    Estimated frame at which IoU reaches half its value at frame ``burn_in``.
    Computed from the linear regression; ``None`` when the tracker is stable
    (positive or zero slope).  Useful as a practical deployment timeout.

Typical usage::

    from eovot.analysis.drift import DriftAnalyzer
    from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult

    # After running a benchmark:
    result: BenchmarkResult = engine.run(tracker, dataset)

    analyzer = DriftAnalyzer(stable_threshold=0.3)
    report   = analyzer.analyze_benchmark(result)

    print(f"Mean stable duration: {report['aggregate']['mean_stable_duration_frames']:.1f} frames")
    print(report['aggregate']['verdict'])
    print(analyzer.to_markdown_table(report['per_sequence'].values()))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


@dataclass
class DriftResult:
    """Drift statistics for a single tracking sequence.

    All frame counts are relative to the start of the sequence.
    IoU arrays are 0-indexed with frame 0 being the initialisation frame.
    """

    tracker_name: str
    sequence_name: str
    n_frames: int
    """Total number of frames in the sequence."""

    mean_iou: float
    """Mean IoU across all frames (including burn-in)."""

    stable_duration: int
    """Number of consecutive frames at the start (after burn-in) where
    IoU stays at or above ``stable_threshold``.  A sequence where the
    tracker never drops below the threshold has ``stable_duration == n_frames``."""

    decay_rate: float
    """Linear regression slope of IoU vs. frame index (IoU per frame).
    Negative indicates degradation; near-zero or positive indicates stability."""

    stability_ratio: float
    """Mean IoU over the second half of the sequence divided by the mean IoU
    over the first half.  Values < 1.0 confirm drift; > 1.0 indicate recovery."""

    half_life_frames: Optional[float]
    """Estimated frame at which IoU reaches half its value at ``burn_in``,
    derived from the linear model.  ``None`` when ``decay_rate >= 0`` (stable)."""

    collapse_frame: Optional[int]
    """First frame where IoU drops below ``collapse_threshold`` and does not
    recover within the sequence.  ``None`` if no permanent collapse is detected."""

    def is_drifting(self) -> bool:
        """Return ``True`` when the tracker shows statistically meaningful drift.

        Drift is declared when ``decay_rate < 0`` and ``stability_ratio < 0.9``
        (IoU in the second half is more than 10 % lower than in the first half).
        """
        return self.decay_rate < 0 and self.stability_ratio < 0.9

    def __str__(self) -> str:
        half = f"{self.half_life_frames:.0f}" if self.half_life_frames is not None else "∞"
        collapse = f"{self.collapse_frame}" if self.collapse_frame is not None else "none"
        return (
            f"DriftResult[{self.tracker_name} on {self.sequence_name}] "
            f"stable={self.stable_duration}fr  decay={self.decay_rate:+.6f}/fr  "
            f"ratio={self.stability_ratio:.3f}  half_life={half}fr  "
            f"collapse={collapse}"
        )


class DriftAnalyzer:
    """Measure how tracker accuracy degrades over sequence length.

    Args:
        stable_threshold: IoU value at or above which the tracker is
            considered *stable*.  Used to compute ``stable_duration``.
            Default: ``0.3``.
        collapse_threshold: IoU below which the tracker is considered
            *collapsed* (permanently lost the target).  Default: ``0.1``.
        burn_in_frames: Frames at the start of each sequence to skip before
            analysis.  Frame 0 is the ground-truth initialisation frame
            (IoU = 1.0 by construction) and is always excluded.
            Default: ``5``.
    """

    def __init__(
        self,
        stable_threshold: float = 0.3,
        collapse_threshold: float = 0.1,
        burn_in_frames: int = 5,
    ) -> None:
        self.stable_threshold = stable_threshold
        self.collapse_threshold = collapse_threshold
        self.burn_in_frames = burn_in_frames

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_sequence(
        self,
        ious: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> DriftResult:
        """Compute drift metrics for a single sequence.

        Args:
            ious:          Per-frame IoU array, shape ``(N,)``.  Frame 0 is
                           the initialisation frame.
            tracker_name:  Tracker identifier stored in the result.
            sequence_name: Sequence identifier stored in the result.

        Returns:
            :class:`DriftResult` populated with all drift statistics.
        """
        ious = np.asarray(ious, dtype=np.float64)
        n = len(ious)

        # Work on the post-burn-in portion.
        start = min(self.burn_in_frames, n)
        active = ious[start:] if start < n else np.empty(0)

        mean_iou = float(ious.mean()) if n > 0 else 0.0

        stable_duration = self._stable_duration(active)
        decay_rate, half_life = self._decay_metrics(active)
        stability_ratio = self._stability_ratio(active)
        collapse_frame = self._collapse_frame(active, offset=start)

        return DriftResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            n_frames=n,
            mean_iou=mean_iou,
            stable_duration=stable_duration,
            decay_rate=decay_rate,
            stability_ratio=stability_ratio,
            half_life_frames=half_life,
            collapse_frame=collapse_frame,
        )

    def analyze_benchmark(
        self,
        result: "BenchmarkResult",
    ) -> Dict:
        """Aggregate drift analysis across all sequences in a benchmark run.

        Args:
            result: :class:`~eovot.benchmark.engine.BenchmarkResult` from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: DriftResult}``
            * ``"aggregate"`` — scalar summary across all sequences
        """
        per_seq: Dict[str, DriftResult] = {}
        for sr in result.sequence_results:
            per_seq[sr.sequence_name] = self.analyze_sequence(
                sr.ious,
                tracker_name=result.tracker_name,
                sequence_name=sr.sequence_name,
            )

        agg = self._aggregate(list(per_seq.values()), result.tracker_name)
        return {"per_sequence": per_seq, "aggregate": agg}

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_markdown_table(self, results: Iterable[DriftResult]) -> str:
        """Format per-sequence drift results as a Markdown table.

        Args:
            results: Iterable of :class:`DriftResult` objects.

        Returns:
            Multi-line Markdown table string ready to embed in reports.
        """
        rows = list(results)
        if not rows:
            return "_No drift results to display._\n"

        lines = [
            "| Sequence | mIoU | Stable (fr) | Decay (IoU/fr) | Ratio | Half-life | Collapse |",
            "|----------|-----:|------------:|---------------:|------:|----------:|---------:|",
        ]
        for r in sorted(rows, key=lambda x: x.stable_duration, reverse=True):
            half = f"{r.half_life_frames:.0f}" if r.half_life_frames is not None else "∞"
            col = f"{r.collapse_frame}" if r.collapse_frame is not None else "—"
            lines.append(
                f"| {r.sequence_name} "
                f"| {r.mean_iou:.3f} "
                f"| {r.stable_duration} "
                f"| {r.decay_rate:+.6f} "
                f"| {r.stability_ratio:.3f} "
                f"| {half} "
                f"| {col} |"
            )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stable_duration(self, active: np.ndarray) -> int:
        """Count consecutive frames at the start where IoU >= stable_threshold."""
        for i, v in enumerate(active):
            if float(v) < self.stable_threshold:
                return i
        return len(active)

    def _decay_metrics(
        self, active: np.ndarray
    ) -> tuple[float, Optional[float]]:
        """Compute linear decay rate and half-life from the active IoU array.

        Returns:
            ``(slope, half_life_frames)`` where slope is in IoU/frame.
            ``half_life_frames`` is ``None`` when slope >= 0 (no decay).
        """
        n = len(active)
        if n < 2:
            return 0.0, None

        x = np.arange(n, dtype=np.float64)
        # Least-squares linear fit.
        x_mean = x.mean()
        y_mean = active.mean()
        ss_xy = float(np.dot(x - x_mean, active - y_mean))
        ss_xx = float(np.dot(x - x_mean, x - x_mean))
        slope = ss_xy / ss_xx if ss_xx > 0 else 0.0

        half_life: Optional[float] = None
        if slope < 0:
            # IoU(t) ≈ intercept + slope * t; solve for IoU(t) = IoU(0) / 2
            intercept = y_mean - slope * x_mean
            iou0 = float(active[0]) if len(active) > 0 else intercept
            target = iou0 / 2.0
            if slope < 0 and intercept > target:
                half_life = (target - intercept) / slope

        return float(slope), half_life

    def _stability_ratio(self, active: np.ndarray) -> float:
        """Ratio of mean IoU in the second half to the first half.

        Returns 1.0 for very short sequences or when the first-half mean is zero.
        """
        n = len(active)
        if n < 2:
            return 1.0
        mid = n // 2
        first = float(active[:mid].mean())
        second = float(active[mid:].mean())
        return second / first if first > 1e-6 else 1.0

    def _collapse_frame(
        self, active: np.ndarray, offset: int
    ) -> Optional[int]:
        """Find the first frame where IoU stays below collapse_threshold to the end.

        Scans backwards: if ``active[i:]`` is entirely below
        ``collapse_threshold``, frame ``i + offset`` is the collapse point.

        Returns:
            Absolute frame index (including burn-in offset), or ``None``.
        """
        n = len(active)
        for i in range(n - 1, -1, -1):
            if float(active[i]) >= self.collapse_threshold:
                # First frame from the end that is still alive.
                if i + 1 < n:
                    return int(i + 1 + offset)
                return None  # never collapsed
        # All frames below threshold — collapsed from the very start.
        return int(offset) if n > 0 else None

    def _aggregate(self, results: List[DriftResult], tracker_name: str) -> Dict:
        """Compute aggregate statistics and a plain-language deployment verdict."""
        n = len(results)
        if n == 0:
            return {"tracker_name": tracker_name, "num_sequences": 0}

        mean_stable = float(np.mean([r.stable_duration for r in results]))
        mean_decay = float(np.mean([r.decay_rate for r in results]))
        mean_ratio = float(np.mean([r.stability_ratio for r in results]))
        n_drifting = sum(1 for r in results if r.is_drifting())
        n_collapsed = sum(1 for r in results if r.collapse_frame is not None)

        # Sequences with a finite half-life only.
        half_lives = [r.half_life_frames for r in results if r.half_life_frames is not None]
        mean_half_life: Optional[float] = float(np.mean(half_lives)) if half_lives else None

        # Plain-language verdict for quick report scanning.
        drift_pct = 100.0 * n_drifting / n
        if drift_pct == 0:
            verdict = "Stable — no drift detected across the dataset."
        elif drift_pct < 25:
            verdict = f"Mostly stable — drift observed in {n_drifting}/{n} sequences."
        elif drift_pct < 75:
            verdict = (
                f"Moderate drift — {n_drifting}/{n} sequences show degradation. "
                "Consider periodic re-initialisation."
            )
        else:
            verdict = (
                f"Severe drift — {n_drifting}/{n} sequences degrade significantly. "
                "Recommend automatic re-initialisation or a more robust tracker."
            )

        agg: Dict = {
            "tracker_name": tracker_name,
            "num_sequences": n,
            "mean_stable_duration_frames": round(mean_stable, 1),
            "mean_decay_rate_per_frame": round(mean_decay, 8),
            "mean_stability_ratio": round(mean_ratio, 4),
            "n_drifting_sequences": n_drifting,
            "n_collapsed_sequences": n_collapsed,
            "verdict": verdict,
        }
        if mean_half_life is not None:
            agg["mean_half_life_frames"] = round(mean_half_life, 1)
        return agg
