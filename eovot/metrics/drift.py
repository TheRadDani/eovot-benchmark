"""Gradient-based tracker drift detection for edge deployment early warning.

Standard robustness metrics detect *failures* — frames where IoU drops below
a threshold — but by then the tracker has already lost the target.  On an edge
device this is too late: a robot controller or video-analytics pipeline may
have already acted on several frames of bad predictions.

This module provides **early warning** by modelling the *trend* of IoU over a
sliding window rather than waiting for a binary threshold crossing.  Drift is
defined as a statistically significant negative slope in the recent IoU series
— the tracker is gradually losing the target before catastrophic failure.

Detection method
~~~~~~~~~~~~~~~~
For each window of ``window_size`` frames, a linear regression is fitted to
the IoU values.  The slope (Δ IoU / frame) is the primary drift indicator:

- ``slope > drift_slope_threshold`` → tracker is stable or improving
- ``slope ≤ drift_slope_threshold`` → drift detected in this window

A window-level drift score is computed as::

    window_drift_score = max(0, −slope) / |slope_threshold|

so that steeper negative slopes produce higher scores.

Aggregate drift metrics over a full sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Drift score** (0–1): fraction of windows in drift, weighted by their
  per-window drift score.  Higher = more severe drift.
- **Drift onset frame**: first frame of the first drifting window, or ``None``
  if no drift was detected.
- **Time-to-failure estimate**: frames from drift onset to the first hard
  failure (IoU < ``failure_threshold``), or ``None`` if no failure followed.

Online use (real-time inference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``DriftDetector.is_drifting(recent_ious)`` accepts the last ``window_size``
IoU values from a live tracker and returns ``True`` when the linear trend is
negative enough to trigger a warning.  This is designed to be called after
every frame with a deque of recent IoU observations::

    from collections import deque
    detector = DriftDetector()
    iou_window = deque(maxlen=detector.window_size)

    for frame in video:
        bbox   = tracker.update(frame)
        gt_box = get_ground_truth(frame)  # if available
        iou_window.append(compute_iou(bbox, gt_box))

        if len(iou_window) == detector.window_size:
            if detector.is_drifting(np.array(iou_window)):
                trigger_reinitialization()

Typical offline usage::

    from eovot.metrics.drift import DriftDetector

    detector = DriftDetector(window_size=10, drift_slope_threshold=-0.005)

    result = detector.analyze(ious, tracker_name="MOSSE", sequence_name="car1")
    print(result.drift_score, result.drift_onset_frame)

    # Aggregate across a full benchmark run
    summary = detector.analyze_benchmark(benchmark_result)
    print(summary["aggregate"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Per-sequence drift analysis output.

    Attributes:
        tracker_name: Identifier for the evaluated tracker.
        sequence_name: Identifier for the evaluated sequence.
        drift_score: Aggregate drift severity in ``[0, 1]``.  A score of 0
            means no drift was detected in any window; 1 means every window
            showed maximum-severity drift.
        drift_onset_frame: Frame index at which the first drifting window
            begins, or ``None`` if no drift was detected.
        time_to_failure_frames: Frames from the drift onset to the first hard
            failure (IoU < ``failure_threshold``), or ``None`` when no failure
            followed the onset (or when no drift was detected).
        trend_slope: Linear regression slope over the *entire* IoU sequence
            (Δ IoU / frame).  Negative values indicate an overall declining
            trend; useful as a single-number drift summary for leaderboards.
        window_slopes: Per-window linear regression slopes in chronological
            order.  Length equals ``len(ious) - window_size + 1``.
        warning_frames: Frame indices at which the online ``is_drifting``
            check would have fired (start of each drifting window).
        num_drift_windows: Number of windows classified as drifting.
        num_total_windows: Total number of windows analysed.
    """

    tracker_name: str
    sequence_name: str
    drift_score: float
    drift_onset_frame: Optional[int]
    time_to_failure_frames: Optional[int]
    trend_slope: float
    window_slopes: List[float]
    warning_frames: List[int]
    num_drift_windows: int
    num_total_windows: int

    def __str__(self) -> str:
        onset = (
            f"onset@{self.drift_onset_frame}" if self.drift_onset_frame is not None else "no_onset"
        )
        ttf = (
            f"  ttf={self.time_to_failure_frames}fr"
            if self.time_to_failure_frames is not None
            else ""
        )
        return (
            f"DriftResult[{self.tracker_name} on {self.sequence_name}] "
            f"score={self.drift_score:.4f}  slope={self.trend_slope:+.5f}  "
            f"{onset}{ttf}  "
            f"drift_windows={self.num_drift_windows}/{self.num_total_windows}"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Detect gradual tracker drift from per-frame IoU sequences.

    Args:
        window_size: Number of frames in each sliding window for local trend
            estimation.  Larger windows are more robust to noise but detect
            drift later.  Default: ``10``.
        drift_slope_threshold: A window's linear-regression slope below this
            value (Δ IoU / frame) is classified as drifting.  The default
            ``-0.005`` means the tracker is drifting when IoU is declining at
            more than 0.5 percentage points per frame.  Default: ``-0.005``.
        failure_threshold: IoU below which a frame is counted as a hard
            failure (used to compute time-to-failure).  Should match the value
            used in ``RobustnessAnalyzer``.  Default: ``0.1``.
        stride: Number of frames to advance the window at each step.
            ``stride=1`` gives maximum temporal resolution; larger values
            reduce computation on very long sequences.  Default: ``1``.
    """

    def __init__(
        self,
        window_size: int = 10,
        drift_slope_threshold: float = -0.005,
        failure_threshold: float = 0.1,
        stride: int = 1,
    ) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be ≥ 2, got {window_size}.")
        if stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {stride}.")
        self.window_size = window_size
        self.drift_slope_threshold = drift_slope_threshold
        self.failure_threshold = failure_threshold
        self.stride = stride

        # Pre-compute the OLS X matrix for efficiency when called repeatedly
        # with the same window_size (e.g. in the online loop).
        xs = np.arange(window_size, dtype=np.float64)
        self._xs_mean = xs.mean()
        self._xs_var = float(np.sum((xs - self._xs_mean) ** 2))

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _window_slope(self, ious: np.ndarray) -> float:
        """Compute the OLS linear regression slope for a single window.

        Args:
            ious: 1-D array of length ``window_size``.

        Returns:
            Slope in Δ IoU / frame.  Negative means declining quality.
        """
        if self._xs_var == 0:
            return 0.0
        ys_mean = ious.mean()
        cov = float(np.sum((np.arange(len(ious), dtype=np.float64) - self._xs_mean) * (ious - ys_mean)))
        return cov / self._xs_var

    def _sliding_slopes(self, ious: np.ndarray) -> List[float]:
        """Compute per-window slopes over the full sequence."""
        n = len(ious)
        slopes: List[float] = []
        start = 0
        while start + self.window_size <= n:
            window = ious[start: start + self.window_size]
            slopes.append(self._window_slope(window))
            start += self.stride
        return slopes

    def _overall_slope(self, ious: np.ndarray) -> float:
        """Linear regression slope over the *entire* IoU sequence."""
        n = len(ious)
        if n < 2:
            return 0.0
        xs = np.arange(n, dtype=np.float64)
        xs_mean = xs.mean()
        xs_var = float(np.sum((xs - xs_mean) ** 2))
        if xs_var == 0:
            return 0.0
        ys_mean = ious.mean()
        cov = float(np.sum((xs - xs_mean) * (ious - ys_mean)))
        return cov / xs_var

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_drifting(self, recent_ious: np.ndarray) -> bool:
        """Online check: returns True when the last window shows drift.

        Designed to be called at every frame with a rolling buffer of the
        most recent ``window_size`` IoU values.  Returns immediately without
        storing state, making it safe to call from a tracking loop.

        Args:
            recent_ious: 1-D array of the most recent IoU observations.
                Must have at least ``window_size`` elements; if shorter,
                returns ``False`` (insufficient data).

        Returns:
            ``True`` when a statistically meaningful negative slope is
            detected — the tracker should be considered at risk of failure.
        """
        if len(recent_ious) < self.window_size:
            return False
        window = np.asarray(recent_ious[-self.window_size:], dtype=np.float64)
        slope = self._window_slope(window)
        return slope <= self.drift_slope_threshold

    def analyze(
        self,
        ious: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> DriftResult:
        """Run full drift analysis on a single sequence.

        Args:
            ious:          Per-frame IoU array, shape ``(N,)``.
            tracker_name:  Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`DriftResult` with all drift statistics populated.
        """
        ious = np.asarray(ious, dtype=np.float64)
        n = len(ious)

        if n < self.window_size:
            return DriftResult(
                tracker_name=tracker_name,
                sequence_name=sequence_name,
                drift_score=0.0,
                drift_onset_frame=None,
                time_to_failure_frames=None,
                trend_slope=self._overall_slope(ious),
                window_slopes=[],
                warning_frames=[],
                num_drift_windows=0,
                num_total_windows=0,
            )

        slopes = self._sliding_slopes(ious)
        num_total = len(slopes)
        abs_thr = abs(self.drift_slope_threshold) or 1e-9

        drift_onset_frame: Optional[int] = None
        warning_frames: List[int] = []
        weighted_drift_sum = 0.0
        num_drift = 0

        for i, slope in enumerate(slopes):
            frame_start = i * self.stride
            if slope <= self.drift_slope_threshold:
                num_drift += 1
                per_window_score = min(1.0, max(0.0, -slope / abs_thr))
                weighted_drift_sum += per_window_score
                warning_frames.append(frame_start)
                if drift_onset_frame is None:
                    drift_onset_frame = frame_start

        # Aggregate drift score: weighted fraction of drifting windows.
        drift_score = weighted_drift_sum / num_total if num_total > 0 else 0.0
        drift_score = min(1.0, drift_score)

        # Time-to-failure: frames from drift onset to first hard failure.
        time_to_failure: Optional[int] = None
        if drift_onset_frame is not None:
            for fi in range(drift_onset_frame, n):
                if float(ious[fi]) < self.failure_threshold:
                    time_to_failure = fi - drift_onset_frame
                    break

        return DriftResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            drift_score=drift_score,
            drift_onset_frame=drift_onset_frame,
            time_to_failure_frames=time_to_failure,
            trend_slope=self._overall_slope(ious),
            window_slopes=slopes,
            warning_frames=warning_frames,
            num_drift_windows=num_drift,
            num_total_windows=num_total,
        )

    def analyze_benchmark(self, result: "BenchmarkResult") -> Dict:
        """Aggregate drift analysis across all sequences in a benchmark run.

        Args:
            result: A ``BenchmarkResult`` from ``BenchmarkEngine.run()``.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: DriftResult}``
            * ``"aggregate"`` — summary scalars across all sequences:

              - ``"tracker_name"``
              - ``"num_sequences"``
              - ``"mean_drift_score"``
              - ``"fraction_sequences_with_drift"``
              - ``"mean_time_to_failure_frames"`` (over sequences where
                onset preceded a failure)
              - ``"mean_trend_slope"``
        """
        per_seq: Dict[str, DriftResult] = {}
        for seq_result in result.sequence_results:
            dr = self.analyze(
                ious=seq_result.ious,
                tracker_name=result.tracker_name,
                sequence_name=seq_result.sequence_name,
            )
            per_seq[seq_result.sequence_name] = dr

        n = len(per_seq)
        results_list = list(per_seq.values())

        mean_drift = float(np.mean([r.drift_score for r in results_list])) if n else 0.0
        frac_drift = (
            sum(1 for r in results_list if r.drift_onset_frame is not None) / n
        ) if n else 0.0
        ttf_values = [
            r.time_to_failure_frames
            for r in results_list
            if r.time_to_failure_frames is not None
        ]
        mean_ttf = float(np.mean(ttf_values)) if ttf_values else None
        mean_slope = float(np.mean([r.trend_slope for r in results_list])) if n else 0.0

        return {
            "per_sequence": per_seq,
            "aggregate": {
                "tracker_name": result.tracker_name,
                "num_sequences": n,
                "mean_drift_score": round(mean_drift, 6),
                "fraction_sequences_with_drift": round(frac_drift, 4),
                "mean_time_to_failure_frames": (
                    round(mean_ttf, 2) if mean_ttf is not None else None
                ),
                "mean_trend_slope": round(mean_slope, 8),
            },
        }

    def to_markdown_table(self, results: Dict[str, "DriftResult"]) -> str:
        """Format per-sequence drift results as a Markdown table.

        Args:
            results: ``{sequence_name: DriftResult}`` from
                :meth:`analyze_benchmark`.

        Returns:
            Multi-line Markdown table string sorted by drift score descending.
        """
        rows = sorted(results.items(), key=lambda kv: kv[1].drift_score, reverse=True)
        lines = [
            "| Sequence | Drift Score | Trend Slope | Onset Frame | TTF (fr) | Warnings |",
            "|----------|------------:|------------:|------------:|---------:|---------:|",
        ]
        for seq_name, dr in rows:
            onset = str(dr.drift_onset_frame) if dr.drift_onset_frame is not None else "—"
            ttf = str(dr.time_to_failure_frames) if dr.time_to_failure_frames is not None else "—"
            lines.append(
                f"| {seq_name} | {dr.drift_score:.4f} "
                f"| {dr.trend_slope:+.5f} | {onset} | {ttf} "
                f"| {len(dr.warning_frames)} |"
            )
        return "\n".join(lines)
