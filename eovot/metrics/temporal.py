"""Temporal consistency metrics for edge deployment quality assessment.

Standard VOT benchmarks measure *what* a tracker predicts (accuracy via IoU,
success curves) and *how fast* it runs (FPS/latency).  They do not measure
*how smoothly* it predicts — a critical property for real edge deployments
where downstream consumers (robot controllers, video analytics pipelines) rely
on stable, jitter-free bounding-box streams.

This module quantifies four complementary dimensions of prediction smoothness:

Position Jitter (σ_pos)
~~~~~~~~~~~~~~~~~~~~~~~
Normalised standard deviation of the frame-to-frame displacement vector.
High jitter means the tracker oscillates around the target without smoothly
following its motion — problematic for Kalman-filter fusion or PID control.

    jitter = std(||Δcenter_t||) / mean_diagonal

Normalisation by the mean box diagonal makes the metric scale-invariant
across sequences with different target sizes.

Scale Jitter (σ_scale)
~~~~~~~~~~~~~~~~~~~~~~
Standard deviation of the ratio of consecutive bounding-box areas.
Trackers that continuously expand and contract the predicted box create
instability in downstream object-size estimation.

    scale_jitter = std(area_{t+1} / area_t)

Velocity Outlier Ratio (VOR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fraction of frames where the magnitude of acceleration (second derivative of
position) exceeds three standard deviations above the mean.  Outliers
correspond to sudden trajectory jumps — a common failure mode when
correlation-filter trackers lose the target and snap to a nearby distractor.

    VOR = |{t : |a_t - μ_a| > 3σ_a}| / N

Smoothness Score
~~~~~~~~~~~~~~~~
A single composite scalar combining the three raw metrics into a
deployment-readiness score:

    S = 1 / (1 + σ_pos + σ_scale + VOR)

``S ∈ (0, 1]``.  Higher is smoother.  A perfectly smooth tracker (zero
jitter, zero outliers) achieves ``S = 1.0``.

Typical usage::

    from eovot.metrics.temporal import TemporalConsistencyAnalyzer

    analyzer = TemporalConsistencyAnalyzer()

    # Per-sequence analysis
    result = analyzer.analyze(predictions, tracker_name="MOSSE", sequence_name="car1")
    print(result)
    # TemporalResult[MOSSE on car1] S=0.743 jitter=0.052 scale_jitter=0.061 VOR=0.000

    # Aggregate over a full benchmark run
    seq_preds = {r.sequence_name: r.predictions for r in benchmark_result.sequence_results}
    agg = analyzer.analyze_benchmark(seq_preds, tracker_name="MOSSE")
    print(agg["aggregate"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TemporalConsistencyResult:
    """Per-sequence temporal consistency summary.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        sequence_name: Sequence identifier.
        position_jitter: Normalised std of frame-to-frame displacement (lower = smoother).
        scale_jitter: Std of consecutive area ratios (lower = smoother).
        velocity_outlier_ratio: Fraction of frames with acceleration outliers (lower = smoother).
        smoothness_score: Composite smoothness scalar in ``(0, 1]`` (higher = smoother).
        mean_velocity_px: Mean per-frame displacement in pixels.
        num_frames: Number of frames in the analysed sequence.
    """

    tracker_name: str
    sequence_name: str
    position_jitter: float
    scale_jitter: float
    velocity_outlier_ratio: float
    smoothness_score: float
    mean_velocity_px: float
    num_frames: int

    def __str__(self) -> str:
        return (
            f"TemporalResult[{self.tracker_name} on {self.sequence_name}] "
            f"S={self.smoothness_score:.3f}  "
            f"jitter={self.position_jitter:.4f}  "
            f"scale_jitter={self.scale_jitter:.4f}  "
            f"VOR={self.velocity_outlier_ratio:.3f}  "
            f"mean_vel={self.mean_velocity_px:.1f}px"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "position_jitter": round(self.position_jitter, 6),
            "scale_jitter": round(self.scale_jitter, 6),
            "velocity_outlier_ratio": round(self.velocity_outlier_ratio, 4),
            "smoothness_score": round(self.smoothness_score, 4),
            "mean_velocity_px": round(self.mean_velocity_px, 3),
            "num_frames": self.num_frames,
        }


class TemporalConsistencyAnalyzer:
    """Analyse the smoothness of bounding-box prediction streams.

    All methods accept ``(N, 4)`` numpy arrays of bounding boxes in
    ``(x, y, w, h)`` format — the same convention used throughout EOVOT.

    Args:
        outlier_threshold: Number of standard deviations above the mean
            acceleration that defines a velocity outlier.  Default: ``3.0``
            (3σ rule, ~0.3 % false-positive rate under Gaussian motion).

    Example::

        analyzer = TemporalConsistencyAnalyzer()
        result = analyzer.analyze(seq_result.predictions, "MOSSE", "car1")
        print(result.smoothness_score)
    """

    def __init__(self, outlier_threshold: float = 3.0) -> None:
        if outlier_threshold <= 0:
            raise ValueError(
                f"outlier_threshold must be positive, got {outlier_threshold}"
            )
        self.outlier_threshold = outlier_threshold

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    def compute_position_jitter(
        self, predictions: np.ndarray
    ) -> Tuple[float, float]:
        """Compute normalised position jitter and mean velocity.

        Args:
            predictions: ``(N, 4)`` array of predicted boxes ``(x, y, w, h)``.

        Returns:
            ``(jitter, mean_velocity_px)`` where *jitter* is normalised by
            the mean box diagonal and *mean_velocity_px* is the mean
            frame-to-frame displacement in pixels.  Both are ``0.0`` for
            sequences with fewer than two frames.
        """
        if len(predictions) < 2:
            return 0.0, 0.0

        centers = self._box_centers(predictions)
        displacements = np.diff(centers, axis=0)          # (N-1, 2)
        speeds = np.linalg.norm(displacements, axis=1)    # (N-1,)

        mean_vel = float(speeds.mean())
        # Normalise jitter by mean box diagonal for scale invariance
        diagonals = np.sqrt(predictions[:, 2] ** 2 + predictions[:, 3] ** 2)
        mean_diag = float(diagonals.mean())
        norm = mean_diag if mean_diag > 1e-6 else 1.0
        jitter = float(speeds.std() / norm)
        return jitter, mean_vel

    def compute_scale_jitter(self, predictions: np.ndarray) -> float:
        """Compute standard deviation of consecutive area ratios.

        Args:
            predictions: ``(N, 4)`` array of predicted boxes.

        Returns:
            Scale jitter scalar ≥ 0.  Returns ``0.0`` for fewer than two frames.
        """
        if len(predictions) < 2:
            return 0.0

        areas = predictions[:, 2] * predictions[:, 3]            # (N,)
        # Avoid division by near-zero area (degenerate boxes)
        safe_areas = np.where(areas > 1e-6, areas, 1e-6)
        ratios = safe_areas[1:] / safe_areas[:-1]                # (N-1,)
        return float(ratios.std())

    def compute_velocity_outlier_ratio(self, predictions: np.ndarray) -> float:
        """Compute the fraction of frames with acceleration outliers.

        An outlier is a frame where the change in speed (acceleration magnitude)
        exceeds ``mean_acc + outlier_threshold × std_acc``.

        Args:
            predictions: ``(N, 4)`` array of predicted boxes.

        Returns:
            VOR in ``[0, 1]``.  Returns ``0.0`` for sequences with fewer
            than three frames or zero acceleration variance.
        """
        if len(predictions) < 3:
            return 0.0

        centers = self._box_centers(predictions)
        velocities = np.diff(centers, axis=0)               # (N-1, 2)
        speeds = np.linalg.norm(velocities, axis=1)         # (N-1,)
        accelerations = np.abs(np.diff(speeds))             # (N-2,)

        if len(accelerations) == 0:
            return 0.0

        mean_acc = float(accelerations.mean())
        std_acc = float(accelerations.std())
        if std_acc < 1e-9:
            return 0.0  # perfectly uniform acceleration — no outliers

        threshold = mean_acc + self.outlier_threshold * std_acc
        n_outliers = int((accelerations > threshold).sum())
        return float(n_outliers / len(accelerations))

    # ------------------------------------------------------------------
    # High-level analysis entry points
    # ------------------------------------------------------------------

    def analyze(
        self,
        predictions: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> TemporalConsistencyResult:
        """Run full temporal consistency analysis on one prediction sequence.

        Args:
            predictions: ``(N, 4)`` array of predicted bounding boxes.
                Use ``SequenceResult.predictions`` from the benchmark engine.
            tracker_name: Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`TemporalConsistencyResult` with all metrics populated.
        """
        predictions = np.asarray(predictions, dtype=np.float64)

        if len(predictions) < 2:
            return TemporalConsistencyResult(
                tracker_name=tracker_name,
                sequence_name=sequence_name,
                position_jitter=0.0,
                scale_jitter=0.0,
                velocity_outlier_ratio=0.0,
                smoothness_score=1.0,
                mean_velocity_px=0.0,
                num_frames=len(predictions),
            )

        pos_jitter, mean_vel = self.compute_position_jitter(predictions)
        scale_jitter = self.compute_scale_jitter(predictions)
        vor = self.compute_velocity_outlier_ratio(predictions)
        smoothness = 1.0 / (1.0 + pos_jitter + scale_jitter + vor)

        return TemporalConsistencyResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            position_jitter=pos_jitter,
            scale_jitter=scale_jitter,
            velocity_outlier_ratio=vor,
            smoothness_score=smoothness,
            mean_velocity_px=mean_vel,
            num_frames=len(predictions),
        )

    def analyze_benchmark(
        self,
        sequence_predictions: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> Dict:
        """Aggregate temporal consistency analysis across all sequences.

        Args:
            sequence_predictions: Mapping ``{sequence_name: predictions_array}``.
                Each value is an ``(N, 4)`` float array.  Use
                ``{r.sequence_name: r.predictions for r in result.sequence_results}``
                to build this from a :class:`~eovot.benchmark.engine.BenchmarkResult`.
            tracker_name: Identifier for the tracker under evaluation.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: TemporalConsistencyResult}``
            * ``"aggregate"`` — summary scalars averaged across all sequences
        """
        per_seq: Dict[str, TemporalConsistencyResult] = {}
        for seq_name, preds in sequence_predictions.items():
            per_seq[seq_name] = self.analyze(
                preds, tracker_name=tracker_name, sequence_name=seq_name
            )

        if not per_seq:
            return {"per_sequence": {}, "aggregate": {}}

        results = list(per_seq.values())
        return {
            "per_sequence": per_seq,
            "aggregate": {
                "tracker_name": tracker_name,
                "num_sequences": len(results),
                "mean_position_jitter": round(
                    float(np.mean([r.position_jitter for r in results])), 6
                ),
                "mean_scale_jitter": round(
                    float(np.mean([r.scale_jitter for r in results])), 6
                ),
                "mean_velocity_outlier_ratio": round(
                    float(np.mean([r.velocity_outlier_ratio for r in results])), 4
                ),
                "mean_smoothness_score": round(
                    float(np.mean([r.smoothness_score for r in results])), 4
                ),
                "mean_velocity_px": round(
                    float(np.mean([r.mean_velocity_px for r in results])), 3
                ),
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _box_centers(boxes: np.ndarray) -> np.ndarray:
        """Return ``(N, 2)`` array of box centre coordinates from ``(N, 4)``."""
        return boxes[:, :2] + boxes[:, 2:] / 2.0
