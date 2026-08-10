"""Sequence difficulty scoring and stratified leaderboard analysis.

This module quantifies how hard each tracking sequence is, enabling
difficulty-stratified benchmarks that reveal where trackers succeed or fail.

Difficulty is a composite of three signals:

1. **Failure rate** (weight 0.45) — fraction of frames where the tracker's
   IoU falls below 0.5.  Requires a :class:`~eovot.benchmark.engine.SequenceResult`.
   This is the primary signal: sustained tracker failure is the strongest
   evidence of a hard sequence.

2. **Motion complexity** (weight 0.25) — frame-to-frame GT centre displacement
   normalised by the mean box diagonal, then compressed with ``tanh``.
   A score of 0.5 corresponds to the OTB "fast motion" threshold (20 % of
   diagonal per frame).

3. **Attribute burden** (weight 0.30) — weighted sum of active VOT challenge
   attributes (see :data:`ATTRIBUTE_WEIGHTS`).  Fast motion and occlusion
   contribute more than long sequence or out-of-plane rotation.

Difficulty levels:

+------------+-------------------+
| Level      | Composite score   |
+============+===================+
| EASY       | < 0.25            |
+------------+-------------------+
| MEDIUM     | 0.25 – 0.50       |
+------------+-------------------+
| HARD       | 0.50 – 0.75       |
+------------+-------------------+
| VERY_HARD  | ≥ 0.75            |
+------------+-------------------+

Typical usage::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.analysis.difficulty import DifficultyAnalyzer

    engine   = BenchmarkEngine(verbose=False)
    result   = engine.run(MOSSETracker(), SyntheticDataset(10), dataset_name="Syn")

    analyzer = DifficultyAnalyzer()
    scores   = analyzer.analyze_benchmark(result)
    for s in scores:
        print(f"{s.sequence_name:30s} {s.difficulty_level.value:10s} "
              f"score={s.difficulty_score:.3f}  dominant={s.dominant_challenge}")

    print()
    print(analyzer.to_stratified_leaderboard(scores, result))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..metrics.attributes import ALL_ATTRIBUTES, AttributeDetector, SequenceAttributes

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult, SequenceResult


# ---------------------------------------------------------------------------
# Difficulty levels
# ---------------------------------------------------------------------------

class DifficultyLevel(str, Enum):
    """Categorical difficulty label for a tracking sequence."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


# Composite score thresholds: (upper_bound, label) in ascending order.
_LEVEL_THRESHOLDS = [
    (0.25, DifficultyLevel.EASY),
    (0.50, DifficultyLevel.MEDIUM),
    (0.75, DifficultyLevel.HARD),
]


# ---------------------------------------------------------------------------
# Per-attribute contribution weights
# ---------------------------------------------------------------------------

#: Contribution of each attribute to the attribute-burden component.
#: Weights sum to 1.0 — ``_compute_attribute_score`` returns a value in [0, 1].
ATTRIBUTE_WEIGHTS: Dict[str, float] = {
    "fast_motion":        0.25,
    "partial_occlusion":  0.25,
    "scale_variation":    0.20,
    "low_resolution":     0.15,
    "deformation":        0.10,
    "out_plane_rotation": 0.03,
    "long_sequence":      0.02,
}

_MAX_ATTRIBUTE_SCORE: float = sum(ATTRIBUTE_WEIGHTS.values())  # 1.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SequenceDifficultyScore:
    """Difficulty profile for a single tracking sequence.

    Attributes:
        sequence_name: Identifier matching the source sequence.
        failure_rate: Fraction of frames where the tracker's IoU < threshold.
            Set to ``0.0`` when computed without tracker results (GT-only mode).
        motion_score: Normalised motion complexity ∈ [0, 1].  Derived from
            frame-to-frame GT centre displacements.
        attribute_score: Weighted fraction of active VOT challenge attributes,
            ∈ [0, 1].  Uses :data:`ATTRIBUTE_WEIGHTS`.
        difficulty_score: Final composite score ∈ [0, 1].
        difficulty_level: Categorical label from :class:`DifficultyLevel`.
        dominant_challenge: Highest-weight active attribute name, or
            ``"none"`` when no attributes are detected.
        active_attributes: VOT attribute names detected as present.
        num_frames: Frame count in the scored sequence.
    """

    sequence_name: str
    failure_rate: float
    motion_score: float
    attribute_score: float
    difficulty_score: float
    difficulty_level: DifficultyLevel
    dominant_challenge: str
    active_attributes: List[str] = field(default_factory=list)
    num_frames: int = 0

    def __str__(self) -> str:
        attrs = ", ".join(self.active_attributes) or "none"
        return (
            f"SequenceDifficultyScore({self.sequence_name!r}  "
            f"level={self.difficulty_level.value}  "
            f"score={self.difficulty_score:.3f}  "
            f"failure={self.failure_rate:.2f}  "
            f"motion={self.motion_score:.2f}  "
            f"attrs=[{attrs}])"
        )

    def to_dict(self) -> Dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "sequence_name": self.sequence_name,
            "difficulty_level": self.difficulty_level.value,
            "difficulty_score": round(self.difficulty_score, 4),
            "failure_rate": round(self.failure_rate, 4),
            "motion_score": round(self.motion_score, 4),
            "attribute_score": round(self.attribute_score, 4),
            "dominant_challenge": self.dominant_challenge,
            "active_attributes": list(self.active_attributes),
            "num_frames": self.num_frames,
        }


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class DifficultyAnalyzer:
    """Assign difficulty scores to tracking sequences.

    Two scoring modes are provided:

    * :meth:`score_from_groundtruth` — tracker-independent difficulty derived
      from GT box statistics only (motion and attribute burden).  Useful for
      characterising datasets without running any tracker.

    * :meth:`score_from_result` — tracker-relative difficulty that also
      incorporates the tracker's per-frame failure rate.  Use this after
      running :class:`~eovot.benchmark.engine.BenchmarkEngine` when you want
      to understand *where* a specific tracker struggles.

    :meth:`analyze_benchmark` is a convenience wrapper over
    :meth:`score_from_result` for an entire
    :class:`~eovot.benchmark.engine.BenchmarkResult`.

    Args:
        failure_iou_threshold: IoU threshold below which a frame is counted
            as a tracker failure.  Default: ``0.50``.
        failure_weight: Composite weight for the failure-rate component.
        motion_weight: Composite weight for the motion-complexity component.
        attribute_weight: Composite weight for the attribute-burden component.

    Raises:
        ValueError: If ``failure_iou_threshold`` is outside ``(0, 1]`` or if
            the three weights do not sum to 1.0.
    """

    def __init__(
        self,
        failure_iou_threshold: float = 0.50,
        failure_weight: float = 0.45,
        motion_weight: float = 0.25,
        attribute_weight: float = 0.30,
    ) -> None:
        if not 0.0 < failure_iou_threshold <= 1.0:
            raise ValueError(
                f"failure_iou_threshold must be in (0, 1], got {failure_iou_threshold}"
            )
        total = failure_weight + motion_weight + attribute_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f} "
                f"(failure={failure_weight} + motion={motion_weight} "
                f"+ attribute={attribute_weight})"
            )
        self._iou_thresh = failure_iou_threshold
        self._fw = failure_weight
        self._mw = motion_weight
        self._aw = attribute_weight
        self._detector = AttributeDetector()

    # ------------------------------------------------------------------
    # Scoring — GT-only mode
    # ------------------------------------------------------------------

    def score_from_groundtruth(
        self,
        ground_truths: np.ndarray,
        sequence_name: str = "",
    ) -> SequenceDifficultyScore:
        """Score a sequence using only ground-truth box statistics.

        The failure-rate component is fixed at 0.  Motion complexity and
        attribute burden are re-weighted proportionally so that an extremely
        challenging sequence (fast motion + all attributes) can still score
        near 1.0.

        Args:
            ground_truths: ``(N, 4)`` float array of GT boxes ``(x, y, w, h)``.
            sequence_name: Optional sequence identifier stored in the result.

        Returns:
            :class:`SequenceDifficultyScore` with ``failure_rate=0.0``.
        """
        gt = np.asarray(ground_truths, dtype=np.float64)
        motion_score = _compute_motion_score(gt)
        attrs = self._detector.detect(gt, sequence_name)
        attr_score = _compute_attribute_score(attrs)

        remaining = self._mw + self._aw
        if remaining > 1e-9:
            score = motion_score * (self._mw / remaining) + attr_score * (self._aw / remaining)
        else:
            score = 0.0

        score = min(1.0, max(0.0, score))
        return SequenceDifficultyScore(
            sequence_name=sequence_name,
            failure_rate=0.0,
            motion_score=motion_score,
            attribute_score=attr_score,
            difficulty_score=score,
            difficulty_level=_level_from_score(score),
            dominant_challenge=_dominant_attribute(attrs),
            active_attributes=attrs.active_attributes,
            num_frames=len(gt),
        )

    # ------------------------------------------------------------------
    # Scoring — result mode
    # ------------------------------------------------------------------

    def score_from_result(
        self,
        seq_result: "SequenceResult",
    ) -> SequenceDifficultyScore:
        """Score a sequence using both tracker IoUs and GT-box statistics.

        All three components (failure rate, motion complexity, attribute burden)
        contribute to the composite score.

        Args:
            seq_result: A single :class:`~eovot.benchmark.engine.SequenceResult`
                from :class:`~eovot.benchmark.engine.BenchmarkEngine`.

        Returns:
            :class:`SequenceDifficultyScore`.
        """
        ious = np.asarray(seq_result.ious, dtype=np.float64)
        gt = (
            np.asarray(seq_result.ground_truths, dtype=np.float64)
            if seq_result.ground_truths is not None
            else np.empty((0, 4), dtype=np.float64)
        )

        failure_rate = float((ious < self._iou_thresh).mean()) if len(ious) else 0.0
        motion_score = _compute_motion_score(gt)
        attrs = self._detector.detect(gt, seq_result.sequence_name)
        attr_score = _compute_attribute_score(attrs)

        score = (
            self._fw * failure_rate
            + self._mw * motion_score
            + self._aw * attr_score
        )
        score = min(1.0, max(0.0, score))

        return SequenceDifficultyScore(
            sequence_name=seq_result.sequence_name,
            failure_rate=failure_rate,
            motion_score=motion_score,
            attribute_score=attr_score,
            difficulty_score=score,
            difficulty_level=_level_from_score(score),
            dominant_challenge=_dominant_attribute(attrs),
            active_attributes=attrs.active_attributes,
            num_frames=len(ious),
        )

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_benchmark(
        self,
        result: "BenchmarkResult",
    ) -> List[SequenceDifficultyScore]:
        """Score all sequences in a benchmark result.

        Args:
            result: Full benchmark result from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.

        Returns:
            List of :class:`SequenceDifficultyScore`, one per sequence,
            in the same order as ``result.sequence_results``.
        """
        return [self.score_from_result(sr) for sr in result.sequence_results]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def stratified_summary(
        self,
        scores: List[SequenceDifficultyScore],
        result: "BenchmarkResult",
    ) -> Dict[str, Dict]:
        """Aggregate tracker performance per difficulty level.

        Args:
            scores: Output of :meth:`analyze_benchmark`.
            result: The benchmark result the scores were derived from.

        Returns:
            Dict keyed by difficulty-level string (``"easy"``, ``"medium"``,
            ``"hard"``, ``"very_hard"``).  Each value has keys:
            ``"n_sequences"``, ``"mean_iou"``, ``"mean_failure_rate"``.
        """
        iou_map = {
            sr.sequence_name: float(sr.ious.mean()) if len(sr.ious) else 0.0
            for sr in result.sequence_results
        }

        groups: Dict[str, List[float]] = {lv.value: [] for lv in DifficultyLevel}
        fail_groups: Dict[str, List[float]] = {lv.value: [] for lv in DifficultyLevel}

        for s in scores:
            k = s.difficulty_level.value
            groups[k].append(iou_map.get(s.sequence_name, 0.0))
            fail_groups[k].append(s.failure_rate)

        out: Dict[str, Dict] = {}
        for lv in DifficultyLevel:
            k = lv.value
            ious_g = groups[k]
            fails_g = fail_groups[k]
            out[k] = {
                "n_sequences": len(ious_g),
                "mean_iou": round(float(np.mean(ious_g)), 4) if ious_g else 0.0,
                "mean_failure_rate": round(float(np.mean(fails_g)), 4) if fails_g else 0.0,
            }
        return out

    def to_stratified_leaderboard(
        self,
        scores: List[SequenceDifficultyScore],
        result: "BenchmarkResult",
    ) -> str:
        """Format a difficulty-stratified performance table as a Markdown string.

        Produces a publication-ready table that separates tracker performance
        on easy vs. hard sequences — a common requirement in tracking papers.

        Args:
            scores: Output of :meth:`analyze_benchmark`.
            result: The benchmark result the scores were derived from.

        Returns:
            Multi-line Markdown string.
        """
        summary = self.stratified_summary(scores, result)

        lines = [
            f"## Difficulty-Stratified Performance: "
            f"{result.tracker_name} on {result.dataset_name}",
            "",
            "| Difficulty | Sequences | Mean IoU | Failure Rate |",
            "|------------|----------:|---------:|-------------:|",
        ]
        for lv in DifficultyLevel:
            s = summary[lv.value]
            if s["n_sequences"] == 0:
                continue
            label = lv.value.capitalize().replace("_", " ")
            lines.append(
                f"| {label:<10s} "
                f"| {s['n_sequences']:>9d} "
                f"| {s['mean_iou']:>8.4f} "
                f"| {s['mean_failure_rate']:>12.4f} |"
            )

        lines.append("")
        total = sum(s["n_sequences"] for s in summary.values())
        all_ious = [
            float(sr.ious.mean())
            for sr in result.sequence_results
            if len(sr.ious)
        ]
        overall_iou = round(float(np.mean(all_ious)), 4) if all_ious else 0.0
        lines.append(
            f"*{total} sequences total — overall mIoU: {overall_iou}*"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------

def _level_from_score(score: float) -> DifficultyLevel:
    """Map a composite score to a :class:`DifficultyLevel`."""
    for threshold, level in _LEVEL_THRESHOLDS:
        if score < threshold:
            return level
    return DifficultyLevel.VERY_HARD


def _dominant_attribute(attrs: SequenceAttributes) -> str:
    """Return the highest-weight active attribute name, or ``'none'``."""
    active = attrs.active_attributes
    if not active:
        return "none"
    return max(active, key=lambda a: ATTRIBUTE_WEIGHTS.get(a, 0.0))


def _compute_motion_score(gt: np.ndarray) -> float:
    """Normalised motion complexity from GT boxes, ∈ [0, 1].

    Computes mean frame-to-frame centre displacement divided by the mean box
    diagonal, then maps with ``tanh`` so that the OTB "fast motion" threshold
    (20 % of diagonal) maps to approximately 0.5.

    Args:
        gt: ``(N, 4)`` float array in ``(x, y, w, h)`` format.

    Returns:
        Motion complexity in ``[0, 1]``.  Returns ``0.0`` for empty or
        single-frame sequences.
    """
    if len(gt) < 2:
        return 0.0

    gt = np.asarray(gt, dtype=np.float64)
    cx = gt[:, 0] + gt[:, 2] / 2.0
    cy = gt[:, 1] + gt[:, 3] / 2.0

    displacements = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    diags = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)
    mean_diag = float(diags.mean())

    if mean_diag < 1e-6:
        return 0.0

    mean_norm = float(displacements.mean()) / mean_diag
    # tanh(x / 0.20) maps OTB fast-motion threshold (0.20) to tanh(1) ≈ 0.76
    return float(min(1.0, math.tanh(mean_norm / 0.20)))


def _compute_attribute_score(attrs: SequenceAttributes) -> float:
    """Weighted attribute burden normalised to [0, 1].

    Args:
        attrs: :class:`~eovot.metrics.attributes.SequenceAttributes`.

    Returns:
        Score in ``[0, 1]``; 0 = no attributes present, 1 = all active.
    """
    raw = sum(ATTRIBUTE_WEIGHTS.get(a, 0.0) for a in attrs.active_attributes)
    if _MAX_ATTRIBUTE_SCORE < 1e-9:
        return 0.0
    return min(1.0, raw / _MAX_ATTRIBUTE_SCORE)
