"""Sequence difficulty profiling for EOVOT.

Assigns a difficulty score to each tracking sequence based on observable
ground-truth statistics, without requiring tracker predictions.  The score
enables difficulty-stratified analysis — e.g. "how does tracker X perform
on hard vs. easy sequences?" — which is a key tool for understanding where
a tracker fails and for fair comparison across datasets with varying content.

Difficulty Components
~~~~~~~~~~~~~~~~~~~~~

**Motion magnitude** — normalised mean displacement of the bounding-box
centre between consecutive frames, divided by the geometric mean of the box
dimensions so that fast-moving small targets score higher than slow-moving
large ones.  Range: ``[0, ∞)`` (practically 0–3 on typical sequences).

**Scale variation** — standard deviation of the ratio between consecutive
frame bounding-box areas, capped at 1.0.  High variance means the target
is rapidly growing or shrinking (zoom changes, approach/recede).

**Aspect ratio drift** — standard deviation of the aspect ratio
(``width / height``) across the sequence, capped at 1.0.  Captures non-rigid
deformation or rotation-induced shape changes.

**Out-of-view frames** — fraction of frames where the bounding box has zero
or negative area (missing annotations).  Even a small fraction significantly
raises difficulty because the tracker receives no feedback.

**Composite score** — a weighted sum of the four components normalized to
``[0, 1]`` via a sigmoid-like clipping:

    score = clip(α·motion + β·scale + γ·aspect + δ·oov, 0, 1)

Default weights: ``α=0.4, β=0.25, γ=0.15, δ=0.20``.

Typical usage::

    from eovot.metrics.difficulty import SequenceDifficultyProfiler
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=10)
    profiler = SequenceDifficultyProfiler()

    for seq in dataset:
        result = profiler.profile(seq.ground_truth, seq.name)
        print(result)

    # Analyse a full benchmark run
    all_gt = {seq.name: seq.ground_truth for seq in dataset}
    report = profiler.profile_dataset(all_gt)
    print(report["summary"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DifficultyResult:
    """Per-sequence difficulty breakdown."""

    sequence_name: str
    """Human-readable name of the sequence."""

    motion_score: float
    """Normalised mean frame-to-frame displacement (0 = static, higher = faster)."""

    scale_variation_score: float
    """Normalised standard deviation of consecutive area ratios (0 = constant size)."""

    aspect_drift_score: float
    """Normalised standard deviation of aspect ratio across the sequence."""

    out_of_view_fraction: float
    """Fraction of frames with missing/zero-area bounding boxes."""

    composite_score: float
    """Weighted composite difficulty in ``[0, 1]``.  Higher = harder."""

    num_frames: int
    """Total number of frames in the sequence."""

    def bucket(self) -> str:
        """Return a human-readable difficulty tier based on composite score.

        Returns:
            ``"easy"`` (< 0.25), ``"medium"`` (0.25–0.60), or ``"hard"`` (> 0.60).
        """
        if self.composite_score < 0.25:
            return "easy"
        if self.composite_score < 0.60:
            return "medium"
        return "hard"

    def __str__(self) -> str:
        return (
            f"DifficultyResult[{self.sequence_name}] "
            f"composite={self.composite_score:.3f} ({self.bucket()})  "
            f"motion={self.motion_score:.3f}  scale_var={self.scale_variation_score:.3f}  "
            f"aspect_drift={self.aspect_drift_score:.3f}  "
            f"oov={self.out_of_view_fraction:.3f}  "
            f"frames={self.num_frames}"
        )


class SequenceDifficultyProfiler:
    """Score tracking sequences by ground-truth complexity.

    All four components are computed from the ground-truth box sequence alone,
    so the profiler can be used before any tracker is run.

    Args:
        weight_motion:      Weight for the motion component.   Default: ``0.40``.
        weight_scale:       Weight for the scale variation.    Default: ``0.25``.
        weight_aspect:      Weight for the aspect ratio drift. Default: ``0.15``.
        weight_oov:         Weight for out-of-view fraction.   Default: ``0.20``.
        motion_clip:        Maximum motion score before clipping (normalisation
            ceiling).  Default: ``3.0`` (covers most realistic sequences).
        scale_clip:         Maximum scale-variation score before clipping.
            Default: ``1.0``.
        aspect_clip:        Maximum aspect-drift score before clipping.
            Default: ``1.0``.

    Raises:
        ValueError: If weights do not sum to a positive value.
    """

    def __init__(
        self,
        weight_motion: float = 0.40,
        weight_scale: float = 0.25,
        weight_aspect: float = 0.15,
        weight_oov: float = 0.20,
        motion_clip: float = 3.0,
        scale_clip: float = 1.0,
        aspect_clip: float = 1.0,
    ) -> None:
        total = weight_motion + weight_scale + weight_aspect + weight_oov
        if total <= 0:
            raise ValueError("Component weights must sum to a positive value.")
        self.weight_motion = weight_motion
        self.weight_scale = weight_scale
        self.weight_aspect = weight_aspect
        self.weight_oov = weight_oov
        self.motion_clip = motion_clip
        self.scale_clip = scale_clip
        self.aspect_clip = aspect_clip

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        ground_truth: np.ndarray,
        sequence_name: str = "",
    ) -> DifficultyResult:
        """Compute the difficulty profile for a single sequence.

        Args:
            ground_truth: ``(N, 4)`` array of bounding boxes ``(x, y, w, h)``.
            sequence_name: Optional label stored in the result.

        Returns:
            :class:`DifficultyResult` with all component scores and the
            composite difficulty score.

        Raises:
            ValueError: If *ground_truth* has fewer than 2 frames.
        """
        gt = np.asarray(ground_truth, dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(f"ground_truth must be (N, 4), got {gt.shape}")
        if len(gt) < 2:
            raise ValueError("Sequence must have at least 2 frames to profile.")

        motion = self._motion_score(gt)
        scale = self._scale_variation_score(gt)
        aspect = self._aspect_drift_score(gt)
        oov = self._out_of_view_fraction(gt)

        composite = self._composite(motion, scale, aspect, oov)

        return DifficultyResult(
            sequence_name=sequence_name,
            motion_score=round(motion, 6),
            scale_variation_score=round(scale, 6),
            aspect_drift_score=round(aspect, 6),
            out_of_view_fraction=round(oov, 6),
            composite_score=round(composite, 6),
            num_frames=len(gt),
        )

    def profile_dataset(
        self,
        sequences: Dict[str, np.ndarray],
    ) -> Dict:
        """Profile every sequence in a dataset and return aggregated statistics.

        Args:
            sequences: Mapping ``{sequence_name: ground_truth_array}``.

        Returns:
            Dict with three keys:

            * ``"per_sequence"`` — ``{name: DifficultyResult}``
            * ``"summary"`` — dataset-level aggregate stats
            * ``"buckets"`` — ``{"easy": [...names], "medium": [...], "hard": [...]}``
        """
        per_seq: Dict[str, DifficultyResult] = {}
        for name, gt in sequences.items():
            try:
                per_seq[name] = self.profile(gt, sequence_name=name)
            except ValueError:
                continue  # skip degenerate sequences

        if not per_seq:
            return {"per_sequence": {}, "summary": {}, "buckets": {}}

        scores = [r.composite_score for r in per_seq.values()]
        buckets: Dict[str, List[str]] = {"easy": [], "medium": [], "hard": []}
        for name, r in per_seq.items():
            buckets[r.bucket()].append(name)

        summary = {
            "num_sequences": len(per_seq),
            "mean_composite": round(float(np.mean(scores)), 4),
            "std_composite": round(float(np.std(scores)), 4),
            "min_composite": round(float(np.min(scores)), 4),
            "max_composite": round(float(np.max(scores)), 4),
            "easy_count": len(buckets["easy"]),
            "medium_count": len(buckets["medium"]),
            "hard_count": len(buckets["hard"]),
        }

        return {"per_sequence": per_seq, "summary": summary, "buckets": buckets}

    def rank_by_difficulty(
        self,
        sequences: Dict[str, np.ndarray],
        descending: bool = True,
    ) -> List[DifficultyResult]:
        """Return sequences sorted by composite difficulty score.

        Args:
            sequences:  Mapping ``{name: ground_truth_array}``.
            descending: ``True`` (default) → hardest first; ``False`` → easiest first.

        Returns:
            List of :class:`DifficultyResult` sorted by composite score.
        """
        report = self.profile_dataset(sequences)
        results = list(report["per_sequence"].values())
        results.sort(key=lambda r: r.composite_score, reverse=descending)
        return results

    def to_markdown_table(self, results: List[DifficultyResult]) -> str:
        """Format difficulty results as a Markdown table.

        Args:
            results: List of :class:`DifficultyResult` (from
                :meth:`rank_by_difficulty` or :meth:`profile_dataset`).

        Returns:
            Multi-line Markdown table string.
        """
        lines = [
            "| Sequence | Tier | Score | Motion | Scale Var | Aspect Drift | OOV | Frames |",
            "|----------|:----:|------:|-------:|----------:|-------------:|----:|-------:|",
        ]
        for r in results:
            lines.append(
                f"| {r.sequence_name} | {r.bucket()} | {r.composite_score:.3f} "
                f"| {r.motion_score:.3f} | {r.scale_variation_score:.3f} "
                f"| {r.aspect_drift_score:.3f} | {r.out_of_view_fraction:.3f} "
                f"| {r.num_frames} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------

    def _motion_score(self, gt: np.ndarray) -> float:
        """Normalised mean frame-to-frame centre displacement."""
        centres = gt[:, :2] + gt[:, 2:] / 2.0          # (N, 2)
        valid_w = gt[:, 2] > 0
        valid_h = gt[:, 3] > 0
        valid = valid_w & valid_h

        if valid.sum() < 2:
            return 0.0

        # Compute displacements only between consecutive valid frames
        displacements: List[float] = []
        sizes: List[float] = []
        prev_idx: Optional[int] = None
        for i in range(len(gt)):
            if not valid[i]:
                prev_idx = None
                continue
            if prev_idx is not None:
                dc = centres[i] - centres[prev_idx]
                disp = float(np.sqrt((dc ** 2).sum()))
                size = float(np.sqrt(gt[prev_idx, 2] * gt[prev_idx, 3]))
                if size > 0:
                    displacements.append(disp / size)
                    sizes.append(size)
            prev_idx = i

        if not displacements:
            return 0.0

        raw = float(np.mean(displacements))
        return float(np.clip(raw / self.motion_clip, 0.0, 1.0))

    def _scale_variation_score(self, gt: np.ndarray) -> float:
        """Normalised std of consecutive area ratios."""
        areas = gt[:, 2] * gt[:, 3]
        valid_mask = areas > 0
        valid_areas = areas[valid_mask]

        if len(valid_areas) < 2:
            return 0.0

        ratios = valid_areas[1:] / np.maximum(valid_areas[:-1], 1e-6)
        raw = float(np.std(ratios))
        return float(np.clip(raw / self.scale_clip, 0.0, 1.0))

    def _aspect_drift_score(self, gt: np.ndarray) -> float:
        """Normalised std of aspect ratio (w/h) across the sequence."""
        valid_h = gt[:, 3] > 0
        valid_w = gt[:, 2] > 0
        valid = valid_w & valid_h

        if valid.sum() < 2:
            return 0.0

        aspects = gt[valid, 2] / np.maximum(gt[valid, 3], 1e-6)
        raw = float(np.std(aspects))
        return float(np.clip(raw / self.aspect_clip, 0.0, 1.0))

    def _out_of_view_fraction(self, gt: np.ndarray) -> float:
        """Fraction of frames with zero or negative bounding-box area."""
        areas = gt[:, 2] * gt[:, 3]
        oov = int((areas <= 0).sum())
        return float(oov / len(gt))

    def _composite(
        self,
        motion: float,
        scale: float,
        aspect: float,
        oov: float,
    ) -> float:
        """Compute the weighted composite difficulty score in ``[0, 1]``."""
        raw = (
            self.weight_motion * motion
            + self.weight_scale * scale
            + self.weight_aspect * aspect
            + self.weight_oov * oov
        )
        return float(np.clip(raw, 0.0, 1.0))
