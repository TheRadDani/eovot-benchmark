"""Sequence difficulty profiler for per-dataset characterisation.

Addresses project issue #63.  Understanding *why* a dataset is hard — fast
motion, large-scale change, small targets, aspect-ratio deformation — is
essential for:

* **Difficulty-stratified benchmarking**: compare trackers fairly by grouping
  sequences with similar challenge profiles rather than averaging over mixed
  difficulty.
* **Ablation attribution**: explain *which* sequences drive a tracker's IoU
  drop rather than lumping all sequences together.
* **Dataset documentation**: give downstream users a one-line characterisation
  of a dataset's composition (e.g. "60 % of sequences have fast-motion frames,
  median scale-change ratio 3.1×").

All analysis is **ground-truth only** — no tracker predictions are required,
making the profiler usable at dataset-design time before any tracker is run.

Typical usage::

    import numpy as np
    from eovot.analysis.difficulty import SequenceDifficultyProfiler

    gt = {
        "car-1": np.array([[100, 80, 60, 40], [130, 82, 62, 41], ...]),
        "person-3": ...,
    }
    profiler = SequenceDifficultyProfiler(frame_size=(480, 640))
    results = profiler.analyze_dataset(gt)
    print(profiler.to_markdown_table(results))
    print(profiler.dataset_summary(results))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SequenceDifficulty:
    """Difficulty characterisation of a single tracking sequence.

    All motion metrics are normalised by the frame diagonal so results are
    resolution-agnostic and comparable across datasets with different frame
    sizes.  Scale metrics use the ratio of bounding-box areas.
    """

    sequence_name: str

    # Motion
    motion_magnitude: float
    """Mean frame-to-frame centre displacement normalised by frame diagonal.

    Values above ~0.05 correspond to fast-moving targets relative to the
    frame size.  Most challenging for correlation-filter trackers on edge
    hardware because the search region must be large.
    """

    fast_motion_fraction: float
    """Fraction of inter-frame transitions where normalised displacement
    exceeds the ``fast_motion_threshold`` (default 0.05).  A sequence is
    considered fast-motion-dominated above 0.3 (30 % of transitions).
    """

    # Scale change
    scale_variability: float
    """Coefficient of variation (std / mean) of bounding-box area across all
    frames.  A high CV (> 0.3) indicates significant target scale change,
    requiring scale-adaptive trackers to maintain accuracy.
    """

    scale_change_ratio: float
    """Ratio of maximum to minimum bounding-box area across all frames.
    A ratio above 4× indicates severe scale change; ratios > 10× are extreme
    and rarely handled well by classical trackers.
    """

    # Aspect ratio
    aspect_ratio_variability: float
    """Standard deviation of per-frame aspect ratio (width / height).  High
    variability (> 0.2) indicates target deformation or rotation — a challenge
    that standard fixed-size templates cannot model.
    """

    # Target size relative to frame
    mean_target_size: float
    """Mean bounding-box area as a fraction of the total frame area.  Small
    values (< 0.01) indicate small-target sequences where the tracker must
    operate at low pixel resolution.
    """

    small_target_fraction: float
    """Fraction of frames where the target occupies less than
    ``small_target_threshold`` of the frame area (default 0.01 — 1 %).
    Small targets have low signal-to-noise ratio and are particularly
    problematic on edge hardware with quantised or downscaled inputs.
    """

    # Composite
    difficulty_score: float
    """Composite difficulty score in [0, 1].  Higher = harder.

    Computed as a weighted combination of motion, scale, small-target, and
    aspect-ratio sub-scores (see :meth:`SequenceDifficultyProfiler._compute_difficulty_score`).
    The weighting reflects typical challenge severity for edge tracker deployment.
    """

    difficulty_tier: str
    """Categorical tier derived from :attr:`difficulty_score`:

    * ``'easy'``   — score < 0.33
    * ``'medium'`` — 0.33 ≤ score < 0.67
    * ``'hard'``   — score ≥ 0.67
    """

    def __str__(self) -> str:
        return (
            f"SequenceDifficulty[{self.sequence_name}] "
            f"score={self.difficulty_score:.3f} ({self.difficulty_tier})  "
            f"motion={self.motion_magnitude:.4f}  "
            f"fast_motion={self.fast_motion_fraction:.2f}  "
            f"scale_CV={self.scale_variability:.3f}  "
            f"scale_ratio={self.scale_change_ratio:.2f}×  "
            f"small_target={self.small_target_fraction:.2f}"
        )


class SequenceDifficultyProfiler:
    """Characterise tracking sequences from ground-truth bounding boxes.

    Args:
        frame_size: Common ``(height, width)`` of all frames.  If ``None``,
            the profiler estimates frame dimensions from the GT box positions
            of each sequence individually (a conservative upper bound).
        fast_motion_threshold: Normalised displacement per frame above which
            a transition is classified as "fast motion".  Default ``0.05``
            (target centre moves > 5 % of the frame diagonal per frame).
        small_target_threshold: Target area as a fraction of frame area below
            which the target is considered "small".  Default ``0.01`` (< 1 %
            of frame pixels).
    """

    def __init__(
        self,
        frame_size: Optional[Tuple[int, int]] = None,
        fast_motion_threshold: float = 0.05,
        small_target_threshold: float = 0.01,
    ) -> None:
        self.frame_size = frame_size
        self.fast_motion_threshold = fast_motion_threshold
        self.small_target_threshold = small_target_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_sequence(
        self,
        ground_truth: np.ndarray,
        sequence_name: str = "",
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> SequenceDifficulty:
        """Compute difficulty metrics for a single sequence.

        Args:
            ground_truth: ``(N, 4)`` array of GT bounding boxes ``(x, y, w, h)``
                in pixel coordinates.  Requires at least 2 frames.
            sequence_name: Human-readable identifier stored in the result.
            frame_size: ``(height, width)`` override for this sequence only.
                Falls back to the profiler-level ``frame_size``, then to an
                estimate from the GT positions.

        Returns:
            :class:`SequenceDifficulty` with all metrics populated.
        """
        gt = np.asarray(ground_truth, dtype=np.float64)
        if len(gt) < 2:
            return self._empty_result(sequence_name)

        fh, fw = frame_size or self.frame_size or self._estimate_frame_size(gt)
        diagonal = np.sqrt(float(fh) ** 2 + float(fw) ** 2)
        frame_area = float(fh) * float(fw)

        # Centre positions
        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0

        # Frame-to-frame displacement — shape (N-1,)
        displacements = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        norm_disp = displacements / (diagonal + 1e-8)
        motion_magnitude = float(norm_disp.mean())
        fast_motion_fraction = float((norm_disp > self.fast_motion_threshold).mean())

        # Bounding-box areas
        areas = gt[:, 2] * gt[:, 3]
        areas = np.maximum(areas, 1.0)
        area_mean = float(areas.mean())
        scale_variability = float(areas.std() / (area_mean + 1e-8))
        scale_change_ratio = float(areas.max() / (areas.min() + 1e-8))

        # Aspect ratios
        ar = gt[:, 2] / (gt[:, 3] + 1e-8)
        aspect_ratio_variability = float(ar.std())

        # Target size relative to frame
        rel_areas = areas / (frame_area + 1e-8)
        mean_target_size = float(rel_areas.mean())
        small_target_fraction = float((rel_areas < self.small_target_threshold).mean())

        difficulty_score = self._compute_difficulty_score(
            motion_magnitude=motion_magnitude,
            fast_motion_fraction=fast_motion_fraction,
            scale_variability=scale_variability,
            scale_change_ratio=scale_change_ratio,
            small_target_fraction=small_target_fraction,
            aspect_ratio_variability=aspect_ratio_variability,
        )
        difficulty_tier = _classify_tier(difficulty_score)

        return SequenceDifficulty(
            sequence_name=sequence_name,
            motion_magnitude=motion_magnitude,
            fast_motion_fraction=fast_motion_fraction,
            scale_variability=scale_variability,
            scale_change_ratio=scale_change_ratio,
            aspect_ratio_variability=aspect_ratio_variability,
            mean_target_size=mean_target_size,
            small_target_fraction=small_target_fraction,
            difficulty_score=difficulty_score,
            difficulty_tier=difficulty_tier,
        )

    def analyze_dataset(
        self,
        ground_truths: Dict[str, np.ndarray],
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, SequenceDifficulty]:
        """Profile every sequence in a dataset.

        Args:
            ground_truths: Mapping ``{sequence_name: (N, 4) GT array}``.
            frame_size: Common frame size override.  Falls back to the
                profiler-level ``frame_size`` then to per-sequence estimation.

        Returns:
            ``{sequence_name: SequenceDifficulty}`` dict ordered by
            descending difficulty score.
        """
        results = {
            name: self.analyze_sequence(gt, sequence_name=name, frame_size=frame_size)
            for name, gt in ground_truths.items()
        }
        return dict(
            sorted(results.items(), key=lambda kv: kv[1].difficulty_score, reverse=True)
        )

    def dataset_summary(
        self, results: Dict[str, SequenceDifficulty]
    ) -> Dict:
        """Aggregate dataset-level difficulty statistics.

        Args:
            results: Output of :meth:`analyze_dataset`.

        Returns:
            Dict with aggregate scalars and per-tier sequence counts, suitable
            for embedding in benchmark reports.
        """
        if not results:
            return {}
        vals = list(results.values())
        tier_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        for v in vals:
            tier_counts[v.difficulty_tier] = tier_counts.get(v.difficulty_tier, 0) + 1
        return {
            "num_sequences": len(vals),
            "mean_difficulty_score": round(float(np.mean([v.difficulty_score for v in vals])), 4),
            "mean_motion_magnitude": round(float(np.mean([v.motion_magnitude for v in vals])), 5),
            "mean_fast_motion_fraction": round(float(np.mean([v.fast_motion_fraction for v in vals])), 4),
            "mean_scale_variability": round(float(np.mean([v.scale_variability for v in vals])), 4),
            "mean_scale_change_ratio": round(float(np.mean([v.scale_change_ratio for v in vals])), 3),
            "mean_small_target_fraction": round(float(np.mean([v.small_target_fraction for v in vals])), 4),
            "tier_counts": tier_counts,
        }

    def to_markdown_table(
        self,
        results: Dict[str, SequenceDifficulty],
    ) -> str:
        """Render difficulty results as a Markdown table, hardest-first.

        Args:
            results: Output of :meth:`analyze_dataset` or :meth:`analyze_sequence`
                wrapped in a dict.

        Returns:
            Multi-line Markdown table string ready for inclusion in reports.
        """
        rows = sorted(results.values(), key=lambda r: r.difficulty_score, reverse=True)
        lines = [
            "| Sequence | Score | Tier | Motion | Fast% | Scale CV | Scale Ratio | SmallTgt% |",
            "|----------|------:|------|-------:|------:|---------:|------------:|----------:|",
        ]
        for r in rows:
            lines.append(
                f"| {r.sequence_name} "
                f"| {r.difficulty_score:.3f} "
                f"| {r.difficulty_tier} "
                f"| {r.motion_magnitude:.4f} "
                f"| {r.fast_motion_fraction * 100:.1f} "
                f"| {r.scale_variability:.3f} "
                f"| {r.scale_change_ratio:.2f}× "
                f"| {r.small_target_fraction * 100:.1f} |"
            )
        return "\n".join(lines)

    def filter_by_tier(
        self,
        results: Dict[str, SequenceDifficulty],
        tier: str,
    ) -> Dict[str, SequenceDifficulty]:
        """Return only sequences belonging to a given difficulty tier.

        Args:
            results: Output of :meth:`analyze_dataset`.
            tier:    One of ``'easy'``, ``'medium'``, or ``'hard'``.

        Returns:
            Filtered subset of *results* with the matching tier.

        Raises:
            ValueError: If *tier* is not a recognised tier name.
        """
        valid = {"easy", "medium", "hard"}
        if tier not in valid:
            raise ValueError(f"Invalid tier {tier!r}. Must be one of {sorted(valid)}.")
        return {name: r for name, r in results.items() if r.difficulty_tier == tier}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_difficulty_score(
        self,
        *,
        motion_magnitude: float,
        fast_motion_fraction: float,
        scale_variability: float,
        scale_change_ratio: float,
        small_target_fraction: float,
        aspect_ratio_variability: float,
    ) -> float:
        """Compute a composite difficulty score in [0, 1].

        Weights reflect the relative challenge severity for edge-deployed
        correlation-filter trackers:

        * Motion (0.35): Fast, large-displacement motion is the primary
          failure mode for fixed-search-region trackers on bandwidth-limited
          edge devices.
        * Scale change (0.30): Scale change requires adaptive mechanisms
          absent in classical trackers (KCF, MOSSE, CSRT).
        * Small targets (0.20): Low-resolution patches have high noise, and
          quantisation on edge hardware makes this worse.
        * Aspect-ratio deformation (0.15): Non-rigid targets or rotating
          objects that change shape across frames.

        Each sub-score is normalised to [0, 1] via a soft ceiling (``min(x/cap, 1)``).
        """
        motion_score = min(motion_magnitude / 0.15, 1.0)
        fast_score = fast_motion_fraction

        scale_cv_score = min(scale_variability / 0.5, 1.0)
        scale_ratio_score = min((scale_change_ratio - 1.0) / 9.0, 1.0)
        scale_score = 0.5 * scale_cv_score + 0.5 * scale_ratio_score

        small_score = small_target_fraction
        ar_score = min(aspect_ratio_variability / 0.5, 1.0)

        score = (
            0.35 * (0.5 * motion_score + 0.5 * fast_score)
            + 0.30 * scale_score
            + 0.20 * small_score
            + 0.15 * ar_score
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _estimate_frame_size(gt: np.ndarray) -> Tuple[int, int]:
        """Estimate frame dimensions from GT box positions.

        Uses a 20 % margin above the max observed x2/y2 as a conservative
        upper bound.  The estimate is used only when no ``frame_size`` is
        supplied; passing an explicit size is always preferred.
        """
        x2_max = float((gt[:, 0] + gt[:, 2]).max())
        y2_max = float((gt[:, 1] + gt[:, 3]).max())
        fw = max(int(x2_max * 1.2), 64)
        fh = max(int(y2_max * 1.2), 64)
        return fh, fw

    def _empty_result(self, sequence_name: str) -> SequenceDifficulty:
        return SequenceDifficulty(
            sequence_name=sequence_name,
            motion_magnitude=0.0,
            fast_motion_fraction=0.0,
            scale_variability=0.0,
            scale_change_ratio=1.0,
            aspect_ratio_variability=0.0,
            mean_target_size=0.0,
            small_target_fraction=0.0,
            difficulty_score=0.0,
            difficulty_tier="easy",
        )


def _classify_tier(score: float) -> str:
    """Map a difficulty score to a categorical tier."""
    if score < 0.33:
        return "easy"
    if score < 0.67:
        return "medium"
    return "hard"
