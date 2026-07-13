"""Sequence difficulty analysis for EOVOT benchmark results.

Research-grade tracking evaluation requires more than dataset-level mean
metrics.  The *same* tracker may achieve 0.9 mIoU on slow sequences and
0.3 mIoU on fast-moving or scale-changing ones.  Reporting a single mean
hides this variation and makes it impossible to understand where a tracker
actually fails.

This module provides:

SequenceAttributes
    A dataclass holding per-sequence characterisation scalars derived from
    the ground-truth bounding box array: mean displacement, scale-change
    ratio, aspect-ratio instability, and an overall difficulty score.

DifficultyTier
    Enum with three values: ``EASY``, ``MEDIUM``, ``HARD``, determined from
    the per-sequence difficulty score relative to the dataset distribution.

SequenceAnalyzer
    Orchestrates attribute computation, tier assignment, and per-tier
    performance breakdown across a full benchmark run.

Design philosophy
~~~~~~~~~~~~~~~~~
All attributes are computed from the ground-truth array only — no tracker
output is required for characterisation.  This ensures the difficulty
classification is independent of any tracker and can be reused across runs.

Typical usage::

    import numpy as np
    from eovot.analysis.sequence_analyzer import SequenceAnalyzer
    from eovot.benchmark.engine import BenchmarkResult

    # Collect ground-truth arrays and IoU arrays from a BenchmarkResult
    analyzer = SequenceAnalyzer()

    gt_arrays = {r.sequence_name: r.ground_truths for r in result.sequence_results}
    iou_arrays = {r.sequence_name: r.ious for r in result.sequence_results}

    report = analyzer.analyze(gt_arrays, iou_arrays, tracker_name=result.tracker_name)
    print(report["tier_summary"])
    print(report["markdown_table"])

Attribute definitions
~~~~~~~~~~~~~~~~~~~~~
mean_speed_px
    Mean Euclidean displacement of the GT box centre between consecutive
    frames (pixels/frame).  High values indicate fast-moving targets.

scale_change_ratio
    Ratio ``max_area / min_area`` across all frames.  A ratio close to 1
    means constant-size; large values (> 4) mean significant size change.

aspect_ratio_std
    Standard deviation of ``w / h`` across all frames.  High values
    indicate the target is deforming or rotating.

difficulty_score
    Weighted composite::

        difficulty = w_spd × norm(speed) + w_sc × norm(log(scale_ratio))
                   + w_ar × norm(ar_std)

    where ``norm(·)`` maps each attribute to ``[0, 1]`` relative to the
    full sequence set, and the weights default to ``(0.5, 0.35, 0.15)``.

Tier thresholds
~~~~~~~~~~~~~~~
The three tiers are defined by percentile cutoffs on the difficulty score
across the full sequence set:

* **EASY**   — difficulty < 33rd percentile
* **MEDIUM** — 33rd ≤ difficulty < 67th percentile
* **HARD**   — difficulty ≥ 67th percentile
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class DifficultyTier(Enum):
    """Difficulty tier assigned to a tracking sequence."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class SequenceAttributes:
    """Per-sequence characterisation derived from ground-truth bounding boxes.

    Attributes:
        sequence_name: Identifier from the dataset.
        num_frames: Total number of frames in the sequence.
        mean_speed_px: Mean centre displacement per frame (pixels/frame).
        max_speed_px: Maximum per-frame displacement (pixels/frame).
        scale_change_ratio: ``max_area / min_area`` across the sequence.
        aspect_ratio_std: Std of ``w/h`` across frames.
        difficulty_score: Composite difficulty in ``[0, 1]`` (higher = harder).
            Populated by :meth:`SequenceAnalyzer.analyze`; ``None`` until then.
        tier: Difficulty tier; ``None`` until assigned by the analyzer.
    """

    sequence_name: str
    num_frames: int
    mean_speed_px: float
    max_speed_px: float
    scale_change_ratio: float
    aspect_ratio_std: float
    difficulty_score: Optional[float] = field(default=None)
    tier: Optional[DifficultyTier] = field(default=None)

    def __str__(self) -> str:
        tier_str = self.tier.value if self.tier else "?"
        return (
            f"SequenceAttributes({self.sequence_name}  "
            f"frames={self.num_frames}  "
            f"speed={self.mean_speed_px:.1f}px/f  "
            f"scale_ratio={self.scale_change_ratio:.2f}  "
            f"ar_std={self.aspect_ratio_std:.3f}  "
            f"difficulty={self.difficulty_score:.3f}  "
            f"tier={tier_str})"
        )


@dataclass
class TierPerformance:
    """Per-difficulty-tier aggregated performance for one tracker.

    Attributes:
        tier: The difficulty tier.
        num_sequences: Number of sequences in this tier.
        mean_iou: Mean IoU across all frames in the tier.
        mean_success_rate: Fraction of frames with IoU ≥ 0.5.
        mean_survival_rate: Fraction of frames with IoU ≥ 0.1.
        sequence_names: Names of sequences in this tier.
    """

    tier: DifficultyTier
    num_sequences: int
    mean_iou: float
    mean_success_rate: float
    mean_survival_rate: float
    sequence_names: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"TierPerformance({self.tier.value}  "
            f"n={self.num_sequences}  "
            f"mIoU={self.mean_iou:.4f}  "
            f"success={self.mean_success_rate:.4f}  "
            f"survival={self.mean_survival_rate:.4f})"
        )


# ---------------------------------------------------------------------------
# Attribute computation
# ---------------------------------------------------------------------------

def _compute_attributes(
    sequence_name: str,
    gt: np.ndarray,
) -> SequenceAttributes:
    """Compute characterisation attributes from a ground-truth array.

    Args:
        sequence_name: Sequence identifier.
        gt: Ground-truth bounding boxes, shape ``(N, 4)`` in ``(x, y, w, h)``
            format.  Must have at least 2 rows.

    Returns:
        :class:`SequenceAttributes` with ``difficulty_score`` and ``tier``
        set to ``None`` (assigned later by the analyzer).
    """
    gt = np.asarray(gt, dtype=np.float64)
    N = len(gt)

    # --- Centre-point displacements ---
    centres = gt[:, :2] + gt[:, 2:] / 2.0  # shape (N, 2)
    if N >= 2:
        diffs = np.diff(centres, axis=0)  # (N-1, 2)
        displacements = np.sqrt(np.sum(diffs ** 2, axis=1))  # (N-1,)
        mean_speed = float(displacements.mean())
        max_speed = float(displacements.max())
    else:
        mean_speed = 0.0
        max_speed = 0.0

    # --- Scale change ratio ---
    areas = gt[:, 2] * gt[:, 3]  # w × h per frame
    valid_areas = areas[areas > 0]
    if len(valid_areas) >= 2:
        scale_ratio = float(valid_areas.max() / valid_areas.min())
    else:
        scale_ratio = 1.0

    # --- Aspect-ratio instability ---
    widths = gt[:, 2]
    heights = gt[:, 3]
    valid_mask = (widths > 0) & (heights > 0)
    if valid_mask.sum() >= 2:
        ar = widths[valid_mask] / heights[valid_mask]
        ar_std = float(ar.std())
    else:
        ar_std = 0.0

    return SequenceAttributes(
        sequence_name=sequence_name,
        num_frames=N,
        mean_speed_px=mean_speed,
        max_speed_px=max_speed,
        scale_change_ratio=scale_ratio,
        aspect_ratio_std=ar_std,
    )


def _normalize_array(values: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to ``[0, 1]``.

    Returns a zero array if all values are equal (zero variance).
    """
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class SequenceAnalyzer:
    """Compute difficulty attributes and per-tier performance for a benchmark run.

    The analyzer works in two stages:

    1. **Attribute computation** — each sequence is characterised by
       ``mean_speed_px``, ``scale_change_ratio``, and ``aspect_ratio_std``
       computed from its ground-truth boxes.
    2. **Tier assignment** — attributes are combined into a ``difficulty_score``
       relative to the full dataset, then sequences are assigned to EASY /
       MEDIUM / HARD tiers by percentile thresholds.

    Per-tier performance tables break down tracker accuracy by tier, making it
    possible to identify whether a tracker fails specifically on fast-motion
    sequences, scale-change sequences, or all categories equally.

    Args:
        speed_weight:    Contribution of motion speed to the difficulty score.
                         Default: 0.50.
        scale_weight:    Contribution of scale change.  Default: 0.35.
        ar_weight:       Contribution of aspect-ratio instability. Default: 0.15.
        easy_percentile: Percentile boundary between EASY and MEDIUM.
                         Default: 33.
        hard_percentile: Percentile boundary between MEDIUM and HARD.
                         Default: 67.

    Example::

        analyzer = SequenceAnalyzer()
        gt_arrays = {"seq1": gt1, "seq2": gt2}
        iou_arrays = {"seq1": iou1, "seq2": iou2}
        report = analyzer.analyze(gt_arrays, iou_arrays, tracker_name="KCF")
        print(report["markdown_table"])
    """

    def __init__(
        self,
        speed_weight: float = 0.50,
        scale_weight: float = 0.35,
        ar_weight: float = 0.15,
        easy_percentile: float = 33.0,
        hard_percentile: float = 67.0,
    ) -> None:
        if abs(speed_weight + scale_weight + ar_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Difficulty weights must sum to 1.0; "
                f"got {speed_weight + scale_weight + ar_weight:.4f}."
            )
        self.speed_weight = speed_weight
        self.scale_weight = scale_weight
        self.ar_weight = ar_weight
        self.easy_percentile = easy_percentile
        self.hard_percentile = hard_percentile

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        gt_arrays: Dict[str, np.ndarray],
        iou_arrays: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> Dict:
        """Run full difficulty analysis and per-tier breakdown.

        Args:
            gt_arrays:   ``{sequence_name: ground_truth_array}`` where each
                array has shape ``(N, 4)`` in ``(x, y, w, h)`` format.
            iou_arrays:  ``{sequence_name: iou_array}`` where each array has
                shape ``(N,)`` and values in ``[0, 1]``.  Keys must be a
                superset of ``gt_arrays.keys()``.
            tracker_name: Human-readable label for the tracker (used in output).

        Returns:
            Dict with the following keys:

            * ``"attributes"``    — ``{seq_name: SequenceAttributes}``
            * ``"tier_performance"`` — ``{DifficultyTier: TierPerformance}``
            * ``"tier_summary"``  — summary string printed to console
            * ``"markdown_table"`` — Markdown table string for reports
        """
        if not gt_arrays:
            return {
                "attributes": {},
                "tier_performance": {},
                "tier_summary": "No sequences to analyse.",
                "markdown_table": "",
            }

        # --- Stage 1: per-sequence attribute computation ---
        attrs: Dict[str, SequenceAttributes] = {}
        for name, gt in gt_arrays.items():
            attrs[name] = _compute_attributes(name, gt)

        # --- Stage 2: difficulty scoring and tier assignment ---
        self._assign_difficulty_scores(attrs)

        # --- Stage 3: per-tier IoU breakdown ---
        tier_perf = self._compute_tier_performance(attrs, iou_arrays, tracker_name)

        return {
            "attributes": attrs,
            "tier_performance": tier_perf,
            "tier_summary": self._format_summary(tracker_name, tier_perf),
            "markdown_table": self._build_markdown_table(tracker_name, attrs, tier_perf),
        }

    # ------------------------------------------------------------------
    # Difficulty scoring
    # ------------------------------------------------------------------

    def _assign_difficulty_scores(
        self, attrs: Dict[str, SequenceAttributes]
    ) -> None:
        """Compute composite difficulty scores and tier labels in-place."""
        names = list(attrs)

        speeds = np.array([attrs[n].mean_speed_px for n in names])
        log_scale_ratios = np.array(
            [math.log(max(attrs[n].scale_change_ratio, 1.0)) for n in names]
        )
        ar_stds = np.array([attrs[n].aspect_ratio_std for n in names])

        norm_speed = _normalize_array(speeds)
        norm_scale = _normalize_array(log_scale_ratios)
        norm_ar = _normalize_array(ar_stds)

        difficulties = (
            self.speed_weight * norm_speed
            + self.scale_weight * norm_scale
            + self.ar_weight * norm_ar
        )

        easy_thr = float(np.percentile(difficulties, self.easy_percentile))
        hard_thr = float(np.percentile(difficulties, self.hard_percentile))

        for i, name in enumerate(names):
            d = float(difficulties[i])
            attrs[name].difficulty_score = d
            if d < easy_thr:
                attrs[name].tier = DifficultyTier.EASY
            elif d < hard_thr:
                attrs[name].tier = DifficultyTier.MEDIUM
            else:
                attrs[name].tier = DifficultyTier.HARD

    # ------------------------------------------------------------------
    # Per-tier performance
    # ------------------------------------------------------------------

    def _compute_tier_performance(
        self,
        attrs: Dict[str, SequenceAttributes],
        iou_arrays: Dict[str, np.ndarray],
        tracker_name: str,
    ) -> Dict[DifficultyTier, TierPerformance]:
        """Aggregate IoU statistics per difficulty tier."""
        tier_ious: Dict[DifficultyTier, List[np.ndarray]] = {
            t: [] for t in DifficultyTier
        }
        tier_seqs: Dict[DifficultyTier, List[str]] = {t: [] for t in DifficultyTier}

        for name, attr in attrs.items():
            iou = iou_arrays.get(name)
            if iou is None or attr.tier is None:
                continue
            tier_ious[attr.tier].append(np.asarray(iou, dtype=np.float64))
            tier_seqs[attr.tier].append(name)

        performances: Dict[DifficultyTier, TierPerformance] = {}
        for tier in DifficultyTier:
            ious_list = tier_ious[tier]
            if not ious_list:
                performances[tier] = TierPerformance(
                    tier=tier,
                    num_sequences=0,
                    mean_iou=0.0,
                    mean_success_rate=0.0,
                    mean_survival_rate=0.0,
                    sequence_names=[],
                )
                continue

            all_ious = np.concatenate(ious_list)
            mean_iou = float(all_ious.mean()) if len(all_ious) else 0.0
            success_rate = float((all_ious >= 0.5).mean()) if len(all_ious) else 0.0
            survival_rate = float((all_ious >= 0.1).mean()) if len(all_ious) else 0.0

            performances[tier] = TierPerformance(
                tier=tier,
                num_sequences=len(ious_list),
                mean_iou=mean_iou,
                mean_success_rate=success_rate,
                mean_survival_rate=survival_rate,
                sequence_names=tier_seqs[tier],
            )

        return performances

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _format_summary(
        self,
        tracker_name: str,
        tier_perf: Dict[DifficultyTier, TierPerformance],
    ) -> str:
        label = f" [{tracker_name}]" if tracker_name else ""
        lines = [
            f"\n{'=' * 60}",
            f"  DIFFICULTY-STRATIFIED PERFORMANCE{label}",
            f"{'=' * 60}",
        ]
        for tier in DifficultyTier:
            tp = tier_perf.get(tier)
            if tp is None or tp.num_sequences == 0:
                lines.append(f"  {tier.value.upper():6s} : no sequences")
                continue
            lines.append(
                f"  {tier.value.upper():6s} "
                f"({tp.num_sequences:>3d} seq) : "
                f"mIoU={tp.mean_iou:.4f}  "
                f"success@0.5={tp.mean_success_rate:.4f}  "
                f"survival@0.1={tp.mean_survival_rate:.4f}"
            )
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    def _build_markdown_table(
        self,
        tracker_name: str,
        attrs: Dict[str, SequenceAttributes],
        tier_perf: Dict[DifficultyTier, TierPerformance],
    ) -> str:
        """Build a Markdown report with tier summary and per-sequence details."""
        label = f" — {tracker_name}" if tracker_name else ""
        lines = [
            f"# Sequence Difficulty Analysis{label}\n",
            "## Per-Tier Performance\n",
            "| Tier | Sequences | mIoU | Success@0.5 | Survival@0.1 |",
            "|:-----|----------:|-----:|------------:|-------------:|",
        ]
        for tier in DifficultyTier:
            tp = tier_perf.get(tier)
            if tp is None or tp.num_sequences == 0:
                lines.append(f"| {tier.value} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {tier.value} | {tp.num_sequences} "
                f"| {tp.mean_iou:.4f} "
                f"| {tp.mean_success_rate:.4f} "
                f"| {tp.mean_survival_rate:.4f} |"
            )

        lines += [
            "",
            "## Per-Sequence Attributes\n",
            "| Sequence | Frames | Speed (px/f) | Scale Ratio | AR Std | Difficulty | Tier |",
            "|----------|-------:|-------------:|------------:|-------:|-----------:|:-----|",
        ]
        for name, attr in sorted(attrs.items()):
            diff_str = f"{attr.difficulty_score:.3f}" if attr.difficulty_score is not None else "?"
            tier_str = attr.tier.value if attr.tier else "?"
            lines.append(
                f"| {name} | {attr.num_frames} "
                f"| {attr.mean_speed_px:.2f} "
                f"| {attr.scale_change_ratio:.2f} "
                f"| {attr.aspect_ratio_std:.3f} "
                f"| {diff_str} "
                f"| {tier_str} |"
            )
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience: build from BenchmarkResult
    # ------------------------------------------------------------------

    @classmethod
    def from_benchmark_result(cls, result: "BenchmarkResult", **kwargs) -> Dict:  # type: ignore[name-defined]
        """Convenience wrapper to analyze a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            result: Output of :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            **kwargs: Forwarded to :class:`SequenceAnalyzer` constructor.

        Returns:
            Same dict structure as :meth:`analyze`.

        Raises:
            ValueError: If no sequence result has ground-truth arrays stored.

        Example::

            from eovot.analysis.sequence_analyzer import SequenceAnalyzer

            report = SequenceAnalyzer.from_benchmark_result(result)
            print(report["markdown_table"])
            with open("difficulty_report.md", "w") as f:
                f.write(report["markdown_table"])
        """
        gt_arrays = {}
        iou_arrays = {}
        for sr in result.sequence_results:
            if sr.ground_truths is not None:
                gt_arrays[sr.sequence_name] = sr.ground_truths
            if sr.ious is not None and len(sr.ious):
                iou_arrays[sr.sequence_name] = sr.ious

        if not gt_arrays:
            raise ValueError(
                "No ground-truth arrays found in sequence results. "
                "Ensure BenchmarkEngine stores predictions and ground-truths."
            )

        analyzer = cls(**kwargs)
        return analyzer.analyze(
            gt_arrays=gt_arrays,
            iou_arrays=iou_arrays,
            tracker_name=result.tracker_name,
        )
