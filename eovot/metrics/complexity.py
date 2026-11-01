"""Sequence complexity scoring for edge-aware dataset analysis.

Quantifies how challenging a tracking sequence is, purely from its
ground-truth bounding boxes — no tracker predictions required.  This
enables researchers to:

- **Subsample large datasets** by selecting a representative mix of
  easy, medium, and hard sequences without running any tracker.
- **Report per-difficulty-tier breakdowns** ("tracker A outperforms B
  on hard sequences, but not on easy ones").
- **Design fairer comparisons** by matching sequences across difficulty
  tiers when datasets differ in composition.

Three orthogonal challenge dimensions are measured:

Motion Intensity (w=0.50)
~~~~~~~~~~~~~~~~~~~~~~~~~
Mean frame-to-frame GT center displacement, normalised by the mean box
diagonal.  A value of 0 means the target never moves; a value of 1 means
it displaces by a full box-diagonal per frame.

    motion_intensity = mean(||Δcenter_t||) / mean_diagonal

Scale Change (w=0.30)
~~~~~~~~~~~~~~~~~~~~~
Log-ratio of the maximum to minimum GT-box area, normalised by log(100)
so that a 100× area change maps to 1.0.

    scale_change = log(max_area / max(min_area, 1)) / log(100)

Aspect-Ratio Variation (w=0.20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Coefficient of variation of the GT bounding-box aspect ratio (w/h).
Captures deformation and out-of-plane rotation.

    ar_variation = std(w/h) / max(mean(w/h), 1e-6)

Combined Complexity Score
~~~~~~~~~~~~~~~~~~~~~~~~~
A weighted sum in ``[0, 1]``::

    score = 0.50 × clip(motion_intensity, 0, 1)
          + 0.30 × clip(scale_change, 0, 1)
          + 0.20 × clip(ar_variation, 0, 1)

Difficulty Tiers
~~~~~~~~~~~~~~~~
- **EASY**   — score < 0.25
- **MEDIUM** — 0.25 ≤ score < 0.55
- **HARD**   — score ≥ 0.55

Typical usage::

    from eovot.metrics.complexity import SequenceComplexityScorer, ComplexityAnalyzer

    scorer = SequenceComplexityScorer()

    # Score one sequence from GT boxes
    result = scorer.score(gt_boxes, sequence_name="car1")
    print(result)  # ComplexityResult[car1]  score=0.612  HARD
    print(result.tier.name)  # HARD

    # Analyse all sequences in a benchmark run
    analyzer = ComplexityAnalyzer()
    breakdown = analyzer.breakdown(benchmark_result)
    print(breakdown["tier_summary"])
    # {'EASY': {'count': 4, 'mean_iou': 0.73, ...}, 'MEDIUM': {...}, 'HARD': {...}}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Difficulty tier
# ---------------------------------------------------------------------------

class DifficultyTier(Enum):
    """Categorical difficulty level for a tracking sequence."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"

    @classmethod
    def from_score(cls, score: float) -> "DifficultyTier":
        """Map a complexity score to a tier.

        Args:
            score: Value in ``[0, 1]`` from :class:`SequenceComplexityScorer`.

        Returns:
            :attr:`EASY` if ``score < 0.25``,
            :attr:`MEDIUM` if ``0.25 ≤ score < 0.55``,
            :attr:`HARD` if ``score ≥ 0.55``.
        """
        if score < 0.25:
            return cls.EASY
        if score < 0.55:
            return cls.MEDIUM
        return cls.HARD


# ---------------------------------------------------------------------------
# Per-sequence result
# ---------------------------------------------------------------------------

@dataclass
class ComplexityResult:
    """Complexity analysis for a single tracking sequence.

    Attributes:
        sequence_name: Identifier of the analysed sequence.
        complexity_score: Weighted composite difficulty score in ``[0, 1]``.
        tier: :class:`DifficultyTier` bucket (EASY / MEDIUM / HARD).
        motion_intensity: Normalised mean GT displacement per frame.
        scale_change: Log-normalised GT area range.
        ar_variation: Coefficient of variation of GT aspect ratio.
        num_frames: Number of frames in the sequence.
    """

    sequence_name: str
    complexity_score: float
    tier: DifficultyTier
    motion_intensity: float
    scale_change: float
    ar_variation: float
    num_frames: int

    def __str__(self) -> str:
        return (
            f"ComplexityResult[{self.sequence_name}]  "
            f"score={self.complexity_score:.3f}  {self.tier.name}  "
            f"(motion={self.motion_intensity:.3f}  "
            f"scale={self.scale_change:.3f}  "
            f"ar_var={self.ar_variation:.3f})"
        )

    def to_dict(self) -> Dict:
        return {
            "sequence_name": self.sequence_name,
            "complexity_score": round(self.complexity_score, 4),
            "tier": self.tier.name,
            "motion_intensity": round(self.motion_intensity, 4),
            "scale_change": round(self.scale_change, 4),
            "ar_variation": round(self.ar_variation, 4),
            "num_frames": self.num_frames,
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class SequenceComplexityScorer:
    """Score tracking sequences by difficulty from GT bounding boxes only.

    Args:
        motion_weight:    Weight for the motion-intensity component.
            Default: ``0.50``.
        scale_weight:     Weight for the scale-change component.
            Default: ``0.30``.
        ar_weight:        Weight for the aspect-ratio-variation component.
            Default: ``0.20``.

    Weights must sum to 1.0.  Passing custom weights lets you emphasise
    the challenge dimension most relevant to your deployment scenario —
    e.g. a drone tracker for slow-moving objects cares more about scale
    change than raw motion speed.

    Example::

        scorer = SequenceComplexityScorer()
        result = scorer.score(gt_boxes, sequence_name="walking_person")
        if result.tier == DifficultyTier.HARD:
            print("Skip this sequence for a quick smoke test")
    """

    def __init__(
        self,
        motion_weight: float = 0.50,
        scale_weight: float = 0.30,
        ar_weight: float = 0.20,
    ) -> None:
        total = motion_weight + scale_weight + ar_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f}. "
                f"Adjust motion_weight, scale_weight, ar_weight."
            )
        self.motion_weight = motion_weight
        self.scale_weight = scale_weight
        self.ar_weight = ar_weight

    # ------------------------------------------------------------------
    # Individual component computations
    # ------------------------------------------------------------------

    def motion_intensity(self, gt: np.ndarray) -> float:
        """Compute normalised mean GT displacement per frame.

        Args:
            gt: ``(N, 4)`` GT bounding boxes ``(x, y, w, h)``.

        Returns:
            Normalised displacement in ``[0, ∞)``.  Returns ``0.0``
            for sequences with fewer than two frames.
        """
        if len(gt) < 2:
            return 0.0
        g = np.asarray(gt, dtype=np.float64)
        centers = g[:, :2] + g[:, 2:] / 2.0          # (N, 2)
        disps = np.linalg.norm(np.diff(centers, axis=0), axis=1)  # (N-1,)
        diags = np.sqrt(g[:, 2] ** 2 + g[:, 3] ** 2)
        mean_diag = float(diags.mean())
        if mean_diag < 1e-6:
            return 0.0
        return float(disps.mean() / mean_diag)

    def scale_change(self, gt: np.ndarray) -> float:
        """Log-normalised GT area range.

        Args:
            gt: ``(N, 4)`` GT bounding boxes ``(x, y, w, h)``.

        Returns:
            Value in ``[0, 1]`` where 1.0 corresponds to a 100× area
            change.  Returns ``0.0`` for degenerate boxes.
        """
        g = np.asarray(gt, dtype=np.float64)
        areas = g[:, 2] * g[:, 3]
        pos_areas = areas[areas > 0]
        if len(pos_areas) < 2:
            return 0.0
        ratio = float(pos_areas.max()) / float(pos_areas.min())
        if ratio <= 1.0:
            return 0.0
        return min(math.log(ratio) / math.log(100.0), 1.0)

    def ar_variation(self, gt: np.ndarray) -> float:
        """Coefficient of variation of GT aspect ratio.

        Args:
            gt: ``(N, 4)`` GT bounding boxes ``(x, y, w, h)``.

        Returns:
            Non-negative float; 0.0 for constant aspect ratio.
        """
        g = np.asarray(gt, dtype=np.float64)
        valid = g[(g[:, 2] > 0) & (g[:, 3] > 0)]
        if len(valid) < 2:
            return 0.0
        ratios = valid[:, 2] / valid[:, 3]
        mean_r = float(ratios.mean())
        if mean_r < 1e-6:
            return 0.0
        return float(ratios.std() / mean_r)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def score(
        self,
        gt: np.ndarray,
        sequence_name: str = "",
    ) -> ComplexityResult:
        """Compute a composite complexity score for one sequence.

        Args:
            gt:            ``(N, 4)`` GT bounding boxes ``(x, y, w, h)``.
            sequence_name: Human-readable sequence identifier for reports.

        Returns:
            :class:`ComplexityResult` with all component scores and the
            overall weighted complexity score.
        """
        gt = np.asarray(gt, dtype=np.float64)
        n = len(gt)

        mi = self.motion_intensity(gt)
        sc = self.scale_change(gt)
        arv = self.ar_variation(gt)

        # Clip each component to [0, 1] before weighting
        score = (
            self.motion_weight * min(mi, 1.0)
            + self.scale_weight * min(sc, 1.0)
            + self.ar_weight * min(arv, 1.0)
        )

        return ComplexityResult(
            sequence_name=sequence_name,
            complexity_score=round(float(score), 6),
            tier=DifficultyTier.from_score(score),
            motion_intensity=round(mi, 6),
            scale_change=round(sc, 6),
            ar_variation=round(arv, 6),
            num_frames=n,
        )

    def score_many(
        self,
        sequences: Dict[str, np.ndarray],
    ) -> List[ComplexityResult]:
        """Score a collection of sequences and return them sorted by complexity.

        Args:
            sequences: Mapping ``{sequence_name: gt_array}``.

        Returns:
            List of :class:`ComplexityResult` sorted by
            ``complexity_score`` descending (hardest first).
        """
        results = [
            self.score(gt, seq_name) for seq_name, gt in sequences.items()
        ]
        results.sort(key=lambda r: r.complexity_score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Benchmark-level analyser
# ---------------------------------------------------------------------------

class ComplexityAnalyzer:
    """Integrate complexity scoring with a full benchmark run.

    Takes a :class:`~eovot.benchmark.engine.BenchmarkResult` (which stores
    per-sequence GT boxes) and produces a difficulty-stratified breakdown of
    accuracy and efficiency metrics.

    Args:
        scorer: :class:`SequenceComplexityScorer` instance to use.
            Defaults to one with the standard weights.

    Example::

        from eovot.metrics.complexity import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer()
        breakdown = analyzer.breakdown(benchmark_result)

        for tier, stats in breakdown["tier_summary"].items():
            print(f"{tier}: n={stats['count']}  mIoU={stats['mean_iou']:.4f}")
    """

    def __init__(self, scorer: Optional[SequenceComplexityScorer] = None) -> None:
        self._scorer = scorer or SequenceComplexityScorer()

    def breakdown(self, result: "BenchmarkResult") -> Dict:
        """Compute per-sequence complexity scores and tier-stratified metrics.

        Sequences without stored GT boxes are skipped (ground-truth is
        required for the GT-based complexity computation).

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` from
                the benchmark engine.  Each
                :class:`~eovot.benchmark.engine.SequenceResult` must have
                ``ground_truths`` populated (set when
                :class:`~eovot.benchmark.engine.BenchmarkEngine` runs the
                sequence).

        Returns:
            Dict with three keys:

            * ``"per_sequence"`` — ``{seq_name: ComplexityResult}``
            * ``"tier_summary"`` — aggregate metrics per tier
              (``EASY``, ``MEDIUM``, ``HARD``)
            * ``"overall_complexity"`` — mean complexity score across all
              sequences
        """
        per_seq: Dict[str, ComplexityResult] = {}
        tier_data: Dict[str, List] = {t.name: [] for t in DifficultyTier}

        for sr in result.sequence_results:
            if sr.ground_truths is None or len(sr.ground_truths) == 0:
                continue
            cr = self._scorer.score(sr.ground_truths, sr.sequence_name)
            per_seq[sr.sequence_name] = cr

            # Collect per-tier accuracy
            entry = {"mean_iou": sr.mean_iou}
            if sr.accuracy_metrics is not None:
                entry["success_auc"] = sr.accuracy_metrics.success_auc
                entry["precision_auc"] = sr.accuracy_metrics.precision_auc
            entry["fps"] = sr.profiling.fps
            tier_data[cr.tier.name].append(entry)

        tier_summary: Dict[str, Dict] = {}
        for tier_name, entries in tier_data.items():
            if not entries:
                tier_summary[tier_name] = {"count": 0}
                continue
            ious = [e["mean_iou"] for e in entries]
            fps_vals = [e["fps"] for e in entries]
            s: Dict = {
                "count": len(entries),
                "mean_iou": round(float(np.mean(ious)), 4),
                "mean_fps": round(float(np.mean(fps_vals)), 2),
            }
            sauc_vals = [e["success_auc"] for e in entries if "success_auc" in e]
            if sauc_vals:
                s["mean_success_auc"] = round(float(np.mean(sauc_vals)), 4)
            pauc_vals = [e["precision_auc"] for e in entries if "precision_auc" in e]
            if pauc_vals:
                s["mean_precision_auc"] = round(float(np.mean(pauc_vals)), 4)
            tier_summary[tier_name] = s

        all_scores = [cr.complexity_score for cr in per_seq.values()]
        overall = round(float(np.mean(all_scores)), 4) if all_scores else 0.0

        return {
            "per_sequence": per_seq,
            "tier_summary": tier_summary,
            "overall_complexity": overall,
        }

    def select_representative_subset(
        self,
        result: "BenchmarkResult",
        n_easy: int = 3,
        n_medium: int = 3,
        n_hard: int = 3,
    ) -> List[str]:
        """Select a balanced subset of sequence names across difficulty tiers.

        Useful for creating a fast evaluation suite that exercises trackers
        across the full difficulty spectrum without running the full dataset.

        Args:
            result:   Benchmark result with GT boxes stored per sequence.
            n_easy:   Number of easy sequences to include.  Default: ``3``.
            n_medium: Number of medium sequences to include.  Default: ``3``.
            n_hard:   Number of hard sequences to include.  Default: ``3``.

        Returns:
            List of sequence names selected from each tier, sorted by
            complexity score within each tier (easy → hard).  Fewer sequences
            are returned when a tier has fewer candidates than requested.
        """
        bd = self.breakdown(result)
        per_seq: Dict[str, ComplexityResult] = bd["per_sequence"]

        tier_seqs: Dict[str, List[ComplexityResult]] = {t.name: [] for t in DifficultyTier}
        for cr in per_seq.values():
            tier_seqs[cr.tier.name].append(cr)

        for tier_name in tier_seqs:
            tier_seqs[tier_name].sort(key=lambda r: r.complexity_score)

        quotas = {
            DifficultyTier.EASY.name: n_easy,
            DifficultyTier.MEDIUM.name: n_medium,
            DifficultyTier.HARD.name: n_hard,
        }
        selected: List[str] = []
        for tier_name, quota in quotas.items():
            candidates = tier_seqs[tier_name]
            selected.extend(cr.sequence_name for cr in candidates[:quota])
        return selected
