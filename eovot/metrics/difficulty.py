"""Sequence difficulty scoring for stratified VOT benchmarking.

Computes a per-sequence difficulty score from ground-truth bounding-box
trajectories — no visual features required.  Four orthogonal difficulty
dimensions are measured and fused into a single scalar in ``[0, 1]``:

+---------------------+------------------------------------------------------+
| Dimension           | What it captures                                     |
+=====================+======================================================+
| motion_score        | Mean frame-to-frame GT displacement / box diagonal.  |
|                     | High value → fast or erratic target motion.          |
+---------------------+------------------------------------------------------+
| scale_score         | log(max GT area / min GT area), normalised to [0,1]. |
|                     | High value → dramatic scale change across sequence.  |
+---------------------+------------------------------------------------------+
| aspect_score        | Coefficient of variation of w/h ratio.               |
|                     | High value → frequent deformation / rotation.        |
+---------------------+------------------------------------------------------+
| length_score        | Sigmoid of sequence length (frames).                 |
|                     | High value → long sequence; more failure chances.    |
+---------------------+------------------------------------------------------+

Overall score (default weights):

    overall = 0.40 × motion + 0.30 × scale + 0.15 × aspect + 0.15 × length

Difficulty tiers:

    easy   →  overall < 0.35
    medium →  0.35 ≤ overall < 0.65
    hard   →  overall ≥ 0.65

Stratified benchmarking groups sequences by tier and reports per-tier mean
accuracy, enabling researchers to isolate where trackers succeed or struggle.

Typical usage::

    import numpy as np
    from eovot.metrics.difficulty import SequenceDifficultyAnalyzer

    gt = np.array([[x, y, w, h], ...])   # (N, 4) ground-truth boxes
    analyzer = SequenceDifficultyAnalyzer()
    diff = analyzer.score_sequence(gt, sequence_name="car1")
    print(diff)
    # SequenceDifficulty[car1] tier=hard  overall=0.728  (motion=0.81 scale=0.64 aspect=0.21 length=0.68)

    # Full benchmark stratification
    from eovot.benchmark.engine import BenchmarkResult
    report = analyzer.stratified_report(benchmark_result)
    print(report.to_markdown())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default per-dimension weights for overall difficulty score.
DEFAULT_WEIGHTS = {
    "motion": 0.40,
    "scale": 0.30,
    "aspect": 0.15,
    "length": 0.15,
}

#: Length (frames) at which the sigmoid-based length score equals 0.5.
LENGTH_MIDPOINT = 300

TIER_EASY_THRESHOLD = 0.35
TIER_HARD_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Per-sequence difficulty
# ---------------------------------------------------------------------------

@dataclass
class SequenceDifficulty:
    """Difficulty profile for a single tracking sequence.

    Attributes:
        sequence_name: Human-readable identifier.
        motion_score:  Normalised mean displacement (0 = stationary, 1 = very fast).
        scale_score:   Normalised scale variation (0 = fixed size, 1 = extreme change).
        aspect_score:  Coefficient of variation of w/h ratio, capped at 1.
        length_score:  Sigmoid of sequence length — proxy for sustained difficulty.
        overall_score: Weighted combination of the four scores (see module docs).
        tier:          ``"easy"``, ``"medium"``, or ``"hard"``.
        num_frames:    Number of frames in the sequence.
    """

    sequence_name: str
    motion_score: float
    scale_score: float
    aspect_score: float
    length_score: float
    overall_score: float
    tier: str
    num_frames: int

    def __str__(self) -> str:
        return (
            f"SequenceDifficulty[{self.sequence_name}] "
            f"tier={self.tier}  overall={self.overall_score:.3f}  "
            f"(motion={self.motion_score:.2f} "
            f"scale={self.scale_score:.2f} "
            f"aspect={self.aspect_score:.2f} "
            f"length={self.length_score:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "tier": self.tier,
            "overall_score": round(self.overall_score, 4),
            "motion_score": round(self.motion_score, 4),
            "scale_score": round(self.scale_score, 4),
            "aspect_score": round(self.aspect_score, 4),
            "length_score": round(self.length_score, 4),
        }


# ---------------------------------------------------------------------------
# Stratified report
# ---------------------------------------------------------------------------

@dataclass
class StratifiedReport:
    """Per-tier accuracy breakdown for a benchmark run.

    Attributes:
        tracker_name:   Name of the evaluated tracker.
        dataset_name:   Name of the evaluated dataset.
        difficulties:   Per-sequence :class:`SequenceDifficulty` objects.
        tier_stats:     Mapping from tier name to per-tier statistics dict.
    """

    tracker_name: str
    dataset_name: str
    difficulties: List[SequenceDifficulty]
    tier_stats: Dict[str, Dict]

    def to_markdown(self) -> str:
        """Render a Markdown table summarising per-tier accuracy.

        Returns:
            Multi-line string with a Markdown table and a summary of sequence
            counts per tier.
        """
        header = (
            f"## Difficulty-Stratified Results: {self.tracker_name} on {self.dataset_name}\n\n"
        )
        col_names = ["Tier", "Sequences", "Mean IoU", "Success AUC", "FPS"]
        row_fmt = "| {:<8} | {:>9} | {:>8} | {:>11} | {:>6} |"
        sep = "| " + " | ".join(["---"] * len(col_names)) + " |"
        header_row = "| " + " | ".join(f"{c}" for c in col_names) + " |"

        rows = [header, header_row, sep]
        for tier in ("easy", "medium", "hard"):
            stats = self.tier_stats.get(tier, {})
            n = stats.get("num_sequences", 0)
            miou = f"{stats['mean_iou']:.4f}" if "mean_iou" in stats else "—"
            sauc = f"{stats['success_auc']:.4f}" if "success_auc" in stats else "—"
            fps = f"{stats['mean_fps']:.1f}" if "mean_fps" in stats else "—"
            rows.append(row_fmt.format(tier.capitalize(), n, miou, sauc, fps))

        rows.append("")
        tier_counts = {t: sum(1 for d in self.difficulties if d.tier == t)
                       for t in ("easy", "medium", "hard")}
        rows.append(
            f"> Tier thresholds: easy < {TIER_EASY_THRESHOLD}, "
            f"hard ≥ {TIER_HARD_THRESHOLD}. "
            f"Counts: easy={tier_counts['easy']}, "
            f"medium={tier_counts['medium']}, hard={tier_counts['hard']}."
        )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class SequenceDifficultyAnalyzer:
    """Score the tracking difficulty of sequences from ground-truth boxes.

    Args:
        weights: Override default dimension weights.  Keys must be a subset
            of ``{"motion", "scale", "aspect", "length"}``.  Missing keys use
            defaults; weights are renormalised to sum to 1.
        length_midpoint: Sequence length (frames) at which the length score
            equals 0.5.  Default: ``300`` (typical medium-length sequence).

    Example::

        analyzer = SequenceDifficultyAnalyzer()
        diff = analyzer.score_sequence(gt_boxes, sequence_name="dog1")
        print(diff.tier)  # "medium"
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        length_midpoint: int = LENGTH_MIDPOINT,
    ) -> None:
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        total = sum(w.values())
        self._weights = {k: v / total for k, v in w.items()}
        self._length_midpoint = length_midpoint

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_sequence(
        self,
        gt_boxes: np.ndarray,
        sequence_name: str = "",
    ) -> SequenceDifficulty:
        """Compute the difficulty profile for a single sequence.

        Args:
            gt_boxes:      ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.
            sequence_name: Human-readable identifier embedded in the result.

        Returns:
            :class:`SequenceDifficulty` with all scores populated.

        Raises:
            ValueError: If ``gt_boxes`` has fewer than 2 frames.
        """
        gt = np.asarray(gt_boxes, dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(f"gt_boxes must be (N, 4), got shape {gt.shape}")
        if len(gt) < 2:
            raise ValueError("score_sequence requires at least 2 GT frames.")

        motion = self._motion_score(gt)
        scale = self._scale_score(gt)
        aspect = self._aspect_score(gt)
        length = self._length_score(len(gt))

        w = self._weights
        overall = (
            w["motion"] * motion
            + w["scale"] * scale
            + w["aspect"] * aspect
            + w["length"] * length
        )
        overall = float(np.clip(overall, 0.0, 1.0))
        tier = _assign_tier(overall)

        return SequenceDifficulty(
            sequence_name=sequence_name,
            motion_score=round(motion, 4),
            scale_score=round(scale, 4),
            aspect_score=round(aspect, 4),
            length_score=round(length, 4),
            overall_score=round(overall, 4),
            tier=tier,
            num_frames=len(gt),
        )

    def score_dataset(
        self,
        sequences_gt: Dict[str, np.ndarray],
    ) -> List[SequenceDifficulty]:
        """Score every sequence in a dataset.

        Args:
            sequences_gt: Mapping ``{sequence_name: gt_boxes}``.

        Returns:
            List of :class:`SequenceDifficulty` objects, one per sequence.
        """
        results = []
        for name, gt in sequences_gt.items():
            try:
                results.append(self.score_sequence(gt, sequence_name=name))
            except ValueError:
                pass  # skip sequences with < 2 frames
        return results

    def stratified_report(
        self,
        benchmark_result: "BenchmarkResult",
    ) -> StratifiedReport:
        """Group a :class:`~eovot.benchmark.engine.BenchmarkResult` by difficulty tier.

        Sequences that lack ground-truth boxes in the result are scored from
        the stored per-frame IoU array (fallback: motion is not computable,
        so only length is used).

        Args:
            benchmark_result: Full benchmark result with per-sequence data.

        Returns:
            :class:`StratifiedReport` with per-tier mean IoU, success AUC,
            and FPS computed from the sequences in each tier.
        """
        from ..benchmark.engine import SequenceResult

        difficulties: Dict[str, SequenceDifficulty] = {}
        for sr in benchmark_result.sequence_results:
            if sr.ground_truths is not None and len(sr.ground_truths) >= 2:
                diff = self.score_sequence(sr.ground_truths, sequence_name=sr.sequence_name)
            else:
                # Fallback: length-only score
                n = len(sr.ious)
                length = self._length_score(n)
                overall = float(np.clip(self._weights["length"] * length, 0.0, 1.0))
                diff = SequenceDifficulty(
                    sequence_name=sr.sequence_name,
                    motion_score=0.0,
                    scale_score=0.0,
                    aspect_score=0.0,
                    length_score=round(length, 4),
                    overall_score=round(overall, 4),
                    tier=_assign_tier(overall),
                    num_frames=n,
                )
            difficulties[sr.sequence_name] = diff

        # Group sequence results by tier
        tier_results: Dict[str, List[SequenceResult]] = {
            "easy": [], "medium": [], "hard": []
        }
        for sr in benchmark_result.sequence_results:
            tier = difficulties.get(sr.sequence_name, SequenceDifficulty(
                sequence_name=sr.sequence_name,
                motion_score=0.0, scale_score=0.0, aspect_score=0.0,
                length_score=0.0, overall_score=0.0, tier="easy", num_frames=0,
            )).tier
            tier_results[tier].append(sr)

        # Compute per-tier statistics
        tier_stats: Dict[str, Dict] = {}
        for tier, srs in tier_results.items():
            if not srs:
                tier_stats[tier] = {"num_sequences": 0}
                continue
            all_ious = np.concatenate([r.ious for r in srs]) if any(len(r.ious) for r in srs) else np.array([])
            mean_iou = float(all_ious.mean()) if len(all_ious) else 0.0
            mean_fps = float(np.mean([r.profiling.fps for r in srs]))

            aucs = [r.accuracy_metrics.success_auc for r in srs if r.accuracy_metrics is not None]
            success_auc = float(np.mean(aucs)) if aucs else None

            stat: Dict = {
                "num_sequences": len(srs),
                "mean_iou": round(mean_iou, 4),
                "mean_fps": round(mean_fps, 2),
            }
            if success_auc is not None:
                stat["success_auc"] = round(success_auc, 4)
            tier_stats[tier] = stat

        return StratifiedReport(
            tracker_name=benchmark_result.tracker_name,
            dataset_name=benchmark_result.dataset_name,
            difficulties=list(difficulties.values()),
            tier_stats=tier_stats,
        )

    # ------------------------------------------------------------------
    # Dimension scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_score(gt: np.ndarray) -> float:
        """Normalised mean frame-to-frame centre displacement.

        Computes the Euclidean distance between consecutive GT centres,
        normalises each displacement by the geometric mean of the GT box
        size (diagonal proxy), and maps the result to ``[0, 1]`` via a
        sigmoid-like transform with midpoint at 0.3 (30 % of box diagonal
        per frame ≈ moderate fast motion).

        Args:
            gt: ``(N, 4)`` GT boxes ``(x, y, w, h)``.

        Returns:
            Float in ``[0, 1]``.
        """
        centres = gt[:, :2] + gt[:, 2:] / 2.0  # (N, 2)
        disps = np.sqrt(np.sum(np.diff(centres, axis=0) ** 2, axis=1))  # (N-1,)
        diagonals = np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2)  # (N,)
        mean_diag = float(np.mean(diagonals)) + 1e-6
        norm_disp = float(np.mean(disps)) / mean_diag
        # Logistic with midpoint 0.30 (30 % of box diagonal / frame)
        score = 1.0 / (1.0 + math.exp(-10.0 * (norm_disp - 0.30)))
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _scale_score(gt: np.ndarray) -> float:
        """Normalised log area ratio: log(max_area / min_area).

        Maps the log ratio to ``[0, 1]`` by dividing by ``log(100)``
        (a 100× area change is considered maximal difficulty).

        Args:
            gt: ``(N, 4)`` GT boxes ``(x, y, w, h)``.

        Returns:
            Float in ``[0, 1]``.
        """
        areas = gt[:, 2] * gt[:, 3]
        valid = areas[areas > 0]
        if len(valid) < 2:
            return 0.0
        ratio = float(valid.max()) / float(valid.min())
        score = math.log(max(ratio, 1.0)) / math.log(100.0)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _aspect_score(gt: np.ndarray) -> float:
        """Coefficient of variation of the width/height aspect ratio.

        High CV means the aspect ratio changes substantially across the
        sequence — typical of out-of-plane rotation or non-rigid deformation.

        Args:
            gt: ``(N, 4)`` GT boxes ``(x, y, w, h)``.

        Returns:
            Float in ``[0, 1]`` (CV capped at 1.0).
        """
        h = gt[:, 3]
        valid_h = h > 0
        if not np.any(valid_h):
            return 0.0
        aspect = gt[valid_h, 2] / gt[valid_h, 3]
        mean_asp = float(np.mean(aspect))
        if mean_asp < 1e-6:
            return 0.0
        cv = float(np.std(aspect)) / mean_asp
        return float(np.clip(cv, 0.0, 1.0))

    def _length_score(self, num_frames: int) -> float:
        """Sigmoid-based length score.

        Returns 0.5 when ``num_frames == length_midpoint``, approaching 1 for
        very long sequences and 0 for very short ones.

        Args:
            num_frames: Total frames in the sequence.

        Returns:
            Float in ``(0, 1)``.
        """
        x = (num_frames - self._length_midpoint) / (self._length_midpoint * 0.5)
        return float(1.0 / (1.0 + math.exp(-x)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_tier(overall: float) -> str:
    if overall < TIER_EASY_THRESHOLD:
        return "easy"
    if overall < TIER_HARD_THRESHOLD:
        return "medium"
    return "hard"
