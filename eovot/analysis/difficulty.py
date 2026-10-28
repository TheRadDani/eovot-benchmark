"""Quantitative sequence difficulty profiling for EOVOT.

Computes a multi-dimensional difficulty score for tracking sequences purely
from their ground-truth bounding-box trajectories — no image frames required.
This makes the profiler fast, dataset-agnostic, and usable offline.

Difficulty dimensions
~~~~~~~~~~~~~~~~~~~~~

+---------------------+----------------------------------------------------+
| Metric              | Description                                        |
+=====================+====================================================+
| motion_speed        | Mean centre displacement between consecutive GT    |
|                     | boxes, normalised by mean box diagonal (px → %).  |
|                     | High → fast motion.                               |
+---------------------+----------------------------------------------------+
| scale_volatility    | Coefficient of variation (std/mean) of GT box      |
|                     | area across the sequence.  High → dramatic size   |
|                     | changes (zoom, depth change).                     |
+---------------------+----------------------------------------------------+
| aspect_deformation  | Std of GT aspect ratio (w/h) across the sequence. |
|                     | High → object deforms non-rigidly.                |
+---------------------+----------------------------------------------------+
| occlusion_events    | Number of abrupt area drops (> 40 % in one step)  |
|                     | per 100 frames — proxy for partial occlusion.     |
+---------------------+----------------------------------------------------+
| difficulty_score    | Weighted composite ∈ [0, 1].  0 = trivial,        |
|                     | 1 = extreme.                                      |
+---------------------+----------------------------------------------------+
| difficulty_tier     | Human label: easy / medium / hard / extreme.      |
+---------------------+----------------------------------------------------+

Why this matters
~~~~~~~~~~~~~~~~
The existing attribute detector (`eovot.metrics.attributes`) flags sequences
with *binary* challenge labels (fast_motion: yes/no).  The difficulty profiler
produces *continuous* scores that let researchers:

- Rank sequences from easiest to hardest within a dataset.
- Identify which sequences drive a tracker's aggregate score.
- Stratify benchmark results by difficulty tier for publication tables.
- Compare dataset difficulty distributions across GOT-10k, LaSOT, and OTB.

Typical usage::

    from eovot.analysis.difficulty import SequenceDifficultyProfiler

    profiler = SequenceDifficultyProfiler()

    # From a BenchmarkResult:
    from eovot.analysis.difficulty import profile_benchmark_result
    breakdown = profile_benchmark_result(benchmark_result)
    for name, result in sorted(breakdown.items(), key=lambda x: -x[1].difficulty_score):
        print(f"{name:<30s} {result.difficulty_tier:<8s} score={result.difficulty_score:.3f}")

    # From raw GT boxes (N, 4) array:
    result = profiler.profile_from_gts(gt_array, sequence_name="car1")
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# Bounding box: (x, y, width, height)
BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Difficulty tiers
# ---------------------------------------------------------------------------

DIFFICULTY_TIERS: List[Tuple[float, str]] = [
    (0.25, "easy"),
    (0.50, "medium"),
    (0.75, "hard"),
    (1.01, "extreme"),
]
"""Score thresholds that map a composite difficulty score to a tier label.

Each entry is ``(upper_exclusive_bound, tier_name)``.  Scores in [0, 0.25)
are ``"easy"``; [0.25, 0.50) are ``"medium"``; etc.
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DifficultyResult:
    """Quantitative difficulty profile for a single tracking sequence.

    Attributes:
        sequence_name:      Identifier of the evaluated sequence.
        num_frames:         Total number of frames (GT boxes) in the sequence.
        motion_speed:       Mean per-frame centre displacement normalised by
                            mean GT box diagonal.  Range: [0, ∞).
        scale_volatility:   Coefficient of variation of GT box area.
                            Range: [0, ∞).
        aspect_deformation: Standard deviation of GT aspect ratio (w/h).
                            Range: [0, ∞).
        occlusion_events:   Detected abrupt area-drop events per 100 frames.
        difficulty_score:   Weighted composite difficulty score in [0, 1].
        difficulty_tier:    Human-readable tier: ``"easy"`` / ``"medium"`` /
                            ``"hard"`` / ``"extreme"``.
    """

    sequence_name: str
    num_frames: int
    motion_speed: float
    scale_volatility: float
    aspect_deformation: float
    occlusion_events: float
    difficulty_score: float
    difficulty_tier: str

    def __str__(self) -> str:
        return (
            f"DifficultyResult[{self.sequence_name}] "
            f"tier={self.difficulty_tier}  score={self.difficulty_score:.3f}  "
            f"motion={self.motion_speed:.3f}  scale_cv={self.scale_volatility:.3f}  "
            f"deform={self.aspect_deformation:.3f}  "
            f"occlusion={self.occlusion_events:.2f}/100fr  "
            f"({self.num_frames} frames)"
        )

    def to_dict(self) -> Dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "motion_speed": round(self.motion_speed, 5),
            "scale_volatility": round(self.scale_volatility, 5),
            "aspect_deformation": round(self.aspect_deformation, 5),
            "occlusion_events_per_100fr": round(self.occlusion_events, 3),
            "difficulty_score": round(self.difficulty_score, 4),
            "difficulty_tier": self.difficulty_tier,
        }


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class SequenceDifficultyProfiler:
    """Compute quantitative difficulty scores from GT bounding-box trajectories.

    Args:
        motion_weight:      Relative weight of motion speed in the composite
                            difficulty score.  Default: 0.40.
        scale_weight:       Relative weight of scale volatility.  Default: 0.30.
        deform_weight:      Relative weight of aspect deformation.  Default: 0.20.
        occlusion_weight:   Relative weight of occlusion event rate.  Default: 0.10.
        occlusion_drop_pct: Area-drop fraction threshold for occlusion detection.
                            Default: 0.40 (40 % sudden area reduction).

    Weights are renormalised to sum to 1.0 internally.

    Example::

        profiler = SequenceDifficultyProfiler()
        result = profiler.profile_from_gts(gt_boxes, sequence_name="bird1")
        print(result.difficulty_tier)  # 'hard'
    """

    # Empirical saturation caps — raw metrics beyond these values are all
    # treated as "maximally hard" for that dimension.  Derived from OTB-100
    # statistics; can be overridden via constructor kwargs.
    _CAP_MOTION: float = 0.40        # normalised displacement per frame
    _CAP_SCALE_CV: float = 0.60      # coefficient of variation
    _CAP_DEFORM: float = 0.80        # std of aspect ratio
    _CAP_OCCLUSION: float = 5.0      # events per 100 frames

    def __init__(
        self,
        motion_weight: float = 0.40,
        scale_weight: float = 0.30,
        deform_weight: float = 0.20,
        occlusion_weight: float = 0.10,
        occlusion_drop_pct: float = 0.40,
    ) -> None:
        weights = np.array(
            [motion_weight, scale_weight, deform_weight, occlusion_weight],
            dtype=np.float64,
        )
        if (weights < 0).any():
            raise ValueError("All weights must be non-negative.")
        if weights.sum() <= 0:
            raise ValueError("At least one weight must be positive.")
        self._weights = weights / weights.sum()
        self._occlusion_drop = occlusion_drop_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_from_gts(
        self,
        gts: np.ndarray,
        sequence_name: str = "",
    ) -> DifficultyResult:
        """Profile a sequence from its ground-truth bounding-box array.

        Args:
            gts:           ``(N, 4)`` array of GT boxes ``(x, y, w, h)``.
                           Must contain at least 2 frames.
            sequence_name: Identifier stored in the result.

        Returns:
            Populated :class:`DifficultyResult`.

        Raises:
            ValueError: If *gts* has fewer than 2 rows.
        """
        gts = np.asarray(gts, dtype=np.float64)
        if gts.ndim != 2 or gts.shape[1] < 4:
            raise ValueError("gts must be an (N, 4) array of (x, y, w, h) boxes.")
        if len(gts) < 2:
            raise ValueError("Need at least 2 GT frames to compute difficulty.")

        motion = self._compute_motion_speed(gts)
        scale_cv = self._compute_scale_volatility(gts)
        deform = self._compute_aspect_deformation(gts)
        occlusion = self._compute_occlusion_events(gts)

        score = self._composite_score(motion, scale_cv, deform, occlusion)
        tier = _score_to_tier(score)

        return DifficultyResult(
            sequence_name=sequence_name,
            num_frames=len(gts),
            motion_speed=motion,
            scale_volatility=scale_cv,
            aspect_deformation=deform,
            occlusion_events=occlusion,
            difficulty_score=score,
            difficulty_tier=tier,
        )

    def profile_benchmark(
        self,
        benchmark_result: "BenchmarkResult",
    ) -> Dict[str, DifficultyResult]:
        """Profile every sequence in a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Requires that ``SequenceResult.ground_truths`` is populated (it is
        populated by default when using :class:`~eovot.benchmark.engine.BenchmarkEngine`).

        Args:
            benchmark_result: Full benchmark run output.

        Returns:
            Dict mapping ``sequence_name → DifficultyResult``.
            Sequences without stored GT boxes are silently skipped.
        """
        out: Dict[str, DifficultyResult] = {}
        for seq in benchmark_result.sequence_results:
            if seq.ground_truths is None or len(seq.ground_truths) < 2:
                continue
            out[seq.sequence_name] = self.profile_from_gts(
                seq.ground_truths, sequence_name=seq.sequence_name
            )
        return out

    def aggregate_stats(
        self, results: Dict[str, DifficultyResult]
    ) -> Dict:
        """Compute dataset-level difficulty statistics.

        Args:
            results: Output of :meth:`profile_benchmark` or a custom dict.

        Returns:
            Dict with mean/std of each difficulty metric and tier distribution.
        """
        if not results:
            return {}

        vals = list(results.values())
        scores = np.array([r.difficulty_score for r in vals])
        motions = np.array([r.motion_speed for r in vals])
        scale_cvs = np.array([r.scale_volatility for r in vals])
        deforms = np.array([r.aspect_deformation for r in vals])
        occs = np.array([r.occlusion_events for r in vals])

        tier_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "extreme": 0}
        for r in vals:
            tier_counts[r.difficulty_tier] = tier_counts.get(r.difficulty_tier, 0) + 1

        return {
            "num_sequences": len(vals),
            "mean_difficulty_score": round(float(scores.mean()), 4),
            "std_difficulty_score": round(float(scores.std()), 4),
            "mean_motion_speed": round(float(motions.mean()), 4),
            "mean_scale_volatility": round(float(scale_cvs.mean()), 4),
            "mean_aspect_deformation": round(float(deforms.mean()), 4),
            "mean_occlusion_events_per_100fr": round(float(occs.mean()), 3),
            "tier_distribution": tier_counts,
        }

    def to_markdown_table(self, results: Dict[str, DifficultyResult]) -> str:
        """Format difficulty results as a Markdown table sorted by score.

        Args:
            results: Output of :meth:`profile_benchmark`.

        Returns:
            Markdown table string with one row per sequence.
        """
        lines = [
            "| Sequence | Frames | Tier | Score | Motion | Scale CV | Deform | Occlusion/100fr |",
            "|----------|-------:|:-----|------:|-------:|---------:|-------:|----------------:|",
        ]
        for seq_name, r in sorted(results.items(), key=lambda x: -x[1].difficulty_score):
            lines.append(
                f"| {r.sequence_name} | {r.num_frames} | {r.difficulty_tier} "
                f"| {r.difficulty_score:.3f} | {r.motion_speed:.3f} "
                f"| {r.scale_volatility:.3f} | {r.aspect_deformation:.3f} "
                f"| {r.occlusion_events:.2f} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _compute_motion_speed(self, gts: np.ndarray) -> float:
        """Mean per-frame centre displacement normalised by mean box diagonal."""
        centres = gts[:, :2] + gts[:, 2:4] / 2.0   # (N, 2)
        displacements = np.sqrt(np.sum(np.diff(centres, axis=0) ** 2, axis=1))  # (N-1,)
        diagonals = np.sqrt(gts[:, 2] ** 2 + gts[:, 3] ** 2)  # (N,)
        mean_diag = float(diagonals.mean())
        if mean_diag < 1e-6:
            return 0.0
        return float(displacements.mean() / mean_diag)

    def _compute_scale_volatility(self, gts: np.ndarray) -> float:
        """Coefficient of variation (std/mean) of GT box area."""
        areas = gts[:, 2] * gts[:, 3]
        mean_area = float(areas.mean())
        if mean_area < 1e-6:
            return 0.0
        return float(areas.std() / mean_area)

    def _compute_aspect_deformation(self, gts: np.ndarray) -> float:
        """Std of GT aspect ratio (w/h) across the sequence."""
        widths = gts[:, 2]
        heights = np.maximum(gts[:, 3], 1e-6)
        aspect = widths / heights
        return float(aspect.std())

    def _compute_occlusion_events(self, gts: np.ndarray) -> float:
        """Abrupt area-drop events per 100 frames."""
        areas = gts[:, 2] * gts[:, 3]
        # Avoid division by zero for degenerate boxes.
        prev_areas = np.maximum(areas[:-1], 1e-6)
        area_drops = (areas[1:] - prev_areas) / prev_areas  # negative = area shrink
        events = int(np.sum(area_drops < -self._occlusion_drop))
        n_transitions = max(len(gts) - 1, 1)
        return float(events / n_transitions * 100.0)

    def _composite_score(
        self,
        motion: float,
        scale_cv: float,
        deform: float,
        occlusion: float,
    ) -> float:
        """Compute composite difficulty score in [0, 1]."""
        norm_motion = min(motion / self._CAP_MOTION, 1.0)
        norm_scale = min(scale_cv / self._CAP_SCALE_CV, 1.0)
        norm_deform = min(deform / self._CAP_DEFORM, 1.0)
        norm_occ = min(occlusion / self._CAP_OCCLUSION, 1.0)

        raw = (
            self._weights[0] * norm_motion
            + self._weights[1] * norm_scale
            + self._weights[2] * norm_deform
            + self._weights[3] * norm_occ
        )
        return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def profile_benchmark_result(
    benchmark_result: "BenchmarkResult",
    profiler: Optional[SequenceDifficultyProfiler] = None,
) -> Dict[str, DifficultyResult]:
    """Profile all sequences in *benchmark_result* using a shared profiler.

    Convenience wrapper around :meth:`SequenceDifficultyProfiler.profile_benchmark`.

    Args:
        benchmark_result: Full benchmark run with stored GT boxes.
        profiler:         Custom profiler instance, or ``None`` to use defaults.

    Returns:
        Dict mapping ``sequence_name → DifficultyResult``.
    """
    p = profiler or SequenceDifficultyProfiler()
    return p.profile_benchmark(benchmark_result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_tier(score: float) -> str:
    """Map a composite score in [0, 1] to a difficulty tier label."""
    for upper, label in DIFFICULTY_TIERS:
        if score < upper:
            return label
    return "extreme"
