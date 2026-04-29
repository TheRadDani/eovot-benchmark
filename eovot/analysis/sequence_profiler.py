"""Sequence difficulty profiler for EOVOT datasets.

Characterises each tracking sequence by its intrinsic difficulty along five
dimensions derived solely from ground-truth bounding box trajectories — no
tracker inference is needed.

Difficulty dimensions
---------------------
* **Scale change** (``target_area_cv``) — coefficient of variation of the
  target area; high means the target grows or shrinks drastically.
* **Target size** (``target_area_mean``) — small targets are harder; expressed
  as a fraction of total frame area so the metric is resolution-independent.
* **Motion magnitude** (``motion_mean``) — mean frame-to-frame centre
  displacement normalised by the frame diagonal.
* **Motion irregularity** (``motion_cv``) — coefficient of variation of
  per-frame displacement; high means the target moves erratically.
* **Deformation** (``aspect_ratio_std``) — standard deviation of the target's
  width/height ratio; high means the target changes shape (e.g. a person
  turning sideways).

A weighted composite ``difficulty_score ∈ [0, 1]`` combines all five
dimensions: 0 = trivially easy, 1 = maximally hard.

Typical usage::

    from eovot.analysis.sequence_profiler import SequenceDifficultyProfiler
    from eovot.datasets.base import OTBDataset

    dataset  = OTBDataset("/data/OTB100")
    profiler = SequenceDifficultyProfiler()

    difficulties = profiler.profile_dataset(dataset)
    ranked       = profiler.sort_by_difficulty(difficulties)   # hardest first
    stats        = profiler.summary_stats(difficulties)

    print(f"Hardest sequence: {ranked[0].sequence_name}  "
          f"(score={ranked[0].difficulty_score:.3f})")
    print(stats)

References
----------
* Wu et al., "Object Tracking Benchmark." TPAMI 2015 — OTB attribute taxonomy.
* Müller et al., "TrackingNet: A Large-Scale Dataset …" ECCV 2018 — motion
  and size analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..datasets.base import BaseDataset, Sequence

# ---------------------------------------------------------------------------
# Composite score weights — must sum to 1.0
# ---------------------------------------------------------------------------
_W_SCALE_CHANGE   = 0.30
_W_MOTION_MAG     = 0.35
_W_MOTION_CV      = 0.20
_W_DEFORMATION    = 0.15

# Normalisation constants for mapping raw metrics to [0, 1]
_NORM_MOTION_MAG  = 0.20   # frame-diagonal fraction above which motion is "fast"
_NORM_AR_STD      = 0.50   # aspect-ratio std above which deformation is "severe"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class SequenceDifficulty:
    """Difficulty characterisation of a single tracking sequence.

    All spatial measurements are normalised by frame dimensions so that
    metrics are comparable across datasets with different resolutions.

    Attributes:
        sequence_name: Identifier of the sequence.
        num_frames:    Number of frames (= rows in ground-truth array).
        target_area_mean: Mean target area as a fraction of total frame area.
            Small values (< 0.01) indicate tiny-target challenge.
        target_area_cv: Coefficient of variation of target area in [0, ∞).
            High values indicate significant scale change.
        motion_mean: Mean per-frame centre displacement normalised by frame
            diagonal.  Values > 0.05 are considered fast-motion.
        motion_cv:  Coefficient of variation of per-frame displacement.
            High values indicate erratic / unpredictable motion.
        aspect_ratio_std: Standard deviation of the target width/height ratio.
            Values > 0.2 indicate significant shape deformation.
        difficulty_score: Weighted composite difficulty in [0, 1].
            Higher → harder for most standard trackers.
    """

    sequence_name: str
    num_frames: int

    target_area_mean: float
    target_area_cv: float
    motion_mean: float
    motion_cv: float
    aspect_ratio_std: float
    difficulty_score: float

    def __str__(self) -> str:
        return (
            f"SequenceDifficulty({self.sequence_name!r}  "
            f"frames={self.num_frames}  "
            f"score={self.difficulty_score:.3f}  "
            f"area={self.target_area_mean:.4f}  "
            f"motion={self.motion_mean:.4f})"
        )

    def to_dict(self) -> Dict[str, object]:
        """Serialise to a plain dict suitable for JSON export or pandas."""
        return {
            "sequence_name":   self.sequence_name,
            "num_frames":      self.num_frames,
            "target_area_mean": round(self.target_area_mean, 6),
            "target_area_cv":  round(self.target_area_cv,   4),
            "motion_mean":     round(self.motion_mean,      6),
            "motion_cv":       round(self.motion_cv,        4),
            "aspect_ratio_std": round(self.aspect_ratio_std, 4),
            "difficulty_score": round(self.difficulty_score, 4),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_frame_size(sequence: Sequence) -> tuple:
    """Return ``(width, height)`` by loading only the first frame."""
    frame = next(iter(sequence))
    h, w = frame.shape[:2]
    return w, h


def _compute_metrics(
    gt: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Dict[str, float]:
    """Compute raw difficulty metrics from a GT array and frame dimensions.

    Args:
        gt:      Ground-truth array, shape ``(N, 4)`` in ``(x, y, w, h)`` format.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Dict with keys ``target_area_mean``, ``target_area_cv``, ``motion_mean``,
        ``motion_cv``, ``aspect_ratio_std``.
    """
    n = len(gt)
    frame_area = float(frame_w * frame_h)
    frame_diag = float(np.sqrt(frame_w ** 2 + frame_h ** 2))

    # ---- target area (normalised) ----
    areas = gt[:, 2] * gt[:, 3]                          # w * h per frame
    valid = areas[areas > 0]
    if len(valid) == 0:
        return _zero_metrics()

    target_area_mean = float(valid.mean()) / frame_area
    mean_area = float(valid.mean())
    target_area_cv = float(valid.std() / mean_area) if mean_area > 0 else 0.0

    # ---- motion (normalised by frame diagonal) ----
    cx = gt[:, 0] + gt[:, 2] / 2.0
    cy = gt[:, 1] + gt[:, 3] / 2.0
    if n > 1:
        displacements = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / frame_diag
        motion_mean = float(displacements.mean())
        motion_cv   = float(displacements.std() / motion_mean) if motion_mean > 0 else 0.0
    else:
        motion_mean = 0.0
        motion_cv   = 0.0

    # ---- aspect ratio deformation ----
    valid_boxes = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    if valid_boxes.sum() > 1:
        ars = gt[valid_boxes, 2] / gt[valid_boxes, 3]
        aspect_ratio_std = float(ars.std())
    else:
        aspect_ratio_std = 0.0

    return {
        "target_area_mean": target_area_mean,
        "target_area_cv":   target_area_cv,
        "motion_mean":      motion_mean,
        "motion_cv":        motion_cv,
        "aspect_ratio_std": aspect_ratio_std,
    }


def _zero_metrics() -> Dict[str, float]:
    return {
        "target_area_mean": 0.0,
        "target_area_cv":   0.0,
        "motion_mean":      0.0,
        "motion_cv":        0.0,
        "aspect_ratio_std": 0.0,
    }


def _composite_score(m: Dict[str, float]) -> float:
    """Map raw difficulty metrics to a scalar score in [0, 1].

    Each component is independently clipped to [0, 1] before weighting so
    extreme outliers in one dimension don't dominate the overall score.

    Weights::

        scale_change  0.30   (target_area_cv)
        motion_mag    0.35   (motion_mean / _NORM_MOTION_MAG)
        motion_cv     0.20   (motion_cv, capped at 1)
        deformation   0.15   (aspect_ratio_std / _NORM_AR_STD)
    """
    scale_ch  = min(1.0, m["target_area_cv"])
    motion_mg = min(1.0, m["motion_mean"] / _NORM_MOTION_MAG)
    motion_cv = min(1.0, m["motion_cv"])
    deform    = min(1.0, m["aspect_ratio_std"] / _NORM_AR_STD)

    return float(
        _W_SCALE_CHANGE * scale_ch
        + _W_MOTION_MAG * motion_mg
        + _W_MOTION_CV  * motion_cv
        + _W_DEFORMATION * deform
    )


# ---------------------------------------------------------------------------
# Public profiler class
# ---------------------------------------------------------------------------

class SequenceDifficultyProfiler:
    """Profile the difficulty of tracking sequences in a dataset.

    Computes five difficulty dimensions per sequence from ground-truth box
    trajectories alone — no tracker inference is required.  Results can be
    used to:

    * Rank sequences from hardest to easiest for ablation studies.
    * Correlate per-sequence tracker accuracy with difficulty dimensions.
    * Characterise dataset difficulty distributions for paper tables.

    Args:
        verbose: If ``True``, print per-sequence progress to stdout.

    Example::

        profiler    = SequenceDifficultyProfiler()
        diffs       = profiler.profile_dataset(dataset)
        ranked      = profiler.sort_by_difficulty(diffs)           # hardest first
        stats       = profiler.summary_stats(diffs)

        hard = [d for d in diffs if d.difficulty_score > 0.6]
        easy = [d for d in diffs if d.difficulty_score < 0.2]
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def profile_sequence(self, sequence: Sequence) -> SequenceDifficulty:
        """Profile the difficulty of a single sequence.

        Reads only the first frame to determine resolution; all remaining
        metrics come from the ground-truth array.

        Args:
            sequence: :class:`~eovot.datasets.base.Sequence` to characterise.

        Returns:
            :class:`SequenceDifficulty` with all five dimension metrics and
            the composite ``difficulty_score``.
        """
        gt       = np.asarray(sequence.ground_truth, dtype=np.float64)
        if gt.ndim == 1:
            gt = gt[np.newaxis, :]

        frame_w, frame_h = _first_frame_size(sequence)
        raw_metrics      = _compute_metrics(gt, frame_w, frame_h)
        score            = _composite_score(raw_metrics)

        return SequenceDifficulty(
            sequence_name=sequence.name,
            num_frames=len(gt),
            difficulty_score=round(score, 6),
            **{k: round(v, 8) for k, v in raw_metrics.items()},
        )

    def profile_dataset(
        self,
        dataset: BaseDataset,
        max_sequences: Optional[int] = None,
    ) -> List[SequenceDifficulty]:
        """Profile all sequences in *dataset*.

        Args:
            dataset:       Dataset whose sequences to profile.
            max_sequences: Analyse only the first *N* sequences.
                           ``None`` (default) profiles all.

        Returns:
            List of :class:`SequenceDifficulty`, one per sequence, in
            dataset order.
        """
        n = min(len(dataset), max_sequences) if max_sequences is not None else len(dataset)
        results: List[SequenceDifficulty] = []
        for i in range(n):
            seq  = dataset[i]
            diff = self.profile_sequence(seq)
            results.append(diff)
            if self.verbose:
                print(f"  [{i + 1:>3}/{n}] {diff}")
        return results

    def sort_by_difficulty(
        self,
        difficulties: List[SequenceDifficulty],
        descending: bool = True,
    ) -> List[SequenceDifficulty]:
        """Return a copy of *difficulties* sorted by composite score.

        Args:
            difficulties: Output of :meth:`profile_dataset` or a curated list.
            descending:   ``True`` (default) → hardest sequences first.

        Returns:
            New sorted list; the original is not modified.
        """
        return sorted(
            difficulties,
            key=lambda d: d.difficulty_score,
            reverse=descending,
        )

    def summary_stats(
        self,
        difficulties: List[SequenceDifficulty],
    ) -> Dict[str, object]:
        """Compute aggregate statistics over all profiled sequences.

        Useful for characterising overall dataset difficulty in a paper table.

        Args:
            difficulties: Output of :meth:`profile_dataset`.

        Returns:
            Dict with ``num_sequences`` and per-dimension dicts each containing
            ``mean``, ``std``, ``min``, ``max``.  Returns ``{}`` for an empty list.

        Example output::

            {
                "num_sequences": 100,
                "difficulty_score": {"mean": 0.31, "std": 0.14, "min": 0.03, "max": 0.89},
                "motion_mean":      {"mean": 0.021, "std": 0.018},
                "target_area_mean": {"mean": 0.032, "std": 0.027},
            }
        """
        if not difficulties:
            return {}

        def _stat(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean": round(float(arr.mean()), 6),
                "std":  round(float(arr.std()),  6),
                "min":  round(float(arr.min()),  6),
                "max":  round(float(arr.max()),  6),
            }

        scores  = np.array([d.difficulty_score  for d in difficulties])
        motions = np.array([d.motion_mean        for d in difficulties])
        areas   = np.array([d.target_area_mean   for d in difficulties])
        scale   = np.array([d.target_area_cv     for d in difficulties])
        deform  = np.array([d.aspect_ratio_std   for d in difficulties])

        return {
            "num_sequences":    len(difficulties),
            "difficulty_score": _stat(scores),
            "motion_mean":      _stat(motions),
            "target_area_mean": _stat(areas),
            "target_area_cv":   _stat(scale),
            "aspect_ratio_std": _stat(deform),
        }
