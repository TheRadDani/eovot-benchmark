"""Sequence difficulty analysis for VOT benchmark datasets.

Standard VOT benchmarks (OTB, LaSOT, GOT-10k) annotate sequences with
challenge attributes (occlusion, fast motion, scale change, etc.) but those
labels are dataset-specific and require manual annotation.  This module
*derives* challenge attributes automatically from ground-truth bounding-box
sequences, enabling difficulty-stratified benchmarking on any dataset —
including custom or synthetic sequences — without relying on pre-existing
attribute labels.

Computed challenge dimensions
------------------------------
Scale change ratio (SCR)
    ``std(sqrt(area)) / mean(sqrt(area))`` — coefficient of variation of the
    square-root area.  High SCR signals that the target substantially changes
    size across the sequence, stressing fixed-template correlation filters.

Mean velocity (MV)
    ``mean(||Δcenter||) / mean_diagonal`` — frame-to-frame displacement
    normalised by the mean box diagonal, making the metric scale-invariant.
    High MV corresponds to fast-moving targets that easily drift out of a
    tracker's search window.

Aspect-ratio jitter (ARJ)
    ``std(w / h)`` — standard deviation of the bounding-box aspect ratio.
    High ARJ indicates target deformation (e.g. a person rotating or a
    vehicle changing orientation), which causes template mismatch in
    appearance-based trackers.

Degenerate frame ratio (DFR)
    Fraction of frames where the GT box has zero or negative area.  This
    covers both explicit ``(0, 0, 0, 0)`` annotations (fully occluded or
    out-of-view in some datasets) and pathological boxes.  High DFR means
    the tracker is expected to handle prolonged visibility loss.

Difficulty score
~~~~~~~~~~~~~~~~
A single scalar in ``[0, 1]`` that combines the four raw signals::

    D = clip(w_scr·SCR + w_mv·MV + w_arj·ARJ + w_dfr·DFR, 0, 1)

Weights: SCR=0.30, MV=0.30, ARJ=0.20, DFR=0.20.
Hard reference values for full score are 0.50 (SCR/MV) and 0.30 (ARJ/DFR).

Difficulty tiers
~~~~~~~~~~~~~~~~
- **easy**   — D < 0.33
- **medium** — 0.33 ≤ D < 0.66
- **hard**   — D ≥ 0.66

Challenge tags
~~~~~~~~~~~~~~
Each sequence is tagged with any of the following strings when the
corresponding metric exceeds its threshold:

=================  ==========  ==================================================
Tag                Default thr  Meaning
=================  ==========  ==================================================
``SCALE_CHANGE``   0.15         Target area varies by ≥15% (relative std)
``FAST_MOTION``    0.10         Mean velocity ≥ 10% of box diagonal per frame
``DEFORMATION``    0.12         Aspect-ratio std ≥ 0.12
``OCCLUSION``      0.05         ≥ 5% of GT boxes are zero-area (occluded)
=================  ==========  ==================================================

Typical usage
~~~~~~~~~~~~~
::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.datasets.difficulty import SequenceDifficultyAnalyzer, DifficultyFilteredDataset

    dataset  = SyntheticDataset(num_sequences=20, num_frames=100, motion="random")
    analyzer = SequenceDifficultyAnalyzer()

    difficulties = analyzer.analyze_dataset(dataset)
    print(analyzer.to_markdown_table(difficulties))

    # Keep only hard sequences for a stress test
    hard_ds = DifficultyFilteredDataset(dataset, tiers=["hard"], analyzer=analyzer)
    print(f"Hard sequences: {len(hard_ds)}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

DifficultyTier = Literal["easy", "medium", "hard"]

# Challenge tag constants
TAG_SCALE_CHANGE = "SCALE_CHANGE"
TAG_FAST_MOTION = "FAST_MOTION"
TAG_DEFORMATION = "DEFORMATION"
TAG_OCCLUSION = "OCCLUSION"


@dataclass
class SequenceDifficulty:
    """Difficulty profile for a single tracking sequence.

    All raw metrics are in ``[0, ∞)``.  The ``difficulty_score`` is normalised
    to ``[0, 1]``.  The ``tier`` and ``challenges`` fields are derived from
    thresholds configured on :class:`SequenceDifficultyAnalyzer`.

    Attributes:
        name: Sequence identifier.
        scale_change_ratio: Coefficient of variation of sqrt(box area).
        mean_velocity: Mean frame-to-frame displacement normalised by diagonal.
        aspect_ratio_jitter: Std of bounding-box aspect ratio (w / h).
        degenerate_frame_ratio: Fraction of frames with zero-area GT boxes.
        difficulty_score: Weighted composite scalar in ``[0, 1]``.
        tier: ``"easy"``, ``"medium"``, or ``"hard"``.
        challenges: List of applicable challenge tag strings.
        num_frames: Number of frames in the sequence.
    """

    name: str
    scale_change_ratio: float
    mean_velocity: float
    aspect_ratio_jitter: float
    degenerate_frame_ratio: float
    difficulty_score: float
    tier: DifficultyTier
    challenges: List[str] = field(default_factory=list)
    num_frames: int = 0

    def __str__(self) -> str:
        tags = ", ".join(self.challenges) if self.challenges else "none"
        return (
            f"SequenceDifficulty[{self.name}]  tier={self.tier}  "
            f"score={self.difficulty_score:.3f}  challenges=[{tags}]  "
            f"SCR={self.scale_change_ratio:.3f}  MV={self.mean_velocity:.3f}  "
            f"ARJ={self.aspect_ratio_jitter:.3f}  DFR={self.degenerate_frame_ratio:.3f}"
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "difficulty_score": round(self.difficulty_score, 4),
            "scale_change_ratio": round(self.scale_change_ratio, 4),
            "mean_velocity": round(self.mean_velocity, 4),
            "aspect_ratio_jitter": round(self.aspect_ratio_jitter, 4),
            "degenerate_frame_ratio": round(self.degenerate_frame_ratio, 4),
            "challenges": self.challenges,
            "num_frames": self.num_frames,
        }


class SequenceDifficultyAnalyzer:
    """Derive difficulty scores and challenge tags from GT bounding boxes.

    All analysis is performed on the ``(N, 4)`` ground-truth array associated
    with a :class:`~eovot.datasets.base.Sequence`, so no frame images need
    to be loaded.

    Args:
        scr_threshold: Scale-change-ratio threshold for the ``SCALE_CHANGE`` tag.
            Default ``0.15``.
        mv_threshold: Mean-velocity threshold for the ``FAST_MOTION`` tag.
            Default ``0.10``.
        arj_threshold: Aspect-ratio-jitter threshold for the ``DEFORMATION`` tag.
            Default ``0.12``.
        dfr_threshold: Degenerate-frame-ratio threshold for the ``OCCLUSION`` tag.
            Default ``0.05``.
        easy_max: Difficulty score below which a sequence is ``"easy"``.
            Default ``0.33``.
        hard_min: Difficulty score at or above which a sequence is ``"hard"``.
            Default ``0.66``.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.datasets.difficulty import SequenceDifficultyAnalyzer

        ds = SyntheticDataset(num_sequences=10)
        analyzer = SequenceDifficultyAnalyzer()
        results = analyzer.analyze_dataset(ds)
        print(analyzer.to_markdown_table(results))
    """

    # Normalisation references (values at which each raw metric contributes
    # its full weight to the difficulty score).
    _SCR_REF = 0.50
    _MV_REF = 0.50
    _ARJ_REF = 0.30
    _DFR_REF = 0.30

    # Composite score weights (must sum to 1.0).
    _W_SCR = 0.30
    _W_MV = 0.30
    _W_ARJ = 0.20
    _W_DFR = 0.20

    def __init__(
        self,
        scr_threshold: float = 0.15,
        mv_threshold: float = 0.10,
        arj_threshold: float = 0.12,
        dfr_threshold: float = 0.05,
        easy_max: float = 0.33,
        hard_min: float = 0.66,
    ) -> None:
        self.scr_threshold = scr_threshold
        self.mv_threshold = mv_threshold
        self.arj_threshold = arj_threshold
        self.dfr_threshold = dfr_threshold
        self.easy_max = easy_max
        self.hard_min = hard_min

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    def compute_scale_change_ratio(self, gt: np.ndarray) -> float:
        """Coefficient of variation of the square-root bounding-box area.

        Only valid (positive-area) frames contribute to the computation.

        Args:
            gt: ``(N, 4)`` array of GT boxes ``(x, y, w, h)``.

        Returns:
            SCR ≥ 0.  Returns ``0.0`` when fewer than 2 valid frames exist.
        """
        areas = gt[:, 2] * gt[:, 3]
        valid = areas[areas > 0]
        if len(valid) < 2:
            return 0.0
        sqrt_areas = np.sqrt(valid)
        mean_val = float(sqrt_areas.mean())
        if mean_val < 1e-9:
            return 0.0
        return float(sqrt_areas.std() / mean_val)

    def compute_mean_velocity(self, gt: np.ndarray) -> float:
        """Mean frame-to-frame centre displacement normalised by box diagonal.

        Only transitions between consecutive valid frames are counted.

        Args:
            gt: ``(N, 4)`` array of GT boxes ``(x, y, w, h)``.

        Returns:
            MV ≥ 0.  Returns ``0.0`` for sequences with fewer than 2 valid frames.
        """
        areas = gt[:, 2] * gt[:, 3]
        valid_mask = areas > 0
        valid_gt = gt[valid_mask]
        if len(valid_gt) < 2:
            return 0.0

        centers = valid_gt[:, :2] + valid_gt[:, 2:] / 2.0
        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        diagonals = np.sqrt(valid_gt[:, 2] ** 2 + valid_gt[:, 3] ** 2)
        mean_diag = float(diagonals.mean())
        if mean_diag < 1e-6:
            return 0.0
        return float(displacements.mean() / mean_diag)

    def compute_aspect_ratio_jitter(self, gt: np.ndarray) -> float:
        """Standard deviation of the bounding-box aspect ratio (w / h).

        Only valid (positive-area) frames contribute.

        Args:
            gt: ``(N, 4)`` array of GT boxes ``(x, y, w, h)``.

        Returns:
            ARJ ≥ 0.  Returns ``0.0`` when fewer than 2 valid frames exist.
        """
        valid = gt[(gt[:, 2] > 0) & (gt[:, 3] > 0)]
        if len(valid) < 2:
            return 0.0
        aspect_ratios = valid[:, 2] / (valid[:, 3] + 1e-9)
        return float(aspect_ratios.std())

    def compute_degenerate_frame_ratio(self, gt: np.ndarray) -> float:
        """Fraction of frames with zero or negative box area.

        Args:
            gt: ``(N, 4)`` array of GT boxes ``(x, y, w, h)``.

        Returns:
            DFR in ``[0, 1]``.  Returns ``0.0`` for empty arrays.
        """
        if len(gt) == 0:
            return 0.0
        areas = gt[:, 2] * gt[:, 3]
        return float((areas <= 0).sum() / len(gt))

    # ------------------------------------------------------------------
    # Composite scoring and tagging
    # ------------------------------------------------------------------

    def _difficulty_score(
        self,
        scr: float,
        mv: float,
        arj: float,
        dfr: float,
    ) -> float:
        """Compute weighted composite difficulty score in ``[0, 1]``."""
        score = (
            self._W_SCR * min(scr / self._SCR_REF, 1.0)
            + self._W_MV * min(mv / self._MV_REF, 1.0)
            + self._W_ARJ * min(arj / self._ARJ_REF, 1.0)
            + self._W_DFR * min(dfr / self._DFR_REF, 1.0)
        )
        return float(np.clip(score, 0.0, 1.0))

    def _tier(self, score: float) -> DifficultyTier:
        if score < self.easy_max:
            return "easy"
        if score < self.hard_min:
            return "medium"
        return "hard"

    def _challenge_tags(
        self,
        scr: float,
        mv: float,
        arj: float,
        dfr: float,
    ) -> List[str]:
        tags: List[str] = []
        if scr >= self.scr_threshold:
            tags.append(TAG_SCALE_CHANGE)
        if mv >= self.mv_threshold:
            tags.append(TAG_FAST_MOTION)
        if arj >= self.arj_threshold:
            tags.append(TAG_DEFORMATION)
        if dfr >= self.dfr_threshold:
            tags.append(TAG_OCCLUSION)
        return tags

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    def analyze(
        self,
        gt: np.ndarray,
        name: str = "",
    ) -> SequenceDifficulty:
        """Analyse one GT sequence and return its difficulty profile.

        Args:
            gt:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.
            name: Sequence identifier stored in the result.

        Returns:
            :class:`SequenceDifficulty` with all metrics populated.
        """
        gt = np.asarray(gt, dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(
                f"gt must be an (N, 4) array, got shape {gt.shape}."
            )

        scr = self.compute_scale_change_ratio(gt)
        mv = self.compute_mean_velocity(gt)
        arj = self.compute_aspect_ratio_jitter(gt)
        dfr = self.compute_degenerate_frame_ratio(gt)
        score = self._difficulty_score(scr, mv, arj, dfr)

        return SequenceDifficulty(
            name=name,
            scale_change_ratio=scr,
            mean_velocity=mv,
            aspect_ratio_jitter=arj,
            degenerate_frame_ratio=dfr,
            difficulty_score=score,
            tier=self._tier(score),
            challenges=self._challenge_tags(scr, mv, arj, dfr),
            num_frames=len(gt),
        )

    def analyze_dataset(
        self,
        dataset: BaseDataset,
    ) -> List[SequenceDifficulty]:
        """Analyse every sequence in a dataset without loading frame images.

        Iterates over :meth:`dataset.__getitem__` and reads only the
        ``ground_truth`` attribute from each :class:`~.base.Sequence`.

        Args:
            dataset: Any :class:`BaseDataset` implementation.

        Returns:
            List of :class:`SequenceDifficulty`, one per sequence, in the
            same order as ``dataset[0], dataset[1], …``.
        """
        results: List[SequenceDifficulty] = []
        for idx in range(len(dataset)):
            seq: Sequence = dataset[idx]
            results.append(self.analyze(seq.ground_truth, name=seq.name))
        return results

    def filter_by_tier(
        self,
        difficulties: List[SequenceDifficulty],
        tiers: List[DifficultyTier],
    ) -> List[int]:
        """Return dataset indices whose difficulty tier is in *tiers*.

        Args:
            difficulties: Output of :meth:`analyze_dataset`.
            tiers: List of accepted tier strings (``"easy"``, ``"medium"``,
                ``"hard"``).

        Returns:
            List of integer indices into the original dataset.
        """
        return [
            i for i, d in enumerate(difficulties) if d.tier in tiers
        ]

    def filter_by_challenge(
        self,
        difficulties: List[SequenceDifficulty],
        challenges: List[str],
        require_all: bool = False,
    ) -> List[int]:
        """Return dataset indices matching the given challenge tags.

        Args:
            difficulties: Output of :meth:`analyze_dataset`.
            challenges: Challenge tags to match, e.g. ``["SCALE_CHANGE", "FAST_MOTION"]``.
            require_all: If ``True``, a sequence must possess *all* listed tags
                to be included.  If ``False`` (default), *any* matching tag
                is sufficient.

        Returns:
            List of integer indices into the original dataset.
        """
        ch_set = set(challenges)
        out = []
        for i, d in enumerate(difficulties):
            seq_tags = set(d.challenges)
            if require_all:
                if ch_set.issubset(seq_tags):
                    out.append(i)
            else:
                if ch_set & seq_tags:
                    out.append(i)
        return out

    def dataset_summary(
        self,
        difficulties: List[SequenceDifficulty],
    ) -> Dict:
        """Aggregate statistics across all analysed sequences.

        Args:
            difficulties: Output of :meth:`analyze_dataset`.

        Returns:
            Dict with count-by-tier, mean difficulty, and tag frequency.
        """
        if not difficulties:
            return {}

        tier_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        tag_counts: Dict[str, int] = {
            TAG_SCALE_CHANGE: 0,
            TAG_FAST_MOTION: 0,
            TAG_DEFORMATION: 0,
            TAG_OCCLUSION: 0,
        }
        for d in difficulties:
            tier_counts[d.tier] += 1
            for tag in d.challenges:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        scores = [d.difficulty_score for d in difficulties]
        return {
            "num_sequences": len(difficulties),
            "mean_difficulty": round(float(np.mean(scores)), 4),
            "std_difficulty": round(float(np.std(scores)), 4),
            "tier_counts": tier_counts,
            "challenge_counts": tag_counts,
            "mean_scr": round(float(np.mean([d.scale_change_ratio for d in difficulties])), 4),
            "mean_mv": round(float(np.mean([d.mean_velocity for d in difficulties])), 4),
            "mean_arj": round(float(np.mean([d.aspect_ratio_jitter for d in difficulties])), 4),
            "mean_dfr": round(float(np.mean([d.degenerate_frame_ratio for d in difficulties])), 4),
        }

    def to_markdown_table(
        self,
        difficulties: List[SequenceDifficulty],
        top_n: Optional[int] = None,
        sort_by: str = "difficulty_score",
        ascending: bool = False,
    ) -> str:
        """Format difficulty profiles as a Markdown table.

        Args:
            difficulties: Output of :meth:`analyze_dataset`.
            top_n: If set, include only the first ``top_n`` rows after sorting.
            sort_by: Column to sort by.  One of ``"difficulty_score"``,
                ``"scale_change_ratio"``, ``"mean_velocity"``,
                ``"aspect_ratio_jitter"``, ``"degenerate_frame_ratio"``.
                Default ``"difficulty_score"``.
            ascending: Sort order.  Default ``False`` (hardest first).

        Returns:
            Multi-line Markdown string ready for embedding in reports.
        """
        valid_keys = {
            "difficulty_score", "scale_change_ratio",
            "mean_velocity", "aspect_ratio_jitter", "degenerate_frame_ratio",
        }
        if sort_by not in valid_keys:
            sort_by = "difficulty_score"

        sorted_ds = sorted(
            difficulties,
            key=lambda d: getattr(d, sort_by),
            reverse=not ascending,
        )
        if top_n is not None:
            sorted_ds = sorted_ds[:top_n]

        lines = [
            "| Rank | Sequence | Tier | Score | SCR | MV | ARJ | DFR | Challenges |",
            "|------|----------|------|------:|----:|---:|----:|----:|-----------|",
        ]
        for rank, d in enumerate(sorted_ds, start=1):
            tags = " ".join(d.challenges) if d.challenges else "—"
            lines.append(
                f"| {rank} | {d.name} | {d.tier} "
                f"| {d.difficulty_score:.3f} "
                f"| {d.scale_change_ratio:.3f} "
                f"| {d.mean_velocity:.3f} "
                f"| {d.aspect_ratio_jitter:.3f} "
                f"| {d.degenerate_frame_ratio:.3f} "
                f"| {tags} |"
            )
        return "\n".join(lines)


class DifficultyFilteredDataset(BaseDataset):
    """Dataset wrapper that exposes only sequences matching given difficulty tiers.

    Wraps any :class:`~eovot.datasets.base.BaseDataset` and filters it to a
    subset of sequences whose computed difficulty tier is in *tiers*.  The
    wrapped dataset's sequences are re-indexed from zero so the result is a
    drop-in replacement for the original dataset.

    This is useful for:
    - **Stress testing** — evaluate only hard sequences.
    - **Ablation studies** — compare trackers on different difficulty strata.
    - **Curriculum experiments** — progressively increase difficulty.

    Args:
        dataset: Any :class:`BaseDataset` to wrap.
        tiers: Accepted difficulty tiers.  Sequences not in these tiers are
            excluded.  Valid values: ``"easy"``, ``"medium"``, ``"hard"``.
        analyzer: :class:`SequenceDifficultyAnalyzer` instance to use.
            If ``None``, a default analyzer with default thresholds is created.

    Raises:
        ValueError: If ``tiers`` is empty or contains invalid tier names.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.datasets.difficulty import DifficultyFilteredDataset

        base = SyntheticDataset(num_sequences=20, motion="random")
        hard_ds = DifficultyFilteredDataset(base, tiers=["hard"])
        print(f"Hard sequences: {len(hard_ds)} / {len(base)}")
    """

    _VALID_TIERS = {"easy", "medium", "hard"}

    def __init__(
        self,
        dataset: BaseDataset,
        tiers: List[DifficultyTier],
        analyzer: Optional[SequenceDifficultyAnalyzer] = None,
    ) -> None:
        if not tiers:
            raise ValueError("tiers must not be empty.")
        invalid = set(tiers) - self._VALID_TIERS
        if invalid:
            raise ValueError(
                f"Invalid tier(s): {invalid}. Must be one of {self._VALID_TIERS}."
            )

        if analyzer is None:
            analyzer = SequenceDifficultyAnalyzer()

        self._dataset = dataset
        self.tiers = list(tiers)
        self.analyzer = analyzer

        difficulties = analyzer.analyze_dataset(dataset)
        self._indices = analyzer.filter_by_tier(difficulties, tiers)
        self._difficulties: Dict[int, SequenceDifficulty] = {
            self._indices[k]: difficulties[self._indices[k]]
            for k in range(len(self._indices))
        }

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(
                f"Index {idx} out of range [0, {len(self._indices)})."
            )
        return self._dataset[self._indices[idx]]

    def get_difficulty(self, idx: int) -> SequenceDifficulty:
        """Return the difficulty profile for sequence ``idx`` in this dataset.

        Args:
            idx: Index in the filtered dataset (not the original dataset index).
        """
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(
                f"Index {idx} out of range [0, {len(self._indices)})."
            )
        return self._difficulties[self._indices[idx]]

    def __repr__(self) -> str:
        return (
            f"DifficultyFilteredDataset("
            f"tiers={self.tiers!r}, "
            f"n_sequences={len(self)}/{len(self._dataset)})"
        )
