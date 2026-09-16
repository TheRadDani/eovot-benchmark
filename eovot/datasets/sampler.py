"""Sequence difficulty scoring and stratified sampling for EOVOT datasets.

Large tracking benchmarks (GOT-10k: 180 test sequences, LaSOT: 280
sequences) are too slow to evaluate exhaustively during development.
This module provides:

1.  :class:`SequenceDifficultyScorer` — scores every sequence in a dataset
    by tracking difficulty, using only its ground-truth trajectory (no frame
    images are loaded, so scoring is fast even for large datasets).

2.  :class:`StratifiedSampler` — samples a representative subset of sequences
    that spans the full difficulty spectrum via equal-width percentile binning.

Difficulty components
---------------------
All components are normalised to ``[0, 1]``:

motion_speed
    Mean frame-to-frame centre displacement as a fraction of the mean target
    diagonal.  Fast-moving targets on a small scale are harder to follow.

scale_variability
    Coefficient of variation of bounding-box area.  Sequences with large
    scale changes challenge appearance models.

aspect_change
    Mean absolute change of the width/height ratio between consecutive frames.
    Deforming targets (people rotating, squatting) increase tracker difficulty.

target_smallness
    Median target area relative to a reference area (320×240 px by default).
    Small targets have low resolution and blurry appearance cues.

The composite difficulty score is a weighted mean of these four components.
Default weights are equal; researchers may supply custom weights.

Typical usage::

    from eovot.datasets.sampler import SequenceDifficultyScorer, StratifiedSampler

    dataset = OTBDataset("/data/OTB100")
    scorer = SequenceDifficultyScorer()
    scored = scorer.score_dataset(dataset)

    # Inspect the five hardest sequences
    for s in scored[-5:]:
        print(s.name, s.difficulty_score)

    # Sample 20 sequences spanning the full difficulty range
    sampler = StratifiedSampler(n_bins=5, seed=42)
    subset = sampler.sample(scored, n=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ScoredSequence:
    """A dataset sequence annotated with its difficulty score."""

    name: str
    """Sequence identifier."""

    difficulty_score: float
    """Composite difficulty score in ``[0, 1]``.  Higher = harder."""

    components: Dict[str, float] = field(default_factory=dict)
    """Individual difficulty component scores keyed by component name."""

    num_frames: int = 0
    """Number of frames in the sequence."""

    def __repr__(self) -> str:
        return (
            f"ScoredSequence(name={self.name!r}, "
            f"difficulty={self.difficulty_score:.3f}, "
            f"frames={self.num_frames})"
        )


class SequenceDifficultyScorer:
    """Score dataset sequences by tracking difficulty from their GT trajectories.

    Operates entirely on ``Sequence.ground_truth`` (``(N, 4)`` bbox arrays)
    so no frame images are loaded — scoring is fast even for large datasets.

    Args:
        weights: Optional dict mapping component name to a non-negative weight.
            Components: ``motion_speed``, ``scale_variability``,
            ``aspect_change``, ``target_smallness``.
            If ``None``, equal weights are used.  Values are normalised
            internally so they need not sum to 1.
        reference_area: Reference image area in pixels used to normalise
            ``target_smallness``.  Default: ``76800`` (320×240 px).

    Example::

        scorer = SequenceDifficultyScorer(
            weights={"motion_speed": 2.0, "target_smallness": 1.5}
        )
        scored = scorer.score_dataset(dataset)
    """

    _COMPONENTS = ("motion_speed", "scale_variability", "aspect_change", "target_smallness")

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        reference_area: float = 76800.0,
    ) -> None:
        if weights is None:
            weights = {c: 1.0 for c in self._COMPONENTS}
        total = sum(weights.get(c, 0.0) for c in self._COMPONENTS)
        if total <= 0:
            raise ValueError(
                "At least one difficulty component weight must be positive. "
                f"Components: {self._COMPONENTS}"
            )
        self._weights = {c: weights.get(c, 0.0) / total for c in self._COMPONENTS}
        self.reference_area = reference_area

    def score_sequence(self, gt: np.ndarray, name: str = "") -> ScoredSequence:
        """Score a single sequence from its ground-truth bbox array.

        Args:
            gt:   ``(N, 4)`` array of ground-truth boxes ``(x, y, w, h)``.
            name: Sequence identifier stored in the result.

        Returns:
            :class:`ScoredSequence` with all component scores populated.
        """
        gt = np.asarray(gt, dtype=np.float64)
        n = len(gt)

        components: Dict[str, float] = {c: 0.0 for c in self._COMPONENTS}

        if n >= 2:
            components["motion_speed"]     = self._motion_speed(gt)
            components["scale_variability"] = self._scale_variability(gt)
            components["aspect_change"]     = self._aspect_change(gt)
        if n >= 1:
            components["target_smallness"]  = self._target_smallness(gt)

        score = sum(self._weights[c] * components[c] for c in self._COMPONENTS)
        score = float(np.clip(score, 0.0, 1.0))

        return ScoredSequence(
            name=name,
            difficulty_score=score,
            components=dict(components),
            num_frames=n,
        )

    def score_dataset(self, dataset) -> List[ScoredSequence]:
        """Score every sequence in *dataset*, sorted by ascending difficulty.

        Only ``Sequence.ground_truth`` and ``Sequence.name`` are accessed;
        no frame images are loaded.

        Args:
            dataset: Any :class:`~eovot.datasets.base.BaseDataset` instance.

        Returns:
            List of :class:`ScoredSequence` objects sorted from easiest to
            hardest (ascending ``difficulty_score``).
        """
        scored: List[ScoredSequence] = []
        for i in range(len(dataset)):
            seq = dataset[i]
            scored.append(self.score_sequence(seq.ground_truth, name=seq.name))
        scored.sort(key=lambda s: s.difficulty_score)
        return scored

    # ------------------------------------------------------------------
    # Component computations
    # ------------------------------------------------------------------

    def _motion_speed(self, gt: np.ndarray) -> float:
        """Normalised mean centre-displacement; mapped to [0,1] via 1-exp(-speed)."""
        centres = gt[:, :2] + gt[:, 2:] / 2.0
        displacements = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        diags = np.sqrt(np.maximum(gt[:, 2] ** 2 + gt[:, 3] ** 2, 1e-6))
        mean_diag = float(diags.mean())
        norm_speed = float(displacements.mean()) / max(mean_diag, 1e-6)
        return float(1.0 - np.exp(-norm_speed))

    def _scale_variability(self, gt: np.ndarray) -> float:
        """Coefficient of variation of bbox areas, clipped to [0, 1]."""
        areas = gt[:, 2] * gt[:, 3]
        mean_area = float(areas.mean())
        if mean_area < 1e-6:
            return 0.0
        return float(np.clip(areas.std() / mean_area, 0.0, 1.0))

    def _aspect_change(self, gt: np.ndarray) -> float:
        """Mean absolute aspect-ratio change between consecutive frames, clipped to [0,1]."""
        w = np.maximum(gt[:, 2], 1e-6)
        h = np.maximum(gt[:, 3], 1e-6)
        ratios = w / h
        return float(np.clip(np.abs(np.diff(ratios)).mean(), 0.0, 1.0))

    def _target_smallness(self, gt: np.ndarray) -> float:
        """Inverted relative area: small targets score high (harder)."""
        areas = gt[:, 2] * gt[:, 3]
        median_area = float(np.median(areas))
        rel = median_area / self.reference_area
        return float(np.clip(1.0 - np.sqrt(rel), 0.0, 1.0))


class StratifiedSampler:
    """Sample a representative subset spanning the full difficulty spectrum.

    Divides a list of :class:`ScoredSequence` objects into equal-width
    difficulty bins and samples uniformly from each bin, ensuring the subset
    contains both easy and hard sequences.

    Args:
        n_bins: Number of difficulty strata.  Default: ``5`` (quintiles).
        seed:   Optional random seed for reproducibility.

    Example::

        sampler = StratifiedSampler(n_bins=5, seed=42)
        subset = sampler.sample(scored_sequences, n=20)
    """

    def __init__(self, n_bins: int = 5, seed: Optional[int] = None) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_bins = n_bins
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        scored: List[ScoredSequence],
        n: int,
    ) -> List[ScoredSequence]:
        """Sample *n* sequences from *scored* using stratified difficulty bins.

        Args:
            scored: List of :class:`ScoredSequence` objects (output of
                :meth:`SequenceDifficultyScorer.score_dataset`).
            n: Number of sequences to sample.  Clamped to ``len(scored)``.

        Returns:
            List of *n* :class:`ScoredSequence` objects drawn proportionally
            from each difficulty stratum, sorted by ascending difficulty.
        """
        if not scored:
            return []
        n = min(n, len(scored))
        if n == len(scored):
            return list(sorted(scored, key=lambda s: s.difficulty_score))

        sorted_seqs = sorted(scored, key=lambda s: s.difficulty_score)
        bins: List[List[ScoredSequence]] = [[] for _ in range(self.n_bins)]
        for s in sorted_seqs:
            bin_idx = min(int(s.difficulty_score * self.n_bins), self.n_bins - 1)
            bins[bin_idx].append(s)

        non_empty = [(i, b) for i, b in enumerate(bins) if b]
        n_active = len(non_empty)
        base = n // n_active
        extra = n % n_active

        selected: List[ScoredSequence] = []
        for rank, (_, bin_seqs) in enumerate(non_empty):
            quota = min(base + (1 if rank < extra else 0), len(bin_seqs))
            idxs = self._rng.choice(len(bin_seqs), size=quota, replace=False)
            selected.extend(bin_seqs[int(i)] for i in sorted(idxs))

        selected_names = {s.name for s in selected}
        remaining = [s for s in sorted_seqs if s.name not in selected_names]
        while len(selected) < n and remaining:
            selected.append(remaining.pop(0))

        selected.sort(key=lambda s: s.difficulty_score)
        return selected[:n]

    def sample_names(self, scored: List[ScoredSequence], n: int) -> List[str]:
        """Return sequence names of the sampled subset.

        Convenience wrapper around :meth:`sample`.

        Args:
            scored: Scored sequence list.
            n:      Number to sample.

        Returns:
            List of sequence name strings in ascending difficulty order.
        """
        return [s.name for s in self.sample(scored, n)]
