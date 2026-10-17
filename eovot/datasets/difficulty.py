"""Sequence difficulty analysis for EOVOT datasets.

Provides tools to characterize tracking sequences by difficulty attributes
derived entirely from ground-truth annotations — no frame decoding required.

Difficulty dimensions
---------------------
- **motion_speed**: Mean per-frame centroid displacement (px/frame).  High
  values indicate fast-moving targets that outpace slower trackers.
- **scale_variation**: Std-dev of relative bounding-box area change between
  consecutive frames.  High values indicate targets that zoom in/out.
- **aspect_ratio_change**: Std-dev of width/height ratio across all frames.
  Indicates non-rigid or rotating targets.
- **out_of_view_ratio**: Fraction of frames where the GT centroid is within
  ``boundary_margin`` pixels of the frame edge (or outside it).  Requires
  the caller to supply ``frame_size``; returns 0.0 otherwise.
- **overall_score**: Normalised composite score in [0, 1] (higher = harder),
  combining the four attributes with configurable weights via a soft-sigmoid
  saturation model so no single extreme outlier dominates.

Usage::

    from eovot.datasets.difficulty import DifficultyAnalyzer
    from eovot.datasets.synthetic import SyntheticDataset

    ds = SyntheticDataset(num_sequences=5, motion="random")
    analyzer = DifficultyAnalyzer(frame_size=(320, 240))
    for seq in ds:
        d = analyzer.analyze(seq)
        print(seq.name, f"score={d.overall_score:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence


@dataclass
class SequenceDifficulty:
    """Difficulty attributes for a single tracking sequence.

    All numeric fields are Python floats for easy JSON serialisation.
    """

    name: str
    num_frames: int
    motion_speed: float
    scale_variation: float
    aspect_ratio_change: float
    out_of_view_ratio: float
    overall_score: float

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dict representation."""
        return {
            "name": self.name,
            "num_frames": self.num_frames,
            "motion_speed": round(self.motion_speed, 4),
            "scale_variation": round(self.scale_variation, 4),
            "aspect_ratio_change": round(self.aspect_ratio_change, 4),
            "out_of_view_ratio": round(self.out_of_view_ratio, 4),
            "overall_score": round(self.overall_score, 4),
        }


class DifficultyAnalyzer:
    """Analyze tracking sequences to quantify difficulty from GT annotations.

    Analysis operates purely on ground-truth bounding boxes — no frame images
    are decoded — making it fast even on datasets with thousands of sequences.

    Args:
        frame_size: ``(width, height)`` in pixels, used to compute the
            out-of-view ratio.  Pass ``None`` to disable that attribute
            (it will always be 0.0).
        boundary_margin: Pixels from the frame edge within which a target
            centroid counts as "near boundary".  Default: ``10``.
        weights: Per-attribute weights for the composite score.  Unspecified
            keys fall back to their defaults.  Weights are normalised
            internally so only relative magnitudes matter.

    Example::

        analyzer = DifficultyAnalyzer(frame_size=(320, 240))
        difficulty = analyzer.analyze(sequence)
        print(f"overall difficulty = {difficulty.overall_score:.3f}")
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "motion_speed": 1.0,
        "scale_variation": 1.2,
        "aspect_ratio_change": 0.8,
        "out_of_view_ratio": 0.6,
    }

    # Saturation scales for the soft-sigmoid mapping (x -> 1 - exp(-x/scale)).
    # Chosen so that a "typical hard" value maps to ~0.63 and "extreme" → 1.
    _SIGMOID_SCALES: Dict[str, float] = {
        "motion_speed": 20.0,      # 20 px/frame is fast (near saturation)
        "scale_variation": 0.30,   # 30 % area std-dev is significant
        "aspect_ratio_change": 0.20,
    }

    def __init__(
        self,
        frame_size: Optional[Tuple[int, int]] = None,
        boundary_margin: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.frame_size = frame_size
        self.boundary_margin = boundary_margin
        w = dict(self._DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        self._weights = w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, seq: Sequence) -> SequenceDifficulty:
        """Compute difficulty attributes for *seq* from its ground truth.

        Args:
            seq: A :class:`~eovot.datasets.base.Sequence` with valid
                ``ground_truth`` of shape ``(N, 4)``.

        Returns:
            :class:`SequenceDifficulty` populated with computed attributes.
        """
        gt = seq.ground_truth  # (N, 4) — x, y, w, h

        ms = self._motion_speed(gt)
        sv = self._scale_variation(gt)
        arc = self._aspect_ratio_change(gt)
        oov = self._out_of_view_ratio(gt)
        score = self._composite_score(ms, sv, arc, oov)

        return SequenceDifficulty(
            name=seq.name,
            num_frames=len(gt),
            motion_speed=ms,
            scale_variation=sv,
            aspect_ratio_change=arc,
            out_of_view_ratio=oov,
            overall_score=score,
        )

    def analyze_dataset(self, dataset: BaseDataset) -> List[SequenceDifficulty]:
        """Analyze every sequence in *dataset* and return results in order."""
        return [self.analyze(dataset[i]) for i in range(len(dataset))]

    def rank(
        self,
        difficulties: List[SequenceDifficulty],
        ascending: bool = False,
    ) -> List[SequenceDifficulty]:
        """Sort *difficulties* by ``overall_score`` (hardest first by default).

        Args:
            ascending: If ``True``, easiest sequences appear first.

        Returns:
            A new sorted list; the original list is not modified.
        """
        return sorted(difficulties, key=lambda d: d.overall_score, reverse=not ascending)

    def filter(
        self,
        difficulties: List[SequenceDifficulty],
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> List[SequenceDifficulty]:
        """Return only sequences whose ``overall_score`` is in [*min_score*, *max_score*].

        Args:
            min_score: Inclusive lower bound.
            max_score: Inclusive upper bound.
        """
        return [d for d in difficulties if min_score <= d.overall_score <= max_score]

    def summary(self, difficulties: List[SequenceDifficulty]) -> Dict[str, float]:
        """Compute mean / std / min / max for each difficulty attribute.

        Returns an empty dict if *difficulties* is empty.
        """
        if not difficulties:
            return {}
        attrs = [
            "motion_speed",
            "scale_variation",
            "aspect_ratio_change",
            "out_of_view_ratio",
            "overall_score",
        ]
        stats: Dict[str, float] = {}
        for attr in attrs:
            vals = np.array([getattr(d, attr) for d in difficulties])
            stats[f"{attr}_mean"] = float(vals.mean())
            stats[f"{attr}_std"] = float(vals.std())
            stats[f"{attr}_min"] = float(vals.min())
            stats[f"{attr}_max"] = float(vals.max())
        return stats

    # ------------------------------------------------------------------
    # Attribute computations (private)
    # ------------------------------------------------------------------

    def _motion_speed(self, gt: np.ndarray) -> float:
        """Mean per-frame centroid displacement in pixels."""
        if len(gt) < 2:
            return 0.0
        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0
        dx = np.diff(cx)
        dy = np.diff(cy)
        return float(np.sqrt(dx ** 2 + dy ** 2).mean())

    def _scale_variation(self, gt: np.ndarray) -> float:
        """Std-dev of relative bounding-box area change between frames."""
        if len(gt) < 2:
            return 0.0
        areas = gt[:, 2] * gt[:, 3]
        safe_prev = np.maximum(areas[:-1], 1e-6)
        ratios = areas[1:] / safe_prev
        # Mask degenerate source boxes to avoid inflating variance.
        valid = areas[:-1] > 0
        if not valid.any():
            return 0.0
        return float(ratios[valid].std())

    def _aspect_ratio_change(self, gt: np.ndarray) -> float:
        """Std-dev of width/height ratio across all frames."""
        heights = gt[:, 3]
        valid = heights > 0
        if not valid.any():
            return 0.0
        ratios = gt[valid, 2] / np.maximum(heights[valid], 1e-6)
        return float(ratios.std())

    def _out_of_view_ratio(self, gt: np.ndarray) -> float:
        """Fraction of frames where the centroid is near or outside the frame."""
        if self.frame_size is None or len(gt) == 0:
            return 0.0
        W, H = self.frame_size
        m = self.boundary_margin
        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0
        near = (cx < m) | (cx > W - m) | (cy < m) | (cy > H - m)
        return float(near.mean())

    def _composite_score(
        self,
        motion_speed: float,
        scale_variation: float,
        aspect_ratio_change: float,
        out_of_view_ratio: float,
    ) -> float:
        """Weighted average of per-attribute scores, each mapped to [0, 1]."""

        def _soft_sigmoid(x: float, scale: float) -> float:
            return float(1.0 - np.exp(-x / max(scale, 1e-9)))

        attr_scores = {
            "motion_speed": _soft_sigmoid(
                motion_speed, self._SIGMOID_SCALES["motion_speed"]
            ),
            "scale_variation": _soft_sigmoid(
                scale_variation, self._SIGMOID_SCALES["scale_variation"]
            ),
            "aspect_ratio_change": _soft_sigmoid(
                aspect_ratio_change, self._SIGMOID_SCALES["aspect_ratio_change"]
            ),
            "out_of_view_ratio": float(np.clip(out_of_view_ratio, 0.0, 1.0)),
        }

        total_weight = sum(self._weights.values())
        composite = sum(
            self._weights.get(k, 1.0) * v for k, v in attr_scores.items()
        ) / total_weight
        return float(np.clip(composite, 0.0, 1.0))
