"""Sequence difficulty scoring from ground-truth bounding-box annotations.

Ranks tracking sequences by inherent challenge without running any tracker,
using six factors derived purely from ground-truth box sequences.

Factors (all normalised to [0, 1], higher = harder)
----------------------------------------------------
- **motion_speed** -- mean inter-frame displacement of the target centre,
  normalised by the frame diagonal (or the median box diagonal when no
  frame size is provided).
- **scale_change** -- coefficient of variation of target area; captures
  sequences where the target grows or shrinks dramatically.
- **aspect_ratio_change** -- coefficient of variation of target w/h ratio;
  high when the target deforms (e.g. a person sits down or a car turns).
- **small_target** -- fraction of frames where target area is below 5% of
  the frame area; small targets are notoriously hard for feature-based trackers.
- **out_of_view_risk** -- fraction of frames where the target box touches a
  frame boundary; proxies for partial or full out-of-view events.
- **deformation** -- 1 - mean(consecutive-frame IoU on GT boxes); high when
  the GT box shape changes substantially between frames.

All six factors are combined with configurable weights into a scalar
``difficulty_score`` in ``[0, 1]``.

Example::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.metrics.difficulty import score_dataset

    ds = SyntheticDataset(num_sequences=5, motion="random")
    report = score_dataset(ds, frame_size=(240, 320))
    print(report.to_markdown())
    print("Hardest:", report.hardest(1)[0].sequence_name)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class DifficultyFactors:
    """Per-sequence difficulty factors and aggregate score.

    All fields are in ``[0, 1]``; 1.0 = maximum difficulty for that factor.
    """

    motion_speed: float = 0.0
    scale_change: float = 0.0
    aspect_ratio_change: float = 0.0
    small_target: float = 0.0
    out_of_view_risk: float = 0.0
    deformation: float = 0.0
    difficulty_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return a ``{name: value}`` dict with values rounded to 4 dp."""
        return {k: round(float(v), 4) for k, v in asdict(self).items()}

    def __str__(self) -> str:
        return (
            f"DifficultyFactors("
            f"score={self.difficulty_score:.3f}, "
            f"motion={self.motion_speed:.3f}, "
            f"scale={self.scale_change:.3f}, "
            f"aspect={self.aspect_ratio_change:.3f}, "
            f"small={self.small_target:.3f}, "
            f"oov={self.out_of_view_risk:.3f}, "
            f"deform={self.deformation:.3f})"
        )


class SequenceDifficultyScorer:
    """Compute per-sequence difficulty factors from GT bounding boxes.

    Args:
        frame_size: ``(height, width)`` of video frames.  Required for
            ``small_target`` (percentage of frame area) and
            ``out_of_view_risk`` (boundary proximity).  When ``None`` those
            two factors use fallback estimates.
        weights: Mapping from factor keys to non-negative floats.  Valid keys:
            ``"motion"``, ``"scale"``, ``"aspect"``, ``"small"``,
            ``"oov"``, ``"deform"``.  Values are L1-normalised internally
            so they need not sum to 1.

    Raises:
        ValueError: If *weights* contains unknown keys or all weights are zero.

    Example::

        scorer = SequenceDifficultyScorer(frame_size=(480, 640))
        factors = scorer.score(sequence.ground_truth)
        print(factors.difficulty_score)
    """

    _WEIGHT_KEYS = ("motion", "scale", "aspect", "small", "oov", "deform")

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "motion": 0.25,
        "scale": 0.20,
        "aspect": 0.10,
        "small": 0.20,
        "oov": 0.15,
        "deform": 0.10,
    }

    def __init__(
        self,
        frame_size: Optional[Tuple[int, int]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.frame_size = frame_size
        raw = dict(weights or self._DEFAULT_WEIGHTS)
        unknown = set(raw) - set(self._WEIGHT_KEYS)
        if unknown:
            raise ValueError(f"Unknown weight keys: {sorted(unknown)!r}")
        total = sum(raw.values())
        if total <= 0.0:
            raise ValueError("Weights must sum to a positive value.")
        self._w: Dict[str, float] = {k: raw.get(k, 0.0) / total for k in self._WEIGHT_KEYS}

    def score(self, gt_boxes: np.ndarray) -> DifficultyFactors:
        """Compute difficulty factors from a sequence's ground-truth array.

        Args:
            gt_boxes: ``(N, 4)`` float array of boxes in ``(x, y, w, h)``
                format.  Must have at least 2 rows.

        Returns:
            :class:`DifficultyFactors` with all six factor values and the
            aggregate ``difficulty_score`` populated.

        Raises:
            ValueError: If *gt_boxes* shape is not ``(N, 4)`` with N >= 2.
        """
        gt = np.asarray(gt_boxes, dtype=np.float64)
        if gt.ndim != 2 or gt.shape[1] != 4:
            raise ValueError(
                f"gt_boxes must have shape (N, 4), got {gt.shape}"
            )
        if len(gt) < 2:
            raise ValueError(
                "Need at least 2 frames to compute difficulty factors."
            )

        motion = float(np.clip(self._motion_speed(gt), 0.0, 1.0))
        scale = float(np.clip(self._scale_change(gt), 0.0, 1.0))
        aspect = float(np.clip(self._aspect_ratio_change(gt), 0.0, 1.0))
        small = float(np.clip(self._small_target(gt), 0.0, 1.0))
        oov = float(np.clip(self._out_of_view_risk(gt), 0.0, 1.0))
        deform = float(np.clip(self._deformation(gt), 0.0, 1.0))

        agg = (
            self._w["motion"] * motion
            + self._w["scale"] * scale
            + self._w["aspect"] * aspect
            + self._w["small"] * small
            + self._w["oov"] * oov
            + self._w["deform"] * deform
        )

        return DifficultyFactors(
            motion_speed=round(motion, 4),
            scale_change=round(scale, 4),
            aspect_ratio_change=round(aspect, 4),
            small_target=round(small, 4),
            out_of_view_risk=round(oov, 4),
            deformation=round(deform, 4),
            difficulty_score=round(min(float(agg), 1.0), 4),
        )

    # ------------------------------------------------------------------
    # Individual factor computations (vectorised NumPy, no external deps)
    # ------------------------------------------------------------------

    def _motion_speed(self, gt: np.ndarray) -> float:
        cx = gt[:, 0] + gt[:, 2] / 2.0
        cy = gt[:, 1] + gt[:, 3] / 2.0
        disps = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        mean_disp = float(disps.mean())
        if self.frame_size is not None:
            h, w = self.frame_size
            ref = math.sqrt(h ** 2 + w ** 2) * 0.10  # 10% of diagonal = max
        else:
            ref = float(np.sqrt(gt[:, 2] ** 2 + gt[:, 3] ** 2).mean()) * 0.5
        return mean_disp / max(ref, 1e-6)

    def _scale_change(self, gt: np.ndarray) -> float:
        areas = gt[:, 2] * gt[:, 3]
        mean_area = float(areas.mean())
        if mean_area < 1.0:
            return 0.0
        cv = float(areas.std()) / mean_area  # coefficient of variation
        return cv / 0.5  # 50% CoV maps to difficulty=1

    def _aspect_ratio_change(self, gt: np.ndarray) -> float:
        valid = gt[:, 3] > 0
        if not valid.any():
            return 0.0
        ratios = gt[valid, 2] / np.maximum(gt[valid, 3], 1e-6)
        mean_r = float(ratios.mean())
        if mean_r < 1e-6:
            return 0.0
        cv = float(ratios.std()) / mean_r
        return cv / 0.30  # 30% CoV maps to difficulty=1

    def _small_target(self, gt: np.ndarray) -> float:
        areas = gt[:, 2] * gt[:, 3]
        if self.frame_size is not None:
            h, w = self.frame_size
            threshold = 0.05 * h * w
        else:
            threshold = 0.01 * float(np.median(areas))
        return float((areas < threshold).mean())

    def _out_of_view_risk(self, gt: np.ndarray) -> float:
        if self.frame_size is None:
            return 0.0
        h, w = self.frame_size
        margin = 5.0
        at_boundary = (
            (gt[:, 0] <= margin)
            | (gt[:, 1] <= margin)
            | ((gt[:, 0] + gt[:, 2]) >= (w - margin))
            | ((gt[:, 1] + gt[:, 3]) >= (h - margin))
        )
        return float(at_boundary.mean())

    def _deformation(self, gt: np.ndarray) -> float:
        """1 - mean(consecutive-frame IoU on GT boxes)."""
        prev = gt[:-1]
        curr = gt[1:]
        ix1 = np.maximum(prev[:, 0], curr[:, 0])
        iy1 = np.maximum(prev[:, 1], curr[:, 1])
        ix2 = np.minimum(prev[:, 0] + prev[:, 2], curr[:, 0] + curr[:, 2])
        iy2 = np.minimum(prev[:, 1] + prev[:, 3], curr[:, 1] + curr[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = prev[:, 2] * prev[:, 3] + curr[:, 2] * curr[:, 3] - inter
        valid = (
            (prev[:, 2] > 0) & (prev[:, 3] > 0)
            & (curr[:, 2] > 0) & (curr[:, 3] > 0)
            & (union > 0)
        )
        ious = np.where(valid, inter / np.maximum(union, 1e-9), 0.0)
        mean_iou = float(ious.mean()) if len(ious) > 0 else 1.0
        return 1.0 - mean_iou


@dataclass
class SequenceDifficultyEntry:
    """Difficulty score and factors for a single named sequence."""

    sequence_name: str
    factors: DifficultyFactors

    @property
    def difficulty_score(self) -> float:
        return self.factors.difficulty_score


class DifficultyReport:
    """Ranked sequence difficulty report.

    Sequences are sorted descending by ``difficulty_score`` on construction.

    Args:
        entries: One :class:`SequenceDifficultyEntry` per sequence.

    Example::

        report = DifficultyReport(entries)
        print(report.to_markdown())
        for entry in report.hardest(3):
            print(entry.sequence_name, entry.difficulty_score)
    """

    def __init__(self, entries: List[SequenceDifficultyEntry]) -> None:
        self.entries = sorted(
            entries, key=lambda e: e.difficulty_score, reverse=True
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SequenceDifficultyEntry]:
        return iter(self.entries)

    def __repr__(self) -> str:
        if not self.entries:
            return "DifficultyReport(empty)"
        top = self.entries[0]
        return (
            f"DifficultyReport({len(self.entries)} sequences, "
            f"hardest={top.sequence_name!r} score={top.difficulty_score:.3f})"
        )

    def hardest(self, n: int = 10) -> List[SequenceDifficultyEntry]:
        """Return the *n* hardest sequences (highest difficulty score)."""
        return self.entries[:n]

    def easiest(self, n: int = 10) -> List[SequenceDifficultyEntry]:
        """Return the *n* easiest sequences (lowest difficulty score), easiest first."""
        return list(reversed(self.entries[-n:]))

    def to_markdown(self) -> str:
        """Render the ranked difficulty table as a Markdown string."""
        header = (
            "| Rank | Sequence                 | Score "
            "| Motion | Scale | Aspect | Small |   OOV | Deform |"
        )
        sep = (
            "|------|--------------------------|-------"
            "|--------|-------|--------|-------|-------|--------|"
        )
        rows = [header, sep]
        for rank, entry in enumerate(self.entries, 1):
            f = entry.factors
            rows.append(
                f"| {rank:>4} | {entry.sequence_name:<24} "
                f"| {f.difficulty_score:.3f} "
                f"| {f.motion_speed:.3f}  "
                f"| {f.scale_change:.3f} "
                f"| {f.aspect_ratio_change:.3f}  "
                f"| {f.small_target:.3f} "
                f"| {f.out_of_view_risk:.3f} "
                f"| {f.deformation:.3f}  |"
            )
        return "\n".join(rows)

    def to_dict(self) -> List[Dict]:
        """Serialise all entries to a list of dicts for JSON export."""
        return [
            {"sequence": e.sequence_name, **e.factors.to_dict()}
            for e in self.entries
        ]


def score_dataset(
    dataset,
    frame_size: Optional[Tuple[int, int]] = None,
    scorer: Optional[SequenceDifficultyScorer] = None,
) -> DifficultyReport:
    """Score every sequence in *dataset* and return a ranked report.

    Args:
        dataset: Any :class:`~eovot.datasets.base.BaseDataset` instance.
        frame_size: ``(height, width)`` passed to
            :class:`SequenceDifficultyScorer` when *scorer* is ``None``.
        scorer: Optional pre-configured scorer; when given, *frame_size*
            is ignored.

    Returns:
        :class:`DifficultyReport` with all sequences ranked by difficulty.

    Example::

        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.metrics.difficulty import score_dataset

        report = score_dataset(SyntheticDataset(num_sequences=10), frame_size=(240, 320))
        print(report.to_markdown())
    """
    if scorer is None:
        scorer = SequenceDifficultyScorer(frame_size=frame_size)
    entries: List[SequenceDifficultyEntry] = []
    for seq in dataset:
        factors = scorer.score(seq.ground_truth)
        entries.append(SequenceDifficultyEntry(sequence_name=seq.name, factors=factors))
    return DifficultyReport(entries)
