"""Sequence attribute analysis for EOVOT.

Automatically detects challenging tracking conditions from ground-truth
bounding boxes, enabling per-attribute performance breakdown in benchmark
reports.

Attributes detected
-------------------
- ``fast_motion``        — consecutive GT displacement > 20% of box diagonal.
- ``scale_variation``    — max/min GT box area ratio > 4.0 (≈ 2× scale).
- ``aspect_ratio_change``— max/min GT box aspect ratio > 2.0.
- ``out_of_view``        — GT box extends beyond frame boundaries.
- ``low_resolution``     — GT box area < 400 px² (< 20 × 20 target).
- ``motion_blur``        — per-frame velocity > 15 px/frame (proxy for blur).

Usage::

    from eovot.analysis import AttributeAnalyzer
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
    from eovot.trackers.mosse import MOSSETracker

    dataset  = SyntheticDataset(num_sequences=10, num_frames=100)
    analyzer = AttributeAnalyzer()

    # Tag every sequence in the dataset.
    tags = analyzer.tag_dataset(dataset, frame_size=(320, 240))

    # Run your tracker and get a BenchmarkResult.
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")

    # Break down performance by attribute.
    breakdown = analyzer.breakdown(result, tags)
    table     = analyzer.report_table({"MOSSE": breakdown})
    print(table)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..benchmark.engine import BenchmarkResult
from ..datasets.base import BaseDataset, Sequence


# ---------------------------------------------------------------------------
# Attribute flag names — single source of truth used in detection & reporting.
# ---------------------------------------------------------------------------
_ALL_ATTRS: Tuple[str, ...] = (
    "fast_motion",
    "scale_variation",
    "aspect_ratio_change",
    "out_of_view",
    "low_resolution",
    "motion_blur",
)

_ATTR_LABELS: Dict[str, str] = {
    "fast_motion": "Fast Motion",
    "scale_variation": "Scale Var.",
    "aspect_ratio_change": "AR Change",
    "out_of_view": "Out of View",
    "low_resolution": "Low Res.",
    "motion_blur": "Motion Blur",
}


@dataclass
class SequenceAttributes:
    """Per-sequence challenge attribute flags derived from ground-truth boxes.

    Each boolean flag indicates whether the corresponding tracking challenge
    is present in the sequence.  Use :meth:`active` for a compact list of
    flagged attributes.

    Attributes:
        name:               Sequence identifier (must match BenchmarkResult).
        fast_motion:        Target moves > 20% of its diagonal per frame.
        scale_variation:    Target area changes by more than 4× across the sequence.
        aspect_ratio_change:Target aspect ratio changes by more than 2×.
        out_of_view:        Target box extends beyond the frame boundary at some frame.
        low_resolution:     Target area falls below 400 px² in at least one frame.
        motion_blur:        Per-frame velocity exceeds 15 px/frame (blur proxy).
    """

    name: str
    fast_motion: bool = False
    scale_variation: bool = False
    aspect_ratio_change: bool = False
    out_of_view: bool = False
    low_resolution: bool = False
    motion_blur: bool = False

    def active(self) -> List[str]:
        """Return the names of all attributes flagged as ``True``."""
        return [a for a in _ALL_ATTRS if getattr(self, a)]


@dataclass
class AttributeBreakdown:
    """Per-attribute IoU summary for a single tracker's benchmark results.

    Attributes:
        tracker_name:      Name of the evaluated tracker.
        overall_iou:       Mean IoU across all matched sequences.
        overall_sequences: Total number of matched sequences.
        attribute_iou:     Mapping ``attribute_name → (mean_iou, num_sequences)``.
                           Only attributes with at least one matching sequence appear.
    """

    tracker_name: str
    overall_iou: float = 0.0
    overall_sequences: int = 0
    attribute_iou: Dict[str, Tuple[float, int]] = field(default_factory=dict)


class AttributeAnalyzer:
    """Detect per-sequence challenge attributes and break down tracker performance.

    Detection relies exclusively on the ground-truth bounding box trajectory;
    no raw video frames are required (except for ``out_of_view``, which
    additionally needs the frame dimensions).

    Args:
        fast_motion_threshold:  Minimum displacement relative to box diagonal
            (fraction) to flag fast motion.  Default: ``0.20``.
        scale_ratio_threshold:  Minimum max/min area ratio to flag scale
            variation.  Default: ``4.0`` (≈ 2× linear scale change).
        ar_ratio_threshold:     Minimum max/min aspect-ratio ratio to flag
            aspect-ratio change.  Default: ``2.0``.
        low_res_threshold:      Maximum box area (px²) to flag low resolution.
            Default: ``400.0`` (≈ 20 × 20 px target).
        motion_blur_threshold:  Velocity (px/frame) above which motion blur
            is expected.  Default: ``15.0``.

    Example::

        analyzer = AttributeAnalyzer(fast_motion_threshold=0.15)
        tags = analyzer.tag_dataset(my_dataset, frame_size=(1280, 720))
    """

    def __init__(
        self,
        fast_motion_threshold: float = 0.20,
        scale_ratio_threshold: float = 4.0,
        ar_ratio_threshold: float = 2.0,
        low_res_threshold: float = 400.0,
        motion_blur_threshold: float = 15.0,
    ) -> None:
        self.fast_motion_threshold = fast_motion_threshold
        self.scale_ratio_threshold = scale_ratio_threshold
        self.ar_ratio_threshold = ar_ratio_threshold
        self.low_res_threshold = low_res_threshold
        self.motion_blur_threshold = motion_blur_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_sequence(
        self,
        seq: Sequence,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> SequenceAttributes:
        """Detect challenge attributes in a single sequence.

        Args:
            seq:        Tracking sequence with ground-truth annotations.
            frame_size: ``(width, height)`` in pixels.  Required to detect
                ``out_of_view``; attribute is skipped when ``None``.

        Returns:
            :class:`SequenceAttributes` with a flag per detected challenge.
        """
        gt = seq.ground_truth  # (N, 4) — x, y, w, h
        attrs = SequenceAttributes(name=seq.name)

        if len(gt) < 2:
            return attrs

        x, y, w, h = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Displacement-based attributes
        dcx = np.diff(cx)
        dcy = np.diff(cy)
        displacements = np.sqrt(dcx ** 2 + dcy ** 2)

        diag = np.sqrt(w[:-1] ** 2 + h[:-1] ** 2) + 1e-6
        attrs.fast_motion = bool(np.any(displacements / diag > self.fast_motion_threshold))
        attrs.motion_blur = bool(np.any(displacements > self.motion_blur_threshold))

        # Scale variation: ratio of largest to smallest box area
        areas = w * h
        valid = areas[areas > 0]
        if len(valid) >= 2:
            attrs.scale_variation = bool(
                valid.max() / (valid.min() + 1e-6) > self.scale_ratio_threshold
            )

        # Aspect ratio change
        ar = w / (h + 1e-6)
        attrs.aspect_ratio_change = bool(
            ar.max() / (ar.min() + 1e-6) > self.ar_ratio_threshold
        )

        # Low resolution: target too small to reliably track
        attrs.low_resolution = bool(np.any(areas < self.low_res_threshold))

        # Out of view: requires the frame dimensions
        if frame_size is not None:
            fw, fh = frame_size
            oov = (x < 0) | (y < 0) | (x + w > fw) | (y + h > fh)
            attrs.out_of_view = bool(np.any(oov))

        return attrs

    def tag_dataset(
        self,
        dataset: BaseDataset,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> List[SequenceAttributes]:
        """Tag every sequence in *dataset* with challenge attributes.

        Args:
            dataset:    Any :class:`~eovot.datasets.base.BaseDataset` instance.
            frame_size: Passed through to :meth:`tag_sequence`.

        Returns:
            One :class:`SequenceAttributes` per sequence, in dataset order.
        """
        return [self.tag_sequence(seq, frame_size=frame_size) for seq in dataset]

    def breakdown(
        self,
        result: BenchmarkResult,
        tags: List[SequenceAttributes],
    ) -> AttributeBreakdown:
        """Compute per-attribute mean IoU for one tracker's benchmark result.

        Sequences are matched by name between *result* and *tags*.
        Unmatched sequences are silently skipped.

        Args:
            result: Benchmark result for a single tracker.
            tags:   Attribute tags produced by :meth:`tag_dataset`.

        Returns:
            :class:`AttributeBreakdown` with per-attribute statistics.
        """
        tag_map: Dict[str, SequenceAttributes] = {t.name: t for t in tags}
        iou_lists: Dict[str, List[float]] = {a: [] for a in _ALL_ATTRS}
        overall: List[float] = []

        for seq_res in result.sequence_results:
            miou = seq_res.mean_iou
            overall.append(miou)
            seq_attrs = tag_map.get(seq_res.sequence_name)
            if seq_attrs is None:
                continue
            for attr in _ALL_ATTRS:
                if getattr(seq_attrs, attr):
                    iou_lists[attr].append(miou)

        bd = AttributeBreakdown(tracker_name=result.tracker_name)
        bd.overall_iou = float(np.mean(overall)) if overall else 0.0
        bd.overall_sequences = len(overall)
        for attr, ious in iou_lists.items():
            if ious:
                bd.attribute_iou[attr] = (float(np.mean(ious)), len(ious))
        return bd

    def report_table(
        self,
        breakdowns: Dict[str, "AttributeBreakdown"],
    ) -> str:
        """Render a Markdown comparison table of trackers broken down by attribute.

        Args:
            breakdowns: Mapping ``tracker_name → AttributeBreakdown``.

        Returns:
            Multi-line Markdown string suitable for README or paper supplementary.

        Example output::

            | Tracker | Overall       | Fast Motion | Scale Var. | ... |
            |---------|---------------|:-----------:|:----------:|-----|
            | MOSSE   | 0.712 (10 seq)| 0.581 (4)   | 0.693 (3)  | ... |
        """
        header = "| Tracker | Overall |" + "".join(
            f" {_ATTR_LABELS[a]} |" for a in _ALL_ATTRS
        )
        divider = "|---------|---------|" + "".join(":-----------:|" for _ in _ALL_ATTRS)

        rows = [header, divider]
        for tracker_name, bd in breakdowns.items():
            cells = [f"| {tracker_name} | {bd.overall_iou:.3f} ({bd.overall_sequences}) |"]
            for attr in _ALL_ATTRS:
                if attr in bd.attribute_iou:
                    miou, n = bd.attribute_iou[attr]
                    cells.append(f" {miou:.3f} ({n}) |")
                else:
                    cells.append(" — |")
            rows.append("".join(cells))

        return "\n".join(rows)
