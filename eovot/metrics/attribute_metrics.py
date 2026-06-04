"""Per-attribute performance breakdown for challenge-conditioned evaluation.

Provides :class:`AttributeMetricsEngine` which aggregates tracker accuracy
and efficiency metrics grouped by sequence challenge attributes (fast motion,
scale variation, etc.).

This is a standard component of research-grade VOT benchmarks (OTB-100,
LaSOT, GOT-10k).  Per-attribute breakdown answers questions like:

- "Does tracker X degrade specifically on fast-motion sequences?"
- "Which tracker handles scale variation best at < 30 FPS?"

These insights guide model design decisions and deployment choices for
resource-constrained edge devices.

Usage::

    from eovot.metrics.attribute_metrics import AttributeMetricsEngine
    from eovot.datasets.attributes import SequenceAttributeAnalyzer, SequenceAttributes
    import numpy as np

    analyzer = SequenceAttributeAnalyzer()
    gt_map = {"seq0": np.array([...]), "seq1": np.array([...])}
    attrs_map = {name: analyzer.analyze(gt) for name, gt in gt_map.items()}

    engine = AttributeMetricsEngine()
    results = engine.compute(
        sequence_names=list(gt_map.keys()),
        attributes_map=attrs_map,
        ious_map={"seq0": np.array([0.8, 0.75, ...]), "seq1": np.array([...])},
        fps_map={"seq0": 120.0, "seq1": 95.0},
    )
    for attr, perf in results.items():
        print(perf)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..datasets.attributes import SequenceAttributes


# Canonical attribute ordering for consistent table output.
ALL_ATTRIBUTES: List[str] = [
    "fast_motion",
    "scale_variation",
    "low_resolution",
    "aspect_ratio_change",
    "out_of_view",
    "partial_occlusion",
]


@dataclass
class AttributePerformance:
    """Tracker performance statistics for a single challenge attribute.

    Attributes:
        attribute: Challenge attribute name (e.g. ``"fast_motion"``).
        sequence_count: Number of sequences that exhibit this attribute.
        mean_iou: Mean IoU across all frames in attribute-positive sequences.
        mean_fps: Mean FPS across attribute-positive sequences.
        success_auc: Mean success-curve AUC across attribute-positive sequences.
    """

    attribute: str
    sequence_count: int
    mean_iou: float
    mean_fps: float
    success_auc: float

    def __repr__(self) -> str:
        return (
            f"AttributePerformance({self.attribute!r}: "
            f"n={self.sequence_count}, "
            f"mIoU={self.mean_iou:.3f}, "
            f"AUC={self.success_auc:.3f}, "
            f"FPS={self.mean_fps:.1f})"
        )

    def to_dict(self) -> Dict:
        """Serialise to a plain dict (JSON-compatible)."""
        return {
            "attribute": self.attribute,
            "sequence_count": self.sequence_count,
            "mean_iou": round(self.mean_iou, 4),
            "success_auc": round(self.success_auc, 4),
            "mean_fps": round(self.mean_fps, 2),
        }


class AttributeMetricsEngine:
    """Aggregate tracker performance broken down by sequence challenge attribute.

    For each attribute, sequences that exhibit that attribute form a
    sub-benchmark.  Metrics are computed only over those sequences, giving
    a fine-grained view of tracker strengths and weaknesses.

    Note:
        A single sequence may contribute to multiple attribute groups
        (e.g. both ``fast_motion`` and ``out_of_view``).

    Args:
        success_thresholds: IoU thresholds for the success curve.  Defaults
            to 101 linearly-spaced points in ``[0, 1]``.
    """

    def __init__(
        self,
        success_thresholds: Optional[np.ndarray] = None,
    ) -> None:
        self._thresholds = (
            success_thresholds
            if success_thresholds is not None
            else np.linspace(0.0, 1.0, 101)
        )

    def compute(
        self,
        sequence_names: List[str],
        attributes_map: Dict[str, SequenceAttributes],
        ious_map: Dict[str, np.ndarray],
        fps_map: Dict[str, float],
    ) -> Dict[str, AttributePerformance]:
        """Compute per-attribute performance aggregates.

        Args:
            sequence_names: Sequence identifiers (determines iteration order).
            attributes_map: Maps sequence name → :class:`~eovot.datasets.attributes.SequenceAttributes`.
            ious_map: Maps sequence name → per-frame IoU array ``(N,)``.
            fps_map: Maps sequence name → measured FPS.

        Returns:
            Dict mapping attribute name → :class:`AttributePerformance`.
            Attributes with zero matching sequences are omitted.
        """
        # Group sequences by attribute.
        attr_seqs: Dict[str, List[str]] = {a: [] for a in ALL_ATTRIBUTES}
        for seq in sequence_names:
            sa = attributes_map.get(seq)
            if sa is None:
                continue
            for attr, active in sa.attribute_vector().items():
                if active:
                    attr_seqs[attr].append(seq)

        results: Dict[str, AttributePerformance] = {}
        for attr in ALL_ATTRIBUTES:
            seqs = attr_seqs[attr]
            if not seqs:
                continue

            per_seq_mean_iou: List[float] = []
            per_seq_fps: List[float] = []
            per_seq_auc: List[float] = []

            for seq in seqs:
                ious = ious_map.get(seq)
                fps = fps_map.get(seq)
                if ious is None or len(ious) == 0:
                    continue
                per_seq_mean_iou.append(float(np.mean(ious)))
                if fps is not None:
                    per_seq_fps.append(float(fps))
                per_seq_auc.append(self._success_auc(ious))

            if not per_seq_mean_iou:
                continue

            results[attr] = AttributePerformance(
                attribute=attr,
                sequence_count=len(seqs),
                mean_iou=float(np.mean(per_seq_mean_iou)),
                mean_fps=float(np.mean(per_seq_fps)) if per_seq_fps else 0.0,
                success_auc=float(np.mean(per_seq_auc)),
            )

        return results

    def to_markdown_table(
        self,
        results: Dict[str, AttributePerformance],
        tracker_name: str = "",
    ) -> str:
        """Format attribute performance as a Markdown table.

        Args:
            results: Output of :meth:`compute`.
            tracker_name: Optional tracker label for the table header.

        Returns:
            Multi-line Markdown string ready to embed in a README or paper.
        """
        header_label = f" — {tracker_name}" if tracker_name else ""
        lines = [
            f"### Attribute-Conditioned Performance{header_label}\n",
            "| Attribute | # Seqs | mIoU | Success AUC | FPS |",
            "|-----------|-------:|-----:|------------:|----:|",
        ]
        for attr in ALL_ATTRIBUTES:
            if attr not in results:
                continue
            p = results[attr]
            lines.append(
                f"| {attr.replace('_', ' ').title()} "
                f"| {p.sequence_count} "
                f"| {p.mean_iou:.4f} "
                f"| {p.success_auc:.4f} "
                f"| {p.mean_fps:.1f} |"
            )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _success_auc(self, ious: np.ndarray) -> float:
        """AUC of the success curve over configured thresholds."""
        success = np.mean(ious[:, None] >= self._thresholds[None, :], axis=0)
        # Use whichever trapz variant is available (NumPy 2.0 renamed it).
        try:
            return float(np.trapezoid(success, self._thresholds))  # type: ignore[attr-defined]
        except AttributeError:
            return float(np.trapz(success, self._thresholds))
