"""Sequence attribute tagging and per-attribute performance analysis.

VOT challenge benchmarks annotate each sequence with difficulty attributes
that describe the visual challenges present — occlusion, fast motion, scale
change, etc.  Stratifying benchmark results by these attributes reveals
*where* a tracker fails and which challenge factors drive performance
differences between trackers.

This module provides:

- :class:`SequenceAttribute` — standard VOT/OTB/LaSOT challenge attribute
  enumeration.  Custom attributes can be added; only use members of this
  enum when tagging sequences so that the analyzer can aggregate consistently.

- :class:`AttributePerformance` — per-attribute performance summary (mIoU,
  success AUC, precision AUC, FPS) for one tracker.

- :class:`AttributeAnalyzer` — stratifies a
  :class:`~eovot.benchmark.engine.BenchmarkResult` by attribute and produces
  per-attribute tables.  Supports single-tracker and multi-tracker analysis.

Typical usage::

    from eovot.analysis.attributes import AttributeAnalyzer, SequenceAttribute
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=10, motion="linear")
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    # Build attribute map from the dataset (sequences auto-tagged for synthetic data)
    attr_map = {seq.name: seq.attributes for seq in dataset if seq.attributes}

    analyzer = AttributeAnalyzer()
    analysis = analyzer.analyze(result, attr_map)
    print(analyzer.to_markdown_table(analysis, tracker_name="KCF"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult, SequenceResult


class SequenceAttribute(Enum):
    """Standard difficulty attributes used in VOT, OTB-100, and LaSOT protocols.

    Members correspond to the challenge categories defined in the OTB-100 paper
    (Wu et al., 2015) and extended in the LaSOT benchmark (Fan et al., 2019).
    Custom attributes may be added for project-specific analysis.

    References:
        Wu et al., "Object Tracking Benchmark." TPAMI 2015.
        Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single
        Object Tracking." CVPR 2019.
    """

    OCCLUSION = "occlusion"
    """Target is partially or fully occluded by another object."""

    FAST_MOTION = "fast_motion"
    """Per-frame displacement > 20 % of target size (OTB definition)."""

    SCALE_CHANGE = "scale_change"
    """Target bounding box area changes by > 25 % relative to the first frame."""

    ILLUMINATION_CHANGE = "illumination_change"
    """Significant change in scene illumination affecting target appearance."""

    OUT_OF_VIEW = "out_of_view"
    """Target partially or fully leaves the field of view."""

    DEFORMATION = "deformation"
    """Non-rigid deformation of the target shape (e.g. a walking person)."""

    BACKGROUND_CLUTTER = "background_clutter"
    """Background region resembles the target in appearance or colour."""

    LOW_RESOLUTION = "low_resolution"
    """Target occupies fewer than 400 px² (< 20×20 px effective area)."""

    MOTION_BLUR = "motion_blur"
    """Target or camera motion causes perceptible blur in the target region."""

    ROTATION = "rotation"
    """Target undergoes significant in-plane or out-of-plane rotation."""

    ASPECT_RATIO_CHANGE = "aspect_ratio_change"
    """Bounding-box aspect ratio changes by > 40 % relative to the first frame."""


@dataclass
class AttributePerformance:
    """Per-attribute performance summary for one tracker.

    Attributes:
        attribute: The :class:`SequenceAttribute` this entry describes.
        num_sequences: Number of sequences tagged with this attribute.
        mean_iou: Mean IoU across all frames in attribute-tagged sequences.
        mean_success_auc: Mean success-curve AUC, or ``None`` if not computed.
        mean_precision_auc: Mean precision-curve AUC, or ``None`` if not computed.
        mean_fps: Mean tracker throughput across attribute-tagged sequences.
        tracker_name: Optional tracker identifier for display purposes.
    """

    attribute: SequenceAttribute
    num_sequences: int
    mean_iou: float
    mean_success_auc: Optional[float]
    mean_precision_auc: Optional[float]
    mean_fps: float
    tracker_name: str = ""

    def __str__(self) -> str:
        sauc = f"{self.mean_success_auc:.4f}" if self.mean_success_auc is not None else "N/A"
        return (
            f"AttributePerformance({self.attribute.value}: "
            f"n={self.num_sequences}  mIoU={self.mean_iou:.4f}  "
            f"success_AUC={sauc}  fps={self.mean_fps:.1f})"
        )

    def to_dict(self) -> Dict:
        return {
            "attribute": self.attribute.value,
            "num_sequences": self.num_sequences,
            "mean_iou": round(self.mean_iou, 4),
            "mean_success_auc": round(self.mean_success_auc, 4) if self.mean_success_auc is not None else None,
            "mean_precision_auc": round(self.mean_precision_auc, 4) if self.mean_precision_auc is not None else None,
            "mean_fps": round(self.mean_fps, 2),
        }


class AttributeAnalyzer:
    """Analyze tracker performance stratified by sequence difficulty attributes.

    Sequences tagged with a given attribute are grouped and their metrics
    averaged, enabling comparison like:
    "KCF loses 0.12 mIoU on occluded sequences vs its average performance."

    This analysis is especially useful for identifying which challenge factors
    cause a tracker to regress relative to state-of-the-art baselines.

    Args:
        min_sequences: Minimum number of sequences required for an attribute to
            appear in the analysis.  Attributes with fewer sequences produce
            statistically unreliable averages.  Default: ``2``.

    Example::

        analyzer = AttributeAnalyzer(min_sequences=3)
        attr_map = {seq.name: seq.attributes for seq in dataset if seq.attributes}
        analysis = analyzer.analyze(result, attr_map)
        print(analyzer.to_markdown_table(analysis, tracker_name="KCF"))
    """

    def __init__(self, min_sequences: int = 2) -> None:
        if min_sequences < 1:
            raise ValueError(f"min_sequences must be >= 1, got {min_sequences}")
        self.min_sequences = min_sequences

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        result: "BenchmarkResult",
        attribute_map: Dict[str, Set[SequenceAttribute]],
    ) -> Dict[SequenceAttribute, AttributePerformance]:
        """Compute per-attribute performance for a single tracker result.

        Each sequence can carry multiple attributes; a sequence tagged with
        both ``OCCLUSION`` and ``FAST_MOTION`` contributes to both groups.

        Args:
            result: Full benchmark result from
                :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            attribute_map: ``{sequence_name: set_of_attributes}``.  Sequences
                not present in the map are silently skipped.  Pass
                ``{seq.name: seq.attributes for seq in dataset if seq.attributes}``
                to use metadata auto-tagged by the dataset loader.

        Returns:
            ``{SequenceAttribute: AttributePerformance}`` — only attributes
            with at least ``min_sequences`` sequences are included.
        """
        attr_to_seqs: Dict[SequenceAttribute, List["SequenceResult"]] = {
            a: [] for a in SequenceAttribute
        }
        for seq_result in result.sequence_results:
            attrs = attribute_map.get(seq_result.sequence_name) or set()
            for attr in attrs:
                if isinstance(attr, SequenceAttribute):
                    attr_to_seqs[attr].append(seq_result)

        output: Dict[SequenceAttribute, AttributePerformance] = {}
        for attr, seq_results in attr_to_seqs.items():
            if len(seq_results) < self.min_sequences:
                continue
            output[attr] = self._compute_performance(
                attr=attr,
                seq_results=seq_results,
                tracker_name=result.tracker_name,
            )
        return output

    def analyze_dataset(
        self,
        result: "BenchmarkResult",
        dataset,
    ) -> Dict[SequenceAttribute, AttributePerformance]:
        """Convenience wrapper: build the attribute map directly from a dataset.

        Sequences whose ``attributes`` field is ``None`` are skipped.

        Args:
            result: Benchmark result from :class:`~eovot.benchmark.engine.BenchmarkEngine`.
            dataset: Any :class:`~eovot.datasets.base.BaseDataset` whose sequences
                carry :attr:`~eovot.datasets.base.Sequence.attributes` metadata.

        Returns:
            Same as :meth:`analyze`.
        """
        attr_map: Dict[str, Set[SequenceAttribute]] = {}
        for seq in dataset:
            if seq.attributes:
                attr_map[seq.name] = seq.attributes
        return self.analyze(result, attr_map)

    def compare_trackers(
        self,
        results: List["BenchmarkResult"],
        attribute_map: Dict[str, Set[SequenceAttribute]],
    ) -> Dict[SequenceAttribute, List[AttributePerformance]]:
        """Compare multiple trackers per attribute.

        Each result in *results* is analyzed independently against the same
        *attribute_map*.  The output groups per-attribute performances by
        attribute for easy side-by-side comparison.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker.
            attribute_map: Shared ``{sequence_name: set_of_attributes}`` map.
                All trackers must have been evaluated on the same sequences.

        Returns:
            ``{SequenceAttribute: [AttributePerformance, ...]}`` — one
            ``AttributePerformance`` per tracker per attribute.  Only
            attributes that appear in at least one tracker's analysis are
            included.
        """
        per_tracker: Dict[str, Dict[SequenceAttribute, AttributePerformance]] = {}
        for result in results:
            per_tracker[result.tracker_name] = self.analyze(result, attribute_map)

        all_attrs: Set[SequenceAttribute] = set()
        for analysis in per_tracker.values():
            all_attrs.update(analysis.keys())

        output: Dict[SequenceAttribute, List[AttributePerformance]] = {}
        tracker_order = [r.tracker_name for r in results]
        for attr in sorted(all_attrs, key=lambda a: a.value):
            perfs = [
                per_tracker[tn][attr]
                for tn in tracker_order
                if attr in per_tracker.get(tn, {})
            ]
            if perfs:
                output[attr] = perfs
        return output

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown_table(
        self,
        analysis: Dict[SequenceAttribute, AttributePerformance],
        tracker_name: str = "",
    ) -> str:
        """Format per-attribute analysis as a Markdown table.

        Rows are sorted by mIoU descending so the easiest challenges
        (where the tracker performs best) appear first.

        Args:
            analysis: Output of :meth:`analyze`.
            tracker_name: Optional tracker identifier for the table title.

        Returns:
            Multi-line Markdown string ready to embed in reports or READMEs.
        """
        if not analysis:
            return "_No attribute data available (min_sequences threshold not met)._\n"

        title = f"Per-Attribute Performance — {tracker_name}" if tracker_name else "Per-Attribute Performance"
        lines = [
            f"### {title}\n",
            "| Attribute | # Seqs | mIoU | Success AUC | Precision AUC | FPS |",
            "|-----------|-------:|-----:|------------:|--------------:|----:|",
        ]
        for attr, perf in sorted(analysis.items(), key=lambda x: -x[1].mean_iou):
            sauc = f"{perf.mean_success_auc:.4f}" if perf.mean_success_auc is not None else "N/A"
            pauc = f"{perf.mean_precision_auc:.4f}" if perf.mean_precision_auc is not None else "N/A"
            lines.append(
                f"| {attr.value} | {perf.num_sequences} "
                f"| {perf.mean_iou:.4f} | {sauc} | {pauc} "
                f"| {perf.mean_fps:.1f} |"
            )
        return "\n".join(lines)

    def to_multi_tracker_table(
        self,
        comparison: Dict[SequenceAttribute, List[AttributePerformance]],
        tracker_names: Optional[List[str]] = None,
    ) -> str:
        """Format a multi-tracker attribute comparison as a Markdown table.

        One column per tracker shows mIoU, making it easy to spot which
        tracker is most robust to each challenge type.

        Args:
            comparison: Output of :meth:`compare_trackers`.
            tracker_names: Ordered list of tracker names for the header.
                Inferred from the first attribute's performance list if omitted.

        Returns:
            Multi-line Markdown table, one row per attribute.
        """
        if not comparison:
            return "_No attribute comparison data available._\n"

        if tracker_names is None:
            first = next(iter(comparison.values()))
            tracker_names = [p.tracker_name for p in first]

        header_cols = " | ".join(f"{name} mIoU" for name in tracker_names)
        sep_cols = " | ".join("-----:" for _ in tracker_names)

        lines = [
            "### Multi-Tracker Attribute Analysis\n",
            f"| Attribute | # Seqs | {header_cols} |",
            f"|-----------|-------:|{sep_cols}|",
        ]

        for attr in sorted(comparison.keys(), key=lambda a: a.value):
            perfs = comparison[attr]
            n_seqs = perfs[0].num_sequences if perfs else 0
            iou_cols = " | ".join(f"{p.mean_iou:.4f}" for p in perfs)
            lines.append(f"| {attr.value} | {n_seqs} | {iou_cols} |")

        return "\n".join(lines)

    def degradation_report(
        self,
        result: "BenchmarkResult",
        attribute_map: Dict[str, Set[SequenceAttribute]],
    ) -> str:
        """Report how much each attribute degrades tracker performance vs average.

        Computes the mIoU delta between attribute-specific performance and the
        overall tracker mIoU.  Negative deltas identify the challenge types that
        most hurt this tracker.

        Args:
            result: Full benchmark result.
            attribute_map: ``{sequence_name: set_of_attributes}``.

        Returns:
            Multi-line Markdown string showing mIoU delta per attribute,
            sorted from most harmful (most negative) to least.
        """
        overall_iou = result.mean_iou
        analysis = self.analyze(result, attribute_map)

        if not analysis:
            return "_No attribute data available._\n"

        rows = [
            (attr.value, perf.num_sequences, perf.mean_iou - overall_iou, perf.mean_iou)
            for attr, perf in analysis.items()
        ]
        rows.sort(key=lambda r: r[2])  # most harmful first

        lines = [
            f"### Attribute Degradation Report — {result.tracker_name}\n",
            f"*Overall mIoU: {overall_iou:.4f}*\n",
            "| Attribute | # Seqs | mIoU | Δ vs average |",
            "|-----------|-------:|-----:|-------------:|",
        ]
        for attr_name, n, delta, miou in rows:
            sign = "▼" if delta < 0 else "▲"
            lines.append(
                f"| {attr_name} | {n} | {miou:.4f} | {sign} {abs(delta):.4f} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_performance(
        attr: SequenceAttribute,
        seq_results: List["SequenceResult"],
        tracker_name: str,
    ) -> AttributePerformance:
        mean_iou = float(np.mean([r.mean_iou for r in seq_results]))
        mean_fps = float(np.mean([r.profiling.fps for r in seq_results]))

        aucs = [
            r.accuracy_metrics.success_auc
            for r in seq_results
            if r.accuracy_metrics is not None
        ]
        mean_sauc: Optional[float] = float(np.mean(aucs)) if aucs else None

        paucs = [
            r.accuracy_metrics.precision_auc
            for r in seq_results
            if r.accuracy_metrics is not None
        ]
        mean_pauc: Optional[float] = float(np.mean(paucs)) if paucs else None

        return AttributePerformance(
            attribute=attr,
            num_sequences=len(seq_results),
            mean_iou=mean_iou,
            mean_success_auc=mean_sauc,
            mean_precision_auc=mean_pauc,
            mean_fps=mean_fps,
            tracker_name=tracker_name,
        )
