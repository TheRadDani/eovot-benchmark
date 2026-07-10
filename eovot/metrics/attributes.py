"""Per-challenge-attribute metrics aggregation for EOVOT.

Standard tracking benchmarks (OTB, LaSOT, TrackingNet) report *average*
performance across all sequences.  This hides the fact that a tracker may
excel on clean sequences but fail on occlusion or fast-motion ones.
Per-attribute breakdown is what reveals which tracking challenges a method
actually solves.

This module aggregates :class:`~eovot.benchmark.engine.BenchmarkResult`
sequences by :class:`~eovot.datasets.synthetic_challenges.ChallengeAttribute`
tag, computes standard VOT metrics for each attribute group, and renders
publication-ready Markdown tables for direct embedding in papers.

Typical usage::

    from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
    from eovot.datasets.synthetic_challenges import ChallengeDataset
    from eovot.metrics.attributes import AttributeMetricsAggregator
    from eovot.trackers.mosse import MOSSETracker

    ds = ChallengeDataset(num_sequences=12, num_frames=80)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="ChallengeSet")

    # Build the attribute index from the dataset
    attr_index = ds.attribute_index()

    agg = AttributeMetricsAggregator()
    report = agg.compute(result, attr_index)
    print(agg.to_markdown(report))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..datasets.synthetic_challenges import ChallengeAttribute

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult, SequenceResult


# ---------------------------------------------------------------------------
# Per-attribute summary
# ---------------------------------------------------------------------------

@dataclass
class AttributeMetrics:
    """Aggregated accuracy and efficiency metrics for one challenge attribute.

    Attributes:
        attribute: The :class:`~eovot.datasets.synthetic_challenges.ChallengeAttribute`
            this summary covers.
        num_sequences: Number of sequences with this attribute.
        mean_iou: Mean IoU across all sequences in this attribute group.
        success_auc: Mean success-curve AUC, or ``None`` if not available.
        precision_auc: Mean precision-curve AUC, or ``None`` if not available.
        mean_fps: Mean FPS throughput.
        peak_memory_mb: Maximum peak-memory footprint across sequences.
    """

    attribute: ChallengeAttribute
    num_sequences: int
    mean_iou: float
    success_auc: Optional[float]
    precision_auc: Optional[float]
    mean_fps: float
    peak_memory_mb: float

    def __str__(self) -> str:
        sauc = f"  AUC={self.success_auc:.4f}" if self.success_auc is not None else ""
        return (
            f"AttributeMetrics[{self.attribute.label()}] "
            f"n={self.num_sequences}  mIoU={self.mean_iou:.4f}{sauc}  "
            f"FPS={self.mean_fps:.1f}  mem={self.peak_memory_mb:.1f} MB"
        )


@dataclass
class AttributeReport:
    """Full per-attribute breakdown for one benchmark run.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset used.
        per_attribute: Mapping from :class:`ChallengeAttribute` to its
            :class:`AttributeMetrics`.
        overall_mean_iou: Mean IoU across **all** sequences.
    """

    tracker_name: str
    dataset_name: str
    per_attribute: Dict[ChallengeAttribute, AttributeMetrics] = field(
        default_factory=dict
    )
    overall_mean_iou: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"AttributeReport[{self.tracker_name} / {self.dataset_name}]  "
            f"overall_mIoU={self.overall_mean_iou:.4f}"
        ]
        for metrics in self.per_attribute.values():
            lines.append(f"  {metrics}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class AttributeMetricsAggregator:
    """Aggregate :class:`~eovot.benchmark.engine.BenchmarkResult` by challenge tag.

    Example::

        agg = AttributeMetricsAggregator()
        report = agg.compute(benchmark_result, ds.attribute_index())
        print(agg.to_markdown(report))
    """

    def compute(
        self,
        result: "BenchmarkResult",
        attribute_index: Dict[ChallengeAttribute, List[int]],
    ) -> AttributeReport:
        """Build an :class:`AttributeReport` from a benchmark result.

        Args:
            result: A completed :class:`~eovot.benchmark.engine.BenchmarkResult`
                from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            attribute_index: Mapping from :class:`ChallengeAttribute` to the
                **sequence indices** (into ``result.sequence_results``) that
                carry that attribute.  Produced by
                :meth:`~eovot.datasets.synthetic_challenges.ChallengeDataset.attribute_index`.

        Returns:
            :class:`AttributeReport` with per-attribute metric summaries.
        """
        seq_results = result.sequence_results
        overall_ious: List[float] = []
        for r in seq_results:
            if len(r.ious) > 0:
                overall_ious.extend(r.ious.tolist())
        overall_mean = float(np.mean(overall_ious)) if overall_ious else 0.0

        per_attribute: Dict[ChallengeAttribute, AttributeMetrics] = {}

        for attr, indices in attribute_index.items():
            valid = [seq_results[i] for i in indices if i < len(seq_results)]
            if not valid:
                continue
            per_attribute[attr] = self._aggregate_group(attr, valid)

        return AttributeReport(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            per_attribute=per_attribute,
            overall_mean_iou=overall_mean,
        )

    @staticmethod
    def _aggregate_group(
        attr: ChallengeAttribute,
        sequences: List["SequenceResult"],
    ) -> AttributeMetrics:
        """Compute summary metrics for a list of sequence results."""
        all_ious: List[float] = []
        fps_list: List[float] = []
        mem_list: List[float] = []
        sauc_list: List[float] = []
        pauc_list: List[float] = []

        for r in sequences:
            if len(r.ious) > 0:
                all_ious.extend(r.ious.tolist())
            fps_list.append(r.profiling.fps)
            mem_list.append(r.profiling.peak_memory_mb)
            if r.accuracy_metrics is not None:
                sauc_list.append(r.accuracy_metrics.success_auc)
                pauc_list.append(r.accuracy_metrics.precision_auc)

        return AttributeMetrics(
            attribute=attr,
            num_sequences=len(sequences),
            mean_iou=float(np.mean(all_ious)) if all_ious else 0.0,
            success_auc=float(np.mean(sauc_list)) if sauc_list else None,
            precision_auc=float(np.mean(pauc_list)) if pauc_list else None,
            mean_fps=float(np.mean(fps_list)) if fps_list else 0.0,
            peak_memory_mb=float(np.max(mem_list)) if mem_list else 0.0,
        )

    @staticmethod
    def to_markdown(report: AttributeReport) -> str:
        """Render the attribute report as a Markdown table.

        Columns: Challenge | Sequences | mIoU | Success AUC | Precision AUC | FPS | Mem (MB)

        Rows are sorted by descending mIoU so the most-challenging attributes
        (lowest scores) appear at the bottom — matching VOT paper conventions.

        Args:
            report: Output of :meth:`compute`.

        Returns:
            Multi-line Markdown string ready to embed in a paper or README.
        """
        lines = [
            f"## Per-Attribute Metrics: {report.tracker_name} on {report.dataset_name}\n",
            f"Overall mean IoU: **{report.overall_mean_iou:.4f}**\n",
            "| Challenge | Seqs | mIoU | Success AUC | Precision AUC | FPS | Mem (MB) |",
            "|-----------|-----:|-----:|------------:|--------------:|----:|---------:|",
        ]

        sorted_attrs = sorted(
            report.per_attribute.values(),
            key=lambda m: m.mean_iou,
            reverse=True,
        )

        for m in sorted_attrs:
            sauc = f"{m.success_auc:.4f}" if m.success_auc is not None else "—"
            pauc = f"{m.precision_auc:.4f}" if m.precision_auc is not None else "—"
            lines.append(
                f"| {m.attribute.label():<20s} "
                f"| {m.num_sequences} "
                f"| {m.mean_iou:.4f} "
                f"| {sauc} "
                f"| {pauc} "
                f"| {m.mean_fps:.1f} "
                f"| {m.peak_memory_mb:.1f} |"
            )
        return "\n".join(lines)

    def to_json(self, report: AttributeReport) -> dict:
        """Serialize an :class:`AttributeReport` to a plain dict.

        Suitable for ``json.dump`` and downstream analysis in pandas.

        Args:
            report: Output of :meth:`compute`.

        Returns:
            Dict with keys ``tracker``, ``dataset``, ``overall_mean_iou``,
            and ``per_attribute`` (list of per-attribute dicts).
        """
        attrs = []
        for m in report.per_attribute.values():
            entry: dict = {
                "attribute": m.attribute.label(),
                "num_sequences": m.num_sequences,
                "mean_iou": round(m.mean_iou, 6),
                "mean_fps": round(m.mean_fps, 2),
                "peak_memory_mb": round(m.peak_memory_mb, 2),
            }
            if m.success_auc is not None:
                entry["success_auc"] = round(m.success_auc, 6)
            if m.precision_auc is not None:
                entry["precision_auc"] = round(m.precision_auc, 6)
            attrs.append(entry)
        return {
            "tracker": report.tracker_name,
            "dataset": report.dataset_name,
            "overall_mean_iou": round(report.overall_mean_iou, 6),
            "per_attribute": attrs,
        }
