"""Per-challenge-attribute tracker performance analysis.

Breaks down benchmark accuracy metrics by the challenge attribute present
in each sequence (fast motion, scale change, etc.).  Enables fine-grained
comparison of tracker strengths and weaknesses — a standard evaluation
protocol in major VOT benchmarks (OTB100, TrackingNet, LaSOT).

Typical usage::

    from eovot.metrics.attribute_analysis import AttributeAnalyzer
    from eovot.datasets.attributes import AttributeTagger, TrackingAttribute

    # 1. Tag sequences with their challenge attributes.
    tagger = AttributeTagger()
    tagged = tagger.tag_dataset(
        {seq.name: seq.ground_truth for seq in dataset},
        frame_sizes={(seq.name: (640, 480)) for seq in dataset},
    )

    # 2. Collect benchmark results (BenchmarkResult from engine.run()).
    result = engine.run(tracker, dataset)

    # 3. Analyse per-attribute performance.
    analyzer = AttributeAnalyzer(tagged)
    report = analyzer.analyze(result)
    print(analyzer.to_markdown(report))

Output example::

    | Attribute     | Sequences | mIoU   | Success AUC |
    |---------------|-----------|--------|-------------|
    | Fast Motion   | 12        | 0.4321 | 0.5102      |
    | Scale Change  | 8         | 0.5618 | 0.6230      |
    | All           | 20        | 0.5134 | 0.5871      |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .accuracy import MetricsEngine
from ..datasets.attributes import (
    ATTRIBUTE_DISPLAY_NAMES,
    SequenceAttributes,
    TrackingAttribute,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AttributeSliceResult:
    """Aggregate metrics for sequences that share a challenge attribute.

    Attributes:
        attribute: Attribute key (or ``"all"`` for the full dataset).
        display_name: Human-readable label.
        n_sequences: Number of sequences in this slice.
        mean_iou: Mean IoU averaged across all frames in the slice.
        success_auc: Area under the success curve (IoU threshold sweep).
        precision_auc: Area under the precision curve (distance threshold).
    """

    attribute: str
    display_name: str
    n_sequences: int
    mean_iou: float
    success_auc: float
    precision_auc: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "display_name": self.display_name,
            "n_sequences": self.n_sequences,
            "mean_iou": round(self.mean_iou, 6),
            "success_auc": round(self.success_auc, 6),
            "precision_auc": round(self.precision_auc, 6),
        }


@dataclass
class AttributeAnalysisReport:
    """Full per-attribute analysis report for one tracker on one dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the dataset.
        slices: One :class:`AttributeSliceResult` per attribute, plus ``"all"``.
        overall: The ``"all"`` slice for quick access.
    """

    tracker_name: str
    dataset_name: str
    slices: List[AttributeSliceResult] = field(default_factory=list)

    @property
    def overall(self) -> Optional[AttributeSliceResult]:
        for s in self.slices:
            if s.attribute == "all":
                return s
        return None

    def by_attribute(self, key: str) -> Optional[AttributeSliceResult]:
        for s in self.slices:
            if s.attribute == key:
                return s
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "slices": [s.to_dict() for s in self.slices],
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class AttributeAnalyzer:
    """Compute per-attribute performance statistics from benchmark results.

    Args:
        tagged: Output of :meth:`~eovot.datasets.attributes.AttributeTagger.tag_dataset` —
            mapping of ``sequence_name`` → :class:`~eovot.datasets.attributes.SequenceAttributes`.
        min_sequences: Minimum sequences required to include an attribute slice.
            Attributes with fewer matching sequences are omitted.
    """

    def __init__(
        self,
        tagged: Dict[str, SequenceAttributes],
        min_sequences: int = 1,
    ) -> None:
        self._tagged = tagged
        self._min_sequences = min_sequences
        self._metrics = MetricsEngine()

    # ------------------------------------------------------------------ #
    # Core analysis                                                       #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        result: Any,  # BenchmarkResult (avoid circular import)
    ) -> AttributeAnalysisReport:
        """Break down *result* by challenge attribute.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` object.

        Returns:
            :class:`AttributeAnalysisReport` with one slice per attribute plus
            an ``"all"`` slice.
        """
        slices: List[AttributeSliceResult] = []

        # ---- Overall (all sequences) ----
        all_slice = self._compute_slice(result.sequence_results, "all", "All Sequences")
        if all_slice is not None:
            slices.append(all_slice)

        # ---- Per-attribute slices ----
        for attr in TrackingAttribute:
            matching = [
                sr for sr in result.sequence_results
                if self._tagged.get(sr.sequence_name, SequenceAttributes("?")).tags.get(attr, False)
            ]
            if len(matching) < self._min_sequences:
                continue
            display = ATTRIBUTE_DISPLAY_NAMES.get(attr, attr)
            s = self._compute_slice(matching, attr, display)
            if s is not None:
                slices.append(s)

        return AttributeAnalysisReport(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            slices=slices,
        )

    def _compute_slice(
        self,
        sequence_results: list,
        attribute: str,
        display_name: str,
    ) -> Optional[AttributeSliceResult]:
        """Aggregate metrics for a list of sequence results."""
        if not sequence_results:
            return None

        all_ious = np.concatenate([sr.ious for sr in sequence_results])
        accuracy = self._metrics.compute_all(all_ious)

        return AttributeSliceResult(
            attribute=attribute,
            display_name=display_name,
            n_sequences=len(sequence_results),
            mean_iou=float(accuracy.mean_iou),
            success_auc=float(accuracy.success_auc),
            precision_auc=float(accuracy.precision_auc),
        )

    # ------------------------------------------------------------------ #
    # Multi-tracker comparison                                            #
    # ------------------------------------------------------------------ #

    def compare(
        self,
        results: List[Any],  # List[BenchmarkResult]
    ) -> Dict[str, AttributeAnalysisReport]:
        """Analyse multiple trackers and return a report per tracker.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

        Returns:
            ``{tracker_name: AttributeAnalysisReport}``
        """
        return {r.tracker_name: self.analyze(r) for r in results}

    # ------------------------------------------------------------------ #
    # Reporting                                                           #
    # ------------------------------------------------------------------ #

    def to_markdown(
        self,
        report: AttributeAnalysisReport,
        metric: str = "mean_iou",
        min_sequences: Optional[int] = None,
    ) -> str:
        """Format a single tracker's attribute analysis as Markdown.

        Args:
            report: Output of :meth:`analyze`.
            metric: Primary metric column to highlight.
            min_sequences: Override minimum sequences threshold for display.

        Returns:
            Markdown table string.
        """
        min_seq = min_sequences if min_sequences is not None else self._min_sequences
        rows = [s for s in report.slices if s.n_sequences >= min_seq]

        lines = [
            f"## Attribute Analysis — {report.tracker_name} on {report.dataset_name}\n",
            "| Attribute | Seq | mIoU | Success AUC | Precision AUC |",
            "|-----------|-----|------|-------------|---------------|",
        ]
        for s in rows:
            marker = " **" if s.attribute == "all" else ""
            name_col = f"{marker}{s.display_name}{marker.replace(' **', '**')}"
            lines.append(
                f"| {name_col} | {s.n_sequences} "
                f"| {s.mean_iou:.4f} "
                f"| {s.success_auc:.4f} "
                f"| {s.precision_auc:.4f} |"
            )
        return "\n".join(lines) + "\n"

    def compare_to_markdown(
        self,
        reports: Dict[str, AttributeAnalysisReport],
        attribute: str = "all",
    ) -> str:
        """Multi-tracker comparison table for a single attribute.

        Args:
            reports: Output of :meth:`compare`.
            attribute: Attribute key to compare across trackers.

        Returns:
            Markdown table string.
        """
        display = ATTRIBUTE_DISPLAY_NAMES.get(attribute, attribute) if attribute != "all" else "All Sequences"
        lines = [
            f"## Tracker Comparison — {display}\n",
            "| Tracker | Seq | mIoU | Success AUC | Precision AUC |",
            "|---------|-----|------|-------------|---------------|",
        ]
        for tracker_name, rpt in reports.items():
            s = rpt.by_attribute(attribute)
            if s is None:
                continue
            lines.append(
                f"| {tracker_name} | {s.n_sequences} "
                f"| {s.mean_iou:.4f} "
                f"| {s.success_auc:.4f} "
                f"| {s.precision_auc:.4f} |"
            )
        return "\n".join(lines) + "\n"

    def coverage_to_markdown(
        self,
        reports: Dict[str, AttributeAnalysisReport],
    ) -> str:
        """Attribute × Tracker mIoU coverage matrix (Markdown).

        Rows = attributes, columns = trackers.

        Args:
            reports: Output of :meth:`compare`.

        Returns:
            Markdown table string.
        """
        tracker_names = list(reports.keys())
        all_attributes = [
            s.attribute
            for s in next(iter(reports.values())).slices
            if s.attribute != "all"
        ]

        header = "| Attribute | " + " | ".join(tracker_names) + " |"
        sep = "|-----------|" + "--------|" * len(tracker_names)
        lines = [header, sep]

        for attr in all_attributes:
            display = ATTRIBUTE_DISPLAY_NAMES.get(attr, attr)
            row_vals = []
            for tname in tracker_names:
                s = reports[tname].by_attribute(attr)
                row_vals.append(f"{s.mean_iou:.4f}" if s else "N/A")
            lines.append(f"| {display} | " + " | ".join(row_vals) + " |")

        return "\n".join(lines) + "\n"
