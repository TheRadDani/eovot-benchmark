"""Attribute-based tracker performance analysis for OTB-style benchmarks.

Visual tracking benchmarks annotate sequences with challenge attributes such as
Occlusion, Fast Motion, and Scale Variation.  Grouping per-sequence results by
these attributes exposes *why* a tracker succeeds or fails — information that
aggregate mIoU scores completely hide.

This module provides :class:`AttributeAnalyzer`, which accepts benchmark results
(from :class:`~eovot.benchmark.engine.BenchmarkResult`) and a mapping of sequence
names to their challenge attributes, then produces:

- Per-attribute mean IoU and failure statistics for each tracker
- A tracker × attribute comparison matrix (research-grade table)
- Identification of the *hardest* attribute per tracker
- Identification of the *best* tracker per attribute
- A self-contained Markdown report suitable for inclusion in papers

The analyzer is dataset-agnostic: any ``Dict[str, Collection[str]]`` mapping
sequence names to attribute codes works — not just OTB.  The
:meth:`~eovot.datasets.otb.OTBDataset.attribute_map` method on ``OTBDataset``
returns exactly this format.

Typical usage::

    from eovot.analysis import AttributeAnalyzer
    from eovot.datasets.otb import OTBDataset

    dataset = OTBDataset("/data/OTB100")
    attr_map = dataset.attribute_map()

    # Assume benchmark_results is a list of BenchmarkResult objects,
    # one per tracker, obtained from BenchmarkEngine.
    analyzer = AttributeAnalyzer(attr_map)
    report = analyzer.generate_report(benchmark_results)
    print(report.to_markdown())

    # Save to file
    with open("attribute_analysis.md", "w") as f:
        f.write(report.to_markdown())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Collection, Dict, FrozenSet, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AttributeStats:
    """Per-attribute performance statistics for a single tracker.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        attribute: Attribute code (e.g. ``"FM"``).
        attribute_name: Full attribute name (e.g. ``"Fast Motion"``).
        n_sequences: Number of sequences in the dataset that carry this attribute.
        mean_iou: Mean IoU across all frames in those sequences.
        std_iou: Standard deviation of per-sequence mean IoU values.
        min_iou: Worst per-sequence mean IoU (most difficult sequence for this attribute).
        max_iou: Best per-sequence mean IoU.
        sequence_names: Sequence names included in this attribute group.
    """

    tracker_name: str
    attribute: str
    attribute_name: str
    n_sequences: int
    mean_iou: float
    std_iou: float
    min_iou: float
    max_iou: float
    sequence_names: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"AttributeStats({self.tracker_name} | {self.attribute} | "
            f"n={self.n_sequences} | mIoU={self.mean_iou:.4f} ± {self.std_iou:.4f})"
        )


@dataclass
class AttributeReport:
    """Full attribute-analysis report for one or more trackers.

    Attributes:
        tracker_names: Ordered list of tracker names in this report.
        attributes: Ordered list of attribute codes covered.
        attribute_names: Full names corresponding to each code in ``attributes``.
        stats: Nested mapping ``tracker_name → attribute → AttributeStats``.
        dataset_name: Optional dataset label for the report header.
    """

    tracker_names: List[str]
    attributes: List[str]
    attribute_names: List[str]
    stats: Dict[str, Dict[str, AttributeStats]]
    dataset_name: str = ""

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def iou_matrix(self) -> np.ndarray:
        """Return a (n_trackers × n_attributes) mean-IoU matrix.

        Rows correspond to ``tracker_names``; columns to ``attributes``.
        Missing cells (attribute not present for a tracker) are filled with
        ``np.nan``.
        """
        mat = np.full((len(self.tracker_names), len(self.attributes)), np.nan)
        for i, tracker in enumerate(self.tracker_names):
            for j, attr in enumerate(self.attributes):
                entry = self.stats.get(tracker, {}).get(attr)
                if entry is not None:
                    mat[i, j] = entry.mean_iou
        return mat

    def hardest_attribute(self, tracker_name: str) -> Optional[Tuple[str, float]]:
        """Return ``(attribute_code, mean_iou)`` for the tracker's weakest attribute.

        Returns ``None`` if the tracker has no attribute stats.
        """
        tracker_stats = self.stats.get(tracker_name, {})
        if not tracker_stats:
            return None
        worst = min(tracker_stats.values(), key=lambda s: s.mean_iou)
        return worst.attribute, worst.mean_iou

    def best_tracker_per_attribute(self) -> Dict[str, Tuple[str, float]]:
        """Return ``{attribute: (best_tracker_name, mean_iou)}`` for all attributes."""
        best: Dict[str, Tuple[str, float]] = {}
        for attr in self.attributes:
            candidates = [
                (tracker, self.stats[tracker][attr].mean_iou)
                for tracker in self.tracker_names
                if attr in self.stats.get(tracker, {})
            ]
            if candidates:
                best[attr] = max(candidates, key=lambda x: x[1])
        return best

    # ------------------------------------------------------------------
    # Report serialisation
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render a complete Markdown analysis report.

        The report contains:
        1. Header with dataset name and date context
        2. Tracker × attribute mean-IoU table
        3. Per-tracker hardest attribute summary
        4. Best tracker per attribute table
        5. Methodology note
        """
        lines: List[str] = []

        # --- Header ---
        header = f"# Attribute-Based Tracker Performance Analysis"
        if self.dataset_name:
            header += f" — {self.dataset_name}"
        lines.append(header)
        lines.append("")
        lines.append(
            "Mean IoU grouped by OTB challenge attribute.  "
            "Each cell shows the average IoU over all sequences that carry the given "
            "attribute.  **Bold** marks the best tracker per column."
        )
        lines.append("")

        # --- Tracker × attribute matrix ---
        mat = self.iou_matrix()
        best_per_col = np.nanargmax(mat, axis=0)

        # Build header row
        col_labels = [f"{a}<br>({n})" for a, n in zip(self.attributes, self.attribute_names)]
        header_row = "| Tracker | " + " | ".join(col_labels) + " |"
        sep_row = "|---------|" + "|".join(["------:"] * len(self.attributes)) + "|"
        lines.append(header_row)
        lines.append(sep_row)

        for i, tracker in enumerate(self.tracker_names):
            cells = []
            for j, attr in enumerate(self.attributes):
                val = mat[i, j]
                if math.isnan(val):
                    cells.append(" — ")
                else:
                    formatted = f"{val:.3f}"
                    if best_per_col[j] == i:
                        formatted = f"**{formatted}**"
                    cells.append(formatted)
            lines.append(f"| {tracker} | " + " | ".join(cells) + " |")

        lines.append("")

        # --- Coverage row (how many sequences per attribute) ---
        lines.append("*Sequence counts per attribute:*")
        lines.append("")
        # Use stats from the first tracker to get sequence counts
        first_tracker = self.tracker_names[0] if self.tracker_names else ""
        count_parts = []
        for attr in self.attributes:
            entry = self.stats.get(first_tracker, {}).get(attr)
            n = entry.n_sequences if entry else "?"
            count_parts.append(f"{attr}: {n}")
        lines.append("  " + ",  ".join(count_parts))
        lines.append("")

        # --- Per-tracker hardest attribute ---
        lines.append("## Hardest Attribute per Tracker")
        lines.append("")
        lines.append("| Tracker | Hardest Attribute | mIoU on that Attribute |")
        lines.append("|---------|-------------------|------------------------|")
        for tracker in self.tracker_names:
            result = self.hardest_attribute(tracker)
            if result:
                attr_code, miou = result
                attr_full = self.attribute_names[self.attributes.index(attr_code)]
                lines.append(f"| {tracker} | {attr_code} ({attr_full}) | {miou:.4f} |")
        lines.append("")

        # --- Best tracker per attribute ---
        lines.append("## Best Tracker per Attribute")
        lines.append("")
        lines.append("| Attribute | Best Tracker | mIoU |")
        lines.append("|-----------|-------------|------|")
        best_per_attr = self.best_tracker_per_attribute()
        for attr, attr_name in zip(self.attributes, self.attribute_names):
            if attr in best_per_attr:
                tracker, miou = best_per_attr[attr]
                lines.append(f"| {attr} ({attr_name}) | {tracker} | {miou:.4f} |")
        lines.append("")

        # --- Methodology ---
        lines.append("## Methodology")
        lines.append("")
        lines.append(
            "Attribute membership follows the annotations from "
            "*Wu et al., \"Object Tracking Benchmark\", IEEE TPAMI 2015*. "
            "A sequence may carry multiple attributes; it is included in the "
            "computation for each attribute it carries.  Mean IoU is computed "
            "over all frames in the matched sequences, weighted equally per "
            "sequence."
        )
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the report to a JSON-compatible dict."""
        return {
            "dataset": self.dataset_name,
            "trackers": self.tracker_names,
            "attributes": self.attributes,
            "stats": {
                tracker: {
                    attr: {
                        "n_sequences": s.n_sequences,
                        "mean_iou": round(s.mean_iou, 6),
                        "std_iou": round(s.std_iou, 6),
                        "min_iou": round(s.min_iou, 6),
                        "max_iou": round(s.max_iou, 6),
                        "sequences": s.sequence_names,
                    }
                    for attr, s in attr_map.items()
                }
                for tracker, attr_map in self.stats.items()
            },
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AttributeAnalyzer:
    """Compute per-attribute tracker performance from benchmark results.

    The analyzer is intentionally dataset-agnostic: it accepts any mapping
    ``{sequence_name: set_of_attribute_codes}`` so it works with OTB, custom
    datasets, or any benchmark with sequence-level annotations.

    Args:
        attribute_map: Mapping from sequence name to a collection of attribute
            codes.  Use :meth:`~eovot.datasets.otb.OTBDataset.attribute_map`
            to obtain this from an ``OTBDataset``, or build it manually.
        attribute_names: Optional mapping from attribute code to full name for
            richer report output.  Defaults to using the code itself when absent.
        dataset_name: Optional string used as the report title.

    Example::

        from eovot.analysis import AttributeAnalyzer
        from eovot.datasets.otb import OTBDataset, OTB_ATTRIBUTE_NAMES

        dataset = OTBDataset("/data/OTB100")
        analyzer = AttributeAnalyzer(
            attribute_map=dataset.attribute_map(),
            attribute_names=OTB_ATTRIBUTE_NAMES,
            dataset_name="OTB-100",
        )
        report = analyzer.generate_report(benchmark_results)
        print(report.to_markdown())
    """

    def __init__(
        self,
        attribute_map: Dict[str, Collection[str]],
        attribute_names: Optional[Dict[str, str]] = None,
        dataset_name: str = "",
    ) -> None:
        if not attribute_map:
            raise ValueError("attribute_map must not be empty.")
        self._attr_map: Dict[str, FrozenSet[str]] = {
            seq: frozenset(attrs) for seq, attrs in attribute_map.items()
        }
        self._attr_names: Dict[str, str] = attribute_names or {}
        self.dataset_name = dataset_name

        # Collect all unique attributes across the map, sorted for reproducibility.
        all_attrs: set = set()
        for attrs in self._attr_map.values():
            all_attrs.update(attrs)
        self._all_attributes: List[str] = sorted(all_attrs)

    @property
    def attributes(self) -> List[str]:
        """Sorted list of all attribute codes present in the attribute map."""
        return list(self._all_attributes)

    def sequences_for_attribute(self, attribute: str) -> List[str]:
        """Return sequence names annotated with *attribute*, sorted."""
        return sorted(
            seq for seq, attrs in self._attr_map.items() if attribute in attrs
        )

    def compute_stats(
        self,
        result: "BenchmarkResult",
    ) -> Dict[str, AttributeStats]:
        """Compute per-attribute :class:`AttributeStats` for one tracker result.

        Only sequences present in both *result* and the attribute map contribute
        to the statistics.  Sequences in the result that are not in the attribute
        map are ignored (they were not annotated and cannot be grouped).

        Args:
            result: :class:`~eovot.benchmark.engine.BenchmarkResult` from a
                single tracker evaluated on the benchmark dataset.

        Returns:
            Mapping ``{attribute_code: AttributeStats}`` containing one entry
            per attribute that has at least one matching sequence.
        """
        # Build a lookup from sequence name → per-sequence mean IoU.
        seq_iou: Dict[str, float] = {
            sr.sequence_name: float(sr.ious.mean()) if len(sr.ious) > 0 else 0.0
            for sr in result.sequence_results
        }

        stats: Dict[str, AttributeStats] = {}
        for attr in self._all_attributes:
            matched_seqs = [
                seq for seq in self.sequences_for_attribute(attr) if seq in seq_iou
            ]
            if not matched_seqs:
                continue

            ious = np.array([seq_iou[seq] for seq in matched_seqs])
            stats[attr] = AttributeStats(
                tracker_name=result.tracker_name,
                attribute=attr,
                attribute_name=self._attr_names.get(attr, attr),
                n_sequences=len(matched_seqs),
                mean_iou=float(ious.mean()),
                std_iou=float(ious.std()) if len(ious) > 1 else 0.0,
                min_iou=float(ious.min()),
                max_iou=float(ious.max()),
                sequence_names=matched_seqs,
            )

        return stats

    def generate_report(
        self,
        results: List["BenchmarkResult"],
        attributes: Optional[List[str]] = None,
    ) -> AttributeReport:
        """Build a full :class:`AttributeReport` for all trackers.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
                tracker.  Each result's ``tracker_name`` becomes a row in the
                report.
            attributes: Subset of attribute codes to include.  If ``None``,
                all attributes present in the attribute map are included.

        Returns:
            :class:`AttributeReport` ready for ``to_markdown()`` or
            ``to_dict()`` output.

        Raises:
            ValueError: If *results* is empty or *attributes* contains
                codes absent from the attribute map.
        """
        if not results:
            raise ValueError("results must contain at least one BenchmarkResult.")

        active_attrs = attributes if attributes is not None else self._all_attributes
        unknown = set(active_attrs) - set(self._all_attributes)
        if unknown:
            raise ValueError(
                f"Requested attributes not in attribute_map: {sorted(unknown)}"
            )

        all_stats: Dict[str, Dict[str, AttributeStats]] = {}
        for result in results:
            tracker_stats = self.compute_stats(result)
            # Filter to only the requested attributes.
            all_stats[result.tracker_name] = {
                attr: stat
                for attr, stat in tracker_stats.items()
                if attr in active_attrs
            }

        tracker_names = [r.tracker_name for r in results]
        attr_name_list = [self._attr_names.get(a, a) for a in active_attrs]

        return AttributeReport(
            tracker_names=tracker_names,
            attributes=list(active_attrs),
            attribute_names=attr_name_list,
            stats=all_stats,
            dataset_name=self.dataset_name,
        )
