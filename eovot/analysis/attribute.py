"""Sequence attribute-based performance analysis for VOT benchmarks.

Most VOT datasets (OTB, LaSOT, GOT-10k) tag sequences with visual challenge
attributes such as fast motion, occlusion, or scale variation.  Stratifying
results by attribute exposes *where* a tracker degrades, enabling targeted
improvement and fair comparison between methods with different strengths.

Typical usage::

    from eovot.analysis import AttributeAnalyzer
    from eovot.benchmark.engine import BenchmarkResult

    # Build attribute map from an OTB-style annotation file or manually:
    analyzer = AttributeAnalyzer()
    analyzer.load_otb_attributes("data/otb_attributes.txt")

    # Analyse a benchmark result:
    analysis = analyzer.from_benchmark_result(result)

    # Print per-attribute table for one tracker:
    print(analyzer.format_comparison_table([analysis]))

    # Compare two trackers:
    cmp = analyzer.compare([analysis_mosse, analysis_kcf])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from eovot.metrics.accuracy import AccuracyMetrics

# ---------------------------------------------------------------------------
# Standard attribute vocabulary
# ---------------------------------------------------------------------------

STANDARD_ATTRIBUTES: List[str] = [
    "illumination_variation",
    "scale_variation",
    "occlusion",
    "deformation",
    "motion_blur",
    "fast_motion",
    "in_plane_rotation",
    "out_of_plane_rotation",
    "out_of_view",
    "background_clutter",
    "low_resolution",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AttributeResult:
    """Per-attribute accuracy summary for one tracker."""

    attribute: str
    num_sequences: int
    mean_iou: float
    success_auc: float
    precision_auc: float
    sequence_names: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"AttributeResult({self.attribute!r}  "
            f"n={self.num_sequences}  "
            f"mIoU={self.mean_iou:.3f}  "
            f"succ_AUC={self.success_auc:.3f}  "
            f"prec_AUC={self.precision_auc:.3f})"
        )


@dataclass
class AttributeAnalysis:
    """Attribute-stratified performance report for a single tracker."""

    tracker_name: str
    attribute_results: Dict[str, AttributeResult]
    overall: AccuracyMetrics

    @property
    def sorted_by_difficulty(self) -> List[AttributeResult]:
        """Attributes sorted by mean IoU ascending (hardest challenge first)."""
        return sorted(self.attribute_results.values(), key=lambda r: r.mean_iou)

    @property
    def hardest_attribute(self) -> Optional[AttributeResult]:
        """The attribute with the lowest mean IoU, or None if no data."""
        results = self.sorted_by_difficulty
        return results[0] if results else None

    @property
    def easiest_attribute(self) -> Optional[AttributeResult]:
        """The attribute with the highest mean IoU, or None if no data."""
        results = self.sorted_by_difficulty
        return results[-1] if results else None


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------


class AttributeAnalyzer:
    """Analyse tracker performance stratified by sequence attributes.

    Args:
        attribute_map: Optional pre-loaded mapping
            ``{sequence_name: [attr1, attr2, ...]}``.  Can be extended later
            via :meth:`register_sequence` or :meth:`load_otb_attributes`.
    """

    def __init__(
        self, attribute_map: Optional[Dict[str, List[str]]] = None
    ) -> None:
        self._attribute_map: Dict[str, List[str]] = {}
        if attribute_map:
            for seq, attrs in attribute_map.items():
                self._attribute_map[seq] = [a.lower() for a in attrs]

    # ------------------------------------------------------------------
    # Attribute registration
    # ------------------------------------------------------------------

    def register_sequence(self, sequence_name: str, attributes: List[str]) -> None:
        """Tag a sequence with a list of attribute strings.

        Args:
            sequence_name: Name of the sequence (must match keys used in
                :meth:`analyze`).
            attributes: List of attribute strings (case-insensitive).
        """
        self._attribute_map[sequence_name] = [a.lower() for a in attributes]

    def load_otb_attributes(self, attribute_file: str) -> None:
        """Load OTB-style attribute annotations from a CSV text file.

        Expected format — one sequence per line::

            sequence_name,attr1,attr2,...

        Lines starting with ``#`` and blank lines are ignored.

        Args:
            attribute_file: Path to the attribute annotation file.

        Raises:
            FileNotFoundError: If *attribute_file* does not exist.
        """
        if not os.path.isfile(attribute_file):
            raise FileNotFoundError(f"Attribute file not found: {attribute_file}")
        with open(attribute_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    self._attribute_map[parts[0]] = [a.lower() for a in parts[1:]]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        tracker_name: str,
        sequence_ious: Dict[str, np.ndarray],
        sequence_distances: Optional[Dict[str, np.ndarray]] = None,
    ) -> AttributeAnalysis:
        """Compute per-attribute metrics for one tracker.

        Args:
            tracker_name: Display name for the tracker (e.g. ``"MOSSE"``).
            sequence_ious: Mapping ``{sequence_name: per_frame_iou_array}``.
                Each array has shape ``(N,)`` with values in ``[0, 1]``.
            sequence_distances: Optional mapping
                ``{sequence_name: per_frame_centre_distance_array}`` in pixels.
                When provided, precision AUC is computed; otherwise it is 0.

        Returns:
            :class:`AttributeAnalysis` with per-attribute results and overall
            aggregate metrics.
        """
        if sequence_distances is None:
            sequence_distances = {}

        # Overall metrics across all sequences
        all_ious = np.concatenate(list(sequence_ious.values()))
        all_dists = (
            np.concatenate(
                [sequence_distances[s] for s in sequence_ious if s in sequence_distances]
            )
            if sequence_distances
            else None
        )
        overall = self._compute_metrics(all_ious, all_dists)

        # Group sequences by attribute
        attr_to_seqs: Dict[str, List[str]] = {}
        for seq_name, attrs in self._attribute_map.items():
            if seq_name not in sequence_ious:
                continue
            for attr in attrs:
                attr_to_seqs.setdefault(attr, []).append(seq_name)

        attribute_results: Dict[str, AttributeResult] = {}
        for attr, seq_names in attr_to_seqs.items():
            attr_ious = np.concatenate([sequence_ious[s] for s in seq_names])
            attr_dists: Optional[np.ndarray] = None
            dists_parts = [
                sequence_distances[s] for s in seq_names if s in sequence_distances
            ]
            if dists_parts:
                attr_dists = np.concatenate(dists_parts)

            metrics = self._compute_metrics(attr_ious, attr_dists)
            attribute_results[attr] = AttributeResult(
                attribute=attr,
                num_sequences=len(seq_names),
                mean_iou=metrics.mean_iou,
                success_auc=metrics.success_auc,
                precision_auc=metrics.precision_auc,
                sequence_names=list(seq_names),
            )

        return AttributeAnalysis(
            tracker_name=tracker_name,
            attribute_results=attribute_results,
            overall=overall,
        )

    def from_benchmark_result(self, result: object) -> AttributeAnalysis:
        """Build an :class:`AttributeAnalysis` directly from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        This is the recommended integration point when running the full
        benchmark pipeline.  Sequence IoU arrays and centre-distance arrays
        are extracted from the result's :class:`~eovot.benchmark.engine.SequenceResult`
        objects.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` instance.

        Returns:
            :class:`AttributeAnalysis` for the tracker embedded in *result*.
        """
        seq_ious: Dict[str, np.ndarray] = {}
        seq_dists: Dict[str, np.ndarray] = {}

        for sr in result.sequence_results:  # type: ignore[attr-defined]
            seq_ious[sr.sequence_name] = sr.ious
            if sr.center_distances is not None:
                seq_dists[sr.sequence_name] = sr.center_distances

        return self.analyze(
            tracker_name=result.tracker_name,  # type: ignore[attr-defined]
            sequence_ious=seq_ious,
            sequence_distances=seq_dists if seq_dists else None,
        )

    # ------------------------------------------------------------------
    # Multi-tracker comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        analyses: List[AttributeAnalysis],
        metric: str = "mean_iou",
    ) -> Dict[str, Dict[str, float]]:
        """Build an attribute × tracker metric table.

        Args:
            analyses: One :class:`AttributeAnalysis` per tracker.
            metric: Which scalar to populate — ``"mean_iou"``,
                ``"success_auc"``, or ``"precision_auc"``.

        Returns:
            ``{attribute: {tracker_name: value}}`` nested dict.
        """
        table: Dict[str, Dict[str, float]] = {}
        for analysis in analyses:
            for attr, result in analysis.attribute_results.items():
                table.setdefault(attr, {})[analysis.tracker_name] = float(
                    getattr(result, metric)
                )
        return table

    def format_comparison_table(
        self,
        analyses: List[AttributeAnalysis],
        metric: str = "mean_iou",
    ) -> str:
        """Render a Markdown table of per-attribute metrics.

        Args:
            analyses: One :class:`AttributeAnalysis` per tracker.
            metric: Metric column to display (``"mean_iou"`` by default).

        Returns:
            Markdown-formatted comparison table string.
        """
        tracker_names = [a.tracker_name for a in analyses]
        all_attrs = sorted(
            {attr for a in analyses for attr in a.attribute_results}
        )

        header = "| Attribute | " + " | ".join(tracker_names) + " |"
        sep = "|---|" + "|".join(["---"] * len(tracker_names)) + "|"
        rows = [header, sep]

        for attr in all_attrs:
            row = f"| {attr} |"
            for analysis in analyses:
                result = analysis.attribute_results.get(attr)
                row += f" {getattr(result, metric):.3f} |" if result else " — |"
            rows.append(row)

        # Overall row
        overall_row = "| **Overall** |"
        for analysis in analyses:
            val = getattr(analysis.overall, metric)
            overall_row += f" **{val:.3f}** |"
        rows.append(overall_row)

        return "\n".join(rows)

    def difficulty_ranking(
        self,
        analyses: List[AttributeAnalysis],
        metric: str = "mean_iou",
    ) -> List[Tuple[str, float]]:
        """Rank attributes by average difficulty across all trackers.

        Difficulty is measured as *mean metric across trackers* — lower
        values indicate harder challenges.

        Args:
            analyses: One :class:`AttributeAnalysis` per tracker.
            metric: Metric to average (default ``"mean_iou"``).

        Returns:
            List of ``(attribute, average_metric_value)`` tuples sorted
            ascending (hardest first).
        """
        table = self.compare(analyses, metric=metric)
        ranking = [
            (attr, float(np.mean(list(tracker_vals.values()))))
            for attr, tracker_vals in table.items()
        ]
        return sorted(ranking, key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        ious: np.ndarray,
        dists: Optional[np.ndarray] = None,
    ) -> AccuracyMetrics:
        """Compute AccuracyMetrics from pre-computed IoU (and optional dist) arrays."""
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]

        thresholds_iou = np.linspace(0.0, 1.0, 101)
        sr = np.array([(ious > t).mean() for t in thresholds_iou])
        success_auc = float(_trapz(sr, thresholds_iou))

        if dists is not None and len(dists) > 0:
            thresholds_dist = np.linspace(0.0, 50.0, 51)
            pr = np.array([(dists < t).mean() for t in thresholds_dist])
            prec_auc = float(_trapz(pr, thresholds_dist) / thresholds_dist[-1])
        else:
            prec_auc = 0.0

        return AccuracyMetrics(
            mean_iou=float(ious.mean()),
            success_auc=success_auc,
            precision_auc=prec_auc,
        )
