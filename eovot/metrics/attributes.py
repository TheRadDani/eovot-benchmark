"""Sequence attribute detection and per-attribute performance analysis.

VOT benchmarks annotate sequences with challenge attributes (fast motion,
scale variation, occlusion, etc.) to report where each tracker succeeds or
fails.  This module adds that capability to EOVOT using automatic detection
from ground-truth bounding boxes, so no manual annotation is required.

Seven attributes are implemented, matching the most diagnostic dimensions
from the OTB and VOT challenge attribute sets:

+---------------------+-------------------------------------------------------+
| Attribute           | Detection rule (from GT boxes)                        |
+=====================+=======================================================+
| scale_variation     | max/min GT-box area ratio > 4                         |
+---------------------+-------------------------------------------------------+
| fast_motion         | Any frame displacement > 20 % of mean box diagonal    |
+---------------------+-------------------------------------------------------+
| out_plane_rotation  | max(w/h) / min(w/h) > 2 (3-D rotation proxy)         |
+---------------------+-------------------------------------------------------+
| deformation         | Std of (w/h) across sequence > 0.30                  |
+---------------------+-------------------------------------------------------+
| low_resolution      | Mean GT-box area < 400 px²                           |
+---------------------+-------------------------------------------------------+
| partial_occlusion   | Any single-frame area drop > 50 % from prior frame   |
+---------------------+-------------------------------------------------------+
| long_sequence       | Sequence length > 400 frames                         |
+---------------------+-------------------------------------------------------+

Example::

    from eovot.metrics.attributes import AttributeDetector, AttributeAnalyzer

    detector = AttributeDetector()

    # Per-sequence attribute detection (uses stored GT from BenchmarkResult)
    attrs = detector.detect(seq_result.ground_truths, seq_result.sequence_name)
    print(attrs.active_attributes)   # e.g. ['fast_motion', 'scale_variation']

    # Per-attribute breakdown across a full benchmark run
    analyzer = AttributeAnalyzer()
    table = analyzer.breakdown(benchmark_result)
    print(table.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------

#: Human-readable descriptions for all built-in attributes.
ATTRIBUTE_DESCRIPTIONS: Dict[str, str] = {
    "scale_variation": "Max-to-min GT-box area ratio > 4× (target grows or shrinks significantly)",
    "fast_motion": "Any frame-to-frame center displacement > 20 % of mean box diagonal",
    "out_plane_rotation": "Aspect-ratio range ratio > 2× — proxy for 3-D out-of-plane rotation",
    "deformation": "Aspect-ratio std > 0.30 — non-rigid or articulated target",
    "low_resolution": "Mean GT-box area < 400 px² — small or distant target",
    "partial_occlusion": "Single-frame area drop > 50 % from prior frame — target occluded",
    "long_sequence": "Sequence length > 400 frames",
}

#: Ordered list of all attribute names (matches ``ATTRIBUTE_DESCRIPTIONS``).
ALL_ATTRIBUTES: List[str] = list(ATTRIBUTE_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SequenceAttributes:
    """Attribute annotations for a single tracking sequence.

    Attributes:
        sequence_name: Identifier matching :attr:`SequenceResult.sequence_name`.
        attributes: Mapping from attribute name to ``True`` if that attribute
            is present in this sequence, ``False`` otherwise.
    """

    sequence_name: str
    attributes: Dict[str, bool] = field(default_factory=dict)

    @property
    def active_attributes(self) -> List[str]:
        """Return the list of attribute names that are present (``True``)."""
        return [k for k, v in self.attributes.items() if v]

    def has(self, attribute: str) -> bool:
        """Return whether *attribute* is present in this sequence."""
        return bool(self.attributes.get(attribute, False))

    def __str__(self) -> str:
        active = ", ".join(self.active_attributes) or "none"
        return f"SequenceAttributes({self.sequence_name!r}: [{active}])"


@dataclass
class AttributePerformanceTable:
    """Per-attribute accuracy summary for one tracker on a dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the evaluated dataset.
        entries: Mapping from attribute name to a summary dict with at least
            ``"n_sequences"`` (int) and ``"mean_iou"`` (float).  When
            per-sequence :class:`~eovot.metrics.accuracy.AccuracyMetrics` are
            available, ``"success_auc"`` and ``"precision_auc"`` are also
            present.  Attributes absent from all sequences are omitted.
    """

    tracker_name: str
    dataset_name: str
    entries: Dict[str, Dict] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Format the per-attribute breakdown as a Markdown table.

        Returns:
            Multi-line Markdown string, one row per attribute that is present
            in at least one sequence in the dataset.
        """
        if not self.entries:
            return (
                f"No attribute data available for "
                f"{self.tracker_name} on {self.dataset_name}.\n"
            )

        has_auc = any("success_auc" in v for v in self.entries.values())
        title = (
            f"## Per-Attribute Performance: "
            f"{self.tracker_name} on {self.dataset_name}\n"
        )
        if has_auc:
            header = (
                "| Attribute | Seqs | mIoU | Success AUC | Precision AUC |\n"
                "|-----------|-----:|-----:|------------:|--------------:|"
            )
        else:
            header = "| Attribute | Seqs | mIoU |\n|-----------|-----:|-----:|"

        rows = []
        for attr in ALL_ATTRIBUTES:
            v = self.entries.get(attr)
            if v is None:
                continue
            if has_auc:
                sauc = v.get("success_auc", float("nan"))
                pauc = v.get("precision_auc", float("nan"))
                rows.append(
                    f"| {attr} | {v['n_sequences']} "
                    f"| {v['mean_iou']:.4f} "
                    f"| {sauc:.4f} | {pauc:.4f} |"
                )
            else:
                rows.append(
                    f"| {attr} | {v['n_sequences']} | {v['mean_iou']:.4f} |"
                )

        return title + header + "\n" + "\n".join(rows)

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable plain dict."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "entries": {attr: dict(v) for attr, v in self.entries.items()},
        }


# ---------------------------------------------------------------------------
# Attribute detector
# ---------------------------------------------------------------------------

class AttributeDetector:
    """Automatically detect VOT challenge attributes from GT bounding boxes.

    All thresholds are configurable; defaults match OTB benchmark conventions
    where an equivalent rule exists.

    Args:
        sv_ratio_threshold: Scale-variation trigger: max/min area > this.
            Default: ``4.0``.
        fm_diag_fraction: Fast-motion trigger: per-frame displacement /
            mean_box_diagonal > this.  Default: ``0.20`` (20 %).
        opr_ratio_threshold: Out-of-plane-rotation trigger: max(w/h) / min(w/h)
            > this.  Default: ``2.0``.
        def_ar_std_threshold: Deformation trigger: std(w/h) > this.
            Default: ``0.30``.
        lr_area_threshold: Low-resolution trigger: mean box area < this (px²).
            Default: ``400.0``.
        occ_area_drop_fraction: Partial-occlusion trigger: single-frame area
            drop > this fraction of the previous frame's area.  Default: ``0.50``.
        ls_frames_threshold: Long-sequence trigger: length > this.
            Default: ``400``.

    Example::

        det = AttributeDetector(sv_ratio_threshold=2.0, fm_diag_fraction=0.15)
        attrs = det.detect(seq.ground_truth, sequence_name=seq.name)
        print(attrs.active_attributes)
    """

    def __init__(
        self,
        sv_ratio_threshold: float = 4.0,
        fm_diag_fraction: float = 0.20,
        opr_ratio_threshold: float = 2.0,
        def_ar_std_threshold: float = 0.30,
        lr_area_threshold: float = 400.0,
        occ_area_drop_fraction: float = 0.50,
        ls_frames_threshold: int = 400,
    ) -> None:
        self.sv_ratio = sv_ratio_threshold
        self.fm_frac = fm_diag_fraction
        self.opr_ratio = opr_ratio_threshold
        self.def_std = def_ar_std_threshold
        self.lr_area = lr_area_threshold
        self.occ_drop = occ_area_drop_fraction
        self.ls_frames = ls_frames_threshold

    def detect(
        self,
        gt: np.ndarray,
        sequence_name: str = "",
    ) -> SequenceAttributes:
        """Detect all attributes present in a ground-truth box sequence.

        Args:
            gt: ``(N, 4)`` float array of GT boxes in ``(x, y, w, h)`` format.
            sequence_name: Identifier stored in the returned object.

        Returns:
            :class:`SequenceAttributes` with all seven attribute flags set.
        """
        gt = np.asarray(gt, dtype=np.float64)
        n = len(gt)

        # Long sequence — pure length check, no box arithmetic needed.
        if n == 0:
            return SequenceAttributes(
                sequence_name=sequence_name,
                attributes={k: False for k in ALL_ATTRIBUTES},
            )

        attrs: Dict[str, bool] = {}
        attrs["long_sequence"] = n > self.ls_frames

        w = gt[:, 2]
        h = gt[:, 3]
        areas = w * h
        safe_areas = np.where(areas > 1e-6, areas, 1e-6)

        # Scale variation: max/min area ratio.
        attrs["scale_variation"] = bool(safe_areas.max() / safe_areas.min() > self.sv_ratio)

        # Low resolution: mean box area.
        attrs["low_resolution"] = bool(areas.mean() < self.lr_area)

        # Aspect-ratio-based attributes.
        safe_h = np.where(h > 1e-6, h, 1e-6)
        ar = w / safe_h

        ar_min = float(ar.min())
        ar_max = float(ar.max())
        if ar_min > 1e-6:
            attrs["out_plane_rotation"] = bool(ar_max / ar_min > self.opr_ratio)
        else:
            attrs["out_plane_rotation"] = False

        attrs["deformation"] = bool(float(ar.std()) > self.def_std)

        # Fast motion: any frame-to-frame displacement > fraction of mean diagonal.
        if n < 2:
            attrs["fast_motion"] = False
        else:
            centers = gt[:, :2] + gt[:, 2:] / 2.0
            displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            mean_diag = float(np.sqrt(w ** 2 + h ** 2).mean())
            if mean_diag > 1e-6:
                attrs["fast_motion"] = bool((displacements / mean_diag > self.fm_frac).any())
            else:
                attrs["fast_motion"] = False

        # Partial occlusion: any single-frame area drop > threshold.
        if n < 2:
            attrs["partial_occlusion"] = False
        else:
            area_drops = (safe_areas[:-1] - safe_areas[1:]) / safe_areas[:-1]
            attrs["partial_occlusion"] = bool((area_drops > self.occ_drop).any())

        return SequenceAttributes(sequence_name=sequence_name, attributes=attrs)


# ---------------------------------------------------------------------------
# Per-attribute performance analyzer
# ---------------------------------------------------------------------------

class AttributeAnalyzer:
    """Compute per-attribute accuracy breakdown from a benchmark result.

    Groups sequences by the attributes they exhibit and averages per-sequence
    IoU and AUC values within each group.  Attributes are auto-detected from
    the ``ground_truths`` stored in each
    :class:`~eovot.benchmark.engine.SequenceResult` unless pre-computed
    annotations are supplied.

    Args:
        detector: :class:`AttributeDetector` instance used for auto-detection.
            Defaults to a fresh instance with standard thresholds.

    Example::

        analyzer = AttributeAnalyzer()
        table = analyzer.breakdown(benchmark_result)
        print(table.to_markdown())

        # Save as JSON
        import json
        with open("attributes.json", "w") as f:
            json.dump(table.to_dict(), f, indent=2)
    """

    def __init__(self, detector: Optional[AttributeDetector] = None) -> None:
        self._detector = detector if detector is not None else AttributeDetector()

    def breakdown(
        self,
        result: "BenchmarkResult",
        sequence_attributes: Optional[Dict[str, SequenceAttributes]] = None,
    ) -> AttributePerformanceTable:
        """Compute per-attribute performance from a full benchmark result.

        Args:
            result: A :class:`~eovot.benchmark.engine.BenchmarkResult` from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            sequence_attributes: Optional pre-computed attribute mapping
                ``{sequence_name: SequenceAttributes}``.  When ``None``,
                attributes are auto-detected from the ``ground_truths`` stored
                in each :class:`~eovot.benchmark.engine.SequenceResult`.
                Sequences without stored ground-truths are skipped.

        Returns:
            :class:`AttributePerformanceTable` with per-attribute summaries.
            Only attributes present in at least one sequence appear in
            ``entries``.
        """
        if sequence_attributes is None:
            sequence_attributes = self.detect_all(result)

        attr_ious: Dict[str, List[float]] = {a: [] for a in ALL_ATTRIBUTES}
        attr_sauc: Dict[str, List[float]] = {a: [] for a in ALL_ATTRIBUTES}
        attr_pauc: Dict[str, List[float]] = {a: [] for a in ALL_ATTRIBUTES}

        for sr in result.sequence_results:
            sa = sequence_attributes.get(sr.sequence_name)
            if sa is None:
                continue
            for attr in ALL_ATTRIBUTES:
                if sa.has(attr):
                    attr_ious[attr].append(sr.mean_iou)
                    if sr.accuracy_metrics is not None:
                        attr_sauc[attr].append(sr.accuracy_metrics.success_auc)
                        attr_pauc[attr].append(sr.accuracy_metrics.precision_auc)

        entries: Dict[str, Dict] = {}
        for attr in ALL_ATTRIBUTES:
            ious = attr_ious[attr]
            if not ious:
                continue
            entry: Dict = {
                "n_sequences": len(ious),
                "mean_iou": round(float(np.mean(ious)), 4),
            }
            sauc_list = attr_sauc[attr]
            if sauc_list:
                entry["success_auc"] = round(float(np.mean(sauc_list)), 4)
                entry["precision_auc"] = round(float(np.mean(attr_pauc[attr])), 4)
            entries[attr] = entry

        return AttributePerformanceTable(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            entries=entries,
        )

    def detect_all(
        self,
        result: "BenchmarkResult",
    ) -> Dict[str, SequenceAttributes]:
        """Auto-detect attributes for every sequence in a benchmark result.

        Convenience wrapper over :meth:`AttributeDetector.detect` that
        processes all :class:`~eovot.benchmark.engine.SequenceResult` objects.

        Args:
            result: Benchmark result with ``ground_truths`` stored in each
                :class:`~eovot.benchmark.engine.SequenceResult`.

        Returns:
            Mapping from sequence name to :class:`SequenceAttributes`.
            Sequences without stored ground-truths produce entries with all
            attributes set to ``False``.
        """
        out: Dict[str, SequenceAttributes] = {}
        for sr in result.sequence_results:
            gt = sr.ground_truths
            if gt is not None and len(gt) > 0:
                out[sr.sequence_name] = self._detector.detect(gt, sr.sequence_name)
            else:
                out[sr.sequence_name] = SequenceAttributes(
                    sequence_name=sr.sequence_name,
                    attributes={k: False for k in ALL_ATTRIBUTES},
                )
        return out
