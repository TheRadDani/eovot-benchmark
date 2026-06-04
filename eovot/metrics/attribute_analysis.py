"""Sequence attribute analysis for EOVOT tracker evaluation.

Decomposes benchmark results by sequence-level attributes (e.g. fast_motion,
occlusion, scale_variation) to reveal per-attribute tracker strengths and
weaknesses — a standard analysis step in VOT research papers.

Standard OTB100 attributes
--------------------------
    IV   – Illumination Variation
    SV   – Scale Variation
    OCC  – Occlusion
    DEF  – Deformation
    MB   – Motion Blur
    FM   – Fast Motion
    IPR  – In-Plane Rotation
    OPR  – Out-of-Plane Rotation
    OV   – Out of View
    BC   – Background Clutter
    LR   – Low Resolution

The module is attribute-label-agnostic: supply whatever label set your dataset
provides, or use :func:`assign_synthetic_attributes` to generate labels for
SyntheticDataset experiments.

Typical usage::

    from eovot.metrics.attribute_analysis import AttributeAnalyzer

    # Map each sequence to a set of attribute tags
    seq_attributes = {
        "car1":    {"FM", "SV"},
        "person1": {"OCC", "DEF"},
        "ball1":   {"FM", "MB"},
    }

    # Per-sequence data: {seq_name: {"preds": ndarray, "gts": ndarray}}
    seq_data = {
        "car1":    {"preds": preds_car1,    "gts": gts_car1},
        "person1": {"preds": preds_person1, "gts": gts_person1},
        "ball1":   {"preds": preds_ball1,   "gts": gts_ball1},
    }

    analyzer = AttributeAnalyzer()
    report = analyzer.analyze(seq_data, seq_attributes, tracker_name="MOSSE")
    print(report.to_markdown())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from .accuracy import MetricsEngine


#: Canonical two-letter codes and descriptions for OTB100 sequence attributes.
OTB_ATTRIBUTES: Dict[str, str] = {
    "IV":  "Illumination Variation",
    "SV":  "Scale Variation",
    "OCC": "Occlusion",
    "DEF": "Deformation",
    "MB":  "Motion Blur",
    "FM":  "Fast Motion",
    "IPR": "In-Plane Rotation",
    "OPR": "Out-of-Plane Rotation",
    "OV":  "Out of View",
    "BC":  "Background Clutter",
    "LR":  "Low Resolution",
}


@dataclass
class AttributeMetrics:
    """Per-attribute accuracy summary for one tracker."""

    attribute: str
    """Attribute code or label (e.g. ``"FM"``, ``"OCC"``)."""

    num_sequences: int
    """Number of sequences carrying this attribute in the evaluation set."""

    mean_iou: float
    """Mean IoU across all frames of sequences tagged with this attribute."""

    success_auc: float
    """Area under the success curve for sequences with this attribute."""

    precision_auc: float
    """Normalised AUC of the precision curve for these sequences."""

    def __str__(self) -> str:
        return (
            f"AttributeMetrics({self.attribute!r:6s}  "
            f"n={self.num_sequences:3d}  "
            f"mIoU={self.mean_iou:.4f}  "
            f"AUC={self.success_auc:.4f}  "
            f"prec={self.precision_auc:.4f})"
        )


@dataclass
class AttributeReport:
    """Full per-attribute analysis result for one tracker.

    Attributes:
        tracker_name:    Identifier of the evaluated tracker.
        per_attribute:   Mapping ``{attribute → AttributeMetrics}``.
        best_attribute:  Attribute with the highest success AUC.
        worst_attribute: Attribute with the lowest success AUC.
        coverage:        Total number of (sequence, attribute) pairs analysed.
    """

    tracker_name: str
    per_attribute: Dict[str, AttributeMetrics]
    best_attribute: Optional[str] = None
    worst_attribute: Optional[str] = None
    coverage: int = 0

    def to_markdown(self, full_names: Optional[Dict[str, str]] = None) -> str:
        """Render a Markdown table of per-attribute metrics.

        Args:
            full_names: Optional mapping of attribute code → description.
                Defaults to :data:`OTB_ATTRIBUTES`.

        Returns:
            Multi-line Markdown string ready for embedding in reports.
        """
        names = full_names if full_names is not None else OTB_ATTRIBUTES
        lines = [
            f"## Attribute Analysis — {self.tracker_name}\n",
            "| Attribute | Description | Sequences | mIoU | Success AUC | Precision AUC |",
            "|-----------|-------------|----------:|-----:|------------:|--------------:|",
        ]
        for attr, m in sorted(
            self.per_attribute.items(),
            key=lambda kv: kv[1].success_auc,
            reverse=True,
        ):
            desc = names.get(attr, "—")
            lines.append(
                f"| {attr} | {desc} | {m.num_sequences} "
                f"| {m.mean_iou:.4f} | {m.success_auc:.4f} | {m.precision_auc:.4f} |"
            )
        if self.best_attribute:
            lines.append(f"\n**Best attribute:** {self.best_attribute}")
        if self.worst_attribute:
            lines.append(f"**Worst attribute:** {self.worst_attribute}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialise the report to a plain dict for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "best_attribute": self.best_attribute,
            "worst_attribute": self.worst_attribute,
            "coverage": self.coverage,
            "per_attribute": {
                attr: {
                    "num_sequences": m.num_sequences,
                    "mean_iou": round(m.mean_iou, 4),
                    "success_auc": round(m.success_auc, 4),
                    "precision_auc": round(m.precision_auc, 4),
                }
                for attr, m in self.per_attribute.items()
            },
        }


class AttributeAnalyzer:
    """Stratify tracker accuracy by sequence-level attribute labels.

    Aggregates per-frame predictions and ground truths across all sequences
    sharing a given attribute tag, then computes standard VOT accuracy
    metrics (mIoU, success AUC, precision AUC) for that attribute subset.
    This follows the evaluation protocol used in OTB and similar benchmarks
    where per-attribute breakdown is a standard table in tracker papers.

    Args:
        metrics_engine: Pre-configured :class:`~eovot.metrics.accuracy.MetricsEngine`
            instance.  A default instance is created when ``None``.

    Example::

        analyzer = AttributeAnalyzer()
        seq_data = {
            "car1":    {"preds": np.zeros((100, 4)), "gts": np.zeros((100, 4))},
            "person1": {"preds": np.zeros((80, 4)),  "gts": np.zeros((80, 4))},
        }
        seq_attributes = {"car1": {"FM", "SV"}, "person1": {"OCC"}}
        report = analyzer.analyze(seq_data, seq_attributes, tracker_name="MOSSE")
        print(report.to_markdown())
    """

    def __init__(self, metrics_engine: Optional[MetricsEngine] = None) -> None:
        self._engine = metrics_engine if metrics_engine is not None else MetricsEngine()

    def analyze(
        self,
        seq_data: Dict[str, Dict[str, np.ndarray]],
        seq_attributes: Dict[str, Set[str]],
        tracker_name: str = "",
    ) -> AttributeReport:
        """Compute per-attribute accuracy metrics for one tracker.

        Sequences that appear in ``seq_attributes`` but not in ``seq_data``
        are silently skipped.  Sequences in ``seq_data`` with no attribute
        mapping are excluded from per-attribute breakdowns but do not cause
        an error.

        Args:
            seq_data: ``{seq_name → {"preds": (N, 4) ndarray, "gts": (N, 4) ndarray}}``.
            seq_attributes: ``{seq_name → set of attribute tags}``.
            tracker_name: Identifier embedded in the returned report.

        Returns:
            :class:`AttributeReport` with per-attribute breakdowns.
        """
        # Group sequences by attribute
        attr_seqs: Dict[str, List[str]] = {}
        for seq_name, attrs in seq_attributes.items():
            if seq_name not in seq_data:
                continue
            for attr in attrs:
                attr_seqs.setdefault(attr, []).append(seq_name)

        per_attribute: Dict[str, AttributeMetrics] = {}
        coverage = 0

        for attr, seq_names in sorted(attr_seqs.items()):
            preds_chunks: List[np.ndarray] = []
            gts_chunks: List[np.ndarray] = []

            for seq_name in seq_names:
                entry = seq_data[seq_name]
                preds_chunks.append(np.asarray(entry["preds"], dtype=np.float64))
                gts_chunks.append(np.asarray(entry["gts"], dtype=np.float64))
                coverage += 1

            preds_all = np.concatenate(preds_chunks, axis=0)
            gts_all = np.concatenate(gts_chunks, axis=0)

            acc = self._engine.compute_all(preds_all, gts_all)

            per_attribute[attr] = AttributeMetrics(
                attribute=attr,
                num_sequences=len(seq_names),
                mean_iou=acc.mean_iou,
                success_auc=acc.success_auc,
                precision_auc=acc.precision_auc,
            )

        best = (
            max(per_attribute, key=lambda a: per_attribute[a].success_auc)
            if per_attribute else None
        )
        worst = (
            min(per_attribute, key=lambda a: per_attribute[a].success_auc)
            if per_attribute else None
        )

        return AttributeReport(
            tracker_name=tracker_name,
            per_attribute=per_attribute,
            best_attribute=best,
            worst_attribute=worst,
            coverage=coverage,
        )

    def compare(
        self,
        tracker_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        seq_attributes: Dict[str, Set[str]],
    ) -> str:
        """Build a Markdown cross-tracker comparison table per attribute.

        Runs :meth:`analyze` for each tracker and assembles a single table
        with one column per tracker, one row per attribute, showing success
        AUC — the primary VOT scalar.

        Args:
            tracker_data: ``{tracker_name → seq_data}`` where ``seq_data``
                matches the format used in :meth:`analyze`.
            seq_attributes: ``{seq_name → set of attribute tags}``.

        Returns:
            Multi-line Markdown string with the comparison table.
        """
        reports = {
            name: self.analyze(data, seq_attributes, tracker_name=name)
            for name, data in tracker_data.items()
        }

        all_attrs = sorted(
            {attr for r in reports.values() for attr in r.per_attribute}
        )
        tracker_names = list(reports.keys())

        col_header = " | ".join(f"{t} (AUC)" for t in tracker_names)
        sep = " | ".join(":-----------:" for _ in tracker_names)
        lines = [
            "## Cross-Tracker Attribute Comparison — Success AUC\n",
            f"| Attribute | {col_header} |",
            f"|-----------|{sep}|",
        ]
        for attr in all_attrs:
            vals = []
            for name in tracker_names:
                m = reports[name].per_attribute.get(attr)
                vals.append(f"{m.success_auc:.4f}" if m is not None else "—")
            lines.append(f"| {attr} | {' | '.join(vals)} |")

        return "\n".join(lines)

    def rank_by_attribute(
        self,
        tracker_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        seq_attributes: Dict[str, Set[str]],
        metric: str = "success_auc",
    ) -> Dict[str, List[str]]:
        """Rank trackers per attribute from best to worst.

        Args:
            tracker_data: ``{tracker_name → seq_data}``.
            seq_attributes: ``{seq_name → set of attribute tags}``.
            metric: One of ``"success_auc"``, ``"mean_iou"``, ``"precision_auc"``.

        Returns:
            ``{attribute → [tracker_name, ...]}`` ordered best-to-worst.
        """
        if metric not in ("success_auc", "mean_iou", "precision_auc"):
            raise ValueError(f"Unknown metric {metric!r}. Choose from: success_auc, mean_iou, precision_auc.")

        reports = {
            name: self.analyze(data, seq_attributes, tracker_name=name)
            for name, data in tracker_data.items()
        }

        all_attrs = sorted(
            {attr for r in reports.values() for attr in r.per_attribute}
        )

        ranking: Dict[str, List[str]] = {}
        for attr in all_attrs:
            scores = []
            for name, report in reports.items():
                m = report.per_attribute.get(attr)
                if m is not None:
                    scores.append((name, getattr(m, metric)))
            scores.sort(key=lambda x: x[1], reverse=True)
            ranking[attr] = [t for t, _ in scores]

        return ranking


def assign_synthetic_attributes(
    sequence_names: List[str],
    seed: int = 42,
    attribute_pool: Optional[List[str]] = None,
    min_attrs: int = 1,
    max_attrs: int = 3,
) -> Dict[str, Set[str]]:
    """Randomly assign VOT-style attributes to sequences for testing or demos.

    Useful when working with :class:`~eovot.datasets.synthetic.SyntheticDataset`
    or any dataset without pre-labelled sequence attributes.  Assignment is
    deterministic given the same ``seed``.

    Args:
        sequence_names: List of sequence identifiers to annotate.
        seed: RNG seed for reproducibility.
        attribute_pool: Attribute labels to sample from.  Defaults to the
            11 canonical OTB100 attribute codes.
        min_attrs: Minimum attributes per sequence.  Default: ``1``.
        max_attrs: Maximum attributes per sequence.  Default: ``3``.

    Returns:
        Mapping ``{sequence_name → set of attribute tags}``.
    """
    if attribute_pool is None:
        attribute_pool = list(OTB_ATTRIBUTES.keys())

    rng = np.random.default_rng(seed)
    result: Dict[str, Set[str]] = {}

    for name in sequence_names:
        n_attrs = int(rng.integers(min_attrs, max_attrs + 1))
        n_attrs = min(n_attrs, len(attribute_pool))
        chosen = rng.choice(attribute_pool, size=n_attrs, replace=False).tolist()
        result[name] = set(chosen)

    return result
