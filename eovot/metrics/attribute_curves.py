"""Per-attribute success and precision curves.

Extends :class:`~eovot.metrics.attributes.AttributeAnalyzer` by computing
the full curve arrays — not just AUC scalars — for each VOT challenge
attribute.  Pooling all per-sequence IoU/distance values belonging to each
attribute into a single curve matches the OTB evaluation protocol and allows
multi-attribute overlay plots.

Example::

    from eovot.metrics.attribute_curves import AttributeCurveAnalyzer

    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(benchmark_result)

    for attr, curve in curves.items():
        print(f"{attr}: success_AUC={curve.success_auc:.3f}, "
              f"precision_AUC={curve.precision_auc:.3f}, "
              f"n_frames={curve.n_frames}")

    # Save to disk
    curves.save_json("results/attribute_curves.json")
    curves.save_csv("results/attribute_curves.csv")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

import numpy as np

from .attributes import ALL_ATTRIBUTES, AttributeAnalyzer, SequenceAttributes
from .accuracy import MetricsEngine

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Per-attribute curve dataclass
# ---------------------------------------------------------------------------

@dataclass
class AttributeCurve:
    """Success and precision curve arrays for a single VOT attribute.

    Attributes:
        attribute:        Attribute name (one of :data:`~eovot.metrics.attributes.ALL_ATTRIBUTES`).
        n_sequences:      Number of sequences exhibiting this attribute.
        n_frames:         Total number of frames pooled across those sequences.
        success_thresholds:   IoU thresholds (shape ``(T,)``).
        success_rates:        Fraction of frames with IoU > threshold (shape ``(T,)``).
        precision_thresholds: Distance thresholds in pixels (shape ``(P,)``).
        precision_rates:      Fraction of frames within threshold (shape ``(P,)``).
        success_auc:      AUC of the success curve.
        precision_auc:    Normalised AUC of the precision curve (0 → 50 px).
    """

    attribute: str
    n_sequences: int
    n_frames: int
    success_thresholds: np.ndarray
    success_rates: np.ndarray
    precision_thresholds: np.ndarray
    precision_rates: np.ndarray
    success_auc: float
    precision_auc: float

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return {
            "attribute": self.attribute,
            "n_sequences": self.n_sequences,
            "n_frames": self.n_frames,
            "success_auc": round(self.success_auc, 6),
            "precision_auc": round(self.precision_auc, 6),
            "success_curve": {
                "thresholds": self.success_thresholds.tolist(),
                "rates": self.success_rates.tolist(),
            },
            "precision_curve": {
                "thresholds": self.precision_thresholds.tolist(),
                "rates": self.precision_rates.tolist(),
            },
        }


# ---------------------------------------------------------------------------
# Container for the full per-attribute result
# ---------------------------------------------------------------------------

@dataclass
class AttributeCurveResult:
    """Per-attribute success/precision curves for one tracker on one dataset.

    Attributes:
        tracker_name: Name of the evaluated tracker.
        dataset_name: Name of the evaluated dataset.
        curves:       Mapping from attribute name to :class:`AttributeCurve`.
                      Only attributes present in at least one sequence appear.
    """

    tracker_name: str
    dataset_name: str
    curves: Dict[str, AttributeCurve] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Iteration helpers
    # -----------------------------------------------------------------------

    def __iter__(self) -> Iterator[str]:
        return iter(self.curves)

    def __len__(self) -> int:
        return len(self.curves)

    def items(self):
        """Iterate over ``(attribute_name, AttributeCurve)`` pairs."""
        return self.curves.items()

    def __getitem__(self, attr: str) -> AttributeCurve:
        return self.curves[attr]

    def get(self, attr: str, default=None):
        return self.curves.get(attr, default)

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "curves": {attr: curve.to_dict() for attr, curve in self.curves.items()},
        }

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        """Write curves to a JSON file.

        Args:
            path:   Output file path.
            indent: JSON indentation level.

        Returns:
            Resolved :class:`~pathlib.Path` of the written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=indent))
        return out.resolve()

    def save_csv(self, path: str | Path) -> Path:
        """Write per-attribute scalar summaries to a CSV file.

        Each row corresponds to one attribute and contains
        ``attribute, n_sequences, n_frames, success_auc, precision_auc``.

        Args:
            path: Output file path.

        Returns:
            Resolved :class:`~pathlib.Path` of the written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["attribute", "n_sequences", "n_frames", "success_auc", "precision_auc"]
            )
            for attr in ALL_ATTRIBUTES:
                curve = self.curves.get(attr)
                if curve is None:
                    continue
                writer.writerow(
                    [
                        curve.attribute,
                        curve.n_sequences,
                        curve.n_frames,
                        round(curve.success_auc, 6),
                        round(curve.precision_auc, 6),
                    ]
                )
        return out.resolve()

    def summary_table(self) -> str:
        """Return a Markdown-formatted summary of per-attribute AUC scalars."""
        if not self.curves:
            return (
                f"No attribute curves available for "
                f"{self.tracker_name} on {self.dataset_name}.\n"
            )
        lines = [
            f"## Per-Attribute Curves: {self.tracker_name} on {self.dataset_name}",
            "",
            "| Attribute | Seqs | Frames | Success AUC | Precision AUC |",
            "|-----------|-----:|-------:|------------:|--------------:|",
        ]
        for attr in ALL_ATTRIBUTES:
            c = self.curves.get(attr)
            if c is None:
                continue
            lines.append(
                f"| {attr} | {c.n_sequences} | {c.n_frames} "
                f"| {c.success_auc:.4f} | {c.precision_auc:.4f} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AttributeCurveAnalyzer:
    """Compute per-attribute success and precision curves from a benchmark result.

    Unlike :class:`~eovot.metrics.attributes.AttributeAnalyzer`, which
    averages per-sequence AUC scalars, this class **pools all frames** that
    belong to sequences exhibiting an attribute before computing the curve.
    Pooling matches the OTB benchmark protocol and produces smoother curves
    on datasets with many short sequences.

    Args:
        detector:           :class:`~eovot.metrics.attributes.AttributeDetector`
            used for auto-detection.  ``None`` uses default thresholds.
        success_thresholds: IoU thresholds for success curves.
            Default: ``np.linspace(0, 1, 101)``.
        precision_thresholds: Distance thresholds (px) for precision curves.
            Default: ``np.linspace(0, 50, 51)``.

    Example::

        from eovot.metrics.attribute_curves import AttributeCurveAnalyzer

        analyzer = AttributeCurveAnalyzer()
        result = analyzer.compute(benchmark_result)
        print(result.summary_table())
        result.save_json("curves.json")
        result.save_csv("curves.csv")
    """

    def __init__(
        self,
        detector=None,
        success_thresholds: Optional[np.ndarray] = None,
        precision_thresholds: Optional[np.ndarray] = None,
    ) -> None:
        self._attr_analyzer = AttributeAnalyzer(detector=detector)
        self._engine = MetricsEngine()
        self._s_thr = (
            success_thresholds
            if success_thresholds is not None
            else np.linspace(0.0, 1.0, 101)
        )
        self._p_thr = (
            precision_thresholds
            if precision_thresholds is not None
            else np.linspace(0.0, 50.0, 51)
        )

    def compute(
        self,
        result: "BenchmarkResult",
        sequence_attributes: Optional[Dict[str, SequenceAttributes]] = None,
    ) -> AttributeCurveResult:
        """Compute per-attribute success and precision curves.

        Args:
            result:               A :class:`~eovot.benchmark.engine.BenchmarkResult`.
            sequence_attributes:  Optional pre-computed attribute annotations.
                When ``None``, attributes are auto-detected from each
                sequence's stored ground-truths.

        Returns:
            :class:`AttributeCurveResult` containing one :class:`AttributeCurve`
            per attribute that appears in at least one sequence.  Sequences
            missing predictions or ground-truths are silently skipped.
        """
        if sequence_attributes is None:
            sequence_attributes = self._attr_analyzer.detect_all(result)

        # Pool IoU values and (pred, gt) pairs per attribute.
        attr_ious: Dict[str, List[np.ndarray]] = {a: [] for a in ALL_ATTRIBUTES}
        attr_preds: Dict[str, List[np.ndarray]] = {a: [] for a in ALL_ATTRIBUTES}
        attr_gts: Dict[str, List[np.ndarray]] = {a: [] for a in ALL_ATTRIBUTES}
        attr_seq_counts: Dict[str, int] = {a: 0 for a in ALL_ATTRIBUTES}

        for sr in result.sequence_results:
            if sr.predictions is None or sr.ground_truths is None:
                continue
            sa = sequence_attributes.get(sr.sequence_name)
            if sa is None:
                continue
            for attr in ALL_ATTRIBUTES:
                if sa.has(attr):
                    attr_ious[attr].append(sr.ious)
                    attr_preds[attr].append(sr.predictions)
                    attr_gts[attr].append(sr.ground_truths)
                    attr_seq_counts[attr] += 1

        curves: Dict[str, AttributeCurve] = {}
        try:
            _trapz = np.trapezoid  # numpy ≥ 2.0
        except AttributeError:
            _trapz = np.trapz  # numpy < 2.0

        for attr in ALL_ATTRIBUTES:
            if attr_seq_counts[attr] == 0:
                continue

            pooled_ious = np.concatenate(attr_ious[attr])
            pooled_preds = np.concatenate(attr_preds[attr], axis=0)
            pooled_gts = np.concatenate(attr_gts[attr], axis=0)

            _, s_rates = self._engine.success_curve(pooled_ious, self._s_thr)
            s_auc = float(_trapz(s_rates, self._s_thr))

            _, p_rates = self._engine.precision_curve(pooled_preds, pooled_gts, self._p_thr)
            p_auc = (
                float(_trapz(p_rates, self._p_thr) / self._p_thr[-1])
                if self._p_thr[-1] > 0
                else 0.0
            )

            curves[attr] = AttributeCurve(
                attribute=attr,
                n_sequences=attr_seq_counts[attr],
                n_frames=int(len(pooled_ious)),
                success_thresholds=self._s_thr.copy(),
                success_rates=s_rates,
                precision_thresholds=self._p_thr.copy(),
                precision_rates=p_rates,
                success_auc=s_auc,
                precision_auc=p_auc,
            )

        return AttributeCurveResult(
            tracker_name=result.tracker_name,
            dataset_name=result.dataset_name,
            curves=curves,
        )
