"""Multi-tracker attribute breakdown comparison report.

Generates the per-attribute performance table that is standard in every
tracking paper: rows are VOT challenge attributes (fast motion, occlusion,
scale variation, etc.) and columns are tracker names.  Each cell contains the
mean IoU (and optionally success AUC) of that tracker on sequences that
exhibit the given attribute.

This module fills a gap in EOVOT's reporting pipeline: the existing
:class:`~eovot.metrics.attributes.AttributeAnalyzer` computes the breakdown
for a *single* tracker result, but papers always compare multiple trackers
side by side in one table.

Outputs
-------
- **Markdown** — ready to paste into READMEs or GitHub discussions.
- **LaTeX** — drop into a paper with ``\\input{attr_table.tex}``; the best
  value in each row is automatically ``\\textbf`` bolded.
- **CSV** — import into pandas, Excel, or Google Sheets.
- **JSON** — machine-readable for downstream tooling.

Typical usage
-------------
::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.reporting.attribute_report import MultiTrackerAttributeReport

    ds      = SyntheticDataset(num_sequences=20, num_frames=200)
    engine  = BenchmarkEngine(verbose=False)

    results = [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
    ]

    report = MultiTrackerAttributeReport(results)
    print(report.to_markdown())
    report.save_latex("results/attr_table.tex")
    report.save_csv("results/attr_table.csv")
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..metrics.attributes import (
    ALL_ATTRIBUTES,
    ATTRIBUTE_DESCRIPTIONS,
    AttributeAnalyzer,
    AttributeDetector,
    SequenceAttributes,
)

try:
    from ..benchmark.engine import BenchmarkResult
except ImportError:  # avoid circular imports during type checking
    BenchmarkResult = None  # type: ignore[assignment,misc]


class MultiTrackerAttributeReport:
    """Aggregate per-attribute performance across multiple trackers.

    Builds and exports the N-tracker × M-attribute comparison table that is
    standard in visual tracking papers.  Attributes are auto-detected from the
    ground-truth boxes stored in each :class:`~eovot.benchmark.engine.SequenceResult`;
    sequences without stored ground truths are silently skipped.

    Args:
        results: One :class:`~eovot.benchmark.engine.BenchmarkResult` per
            tracker.  All results should be from the same dataset for the
            comparison to be meaningful.
        metric: Which metric to compare.  One of:
            - ``"mean_iou"`` (default)
            - ``"success_auc"``
            - ``"precision_auc"``
        detector: Custom :class:`~eovot.metrics.attributes.AttributeDetector`.
            Defaults to a fresh instance with standard thresholds.
        min_sequences: Minimum number of sequences that must exhibit an
            attribute for it to appear in the table.  Default: ``1``.

    Example::

        report = MultiTrackerAttributeReport(results, metric="success_auc")
        print(report.to_markdown())
        report.save_latex("paper/tables/attr_table.tex")
    """

    def __init__(
        self,
        results: List["BenchmarkResult"],
        metric: str = "mean_iou",
        detector: Optional[AttributeDetector] = None,
        min_sequences: int = 1,
    ) -> None:
        valid_metrics = {"mean_iou", "success_auc", "precision_auc"}
        if metric not in valid_metrics:
            raise ValueError(
                f"metric must be one of {sorted(valid_metrics)}, got {metric!r}"
            )
        if not results:
            raise ValueError("results must not be empty")

        self.metric = metric
        self.min_sequences = int(min_sequences)
        self._analyzer = AttributeAnalyzer(
            detector=detector if detector is not None else AttributeDetector()
        )

        self._tracker_names: List[str] = [r.tracker_name for r in results]
        self._dataset_name: str = results[0].dataset_name

        # Build shared attribute map from the first result (all results use the
        # same dataset, so attribute detection is identical).
        self._seq_attrs: Dict[str, SequenceAttributes] = self._analyzer.detect_all(
            results[0]
        )

        # Per-tracker, per-attribute performance
        self._table: Dict[str, Dict[str, Optional[float]]] = {}
        self._counts: Dict[str, Dict[str, int]] = {}

        for result in results:
            name = result.tracker_name
            self._table[name] = {}
            self._counts[name] = {}
            breakdown = self._analyzer.breakdown(result, self._seq_attrs)
            for attr in ALL_ATTRIBUTES:
                entry = breakdown.entries.get(attr)
                if entry is None or entry.get("n_sequences", 0) < self.min_sequences:
                    self._table[name][attr] = None
                    self._counts[name][attr] = 0
                else:
                    self._table[name][attr] = float(entry.get(metric, 0.0))
                    self._counts[name][attr] = int(entry.get("n_sequences", 0))

    # ------------------------------------------------------------------
    # Active attributes (present in at least one tracker's dataset)
    # ------------------------------------------------------------------

    def active_attributes(self) -> List[str]:
        """Return attributes present (non-None, ≥ min_sequences) for ≥ 1 tracker."""
        active = []
        for attr in ALL_ATTRIBUTES:
            for name in self._tracker_names:
                if self._table[name].get(attr) is not None:
                    active.append(attr)
                    break
        return active

    # ------------------------------------------------------------------
    # Cell helpers
    # ------------------------------------------------------------------

    def _best_in_row(self, attr: str) -> Optional[float]:
        """Return the highest metric value across all trackers for *attr*."""
        vals = [
            self._table[name][attr]
            for name in self._tracker_names
            if self._table[name].get(attr) is not None
        ]
        return max(vals) if vals else None

    def _cell_value(self, tracker: str, attr: str) -> Optional[float]:
        return self._table[tracker].get(attr)

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def to_markdown(self, show_counts: bool = False) -> str:
        """Return a Markdown table of per-attribute performance.

        Args:
            show_counts: If ``True``, append ``(N)`` to each cell showing the
                number of sequences that exhibit the attribute.

        Returns:
            Multi-line Markdown string.
        """
        attrs = self.active_attributes()
        if not attrs:
            return "No attribute data available.\n"

        metric_label = {
            "mean_iou": "mIoU",
            "success_auc": "Succ. AUC",
            "precision_auc": "Prec. AUC",
        }[self.metric]

        col_names = ["Attribute"] + self._tracker_names
        sep = ["-----------"] + ["---------:" for _ in self._tracker_names]
        header = "| " + " | ".join(col_names) + " |"
        sep_row = "| " + " | ".join(sep) + " |"

        rows = [
            f"## Per-Attribute {metric_label}: {self._dataset_name}",
            "",
            header,
            sep_row,
        ]

        for attr in attrs:
            best = self._best_in_row(attr)
            cells = [attr]
            for name in self._tracker_names:
                val = self._cell_value(name, attr)
                if val is None:
                    cells.append("—")
                else:
                    fmt = f"{val:.4f}"
                    if show_counts:
                        n = self._counts[name].get(attr, 0)
                        fmt = f"{fmt} ({n})"
                    if best is not None and abs(val - best) < 1e-9:
                        fmt = f"**{fmt}**"
                    cells.append(fmt)
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)

    # ------------------------------------------------------------------
    # LaTeX export
    # ------------------------------------------------------------------

    def to_latex(
        self,
        caption: str = "",
        label: str = "tab:attr-breakdown",
        bold_best: bool = True,
    ) -> str:
        """Return a LaTeX ``tabular`` environment string.

        Args:
            caption: Table caption.  Wrapped in a ``table`` float when
                non-empty.
            label: LaTeX label for ``\\ref{}``.
            bold_best: Bold the best (highest) value in each row.

        Returns:
            LaTeX string, ready to ``\\input{}`` or paste into a paper.
        """
        attrs = self.active_attributes()
        n_trackers = len(self._tracker_names)

        col_spec = "l" + "r" * n_trackers
        header_cells = ["Attribute"] + self._tracker_names
        header_row = " & ".join(header_cells) + r" \\"

        data_rows = []
        for attr in attrs:
            best = self._best_in_row(attr) if bold_best else None
            cells = [attr.replace("_", r"\_")]
            for name in self._tracker_names:
                val = self._cell_value(name, attr)
                if val is None:
                    cells.append("—")
                else:
                    fmt = f"{val:.4f}"
                    if best is not None and abs(val - best) < 1e-9:
                        fmt = r"\textbf{" + fmt + "}"
                    cells.append(fmt)
            data_rows.append(" & ".join(cells) + r" \\")

        body = "\n".join([
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
            header_row,
            r"\midrule",
            "\n".join(data_rows),
            r"\bottomrule",
            r"\end{tabular}",
        ])

        if not caption:
            return body

        metric_label = {
            "mean_iou": "mIoU",
            "success_auc": "Success AUC",
            "precision_auc": "Precision AUC",
        }.get(self.metric, self.metric)

        auto_caption = (
            caption
            or f"Per-attribute {metric_label} comparison on {self._dataset_name}."
        )

        return "\n".join([
            r"\begin{table}[t]",
            r"\centering",
            body,
            r"\caption{" + auto_caption + "}",
            r"\label{" + label + "}",
            r"\end{table}",
        ])

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(self) -> str:
        """Return the table as a CSV string.

        Returns:
            CSV text with one row per attribute and one column per tracker.
        """
        attrs = self.active_attributes()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["attribute", "n_sequences"] + self._tracker_names)
        for attr in attrs:
            # Use counts from the first tracker (same dataset → same count)
            n = max(
                self._counts[name].get(attr, 0) for name in self._tracker_names
            )
            cells = [attr, str(n)]
            for name in self._tracker_names:
                val = self._cell_value(name, attr)
                cells.append(f"{val:.6f}" if val is not None else "")
            writer.writerow(cells)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable dict.

        Structure::

            {
                "dataset": "...",
                "metric": "mean_iou",
                "trackers": ["MOSSE", "KCF"],
                "attributes": {
                    "scale_variation": {
                        "n_sequences": 3,
                        "MOSSE": 0.421,
                        "KCF":   0.517
                    },
                    ...
                }
            }
        """
        attrs = self.active_attributes()
        attr_data: Dict[str, Dict] = {}
        for attr in attrs:
            n = max(
                self._counts[name].get(attr, 0) for name in self._tracker_names
            )
            entry: Dict = {"n_sequences": n, "description": ATTRIBUTE_DESCRIPTIONS.get(attr, "")}
            for name in self._tracker_names:
                val = self._cell_value(name, attr)
                entry[name] = round(val, 6) if val is not None else None
            attr_data[attr] = entry

        return {
            "dataset": self._dataset_name,
            "metric": self.metric,
            "trackers": list(self._tracker_names),
            "attributes": attr_data,
        }

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_markdown(self, path: Union[str, Path], **kwargs) -> Path:
        """Write the Markdown table to *path*.

        Args:
            path: Destination file path (a ``.md`` extension is added if absent).
            **kwargs: Forwarded to :meth:`to_markdown`.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(**kwargs), encoding="utf-8")
        return p

    def save_latex(
        self, path: Union[str, Path], caption: str = "", label: str = "tab:attr-breakdown"
    ) -> Path:
        """Write the LaTeX table to *path*.

        Args:
            path: Destination file path (a ``.tex`` extension is added if absent).
            caption: Optional table caption.
            label: LaTeX label.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".tex")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_latex(caption=caption, label=label), encoding="utf-8")
        return p

    def save_csv(self, path: Union[str, Path]) -> Path:
        """Write the CSV table to *path*.

        Args:
            path: Destination file path (a ``.csv`` extension is added if absent).

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_csv(), encoding="utf-8")
        return p

    def save_json(self, path: Union[str, Path]) -> Path:
        """Write the JSON report to *path*.

        Args:
            path: Destination file path (a ``.json`` extension is added if absent).

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------------

    def overall_ranking(self) -> List[Tuple[str, float]]:
        """Rank trackers by their mean metric across all active attributes.

        Returns:
            List of ``(tracker_name, mean_metric)`` tuples, sorted descending.
        """
        attrs = self.active_attributes()
        ranking = []
        for name in self._tracker_names:
            vals = [
                self._table[name][attr]
                for attr in attrs
                if self._table[name].get(attr) is not None
            ]
            mean_val = float(np.mean(vals)) if vals else 0.0
            ranking.append((name, mean_val))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def __repr__(self) -> str:
        trackers = ", ".join(self._tracker_names)
        return (
            f"MultiTrackerAttributeReport("
            f"trackers=[{trackers}], "
            f"metric={self.metric!r}, "
            f"dataset={self._dataset_name!r})"
        )
