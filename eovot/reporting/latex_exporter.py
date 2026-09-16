"""LaTeX table exporter for EOVOT benchmark results.

Generates camera-ready booktabs-style LaTeX tables from
:class:`~eovot.benchmark.engine.BenchmarkResult` objects for IEEE, ACM,
NeurIPS, CVPR, and ECCV paper submissions.

Features
--------
- Single-dataset table (one row per tracker)
- Multi-dataset grouped table (rows separated by \\midrule)
- Per-column best/second-best highlighting (\\textbf / \\underline)
- Metric column selection and custom ordering
- \\caption and \\label injection
- Standalone .tex file export with preamble hints

Required LaTeX packages::

    \\usepackage{booktabs}
    \\usepackage{multirow}  % multi-dataset tables only
    \\usepackage{amsmath}   % \\text inside math mode

Typical usage::

    from eovot.reporting.latex_exporter import LatexTableExporter

    exporter = LatexTableExporter()
    table = exporter.single_dataset_table(
        results=[result_kcf, result_mosse, result_csrt],
        metrics=["success_auc", "precision_auc", "mean_fps", "peak_memory_mb"],
        caption="Tracker comparison on GOT-10k test set.",
        label="tab:got10k",
    )
    exporter.save(table, "results/got10k_comparison.tex")
    print(table)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Metric metadata: (display name, higher_is_better)
# ---------------------------------------------------------------------------
_METRIC_META: Dict[str, Tuple[str, bool]] = {
    "mean_iou":                 (r"mIoU $\uparrow$",                    True),
    "success_auc":              (r"AUC$_\text{S}$ $\uparrow$",          True),
    "precision_auc":            (r"AUC$_\text{P}$ $\uparrow$",          True),
    "normalized_precision_auc": (r"AUC$_\text{NP}$ $\uparrow$",         True),
    "mean_fps":                 (r"FPS $\uparrow$",                     True),
    "mean_center_distance_px":  (r"CD (px) $\downarrow$",              False),
    "peak_memory_mb":           (r"Mem (MiB) $\downarrow$",            False),
    "mean_energy_per_frame_mj": (r"E/frame (mJ) $\downarrow$",         False),
    "total_energy_j":           (r"E$_\text{tot}$ (J) $\downarrow$",    False),
}

DEFAULT_METRICS: List[str] = [
    "success_auc",
    "precision_auc",
    "mean_iou",
    "mean_fps",
    "peak_memory_mb",
]

LATEX_PREAMBLE_HINT = (
    "% Add to your LaTeX preamble:\n"
    "% \\usepackage{booktabs}\n"
    "% \\usepackage{multirow}   % multi-dataset table only\n"
    "% \\usepackage{amsmath}    % \\text inside math mode\n"
)


def _fmt(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _best_and_second(
    values: List[Optional[float]], higher_is_better: bool
) -> Tuple[Optional[int], Optional[int]]:
    """Return indices of the best and second-best non-None values."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None, None
    sign = 1 if higher_is_better else -1
    valid.sort(key=lambda iv: sign * iv[1], reverse=True)
    best = valid[0][0] if len(valid) >= 1 else None
    second = valid[1][0] if len(valid) >= 2 else None
    return best, second


class LatexTableExporter:
    """Export :class:`~eovot.benchmark.engine.BenchmarkResult` objects as
    camera-ready LaTeX booktabs tables.

    Args:
        precision: Decimal places for all numeric cells. Default: ``3``.
        highlight_best: Whether to wrap the best column value in ``\\textbf{}``
            and the second-best in ``\\underline{}``. Default: ``True``.

    Example::

        exporter = LatexTableExporter(precision=3)
        tex = exporter.single_dataset_table(
            results=[r_kcf, r_mosse],
            caption="Classical trackers on OTB-100",
            label="tab:otb100",
        )
        print(tex)
    """

    def __init__(self, precision: int = 3, highlight_best: bool = True) -> None:
        self.precision = precision
        self.highlight_best = highlight_best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def single_dataset_table(
        self,
        results: Sequence,
        metrics: Optional[List[str]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        table_env: str = "table",
        placement: str = "t",
        centering: bool = True,
        font_size: Optional[str] = None,
    ) -> str:
        """Generate a single-dataset comparison table.

        One row per tracker; columns are the requested metrics.

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects for the same dataset.
            metrics: Metric keys to include as columns.  Must be a subset of
                the keys in :data:`_METRIC_META`.  Defaults to
                :data:`DEFAULT_METRICS`.
            caption: Optional ``\\caption{...}`` text.
            label: Optional ``\\label{...}`` text.  A ``tab:`` prefix is
                added automatically when absent.
            table_env: LaTeX environment name. Default: ``"table"``.
            placement: Placement specifier. Default: ``"t"``.
            centering: Emit ``\\centering``. Default: ``True``.
            font_size: Optional LaTeX font-size command, e.g.
                ``"\\small"`` or ``"\\footnotesize"``.

        Returns:
            Complete LaTeX ``\\begin{table}\u2026\\end{table}`` block.
        """
        metrics = self._resolve_metrics(metrics)
        summaries = [r.summary() for r in results]
        tracker_names = [s["tracker"] for s in summaries]

        col_values: Dict[str, List[Optional[float]]] = {
            m: [s.get(m) for s in summaries] for m in metrics
        }
        best, second = self._compute_highlights(col_values, metrics)

        lines: List[str] = [f"\\begin{{{table_env}}}[{placement}]"]
        if centering:
            lines.append("  \\centering")
        if font_size:
            lines.append(f"  {font_size}")
        if caption:
            lines.append(f"  \\caption{{{caption}}}")
        if label:
            lbl = label if label.startswith("tab:") else f"tab:{label}"
            lines.append(f"  \\label{{{lbl}}}")

        col_spec = "l" + "c" * len(metrics)
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append("    \\toprule")

        header_cells = ["Tracker"] + [_METRIC_META[m][0] for m in metrics]
        lines.append("    " + " & ".join(header_cells) + r" \\")
        lines.append("    \\midrule")

        for i, name in enumerate(tracker_names):
            cells = [self._escape(name)]
            for m in metrics:
                v = col_values[m][i]
                cells.append(self._format_cell(v, m, i, best, second))
            lines.append("    " + " & ".join(cells) + r" \\")

        lines.append("    \\bottomrule")
        lines.append(f"  \\end{{tabular}}")
        lines.append(f"\\end{{{table_env}}}")
        return "\n".join(lines)

    def multi_dataset_table(
        self,
        results_by_dataset: Dict[str, Sequence],
        metrics: Optional[List[str]] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        font_size: Optional[str] = None,
    ) -> str:
        """Generate a multi-dataset grouped comparison table.

        Rows are grouped by dataset and separated by ``\\midrule``.
        Best and second-best are highlighted independently within each
        dataset group.

        Args:
            results_by_dataset: ``{dataset_name: [BenchmarkResult, ...]}``
                mapping.  Iteration order determines dataset row order.
            metrics: Metric keys to include.  Defaults to
                :data:`DEFAULT_METRICS`.
            caption: Optional ``\\caption{...}`` text.
            label: Optional ``\\label{...}`` text.
            font_size: Optional font-size command, e.g. ``"\\small"``.

        Returns:
            Complete LaTeX ``\\begin{table}\u2026\\end{table}`` block.
        """
        metrics = self._resolve_metrics(metrics)

        tracker_order: List[str] = []
        seen: Set[str] = set()
        for rs in results_by_dataset.values():
            for r in rs:
                name = r.summary()["tracker"]
                if name not in seen:
                    tracker_order.append(name)
                    seen.add(name)

        col_spec = "ll" + "c" * len(metrics)
        lines: List[str] = ["\\begin{table}[t]"]
        lines.append("  \\centering")
        if font_size:
            lines.append(f"  {font_size}")
        if caption:
            lines.append(f"  \\caption{{{caption}}}")
        if label:
            lbl = label if label.startswith("tab:") else f"tab:{label}"
            lines.append(f"  \\label{{{lbl}}}")

        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append("    \\toprule")
        header_cells = ["Dataset", "Tracker"] + [_METRIC_META[m][0] for m in metrics]
        lines.append("    " + " & ".join(header_cells) + r" \\")
        lines.append("    \\midrule")

        for ds_idx, (dataset_name, ds_results) in enumerate(results_by_dataset.items()):
            if ds_idx > 0:
                lines.append("    \\midrule")

            summaries_map = {r.summary()["tracker"]: r.summary() for r in ds_results}
            col_values: Dict[str, List[Optional[float]]] = {
                m: [summaries_map.get(t, {}).get(m) for t in tracker_order]
                for m in metrics
            }
            best, second = self._compute_highlights(col_values, metrics)

            n_rows = len(tracker_order)
            for row_i, t_name in enumerate(tracker_order):
                ds_cell = (
                    f"\\multirow{{{n_rows}}}{{*}}{{{self._escape(dataset_name)}}}"
                    if row_i == 0 else ""
                )
                cells = [ds_cell, self._escape(t_name)]
                for m in metrics:
                    v = col_values[m][row_i]
                    cells.append(self._format_cell(v, m, row_i, best, second))
                lines.append("    " + " & ".join(cells) + r" \\")

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def save(self, latex: str, path: str, include_preamble_hint: bool = True) -> Path:
        """Write a LaTeX string to a ``.tex`` file.

        Args:
            latex: Output of :meth:`single_dataset_table` or
                :meth:`multi_dataset_table`.
            path:  Destination file path.  A ``.tex`` extension is appended
                when the path has none.
            include_preamble_hint: Prepend a comment with required
                ``\\usepackage`` declarations.  Default: ``True``.

        Returns:
            Resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".tex")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            if include_preamble_hint:
                fh.write(LATEX_PREAMBLE_HINT + "\n")
            fh.write(latex)
            fh.write("\n")
        return p

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_metrics(metrics: Optional[List[str]]) -> List[str]:
        if metrics is None:
            return list(DEFAULT_METRICS)
        return [m for m in metrics if m in _METRIC_META]

    def _compute_highlights(
        self,
        col_values: Dict[str, List[Optional[float]]],
        metrics: List[str],
    ) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]:
        best: Dict[str, Optional[int]] = {}
        second: Dict[str, Optional[int]] = {}
        for m in metrics:
            if not self.highlight_best:
                best[m] = second[m] = None
                continue
            _, higher = _METRIC_META[m]
            b, s = _best_and_second(col_values[m], higher)
            best[m] = b
            second[m] = s
        return best, second

    def _format_cell(
        self,
        value: Optional[float],
        metric: str,
        row_idx: int,
        best: Dict[str, Optional[int]],
        second: Dict[str, Optional[int]],
    ) -> str:
        if value is None:
            return "--"
        cell = _fmt(value, self.precision)
        if row_idx == best.get(metric):
            return f"\\textbf{{{cell}}}"
        if row_idx == second.get(metric):
            return f"\\underline{{{cell}}}"
        return cell

    @staticmethod
    def _escape(text: str) -> str:
        """Escape LaTeX special characters in tracker/dataset names."""
        for ch in ("_", "%", "&", "#", "{", "}"):
            text = text.replace(ch, f"\\{ch}")
        return text
