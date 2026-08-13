"""LaTeX academic table generator for EOVOT benchmark results.

Converts benchmark result dicts produced by
:meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict` into publication-ready
LaTeX table code, ready to paste directly into a research paper.

Features
--------
* **Configurable columns** — choose any subset of the standard scalars
  (mIoU, success AUC, precision AUC, FPS, memory, EES) or pass custom headers.
* **Automatic bold-facing** — optionally highlight the best (or worst) value per
  column to guide readers; direction (↑/↓) is per-metric.
* **Edge Efficiency Score (EES)** — computed on the fly from mIoU and FPS when
  requested; no need to pre-run :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.
* **Multi-dataset support** — pass results from several datasets; the table
  groups rows by dataset with ``\\midrule`` separators.
* **Caption and label** — fully configurable for BibTeX cross-referencing.
* **Booktabs style** — uses ``\\toprule / \\midrule / \\bottomrule`` (requires
  the ``booktabs`` package in the LaTeX preamble).

Typical usage::

    from eovot.reporting.latex_report import LaTeXReportGenerator

    gen = LaTeXReportGenerator()
    table = gen.generate_table(results, caption="Tracker comparison on OTB-100",
                               label="tab:otb100")
    print(table)

    # Or save to a file
    gen.save(results, "paper/tables/otb100.tex",
             caption="Tracker comparison on OTB-100", label="tab:otb100")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Column specification
# ---------------------------------------------------------------------------

#: Built-in column definitions: (header, result_key, higher_is_better, format_spec)
_BUILTIN_COLUMNS: Dict[str, Tuple[str, str, bool, str]] = {
    "tracker":       (r"Tracker",        "tracker",        True,  "s"),
    "dataset":       (r"Dataset",        "dataset",        True,  "s"),
    "miou":          (r"mIoU",           "mean_iou",       True,  ".4f"),
    "success_auc":   (r"Succ. AUC",      "success_auc",    True,  ".4f"),
    "precision_auc": (r"Prec. AUC",      "precision_auc",  True,  ".4f"),
    "fps":           (r"FPS",            "mean_fps",       True,  ".1f"),
    "memory":        (r"Mem (MB)",       "peak_memory_mb", False, ".1f"),
    "latency":       (r"Lat. (ms)",      "mean_latency_ms",False, ".2f"),
    "energy":        (r"E/fr (mJ)",      "mean_energy_per_frame_mj", False, ".3f"),
    "ees":           (r"EES",            "_ees",           True,  ".4f"),
}

#: Default column set — covers the key VOT accuracy + edge efficiency metrics.
DEFAULT_COLUMNS: List[str] = [
    "tracker", "miou", "success_auc", "precision_auc", "fps", "memory",
]

#: Column alignment strings for LaTeX tabular (one character per column).
_ALIGN: Dict[str, str] = {
    "tracker":       "l",
    "dataset":       "l",
    "miou":          "r",
    "success_auc":   "r",
    "precision_auc": "r",
    "fps":           "r",
    "memory":        "r",
    "latency":       "r",
    "energy":        "r",
    "ees":           "r",
}


# ---------------------------------------------------------------------------
# EES helper
# ---------------------------------------------------------------------------

def _compute_ees(
    mean_iou: float,
    fps: float,
    peak_memory_mb: float,
    memory_budget_mb: float = 512.0,
) -> float:
    """Compute Edge Efficiency Score (mirrors :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`)."""
    if fps <= 0 or mean_iou < 0:
        return 0.0
    return (mean_iou * math.log1p(fps)) / (1.0 + peak_memory_mb / memory_budget_mb)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class LaTeXReportGenerator:
    """Generate publication-ready LaTeX tables from EOVOT benchmark results.

    Args:
        columns: List of column keys (from :data:`_BUILTIN_COLUMNS` or custom).
            Defaults to :data:`DEFAULT_COLUMNS`.
        bold_best: When ``True``, wrap the best value in each numeric column
            with ``\\mathbf{...}`` in math mode.  Default: ``True``.
        group_by_dataset: When ``True`` and results span multiple datasets,
            insert ``\\midrule`` between dataset groups.  Default: ``True``.
        memory_budget_mb: Memory ceiling used for EES computation.
            Default: ``512.0`` MB.
        float_precision: Default decimal places for numeric cells when the
            column's format spec is not overridden.  Default: ``4``.

    Example::

        results = [mosse_dict, kcf_dict, csrt_dict]  # from BenchmarkResult.to_dict()
        gen = LaTeXReportGenerator(columns=["tracker", "miou", "success_auc", "fps", "ees"])
        print(gen.generate_table(results, caption="Classical trackers on OTB-100",
                                 label="tab:classical-otb100"))
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        bold_best: bool = True,
        group_by_dataset: bool = True,
        memory_budget_mb: float = 512.0,
        float_precision: int = 4,
    ) -> None:
        self.columns = columns if columns is not None else list(DEFAULT_COLUMNS)
        self.bold_best = bold_best
        self.group_by_dataset = group_by_dataset
        self.memory_budget_mb = memory_budget_mb
        self.float_precision = float_precision

        # Validate column names
        unknown = [c for c in self.columns if c not in _BUILTIN_COLUMNS]
        if unknown:
            raise ValueError(
                f"Unknown column(s): {unknown}. "
                f"Available: {sorted(_BUILTIN_COLUMNS)}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_table(
        self,
        results: List[Dict[str, Any]],
        caption: str = "EOVOT Tracker Comparison",
        label: str = "tab:eovot-comparison",
        position: str = "htbp",
    ) -> str:
        """Generate a complete LaTeX table environment.

        Args:
            results: List of dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            caption: Table caption string (supports LaTeX commands).
            label: BibTeX label for ``\\ref{label}`` cross-references.
            position: LaTeX float placement specifier.  Default: ``"htbp"``.

        Returns:
            Multi-line string containing the full LaTeX ``table`` environment.
        """
        rows = self._extract_rows(results)
        if not rows:
            return "% No results to display.\n"

        col_specs = [_BUILTIN_COLUMNS[c] for c in self.columns]
        best_indices = self._find_best(rows, col_specs) if self.bold_best else {}

        alignment = "".join(_ALIGN.get(c, "r") for c in self.columns)

        header_cells = " & ".join(
            r"\textbf{" + spec[0] + "}" for spec in col_specs
        )

        body_lines: List[str] = []
        prev_dataset: Optional[str] = None

        for row_idx, row in enumerate(rows):
            dataset = row.get("dataset", "")
            if (
                self.group_by_dataset
                and prev_dataset is not None
                and dataset != prev_dataset
            ):
                body_lines.append(r"    \midrule")
            prev_dataset = dataset

            cells: List[str] = []
            for col_idx, (col_key, col_spec) in enumerate(zip(self.columns, col_specs)):
                header, result_key, higher_is_better, fmt = col_spec
                raw = row.get(result_key)

                if raw is None:
                    cells.append("--")
                    continue

                # Format the value
                if fmt == "s":
                    cell = str(raw)
                else:
                    try:
                        cell = format(float(raw), fmt)
                    except (TypeError, ValueError):
                        cell = str(raw)

                # Bold the best numeric cell in this column
                if (
                    self.bold_best
                    and fmt != "s"
                    and row_idx == best_indices.get(col_key)
                    and raw is not None
                ):
                    cell = r"$\mathbf{" + cell + r"}$"

                cells.append(cell)

            body_lines.append("    " + " & ".join(cells) + r" \\")

        lines = [
            f"\\begin{{table}}[{position}]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            f"  \\begin{{tabular}}{{{alignment}}}",
            "    \\toprule",
            f"    {header_cells} \\\\",
            "    \\midrule",
        ]
        lines.extend(body_lines)
        lines += [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
        return "\n".join(lines) + "\n"

    def generate_preamble_comment(self) -> str:
        """Return a LaTeX comment block listing the required packages.

        Returns:
            A multi-line string of ``% ...`` comments to paste at the top of
            the paper's preamble section as a reminder of LaTeX dependencies.
        """
        return (
            "% EOVOT LaTeX table — required packages:\n"
            "% \\usepackage{booktabs}  % \\toprule / \\midrule / \\bottomrule\n"
            "% \\usepackage{amsmath}   % \\mathbf{} in table cells\n"
        )

    def generate_full(
        self,
        results: List[Dict[str, Any]],
        caption: str = "EOVOT Tracker Comparison",
        label: str = "tab:eovot-comparison",
        position: str = "htbp",
    ) -> str:
        """Generate preamble comment + table in one call.

        Args:
            results: List of benchmark result dicts.
            caption: Table caption.
            label: BibTeX cross-reference label.
            position: LaTeX float placement.

        Returns:
            Full string ready to paste into a ``*.tex`` source file.
        """
        return self.generate_preamble_comment() + "\n" + self.generate_table(
            results, caption=caption, label=label, position=position
        )

    def save(
        self,
        results: List[Dict[str, Any]],
        path: str,
        caption: str = "EOVOT Tracker Comparison",
        label: str = "tab:eovot-comparison",
        position: str = "htbp",
        include_preamble: bool = True,
    ) -> Path:
        """Write the LaTeX table to a ``.tex`` file.

        Args:
            results: List of benchmark result dicts.
            path: Destination file path.  Parent directories are created
                automatically.  A ``.tex`` extension is appended when the path
                has none.
            caption: Table caption.
            label: BibTeX cross-reference label.
            position: LaTeX float placement.
            include_preamble: When ``True`` (default) prepend the package
                comment block to the file.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".tex")
        p.parent.mkdir(parents=True, exist_ok=True)

        content = (
            self.generate_full(results, caption=caption, label=label, position=position)
            if include_preamble
            else self.generate_table(results, caption=caption, label=label, position=position)
        )
        p.write_text(content, encoding="utf-8")
        return p

    def to_markdown_preview(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """Return a Markdown table equivalent of the LaTeX output.

        Useful for quickly previewing in GitHub issues, README files, or
        notebook cells without compiling LaTeX.

        Args:
            results: List of benchmark result dicts.

        Returns:
            Multi-line Markdown table string.
        """
        rows = self._extract_rows(results)
        if not rows:
            return "No results.\n"

        col_specs = [_BUILTIN_COLUMNS[c] for c in self.columns]
        headers = [spec[0] for spec in col_specs]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(
            "---:" if _ALIGN.get(col, "r") == "r" else "---"
            for col in self.columns
        ) + " |"

        body: List[str] = []
        for row in rows:
            cells: List[str] = []
            for col_key, (header, result_key, _, fmt) in zip(self.columns, col_specs):
                raw = row.get(result_key)
                if raw is None:
                    cells.append("--")
                elif fmt == "s":
                    cells.append(str(raw))
                else:
                    try:
                        cells.append(format(float(raw), fmt))
                    except (TypeError, ValueError):
                        cells.append(str(raw))
            body.append("| " + " | ".join(cells) + " |")

        return "\n".join([header_row, sep_row] + body) + "\n"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_rows(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten result dicts into one row dict per tracker/dataset."""
        rows: List[Dict[str, Any]] = []
        for result in results:
            s = result.get("summary", {})
            row: Dict[str, Any] = {
                "tracker":       s.get("tracker") or s.get("tracker_name", "?"),
                "dataset":       s.get("dataset") or s.get("dataset_name", "?"),
                "mean_iou":      s.get("mean_iou"),
                "success_auc":   s.get("success_auc"),
                "precision_auc": s.get("precision_auc"),
                "mean_fps":      s.get("mean_fps"),
                "peak_memory_mb":s.get("peak_memory_mb"),
                "mean_latency_ms": s.get("mean_latency_ms"),
                "mean_energy_per_frame_mj": s.get("mean_energy_per_frame_mj"),
            }
            # Compute EES when requested
            if "ees" in self.columns:
                iou = s.get("mean_iou")
                fps = s.get("mean_fps")
                mem = s.get("peak_memory_mb")
                if iou is not None and fps is not None and mem is not None:
                    row["_ees"] = _compute_ees(
                        float(iou), float(fps), float(mem), self.memory_budget_mb
                    )
                else:
                    row["_ees"] = None
            rows.append(row)
        return rows

    def _find_best(
        self,
        rows: List[Dict[str, Any]],
        col_specs: List[Tuple[str, str, bool, str]],
    ) -> Dict[str, int]:
        """Return {column_key: row_index_of_best} for each numeric column."""
        best: Dict[str, int] = {}
        for col_key, (header, result_key, higher_is_better, fmt) in zip(self.columns, col_specs):
            if fmt == "s":
                continue  # skip text columns
            values = [row.get(result_key) for row in rows]
            numeric = [
                (i, float(v)) for i, v in enumerate(values)
                if v is not None
            ]
            if not numeric:
                continue
            if higher_is_better:
                best[col_key] = max(numeric, key=lambda x: x[1])[0]
            else:
                best[col_key] = min(numeric, key=lambda x: x[1])[0]
        return best
