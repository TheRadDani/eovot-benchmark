"""Publication-ready leaderboard exporter for EOVOT benchmark results.

Converts a collection of :class:`~eovot.benchmark.engine.BenchmarkResult`
objects (or their ``to_dict()`` representations) into ranked comparison
tables exportable as **Markdown**, **CSV**, and **LaTeX** — the three
formats most needed in research workflows:

* **Markdown** — paste directly into GitHub READMEs or issue comments.
* **CSV** — import into pandas, Excel, or Google Sheets for further analysis.
* **LaTeX** — drop into a paper with ``\\input{leaderboard.tex}``; best
  values in each column are automatically bolded.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.reporting.leaderboard import LeaderboardExporter

    ds = SyntheticDataset(num_sequences=10, num_frames=200)
    engine = BenchmarkEngine(verbose=False)

    results = [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
    ]

    exporter = LeaderboardExporter()
    print(exporter.to_markdown(results))
    exporter.to_csv(results, "leaderboard.csv")
    print(exporter.to_latex(results))

Loading from saved JSON files is also supported::

    import json
    from eovot.reporting.leaderboard import LeaderboardExporter

    result_dicts = []
    for path in ["results/MOSSE.json", "results/KCF.json"]:
        with open(path) as f:
            result_dicts.append(json.load(f))

    exporter = LeaderboardExporter()
    print(exporter.to_markdown(result_dicts))
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Column specification
# ---------------------------------------------------------------------------

#: Default ordered list of metrics to include in leaderboard tables.
#: Each entry: (column_key, display_name, higher_is_better, format_string)
DEFAULT_COLUMNS: List[Tuple[str, str, bool, str]] = [
    ("mean_iou",              "mIoU",        True,  ".4f"),
    ("success_auc",           "Succ. AUC",   True,  ".4f"),
    ("precision_auc",         "Prec. AUC",   True,  ".4f"),
    ("mean_fps",              "FPS",         True,  ".1f"),
    ("peak_memory_mb",        "Mem (MiB)",   False, ".1f"),
    ("total_energy_j",        "Energy (J)",  False, ".3f"),
    ("mean_center_distance_px", "Ctr Dist",  False, ".2f"),
]


class LeaderboardExporter:
    """Rank and export multi-tracker benchmark results in multiple formats.

    Args:
        columns: Custom column specification overriding :data:`DEFAULT_COLUMNS`.
            Each entry must be a 4-tuple
            ``(key, display_name, higher_is_better, format_string)``
            where *key* matches a field in the result's ``"summary"`` dict.
        primary_metric: Key used to rank trackers.  Must be present in
            ``columns``.  Default: ``"success_auc"``.
        rank_ascending: When ``True``, rank from lowest to highest
            (for lower-is-better primary metrics like memory or latency).
            Inferred automatically from the column's ``higher_is_better``
            flag when ``None``.  Default: ``None`` (auto-infer).

    Example::

        exporter = LeaderboardExporter(primary_metric="mean_fps")
        print(exporter.to_markdown(results))
    """

    def __init__(
        self,
        columns: Optional[List[Tuple[str, str, bool, str]]] = None,
        primary_metric: str = "success_auc",
        rank_ascending: Optional[bool] = None,
    ) -> None:
        self.columns = columns if columns is not None else DEFAULT_COLUMNS
        self.primary_metric = primary_metric
        self._col_map: Dict[str, Tuple[str, bool, str]] = {
            key: (name, hib, fmt) for key, name, hib, fmt in self.columns
        }

        if rank_ascending is None:
            # Infer from the primary metric's higher_is_better flag.
            hib = self._col_map.get(primary_metric, ("", True, ""))[1]
            self._rank_ascending = not hib
        else:
            self._rank_ascending = rank_ascending

    # ------------------------------------------------------------------
    # Core table builder
    # ------------------------------------------------------------------

    def build_table(
        self, results: Sequence[Union[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Build a ranked list of rows from benchmark results.

        Each row is a flat dict with keys ``"rank"``, ``"tracker"``,
        ``"dataset"``, and one key per column in :attr:`columns`.
        Missing metrics are represented as ``None``.

        Args:
            results: Sequence of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects *or* their ``to_dict()`` / ``summary()`` dicts.
                Mixed types in one call are supported.

        Returns:
            List of row dicts, ordered by :attr:`primary_metric`.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("LeaderboardExporter.build_table: results list is empty.")

        rows: List[Dict[str, Any]] = []
        for r in results:
            summary = self._extract_summary(r)
            row: Dict[str, Any] = {
                "tracker": summary.get("tracker") or summary.get("tracker_name", "unknown"),
                "dataset": summary.get("dataset") or summary.get("dataset_name", "—"),
            }
            for key, _name, _hib, _fmt in self.columns:
                row[key] = summary.get(key)
            rows.append(row)

        # Sort by primary metric; rows missing the metric go to the bottom.
        def sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
            val = r.get(self.primary_metric)
            if val is None:
                return (1, 0.0)  # missing → last
            return (0, float(val))

        rows.sort(key=sort_key, reverse=not self._rank_ascending)

        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        return rows

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def to_markdown(
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
        title: Optional[str] = None,
        show_dataset: bool = True,
    ) -> str:
        """Render a ranked leaderboard as a GitHub-flavoured Markdown table.

        Columns with no data across any tracker are omitted automatically
        to keep the table compact.

        Args:
            results: Benchmark results (objects or dicts).
            title: Optional heading printed above the table (``## title``).
            show_dataset: Include a *Dataset* column. Useful when comparing
                trackers across a single shared dataset (``True`` default)
                or when the column would be identical in every row.

        Returns:
            Multi-line string ready to paste into a README or PR comment.
        """
        rows = self.build_table(results)
        active_cols = self._active_columns(rows)

        header_parts = ["Rank", "Tracker"]
        if show_dataset:
            header_parts.append("Dataset")
        for key in active_cols:
            header_parts.append(self._col_map[key][0])

        sep_parts = [":---:"] * len(header_parts)
        sep_parts[1] = ":---"  # left-align tracker name

        lines: List[str] = []
        if title:
            lines.append(f"## {title}")
            lines.append("")

        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("| " + " | ".join(sep_parts) + " |")

        best = self._best_values(rows, active_cols)

        for row in rows:
            cells = [str(row["rank"]), row["tracker"]]
            if show_dataset:
                cells.append(row["dataset"])
            for key in active_cols:
                val = row.get(key)
                if val is None:
                    cells.append("—")
                else:
                    _, hib, fmt = self._col_map[key]
                    formatted = format(float(val), fmt)
                    if self._is_best(val, best.get(key), hib):
                        formatted = f"**{formatted}**"
                    cells.append(formatted)
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
        path: Union[str, Path],
        include_dataset: bool = True,
    ) -> Path:
        """Write a ranked leaderboard to a CSV file.

        Args:
            results: Benchmark results (objects or dicts).
            path: Destination file path.  Parent directories are created
                automatically.
            include_dataset: Add a *dataset* column.  Default: ``True``.

        Returns:
            Resolved :class:`pathlib.Path` of the written file.
        """
        rows = self.build_table(results)
        active_cols = self._active_columns(rows)

        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["rank", "tracker"]
        if include_dataset:
            fieldnames.append("dataset")
        fieldnames.extend(active_cols)

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = {k: row.get(k, "") for k in fieldnames}
                # Format numeric values
                for key in active_cols:
                    val = out.get(key)
                    if val is not None and val != "":
                        _, _hib, fmt = self._col_map[key]
                        out[key] = format(float(val), fmt)
                writer.writerow(out)

        return p

    # ------------------------------------------------------------------
    # LaTeX export
    # ------------------------------------------------------------------

    def to_latex(
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
        caption: str = "Tracker performance comparison.",
        label: str = "tab:leaderboard",
        include_dataset: bool = False,
    ) -> str:
        r"""Render a ranked leaderboard as a LaTeX ``booktabs`` table.

        Best values per column are wrapped in ``\textbf{}``.  The table
        uses the ``booktabs`` package (``\toprule``, ``\midrule``,
        ``\bottomrule``); add ``\usepackage{booktabs}`` to your preamble.

        Args:
            results: Benchmark results (objects or dicts).
            caption: Table caption (``\caption{...}``).
            label: LaTeX label for ``\ref`` and ``\autoref``.
            include_dataset: Add a *Dataset* column.  Default: ``False``
                (most papers compare trackers on a single shared dataset).

        Returns:
            String containing the complete LaTeX ``table`` environment.

        Example output::

            \begin{table}[htbp]
              \centering
              \caption{Tracker performance comparison.}
              \label{tab:leaderboard}
              \begin{tabular}{clrrrr}
                \toprule
                Rank & Tracker & Succ. AUC & Prec. AUC & FPS & Mem (MiB) \\
                \midrule
                1 & MOSSE & \textbf{0.7341} & 0.8910 & \textbf{534.2} & 92.1 \\
                2 & KCF   & 0.6912 & \textbf{0.9023} & 210.5 & \textbf{88.4} \\
                \bottomrule
              \end{tabular}
            \end{table}
        """
        rows = self.build_table(results)
        active_cols = self._active_columns(rows)

        n_extra = 1 if include_dataset else 0
        n_cols = 2 + n_extra + len(active_cols)  # rank + tracker + optional dataset + metrics

        # Column alignment: rank = r, tracker = l, metrics = r
        col_spec = "r" + "l" + ("l" if include_dataset else "") + "r" * len(active_cols)

        header_names = ["Rank", "Tracker"]
        if include_dataset:
            header_names.append("Dataset")
        for key in active_cols:
            header_names.append(self._col_map[key][0])

        best = self._best_values(rows, active_cols)

        buf = io.StringIO()
        buf.write("\\begin{table}[htbp]\n")
        buf.write("  \\centering\n")
        buf.write(f"  \\caption{{{caption}}}\n")
        buf.write(f"  \\label{{{label}}}\n")
        buf.write(f"  \\begin{{tabular}}{{{col_spec}}}\n")
        buf.write("    \\toprule\n")
        buf.write("    " + " & ".join(header_names) + " \\\\\n")
        buf.write("    \\midrule\n")

        for row in rows:
            cells = [str(row["rank"]), self._latex_escape(row["tracker"])]
            if include_dataset:
                cells.append(self._latex_escape(row["dataset"]))
            for key in active_cols:
                val = row.get(key)
                if val is None:
                    cells.append("—")
                else:
                    _, hib, fmt = self._col_map[key]
                    formatted = format(float(val), fmt)
                    if self._is_best(val, best.get(key), hib):
                        formatted = f"\\textbf{{{formatted}}}"
                    cells.append(formatted)
            buf.write("    " + " & ".join(cells) + " \\\\\n")

        buf.write("    \\bottomrule\n")
        buf.write("  \\end{tabular}\n")
        buf.write("\\end{table}\n")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summary(result: Any) -> Dict[str, Any]:
        """Coerce a result to its flat summary dict."""
        if isinstance(result, dict):
            # Accept either a full to_dict() output or a bare summary dict.
            return result.get("summary", result)
        # Assume a BenchmarkResult object.
        if hasattr(result, "summary"):
            return result.summary()
        raise TypeError(
            f"LeaderboardExporter: unsupported result type {type(result).__name__}. "
            "Pass a BenchmarkResult object or its to_dict()/summary() dict."
        )

    def _active_columns(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Return column keys that have at least one non-None value across rows."""
        return [
            key for key, _, _, _ in self.columns
            if any(row.get(key) is not None for row in rows)
        ]

    @staticmethod
    def _best_values(
        rows: List[Dict[str, Any]], active_cols: List[str]
    ) -> Dict[str, Optional[float]]:
        """Compute the best (highlight-worthy) value for each active column."""
        best: Dict[str, Optional[float]] = {}
        return best  # populated lazily in _is_best for simplicity

    def _is_best(
        self, val: Any, best_val: Optional[float], higher_is_better: bool
    ) -> bool:
        """Return True if *val* is the best in its column across *rows*."""
        # We resolve best lazily — this method is called per-cell after rows are built.
        # The caller passes ``best_val`` from a pre-computed dict; we re-compute here
        # to avoid a second pass through rows.  The cost is negligible at table scale.
        return False  # highlight logic is driven by _highlight_best in render methods

    def _best_in_column(
        self, rows: List[Dict[str, Any]], key: str, higher_is_better: bool
    ) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return max(vals) if higher_is_better else min(vals)

    def _render_cell(
        self,
        val: Optional[float],
        best_val: Optional[float],
        higher_is_better: bool,
        fmt: str,
        bold_token: str = "**{}**",
    ) -> str:
        if val is None:
            return "—"
        formatted = format(float(val), fmt)
        if best_val is not None and abs(float(val) - best_val) < 1e-9:
            return bold_token.format(formatted)
        return formatted

    # Override to_markdown and to_latex to use the corrected highlight logic.
    def _render_table(
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
        show_dataset: bool,
        bold_open: str,
        bold_close: str,
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Optional[float]]]:
        rows = self.build_table(results)
        active_cols = self._active_columns(rows)
        best: Dict[str, Optional[float]] = {}
        for key, _, hib, _ in self.columns:
            if key in active_cols:
                best[key] = self._best_in_column(rows, key, hib)
        return rows, active_cols, best

    # Re-implement to_markdown properly using _render_table.
    def to_markdown(  # type: ignore[override]
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
        title: Optional[str] = None,
        show_dataset: bool = True,
    ) -> str:
        """Render a ranked leaderboard as a GitHub-flavoured Markdown table."""
        rows, active_cols, best = self._render_table(results, show_dataset, "**", "**")

        header_parts = ["Rank", "Tracker"]
        if show_dataset:
            header_parts.append("Dataset")
        for key in active_cols:
            header_parts.append(self._col_map[key][0])

        sep_parts = [":---:"] * len(header_parts)
        sep_parts[1] = ":---"

        lines: List[str] = []
        if title:
            lines.append(f"## {title}")
            lines.append("")
        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("| " + " | ".join(sep_parts) + " |")

        for row in rows:
            cells = [str(row["rank"]), row["tracker"]]
            if show_dataset:
                cells.append(row["dataset"])
            for key in active_cols:
                _, hib, fmt = self._col_map[key]
                val = row.get(key)
                cells.append(self._render_cell(val, best.get(key), hib, fmt, "**{}**"))
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def to_latex(  # type: ignore[override]
        self,
        results: Sequence[Union[Any, Dict[str, Any]]],
        caption: str = "Tracker performance comparison.",
        label: str = "tab:leaderboard",
        include_dataset: bool = False,
    ) -> str:
        r"""Render a ranked leaderboard as a LaTeX ``booktabs`` table."""
        rows, active_cols, best = self._render_table(results, include_dataset, "", "")

        col_spec = "r" + "l" + ("l" if include_dataset else "") + "r" * len(active_cols)

        header_names = ["Rank", "Tracker"]
        if include_dataset:
            header_names.append("Dataset")
        for key in active_cols:
            header_names.append(self._col_map[key][0])

        buf = io.StringIO()
        buf.write("\\begin{table}[htbp]\n")
        buf.write("  \\centering\n")
        buf.write(f"  \\caption{{{caption}}}\n")
        buf.write(f"  \\label{{{label}}}\n")
        buf.write(f"  \\begin{{tabular}}{{{col_spec}}}\n")
        buf.write("    \\toprule\n")
        buf.write("    " + " & ".join(header_names) + " \\\\\n")
        buf.write("    \\midrule\n")

        for row in rows:
            cells = [str(row["rank"]), self._latex_escape(row["tracker"])]
            if include_dataset:
                cells.append(self._latex_escape(row["dataset"]))
            for key in active_cols:
                _, hib, fmt = self._col_map[key]
                val = row.get(key)
                cells.append(
                    self._render_cell(val, best.get(key), hib, fmt, "\\textbf{{{}}}")
                )
            buf.write("    " + " & ".join(cells) + " \\\\\n")

        buf.write("    \\bottomrule\n")
        buf.write("  \\end{tabular}\n")
        buf.write("\\end{table}\n")
        return buf.getvalue()

    @staticmethod
    def _latex_escape(text: str) -> str:
        """Escape special LaTeX characters in tracker/dataset names."""
        replacements = [
            ("\\", "\\textbackslash{}"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("$", "\\$"),
            ("#", "\\#"),
            ("_", "\\_"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        for char, escaped in replacements:
            text = text.replace(char, escaped)
        return text
