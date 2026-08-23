"""Results reporting utilities for EOVOT benchmarks.

Provides :class:`BenchmarkReporter` for exporting benchmark results in
multiple formats (JSON, CSV, Markdown) and generating multi-tracker
comparison tables suitable for inclusion in research papers or README files.

Typical usage::

    from eovot.reporting.reporter import BenchmarkReporter
    from eovot.benchmark.engine import BenchmarkEngine

    engine = BenchmarkEngine()
    result = engine.run(tracker, dataset)

    reporter = BenchmarkReporter(output_dir="results/")
    reporter.save_all(result, name="MOSSE-OTB100")
    reporter.print_summary(result)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class BenchmarkReporter:
    """Export and format benchmark results from :class:`~eovot.benchmark.engine.BenchmarkEngine`.

    Supports:

    * **JSON** — full result dict including per-sequence breakdowns.
    * **CSV** — per-sequence metrics table, importable into pandas/Excel.
    * **Markdown** — publication-ready comparison table.
    * **Console** — formatted summary block printed to stdout.

    Args:
        output_dir: Directory where all output files are written.
            Created automatically if it does not exist. Default: ``"results/"``.
    """

    def __init__(self, output_dir: str = "results/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save individual formats
    # ------------------------------------------------------------------

    def save_json(self, result: Dict[str, Any], name: str) -> Path:
        """Serialise the full result dict to a JSON file.

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            name: Base filename without extension.

        Returns:
            :class:`pathlib.Path` of the written file.
        """
        path = self.output_dir / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2, default=_json_default)
        return path

    def save_csv(self, result: Dict[str, Any], name: str) -> Path:
        """Write per-sequence metrics to a CSV file.

        All fields present in the result are exported: accuracy (mIoU,
        success AUC, precision AUC, normalized precision AUC), latency
        statistics (mean, std, p95, p99, CV), memory, and energy fields when
        available.  Columns are ordered consistently; optional fields are
        included only when at least one sequence contains them.

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name: Base filename without extension.

        Returns:
            :class:`pathlib.Path` of the written file.
        """
        path = self.output_dir / f"{name}.csv"
        sequences = result.get("sequences", [])
        if not sequences:
            path.touch()
            return path

        # Core fields always present.
        core_fields = [
            "sequence_name",
            "mean_iou",
            "fps",
            "mean_latency_ms",
            "latency_std_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "latency_cv",
            "peak_memory_mb",
        ]
        # Optional fields: include a column only when at least one sequence has it.
        optional_fields = [
            "success_auc",
            "precision_auc",
            "normalized_precision_auc",
            "energy_j",
            "energy_per_frame_mj",
            "energy_tdp_watts",
            "energy_mean_power_w",
        ]
        present_optional = [
            f for f in optional_fields if any(f in seq for seq in sequences)
        ]
        fieldnames = core_fields + present_optional

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for seq in sequences:
                row: Dict[str, Any] = {
                    "sequence_name": seq.get("sequence_name", ""),
                    "mean_iou": f"{seq.get('mean_iou', 0.0):.4f}",
                    "fps": f"{seq.get('fps', 0.0):.2f}",
                    "mean_latency_ms": f"{seq.get('mean_latency_ms', 0.0):.3f}",
                    "latency_std_ms": f"{seq.get('latency_std_ms', 0.0):.3f}",
                    "latency_p95_ms": f"{seq.get('latency_p95_ms', 0.0):.3f}",
                    "latency_p99_ms": f"{seq.get('latency_p99_ms', 0.0):.3f}",
                    "latency_cv": f"{seq.get('latency_cv', 0.0):.6f}",
                    "peak_memory_mb": f"{seq.get('peak_memory_mb', 0.0):.2f}",
                }
                for field in present_optional:
                    if field in seq:
                        # Energy values in Joules can be very small; use 6 d.p.
                        fmt = ".6f" if field in ("energy_j",) else ".4f"
                        row[field] = f"{seq[field]:{fmt}}"
                writer.writerow(row)
        return path

    def save_summary_csv(
        self, results: List[Dict[str, Any]], name: str = "summary"
    ) -> Path:
        """Write a one-row-per-tracker summary CSV for cross-experiment comparison.

        Each row aggregates the scalar summary fields from one tracker run.
        This is the CSV complement of :meth:`comparison_table`, designed for
        import into pandas, Excel, or spreadsheet tools.

        Args:
            results: List of result dicts, one per tracker / dataset combination.
                Each must match the format produced by
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name: Base filename without extension. Default: ``"summary"``.

        Returns:
            :class:`pathlib.Path` of the written ``.csv`` file.

        Example::

            reporter = BenchmarkReporter(output_dir="results/")
            reporter.save_summary_csv([mosse_dict, kcf_dict], name="classical_trackers")
        """
        path = self.output_dir / f"{name}.csv"
        if not results:
            path.touch()
            return path

        summaries = [r.get("summary", {}) for r in results]

        # Determine which optional summary fields are present across all results.
        core_summary_fields = [
            "tracker",
            "dataset",
            "num_sequences",
            "mean_iou",
            "mean_fps",
            "peak_memory_mb",
        ]
        optional_summary_fields = [
            "mean_center_distance_px",
            "success_auc",
            "precision_auc",
            "normalized_precision_auc",
            "total_energy_j",
            "mean_energy_per_frame_mj",
        ]
        present_summary_optional = [
            f for f in optional_summary_fields if any(f in s for s in summaries)
        ]
        fieldnames = core_summary_fields + present_summary_optional

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in summaries:
                row: Dict[str, Any] = {
                    "tracker": s.get("tracker") or s.get("tracker_name", ""),
                    "dataset": s.get("dataset") or s.get("dataset_name", ""),
                    "num_sequences": s.get("num_sequences", ""),
                    "mean_iou": f"{s.get('mean_iou', 0.0):.4f}",
                    "mean_fps": f"{s.get('mean_fps', 0.0):.2f}",
                    "peak_memory_mb": f"{s.get('peak_memory_mb', 0.0):.2f}",
                }
                for field in present_summary_optional:
                    if field in s:
                        row[field] = f"{s[field]:.4f}"
                writer.writerow(row)
        return path

    def save_all(self, result: Dict[str, Any], name: str) -> Dict[str, Path]:
        """Save JSON and CSV and return a mapping of format → path.

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name: Base filename prefix.

        Returns:
            ``{"json": Path(...), "csv": Path(...)}``.
        """
        return {
            "json": self.save_json(result, name),
            "csv": self.save_csv(result, name),
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(result: Dict[str, Any]) -> None:
        """Print a formatted benchmark summary block to stdout.

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
        """
        summary = result.get("summary", {})
        print("\n" + "=" * 60)
        print(" BENCHMARK SUMMARY")
        print("=" * 60)
        for key, val in summary.items():
            formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  {key:<30s}: {formatted}")
        print("=" * 60 + "\n")

    @staticmethod
    def to_markdown_row(result: Dict[str, Any]) -> str:
        """Format a single benchmark result as one Markdown table row.

        Columns: Tracker | Dataset | mIoU | Success AUC | Precision AUC | FPS | Latency (ms) | Mem (MB)

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            A ``| col | col | ... |`` formatted string (no trailing newline).
        """
        s = result.get("summary", {})
        # "tracker" is the canonical key produced by BenchmarkResult.summary();
        # fall back to "tracker_name" for backward compatibility with older result files.
        tracker = s.get("tracker") or s.get("tracker_name", "?")
        dataset = s.get("dataset") or s.get("dataset_name", "?")
        mean_iou = float(s.get("mean_iou", 0.0))
        # Fall back gracefully when success/precision AUC are absent (pre-existing result files).
        success_auc = float(s.get("success_auc", mean_iou))
        precision_auc = float(s.get("precision_auc", 0.0))
        fps = float(s.get("mean_fps", 0.0))
        lat = float(s.get("mean_latency_ms", 0.0))
        mem = float(s.get("peak_memory_mb", 0.0))
        return (
            f"| {tracker} | {dataset} | {mean_iou:.4f} "
            f"| {success_auc:.4f} | {precision_auc:.4f} "
            f"| {fps:.1f} | {lat:.2f} | {mem:.1f} |"
        )

    @staticmethod
    def comparison_table(results: List[Dict[str, Any]]) -> str:
        """Build a Markdown comparison table from multiple benchmark results.

        Includes the standard VOT scalars (mIoU, success AUC, precision AUC)
        alongside hardware metrics (FPS, latency, memory).

        Args:
            results: List of outputs from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`,
                one entry per tracker / dataset combination.

        Returns:
            A multi-line Markdown string ready to paste into a README or paper.
        """
        header = (
            "| Tracker | Dataset | mIoU | Success AUC | Precision AUC | FPS | Latency (ms) | Mem (MB) |\n"
            "|---------|---------|-----:|------------:|--------------:|----:|-------------:|---------:|\n"
        )
        rows = "\n".join(BenchmarkReporter.to_markdown_row(r) for r in results)
        return header + rows

    def save_comparison(self, results: List[Dict[str, Any]], name: str = "comparison") -> Path:
        """Write a Markdown comparison table to disk.

        Args:
            results: List of benchmark results (one per tracker / dataset combo).
            name: Base filename without extension. Default: ``"comparison"``.

        Returns:
            :class:`pathlib.Path` of the written ``.md`` file.
        """
        table = self.comparison_table(results)
        path = self.output_dir / f"{name}.md"
        with open(path, "w") as fh:
            fh.write("# EOVOT Tracker Comparison\n\n")
            fh.write(table)
            fh.write("\n")
        return path

    def save_html(
        self,
        results: List[Dict[str, Any]],
        name: str = "report",
        title: str = "EOVOT Benchmark Report",
    ) -> Path:
        """Write a self-contained HTML report to disk.

        The report contains an interactive leaderboard table and inline SVG
        charts (success curves, precision curves, FPS-IoU scatter, latency
        percentiles).  No external CDN or matplotlib installation is required.

        Args:
            results: List of result dicts, one per tracker / dataset.  Each
                dict must match the format produced by
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name:  Base filename without extension.  Default: ``"report"``.
            title: HTML document title shown in the page heading and browser
                tab.

        Returns:
            :class:`pathlib.Path` of the written ``.html`` file.

        Example::

            reporter = BenchmarkReporter(output_dir="results/")
            reporter.save_html([mosse_dict, kcf_dict], name="comparison",
                               title="Classical Trackers on OTB100")
        """
        from .html_report import HTMLReportGenerator

        gen = HTMLReportGenerator(title=title)
        path = self.output_dir / f"{name}.html"
        gen.save(results, str(path))
        return path


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard types (e.g. numpy scalars)."""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
