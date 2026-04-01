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

        Columns: ``sequence_name``, ``mean_iou``, ``precision_score``,
        ``fps``, ``mean_latency_ms``.

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            name: Base filename without extension.

        Returns:
            :class:`pathlib.Path` of the written file.
        """
        path = self.output_dir / f"{name}.csv"
        sequences = result.get("sequences", [])
        if not sequences:
            return path

        fieldnames = ["sequence_name", "mean_iou", "precision_score", "fps", "mean_latency_ms"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for seq in sequences:
                writer.writerow({
                    "sequence_name": seq.get("sequence_name", ""),
                    "mean_iou": f"{seq.get('mean_iou', 0.0):.4f}",
                    "precision_score": f"{seq.get('precision_score', 0.0):.4f}",
                    "fps": f"{seq.get('fps', 0.0):.2f}",
                    "mean_latency_ms": f"{seq.get('mean_latency_ms', 0.0):.3f}",
                })
        return path

    def save_all(self, result: Dict[str, Any], name: str) -> Dict[str, Path]:
        """Save JSON and CSV and return a mapping of format → path.

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
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

        Args:
            result: Output dict from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.

        Returns:
            A ``| col | col | ... |`` formatted string (no trailing newline).
        """
        s = result.get("summary", {})
        tracker = s.get("tracker", "?")
        dataset = s.get("dataset", "?")
        mean_iou = s.get("mean_iou", 0.0)
        mean_prec = s.get("mean_precision", 0.0)
        fps = s.get("mean_fps", 0.0)
        lat = s.get("mean_latency_ms", 0.0)
        mem = s.get("peak_memory_mb", 0.0)
        return (
            f"| {tracker} | {dataset} | {mean_iou:.4f} "
            f"| {mean_prec:.4f} | {fps:.1f} | {lat:.2f} | {mem:.1f} |"
        )

    @staticmethod
    def comparison_table(results: List[Dict[str, Any]]) -> str:
        """Build a Markdown comparison table from multiple benchmark results.

        Args:
            results: List of outputs from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`,
                one entry per tracker / dataset combination.

        Returns:
            A multi-line Markdown string ready to paste into a README or paper.
        """
        header = (
            "| Tracker | Dataset | mIoU | Precision | FPS | Latency (ms) | Mem (MB) |\n"
            "|---------|---------|-----:|----------:|----:|-------------:|---------:|\n"
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


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard types (e.g. numpy scalars)."""
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):        # numpy array
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
