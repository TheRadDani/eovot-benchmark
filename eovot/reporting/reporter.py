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

    # Fixed core CSV columns — always present in engine output.
    _CSV_CORE_FIELDS = [
        "sequence_name",
        "mean_iou",
        "fps",
        "mean_latency_ms",
        "peak_memory_mb",
    ]
    # Optional columns — written only when the first sequence carries them.
    _CSV_OPTIONAL_FIELDS = [
        "success_auc",
        "precision_auc",
        "energy_j",
        "energy_per_frame_mj",
    ]

    def save_csv(self, result: Dict[str, Any], name: str) -> Path:
        """Write per-sequence metrics to a CSV file.

        Columns always present: ``sequence_name``, ``mean_iou``, ``fps``,
        ``mean_latency_ms``, ``peak_memory_mb``.

        Optional columns (included when the engine produced them):
        ``success_auc``, ``precision_auc``, ``energy_j``,
        ``energy_per_frame_mj``.

        The previous ``precision_score`` column has been removed — it was
        never populated by the engine and always wrote ``0.0000``.  The
        correct column name is ``precision_auc``.

        Args:
            result: Output dict from
                :meth:`~eovot.benchmark.engine.BenchmarkEngine.run` or
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name: Base filename without extension.

        Returns:
            :class:`pathlib.Path` of the written file.
        """
        path = self.output_dir / f"{name}.csv"
        sequences = result.get("sequences", [])
        if not sequences:
            return path

        # Discover which optional columns are actually present.
        optional_present = [
            col for col in self._CSV_OPTIONAL_FIELDS
            if any(col in seq for seq in sequences)
        ]
        fieldnames = self._CSV_CORE_FIELDS + optional_present

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for seq in sequences:
                row: Dict[str, Any] = {
                    "sequence_name": seq.get("sequence_name", ""),
                    "mean_iou": f"{seq.get('mean_iou', 0.0):.4f}",
                    "fps": f"{seq.get('fps', 0.0):.2f}",
                    "mean_latency_ms": f"{seq.get('mean_latency_ms', 0.0):.3f}",
                    "peak_memory_mb": f"{seq.get('peak_memory_mb', 0.0):.2f}",
                }
                for col in optional_present:
                    val = seq.get(col, "")
                    row[col] = f"{val:.6f}" if isinstance(val, (int, float)) else val
                writer.writerow(row)
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

    def save_comparison_csv(
        self,
        results: List[Dict[str, Any]],
        name: str = "comparison",
        sort_by: str = "success_auc",
    ) -> Path:
        """Write a summary-level comparison CSV, one row per tracker/dataset.

        This is the canonical output for importing benchmark results into
        spreadsheet tools, pandas, or R for downstream statistical analysis.
        Results are sorted by ``sort_by`` (descending); falls back to
        ``"mean_iou"`` when the key is absent.

        Columns: ``tracker``, ``dataset``, ``num_sequences``, ``mean_iou``,
        ``success_auc``, ``precision_auc``, ``mean_fps``, ``peak_memory_mb``,
        ``mean_latency_ms``, ``total_energy_j``, ``mean_energy_per_frame_mj``.

        Optional energy columns are omitted when none of the results carry them.

        Args:
            results: List of dicts from
                :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            name: Output filename without extension. Default: ``"comparison"``.
            sort_by: Summary key to sort by (descending). Default:
                ``"success_auc"``.

        Returns:
            :class:`pathlib.Path` of the written ``.csv`` file.
        """
        summaries = [r.get("summary", r) for r in results]

        # Determine sort key — fall back to mean_iou when absent.
        def _sort_key(s: Dict) -> float:
            val = s.get(sort_by, s.get("mean_iou", 0.0))
            return float(val) if val is not None else 0.0

        summaries_sorted = sorted(summaries, key=_sort_key, reverse=True)

        # Discover optional columns.
        has_energy = any(
            "total_energy_j" in s or "mean_energy_per_frame_mj" in s
            for s in summaries_sorted
        )
        has_sauc = any("success_auc" in s for s in summaries_sorted)
        has_pauc = any("precision_auc" in s for s in summaries_sorted)
        has_lat = any("mean_latency_ms" in s or "mean_fps" in s for s in summaries_sorted)

        fieldnames = ["tracker", "dataset", "num_sequences", "mean_iou"]
        if has_sauc:
            fieldnames.append("success_auc")
        if has_pauc:
            fieldnames.append("precision_auc")
        fieldnames.append("mean_fps")
        fieldnames.append("peak_memory_mb")
        if has_lat:
            fieldnames.append("mean_latency_ms")
        if has_energy:
            fieldnames += ["total_energy_j", "mean_energy_per_frame_mj"]

        path = self.output_dir / f"{name}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries_sorted:
                tracker = s.get("tracker") or s.get("tracker_name", "?")
                dataset = s.get("dataset") or s.get("dataset_name", "?")
                row: Dict[str, Any] = {
                    "tracker": tracker,
                    "dataset": dataset,
                    "num_sequences": s.get("num_sequences", ""),
                    "mean_iou": f"{float(s.get('mean_iou', 0.0)):.4f}",
                }
                if has_sauc:
                    row["success_auc"] = f"{float(s.get('success_auc', s.get('mean_iou', 0.0))):.4f}"
                if has_pauc:
                    row["precision_auc"] = f"{float(s.get('precision_auc', 0.0)):.4f}"
                row["mean_fps"] = f"{float(s.get('mean_fps', 0.0)):.2f}"
                row["peak_memory_mb"] = f"{float(s.get('peak_memory_mb', 0.0)):.2f}"
                if has_lat:
                    lat = s.get("mean_latency_ms") or (
                        1000.0 / float(s["mean_fps"]) if s.get("mean_fps") else ""
                    )
                    row["mean_latency_ms"] = f"{float(lat):.3f}" if lat != "" else ""
                if has_energy:
                    row["total_energy_j"] = (
                        f"{float(s['total_energy_j']):.6f}" if "total_energy_j" in s else ""
                    )
                    row["mean_energy_per_frame_mj"] = (
                        f"{float(s['mean_energy_per_frame_mj']):.4f}"
                        if "mean_energy_per_frame_mj" in s else ""
                    )
                writer.writerow(row)
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
