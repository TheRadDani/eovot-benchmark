"""CLI for per-attribute tracker performance analysis.

Loads one or more saved benchmark result JSON files, computes per-attribute
performance statistics using the sequence attribute metadata stored in the
provided dataset, and outputs a Markdown report.

Usage examples
--------------
Single tracker, using attribute map from a SyntheticDataset::

    python scripts/attribute_analysis.py \\
        --results results/KCF-Synthetic.json \\
        --dataset synthetic --motion linear \\
        --tracker-names KCF

Multi-tracker comparison from pre-saved JSON results::

    python scripts/attribute_analysis.py \\
        --results results/MOSSE-Synthetic.json results/KCF-Synthetic.json \\
        --dataset synthetic --motion linear \\
        --tracker-names MOSSE KCF \\
        --output results/attr_comparison.md

For real datasets (OTB, LaSOT, GOT-10k), provide the dataset root and
populate sequence attributes externally by extending the attribute_map
returned by the dataset loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.attributes import AttributeAnalyzer, SequenceAttribute
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ------------------------------------------------------------------ #
# Helpers to reconstruct BenchmarkResult from a saved JSON dict       #
# ------------------------------------------------------------------ #

def _profiling_stub(fps: float) -> ProfilingResult:
    """Build a minimal ProfilingResult from the serialized fps value."""
    mean_ms = 1000.0 / fps if fps > 0 else 1.0
    return ProfilingResult(
        tracker_name="",
        frame_count=1,
        fps=fps,
        latency_mean_ms=mean_ms,
        latency_std_ms=0.0,
        latency_p95_ms=mean_ms,
        peak_memory_mb=0.0,
    )


def result_from_json(path: str, tracker_name: str) -> BenchmarkResult:
    """Reconstruct a :class:`BenchmarkResult` from a saved JSON file.

    Only the fields needed by :class:`~eovot.analysis.attributes.AttributeAnalyzer`
    are populated: ``mean_iou``, ``profiling.fps``, and ``accuracy_metrics``.

    Args:
        path: Path to the JSON result file.
        tracker_name: Tracker name to stamp on the result.

    Returns:
        A :class:`BenchmarkResult` with per-sequence accuracy data.
    """
    from eovot.metrics.accuracy import AccuracyMetrics
    import numpy as np

    with open(path) as fh:
        data = json.load(fh)

    summary = data.get("summary", {})
    dataset_name = summary.get("dataset", summary.get("dataset_name", "unknown"))
    br = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)

    for seq in data.get("sequences", []):
        miou = float(seq.get("mean_iou", 0.0))
        fps = float(seq.get("fps", 1.0))
        sauc = float(seq.get("success_auc", miou))
        pauc = float(seq.get("precision_auc", 0.0))
        acc = AccuracyMetrics(
            mean_iou=miou,
            success_auc=sauc,
            precision_auc=pauc,
        )
        prof = _profiling_stub(fps)
        prof.tracker_name = tracker_name
        sr = SequenceResult(
            sequence_name=seq.get("sequence_name", ""),
            ious=np.array([miou]),
            profiling=prof,
            accuracy_metrics=acc,
        )
        br.sequence_results.append(sr)

    return br


# ------------------------------------------------------------------ #
# Dataset attribute map builders                                       #
# ------------------------------------------------------------------ #

def _build_synthetic_attr_map(motion: str, num_sequences: int, seed: int) -> dict:
    """Build an attribute map by instantiating a SyntheticDataset."""
    from eovot.datasets.synthetic import SyntheticDataset

    ds = SyntheticDataset(
        num_sequences=num_sequences,
        num_frames=2,        # minimal frames — we only need the attribute tags
        motion=motion,
        seed=seed,
    )
    return {seq.name: seq.attributes for seq in ds if seq.attributes}


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="attribute_analysis",
        description="Per-attribute tracker performance analysis for EOVOT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="JSON",
        help="Path(s) to saved benchmark result JSON files (one per tracker).",
    )
    parser.add_argument(
        "--tracker-names", nargs="+", metavar="NAME",
        help="Tracker names corresponding to each --results file (same order). "
             "If omitted, names are read from the JSON summaries.",
    )
    parser.add_argument(
        "--dataset", choices=["synthetic"], default="synthetic",
        help="Dataset type for building the attribute map (default: synthetic).",
    )
    parser.add_argument(
        "--motion", choices=["linear", "circular", "random"], default="linear",
        help="SyntheticDataset motion pattern (used when --dataset=synthetic).",
    )
    parser.add_argument(
        "--num-sequences", type=int, default=10,
        help="Number of sequences in the SyntheticDataset (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="SyntheticDataset RNG seed (default: 42).",
    )
    parser.add_argument(
        "--min-sequences", type=int, default=2,
        help="Min sequences per attribute to include in the report (default: 2).",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Write the Markdown report to this file (default: print to stdout).",
    )
    args = parser.parse_args()

    # --- Load results ---
    result_paths = args.results
    tracker_names = args.tracker_names

    if tracker_names and len(tracker_names) != len(result_paths):
        parser.error("--tracker-names must have the same count as --results")

    results = []
    for i, path in enumerate(result_paths):
        name = tracker_names[i] if tracker_names else None
        if name is None:
            with open(path) as fh:
                d = json.load(fh)
            s = d.get("summary", {})
            name = s.get("tracker") or s.get("tracker_name") or Path(path).stem
        results.append(result_from_json(path, name))

    # --- Build attribute map ---
    if args.dataset == "synthetic":
        attr_map = _build_synthetic_attr_map(args.motion, args.num_sequences, args.seed)
    else:
        attr_map = {}

    if not attr_map:
        print("[WARNING] Attribute map is empty — no sequences will be tagged.", file=sys.stderr)

    analyzer = AttributeAnalyzer(min_sequences=args.min_sequences)

    # --- Generate report ---
    report_lines = ["# EOVOT Attribute Analysis Report\n"]

    if len(results) == 1:
        analysis = analyzer.analyze(results[0], attr_map)
        report_lines.append(analyzer.to_markdown_table(analysis, tracker_name=results[0].tracker_name))
        report_lines.append("")
        report_lines.append(analyzer.degradation_report(results[0], attr_map))
    else:
        comparison = analyzer.compare_trackers(results, attr_map)
        report_lines.append(
            analyzer.to_multi_tracker_table(
                comparison,
                tracker_names=[r.tracker_name for r in results],
            )
        )
        report_lines.append("")
        for result in results:
            report_lines.append(analyzer.degradation_report(result, attr_map))
            report_lines.append("")

    report = "\n".join(report_lines)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"[attribute_analysis] Report saved → {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
