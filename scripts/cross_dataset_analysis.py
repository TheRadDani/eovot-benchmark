"""Cross-dataset generalization analysis for EOVOT.

Loads per-tracker BenchmarkResult JSON files produced by the benchmark engine,
groups them by dataset, and runs CrossDatasetAnalyzer to produce a
generalization matrix and ranking.

Usage
-----
    # Analyse saved result files:
    python scripts/cross_dataset_analysis.py \\
        results/MOSSE-OTB100.json \\
        results/MOSSE-GOT10k.json \\
        results/KCF-OTB100.json \\
        results/KCF-GOT10k.json \\
        --output-dir results/cross_dataset

    # Demo mode using synthetic datasets (no data download required):
    python scripts/cross_dataset_analysis.py --demo \\
        --trackers MOSSE KCF CSRT LKOpticalFlow \\
        --output-dir results/cross_dataset_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.cross_dataset import CrossDatasetAnalyzer, GeneralizationReport
from eovot.benchmark.engine import BenchmarkResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(paths: List[str]) -> Dict[str, List[BenchmarkResult]]:
    """Load JSON result files and group by dataset name."""
    by_dataset: Dict[str, List[BenchmarkResult]] = {}
    for path in paths:
        result = BenchmarkResult.load(path)
        by_dataset.setdefault(result.dataset_name, []).append(result)
    return by_dataset


def _run_demo(trackers: List[str], sequences: int, frames: int) -> Dict[str, List[BenchmarkResult]]:
    """Run a quick demo on two synthetic datasets with different motion patterns."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker, available_trackers

    print("Running demo on two synthetic datasets (linear + random motion) …\n")
    by_dataset: Dict[str, List[BenchmarkResult]] = {}
    engine = BenchmarkEngine(verbose=False)

    for motion in ("linear", "random"):
        ds_name = f"Synthetic-{motion.capitalize()}"
        dataset = SyntheticDataset(
            num_sequences=sequences,
            num_frames=frames,
            motion=motion,  # type: ignore[arg-type]
            seed=42,
        )
        results_for_ds: List[BenchmarkResult] = []
        for t_name in trackers:
            if t_name not in available_trackers():
                print(f"  Warning: tracker '{t_name}' not registered — skipping.")
                continue
            tracker = build_tracker(t_name)
            result = engine.run(tracker, dataset, dataset_name=ds_name)
            results_for_ds.append(result)
        by_dataset[ds_name] = results_for_ds

    return by_dataset


def _write_report(report: GeneralizationReport, output_dir: Path) -> None:
    """Write all report formats to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "generalization_matrix.md"
    md_path.write_text(
        report.to_markdown_table() + "\n\n" + report.to_generalization_ranking(),
        encoding="utf-8",
    )
    print(f"\nMarkdown report  → {md_path}")

    csv_path = output_dir / "generalization_matrix.csv"
    csv_path.write_text(report.to_csv(), encoding="utf-8")
    print(f"CSV report       → {csv_path}")

    latex_path = output_dir / "generalization_matrix.tex"
    latex_path.write_text(
        report.to_latex_table(caption="Cross-dataset generalization matrix (mean IoU)."),
        encoding="utf-8",
    )
    print(f"LaTeX table      → {latex_path}")

    json_path = output_dir / "generalization_scores.json"
    scores = {
        "datasets": report.datasets,
        "trackers": [
            {
                "tracker": e.tracker_name,
                "generalization_score": round(e.generalization_score, 6),
                "mean_normalized_iou": round(e.mean_normalized_iou, 6),
                "consistency_penalty": round(e.consistency_penalty, 6),
                "per_dataset_mean_iou": {
                    ds: round(e.metrics[ds].mean_iou, 6)
                    for ds in e.datasets
                    if ds in e.metrics
                },
            }
            for e in report.entries
        ],
    }
    json_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"JSON scores      → {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset generalization analysis for EOVOT trackers.",
    )
    parser.add_argument(
        "result_files",
        nargs="*",
        help="Paths to BenchmarkResult JSON files produced by the benchmark engine.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo on two synthetic datasets (no data download required).",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        default=["MOSSE", "KCF", "CSRT", "LKOpticalFlow"],
        help="Tracker names for --demo mode. Default: MOSSE KCF CSRT LKOpticalFlow.",
    )
    parser.add_argument(
        "--demo-sequences",
        type=int,
        default=5,
        help="Sequences per synthetic dataset in --demo mode. Default: 5.",
    )
    parser.add_argument(
        "--demo-frames",
        type=int,
        default=60,
        help="Frames per sequence in --demo mode. Default: 60.",
    )
    parser.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        help="Memory budget in MB for EES computation. Default: 512.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cross_dataset",
        help="Directory to write report files. Default: results/cross_dataset.",
    )
    parser.add_argument(
        "--metric",
        choices=["mean_iou", "success_auc", "fps", "ees"],
        default="mean_iou",
        help="Metric shown in the Markdown matrix table. Default: mean_iou.",
    )
    args = parser.parse_args()

    if args.demo:
        by_dataset = _run_demo(args.trackers, args.demo_sequences, args.demo_frames)
    elif args.result_files:
        print(f"Loading {len(args.result_files)} result file(s) …")
        by_dataset = _load_results(args.result_files)
    else:
        parser.error("Provide result JSON files or use --demo for a synthetic run.")

    analyzer = CrossDatasetAnalyzer(memory_budget_mb=args.memory_budget)
    report = analyzer.analyze(by_dataset)

    print("\n" + report.to_markdown_table(metric=args.metric))
    print("\n" + report.to_generalization_ranking())

    _write_report(report, Path(args.output_dir))


if __name__ == "__main__":
    main()
