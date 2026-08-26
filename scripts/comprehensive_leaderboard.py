"""Generate a comprehensive multi-metric leaderboard from saved benchmark results.

Loads one or more JSON result files produced by ``eovot`` / ``run_benchmark.py``
and ranks them using :class:`~eovot.reporting.comprehensive.ComprehensiveLeaderboard`,
which combines accuracy (Success AUC), robustness (EAO), and efficiency (EES)
into a single composite score.

Usage
-----
    # Rank two pre-computed result files:
    python scripts/comprehensive_leaderboard.py \\
        results/MOSSE-Synthetic.json \\
        results/KCF-Synthetic.json \\
        --output results/leaderboard_comprehensive.md

    # Adjust dimension weights and memory budget:
    python scripts/comprehensive_leaderboard.py results/*.json \\
        --accuracy-weight 0.4 \\
        --robustness-weight 0.4 \\
        --efficiency-weight 0.2 \\
        --memory-budget 256 \\
        --output results/leaderboard_edge.md

    # Print only; no file written:
    python scripts/comprehensive_leaderboard.py results/*.json --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkResult
from eovot.reporting.comprehensive import ComprehensiveLeaderboard


def _load_result(path: Path) -> BenchmarkResult:
    """Load a :class:`BenchmarkResult` from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return BenchmarkResult.from_dict(d)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comprehensive_leaderboard",
        description=(
            "Build a comprehensive EOVOT leaderboard that ranks trackers by "
            "a weighted combination of accuracy, robustness, and efficiency."
        ),
    )
    parser.add_argument(
        "results",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more benchmark result JSON files to compare.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help=(
            "Write the Markdown leaderboard to this file.  "
            "Also writes a companion .json file.  "
            "Default: <first_result_dir>/leaderboard_comprehensive.md"
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout only; do not write output files.",
    )
    parser.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.5,
        metavar="W",
        help="Weight for Success AUC in the composite score (default: 0.5).",
    )
    parser.add_argument(
        "--robustness-weight",
        type=float,
        default=0.3,
        metavar="W",
        help="Weight for EAO in the composite score (default: 0.3).",
    )
    parser.add_argument(
        "--efficiency-weight",
        type=float,
        default=0.2,
        metavar="W",
        help="Weight for normalised EES in the composite score (default: 0.2).",
    )
    parser.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help=(
            "Peak-memory budget in MB for EES computation.  "
            "Trackers exceeding this are soft-penalised.  "
            "Default: 512.0."
        ),
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.1,
        metavar="IOU",
        help="IoU below which a frame is counted as a failure (default: 0.1).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Validate weights ---
    total = args.accuracy_weight + args.robustness_weight + args.efficiency_weight
    if abs(total - 1.0) > 1e-4:
        parser.error(
            f"Weights must sum to 1.0: "
            f"accuracy={args.accuracy_weight} + robustness={args.robustness_weight} + "
            f"efficiency={args.efficiency_weight} = {total:.4f}"
        )

    # --- Load results ---
    results = []
    for path_str in args.results:
        path = Path(path_str)
        if not path.exists():
            print(f"[WARNING] File not found, skipping: {path}", file=sys.stderr)
            continue
        try:
            r = _load_result(path)
            results.append(r)
            print(f"  Loaded: {path.name}  →  {r.tracker_name} on {r.dataset_name}")
        except Exception as exc:
            print(f"[WARNING] Failed to load {path}: {exc}", file=sys.stderr)

    if not results:
        print("[ERROR] No results loaded.  Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Build leaderboard ---
    lb = ComprehensiveLeaderboard(
        memory_budget_mb=args.memory_budget,
        accuracy_weight=args.accuracy_weight,
        robustness_weight=args.robustness_weight,
        efficiency_weight=args.efficiency_weight,
        failure_threshold=args.failure_threshold,
    )
    entries = lb.rank(results)
    md = lb.to_markdown(entries)

    print("\n" + "=" * 70)
    print(md)
    print("=" * 70)

    if not args.no_save:
        out_path = Path(args.output) if args.output else (
            Path(args.results[0]).parent / "leaderboard_comprehensive.md"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\nMarkdown leaderboard saved to: {out_path}")

        json_path = out_path.with_suffix(".json")
        lb.save_json(entries, json_path)
        print(f"JSON leaderboard saved to:     {json_path}")


if __name__ == "__main__":
    main()
