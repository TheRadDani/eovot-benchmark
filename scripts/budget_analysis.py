#!/usr/bin/env python3
"""CLI for hardware-budget constraint analysis of EOVOT benchmark results."""

import argparse
import json
import sys
from pathlib import Path


def _load_results(paths):
    """Load BenchmarkResult JSON files and return namespace objects."""
    from types import SimpleNamespace

    results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        summary = data.get("summary", data)
        results.append(
            SimpleNamespace(
                tracker_name=summary.get("tracker_name") or summary.get("tracker", Path(p).stem),
                dataset_name=summary.get("dataset_name") or summary.get("dataset", "unknown"),
                mean_iou=float(summary.get("mean_iou", 0.0)),
                success_auc=float(summary.get("success_auc") or summary.get("mean_success_auc") or 0.0),
                fps=summary.get("mean_fps") or summary.get("fps"),
                memory_mb=summary.get("peak_memory_mb") or summary.get("memory_mb"),
                energy_mj_per_frame=summary.get("mean_energy_per_frame_mj") or summary.get("energy_mj_per_frame"),
                latency_ms=summary.get("mean_latency_ms") or summary.get("latency_ms"),
                latency_p95_ms=summary.get("latency_p95_ms"),
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyse which trackers satisfy a hardware deployment budget."
    )
    parser.add_argument(
        "results", nargs="+", metavar="RESULT_JSON",
        help="BenchmarkResult JSON file(s) to analyse.",
    )
    parser.add_argument("--min-fps", type=float, default=None)
    parser.add_argument("--max-latency", type=float, default=None, metavar="MS")
    parser.add_argument("--max-latency-p95", type=float, default=None, metavar="MS")
    parser.add_argument("--max-memory", type=float, default=None, metavar="MB")
    parser.add_argument("--max-energy", type=float, default=None, metavar="MJ")
    parser.add_argument("--label", default="target device")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--fps-curve", action="store_true")
    args = parser.parse_args()

    try:
        from eovot.analysis.efficiency_budget import BudgetAnalyzer, HardwareBudget
    except ImportError:
        print("error: eovot not found — run: pip install -e .", file=sys.stderr)
        sys.exit(1)

    results = _load_results(args.results)
    if not results:
        print("No results loaded.", file=sys.stderr)
        sys.exit(1)

    budget = HardwareBudget(
        min_fps=args.min_fps,
        max_latency_ms=args.max_latency,
        max_latency_p95_ms=args.max_latency_p95,
        max_memory_mb=args.max_memory,
        max_energy_mj_per_frame=args.max_energy,
        label=args.label,
    )
    analyzer = BudgetAnalyzer()

    if args.fps_curve:
        fps_values = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]
        curve = analyzer.accuracy_cost_curve(results, fps_values)
        print(f"{'min_fps':>8}  {'accuracy_loss':>14}  best_tracker")
        print("-" * 50)
        for fps, loss, best in curve:
            loss_str = f"{loss:.4f}" if loss is not None else "   N/A"
            print(f"{fps:8.1f}  {loss_str:>14}  {best or chr(8212)}")
        return

    report = analyzer.analyze(results, budget)
    print(f"Budget: {budget.label}")
    print(f"  Feasible trackers : {len(report.feasible)} / {len(report.entries)}")
    if report.best_tracker:
        print(f"  Best tracker      : {report.best_tracker}")
    if report.accuracy_loss_at_budget:
        print(f"  Accuracy cost     : {report.accuracy_loss_at_budget:.4f} IoU")
    print()

    if args.markdown:
        print(report.to_markdown_table())
    else:
        print(f"{'Tracker':<20} {'IoU':>6} {'FPS':>6} {'Mem MB':>8} {'OK':>4}  Violations")
        print("-" * 70)
        for e in sorted(report.entries, key=lambda x: x.accuracy_rank):
            ok_str = "✓" if e.satisfies else "✗"
            fps_str = f"{e.fps:.1f}" if e.fps is not None else "—"
            mem_str = f"{e.memory_mb:.1f}" if e.memory_mb is not None else "—"
            viol = "; ".join(e.violations.values()) if e.violations else ""
            print(f"{e.tracker_name:<20} {e.mean_iou:6.4f} {fps_str:>6} {mem_str:>8} {ok_str:>4}  {viol}")


if __name__ == "__main__":
    main()
