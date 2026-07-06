"""Zero-dependency end-to-end EOVOT demo.

Generates a synthetic tracking dataset, benchmarks every registered
classical tracker, computes accuracy + efficiency metrics, prints a
leaderboard, and (optionally) saves an edge-efficiency scatter plot.

No external dataset downloads are required.  This script is the fastest
way to verify that the full EOVOT pipeline is working correctly on a fresh
installation.

Usage
-----
    # Minimal run (headless, no plot):
    python scripts/run_full_demo.py

    # Save an edge scatter plot:
    python scripts/run_full_demo.py --plot results/demo/edge_scatter.png

    # Larger synthetic dataset:
    python scripts/run_full_demo.py --sequences 10 --frames 150

    # Quiet output:
    python scripts/run_full_demo.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.metrics.efficiency import EfficiencyMetricsEngine
from eovot.trackers.registry import TRACKER_REGISTRY


# ---------------------------------------------------------------------------
# Trackers safe to run without external model files
# ---------------------------------------------------------------------------
_CLASSICAL_TRACKERS = ["MOSSE", "KCF", "CSRT", "MIL", "MedianFlow", "CamShift"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_full_demo",
        description="EOVOT zero-dependency end-to-end demo",
    )
    p.add_argument(
        "--sequences", type=int, default=5, metavar="N",
        help="Number of synthetic sequences (default: 5)",
    )
    p.add_argument(
        "--frames", type=int, default=80, metavar="N",
        help="Frames per sequence (default: 80)",
    )
    p.add_argument(
        "--motion", default="linear",
        choices=["linear", "circular", "random"],
        help="Synthetic motion pattern (default: linear)",
    )
    p.add_argument(
        "--memory-budget", type=float, default=512.0, metavar="MB",
        help="Memory budget for EES computation in MB (default: 512.0)",
    )
    p.add_argument(
        "--output-dir", default="results/demo", metavar="DIR",
        help="Directory for JSON result files (default: results/demo)",
    )
    p.add_argument(
        "--plot", metavar="PATH", default=None,
        help="Save edge scatter plot to this path (e.g. results/demo/scatter.png).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sequence benchmark progress.",
    )
    p.add_argument(
        "--trackers", nargs="+", metavar="NAME",
        default=_CLASSICAL_TRACKERS,
        help=(
            "Space-separated list of trackers to benchmark "
            f"(default: {' '.join(_CLASSICAL_TRACKERS)})"
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------

def run_demo(
    sequences: int = 5,
    frames: int = 80,
    motion: str = "linear",
    memory_budget_mb: float = 512.0,
    output_dir: str = "results/demo",
    plot_path: Optional[str] = None,
    quiet: bool = False,
    trackers: Optional[List[str]] = None,
) -> Dict:
    """Run the full EOVOT demo pipeline and return a summary dict.

    Args:
        sequences: Number of synthetic sequences to generate.
        frames: Frames per sequence.
        motion: Synthetic motion pattern (``"linear"``, ``"circular"``,
            or ``"random"``).
        memory_budget_mb: Reference memory budget for EES computation (MB).
        output_dir: Directory for per-tracker JSON result files.
        plot_path: If given, save an edge scatter plot to this path.
            Silently skipped when ``matplotlib`` is not installed.
        quiet: Suppress per-sequence benchmark progress.
        trackers: List of tracker names to benchmark.  Defaults to all
            classical trackers that ship with ``opencv-python``.

    Returns:
        Dict with keys ``"leaderboard"`` (Markdown string),
        ``"efficiency_ranking"`` (list of :class:`~eovot.metrics.efficiency.EfficiencyEntry`),
        and ``"results"`` (list of per-tracker result dicts).
    """
    if trackers is None:
        trackers = _CLASSICAL_TRACKERS

    # Filter to only registered trackers to avoid confusing errors.
    available = set(TRACKER_REGISTRY)
    run_trackers = [t for t in trackers if t in available]
    skipped = [t for t in trackers if t not in available]
    if skipped and not quiet:
        print(f"[demo] Skipping unknown trackers: {', '.join(skipped)}")

    if not run_trackers:
        print("[demo] No valid trackers to benchmark. Exiting.", file=sys.stderr)
        return {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = SyntheticDataset(
        num_sequences=sequences,
        num_frames=frames,
        motion=motion,
        seed=42,
    )
    dataset_name = f"Synthetic-{motion.capitalize()}"

    if not quiet:
        print("\n" + "=" * 64)
        print(f"  EOVOT End-to-End Demo")
        print(f"  Dataset : {dataset_name}  ({sequences} sequences × {frames} frames)")
        print(f"  Trackers: {', '.join(run_trackers)}")
        print("=" * 64)

    engine = BenchmarkEngine(verbose=not quiet)
    benchmark_results: List[BenchmarkResult] = []
    result_dicts: List[Dict] = []

    for tracker_name in run_trackers:
        tracker = TRACKER_REGISTRY[tracker_name]()
        t0 = time.perf_counter()
        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=sequences,
        )
        elapsed = time.perf_counter() - t0
        benchmark_results.append(result)

        result_dict = result.to_dict()
        result_dicts.append(result_dict)

        # Persist per-tracker JSON
        out_path = Path(output_dir) / f"{tracker_name}-{dataset_name}.json"
        with open(out_path, "w") as fh:
            json.dump(result_dict, fh, indent=2)

        if not quiet:
            s = result_dict["summary"]
            print(
                f"  → {tracker_name:<14s} "
                f"mIoU={s['mean_iou']:.4f}  "
                f"FPS={s['mean_fps']:.1f}  "
                f"mem={s['peak_memory_mb']:.1f} MB  "
                f"({elapsed:.1f}s)"
            )

    # -------------------------------------------------------------------
    # Efficiency ranking + Pareto front
    # -------------------------------------------------------------------
    eff_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
    ranking = eff_engine.rank_trackers(benchmark_results)

    # -------------------------------------------------------------------
    # Leaderboard (sorted by EES)
    # -------------------------------------------------------------------
    leaderboard_lines = [
        "\n" + "=" * 64,
        "  EOVOT LEADERBOARD  (ranked by Edge Efficiency Score)",
        "=" * 64,
        f"  {'Rank':<5} {'Tracker':<14} {'mIoU':>7} {'FPS':>8} "
        f"{'Mem(MB)':>9} {'EES':>8} {'Pareto':>7}",
        "  " + "-" * 60,
    ]
    for rank, entry in enumerate(ranking, start=1):
        pareto = "✓" if entry.on_pareto_front else ""
        leaderboard_lines.append(
            f"  {rank:<5} {entry.tracker_name:<14} "
            f"{entry.mean_iou:>7.4f} "
            f"{entry.fps:>8.1f} "
            f"{entry.peak_memory_mb:>9.1f} "
            f"{entry.ees:>8.4f} "
            f"{pareto:>7}"
        )
    leaderboard_lines.append("=" * 64 + "\n")
    leaderboard = "\n".join(leaderboard_lines)

    if not quiet:
        print(leaderboard)
        print(f"  Results saved to: {Path(output_dir).resolve()}")

    # -------------------------------------------------------------------
    # Edge efficiency Markdown table
    # -------------------------------------------------------------------
    md_table = eff_engine.to_markdown_table(ranking)
    md_path = Path(output_dir) / "efficiency_ranking.md"
    with open(md_path, "w") as fh:
        fh.write(f"# EOVOT Edge Efficiency Ranking — {dataset_name}\n\n")
        fh.write(md_table)
        fh.write("\n")

    # -------------------------------------------------------------------
    # Edge scatter plot (optional — requires matplotlib)
    # -------------------------------------------------------------------
    if plot_path is not None:
        try:
            from eovot.visualization.plots import plot_edge_scatter  # noqa: PLC0415
            plot_edge_scatter(
                result_dicts,
                output_path=plot_path,
                title=f"EOVOT Edge Deployment Trade-off — {dataset_name}",
                memory_budget_mb=memory_budget_mb,
            )
        except ImportError:
            print("[demo] matplotlib not installed — skipping scatter plot.")

    return {
        "leaderboard": leaderboard,
        "efficiency_ranking": ranking,
        "results": result_dicts,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    run_demo(
        sequences=args.sequences,
        frames=args.frames,
        motion=args.motion,
        memory_budget_mb=args.memory_budget,
        output_dir=args.output_dir,
        plot_path=args.plot,
        quiet=args.quiet,
        trackers=args.trackers,
    )


if __name__ == "__main__":
    main()
