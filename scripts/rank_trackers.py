#!/usr/bin/env python3
"""Rank and statistically compare trackers from saved benchmark results.

Loads JSON result files produced by BenchmarkReporter, computes bootstrap
confidence intervals, runs pairwise Wilcoxon signed-rank tests, and prints
a ranked leaderboard — enabling research-grade tracker comparison without
re-running the benchmark.

Usage::

    # Rank by mean IoU (default)
    python -m scripts.rank_trackers results/mosse.json results/kcf.json

    # Rank by FPS (higher is better) and show all pairwise tests
    python -m scripts.rank_trackers results/*.json --metric fps --pairwise

    # Rank by latency (lower is better), save Markdown leaderboard
    python -m scripts.rank_trackers results/*.json \\
        --metric mean_latency_ms --asc --output leaderboard.md

    # Use more bootstrap resamples for tighter CIs
    python -m scripts.rank_trackers results/*.json --bootstrap 20000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.metrics.statistical import (
    PairwiseComparison,
    compare_trackers,
    rank_trackers,
)

_METRICS = ("mean_iou", "fps", "peak_memory_mb", "mean_latency_ms", "energy_per_frame_mj")


# ---------------------------------------------------------------------------
# Lightweight result stubs — parse saved JSON without the full package
# ---------------------------------------------------------------------------

def _load_result_stub(path: Path):
    """Parse a BenchmarkReporter JSON file into a minimal stub object."""

    class _Profiling:
        def __init__(self, fps, lat, mem):
            self.fps = fps
            self.latency_mean_ms = lat
            self.peak_memory_mb = mem

    class _Energy:
        def __init__(self, mj):
            self.energy_per_frame_mj = mj

    class _Sequence:
        def __init__(self, d):
            self._iou = float(d.get("mean_iou", 0.0))
            self.profiling = _Profiling(
                fps=float(d.get("fps", 0.0)),
                lat=float(d.get("mean_latency_ms", 0.0)),
                mem=float(d.get("peak_memory_mb", 0.0)),
            )
            e = d.get("energy_per_frame_mj")
            self.energy = _Energy(float(e)) if e is not None else None

        @property
        def mean_iou(self):
            return self._iou

    class _Result:
        def __init__(self, data, source_path):
            summary = data.get("summary", {})
            self.tracker_name = summary.get("tracker", source_path.stem)
            self.dataset_name = summary.get("dataset", "unknown")
            self.sequence_results = [_Sequence(s) for s in data.get("sequences", [])]

    with open(path) as fh:
        return _Result(json.load(fh), path)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _markdown_leaderboard(rankings, metric: str, higher_better: bool) -> str:
    direction = "↑ higher is better" if higher_better else "↓ lower is better"
    lines = [
        f"## EOVOT Tracker Leaderboard — `{metric}` ({direction})",
        "",
        f"| Rank | Tracker | {metric} | 95% CI | n_seq |",
        "|:----:|:--------|--------:|:------:|:-----:|",
    ]
    for r in rankings:
        ci_str = f"[{r.ci.lower:.4f}, {r.ci.upper:.4f}]"
        lines.append(
            f"| {r.rank} | {r.tracker_name} | {r.mean_value:.4f} | {ci_str} | {r.n_sequences} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rank and statistically compare EOVOT trackers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "results",
        nargs="+",
        type=Path,
        metavar="RESULT_JSON",
        help="Paths to JSON result files from BenchmarkReporter.",
    )
    p.add_argument(
        "--metric",
        default="mean_iou",
        choices=_METRICS,
        help="Metric to rank and compare by (default: mean_iou).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        metavar="α",
        help="Significance level for Wilcoxon test (default: 0.05).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        metavar="N",
        help="Bootstrap resamples for confidence intervals (default: 5000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--asc",
        action="store_true",
        help="Rank ascending (lower is better — use for latency/memory).",
    )
    p.add_argument(
        "--pairwise",
        action="store_true",
        help="Print all pairwise significance test results.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save leaderboard as a Markdown file.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    results = []
    for path in args.results:
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            return 1
        try:
            r = _load_result_stub(path)
            results.append(r)
            print(f"Loaded  {r.tracker_name:<25s}  {len(r.sequence_results)} sequences  [{path}]")
        except Exception as exc:
            print(f"[ERROR] Could not load {path}: {exc}", file=sys.stderr)
            return 1

    if not results:
        print("[ERROR] No results loaded.", file=sys.stderr)
        return 1

    higher_better = not args.asc
    print()

    # Ranked leaderboard
    rankings = rank_trackers(
        results,
        metric=args.metric,
        higher_is_better=higher_better,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    direction = "higher is better" if higher_better else "lower is better"
    width = 70
    print("=" * width)
    print(f"  LEADERBOARD — {args.metric}  ({direction})")
    print("=" * width)
    for r in rankings:
        print(f"  {r}")
    print()

    # Pairwise significance tests
    if args.pairwise and len(results) >= 2:
        print("=" * width)
        print(f"  PAIRWISE SIGNIFICANCE TESTS  (α={args.alpha})")
        print("=" * width)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                try:
                    cmp: PairwiseComparison = compare_trackers(
                        results[i],
                        results[j],
                        metric=args.metric,
                        alpha=args.alpha,
                        n_bootstrap=args.bootstrap,
                        seed=args.seed,
                    )
                    print(f"  {cmp}")
                    if cmp.ci_a:
                        print(f"    {cmp.tracker_a}: {cmp.ci_a}")
                    if cmp.ci_b:
                        print(f"    {cmp.tracker_b}: {cmp.ci_b}")
                    print()
                except Exception as exc:
                    print(f"  [WARN] Comparison skipped: {exc}")

    # Save Markdown
    if args.output is not None:
        md = _markdown_leaderboard(rankings, args.metric, higher_better)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Leaderboard saved → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
