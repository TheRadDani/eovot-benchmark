"""CLI for joint frame-skip × resolution deployment optimization.

Sweeps a tracker across a 2-D grid of (skip_rate × scale_factor) configs,
computes the Pareto-optimal set, and outputs the recommended configuration
for an edge deployment budget.

Usage
-----
    # Quick synthetic demo — no real dataset needed
    python scripts/optimize_deployment.py --tracker KCF --synthetic

    # Explicit skip/scale grid
    python scripts/optimize_deployment.py \\
        --tracker MOSSE --synthetic \\
        --skip-rates 1 2 4 \\
        --scale-factors 1.0 0.75 0.5 0.25

    # Constrain by IoU budget (≥0.80 of baseline) and FPS budget (≤500 FPS)
    python scripts/optimize_deployment.py \\
        --tracker KCF --synthetic \\
        --min-iou-fraction 0.80 \\
        --fps-budget 500

    # Save results as JSON
    python scripts/optimize_deployment.py \\
        --tracker KCF --synthetic \\
        --output results/deploy_opt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make repo root importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.analysis.joint_optimizer import JointDeploymentOptimizer
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker

_TRACKERS = {"MOSSE": MOSSETracker, "KCF": KCFTracker}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint frame-skip × resolution deployment optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--tracker",
        choices=list(_TRACKERS),
        default="KCF",
        help="Tracker to benchmark (default: KCF)",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic dataset (no real data required)",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of synthetic sequences (default: 5)",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Frames per synthetic sequence (default: 100)",
    )
    p.add_argument(
        "--skip-rates",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        metavar="N",
        help="Skip rates to sweep (default: 1 2 4). Baseline (1) is always included.",
    )
    p.add_argument(
        "--scale-factors",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5],
        metavar="F",
        help="Resolution scale factors to sweep (default: 1.0 0.75 0.5). Baseline (1.0) always included.",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of sequences evaluated (for quick tests)",
    )
    p.add_argument(
        "--min-iou-fraction",
        type=float,
        default=None,
        metavar="F",
        help="Minimum IoU as a fraction of baseline IoU for recommend() (e.g. 0.85)",
    )
    p.add_argument(
        "--fps-budget",
        type=float,
        default=None,
        metavar="FPS",
        help="Maximum FPS target for recommend() (e.g. 200)",
    )
    p.add_argument(
        "--dataset-name",
        default="Synthetic",
        help="Display name for the dataset (default: Synthetic)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save results as JSON to PATH",
    )
    p.add_argument(
        "--skip-mode",
        choices=["repeat", "linear"],
        default="repeat",
        help="Frame-skip propagation mode (default: repeat)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.synthetic:
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            seed=42,
        )
        dataset_name = args.dataset_name
    else:
        print("ERROR: only --synthetic is supported in this script. "
              "For real datasets wire up your loader before calling optimize().",
              file=sys.stderr)
        sys.exit(1)

    tracker = _TRACKERS[args.tracker]()
    engine = BenchmarkEngine(verbose=False)
    optimizer = JointDeploymentOptimizer(engine, skip_mode=args.skip_mode)

    print(f"Running joint optimization: {args.tracker} on {dataset_name}")
    print(f"  skip_rates   : {sorted(set(args.skip_rates) | {1})}")
    print(f"  scale_factors: {sorted(set(args.scale_factors) | {1.0}, reverse=True)}")
    print()

    result = optimizer.optimize(
        tracker,
        dataset,
        dataset_name=dataset_name,
        skip_rates=args.skip_rates,
        scale_factors=args.scale_factors,
        max_sequences=args.max_sequences,
    )

    print(result.to_markdown_table())
    print()

    min_iou: float | None = None
    if args.min_iou_fraction is not None:
        baseline = result.baseline()
        if baseline is not None:
            min_iou = baseline.mean_iou * args.min_iou_fraction

    rec = result.recommend(min_iou=min_iou, fps_budget=args.fps_budget)
    if rec is not None:
        print(f"Recommended config: {rec.config_label}")
        print(f"  mIoU={rec.mean_iou:.4f}  FPS={rec.mean_fps:.1f}  "
              f"gain={rec.fps_gain:.2f}×  ΔIoU={rec.iou_degradation:+.4f}")
    else:
        print("No config satisfies the given constraints.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
