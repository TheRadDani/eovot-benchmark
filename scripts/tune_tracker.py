#!/usr/bin/env python3
"""CLI for tracker hyperparameter tuning (grid search or sensitivity analysis).

Usage examples
--------------
Grid search — MOSSE on synthetic data, optimise mean_iou:

    python scripts/tune_tracker.py \\
        --mode grid \\
        --tracker MOSSE \\
        --metric mean_iou \\
        --params "learning_rate=[0.05,0.125,0.2] sigma=[1.0,2.0,3.0]" \\
        --dataset synthetic \\
        --num-sequences 10 \\
        --frames-per-sequence 80 \\
        --output results/mosse_grid.md

Sensitivity analysis — KCF, optimise mean_fps:

    python scripts/tune_tracker.py \\
        --mode sensitivity \\
        --tracker KCF \\
        --metric mean_fps \\
        --base-params "learning_rate=0.075 kernel_sigma=0.5" \\
        --params "learning_rate=[0.03,0.05,0.075,0.1] kernel_sigma=[0.2,0.5,0.8]" \\
        --dataset synthetic \\
        --output results/kcf_sensitivity.md
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Allow running as a top-level script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.tuning.grid_search import GridSearchRunner
from eovot.tuning.sensitivity import SensitivityAnalyzer

_TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
}

_TRACKER_DEFAULTS = {
    "MOSSE": {"learning_rate": 0.125, "sigma": 2.0},
    "KCF": {"learning_rate": 0.075, "kernel_sigma": 0.5, "padding": 1.5, "lambda_": 1e-4},
}


def _parse_params(raw: str) -> dict:
    """Parse ``"key=[v1,v2] key2=[v3,v4]"`` into a dict of lists."""
    result = {}
    for token in raw.strip().split():
        key, _, val_str = token.partition("=")
        result[key.strip()] = ast.literal_eval(val_str.strip())
    return result


def _parse_base_params(raw: str) -> dict:
    """Parse ``"key=val key2=val2"`` into a dict of scalars."""
    result = {}
    for token in raw.strip().split():
        key, _, val_str = token.partition("=")
        result[key.strip()] = ast.literal_eval(val_str.strip())
    return result


def _build_dataset(args: argparse.Namespace) -> SyntheticDataset:
    return SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.frames_per_sequence,
        seed=args.seed,
    )


def run_grid(args: argparse.Namespace) -> None:
    tracker_class = _TRACKER_REGISTRY[args.tracker]
    param_grid = _parse_params(args.params)
    dataset = _build_dataset(args)

    runner = GridSearchRunner(
        tracker_class=tracker_class,
        param_grid=param_grid,
        metric=args.metric,
        verbose=not args.quiet,
        max_sequences=args.max_sequences,
    )
    result = runner.run(dataset, dataset_name=args.dataset)

    report = result.to_markdown()
    print(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        csv_path = out.with_suffix(".csv")
        result.save_csv(str(csv_path))
        print(f"\nSaved: {out}  |  {csv_path}")


def run_sensitivity(args: argparse.Namespace) -> None:
    tracker_class = _TRACKER_REGISTRY[args.tracker]
    param_ranges = _parse_params(args.params)

    if args.base_params:
        base_params = _parse_base_params(args.base_params)
    else:
        base_params = dict(_TRACKER_DEFAULTS.get(args.tracker, {}))

    dataset = _build_dataset(args)

    analyzer = SensitivityAnalyzer(
        tracker_class=tracker_class,
        base_params=base_params,
        param_ranges=param_ranges,
        metric=args.metric,
        verbose=not args.quiet,
        max_sequences=args.max_sequences,
    )
    report = analyzer.run(dataset, dataset_name=args.dataset)

    markdown = report.to_markdown()
    print(markdown)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print(f"\nSaved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tracker hyperparameter tuning — grid search or OAT sensitivity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["grid", "sensitivity"], required=True,
        help="Tuning mode: exhaustive grid search or one-at-a-time sensitivity analysis.",
    )
    parser.add_argument(
        "--tracker", choices=list(_TRACKER_REGISTRY), required=True,
        help="Tracker to tune.",
    )
    parser.add_argument(
        "--metric", default="mean_iou",
        help="BenchmarkResult attribute to optimise (default: mean_iou).",
    )
    parser.add_argument(
        "--params", required=True,
        help=(
            'Parameter grid / ranges as space-separated key=[v1,v2,...] tokens. '
            'Example: "learning_rate=[0.05,0.125] sigma=[1.0,2.0]"'
        ),
    )
    parser.add_argument(
        "--base-params", default="",
        help=(
            "Baseline parameter values for sensitivity mode (default: tracker defaults). "
            'Example: "learning_rate=0.125 sigma=2.0"'
        ),
    )
    parser.add_argument(
        "--dataset", default="synthetic",
        help="Dataset name (currently only 'synthetic' is supported via built-in generator).",
    )
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--frames-per-sequence", type=int, default=60)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="", help="Path for the Markdown report output.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress output.")

    args = parser.parse_args()

    if args.mode == "grid":
        run_grid(args)
    else:
        run_sensitivity(args)


if __name__ == "__main__":
    main()
