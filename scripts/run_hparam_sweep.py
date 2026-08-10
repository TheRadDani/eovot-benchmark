"""CLI tool for hyperparameter sweep over tracker configurations.

Runs a grid or random search over the given parameter grid, evaluating
each configuration on the configured dataset, and saves results as JSON
and CSV.

Usage examples::

    # Grid search over KCF learning rate and lambda
    python scripts/run_hparam_sweep.py \\
        --tracker KCF \\
        --params learning_rate=[0.05,0.075,0.10,0.15] lambda_=[1e-5,1e-4,1e-3] \\
        --dataset synthetic \\
        --metric success_auc \\
        --output-dir results/kcf_sweep

    # Random search over MOSSE parameters (20 random draws)
    python scripts/run_hparam_sweep.py \\
        --tracker MOSSE \\
        --params learning_rate=[0.05,0.1,0.15,0.20,0.25] \\
        --mode random --n-random 10 \\
        --dataset synthetic \\
        --output-dir results/mosse_sweep

    # Sweep over DCF with real dataset
    python scripts/run_hparam_sweep.py \\
        --tracker DCF \\
        --params num_hog_bins=[6,9,12] cell_size=[4,8] \\
        --dataset-root /data/OTB100 --dataset-name OTB100 \\
        --max-sequences 10 \\
        --output-dir results/dcf_sweep
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.experiment.hparam_sweep import SweepConfig, TrackerHparamSweep


def _parse_param_grid(specs: list[str]) -> dict:
    """Parse ``key=[v1,v2,...]`` strings into a param_grid dict."""
    grid = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid param spec {spec!r}. Expected format: name=[v1,v2,...]"
            )
        name, vals_str = spec.split("=", 1)
        name = name.strip()
        try:
            vals = ast.literal_eval(vals_str.strip())
        except Exception as exc:
            raise ValueError(
                f"Cannot parse values for '{name}': {vals_str!r}. "
                "Use Python literal syntax, e.g. [0.05,0.1,0.15]"
            ) from exc
        if not isinstance(vals, list):
            vals = [vals]
        grid[name] = vals
    return grid


def _build_dataset(args):
    if args.dataset == "synthetic":
        from eovot.datasets.synthetic import SyntheticDataset
        return SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            frame_size=(320, 240),
            bbox_size=(40, 30),
            motion=args.motion,
            seed=args.seed,
        ), "SyntheticDataset"
    if args.dataset == "otb":
        from eovot.datasets.base import OTBDataset
        return OTBDataset(
            root=args.dataset_root,
            max_sequences=args.max_sequences,
        ), args.dataset_name or "OTB"
    if args.dataset == "got10k":
        from eovot.datasets.got10k import GOT10kDataset
        return GOT10kDataset(
            root=args.dataset_root,
            split=args.split or "val",
            max_sequences=args.max_sequences,
        ), args.dataset_name or "GOT-10k"
    raise ValueError(f"Unknown dataset type: {args.dataset!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for EOVOT trackers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Tracker
    parser.add_argument(
        "--tracker", required=True,
        help="Tracker name (e.g. KCF, MOSSE, DCF, CamShift)",
    )
    parser.add_argument(
        "--params", nargs="+", required=True, metavar="KEY=[v1,v2,...]",
        help="Parameter grid specs, e.g. learning_rate=[0.05,0.1,0.15]",
    )
    parser.add_argument(
        "--metric", default="success_auc",
        choices=["mean_iou", "success_auc", "precision_auc", "mean_fps"],
        help="Metric to maximise (default: success_auc)",
    )
    # Sweep mode
    parser.add_argument(
        "--mode", default="grid", choices=["grid", "random"],
        help="Search strategy: 'grid' (exhaustive) or 'random' (default: grid)",
    )
    parser.add_argument(
        "--n-random", type=int, default=20,
        help="Number of random configs for random search (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    # Dataset
    parser.add_argument(
        "--dataset", default="synthetic",
        choices=["synthetic", "otb", "got10k"],
        help="Dataset type (default: synthetic)",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--num-sequences", type=int, default=5,
        help="Sequences for synthetic dataset (default: 5)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=50,
        help="Frames per sequence for synthetic dataset (default: 50)",
    )
    parser.add_argument(
        "--motion", default="linear",
        choices=["linear", "circular", "random"],
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Cap evaluation to first N sequences per config",
    )
    # Output
    parser.add_argument(
        "--output-dir", default="results/sweep",
        help="Directory for JSON, CSV, and Markdown output",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-configuration progress output",
    )

    args = parser.parse_args()

    # Build param grid
    try:
        param_grid = _parse_param_grid(args.params)
    except ValueError as exc:
        parser.error(str(exc))
        return

    # Build dataset
    try:
        dataset, dataset_name = _build_dataset(args)
    except Exception as exc:
        parser.error(f"Dataset error: {exc}")
        return

    # Configure sweep
    config = SweepConfig(
        tracker_name=args.tracker,
        param_grid=param_grid,
        metric=args.metric,
        max_sequences=args.max_sequences,
    )

    sweep = TrackerHparamSweep(
        mode=args.mode,
        n_random=args.n_random,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print(f"\nStarting {args.mode} search over {len(list(__import__('itertools').product(*param_grid.values())))} total combos")
    result = sweep.run(config, dataset, dataset_name)

    # Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{args.tracker}_{dataset_name}_{args.metric}"
    json_path = str(out / f"{stem}_sweep.json")
    csv_path = str(out / f"{stem}_sweep.csv")
    md_path = str(out / f"{stem}_sweep.md")

    result.save_json(json_path)
    result.to_csv(csv_path)
    Path(md_path).write_text(result.to_markdown())

    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(result.entries)} configurations evaluated")
    print(f"Best {args.metric}: {result.best_score:.4f}")
    print(f"Best params: {result.best_params}")
    print(f"\nSaved: {json_path}")
    print(f"       {csv_path}")
    print(f"       {md_path}")


if __name__ == "__main__":
    main()
