#!/usr/bin/env python3
"""Hyperparameter grid search CLI for EOVOT trackers.

Evaluates all combinations in a parameter grid and outputs a ranked
Markdown table, a parameter sensitivity report, and a JSON result file.

Usage — YAML-driven (recommended)::

    python scripts/run_grid_search.py --config configs/grid_search/mosse_grid.yaml

Usage — inline flags::

    python scripts/run_grid_search.py \\
        --tracker MOSSE \\
        --param learning_rate 0.075 0.10 0.125 0.15 \\
        --param sigma 1.5 2.0 2.5 \\
        --dataset SyntheticDataset \\
        --num-sequences 10 \\
        --num-frames 100 \\
        --output-dir results/grid_search/

Use ``--max-sequences`` to run a quick subset evaluation::

    python scripts/run_grid_search.py \\
        --config configs/grid_search/mosse_grid.yaml \\
        --max-sequences 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.grid_search import GridSearchEngine
from eovot.trackers.csrt import CSRTTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.median_flow import MedianFlowTracker
from eovot.trackers.mil import MILTracker
from eovot.trackers.mosse import MOSSETracker

_TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "MedianFlow": MedianFlowTracker,
    "CSRT": CSRTTracker,
    "MIL": MILTracker,
}


def _build_engine_from_args(args: argparse.Namespace) -> GridSearchEngine:
    """Construct GridSearchEngine from parsed CLI arguments."""
    tracker_cls = _TRACKER_REGISTRY[args.tracker]
    param_grid = {}
    for name, *values in (args.param or []):
        # Attempt numeric coercion: int → float → str
        parsed = []
        for v in values:
            try:
                parsed.append(int(v))
            except ValueError:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
        param_grid[name] = parsed

    if not param_grid:
        raise SystemExit(
            "[ERROR] No parameters specified. "
            "Use --param <name> <v1> <v2> ... or --config."
        )
    return GridSearchEngine(tracker_cls=tracker_cls, param_grid=param_grid)


def _build_dataset_from_args(args: argparse.Namespace):
    """Build a SyntheticDataset from inline flags."""
    return SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_grid_search",
        description="EOVOT hyperparameter grid search for tracker optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Config file (takes precedence over inline flags) ---
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help="Path to a grid search YAML config file.",
    )

    # --- Inline tracker flags ---
    parser.add_argument(
        "--tracker",
        choices=list(_TRACKER_REGISTRY),
        default="MOSSE",
        help="Tracker to search. Ignored when --config is used.",
    )
    parser.add_argument(
        "--param",
        nargs="+",
        action="append",
        metavar=("NAME", "VALUE"),
        help=(
            "Parameter sweep: --param learning_rate 0.075 0.10 0.125. "
            "Repeat for multiple parameters.  Ignored when --config is used."
        ),
    )

    # --- Inline dataset flags ---
    parser.add_argument(
        "--dataset",
        choices=["SyntheticDataset"],
        default="SyntheticDataset",
        help="Dataset to benchmark on (inline mode only).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Number of synthetic sequences.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        metavar="N",
        help="Frames per synthetic sequence.",
    )
    parser.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default="linear",
        help="Synthetic motion pattern.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset.",
    )

    # --- Common flags ---
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap sequences per combination for a quick test run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/grid_search"),
        metavar="DIR",
        help="Directory for JSON and Markdown outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Show only top-N rows in the Markdown table.",
    )
    parser.add_argument(
        "--metric",
        choices=["mean_iou", "success_auc", "precision_auc", "fps"],
        default="mean_iou",
        help="Primary metric for ranking and sensitivity analysis.",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip the parameter sensitivity report.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Build engine + dataset                                               #
    # ------------------------------------------------------------------ #

    if args.config is not None:
        if not args.config.exists():
            raise SystemExit(f"[ERROR] Config file not found: {args.config}")
        with args.config.open() as f:
            config = yaml.safe_load(f)
        engine = GridSearchEngine.from_config(config)
    else:
        engine = _build_engine_from_args(args)

    # Determine dataset name for report label
    tracker_name = engine.tracker_cls.__name__

    # ------------------------------------------------------------------ #
    # Run search                                                           #
    # ------------------------------------------------------------------ #

    print(f"\n{'=' * 65}")
    print(f"  EOVOT Grid Search — {tracker_name}")
    print(f"  Parameters: {list(engine.param_grid)}")
    n_combos = 1
    for v in engine.param_grid.values():
        n_combos *= len(v)
    print(f"  Total combinations: {n_combos}")
    print(f"{'=' * 65}\n")

    if args.config is not None:
        entries = engine.run_from_config(config)
        ds_label = config.get("dataset", {}).get("name", "dataset")
    else:
        dataset = _build_dataset_from_args(args)
        entries = engine.run(
            dataset,
            dataset_name="Synthetic",
            max_sequences=args.max_sequences,
        )
        ds_label = "Synthetic"

    if not entries:
        print("[WARN] No results produced.")
        return

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #

    best = engine.best_config(entries, metric=args.metric)
    print(f"\n{'=' * 65}")
    print(f"  Best config by {args.metric}:")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"  Score: {_metric_value(entries[0], args.metric):.4f}")
    print(f"{'=' * 65}\n")

    print(engine.to_markdown(entries, top_n=args.top_n))

    if not args.no_sensitivity:
        print(engine.sensitivity_to_markdown(entries, metric=args.metric))

    # ------------------------------------------------------------------ #
    # Save outputs                                                         #
    # ------------------------------------------------------------------ #

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tracker_name}-{ds_label}-grid"

    json_path = engine.save_json(entries, path=str(args.output_dir / f"{stem}.json"))
    print(f"[JSON] saved → {json_path}")

    md_path = args.output_dir / f"{stem}.md"
    md_content = (
        f"# Grid Search: {tracker_name} on {ds_label}\n\n"
        f"## Results\n\n"
        + engine.to_markdown(entries)
        + "\n"
        + engine.sensitivity_to_markdown(entries, metric=args.metric)
    )
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[MD]   saved → {md_path}")


def _metric_value(entry, metric: str) -> float:
    return {
        "mean_iou": entry.mean_iou,
        "success_auc": entry.success_auc,
        "precision_auc": entry.precision_auc,
        "fps": entry.fps,
    }[metric]


if __name__ == "__main__":
    main()
