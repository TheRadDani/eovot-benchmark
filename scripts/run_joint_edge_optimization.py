#!/usr/bin/env python3
"""Joint 2-D edge optimization CLI: frame-skip × resolution-scale grid search.

Sweeps all combinations of skip rates and scale factors for a given tracker,
identifies Pareto-optimal deployment points, and recommends the best operating
configuration within a target FPS budget.

Usage
-----
    # Config-driven run (recommended)
    python scripts/run_joint_edge_optimization.py \\
        --config configs/experiments/joint_edge_optimization.yaml

    # Inline arguments with synthetic dataset (no download needed)
    python scripts/run_joint_edge_optimization.py \\
        --tracker MOSSE --synthetic \\
        --skip-rates 1 2 4 --scale-factors 1.0 0.75 0.5 \\
        --fps-budget 30

    # Inline arguments with a real dataset
    python scripts/run_joint_edge_optimization.py \\
        --tracker KCF \\
        --dataset-root /data/OTB100 --dataset-name OTB100 \\
        --skip-rates 1 2 3 4 --scale-factors 1.0 0.75 0.5 0.25 \\
        --fps-budget 60 --min-iou 0.55

    # List available trackers and exit
    python scripts/run_joint_edge_optimization.py --list-trackers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eovot.analysis.joint_analysis import JointEdgeAnalyzer
from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.trackers.registry import TRACKER_REGISTRY, available_trackers, build_tracker


DATASET_REGISTRY: Dict[str, Any] = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
    "SyntheticDataset": SyntheticDataset,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config_from_args(args: argparse.Namespace) -> Dict:
    """Build a config dict from CLI arguments."""
    opt = args.optimization if hasattr(args, "optimization") else {}

    skip_rates: List[int] = (
        [int(x) for x in args.skip_rates]
        if args.skip_rates
        else [1, 2, 4]
    )
    scale_factors: List[float] = (
        [float(x) for x in args.scale_factors]
        if args.scale_factors
        else [1.0, 0.75, 0.5]
    )

    dataset_cfg: Dict[str, Any]
    if args.synthetic:
        dataset_cfg = {
            "loader": "SyntheticDataset",
            "name": "Synthetic",
            "num_sequences": getattr(args, "synthetic_sequences", 5),
            "num_frames": getattr(args, "synthetic_frames", 100),
        }
    else:
        dataset_cfg = {
            "loader": getattr(args, "dataset_loader", "OTBDataset"),
            "name": getattr(args, "dataset_name", "OTB"),
            "root": getattr(args, "dataset_root", "/data/OTB100"),
        }

    return {
        "experiment": {
            "name": f"joint-{args.tracker}-opt",
            "output_dir": getattr(args, "output_dir", "results/joint_edge"),
        },
        "dataset": dataset_cfg,
        "tracker": {"name": args.tracker, "params": {}},
        "optimization": {
            "skip_rates": skip_rates,
            "scale_factors": scale_factors,
            "memory_budget_mb": getattr(args, "memory_budget_mb", 512.0),
        },
        "reporting": {
            "fps_budget": getattr(args, "fps_budget", 30.0),
            "min_iou": getattr(args, "min_iou", 0.0),
        },
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(cfg: Dict) -> Any:
    """Instantiate the dataset from a config dict."""
    loader_name: str = cfg.get("loader", "SyntheticDataset")
    cls = DATASET_REGISTRY.get(loader_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset loader '{loader_name}'. "
            f"Available: {sorted(DATASET_REGISTRY)}"
        )

    if loader_name == "SyntheticDataset":
        kwargs: Dict[str, Any] = {}
        for key in ("num_sequences", "num_frames", "frame_size", "bbox_size", "seed", "motion"):
            if key in cfg:
                kwargs[key] = cfg[key]
        return cls(**kwargs)

    root = cfg.get("root") or cfg.get("dataset_root")
    if not root:
        raise ValueError("dataset.root is required for non-synthetic datasets.")
    kwargs = {"root": root}
    if "split" in cfg:
        kwargs["split"] = cfg["split"]
    if "max_sequences" in cfg and cfg["max_sequences"] is not None:
        kwargs["max_sequences"] = cfg["max_sequences"]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOVOT joint 2-D edge optimization: skip × scale grid search"
    )
    parser.add_argument("--config", help="Path to YAML experiment config")
    parser.add_argument("--tracker", default="MOSSE", help="Tracker name")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic dataset (no download needed)")
    parser.add_argument("--synthetic-sequences", type=int, default=5,
                        dest="synthetic_sequences")
    parser.add_argument("--synthetic-frames", type=int, default=100,
                        dest="synthetic_frames")
    parser.add_argument("--dataset-root", dest="dataset_root")
    parser.add_argument("--dataset-name", dest="dataset_name", default="OTB100")
    parser.add_argument("--dataset-loader", dest="dataset_loader", default="OTBDataset")
    parser.add_argument("--skip-rates", nargs="+", dest="skip_rates",
                        help="e.g. --skip-rates 1 2 4")
    parser.add_argument("--scale-factors", nargs="+", dest="scale_factors",
                        help="e.g. --scale-factors 1.0 0.75 0.5")
    parser.add_argument("--fps-budget", type=float, default=30.0, dest="fps_budget",
                        help="Minimum FPS for the optimal-point recommendation")
    parser.add_argument("--min-iou", type=float, default=0.0, dest="min_iou",
                        help="Minimum IoU for the optimal-point recommendation")
    parser.add_argument("--memory-budget-mb", type=float, default=512.0,
                        dest="memory_budget_mb")
    parser.add_argument("--output-dir", default="results/joint_edge", dest="output_dir")
    parser.add_argument("--list-trackers", action="store_true", dest="list_trackers")

    args = parser.parse_args()

    if args.list_trackers:
        print("Available trackers:", available_trackers())
        return

    # Build config
    if args.config:
        cfg = _load_config(args.config)
    else:
        if not args.synthetic and not args.dataset_root:
            print(
                "Error: --dataset-root is required for non-synthetic datasets. "
                "Use --synthetic for a quick test.",
                file=sys.stderr,
            )
            sys.exit(1)
        cfg = _config_from_args(args)

    exp_cfg = cfg.get("experiment", {})
    ds_cfg = cfg.get("dataset", {})
    tr_cfg = cfg.get("tracker", {})
    opt_cfg = cfg.get("optimization", {})
    rep_cfg = cfg.get("reporting", {})

    # Build components
    tracker_name: str = tr_cfg.get("name", "MOSSE")
    tracker_params: Dict = tr_cfg.get("params", {})
    tracker = build_tracker(tracker_name, **tracker_params)

    dataset = _build_dataset(ds_cfg)
    dataset_name: str = ds_cfg.get("name", "unknown")

    skip_rates: List[int] = opt_cfg.get("skip_rates", [1, 2, 4])
    scale_factors: List[float] = opt_cfg.get("scale_factors", [1.0, 0.75, 0.5])
    memory_budget_mb: float = float(opt_cfg.get("memory_budget_mb", 512.0))

    fps_budget: float = float(rep_cfg.get("fps_budget", 30.0))
    min_iou: float = float(rep_cfg.get("min_iou", 0.0))

    print(f"\n{'='*60}")
    print(f" EOVOT Joint Edge Optimization")
    print(f" Tracker:  {tracker_name}")
    print(f" Dataset:  {dataset_name}  ({len(dataset)} sequences)")
    print(f" Grid:     skip={skip_rates} × scale={scale_factors}")
    print(f" Budget:   ≥{fps_budget} FPS, IoU≥{min_iou}")
    print(f"{'='*60}\n")

    engine = BenchmarkEngine(verbose=False)
    analyzer = JointEdgeAnalyzer(engine, memory_budget_mb=memory_budget_mb)

    result = analyzer.analyze(
        tracker,
        dataset,
        dataset_name=dataset_name,
        skip_rates=skip_rates,
        scale_factors=scale_factors,
    )

    # Print results
    print(result.to_markdown_table())
    print()
    print(result.to_ranked_markdown())
    print()

    # Recommendation
    best_skip, best_scale, best_entry = result.optimal_point(fps_budget, min_iou)
    print(f"\n{'='*60}")
    print(f" RECOMMENDATION (FPS≥{fps_budget}, IoU≥{min_iou})")
    print(f" skip_rate={best_skip}, scale_factor={best_scale:.2f}")
    print(f" → mIoU={best_entry.mean_iou:.4f}  FPS={best_entry.fps:.1f}  EES={best_entry.ees:.4f}")
    print(f"{'='*60}\n")

    if result.pareto_front():
        print("Pareto-optimal points (sorted by FPS):")
        for pt in result.pareto_front():
            print(f"  {pt}")

    # Save outputs
    output_dir = Path(exp_cfg.get("output_dir", "results/joint_edge"))
    exp_name = exp_cfg.get("name", f"joint-{tracker_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"{exp_name}_grid.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.to_markdown_table())
        f.write("\n\n")
        f.write(result.to_ranked_markdown())
        f.write("\n\n")
        f.write(f"**Recommendation** (FPS≥{fps_budget}, IoU≥{min_iou}): ")
        f.write(f"skip={best_skip}, scale={best_scale:.2f}, "
                f"mIoU={best_entry.mean_iou:.4f}, FPS={best_entry.fps:.1f}\n")

    print(f"Saved grid report → {md_path}")

    summary = {
        "experiment": exp_name,
        "tracker": tracker_name,
        "dataset": dataset_name,
        "grid": {
            f"skip{sr}_scale{sf:.2f}": {
                "mean_iou": round(entry.mean_iou, 4),
                "fps": round(entry.fps, 2),
                "ees": round(entry.ees, 4),
                "on_pareto_front": entry.on_pareto_front,
            }
            for sr, scale_dict in result.grid.items()
            for sf, entry in scale_dict.items()
        },
        "recommendation": {
            "skip_rate": best_skip,
            "scale_factor": best_scale,
            "mean_iou": round(best_entry.mean_iou, 4),
            "fps": round(best_entry.fps, 2),
            "ees": round(best_entry.ees, 4),
        },
    }
    json_path = output_dir / f"{exp_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary → {json_path}")


if __name__ == "__main__":
    main()
