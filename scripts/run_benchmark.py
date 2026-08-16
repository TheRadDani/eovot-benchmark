"""Command-line entry point for EOVOT benchmarking.

Usage
-----
    # Run with a YAML config file:
    python scripts/run_benchmark.py --config configs/default.yaml

    # Quick single-tracker test with explicit arguments:
    python scripts/run_benchmark.py \\
        --tracker MOSSE \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 5

    # Headless smoke test using the built-in synthetic dataset (no download needed):
    python scripts/run_benchmark.py \\
        --tracker MOSSE --synthetic \\
        --synthetic-sequences 5 --synthetic-frames 50

    # Installed as a package entry point:
    eovot --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure the repo root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import TRACKER_REGISTRY


# ------------------------------------------------------------------ #
# Tracker registry — shared with eovot.experiment.runner; see          #
# eovot/trackers/registry.py to add a new tracker in one place.        #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
# Dataset registry                                                     #
# ------------------------------------------------------------------ #

DATASET_REGISTRY: Dict[str, Any] = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
    "SyntheticDataset": SyntheticDataset,
}


# ------------------------------------------------------------------ #
# Config loading                                                       #
# ------------------------------------------------------------------ #

def _load_config(path: str) -> Dict:
    """Load and return a YAML config as a nested dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config_from_args(args: argparse.Namespace) -> Dict:
    """Build a minimal config dict from CLI arguments."""
    if args.synthetic:
        dataset_cfg: Dict[str, Any] = {
            "name": args.dataset_name or "Synthetic",
            "loader": "SyntheticDataset",
            "num_sequences": args.synthetic_sequences,
            "num_frames": args.synthetic_frames,
            "frame_size": args.synthetic_size,
            "motion": args.synthetic_motion,
            "max_sequences": args.max_sequences,
        }
    else:
        dataset_cfg = {
            "name": args.dataset_name,
            "loader": "OTBDataset",
            "root": args.dataset_root,
            "max_sequences": args.max_sequences,
        }
    return {
        "experiment": {
            "name": "cli-run",
            "output_dir": args.output_dir,
        },
        "dataset": dataset_cfg,
        "tracker": {
            "name": args.tracker,
            "params": {},
        },
        "benchmark": {
            "verbose": not args.quiet,
            "tdp_watts": args.tdp_watts,
        },
        "reporting": {
            "formats": ["json"],
            "print_summary": True,
        },
    }


# ------------------------------------------------------------------ #
# Core run logic                                                       #
# ------------------------------------------------------------------ #

def run_from_config(cfg: Dict) -> None:
    """Execute a benchmark run described by *cfg*."""
    # --- Dataset ---
    ds_cfg = cfg["dataset"]
    loader_name = ds_cfg.get("loader", "OTBDataset")
    loader_cls = DATASET_REGISTRY.get(loader_name)
    if loader_cls is None:
        print(f"[ERROR] Unknown dataset loader: {loader_name}", file=sys.stderr)
        sys.exit(1)

    if loader_name == "SyntheticDataset":
        frame_size = ds_cfg.get("frame_size", [320, 240])
        dataset = loader_cls(
            num_sequences=ds_cfg.get("num_sequences", 10),
            num_frames=ds_cfg.get("num_frames", 100),
            frame_size=tuple(frame_size) if not isinstance(frame_size, tuple) else frame_size,
            motion=ds_cfg.get("motion", "linear"),
        )
    elif loader_name in ("GOT10kDataset", "LaSOTDataset"):
        dataset = loader_cls(
            ds_cfg["root"],
            split=ds_cfg.get("split", "val"),
            max_sequences=ds_cfg.get("max_sequences"),
        )
    else:
        dataset = loader_cls(ds_cfg["root"])

    # --- Tracker ---
    tr_cfg = cfg["tracker"]
    tracker_cls = TRACKER_REGISTRY.get(tr_cfg["name"])
    if tracker_cls is None:
        available = ", ".join(TRACKER_REGISTRY)
        print(
            f"[ERROR] Unknown tracker '{tr_cfg['name']}'. "
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    tracker = tracker_cls(**tr_cfg.get("params", {}))

    # --- Engine ---
    bm_cfg = cfg.get("benchmark", {})
    engine = BenchmarkEngine(
        verbose=bm_cfg.get("verbose", True),
        tdp_watts=bm_cfg.get("tdp_watts"),
    )

    result = engine.run(
        tracker=tracker,
        dataset=dataset,
        dataset_name=ds_cfg.get("name", "unknown"),
        max_sequences=ds_cfg.get("max_sequences"),
    )

    # --- Reporting ---
    report_cfg = cfg.get("reporting", {})
    output_dir = cfg["experiment"].get("output_dir", "results/")
    exp_name = cfg["experiment"].get("name", "run")
    os.makedirs(output_dir, exist_ok=True)

    summary = result.summary()

    if report_cfg.get("print_summary", True):
        print("\n" + "=" * 60)
        print(" BENCHMARK SUMMARY")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k:<22s}: {v}")
        print("=" * 60)

    formats = report_cfg.get("formats", ["json"])
    if "json" in formats:
        out_path = os.path.join(output_dir, f"{exp_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {out_path}")

    if "csv" in formats:
        import csv
        out_path = os.path.join(output_dir, f"{exp_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
        print(f"Results saved to {out_path}")


# ------------------------------------------------------------------ #
# CLI argument parsing                                                 #
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eovot",
        description="EOVOT — Edge-Optimized Visual Object Tracking Benchmark",
    )
    parser.add_argument(
        "--config", "-c",
        metavar="PATH",
        help="Path to a YAML experiment config file.",
    )
    # Convenience overrides (used when --config is not provided)
    parser.add_argument("--tracker", default="MOSSE",
                        choices=list(TRACKER_REGISTRY),
                        help="Tracker to evaluate.")
    parser.add_argument("--dataset-root", metavar="DIR",
                        help="Path to dataset root directory.")
    parser.add_argument("--dataset-name", default="dataset",
                        help="Label for the dataset in reports.")
    parser.add_argument("--max-sequences", type=int, default=None,
                        metavar="N",
                        help="Evaluate only the first N sequences.")
    parser.add_argument("--output-dir", default="results/",
                        help="Directory for output reports (default: results/).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sequence progress output.")
    parser.add_argument("--tdp-watts", type=float, default=None, metavar="W",
                        help=(
                            "Enable CPU energy estimation with this TDP (Watts). "
                            "E.g. 6.0 for Raspberry Pi 4, 15.0 for a laptop."
                        ))

    # Synthetic dataset mode — allows headless evaluation without any download
    syn = parser.add_argument_group(
        "synthetic dataset",
        "Use the built-in procedural dataset instead of a file-based one. "
        "Pass --synthetic to activate; no --dataset-root required.",
    )
    syn.add_argument("--synthetic", action="store_true",
                     help="Evaluate against the synthetic dataset (no download needed).")
    syn.add_argument("--synthetic-sequences", type=int, default=5, metavar="N",
                     help="Number of synthetic sequences to generate (default: 5).")
    syn.add_argument("--synthetic-frames", type=int, default=50, metavar="N",
                     help="Frames per synthetic sequence (default: 50).")
    syn.add_argument("--synthetic-size", type=int, nargs=2, default=[320, 240],
                     metavar=("W", "H"),
                     help="Frame size in pixels — width height (default: 320 240).")
    syn.add_argument("--synthetic-motion", default="linear",
                     choices=["linear", "circular", "random"],
                     help="Target motion pattern (default: linear).")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        cfg = _load_config(args.config)
    elif args.synthetic or args.dataset_root:
        cfg = _config_from_args(args)
    else:
        parser.print_help()
        sys.exit(0)

    run_from_config(cfg)


if __name__ == "__main__":
    main()
