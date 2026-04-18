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
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.csrt import CSRTTracker
from eovot.trackers.median_flow import MedianFlowTracker


# ------------------------------------------------------------------ #
# Tracker registry — extend as new trackers are added                 #
# ------------------------------------------------------------------ #

TRACKER_REGISTRY: Dict[str, Any] = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "CSRT": CSRTTracker,
    "MedianFlow": MedianFlowTracker,
}


# ------------------------------------------------------------------ #
# Dataset registry                                                     #
# ------------------------------------------------------------------ #

DATASET_REGISTRY: Dict[str, Any] = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
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
    return {
        "experiment": {
            "name": "cli-run",
            "output_dir": args.output_dir,
        },
        "dataset": {
            "name": args.dataset_name,
            "loader": "OTBDataset",
            "root": args.dataset_root,
            "max_sequences": args.max_sequences,
        },
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
    loader_cls = DATASET_REGISTRY.get(ds_cfg.get("loader", "OTBDataset"))
    if loader_cls is None:
        print(f"[ERROR] Unknown dataset loader: {ds_cfg['loader']}", file=sys.stderr)
        sys.exit(1)
    loader_name = ds_cfg.get("loader", "OTBDataset")
    if loader_name in ("GOT10kDataset", "LaSOTDataset"):
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        cfg = _load_config(args.config)
    elif args.dataset_root:
        cfg = _config_from_args(args)
    else:
        parser.print_help()
        sys.exit(0)

    run_from_config(cfg)


if __name__ == "__main__":
    main()
