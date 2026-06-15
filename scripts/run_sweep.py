#!/usr/bin/env python3
"""Experiment sweep CLI for EOVOT.

Runs multiple trackers against a dataset in a single command, ranks them
by accuracy, and saves a JSON report.  Optionally auto-detects hardware
to configure CPU energy estimation.

Usage
-----
::

    # Classical trackers on OTB-100 (first 10 sequences, auto-detect TDP)
    python scripts/run_sweep.py \\
        --trackers MOSSE KCF CSRT MedianFlow \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 10 \\
        --auto-tdp

    # GOT-10k validation split with explicit TDP for Raspberry Pi
    python scripts/run_sweep.py \\
        --trackers MOSSE KCF \\
        --dataset-loader GOT10kDataset \\
        --dataset-root /data/GOT-10k \\
        --split val \\
        --tdp-watts 6.0 \\
        --output-dir results/rpi-sweep

    # YAML config (recommended for reproducible experiments)
    python scripts/run_sweep.py \\
        --config configs/experiments/classical_sweep.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eovot.benchmark.sweep import SweepConfig, SweepRunner
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.hardware.detector import detect_platform, get_recommended_tdp
from eovot.trackers.csrt import CSRTTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.median_flow import MedianFlowTracker
from eovot.trackers.mosse import MOSSETracker

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

TRACKER_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "CSRT": CSRTTracker,
    "MedianFlow": MedianFlowTracker,
}

DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _build_dataset(loader_name: str, root: str, split: str, max_sequences):
    cls = DATASET_REGISTRY[loader_name]
    if loader_name in ("GOT10kDataset", "LaSOTDataset"):
        return cls(root=root, split=split, max_sequences=max_sequences)
    return cls(root=root)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_from_config(cfg: dict) -> None:
    """Execute a sweep described by *cfg* (loaded from YAML)."""
    exp = cfg.get("experiment", {})
    ds_cfg = cfg["dataset"]
    sw_cfg = cfg.get("sweep", {})
    hw_cfg = cfg.get("hardware", {})

    # Hardware / energy
    tdp_watts = None
    if hw_cfg.get("auto_tdp", False):
        hw = detect_platform()
        tdp_watts = hw.tdp_watts
        print(f"[hardware] {hw} → TDP={tdp_watts} W")
    elif hw_cfg.get("tdp_watts") is not None:
        tdp_watts = float(hw_cfg["tdp_watts"])

    # Dataset
    loader_name = ds_cfg.get("loader", "OTBDataset")
    dataset = _build_dataset(
        loader_name,
        ds_cfg["root"],
        ds_cfg.get("split", "val"),
        ds_cfg.get("max_sequences"),
    )

    # Trackers
    tracker_names = sw_cfg.get("trackers", list(TRACKER_REGISTRY))
    unknown = [t for t in tracker_names if t not in TRACKER_REGISTRY]
    if unknown:
        print(f"[ERROR] Unknown trackers: {unknown}", file=sys.stderr)
        sys.exit(1)
    trackers = {name: TRACKER_REGISTRY[name] for name in tracker_names}

    config = SweepConfig(
        name=exp.get("name", "sweep"),
        trackers=trackers,
        dataset=dataset,
        dataset_name=ds_cfg.get("name", loader_name),
        max_sequences=ds_cfg.get("max_sequences"),
        tdp_watts=tdp_watts,
        verbose=sw_cfg.get("verbose", True),
        output_dir=exp.get("output_dir", "results/"),
    )

    runner = SweepRunner()
    result = runner.run(config)

    out_path = result.save(config.output_dir)
    print(f"\n[Sweep report] saved → {out_path}")


def _run_from_args(args: argparse.Namespace) -> None:
    """Execute a sweep from parsed CLI arguments."""
    if not args.dataset_root:
        print("[ERROR] --dataset-root is required when not using --config", file=sys.stderr)
        sys.exit(1)

    # Hardware / energy
    tdp_watts: float | None = None
    if args.auto_tdp:
        hw = detect_platform()
        tdp_watts = hw.tdp_watts
        print(f"[hardware] {hw} → TDP={tdp_watts} W")
    elif args.tdp_watts is not None:
        tdp_watts = args.tdp_watts

    # Dataset
    dataset = _build_dataset(
        args.dataset_loader,
        args.dataset_root,
        args.split,
        args.max_sequences,
    )

    # Trackers
    unknown = [t for t in args.trackers if t not in TRACKER_REGISTRY]
    if unknown:
        print(f"[ERROR] Unknown trackers: {unknown}", file=sys.stderr)
        sys.exit(1)
    trackers = {name: TRACKER_REGISTRY[name] for name in args.trackers}

    dataset_name = args.dataset_name or args.dataset_loader

    config = SweepConfig(
        name=args.name,
        trackers=trackers,
        dataset=dataset,
        dataset_name=dataset_name,
        max_sequences=args.max_sequences,
        tdp_watts=tdp_watts,
        verbose=not args.quiet,
        output_dir=args.output_dir,
    )

    runner = SweepRunner()
    result = runner.run(config)

    out_path = result.save(config.output_dir)
    print(f"\n[Sweep report] saved → {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sweep",
        description="EOVOT Sweep — compare multiple trackers on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", metavar="PATH",
        help="Path to a YAML sweep config file (overrides all other flags).",
    )
    parser.add_argument(
        "--trackers", nargs="+", default=list(TRACKER_REGISTRY),
        metavar="TRACKER",
        help=f"Trackers to compare. Choices: {list(TRACKER_REGISTRY)}",
    )
    parser.add_argument(
        "--dataset-root", metavar="DIR",
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--dataset-loader", default="OTBDataset",
        choices=list(DATASET_REGISTRY),
        help="Dataset loader class.",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="Human-readable dataset label (defaults to --dataset-loader).",
    )
    parser.add_argument(
        "--split", default="val",
        help="Dataset split (for GOT-10k/LaSOT).",
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None, metavar="N",
        help="Evaluate only the first N sequences.",
    )
    parser.add_argument(
        "--name", default="sweep",
        help="Experiment name used in output filenames.",
    )
    parser.add_argument(
        "--output-dir", default="results/",
        help="Directory for output reports.",
    )
    parser.add_argument(
        "--tdp-watts", type=float, default=None, metavar="W",
        help="Explicit TDP in Watts for energy estimation.",
    )
    parser.add_argument(
        "--auto-tdp", action="store_true",
        help="Auto-detect hardware platform and use its recommended TDP.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sequence progress output.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        cfg = _load_yaml(args.config)
        _run_from_config(cfg)
    else:
        _run_from_args(args)


if __name__ == "__main__":
    main()
