#!/usr/bin/env python3
"""Resolution-scale accuracy–speed tradeoff analysis for EOVOT.

Sweeps a tracker across multiple frame resolutions and reports the
accuracy/FPS tradeoff as a Markdown table.  The optimal scale factor
for a given IoU budget is highlighted.

Usage::

    # Run with the bundled example config:
    python scripts/analyze_resolution.py \\
        --config configs/experiments/resolution_analysis.yaml

    # Quick sweep without a config file:
    python scripts/analyze_resolution.py \\
        --tracker KCF \\
        --scales 1.0 0.75 0.5 0.25 \\
        --num-sequences 5

    # Find the best scale that loses no more than 5% IoU vs. full resolution:
    python scripts/analyze_resolution.py \\
        --tracker MOSSE \\
        --scales 1.0 0.75 0.5 0.25 \\
        --iou-budget 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
from eovot.trackers.registry import build_tracker, available_trackers


def _run_from_config(config: dict, iou_budget: Optional[float]) -> None:
    exp_cfg = config.get("experiment", {})
    ds_cfg = config.get("dataset", {})
    tr_cfg = config.get("tracker", {})
    bm_cfg = config.get("benchmark", {})
    ra_cfg = config.get("resolution_analysis", {})

    loader_name = ds_cfg.get("loader", "SyntheticDataset")
    if loader_name == "SyntheticDataset":
        frame_size = ds_cfg.get("frame_size", [320, 240])
        bbox_size = ds_cfg.get("bbox_size", [40, 40])
        dataset = SyntheticDataset(
            num_sequences=ds_cfg.get("num_sequences", 10),
            num_frames=ds_cfg.get("num_frames", 100),
            frame_size=tuple(frame_size),
            bbox_size=tuple(bbox_size),
            motion=ds_cfg.get("motion", "linear"),
            seed=ds_cfg.get("seed", 42),
        )
    else:
        # Real dataset loaders
        from eovot.datasets.base import OTBDataset
        from eovot.datasets.got10k import GOT10kDataset
        from eovot.datasets.lasot import LaSOTDataset

        loaders = {
            "OTBDataset": OTBDataset,
            "GOT10kDataset": GOT10kDataset,
            "LaSOTDataset": LaSOTDataset,
        }
        cls = loaders.get(loader_name)
        if cls is None:
            print(f"[error] Unknown dataset loader: {loader_name}", file=sys.stderr)
            sys.exit(1)
        root = ds_cfg["root"]
        if loader_name == "OTBDataset":
            dataset = cls(root=root)
        else:
            dataset = cls(root=root, split=ds_cfg.get("split", "val"),
                          max_sequences=ds_cfg.get("max_sequences"))

    dataset_name = ds_cfg.get("name", loader_name)
    tracker = build_tracker(tr_cfg["name"], **tr_cfg.get("params", {}))
    scale_factors = ra_cfg.get("scale_factors", [1.0, 0.75, 0.5, 0.25])
    max_sequences = ra_cfg.get("max_sequences", None)

    tdp = exp_cfg.get("tdp_watts") or bm_cfg.get("tdp_watts")
    engine = BenchmarkEngine(
        verbose=bm_cfg.get("verbose", True),
        tdp_watts=tdp,
    )
    analyzer = ResolutionScaleAnalyzer(engine)
    result = analyzer.analyze(
        tracker,
        dataset,
        dataset_name=dataset_name,
        scale_factors=scale_factors,
        max_sequences=max_sequences,
    )

    print("\n" + str(result))
    print()
    print(result.to_markdown_table())

    if iou_budget is not None:
        _show_recommendation(result, iou_budget)


def _show_recommendation(result, iou_budget: float) -> None:
    print(f"\n  IoU budget: ≤{iou_budget:.2f} absolute drop from full-resolution baseline")
    try:
        scale, iou, fps = result.optimal_scale(
            min_iou=max(0.0, (result.baseline.mean_iou if result.baseline else 0.0) - iou_budget)
        )
        print(f"  → Recommended scale: {scale:.2f}  mIoU={iou:.4f}  FPS={fps:.1f}")
    except ValueError as e:
        print(f"  → {e}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze_resolution",
        description="EOVOT resolution-scale accuracy–speed tradeoff analysis.",
    )
    parser.add_argument("--config", "-c", metavar="PATH",
                        help="Path to a YAML config file.")
    parser.add_argument("--tracker", default="KCF",
                        choices=available_trackers(),
                        help="Tracker to analyse (default: KCF).")
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[1.0, 0.75, 0.5, 0.25],
                        metavar="S",
                        help="Scale factors to sweep (default: 1.0 0.75 0.5 0.25).")
    parser.add_argument("--num-sequences", type=int, default=5, metavar="N",
                        help="Synthetic sequences to generate (default: 5).")
    parser.add_argument("--iou-budget", type=float, default=None, metavar="D",
                        help=(
                            "Print the recommended scale factor that stays within "
                            "this absolute IoU drop from the full-resolution baseline."
                        ))
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sequence verbose output.")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as fh:
            config = yaml.safe_load(fh)
        _run_from_config(config, iou_budget=args.iou_budget)
        return

    # Quick mode: synthetic dataset, no config file needed.
    tracker = build_tracker(args.tracker)
    dataset = SyntheticDataset(num_sequences=args.num_sequences)
    engine = BenchmarkEngine(verbose=not args.quiet)
    analyzer = ResolutionScaleAnalyzer(engine)
    result = analyzer.analyze(
        tracker,
        dataset,
        dataset_name="Synthetic",
        scale_factors=args.scales,
    )
    print("\n" + str(result))
    print()
    print(result.to_markdown_table())

    if args.iou_budget is not None:
        _show_recommendation(result, args.iou_budget)


if __name__ == "__main__":
    main()
