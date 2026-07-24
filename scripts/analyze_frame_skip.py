"""CLI for frame-skip accuracy–speed tradeoff analysis.

Sweeps a tracker across a range of skip rates on a synthetic (or real)
dataset, reporting the IoU degradation and FPS gain at each rate.

Usage
-----
    # Quick synthetic demo — no real dataset needed
    python scripts/analyze_frame_skip.py --tracker MOSSE --synthetic

    # With a YAML config file
    python scripts/analyze_frame_skip.py --config configs/experiments/frame_skip_analysis.yaml

    # Explicit arguments against a real OTB dataset
    python scripts/analyze_frame_skip.py \\
        --tracker KCF \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --skip-rates 1 2 3 4 6 \\
        --max-sequences 10

    # Change the propagation mode and set a minimum IoU budget
    python scripts/analyze_frame_skip.py \\
        --tracker MOSSE --synthetic \\
        --mode linear \\
        --min-iou 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from eovot.benchmark.engine import BenchmarkEngine
from eovot.trackers.registry import TRACKER_REGISTRY
from eovot.datasets.synthetic import SyntheticDataset
from eovot.datasets.base import OTBDataset
from eovot.analysis.skip_analysis import FrameSkipAnalyzer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_frame_skip",
        description="EOVOT frame-skip tradeoff analysis — sweep skip rates and report accuracy vs. FPS.",
    )
    parser.add_argument("--config", "-c", metavar="PATH",
                        help="YAML config file (overrides all other args).")
    parser.add_argument("--tracker", default="MOSSE",
                        choices=list(TRACKER_REGISTRY),
                        help="Tracker to evaluate (default: MOSSE).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic dataset (no real data required).")
    parser.add_argument("--num-sequences", type=int, default=5, metavar="N",
                        help="Number of synthetic sequences (default: 5).")
    parser.add_argument("--dataset-root", metavar="DIR",
                        help="OTB dataset root directory.")
    parser.add_argument("--dataset-name", default="dataset",
                        help="Human-readable dataset label.")
    parser.add_argument("--skip-rates", nargs="+", type=int, default=[1, 2, 3, 4],
                        metavar="K",
                        help="Skip rates to sweep (default: 1 2 3 4).")
    parser.add_argument("--mode", choices=["repeat", "linear"], default="repeat",
                        help="Frame propagation mode for skipped frames (default: repeat).")
    parser.add_argument("--max-sequences", type=int, default=None, metavar="N",
                        help="Limit evaluation to the first N sequences.")
    parser.add_argument("--min-iou", type=float, default=None, metavar="T",
                        help="Report the optimal skip rate at this IoU threshold.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-sequence progress output.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)

    # Resolve tracker
    tracker_name = cfg.get("tracker", {}).get("name", args.tracker)
    tracker_cls = TRACKER_REGISTRY.get(tracker_name)
    if tracker_cls is None:
        print(f"[ERROR] Unknown tracker '{tracker_name}'.", file=sys.stderr)
        sys.exit(1)
    tracker = tracker_cls(**cfg.get("tracker", {}).get("params", {}))

    # Resolve dataset
    ds_cfg = cfg.get("dataset", {})
    use_synthetic = ds_cfg.get("synthetic", args.synthetic)
    if use_synthetic:
        n_seq = ds_cfg.get("num_sequences", args.num_sequences)
        dataset = SyntheticDataset(num_sequences=n_seq, num_frames=100)
        dataset_name = ds_cfg.get("name", "Synthetic")
    elif args.dataset_root or ds_cfg.get("root"):
        root = ds_cfg.get("root") or args.dataset_root
        dataset = OTBDataset(root)
        dataset_name = ds_cfg.get("name", args.dataset_name)
    else:
        print("[ERROR] Provide --synthetic or --dataset-root.", file=sys.stderr)
        sys.exit(1)

    # Resolve analysis params
    skip_rates = cfg.get("skip_rates", args.skip_rates)
    mode = cfg.get("mode", args.mode)
    max_seq = cfg.get("max_sequences", args.max_sequences)
    min_iou = cfg.get("min_iou", args.min_iou)
    verbose = not (cfg.get("quiet", args.quiet))

    engine = BenchmarkEngine(verbose=verbose)
    analyzer = FrameSkipAnalyzer(engine)

    print(f"\nFrame-skip analysis: {tracker_name} on {dataset_name}")
    print(f"Skip rates: {skip_rates}  mode: {mode}\n")

    result = analyzer.analyze(
        tracker,
        dataset,
        dataset_name=dataset_name,
        skip_rates=skip_rates,
        mode=mode,
        max_sequences=max_seq,
    )

    print("\n" + "=" * 60)
    print(" FRAME-SKIP TRADEOFF REPORT")
    print("=" * 60)
    print(result.to_markdown_table())
    print()

    if min_iou is not None:
        try:
            rate, iou, fps = result.optimal_rate(min_iou=min_iou)
            print(
                f"Optimal rate at mIoU >= {min_iou}:  "
                f"skip_rate={rate}  mIoU={iou:.4f}  FPS={fps:.1f}"
            )
        except ValueError as e:
            print(f"[WARNING] {e}")

    print()


if __name__ == "__main__":
    main()
