"""CLI runner for the VOT re-initialization protocol evaluation.

Evaluates one or more trackers using the standard VOT challenge protocol:
run until failure, skip N frames, re-initialize at ground truth.

Usage
-----
    # Quick smoke test with the synthetic dataset (no download needed):
    python scripts/run_reinit_eval.py \\
        --tracker MOSSE --synthetic --synthetic-sequences 5

    # All classical trackers on the synthetic dataset:
    python scripts/run_reinit_eval.py \\
        --trackers MOSSE KCF CSRT --synthetic --synthetic-sequences 10

    # With a YAML config:
    python scripts/run_reinit_eval.py \\
        --config configs/experiments/reinit_eval.yaml

    # With a real OTB dataset:
    python scripts/run_reinit_eval.py \\
        --tracker MOSSE --dataset-root /data/OTB100 --dataset-name OTB100

Output
------
    results/reinit_eval/<tracker>_<dataset>.json   — per-sequence breakdown
    results/reinit_eval/comparison.md              — Markdown comparison table
                                                     (when multiple trackers)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from eovot.benchmark.reinit_protocol import ReinitEvaluator, compare_reinit_results
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker


DATASET_REGISTRY = {
    "OTBDataset": OTBDataset,
    "GOT10kDataset": GOT10kDataset,
    "LaSOTDataset": LaSOTDataset,
    "SyntheticDataset": SyntheticDataset,
}


def _build_dataset(cfg: dict):
    loader_name = cfg.get("loader", "SyntheticDataset")
    loader_cls = DATASET_REGISTRY.get(loader_name)
    if loader_cls is None:
        raise ValueError(f"Unknown dataset loader: {loader_name}")

    if loader_name == "SyntheticDataset":
        frame_size = cfg.get("frame_size", [320, 240])
        return loader_cls(
            num_sequences=cfg.get("num_sequences", 10),
            num_frames=cfg.get("num_frames", 100),
            frame_size=tuple(frame_size) if not isinstance(frame_size, tuple) else frame_size,
            motion=cfg.get("motion", "linear"),
        )
    elif loader_name in ("GOT10kDataset", "LaSOTDataset"):
        return loader_cls(cfg["root"], split=cfg.get("split", "val"),
                          max_sequences=cfg.get("max_sequences"))
    else:
        return loader_cls(cfg["root"])


def run_from_config(cfg: dict) -> None:
    reinit_cfg = cfg.get("reinit", {})
    evaluator = ReinitEvaluator(
        failure_threshold=reinit_cfg.get("failure_threshold", 0.1),
        skip_frames=reinit_cfg.get("skip_frames", 5),
        verbose=cfg.get("output", {}).get("verbose", True),
    )

    dataset = _build_dataset(cfg["dataset"])
    dataset_name = cfg["dataset"].get("name", "dataset")
    output_dir = cfg.get("output", {}).get("dir", "results/reinit_eval/")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for tr_cfg in cfg.get("trackers", []):
        tracker = build_tracker(tr_cfg["name"], **tr_cfg.get("params", {}))
        result = evaluator.evaluate_dataset(tracker, dataset, dataset_name)
        all_results[tracker.name] = result

        summary = result.summary()
        out_path = os.path.join(output_dir, f"{tracker.name}_{dataset_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_path}")

    if len(all_results) > 1:
        table = compare_reinit_results(all_results)
        table_path = os.path.join(output_dir, "comparison.md")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("# Re-initialization Protocol Comparison\n\n")
            f.write(f"Dataset: {dataset_name}  |  ")
            f.write(f"Threshold: {evaluator.failure_threshold}  |  ")
            f.write(f"Skip: {evaluator.skip_frames}\n\n")
            f.write(table + "\n")
        print(f"\nComparison table saved: {table_path}")
        print("\n" + table)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_reinit_eval",
        description="EOVOT — VOT Re-initialization Protocol Evaluator",
    )
    parser.add_argument("--config", "-c", metavar="PATH",
                        help="YAML experiment config.")
    parser.add_argument("--trackers", nargs="+",
                        choices=list(TRACKER_REGISTRY), metavar="NAME",
                        help="One or more trackers to evaluate.")
    parser.add_argument("--tracker",
                        choices=list(TRACKER_REGISTRY), default="MOSSE",
                        help="Single tracker (alternative to --trackers).")
    parser.add_argument("--dataset-root", metavar="DIR")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--failure-threshold", type=float, default=0.1, metavar="T",
                        help="IoU threshold for failure detection (default: 0.1).")
    parser.add_argument("--skip-frames", type=int, default=5, metavar="N",
                        help="Frames to skip after failure (default: 5).")
    parser.add_argument("--max-sequences", type=int, default=None, metavar="N")
    parser.add_argument("--output-dir", default="results/reinit_eval/")
    parser.add_argument("--quiet", action="store_true")

    syn = parser.add_argument_group("synthetic dataset")
    syn.add_argument("--synthetic", action="store_true")
    syn.add_argument("--synthetic-sequences", type=int, default=5)
    syn.add_argument("--synthetic-frames", type=int, default=100)
    syn.add_argument("--synthetic-motion", default="linear",
                     choices=["linear", "circular", "random"])
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        run_from_config(cfg)
        return

    tracker_names = args.trackers if args.trackers else [args.tracker]

    if args.synthetic:
        ds_cfg = {
            "loader": "SyntheticDataset",
            "name": "Synthetic",
            "num_sequences": args.synthetic_sequences,
            "num_frames": args.synthetic_frames,
            "motion": args.synthetic_motion,
        }
    elif args.dataset_root:
        ds_cfg = {
            "loader": "OTBDataset",
            "root": args.dataset_root,
            "name": args.dataset_name,
            "max_sequences": args.max_sequences,
        }
    else:
        parser.print_help()
        sys.exit(0)

    cfg = {
        "reinit": {
            "failure_threshold": args.failure_threshold,
            "skip_frames": args.skip_frames,
        },
        "trackers": [{"name": n, "params": {}} for n in tracker_names],
        "dataset": ds_cfg,
        "output": {
            "dir": args.output_dir,
            "verbose": not args.quiet,
        },
    }
    run_from_config(cfg)


if __name__ == "__main__":
    main()
