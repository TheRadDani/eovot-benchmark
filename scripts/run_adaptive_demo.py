"""Adaptive Confidence-Skip benchmarking demo.

Compares three tracker configurations side-by-side on a dataset:
1. MOSSE at full rate (baseline)
2. MOSSE with fixed frame-skip k=2 (FrameSkipTracker)
3. MOSSE with adaptive PSR-driven frame-skip (AdaptiveSkipTracker)

The script outputs a leaderboard table and — when ``--output-dir`` is given —
writes JSON results to disk for further analysis.

Usage
-----
    # Quick smoke-test with the built-in synthetic dataset:
    python scripts/run_adaptive_demo.py --synthetic

    # Against a real OTB100 dataset:
    python scripts/run_adaptive_demo.py --dataset-root /data/OTB100 --max-sequences 20

    # Load thresholds from a YAML config:
    python scripts/run_adaptive_demo.py \\
        --config configs/experiments/adaptive_skip_demo.yaml --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when running directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import OTBDataset
from eovot.datasets.got10k import GOT10kDataset
from eovot.datasets.lasot import LaSOTDataset
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.frame_skip import FrameSkipTracker
from eovot.trackers.adaptive_skip import AdaptiveSkipTracker
from eovot.reporting.leaderboard import LeaderboardExporter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare full-rate, fixed-skip, and adaptive-skip trackers."
    )
    p.add_argument("--config", default=None, help="Path to a YAML config file.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use the built-in synthetic dataset.")
    p.add_argument("--synthetic-sequences", type=int, default=10)
    p.add_argument("--synthetic-frames", type=int, default=120)
    p.add_argument("--dataset-root", default=None,
                   help="Root directory of an OTB-style dataset.")
    p.add_argument("--dataset-name", default="OTB100")
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--high-threshold", type=float, default=7.0,
                   help="PSR >= this → skip next frame.")
    p.add_argument("--low-threshold", type=float, default=4.0,
                   help="PSR < this → always process next frame.")
    p.add_argument("--max-consecutive-skips", type=int, default=4)
    p.add_argument("--mode", choices=["repeat", "linear"], default="repeat")
    p.add_argument("--output-dir", default=None,
                   help="Write JSON results here (one file per tracker).")
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


def _load_config(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    skip_cfg = cfg.get("adaptive_skip", {})
    return {
        "high_threshold": skip_cfg.get("high_threshold", args.high_threshold),
        "low_threshold": skip_cfg.get("low_threshold", args.low_threshold),
        "max_consecutive_skips": skip_cfg.get("max_consecutive_skips",
                                               args.max_consecutive_skips),
        "mode": skip_cfg.get("mode", args.mode),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    params = _load_config(args)

    # --- Dataset --------------------------------------------------------
    if args.synthetic or (args.dataset_root is None):
        n_seq = args.synthetic_sequences
        n_frm = args.synthetic_frames
        if args.config:
            ds_cfg = yaml.safe_load(open(args.config)) or {}
            ds_section = ds_cfg.get("dataset", {})
            n_seq = ds_section.get("num_sequences", n_seq)
            n_frm = ds_section.get("num_frames", n_frm)
        dataset = SyntheticDataset(num_sequences=n_seq, num_frames=n_frm)
        dataset_name = "Synthetic"
    else:
        dataset = OTBDataset(
            root=args.dataset_root,
            name=args.dataset_name,
            max_sequences=args.max_sequences,
        )
        dataset_name = args.dataset_name

    # --- Trackers -------------------------------------------------------
    base_mosse = MOSSETracker()
    fixed_skip = FrameSkipTracker(MOSSETracker(), skip_rate=2, mode=params["mode"])
    adaptive = AdaptiveSkipTracker(
        MOSSETracker(),
        high_threshold=params["high_threshold"],
        low_threshold=params["low_threshold"],
        max_consecutive_skips=params["max_consecutive_skips"],
        mode=params["mode"],
    )

    trackers = [base_mosse, fixed_skip, adaptive]

    # --- Benchmark ------------------------------------------------------
    engine = BenchmarkEngine(verbose=args.verbose)
    results = []
    for tracker in trackers:
        print(f"  Benchmarking {tracker.name} ...")
        result = engine.run(tracker, dataset, dataset_name=dataset_name)
        results.append(result)

    # --- Reporting -------------------------------------------------------
    exporter = LeaderboardExporter()
    print("\n" + "=" * 72)
    print("ADAPTIVE SKIP BENCHMARK — LEADERBOARD")
    print("=" * 72)
    print(exporter.to_markdown(results))

    # Per-tracker adaptive stats
    print("\nAdaptive tracker statistics:")
    seq_results = results[2].sequence_results  # adaptive tracker
    total_frames = sum(r.profiling.frame_count for r in seq_results)
    # adaptive_frames ~= active frames; we approximate from FPS ratio
    print(f"  mean FPS (full): {results[0].mean_fps:.1f}")
    print(f"  mean FPS (fixed-skip k=2): {results[1].mean_fps:.1f}")
    print(f"  mean FPS (adaptive): {results[2].mean_fps:.1f}")
    print(f"  mIoU (full): {results[0].mean_iou:.4f}")
    print(f"  mIoU (fixed-skip k=2): {results[1].mean_iou:.4f}")
    print(f"  mIoU (adaptive): {results[2].mean_iou:.4f}")

    fps_gain = results[2].mean_fps / results[0].mean_fps
    iou_cost = results[0].mean_iou - results[2].mean_iou
    print(f"\n  FPS gain vs. full rate : {fps_gain:.2f}×")
    print(f"  IoU cost vs. full rate : {iou_cost:.4f}")

    # --- Save results ---------------------------------------------------
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        exporter.to_csv(results, str(out / "leaderboard.csv"))
        for r in results:
            path = out / f"{r.tracker_name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r.to_dict(), f, indent=2)
        print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
