"""Demonstrate Kalman filter smoothing on synthetic sequences.

Compares raw MOSSE / KCF trackers against their Kalman-smoothed counterparts,
showing IoU accuracy, FPS, and temporal smoothness metrics.

Usage::

    python scripts/run_kalman_demo.py
    python scripts/run_kalman_demo.py --num-sequences 5 --num-frames 300
    python scripts/run_kalman_demo.py --motion circular --output-dir results/kalman
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.metrics.temporal import TemporalConsistencyAnalyzer
from eovot.trackers.kalman_smoother import KalmanSmootherTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kalman smoother comparison demo",
    )
    p.add_argument("--num-sequences", type=int, default=5)
    p.add_argument("--num-frames", type=int, default=150)
    p.add_argument(
        "--motion",
        choices=["linear", "circular", "random"],
        default="linear",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--process-noise", type=float, default=1.0)
    p.add_argument("--measurement-noise", type=float, default=4.0)
    p.add_argument("--gate-threshold", type=float, default=16.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ds = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
    )
    engine = BenchmarkEngine(verbose=True)
    analyzer = TemporalConsistencyAnalyzer()

    trackers = [
        MOSSETracker(),
        KalmanSmootherTracker(
            MOSSETracker(),
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            gate_threshold=args.gate_threshold,
            name="Kalman(MOSSE)",
        ),
        KCFTracker(),
        KalmanSmootherTracker(
            KCFTracker(),
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
            gate_threshold=args.gate_threshold,
            name="Kalman(KCF)",
        ),
    ]

    results = []
    for tracker in trackers:
        result = engine.run(
            tracker,
            ds,
            dataset_name=f"Synthetic-{args.motion.capitalize()}",
        )
        results.append(result)

    # Temporal smoothness comparison
    print("\n" + "=" * 72)
    print(f"{'Tracker':<20} {'mIoU':>8} {'FPS':>7} {'Smoothness':>12} {'Jitter':>10}")
    print("=" * 72)

    for result in results:
        preds = [
            sr.predictions
            for sr in result.sequence_results
            if sr.predictions is not None and len(sr.predictions) > 2
        ]
        if preds:
            import numpy as np
            scores = []
            jitters = []
            for p in preds:
                tc = analyzer.analyze(p)
                scores.append(tc.smoothness_score)
                jitters.append(tc.position_jitter)
            smoothness = float(np.mean(scores))
            jitter = float(np.mean(jitters))
        else:
            smoothness = jitter = float("nan")

        print(
            f"{result.tracker_name:<20} "
            f"{result.mean_iou:>8.4f} "
            f"{result.mean_fps:>7.1f} "
            f"{smoothness:>12.4f} "
            f"{jitter:>10.5f}"
        )

    print("=" * 72)

    if args.output_dir:
        import json
        from pathlib import Path as P
        out = P(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for result in results:
            fname = out / f"{result.tracker_name.replace('(', '_').replace(')', '')}.json"
            result.save(fname)
            print(f"Saved {fname}")


if __name__ == "__main__":
    main()
