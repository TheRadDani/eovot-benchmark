#!/usr/bin/env python3
"""Export or import tracker predictions in standard benchmark formats.

This script allows you to:

1. **Export** — run a tracker on a dataset and save raw predictions (bounding
   boxes) to disk in OTB, GOT-10k, VOT, or EOVOT-JSON format.
2. **Import** — load saved predictions (from any system) and compute the
   full EOVOT metrics suite on them without re-running the tracker.

Usage — Export::

    # Save MOSSE predictions on synthetic data in OTB format
    python scripts/export_predictions.py export \\
        --tracker MOSSE \\
        --dataset synthetic \\
        --num-sequences 5 \\
        --format otb \\
        --output-dir predictions/

    # Save KCF predictions in GOT-10k format
    python scripts/export_predictions.py export \\
        --tracker KCF \\
        --dataset synthetic \\
        --format got10k \\
        --output-dir predictions/

Usage — Import and evaluate::

    # Load saved predictions and compute metrics
    python scripts/export_predictions.py evaluate \\
        --predictions-dir predictions/ \\
        --tracker MOSSE \\
        --dataset synthetic \\
        --num-sequences 5 \\
        --format otb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Export sub-command
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker
    from eovot.utils.prediction_io import PredictionFormat, PredictionWriter

    if args.dataset == "synthetic":
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            motion=args.motion,
            seed=args.seed,
        )
        dataset_name = f"Synthetic-{args.motion}"
    else:
        print(f"ERROR: dataset '{args.dataset}' not supported via CLI yet.", file=sys.stderr)
        return 2

    tracker = build_tracker(args.tracker)
    engine = BenchmarkEngine(verbose=True)
    result = engine.run(tracker, dataset, dataset_name=dataset_name)

    fmt = PredictionFormat(args.format)
    writer = PredictionWriter(output_dir=args.output_dir, fmt=fmt)
    paths = writer.write_benchmark_result(result)

    print(f"\nPredictions saved ({fmt.value} format) → {args.output_dir}")
    for seq_name, path in sorted(paths.items()):
        print(f"  {seq_name}: {path}")

    return 0


# ---------------------------------------------------------------------------
# Evaluate sub-command
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    import numpy as np
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.metrics.accuracy import MetricsEngine
    from eovot.utils.prediction_io import PredictionFormat, PredictionReader

    if args.dataset == "synthetic":
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences,
            num_frames=args.num_frames,
            motion=args.motion,
            seed=args.seed,
        )
    else:
        print(f"ERROR: dataset '{args.dataset}' not supported via CLI yet.", file=sys.stderr)
        return 2

    fmt = PredictionFormat(args.format)
    reader = PredictionReader(input_dir=args.predictions_dir, fmt=fmt)

    try:
        loaded = reader.read_tracker(args.tracker)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not loaded:
        print(f"No predictions found for tracker '{args.tracker}'.", file=sys.stderr)
        return 1

    engine = MetricsEngine()
    print(f"\nEvaluating loaded predictions for '{args.tracker}':")
    print("-" * 60)

    all_ious = []
    for seq in dataset:
        preds = loaded.get(seq.name)
        if preds is None:
            print(f"  {seq.name:<30s} — no predictions found, skipping")
            continue
        n = min(len(preds), len(seq.ground_truth))
        result = engine.compute_all(preds[:n], seq.ground_truth[:n])
        ious = engine.batch_iou(preds[:n], seq.ground_truth[:n])
        all_ious.append(ious)
        print(
            f"  {seq.name:<30s}  "
            f"mIoU={result.mean_iou:.4f}  "
            f"AUC={result.success_auc:.4f}  "
            f"prec={result.precision_auc:.4f}"
        )

    if all_ious:
        combined = np.concatenate(all_ious)
        print("-" * 60)
        print(f"  Overall mIoU = {combined.mean():.4f}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export / import tracker predictions in standard formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- export ---
    exp = sub.add_parser("export", help="Run tracker and save predictions to disk.")
    exp.add_argument("--tracker", required=True, help="Tracker name (e.g. MOSSE, KCF)")
    exp.add_argument(
        "--format", choices=["otb", "got10k", "vot", "json"], default="otb",
        help="Output format (default: otb)",
    )
    exp.add_argument("--output-dir", default="predictions/", help="Root output directory")
    _add_dataset_args(exp)

    # --- evaluate ---
    ev = sub.add_parser("evaluate", help="Load saved predictions and compute metrics.")
    ev.add_argument("--tracker", required=True, help="Tracker name to load")
    ev.add_argument("--predictions-dir", required=True, help="Directory with saved predictions")
    ev.add_argument(
        "--format", choices=["otb", "got10k", "vot", "json"], default="otb",
        help="Format of the saved predictions (default: otb)",
    )
    _add_dataset_args(ev)

    return p


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset", choices=["synthetic"], default="synthetic",
        help="Dataset to use (default: synthetic)",
    )
    p.add_argument("--num-sequences", type=int, default=5)
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument(
        "--motion", choices=["linear", "circular", "random"], default="linear",
    )
    p.add_argument("--seed", type=int, default=42)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
