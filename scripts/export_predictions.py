#!/usr/bin/env python3
"""CLI: export tracker predictions to VOT-standard text files for offline analysis.

Runs one classical tracker on a dataset and writes per-frame bounding boxes
to ``<output-dir>/<tracker>/<sequence>.txt`` in the standard VOT format
(``x,y,w,h`` per line; ``nan,nan,nan,nan`` for unavailable frames).

The saved files can be reloaded by
:class:`~eovot.experiment.prediction_io.PredictionLoader` for offline metric
recomputation or cross-tracker comparison without re-running inference.

Usage::

    # Export MOSSE predictions on the built-in synthetic dataset
    python scripts/export_predictions.py --tracker mosse --num-sequences 5

    # Export KCF predictions and enable energy profiling
    python scripts/export_predictions.py \
        --tracker kcf \
        --num-sequences 10 \
        --output-dir results/predictions/ \
        --tdp-watts 6.0

    # Reload and recompute metrics without re-running
    python - <<'EOF'
    from eovot.experiment.prediction_io import PredictionLoader
    loader = PredictionLoader("results/predictions/MOSSE/")
    preds = loader.load_sequence("synthetic_seq_0")
    print(preds.shape)   # (N, 4)
    EOF
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_TRACKER_MAP = {
    "mosse":      ("eovot.trackers.mosse",       "MOSSETracker"),
    "kcf":        ("eovot.trackers.kcf",         "KCFTracker"),
    "csrt":       ("eovot.trackers.csrt",        "CSRTTracker"),
    "camshift":   ("eovot.trackers.camshift",    "CamShiftTracker"),
    "medianflow": ("eovot.trackers.median_flow", "MedianFlowTracker"),
    "mil":        ("eovot.trackers.mil",         "MILTracker"),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export EOVOT tracker predictions to VOT-format text files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tracker",
        choices=sorted(_TRACKER_MAP),
        default="mosse",
        help="Tracker to evaluate.",
    )
    p.add_argument(
        "--dataset",
        choices=["synthetic"],
        default="synthetic",
        help="Dataset (only 'synthetic' ships without external data).",
    )
    p.add_argument(
        "--output-dir",
        default="results/predictions/",
        help="Root directory for saved prediction files.",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of sequences to evaluate (default: 5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic dataset.",
    )
    p.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        help="CPU TDP in Watts for energy profiling (e.g. 6.0 for Raspberry Pi 4).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence progress output.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # --- dataset -----------------------------------------------------------
    if args.dataset == "synthetic":
        from eovot.datasets.synthetic import SyntheticDataset
        dataset = SyntheticDataset(
            num_sequences=args.num_sequences, seed=args.seed
        )
        dataset_name = f"Synthetic(seed={args.seed})"
    else:
        print(f"[error] Dataset '{args.dataset}' not yet supported.", file=sys.stderr)
        return 1

    # --- tracker -----------------------------------------------------------
    module_path, class_name = _TRACKER_MAP[args.tracker]
    try:
        mod = importlib.import_module(module_path)
        TrackerClass = getattr(mod, class_name)
        tracker = TrackerClass()
    except ImportError as exc:
        print(f"[error] Failed to import tracker '{args.tracker}': {exc}", file=sys.stderr)
        return 1

    # --- benchmark ---------------------------------------------------------
    from eovot.benchmark.engine import BenchmarkEngine
    engine = BenchmarkEngine(
        verbose=not args.quiet,
        tdp_watts=args.tdp_watts,
    )
    result = engine.run(
        tracker, dataset,
        dataset_name=dataset_name,
        max_sequences=args.num_sequences,
    )

    # --- export predictions ------------------------------------------------
    from eovot.experiment.prediction_io import PredictionExporter
    exporter = PredictionExporter(args.output_dir)
    paths = exporter.save(result)

    tracker_dir = Path(args.output_dir) / tracker.name
    print(f"\n[export] {len(paths)} sequences saved to: {tracker_dir}")
    for seq_name, path in sorted(paths.items()):
        n_frames = sum(1 for _ in open(path))
        print(f"  {seq_name:<35s}  {n_frames:>4d} frames  →  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
