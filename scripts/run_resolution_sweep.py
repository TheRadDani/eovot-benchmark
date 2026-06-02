"""Resolution-scaling sweep benchmark for EOVOT.

Evaluates one or more base trackers at a range of input-frame scale factors
on a synthetic dataset and prints a Markdown table comparing accuracy vs
throughput at each resolution level.

This sweep characterises the spatial accuracy-vs-latency trade-off, which
complements the temporal frame-skip sweep (``run_frame_skip_sweep.py``).
Together they define the 2D operating envelope for a tracker on a given
edge device.

Usage::

    python scripts/run_resolution_sweep.py [options]

Options::

    --trackers      Comma-separated base tracker names.  Default: MOSSE,KCF
    --scales        Comma-separated scale factors (0 < s ≤ 1). Default: 1.0,0.75,0.5,0.25
    --num-sequences Number of synthetic sequences.     Default: 10
    --num-frames    Frames per sequence.               Default: 200
    --motion        Synthetic motion type.             Default: linear
    --output        Path to write Markdown table.      Default: stdout only
    --seed          RNG seed for reproducibility.      Default: 42

Example::

    python scripts/run_resolution_sweep.py --trackers MOSSE,KCF --scales 1.0,0.75,0.5,0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.csrt import CSRTTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.median_flow import MedianFlowTracker
from eovot.trackers.mil import MILTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.resolution_scaler import ResolutionScalerTracker

_BASE_REGISTRY = {
    "MOSSE": MOSSETracker,
    "KCF": KCFTracker,
    "MIL": MILTracker,
    "CSRT": CSRTTracker,
    "MedianFlow": MedianFlowTracker,
}


def _make_tracker(base_name: str, scale: float):
    base_cls = _BASE_REGISTRY[base_name]
    base = base_cls()
    if scale == 1.0:
        return base  # no wrapping for full-resolution baseline
    return ResolutionScalerTracker(base, scale_factor=scale)


def _run_sweep(
    base_trackers: list[str],
    scales: list[float],
    num_sequences: int,
    num_frames: int,
    motion: str,
    seed: int,
) -> list[dict]:
    """Run the full sweep and return a list of result rows."""
    dataset = SyntheticDataset(
        num_sequences=num_sequences,
        num_frames=num_frames,
        motion=motion,
        seed=seed,
    )
    engine = BenchmarkEngine(verbose=False)
    rows = []

    for base_name in base_trackers:
        for scale in scales:
            tracker = _make_tracker(base_name, scale)
            result = engine.run(tracker, dataset, dataset_name="Synthetic")

            pixel_frac = scale ** 2  # fraction of pixels vs full resolution
            rows.append(
                {
                    "base": base_name,
                    "scale": scale,
                    "pixel_pct": round(pixel_frac * 100, 1),
                    "mean_iou": round(result.mean_iou, 4),
                    "success_auc": round(result.mean_success_auc or 0.0, 4),
                    "mean_fps": round(result.mean_fps, 1),
                    "peak_mem_mb": round(result.peak_memory_mb, 1),
                }
            )
            print(
                f"  {base_name:12s}  scale={scale:.2f}  pixels={pixel_frac*100:5.1f}%  "
                f"mIoU={rows[-1]['mean_iou']:.4f}  "
                f"AUC={rows[-1]['success_auc']:.4f}  "
                f"FPS={rows[-1]['mean_fps']:.1f}"
            )

    return rows


def _rows_to_markdown(rows: list[dict]) -> str:
    lines = [
        "# EOVOT Resolution Sweep Results",
        "",
        "| Tracker | Scale | Pixels % | mIoU | Success AUC | FPS | Mem (MB) |",
        "|---------|------:|---------:|-----:|------------:|----:|---------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['base']} "
            f"| {r['scale']:.2f} "
            f"| {r['pixel_pct']:.1f} "
            f"| {r['mean_iou']:.4f} "
            f"| {r['success_auc']:.4f} "
            f"| {r['mean_fps']:.1f} "
            f"| {r['peak_mem_mb']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolution-scaling accuracy vs throughput sweep."
    )
    parser.add_argument(
        "--trackers",
        default="MOSSE,KCF",
        help="Comma-separated base tracker names (default: MOSSE,KCF)",
    )
    parser.add_argument(
        "--scales",
        default="1.0,0.75,0.5,0.25",
        help="Comma-separated scale factors, e.g. 1.0,0.5,0.25 (default: 1.0,0.75,0.5,0.25)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        help="Number of synthetic sequences (default: 10)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=200,
        help="Frames per sequence (default: 200)",
    )
    parser.add_argument(
        "--motion",
        default="linear",
        choices=["linear", "circular", "random"],
        help="Synthetic target motion pattern (default: linear)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write Markdown table to this file (default: stdout only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic dataset (default: 42)",
    )
    args = parser.parse_args()

    base_trackers = [t.strip() for t in args.trackers.split(",")]
    for bt in base_trackers:
        if bt not in _BASE_REGISTRY:
            parser.error(
                f"Unknown tracker '{bt}'. Available: {list(_BASE_REGISTRY)}"
            )

    scales = [float(s.strip()) for s in args.scales.split(",")]
    for sf in scales:
        if not (0.0 < sf <= 1.0):
            parser.error(
                f"Invalid scale factor '{sf}'. Must be in (0.0, 1.0]."
            )

    print(
        f"\nResolution sweep  "
        f"trackers={base_trackers}  "
        f"scales={scales}  "
        f"seq={args.num_sequences}×{args.num_frames}fr  "
        f"motion={args.motion}\n"
    )

    rows = _run_sweep(
        base_trackers=base_trackers,
        scales=scales,
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        motion=args.motion,
        seed=args.seed,
    )

    md = _rows_to_markdown(rows)
    print("\n" + md)

    if args.output:
        Path(args.output).write_text(md + "\n", encoding="utf-8")
        print(f"\nTable written to {args.output}")


if __name__ == "__main__":
    main()
