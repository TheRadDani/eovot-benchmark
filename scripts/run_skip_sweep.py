#!/usr/bin/env python3
"""Skip-rate sweep: ablation study of the accuracy-throughput trade-off.

Wraps any registered tracker in :class:`~eovot.trackers.adaptive.FrameSkipTracker`
for a range of skip rates, evaluates each on a dataset, and outputs:

- A JSON report with per-rate metrics (mIoU, FPS, skip ratio, throughput ×)
- A Markdown table ready for a paper or README
- An optional PNG Pareto curve of IoU vs. FPS (``--plot``)

Usage::

    # Quick run on the built-in synthetic dataset
    python scripts/run_skip_sweep.py \\
        --tracker MOSSE \\
        --skip-rates 0 1 2 3 5 9 \\
        --extrapolation linear

    # Save JSON + plot
    python scripts/run_skip_sweep.py \\
        --tracker KCF \\
        --skip-rates 0 1 2 4 \\
        --output results/skip_sweep_kcf.json \\
        --plot results/skip_sweep_kcf.png

    # Larger run for a paper figure
    python scripts/run_skip_sweep.py \\
        --tracker MOSSE \\
        --skip-rates 0 1 2 3 5 9 \\
        --num-sequences 20 \\
        --num-frames 200 \\
        --extrapolation linear \\
        --output results/mosse_skip_sweep.json \\
        --plot results/mosse_skip_pareto.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.adaptive import FrameSkipTracker
from eovot.trackers.registry import build_tracker


def _build_dataset(name: str, num_sequences: int, num_frames: int):
    """Construct the requested dataset.

    Only ``synthetic`` is supported without external data files.  Future
    work can extend this to OTB / GOT-10k / LaSOT once a download is
    available.
    """
    if name.lower() in ("synthetic", "syn"):
        return SyntheticDataset(
            num_sequences=num_sequences,
            num_frames=num_frames,
            motion="linear",
            seed=42,
        )
    raise ValueError(
        f"Dataset '{name}' is not supported by this script. "
        "Pass --dataset synthetic to use the built-in dataset."
    )


def run_sweep(
    tracker_name: str,
    skip_rates: List[int],
    extrapolation: str,
    dataset_name: str,
    num_sequences: int,
    num_frames: int,
    verbose: bool,
) -> List[dict]:
    """Run the skip-rate sweep and return one result dict per skip rate."""
    engine = BenchmarkEngine(verbose=verbose)
    results = []

    for skip in sorted(set(skip_rates)):
        tracker = FrameSkipTracker(
            build_tracker(tracker_name),
            skip_rate=skip,
            extrapolation=extrapolation,
        )
        dataset = _build_dataset(dataset_name, num_sequences, num_frames)
        result = engine.run(tracker, dataset, dataset_name=dataset_name)
        s = result.summary()

        entry: dict = {
            "skip_rate": skip,
            "skip_ratio": round(skip / (skip + 1), 4),
            "throughput_multiplier": float(skip + 1),
            "mean_iou": s["mean_iou"],
            "mean_fps": s["mean_fps"],
            "peak_memory_mb": s["peak_memory_mb"],
        }
        if "success_auc" in s:
            entry["success_auc"] = s["success_auc"]
        if "precision_auc" in s:
            entry["precision_auc"] = s["precision_auc"]
        results.append(entry)

    return results


def format_markdown_table(tracker_name: str, sweep: List[dict]) -> str:
    """Render sweep results as a Markdown table."""
    has_auc = "success_auc" in sweep[0] if sweep else False

    header = (
        "| Skip | Skip Ratio | FPS× | mIoU   | FPS    | Mem (MB) |"
        + (" Success AUC |" if has_auc else "")
    )
    sep = (
        "|-----:|-----------:|-----:|-------:|-------:|---------:|"
        + ("-----------:|" if has_auc else "")
    )
    lines = [f"## Skip-Rate Sweep — {tracker_name}\n", header, sep]

    for r in sweep:
        row = (
            f"| {r['skip_rate']:4} "
            f"| {r['skip_ratio']:10.4f} "
            f"| {r['throughput_multiplier']:4.1f}× "
            f"| {r['mean_iou']:6.4f} "
            f"| {r['mean_fps']:6.1f} "
            f"| {r['peak_memory_mb']:8.1f} |"
        )
        if has_auc:
            row += f" {r.get('success_auc', 0.0):10.4f} |"
        lines.append(row)

    return "\n".join(lines) + "\n"


def plot_pareto(
    sweep: List[dict], output_path: str, tracker_name: str
) -> None:
    """Save an IoU-vs-FPS Pareto curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skipping plot.", file=sys.stderr)
        return

    fps_vals = [r["mean_fps"] for r in sweep]
    iou_vals = [r["mean_iou"] for r in sweep]
    labels = [f"skip={r['skip_rate']}" for r in sweep]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fps_vals, iou_vals, "o-", linewidth=2, markersize=8, label=tracker_name)
    for fps, iou, lbl in zip(fps_vals, iou_vals, labels):
        ax.annotate(
            lbl,
            (fps, iou),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Throughput (FPS)", fontsize=12)
    ax.set_ylabel("Mean IoU", fontsize=12)
    ax.set_title(
        f"Skip-Rate Ablation — {tracker_name}: Accuracy vs. Throughput",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skip-rate ablation: measure the accuracy-throughput trade-off.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tracker",
        default="MOSSE",
        metavar="NAME",
        help="Tracker name from the registry (default: MOSSE).",
    )
    parser.add_argument(
        "--dataset",
        default="synthetic",
        metavar="NAME",
        help="Dataset to evaluate on (default: synthetic).",
    )
    parser.add_argument(
        "--skip-rates",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 9],
        metavar="N",
        help="Skip rates to sweep (default: 0 1 2 3 5 9).",
    )
    parser.add_argument(
        "--extrapolation",
        choices=["last", "linear"],
        default="linear",
        help="Box extrapolation for skipped frames (default: linear).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        metavar="N",
        help="Number of synthetic sequences to generate (default: 5).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        metavar="N",
        help="Frames per synthetic sequence (default: 100).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write JSON results to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--plot",
        default=None,
        metavar="PATH",
        help="Save the IoU-vs-FPS Pareto curve PNG to this path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sequence benchmark output.",
    )
    args = parser.parse_args()

    sweep = run_sweep(
        tracker_name=args.tracker,
        skip_rates=args.skip_rates,
        extrapolation=args.extrapolation,
        dataset_name=args.dataset,
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        verbose=not args.quiet,
    )

    output = {
        "tracker": args.tracker,
        "dataset": args.dataset,
        "extrapolation": args.extrapolation,
        "sweep": sweep,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"[saved] JSON → {args.output}")
    else:
        print(json.dumps(output, indent=2))

    print()
    print(format_markdown_table(args.tracker, sweep))

    if args.plot:
        plot_pareto(sweep, args.plot, args.tracker)


if __name__ == "__main__":
    main()
