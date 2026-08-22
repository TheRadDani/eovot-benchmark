"""CLI script: benchmark a tracker on a video file or live webcam feed.

Usage examples
--------------
Benchmark MOSSE on a video file with a manually specified initial bounding box::

    python scripts/benchmark_stream.py \\
        --source path/to/video.mp4 \\
        --bbox 120 45 80 60 \\
        --tracker mosse

Benchmark KCF on a webcam (device 0) for 300 frames::

    python scripts/benchmark_stream.py \\
        --webcam 0 \\
        --bbox 160 120 80 80 \\
        --tracker kcf \\
        --max-frames 300

Benchmark with ground-truth for accuracy metrics and save JSON::

    python scripts/benchmark_stream.py \\
        --source car_track.mp4 \\
        --bbox 50 30 100 70 \\
        --gt car_track_gt.txt \\
        --tracker mosse \\
        --save results/stream_result.json

Enable CPU energy profiling::

    python scripts/benchmark_stream.py \\
        --source video.mp4 \\
        --bbox 0 0 100 80 \\
        --tracker kcf \\
        --tdp 15.0

Enable GPU profiling (requires NVIDIA GPU + nvidia-smi)::

    python scripts/benchmark_stream.py \\
        --source video.mp4 \\
        --bbox 0 0 100 80 \\
        --tracker kcf \\
        --gpu-profiling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package root is importable when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.stream import VideoStreamDataset
from eovot.trackers.registry import available_trackers, build_tracker


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark a tracker on a video file or webcam feed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        metavar="PATH",
        help="Path to a video file (MP4, AVI, MOV, …).",
    )
    source_group.add_argument(
        "--webcam",
        type=int,
        metavar="INDEX",
        help="Webcam device index (e.g. 0 for the default camera).",
    )

    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Initial bounding box on the first frame: x y w h.",
    )
    p.add_argument(
        "--tracker",
        default="mosse",
        metavar="NAME",
        help=(
            f"Tracker name.  Available: {', '.join(available_trackers())}.  "
            "Default: mosse."
        ),
    )
    p.add_argument(
        "--gt",
        metavar="PATH",
        default=None,
        help=(
            "Optional ground-truth file for accuracy metrics.  "
            "One row per frame: x y w h (comma or whitespace separated)."
        ),
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of frames to process.  Useful for webcam feeds.",
    )
    p.add_argument(
        "--tdp",
        type=float,
        default=None,
        metavar="WATTS",
        help=(
            "Enable CPU energy estimation using the given TDP in Watts.  "
            "Example values: RPi 4 → 6.0, laptop → 15.0, desktop → 65.0."
        ),
    )
    p.add_argument(
        "--gpu-profiling",
        action="store_true",
        default=False,
        help="Enable GPU profiling via nvidia-smi (requires NVIDIA GPU).",
    )
    p.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        metavar="INDEX",
        help="CUDA device index for GPU profiling.  Default: 0.",
    )
    p.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save the benchmark result to a JSON file at this path.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-sequence progress output.",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Build dataset
    # ------------------------------------------------------------------ #
    init_bbox = tuple(args.bbox)  # type: ignore[arg-type]

    if args.source is not None:
        dataset = VideoStreamDataset.from_video(
            path=args.source,
            init_bbox=init_bbox,
            gt_path=args.gt,
            max_frames=args.max_frames,
        )
        dataset_name = Path(args.source).name
    else:
        if args.gt is not None:
            parser.error("--gt is not supported with --webcam (no fixed frame count).")
        dataset = VideoStreamDataset.from_webcam(
            device_index=args.webcam,
            init_bbox=init_bbox,
            max_frames=args.max_frames or 300,
        )
        dataset_name = f"webcam_{args.webcam}"

    # ------------------------------------------------------------------ #
    # Build tracker
    # ------------------------------------------------------------------ #
    tracker_name = args.tracker.lower()
    if tracker_name not in available_trackers():
        parser.error(
            f"Unknown tracker '{tracker_name}'.  "
            f"Available: {', '.join(available_trackers())}."
        )
    tracker = build_tracker(tracker_name)

    # ------------------------------------------------------------------ #
    # Build engine and run
    # ------------------------------------------------------------------ #
    engine = BenchmarkEngine(
        verbose=not args.quiet,
        tdp_watts=args.tdp,
        gpu_profiling=args.gpu_profiling,
        gpu_device_index=args.gpu_device,
    )

    result = engine.run(tracker, dataset, dataset_name=dataset_name)

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    summary = result.summary()
    print("\n=== Benchmark Summary ===")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    # GPU summary (if enabled)
    for sr in result.sequence_results:
        if sr.gpu_profiling is not None:
            print(f"\n=== GPU Profiling ({sr.sequence_name}) ===")
            print(f"  {sr.gpu_profiling}")

    # ------------------------------------------------------------------ #
    # Save result
    # ------------------------------------------------------------------ #
    if args.save:
        save_path = result.save(args.save)
        print(f"\nResult saved to: {save_path}")

    # Shutdown GPU profiler thread cleanly.
    if engine._gpu_profiler is not None:
        engine._gpu_profiler.close()


if __name__ == "__main__":
    main()
