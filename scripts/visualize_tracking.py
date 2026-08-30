#!/usr/bin/env python
"""CLI tool to render tracking overlay videos from benchmark prediction files.

Loads ground-truth and tracker predictions from a saved benchmark JSON result
(produced by BenchmarkEngine) and renders annotated MP4 overlays for each
sequence, enabling qualitative inspection of tracker behaviour.

Usage::

    # Overlay one result file using all trackers stored inside it
    python scripts/visualize_tracking.py results/my_benchmark.json \\
        --output out/overlays \\
        --fps 25

    # Render only specific sequences
    python scripts/visualize_tracking.py results/my_benchmark.json \\
        --sequences ball-2 person-1 \\
        --output out/overlays

    # Save individual frames instead of (or in addition to) video
    python scripts/visualize_tracking.py results/my_benchmark.json \\
        --save-frames --no-video

    # Side-by-side comparison strip (one panel per tracker)
    python scripts/visualize_tracking.py results/my_benchmark.json \\
        --mode strip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render tracking overlay videos from a benchmark result JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("result_json", help="Path to a BenchmarkEngine result JSON file.")
    p.add_argument("-o", "--output", default="overlay_output",
                   help="Output directory for rendered overlays.")
    p.add_argument("--fps", type=float, default=25.0,
                   help="Frames-per-second for video output.")
    p.add_argument("--sequences", nargs="*", metavar="SEQ",
                   help="Sequence names to render (default: all in result file).")
    p.add_argument("--save-frames", action="store_true",
                   help="Also save individual JPEG frames.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip MP4 video output (useful with --save-frames).")
    p.add_argument("--mode", choices=["overlay", "strip"], default="overlay",
                   help="'overlay': all trackers on one frame. "
                        "'strip': side-by-side panel per tracker.")
    p.add_argument("--no-iou", action="store_true",
                   help="Hide IoU score in tracker labels.")
    p.add_argument("--no-legend", action="store_true",
                   help="Hide the tracker legend panel.")
    return p.parse_args()


def _load_result(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _bbox_list(data) -> List[Optional[tuple]]:
    """Convert a list of [x,y,w,h] entries (possibly null) to tuples."""
    result = []
    for entry in data:
        if entry is None:
            result.append(None)
        else:
            result.append(tuple(entry))
    return result


def _load_frames_from_paths(frame_paths: List[str]) -> List[np.ndarray]:
    frames = []
    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            h, w = 120, 160
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(frame, "missing", (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 80, 80), 1)
        frames.append(frame)
    return frames


def main() -> None:
    args = _parse_args()

    result = _load_result(args.result_json)

    from eovot.visualization.overlay import TrackingOverlayRenderer, OverlayConfig

    cfg = OverlayConfig(
        show_iou=not args.no_iou,
        show_legend=not args.no_legend,
    )
    renderer = TrackingOverlayRenderer(
        output_dir=args.output,
        config=cfg,
        fps=args.fps,
    )

    sequences = result.get("sequences", [])
    if args.sequences:
        sequences = [s for s in sequences if s.get("name") in args.sequences]

    if not sequences:
        print("No sequences found in result file (or no match for --sequences filter).",
              file=sys.stderr)
        sys.exit(1)

    for seq_data in sequences:
        seq_name = seq_data.get("name", "unknown")
        frame_paths = seq_data.get("frame_paths", [])
        gt_raw = seq_data.get("ground_truth", [])
        predictions_raw = seq_data.get("predictions", {})

        if not frame_paths:
            print(f"[{seq_name}] No frame_paths in result — skipping.", file=sys.stderr)
            continue

        frames = _load_frames_from_paths(frame_paths)
        gt_bboxes = _bbox_list(gt_raw) if gt_raw else None
        predictions: Dict[str, List] = {
            name: _bbox_list(preds)
            for name, preds in predictions_raw.items()
        }

        print(f"Rendering '{seq_name}' ({len(frames)} frames, "
              f"{len(predictions)} tracker(s)) …")

        if args.mode == "strip" and predictions:
            # Side-by-side mode: write one comparison strip per frame
            strip_dir = Path(args.output) / seq_name / "strips"
            strip_dir.mkdir(parents=True, exist_ok=True)
            tracker_names = list(predictions.keys())
            for i, frame in enumerate(frames):
                strip = renderer.render_comparison_strip(
                    frame, gt_bboxes, predictions, frame_idx=i
                )
                cv2.imwrite(str(strip_dir / f"strip_{i:05d}.jpg"), strip)
            print(f"  → Strips saved to {strip_dir}")
        else:
            out_dir = renderer.render_sequence(
                sequence_name=seq_name,
                frames=frames,
                gt_bboxes=gt_bboxes,
                predictions=predictions,
                save_video=not args.no_video,
                save_frames=args.save_frames,
            )
            print(f"  → Output: {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
