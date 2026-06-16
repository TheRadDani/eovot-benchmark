#!/usr/bin/env python3
"""CLI for visualising per-sequence tracking quality from EOVOT JSON results.

Three sub-commands are available:

**timeline** — per-frame IoU for one tracker on one sequence:

.. code-block:: bash

    python scripts/visualize_tracking.py timeline \\
        --result results/MOSSE-OTB100.json \\
        --sequence Basketball \\
        --output plots/basketball_iou.png

**heatmap** — per-sequence metrics grid for one tracker run:

.. code-block:: bash

    python scripts/visualize_tracking.py heatmap \\
        --result results/KCF-OTB100.json \\
        --output plots/kcf_heatmap.png \\
        --metrics mean_iou fps peak_memory_mb

**compare** — overlay IoU timelines for multiple trackers on the same sequence:

.. code-block:: bash

    python scripts/visualize_tracking.py compare \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --sequence Basketball \\
        --output plots/compare_basketball.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running as a top-level script without installing the package.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_result(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _find_sequence(result: dict, name: str) -> dict:
    """Return the sequence sub-dict matching *name*, or raise ``KeyError``."""
    for seq in result.get("sequences", []):
        if seq.get("sequence_name") == name:
            return seq
    available = [s.get("sequence_name") for s in result.get("sequences", [])]
    preview = available[:10]
    suffix = "…" if len(available) > 10 else ""
    raise KeyError(
        f"Sequence '{name}' not found in result.\n"
        f"Available sequences: {preview}{suffix}"
    )


def _require_ious(seq: dict, tracker_name: str) -> np.ndarray:
    raw = seq.get("ious")
    if raw is None:
        print(
            f"[error] Tracker '{tracker_name}' / sequence "
            f"'{seq.get('sequence_name')}' has no per-frame 'ious' data.\n"
            "        Re-run the benchmark to generate per-frame IoU arrays.",
            file=sys.stderr,
        )
        sys.exit(1)
    return np.array(raw, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_timeline(args: argparse.Namespace) -> None:
    """Plot per-frame IoU timeline for one tracker on one sequence."""
    from eovot.visualization.trajectory import plot_iou_timeline

    result = _load_result(args.result)
    seq = _find_sequence(result, args.sequence)
    ious = _require_ious(seq, result.get("summary", {}).get("tracker", "?"))
    tracker_name = result.get("summary", {}).get("tracker", Path(args.result).stem)

    plot_iou_timeline(
        ious=ious,
        sequence_name=args.sequence,
        tracker_name=tracker_name,
        failure_threshold=args.failure_threshold,
        output_path=args.output,
    )


def cmd_heatmap(args: argparse.Namespace) -> None:
    """Plot per-sequence metrics heatmap for one tracker run."""
    from eovot.visualization.trajectory import plot_sequence_heatmap

    result = _load_result(args.result)
    metrics = args.metrics or ["mean_iou", "fps", "peak_memory_mb"]

    plot_sequence_heatmap(
        result_dict=result,
        metrics=metrics,
        output_path=args.output,
        max_sequences=args.max_sequences,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    """Overlay IoU timelines for multiple trackers on the same sequence."""
    from eovot.visualization.trajectory import plot_multi_tracker_iou_timeline

    tracker_ious: Dict[str, np.ndarray] = {}
    for result_path in args.results:
        result = _load_result(result_path)
        tracker_name = result.get("summary", {}).get("tracker", Path(result_path).stem)
        try:
            seq = _find_sequence(result, args.sequence)
        except KeyError as exc:
            print(f"[warning] {exc}\n  Skipping {result_path}.", file=sys.stderr)
            continue
        raw = seq.get("ious")
        if raw is None:
            print(
                f"[warning] '{tracker_name}' has no per-frame ious in "
                f"'{args.sequence}'; skipping.",
                file=sys.stderr,
            )
            continue
        tracker_ious[tracker_name] = np.array(raw, dtype=np.float64)

    if not tracker_ious:
        print(
            "[error] No tracker provided per-frame IoU data for this sequence.",
            file=sys.stderr,
        )
        sys.exit(1)

    plot_multi_tracker_iou_timeline(
        tracker_ious=tracker_ious,
        sequence_name=args.sequence,
        failure_threshold=args.failure_threshold,
        output_path=args.output,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize_tracking",
        description=(
            "Visualise per-sequence tracking quality from EOVOT JSON result files.\n"
            "Use 'timeline', 'heatmap', or 'compare' sub-commands."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── timeline ────────────────────────────────────────────────────────────
    tl = sub.add_parser(
        "timeline",
        help="Plot per-frame IoU timeline for one tracker on one sequence.",
    )
    tl.add_argument("--result", required=True, metavar="FILE",
                    help="Path to JSON result file (e.g. results/MOSSE-OTB100.json).")
    tl.add_argument("--sequence", required=True, metavar="NAME",
                    help="Sequence name exactly as it appears in the result file.")
    tl.add_argument("--output", default=None, metavar="FILE",
                    help="Output image path (PNG/PDF/SVG). Omit for interactive display.")
    tl.add_argument("--failure-threshold", type=float, default=0.1, metavar="F",
                    help="IoU below which a frame is flagged as failure (default 0.1).")

    # ── heatmap ─────────────────────────────────────────────────────────────
    hm = sub.add_parser(
        "heatmap",
        help="Plot per-sequence metrics heatmap for one tracker run.",
    )
    hm.add_argument("--result", required=True, metavar="FILE",
                    help="Path to JSON result file.")
    hm.add_argument("--output", default=None, metavar="FILE",
                    help="Output image path. Omit for interactive display.")
    hm.add_argument("--metrics", nargs="+", default=None, metavar="KEY",
                    help=(
                        "Metric keys to display (default: mean_iou fps peak_memory_mb). "
                        "Any key present in the sequence dicts is accepted."
                    ))
    hm.add_argument("--max-sequences", type=int, default=50, metavar="N",
                    help="Maximum number of sequences shown as rows (default 50).")

    # ── compare ─────────────────────────────────────────────────────────────
    cp = sub.add_parser(
        "compare",
        help="Overlay IoU timelines for multiple trackers on the same sequence.",
    )
    cp.add_argument("--results", nargs="+", required=True, metavar="FILE",
                    help="Paths to JSON result files, one per tracker.")
    cp.add_argument("--sequence", required=True, metavar="NAME",
                    help="Sequence name to compare trackers on.")
    cp.add_argument("--output", default=None, metavar="FILE",
                    help="Output image path. Omit for interactive display.")
    cp.add_argument("--failure-threshold", type=float, default=0.1, metavar="F",
                    help="Reference line drawn at this IoU level (default 0.1).")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "timeline": cmd_timeline,
        "heatmap": cmd_heatmap,
        "compare": cmd_compare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
