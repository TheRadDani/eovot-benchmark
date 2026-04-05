#!/usr/bin/env python3
"""Hardware-aware tracker selection CLI for EOVOT.

Auto-detects (or loads) a hardware profile and ranks all built-in trackers
against the given deployment constraints, printing a recommendation table.

Usage::

    # Auto-detect hardware and recommend with default constraints
    python scripts/select_tracker.py

    # Set a minimum FPS requirement and prefer accuracy
    python scripts/select_tracker.py --target-fps 30 --accuracy-weight 0.8

    # Simulate a Raspberry Pi 4 profile
    python scripts/select_tracker.py \\
        --profile configs/hardware_profiles/raspberry_pi4.yaml

    # Constrain memory (e.g. only 64 MB budget for tracker)
    python scripts/select_tracker.py --max-memory-mb 64

    # Override TDP for energy estimates
    python scripts/select_tracker.py --tdp-watts 6.0 --target-fps 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from eovot.selection.hardware_profile import HardwareProfile
from eovot.selection.tracker_selector import TrackerConstraints, TrackerSelector


def _load_profile(path: str) -> HardwareProfile:
    """Load a hardware profile from a YAML preset file."""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return HardwareProfile.from_dict(d)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="select_tracker",
        description=(
            "EOVOT hardware-aware tracker selector.\n\n"
            "Ranks available trackers by a composite speed+accuracy score "
            "given the target device's hardware profile and your constraints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        metavar="YAML",
        default=None,
        help=(
            "Path to a YAML hardware profile preset "
            "(e.g. configs/hardware_profiles/raspberry_pi4.yaml). "
            "If omitted, the current machine is auto-detected."
        ),
    )
    parser.add_argument(
        "--tdp-watts",
        type=float,
        default=None,
        metavar="W",
        help="Override TDP estimate in Watts (used with auto-detect).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=1.0,
        metavar="FPS",
        help="Minimum acceptable FPS on the target device (default: 1.0).",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=None,
        metavar="MB",
        help="Maximum acceptable peak memory usage in MB (default: no limit).",
    )
    parser.add_argument(
        "--accuracy-weight",
        type=float,
        default=0.5,
        metavar="W",
        help=(
            "How much to weight accuracy vs. speed in the composite score. "
            "0 = pure speed, 1 = pure accuracy (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output a Markdown table instead of plain text.",
    )
    args = parser.parse_args()

    # --- Load or detect hardware profile ---
    if args.profile:
        try:
            profile = _load_profile(args.profile)
        except FileNotFoundError:
            print(f"[ERROR] Profile file not found: {args.profile}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded hardware profile from: {args.profile}")
    else:
        profile = HardwareProfile.detect(tdp_watts=args.tdp_watts)
        print("Auto-detected hardware profile.")

    print(f"\n{profile}\n")

    # --- Build constraints ---
    try:
        constraints = TrackerConstraints(
            target_fps=args.target_fps,
            max_memory_mb=args.max_memory_mb,
            accuracy_weight=args.accuracy_weight,
        )
    except ValueError as exc:
        print(f"[ERROR] Invalid constraint: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Constraints:")
    print(f"  target FPS    : >= {constraints.target_fps}")
    print(f"  max memory    : {constraints.max_memory_mb or 'unlimited'} MB")
    print(f"  accuracy weight: {constraints.accuracy_weight:.2f}  "
          f"(speed weight: {1 - constraints.accuracy_weight:.2f})")
    print()

    # --- Rank trackers ---
    selector = TrackerSelector()
    recs = selector.rank(profile, constraints)

    if not recs:
        print("No trackers satisfy the given constraints.")
        print("Try lowering --target-fps or increasing --max-memory-mb.")
        sys.exit(0)

    if args.markdown:
        print(selector.summary_table(profile, constraints))
    else:
        print("=" * 65)
        print(f"  Tracker Recommendations  ({len(recs)} eligible)")
        print("=" * 65)
        for r in recs:
            marker = "  <<< BEST" if r.rank == 1 else ""
            print(f"  {r}{marker}")
            if r.notes:
                print(f"       Note: {r.notes}")
        print("=" * 65)

    best = recs[0]
    print(f"\nRecommended tracker: {best.tracker_name}")
    print(
        f"  Expected FPS ≈ {best.estimated_fps:.1f}  |  "
        f"AUC = {best.accuracy_auc:.3f}  |  "
        f"Memory ≈ {best.peak_memory_mb:.0f} MB"
    )
    print(
        f"\nTo benchmark this tracker:\n"
        f"  python scripts/run_benchmark.py "
        f"--tracker {best.tracker_name} "
        f"--dataset-root /path/to/dataset "
        f"--tdp-watts {profile.tdp_watts}"
    )


if __name__ == "__main__":
    main()
