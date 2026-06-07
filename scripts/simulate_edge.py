#!/usr/bin/env python3
"""Edge deployment simulation CLI for EOVOT.

Projects benchmark results (saved as JSON) onto edge hardware profiles and
generates a deployment matrix answering: *which trackers can run on which
devices?*

This script is the command-line interface to
:class:`~eovot.reporting.edge_report.EdgeDeploymentReport`.  It reads one or
more JSON result files produced by the benchmark engine (or
:class:`~eovot.experiment.runner.ExperimentRunner`) and outputs:

1. A Markdown deployment matrix (trackers × devices) with FPS, energy, and
   memory feasibility.
2. A JSON export of all simulation data.

Usage
-----
Simulate all trackers in a results directory on all built-in devices::

    python scripts/simulate_edge.py results/my-experiment/

Simulate specific result files, only on embedded Linux devices::

    python scripts/simulate_edge.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        --devices rpi4 rpi5 coral_board \\
        --min-fps 20

Model a 2-minute sustained tracking workload (triggers thermal throttling)::

    python scripts/simulate_edge.py results/experiment/ \\
        --sustained-seconds 120 \\
        --min-fps 10 \\
        --output-dir results/edge_analysis

List available device profiles::

    python scripts/simulate_edge.py --list-devices

Available built-in devices
--------------------------
    rpi4          Raspberry Pi 4B (Cortex-A72, 4 GB, 7.5 W)
    rpi5          Raspberry Pi 5   (Cortex-A76, 8 GB, 12 W)
    jetson_nano   NVIDIA Jetson Nano (Cortex-A57, 4 GB, 10 W)
    jetson_xnx    NVIDIA Jetson Xavier NX (Carmel ARM, 8 GB, 15 W)
    coral_board   Google Coral Dev Board (i.MX 8M, 1 GB, 4 W)
    snapdragon888 Qualcomm Snapdragon 888 mobile (Kryo 680, 6 GB, 15 W)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.profiling.device_sim import KNOWN_DEVICES
from eovot.reporting.edge_report import EdgeDeploymentReport


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _collect_json_files(inputs: List[str]) -> List[Path]:
    """Expand directories to *.json; keep explicit file paths as-is."""
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(p.glob("*.json"))
            if not found:
                print(f"[WARN] No JSON files found in: {p}", file=sys.stderr)
            paths.extend(found)
        elif p.is_file() and p.suffix == ".json":
            paths.append(p)
        elif p.is_file():
            print(f"[WARN] Skipping non-JSON file: {p}", file=sys.stderr)
        else:
            print(f"[WARN] Path not found: {p}", file=sys.stderr)
    return paths


def _load_result(path: Path) -> Optional[dict]:
    """Load one JSON result file; return None if malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Could not read {path.name}: {exc}", file=sys.stderr)
        return None

    # Support both top-level summary dicts and wrapped {summary, sequences} dicts.
    if "summary" not in data:
        # Wrap legacy / plain-summary JSONs
        data = {"summary": data, "sequences": []}

    summary = data["summary"]
    if not summary.get("tracker_name") and not summary.get("tracker"):
        summary["tracker_name"] = path.stem.split("-")[0]

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="simulate_edge",
        description="Project EOVOT benchmark results onto edge device profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="PATH",
        help=(
            "JSON result files or directories containing them.  "
            "Use --list-devices to skip simulation and only list device profiles."
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "Subset of device keys to simulate.  "
            f"Choices: {sorted(KNOWN_DEVICES)}.  "
            "Default: all built-in devices."
        ),
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=10.0,
        metavar="FPS",
        help="Minimum FPS required for a deployment to be marked feasible. Default: 10.",
    )
    parser.add_argument(
        "--sustained-seconds",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Duration of sustained tracking on the device.  "
            "Controls thermal throttling model.  "
            "0 (default) = cold-start / burst scenario."
        ),
    )
    parser.add_argument(
        "--host-calibration",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Calibration multiplier for the benchmark host CPU relative to an "
            "Intel Core i7 reference (default: 1.0).  "
            "Set > 1.0 if your host is faster; < 1.0 if slower."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/edge_report",
        metavar="DIR",
        help="Directory where Markdown and JSON reports are saved. Default: results/edge_report",
    )
    parser.add_argument(
        "--prefix",
        default="edge_report",
        help="Filename prefix for saved reports. Default: edge_report",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print all available device profiles and exit.",
    )
    args = parser.parse_args(argv)

    # --- List devices mode ---
    if args.list_devices:
        print("\nAvailable edge device profiles:")
        print("-" * 60)
        for key, profile in sorted(KNOWN_DEVICES.items()):
            print(
                f"  {key:<20} {profile.display_name}  "
                f"[{profile.cpu_speed_factor:.0%} host speed, "
                f"{profile.memory_limit_mb:.0f} MB RAM, "
                f"{profile.tdp_watts:.1f} W TDP]"
            )
            if profile.notes:
                print(f"    Note: {profile.notes}")
        print()
        return 0

    if not args.inputs:
        parser.print_help()
        return 1

    # --- Validate device choices ---
    if args.devices:
        unknown = [d for d in args.devices if d not in KNOWN_DEVICES]
        if unknown:
            print(
                f"[ERROR] Unknown device(s): {unknown}. "
                f"Run --list-devices to see available options.",
                file=sys.stderr,
            )
            return 1

    # --- Load result files ---
    json_files = _collect_json_files(args.inputs)
    if not json_files:
        print("[ERROR] No JSON result files found.", file=sys.stderr)
        return 1

    result_dicts = []
    for path in json_files:
        data = _load_result(path)
        if data is not None:
            result_dicts.append(data)
            tracker = (
                data.get("summary", {}).get("tracker_name")
                or data.get("summary", {}).get("tracker", path.stem)
            )
            print(f"  Loaded: {tracker!r}  ({path.name})")

    if not result_dicts:
        print("[ERROR] No valid result files could be loaded.", file=sys.stderr)
        return 1

    # --- Run simulation ---
    print(
        f"\nSimulating {len(result_dicts)} tracker(s) on "
        f"{len(args.devices) if args.devices else len(KNOWN_DEVICES)} device(s) "
        f"[min FPS={args.min_fps}, sustained={args.sustained_seconds:.0f}s] …"
    )

    report = EdgeDeploymentReport.from_result_dicts(
        result_dicts,
        min_fps=args.min_fps,
        sustained_seconds=args.sustained_seconds,
        device_names=args.devices,
        host_calibration_factor=args.host_calibration,
    )

    # --- Print deployment matrix to stdout ---
    print()
    print(report.to_markdown())

    # --- Save reports ---
    saved = report.save(output_dir=args.output_dir, prefix=args.prefix)
    print(f"[Markdown] saved → {saved['markdown']}")
    print(f"[JSON]     saved → {saved['json']}")

    n_ok = len(report.deployable_combinations())
    n_total = len(report.tracker_names) * len(report.device_names)
    print(
        f"\n{n_ok}/{n_total} tracker-device combinations are deployable "
        f"at ≥{args.min_fps:.0f} FPS.\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
