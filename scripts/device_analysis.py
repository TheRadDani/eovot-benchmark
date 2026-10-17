#!/usr/bin/env python3
"""Device fleet deployment analysis CLI for EOVOT.

Reads one or more JSON benchmark result files produced by
:meth:`~eovot.benchmark.engine.BenchmarkEngine.run` and projects each tracker's
measured performance onto a configurable fleet of edge devices using
:class:`~eovot.profiling.device_sim.DeviceSimulator`.

Outputs:

* Markdown deployment table (stdout and optional file)
* PNG heatmap of the chosen metric across trackers × devices (requires matplotlib)

Usage
-----
Analyse a single result::

    python scripts/device_analysis.py results/MOSSE-OTB100.json

Compare multiple trackers across all built-in devices::

    python scripts/device_analysis.py \\
        results/MOSSE-synthetic.json \\
        results/KCF-synthetic.json   \\
        results/CSRT-synthetic.json  \\
        --output-dir analysis/

Thermal modelling (60 s sustained load)::

    python scripts/device_analysis.py results/*.json \\
        --sustained 60.0 \\
        --metric energy_mj \\
        --output-dir analysis/thermal/

Target a subset of devices::

    python scripts/device_analysis.py results/MOSSE-synthetic.json \\
        --devices rpi4 jetson_nano coral_board \\
        --metric viability

Correct for a faster host machine::

    python scripts/device_analysis.py results/MOSSE-synthetic.json \\
        --host-calibration 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as ``python scripts/device_analysis.py`` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.profiling.device_sim import KNOWN_DEVICES, DeviceSimulator
from eovot.profiling.profiler import ProfilingResult


_METRIC_CHOICES = ("fps", "latency_ms", "energy_mj", "viability")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _collect_json_paths(inputs: List[str]) -> List[Path]:
    """Expand directory arguments to JSON files; pass plain file paths through."""
    paths: List[Path] = []
    for p_str in inputs:
        p = Path(p_str)
        if p.is_dir():
            found = sorted(p.glob("*.json"))
            if not found:
                print(f"[WARN] No JSON files in directory: {p}", file=sys.stderr)
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[WARN] Path not found: {p}", file=sys.stderr)
    return paths


def _load_result(path: Path) -> Optional[dict]:
    """Parse a JSON benchmark result file.

    Returns:
        Dict with at least a ``"summary"`` key, or ``None`` on error.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
        return None
    if "summary" not in data:
        # Legacy format: the whole file is the summary
        data = {"summary": data, "sequences": []}
    return data


def _build_profiling_result(data: dict, path: Path) -> Optional[ProfilingResult]:
    """Extract a :class:`ProfilingResult` from a benchmark result dict.

    Args:
        data: Parsed benchmark result dict.
        path: Source file path (used to infer tracker name when absent).

    Returns:
        :class:`ProfilingResult` on success, ``None`` when required fields are missing.
    """
    summary = data.get("summary", {})
    tracker_name = (
        summary.get("tracker_name")
        or summary.get("tracker")
        or path.stem.split("-")[0]
    )
    fps = summary.get("mean_fps")
    mem = summary.get("peak_memory_mb")
    if fps is None or mem is None:
        print(
            f"[WARN] Skipping {path.name}: missing 'mean_fps' or 'peak_memory_mb'.",
            file=sys.stderr,
        )
        return None
    fps = float(fps)
    mem = float(mem)
    latency_ms = 1_000.0 / fps if fps > 0 else 1_000.0
    return ProfilingResult(
        tracker_name=str(tracker_name),
        frame_count=int(summary.get("num_sequences", 1)) * 100,
        fps=fps,
        latency_mean_ms=latency_ms,
        latency_std_ms=0.0,
        latency_p95_ms=latency_ms * 1.3,
        peak_memory_mb=mem,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_METRIC_UNITS = {
    "fps": "Est. FPS",
    "latency_ms": "Latency (ms)",
    "energy_mj": "Energy (mJ/fr)",
    "viability": "Viable?",
}


def _format_cell(r, metric: str) -> str:
    """Format a DeviceSimResult cell for the Markdown table."""
    if metric == "fps":
        cell = f"{r.estimated_fps:.1f}"
        if not r.fits_in_memory:
            cell += " ⚠OOM"
    elif metric == "latency_ms":
        cell = f"{r.estimated_latency_ms:.1f}"
        if not r.fits_in_memory:
            cell += " ⚠OOM"
    elif metric == "energy_mj":
        cell = f"{r.estimated_energy_mj_per_frame:.3f}"
        if not r.fits_in_memory:
            cell += " ⚠OOM"
    else:  # viability
        viable = r.fits_in_memory and r.estimated_fps >= 5.0
        cell = "✓" if viable else "✗"
    return cell


def _build_markdown_report(
    sim_matrix: Dict[str, Dict[str, object]],
    device_names: List[str],
    device_display: Dict[str, str],
    metric: str,
    sustained_seconds: float,
) -> str:
    tracker_names = list(sim_matrix.keys())
    lines = [
        "# EOVOT Device Fleet Deployment Analysis\n",
        f"**Metric:** `{metric}` — {_METRIC_UNITS[metric]}  ",
        f"**Sustained load modelled:** {sustained_seconds:.0f} s  ",
        f"**Trackers:** {', '.join(tracker_names)}  \n",
    ]
    header_devs = " | ".join(device_display[d] for d in device_names)
    sep = "|".join(["---"] * (len(device_names) + 1))
    lines.append(f"| Tracker | {header_devs} |")
    lines.append(f"|{sep}|")
    for tracker in tracker_names:
        cells = [
            _format_cell(sim_matrix[tracker][d], metric)
            for d in device_names
        ]
        lines.append(f"| {tracker} | {' | '.join(cells)} |")
    lines.append("")
    lines.append("_⚠OOM = tracker exceeds device RAM.  Viable = FPS ≥ 5 and fits in RAM._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="device_analysis",
        description="Project EOVOT benchmark results onto an edge device fleet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="JSON",
        help="Benchmark result JSON files or directories containing them.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        metavar="DEVICE",
        help=(
            "Device subset to simulate.  "
            f"Available: {list(KNOWN_DEVICES)}.  Default: all built-in devices."
        ),
    )
    parser.add_argument(
        "--sustained",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Sustained-load duration for thermal throttling model "
            "(default: 0 = cold start burst scenario)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=_METRIC_CHOICES,
        default="fps",
        help="Metric to display in the heatmap and table (default: fps).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Save Markdown report (device_fleet_analysis.md) and heatmap PNG "
            "to this directory.  Prints to stdout only when omitted."
        ),
    )
    parser.add_argument(
        "--host-calibration",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Speed calibration factor for the benchmark host.  "
            "Use > 1.0 if your host is faster than an Intel i7 reference, "
            "< 1.0 if slower (default: 1.0)."
        ),
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    json_paths = _collect_json_paths(args.inputs)
    if not json_paths:
        print("[ERROR] No JSON result files found.", file=sys.stderr)
        return 1

    profiling_results: List[ProfilingResult] = []
    for path in json_paths:
        data = _load_result(path)
        if data is None:
            continue
        pr = _build_profiling_result(data, path)
        if pr is not None:
            profiling_results.append(pr)

    if not profiling_results:
        print("[ERROR] No valid profiling results loaded.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Device simulation
    # ------------------------------------------------------------------
    sim = DeviceSimulator(host_calibration_factor=args.host_calibration)
    device_names: List[str] = args.devices if args.devices else sim.list_devices()

    for d in device_names:
        try:
            sim.get_profile(d)
        except KeyError:
            print(
                f"[ERROR] Unknown device '{d}'.  "
                f"Available: {sim.list_devices()}",
                file=sys.stderr,
            )
            return 1

    sim_matrix: Dict[str, Dict[str, object]] = {}
    for pr in profiling_results:
        device_results = sim.simulate_all(
            pr,
            sustained_seconds=args.sustained,
            device_names=device_names,
        )
        sim_matrix[pr.tracker_name] = {r.device_name: r for r in device_results}

    device_display = {d: sim.get_profile(d).display_name for d in device_names}

    # ------------------------------------------------------------------
    # Build and print report
    # ------------------------------------------------------------------
    report = _build_markdown_report(
        sim_matrix,
        device_names,
        device_display,
        args.metric,
        args.sustained,
    )
    print(report)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / "device_fleet_analysis.md"
        md_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n[SAVED] Markdown → {md_path}", file=sys.stderr)

        try:
            from eovot.visualization.plots import plot_device_fleet_heatmap

            heatmap_path = str(out_dir / f"device_fleet_{args.metric}.png")
            plot_device_fleet_heatmap(
                sim_matrix=sim_matrix,
                device_names=device_names,
                metric=args.metric,
                output_path=heatmap_path,
                title=f"Device Fleet — {_METRIC_UNITS[args.metric]}",
            )
            print(f"[SAVED] Heatmap → {heatmap_path}", file=sys.stderr)
        except ImportError:
            print(
                "[INFO] Install matplotlib to generate the heatmap PNG: "
                "pip install matplotlib",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
