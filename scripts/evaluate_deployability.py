#!/usr/bin/env python3
"""Full deployability analysis pipeline for saved EOVOT benchmark results.

Loads one or more ``BenchmarkResult`` JSON files produced by the benchmark
engine, runs the complete post-processing pipeline — robustness analysis,
Edge Efficiency Score, Composite Deployability Score (CDS) — and writes a
structured Markdown report.  Optionally projects every tracker's profiling
data onto a target edge device using :class:`~eovot.profiling.device_sim.DeviceSimulator`.

This tool is the recommended way to turn raw benchmark JSON files into a
publication-quality tracker comparison table without re-running the benchmark.

Usage::

    # Analyse two saved results and print the report
    python scripts/evaluate_deployability.py \\
        results/MOSSE-OTB100.json results/KCF-OTB100.json

    # With edge device projection (Raspberry Pi 4, 2-minute sustained load)
    python scripts/evaluate_deployability.py \\
        results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --device rpi4 --sustained-seconds 120

    # Custom CDS weights (must sum to 1.0)
    python scripts/evaluate_deployability.py results/*.json \\
        --w-acc 0.50 --w-rob 0.15 --w-eff 0.25 --w-smooth 0.10

    # Save Markdown report to file
    python scripts/evaluate_deployability.py results/*.json \\
        --output results/deployability_report.md

    # Use smaller memory budget for heavily constrained devices
    python scripts/evaluate_deployability.py results/*.json \\
        --memory-budget-mb 256 --device coral_board

Available devices: rpi4, rpi5, jetson_nano, jetson_xnx, coral_board, snapdragon888
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python scripts/evaluate_deployability.py`` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.metrics.composite import CompositeDeployabilityScore
from eovot.profiling.device_sim import KNOWN_DEVICES, DeviceSimulator
from eovot.profiling.profiler import ProfilingResult


def _load_result_dict(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_profiling_result(summary: dict, tracker_name: str) -> ProfilingResult:
    """Construct a minimal ProfilingResult from a benchmark summary dict.

    Only FPS and memory are available in the summary; latency percentiles
    are approximated from the mean for device simulation purposes.
    """
    fps = float(summary.get("mean_fps", 1.0))
    lat_ms = 1000.0 / fps if fps > 0 else 0.0
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=int(summary.get("num_sequences", 1)),
        fps=fps,
        latency_mean_ms=lat_ms,
        latency_std_ms=0.0,
        latency_p95_ms=lat_ms,
        latency_p99_ms=lat_ms,
        latency_cv=0.0,
        peak_memory_mb=float(summary.get("peak_memory_mb", 0.0)),
    )


def _build_cds_section(
    engine: CompositeDeployabilityScore,
    result_dicts: list,
    w_acc: float,
    w_rob: float,
    w_eff: float,
    w_smooth: float,
    memory_budget_mb: float,
) -> tuple[list, list]:
    """Run CDS ranking and return (entries, report_lines)."""
    entries = engine.rank_from_dicts(result_dicts)

    lines = [
        "## Composite Deployability Score (CDS) Ranking",
        "",
        f"> **Weights** — accuracy: {w_acc}, robustness: {w_rob}, "
        f"efficiency: {w_eff}, smoothness: {w_smooth} "
        f"(temporal unavailable in JSON mode → smoothness weight redistributed)",
        f">",
        f"> **Memory budget for EES**: {memory_budget_mb:.0f} MB",
        "",
        engine.to_markdown_table(entries),
        "",
    ]
    return entries, lines


def _build_device_section(
    result_dicts: list,
    device_name: str,
    sustained_seconds: float,
) -> list:
    """Run device simulation and return report lines."""
    sim = DeviceSimulator()
    profile = sim.get_profile(device_name)
    lines = [
        f"## Edge Device Projection: {profile.display_name}",
        "",
        f"> Sustained load: **{sustained_seconds:.0f} s** · "
        f"TDP: {profile.tdp_watts} W · "
        f"RAM limit: {profile.memory_limit_mb:.0f} MB",
        f"> _{profile.notes}_",
        "",
        "| Tracker | FPS (host) | FPS (device) | Latency (ms) | Mem (MB) | Fits? | Thermal | mJ/frame |",
        "|---------|----------:|-------------:|-------------:|---------:|:-----:|---------|----------|",
    ]

    for d in result_dicts:
        summary = d.get("summary", {})
        tracker_name = summary.get("tracker", "?")
        prof = _make_profiling_result(summary, tracker_name)
        r = sim.simulate(prof, device_name, sustained_seconds=sustained_seconds)
        fits = "✓" if r.fits_in_memory else "✗ OOM"
        lines.append(
            f"| {tracker_name} "
            f"| {r.host_fps:.1f} "
            f"| {r.estimated_fps:.1f} "
            f"| {r.estimated_latency_ms:.1f} "
            f"| {r.estimated_memory_mb:.0f} / {r.memory_limit_mb:.0f} "
            f"| {fits} "
            f"| {r.thermal_state} "
            f"| {r.estimated_energy_mj_per_frame:.3f} |"
        )

    lines.append("")
    return lines


def _build_robustness_section(entries: list) -> list:
    lines = [
        "## Robustness Summary",
        "",
        "| Tracker | EAO | Failures (total) | Survival Rate |",
        "|---------|----:|-----------------:|--------------:|",
    ]
    for e in entries:
        lines.append(
            f"| {e.tracker_name} "
            f"| {e.eao:.4f} "
            f"| {e.num_failures} "
            f"| {e.survival_rate:.4f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate_deployability",
        description=(
            "Run the full EOVOT deployability analysis pipeline on saved "
            "benchmark result JSON files and write a Markdown report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more BenchmarkResult JSON files.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=list(KNOWN_DEVICES),
        metavar="DEVICE",
        help=f"Project performance onto this edge device. Choices: {list(KNOWN_DEVICES)}",
    )
    parser.add_argument(
        "--sustained-seconds",
        type=float,
        default=60.0,
        metavar="S",
        help="Duration of continuous tracking (seconds) used for thermal throttling model.",
    )
    parser.add_argument(
        "--w-acc",
        type=float,
        default=0.40,
        dest="w_acc",
        help="CDS weight for accuracy (success_auc).",
    )
    parser.add_argument(
        "--w-rob",
        type=float,
        default=0.20,
        dest="w_rob",
        help="CDS weight for robustness (EAO).",
    )
    parser.add_argument(
        "--w-eff",
        type=float,
        default=0.30,
        dest="w_eff",
        help="CDS weight for efficiency (normalised EES).",
    )
    parser.add_argument(
        "--w-smooth",
        type=float,
        default=0.10,
        dest="w_smooth",
        help="CDS weight for temporal smoothness (redistributed in JSON mode).",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget for Edge Efficiency Score computation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write the Markdown report to this file (default: stdout only).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate weights
    # -----------------------------------------------------------------------
    total_w = args.w_acc + args.w_rob + args.w_eff + args.w_smooth
    if abs(total_w - 1.0) > 1e-4:
        print(
            f"[ERROR] CDS weights must sum to 1.0 (got {total_w:.4f}). "
            "Adjust --w-acc, --w-rob, --w-eff, --w-smooth.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load result JSON files
    # -----------------------------------------------------------------------
    result_dicts = []
    for p_str in args.results:
        path = Path(p_str)
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)
        d = _load_result_dict(path)
        result_dicts.append(d)
        tracker = d.get("summary", {}).get("tracker", "?")
        print(f"[loaded] {path}  ({tracker})")

    if not result_dicts:
        print("[ERROR] No result files provided.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # CDS analysis
    # -----------------------------------------------------------------------
    cds_engine = CompositeDeployabilityScore(
        w_acc=args.w_acc,
        w_rob=args.w_rob,
        w_eff=args.w_eff,
        w_smooth=args.w_smooth,
        memory_budget_mb=args.memory_budget_mb,
    )

    entries, cds_lines = _build_cds_section(
        cds_engine,
        result_dicts,
        w_acc=args.w_acc,
        w_rob=args.w_rob,
        w_eff=args.w_eff,
        w_smooth=args.w_smooth,
        memory_budget_mb=args.memory_budget_mb,
    )

    # -----------------------------------------------------------------------
    # Assemble report
    # -----------------------------------------------------------------------
    datasets = {d.get("summary", {}).get("dataset", "?") for d in result_dicts}
    dataset_label = ", ".join(sorted(datasets))
    n_trackers = len(result_dicts)

    report_lines = [
        "# EOVOT Deployability Analysis Report",
        "",
        f"> **Trackers evaluated**: {n_trackers}  ·  **Dataset(s)**: {dataset_label}",
        "",
    ]
    report_lines += cds_lines

    if args.device:
        report_lines += _build_device_section(
            result_dicts, args.device, args.sustained_seconds
        )

    report_lines += _build_robustness_section(entries)

    report = "\n".join(report_lines)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EOVOT DEPLOYABILITY ANALYSIS")
    print("=" * 70)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n[Report saved] → {out_path}")


if __name__ == "__main__":
    main()
