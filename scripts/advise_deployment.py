"""CLI tool: load saved benchmark results and generate a deployment report.

Usage examples::

    # Single device
    python scripts/advise_deployment.py \\
        results/mosse.json results/kcf.json results/csrt.json \\
        --profile jetson_nano

    # All built-in devices
    python scripts/advise_deployment.py results/*.json --all-profiles

    # Save Markdown report
    python scripts/advise_deployment.py results/*.json \\
        --profile raspberry_pi_4 \\
        --output reports/deployment_rpi4.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.profiling.deployment_advisor import DeploymentAdvisor
from eovot.profiling.hardware_profiles import PROFILES, list_profiles


# ---------------------------------------------------------------------------
# Lightweight result stub — mirrors BenchmarkResult public API so the advisor
# works without needing a full dataset on disk.
# ---------------------------------------------------------------------------


class _ProfilingStub:
    def __init__(self, fps: float, latency_ms: float, memory_mb: float) -> None:
        self.fps = fps
        self.latency_mean_ms = latency_ms
        self.peak_memory_mb = memory_mb


class _EnergyStub:
    def __init__(self, mean_power_w: float) -> None:
        self.mean_power_w = mean_power_w


class _SeqStub:
    def __init__(
        self,
        seq_name: str,
        fps: float,
        latency_ms: float,
        memory_mb: float,
        mean_power_w: float | None,
    ) -> None:
        self.sequence_name = seq_name
        self.profiling = _ProfilingStub(fps, latency_ms, memory_mb)
        self.energy = _EnergyStub(mean_power_w) if mean_power_w is not None else None


class _BenchmarkResultStub:
    """Minimal BenchmarkResult reconstructed from a JSON file."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        sequences_data = data.get("sequences", [])

        self.tracker_name: str = summary.get("tracker", path.stem)
        self.dataset_name: str = summary.get("dataset", "unknown")

        # Reconstruct per-sequence stubs from the serialised sequence list.
        # Fields available from BenchmarkResult.to_dict():
        #   sequence_name, mean_iou, fps, mean_latency_ms, peak_memory_mb
        #   [optional] energy_j, energy_per_frame_mj
        self.sequence_results = []
        for s in sequences_data:
            seq_name = s.get("sequence_name", "unknown")
            fps = float(s.get("fps", 0.0))
            lat = float(s.get("mean_latency_ms", 0.0))
            mem = float(s.get("peak_memory_mb", 0.0))
            # Approximate mean power from energy_per_frame_mj if available
            mean_power: float | None = None
            epf = s.get("energy_per_frame_mj")
            lat_s = lat / 1000.0
            if epf is not None and lat_s > 0:
                mean_power = (float(epf) / 1000.0) / lat_s  # mJ → J / s = W
            self.sequence_results.append(
                _SeqStub(seq_name, fps, lat, mem, mean_power)
            )

        # Aggregate properties expected by DeploymentAdvisor
        fpss = [s.profiling.fps for s in self.sequence_results] or [0.0]
        mems = [s.profiling.peak_memory_mb for s in self.sequence_results] or [0.0]
        self._mean_fps = float(sum(fpss) / len(fpss))
        self._peak_memory_mb = float(max(mems))

    @property
    def mean_fps(self) -> float:
        return self._mean_fps

    @property
    def peak_memory_mb(self) -> float:
        return self._peak_memory_mb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a hardware deployment report from EOVOT JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        metavar="RESULT_JSON",
        help="One or more JSON files produced by the EOVOT benchmark engine.",
    )
    profile_group = parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument(
        "--profile",
        metavar="PROFILE_NAME",
        help=(
            "Target hardware profile. "
            f"Choices: {', '.join(sorted(PROFILES))}"
        ),
    )
    profile_group.add_argument(
        "--all-profiles",
        action="store_true",
        help="Evaluate against all built-in hardware profiles.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write Markdown report to this file (default: print to stdout).",
    )
    parser.add_argument(
        "--memory-safety",
        type=float,
        default=0.80,
        metavar="FRACTION",
        help="Fraction of device RAM treated as usable (default: 0.80).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load results
    results: List[_BenchmarkResultStub] = []
    for path_str in args.result_files:
        p = Path(path_str)
        if not p.is_file():
            print(f"[WARNING] File not found, skipping: {p}", file=sys.stderr)
            continue
        try:
            results.append(_BenchmarkResultStub(p))
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[ERROR] Could not parse {p}: {exc}", file=sys.stderr)
            sys.exit(1)

    if not results:
        print("[ERROR] No valid result files loaded.", file=sys.stderr)
        sys.exit(1)

    advisor = DeploymentAdvisor(memory_safety_factor=args.memory_safety)

    if args.all_profiles:
        profiles = list_profiles()
        summary = advisor.multi_profile_summary(results, profiles)
        report = advisor.report_multi_profile_markdown(summary, profiles)
        # Append per-device tables
        per_device_parts: List[str] = []
        for p in profiles:
            ranked = summary[p.name]
            per_device_parts.append(advisor.report_markdown(ranked, p))
        report = report + "\n\n---\n\n" + "\n\n---\n\n".join(per_device_parts)
    else:
        if args.profile not in PROFILES:
            print(
                f"[ERROR] Unknown profile {args.profile!r}. "
                f"Available: {', '.join(sorted(PROFILES))}",
                file=sys.stderr,
            )
            sys.exit(1)
        profile = PROFILES[args.profile]
        ranked = advisor.rank(results, profile)
        report = advisor.report_markdown(ranked, profile)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"Report written to {out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
