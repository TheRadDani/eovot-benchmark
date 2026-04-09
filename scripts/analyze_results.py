#!/usr/bin/env python3
"""Edge deployment leaderboard and Pareto analysis CLI.

Loads one or more benchmark JSON result files produced by EOVOT's
BenchmarkReporter, identifies the Pareto-optimal tracker set across
accuracy / speed / memory objectives, computes per-tracker composite
edge scores, and outputs a ranked leaderboard.

Usage examples::

    # Console leaderboard from a single result file
    python scripts/analyze_results.py results/comparison.json

    # Multiple files (one tracker per file)
    python scripts/analyze_results.py results/mosse.json results/kcf.json results/csrt.json

    # Load all JSONs from a directory
    python scripts/analyze_results.py results/

    # Raspberry Pi 4 profile (FPS target 15, 256 MB budget)
    python scripts/analyze_results.py results/ --fps-target 15 --memory-budget 256

    # Markdown export for research paper table
    python scripts/analyze_results.py results/ --format markdown --output leaderboard.md

    # JSON export for downstream tooling
    python scripts/analyze_results.py results/ --format json --output leaderboard.json

    # Custom scoring weights (accuracy-focused)
    python scripts/analyze_results.py results/ --w-accuracy 0.6 --w-speed 0.2 --w-memory 0.15 --w-energy 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.analysis.pareto import ParetoAnalyzer, TrackerProfile  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_results",
        description="EOVOT edge deployment leaderboard and Pareto analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "results",
        nargs="+",
        metavar="PATH",
        help="Benchmark JSON file(s) or directory containing JSON files",
    )

    # --- Hardware budget ---
    hw = p.add_argument_group("hardware budget")
    hw.add_argument(
        "--fps-target",
        type=float,
        default=30.0,
        metavar="FPS",
        help="Real-time FPS target for normalisation (default: 30.0)",
    )
    hw.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Peak memory budget in MB (default: 512.0)",
    )
    hw.add_argument(
        "--energy-budget",
        type=float,
        default=100.0,
        metavar="mJ",
        help="Energy-per-frame budget in milli-Joules (default: 100.0)",
    )

    # --- Scoring weights ---
    sw = p.add_argument_group("scoring weights (must sum to ≤1; remainder goes to accuracy)")
    sw.add_argument("--w-accuracy", type=float, default=0.40, metavar="W",
                    help="Weight for accuracy / IoU (default: 0.40)")
    sw.add_argument("--w-speed", type=float, default=0.30, metavar="W",
                    help="Weight for throughput / FPS (default: 0.30)")
    sw.add_argument("--w-memory", type=float, default=0.20, metavar="W",
                    help="Weight for memory efficiency (default: 0.20)")
    sw.add_argument("--w-energy", type=float, default=0.10, metavar="W",
                    help="Weight for energy efficiency (default: 0.10)")

    # --- Output ---
    out = p.add_argument_group("output")
    out.add_argument(
        "--format",
        choices=["console", "markdown", "json"],
        default="console",
        help="Output format (default: console)",
    )
    out.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write output to FILE instead of stdout",
    )

    return p


def _collect_json_files(paths: list[str]) -> list[Path]:
    """Expand path arguments: files are used directly, directories are globbed."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: {raw} does not exist, skipping.", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    analyzer = ParetoAnalyzer(
        fps_target=args.fps_target,
        memory_budget_mb=args.memory_budget,
        energy_budget_mj=args.energy_budget,
        w_accuracy=args.w_accuracy,
        w_speed=args.w_speed,
        w_memory=args.w_memory,
        w_energy=args.w_energy,
    )

    # Collect and load all JSON files
    json_files = _collect_json_files(args.results)
    if not json_files:
        print("Error: no JSON files found.", file=sys.stderr)
        return 1

    all_profiles: list[TrackerProfile] = []
    for jf in json_files:
        try:
            profiles = analyzer.load_from_json(str(jf))
            all_profiles.extend(profiles)
            print(f"Loaded {len(profiles)} tracker(s) from {jf.name}", file=sys.stderr)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"Warning: could not parse {jf}: {exc}", file=sys.stderr)

    if not all_profiles:
        print("Error: no tracker profiles loaded.", file=sys.stderr)
        return 1

    # Deduplicate: keep last-seen profile per tracker name
    seen: dict[str, TrackerProfile] = {}
    for p in all_profiles:
        seen[p.name] = p
    profiles = list(seen.values())

    print(f"\nAnalysing {len(profiles)} tracker(s)…", file=sys.stderr)
    result = analyzer.analyze(profiles)

    # Render output
    if args.format == "console":
        analyzer.print_leaderboard(result)
        text: str | None = None
    elif args.format == "markdown":
        text = analyzer.to_markdown(result)
    elif args.format == "json":
        text = analyzer.to_json(result)
    else:
        text = None

    if text is not None:
        if args.output:
            Path(args.output).write_text(text)
            print(f"Leaderboard written to {args.output}", file=sys.stderr)
        else:
            print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
