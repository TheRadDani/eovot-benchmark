"""Command-line interface for ``python -m eovot``.

Provides a unified entry point with subcommands for running benchmarks,
quick demos, listing trackers, and comparing algorithms side by side.

Usage::

    # List all available trackers
    python -m eovot list

    # Demo on synthetic data (no dataset download required)
    python -m eovot demo --trackers MOSSE KCF --sequences 5 --frames 100

    # Demo with robustness and temporal consistency metrics
    python -m eovot demo --trackers MOSSE KCF --robustness --temporal --html

    # Run a full experiment from a YAML config
    python -m eovot run --config configs/default.yaml

    # Compare trackers on synthetic data and print Markdown table
    python -m eovot compare --trackers MOSSE KCF CSRT --sequences 10
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch ``python -m eovot <subcommand>`` calls."""
    parser = argparse.ArgumentParser(
        prog="python -m eovot",
        description="EOVOT — Edge-Optimized Visual Object Tracking Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "subcommands:\n"
            "  run      Run a benchmark from a YAML config file\n"
            "  demo     Quick demo on synthetic data (no download needed)\n"
            "  list     List all available trackers\n"
            "  compare  Compare multiple trackers and print a Markdown table\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ------------------------------------------------------------------ run
    p_run = sub.add_parser(
        "run",
        help="Run a benchmark from a YAML config file",
        description=(
            "Load a YAML experiment config and run the full benchmark pipeline. "
            "Equivalent to: python scripts/run_experiment.py --config <file>"
        ),
    )
    p_run.add_argument(
        "--config", required=True, metavar="FILE",
        help="Path to YAML experiment config (see configs/default.yaml for schema)",
    )
    p_run.add_argument(
        "--output-dir", default="results/", metavar="DIR",
        help="Root directory for output files (default: results/)",
    )
    p_run.add_argument(
        "--resume", action="store_true",
        help="Skip trackers whose results already exist in output-dir",
    )

    # ------------------------------------------------------------------ demo
    p_demo = sub.add_parser(
        "demo",
        help="Quick demo on procedurally-generated sequences (no data download needed)",
        description=(
            "Benchmark one or more trackers on a synthetic moving-rectangle dataset. "
            "No external data download required — ideal for CI and quick validation."
        ),
    )
    p_demo.add_argument(
        "--trackers", nargs="+", default=["MOSSE", "KCF"], metavar="TRACKER",
        help="Tracker names to evaluate (default: MOSSE KCF)",
    )
    p_demo.add_argument(
        "--sequences", type=int, default=5,
        help="Number of synthetic sequences to generate (default: 5)",
    )
    p_demo.add_argument(
        "--frames", type=int, default=100,
        help="Frames per sequence (default: 100)",
    )
    p_demo.add_argument(
        "--motion", default="linear", choices=["linear", "circular", "random"],
        help="Synthetic target motion pattern (default: linear)",
    )
    p_demo.add_argument(
        "--output-dir", default="results/demo/", metavar="DIR",
        help="Output directory (default: results/demo/)",
    )
    p_demo.add_argument(
        "--robustness", action="store_true",
        help="Include robustness analysis: failure counts, EAO, survival rate",
    )
    p_demo.add_argument(
        "--temporal", action="store_true",
        help="Include temporal consistency analysis: jitter, smoothness score",
    )
    p_demo.add_argument(
        "--html", action="store_true",
        help="Generate a self-contained HTML report alongside JSON/CSV",
    )

    # ------------------------------------------------------------------ list
    sub.add_parser(
        "list",
        help="List all trackers available in the EOVOT registry",
        description="Print all tracker names that can be passed to --trackers.",
    )

    # ------------------------------------------------------------------ compare
    p_cmp = sub.add_parser(
        "compare",
        help="Compare multiple trackers on synthetic data and print a Markdown table",
        description=(
            "Benchmark multiple trackers on a synthetic dataset and output "
            "a side-by-side Markdown comparison table ready for papers or READMEs."
        ),
    )
    p_cmp.add_argument(
        "--trackers", nargs="+", required=True, metavar="TRACKER",
        help="Two or more tracker names to compare",
    )
    p_cmp.add_argument(
        "--sequences", type=int, default=5,
        help="Number of synthetic sequences (default: 5)",
    )
    p_cmp.add_argument(
        "--frames", type=int, default=100,
        help="Frames per sequence (default: 100)",
    )
    p_cmp.add_argument(
        "--output-dir", default="results/compare/", metavar="DIR",
        help="Output directory for JSON/CSV/Markdown files (default: results/compare/)",
    )
    p_cmp.add_argument(
        "--robustness", action="store_true",
        help="Include robustness metrics (EAO, failure counts)",
    )
    p_cmp.add_argument(
        "--temporal", action="store_true",
        help="Include temporal consistency metrics (smoothness score)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "list":
        return _cmd_list()
    if args.command == "compare":
        return _cmd_compare(args)

    parser.print_help()
    return 1


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _cmd_list() -> int:
    """Print all tracker names registered in TRACKER_REGISTRY."""
    from eovot.trackers.registry import available_trackers
    print("Available trackers:")
    for name in available_trackers():
        print(f"  {name}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run an experiment from a YAML config file via ExperimentRunner."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required.  Install it with:  pip install pyyaml")
        return 1

    from eovot.experiment.runner import ExperimentRunner

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config}")
        return 1

    runner = ExperimentRunner(output_dir=args.output_dir, resume=args.resume)
    runner.run_from_config(config)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Benchmark trackers on procedurally-generated synthetic sequences."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.reporting.reporter import BenchmarkReporter
    from eovot.trackers.registry import available_trackers, build_tracker

    dataset = SyntheticDataset(
        num_sequences=args.sequences,
        num_frames=args.frames,
        motion=args.motion,
    )
    engine = BenchmarkEngine(
        verbose=True,
        include_robustness=args.robustness,
        include_temporal=args.temporal,
    )
    reporter = BenchmarkReporter(output_dir=args.output_dir)

    results = []
    valid = set(available_trackers())
    for name in args.trackers:
        if name not in valid:
            print(f"[WARN] Unknown tracker '{name}'. Available: {sorted(valid)}")
            continue
        tracker = build_tracker(name)
        result = engine.run(tracker, dataset, dataset_name=f"Synthetic-{args.motion}")
        d = result.to_dict()
        results.append(d)
        saved = reporter.save_all(d, name=f"demo-{name}")
        for fmt, path in saved.items():
            print(f"  [{fmt.upper()}] {path}")

    if len(results) > 1:
        cmp_path = reporter.save_comparison(results, name="demo-comparison")
        print(f"\n[COMPARISON TABLE] → {cmp_path}")
        print(reporter.comparison_table(results))

    if args.html and results:
        html_path = reporter.save_html(
            results, name="demo-report", title="EOVOT Demo Report"
        )
        print(f"[HTML REPORT] → {html_path}")

    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare trackers on synthetic data and print a Markdown table."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.reporting.reporter import BenchmarkReporter
    from eovot.trackers.registry import available_trackers, build_tracker

    dataset = SyntheticDataset(num_sequences=args.sequences, num_frames=args.frames)
    engine = BenchmarkEngine(
        verbose=True,
        include_robustness=args.robustness,
        include_temporal=args.temporal,
    )
    reporter = BenchmarkReporter(output_dir=args.output_dir)

    results = []
    valid = set(available_trackers())
    for name in args.trackers:
        if name not in valid:
            print(f"[WARN] Unknown tracker '{name}'. Available: {sorted(valid)}")
            continue
        tracker = build_tracker(name)
        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        d = result.to_dict()
        results.append(d)
        reporter.save_all(d, name=f"compare-{name}")

    if len(results) > 1:
        cmp_path = reporter.save_comparison(results, name="tracker-comparison")
        print(f"\n[COMPARISON TABLE] saved → {cmp_path}")
        print(reporter.comparison_table(results))
    elif results:
        print("Need at least 2 trackers to compare. Pass: --trackers A B [C ...]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
