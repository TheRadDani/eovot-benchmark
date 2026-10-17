"""Entry point for ``python -m eovot``.

Provides a unified command-line interface for the EOVOT benchmark suite.

Subcommands
-----------
run
    Load a YAML experiment config and execute a multi-tracker benchmark run.
    Wraps :class:`~eovot.experiment.runner.ExperimentRunner` so you get
    resumable, reproducible experiments with a single command.

compare
    Run two or more trackers on a dataset side-by-side and print a Markdown
    comparison table.  A thin convenience wrapper around
    :class:`~eovot.benchmark.engine.BenchmarkEngine` and
    :class:`~eovot.reporting.reporter.BenchmarkReporter`.

list-trackers
    Print every tracker registered in :data:`~eovot.trackers.registry.TRACKER_REGISTRY`
    together with its class path.

info
    Print EOVOT version, Python version, and optional dependency availability.

Examples
--------
Run an experiment from a YAML config::

    python -m eovot run configs/experiments/multi_tracker.yaml
    python -m eovot run configs/experiments/classical_comparison.yaml --resume --quiet

Quick tracker comparison (5 sequences)::

    python -m eovot compare \\
        --trackers MOSSE KCF CSRT \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 5

List registered trackers::

    python -m eovot list-trackers

Show installed versions::

    python -m eovot info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    """Execute a YAML-defined multi-tracker experiment."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required.  Install with: pip install pyyaml", file=sys.stderr)
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    if args.dry_run:
        _print_experiment_plan(config)
        return 0

    from eovot.experiment.runner import ExperimentRunner

    output_dir = args.output_dir or "results/experiments"
    runner = ExperimentRunner(
        output_dir=output_dir,
        verbose=not args.quiet,
        resume=args.resume,
    )

    output = runner.run_from_config(config)

    if args.json:
        print(json.dumps(output["metadata"], indent=2))

    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Compare multiple trackers on a dataset and print a Markdown table."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.reporting.reporter import BenchmarkReporter
    from eovot.trackers.registry import TRACKER_REGISTRY, build_tracker

    unknown = [t for t in args.trackers if t not in TRACKER_REGISTRY]
    if unknown:
        print(
            f"ERROR: Unknown tracker(s): {unknown}\n"
            f"Available: {sorted(TRACKER_REGISTRY)}",
            file=sys.stderr,
        )
        return 1

    dataset = _build_dataset(
        loader_name=args.dataset_loader,
        root=args.dataset_root,
        split=args.split,
        max_sequences=args.max_sequences,
    )
    if dataset is None:
        return 1

    dataset_name = args.dataset_name or args.dataset_loader
    engine = BenchmarkEngine(verbose=not args.quiet, tdp_watts=args.tdp_watts)
    reporter = BenchmarkReporter(output_dir=args.output_dir)
    all_results = []

    for tracker_name in args.trackers:
        if not args.quiet:
            print(f"\n[{tracker_name}] evaluating on {dataset_name} …")
        tracker = build_tracker(tracker_name)
        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=args.max_sequences,
        )
        result_dict = result.to_dict()
        if not args.quiet:
            reporter.print_summary(result_dict)
        run_name = f"{tracker_name}-{dataset_name}"
        reporter.save_all(result_dict, name=run_name)
        all_results.append(result_dict)

        # Re-instantiate dataset for the next tracker (stateless iterators).
        dataset = _build_dataset(
            loader_name=args.dataset_loader,
            root=args.dataset_root,
            split=args.split,
            max_sequences=args.max_sequences,
        )

    if len(all_results) > 1:
        table = reporter.comparison_table(all_results)
        cmp_path = reporter.save_comparison(all_results, name=f"comparison-{dataset_name}")
        print("\n" + table)
        if not args.quiet:
            print(f"\n[saved] {cmp_path}")

    return 0


def _cmd_list_trackers(_args: argparse.Namespace) -> int:
    """Print every registered tracker."""
    from eovot.trackers.registry import TRACKER_REGISTRY

    print(f"\nEOVOT Tracker Registry  ({len(TRACKER_REGISTRY)} trackers)\n")
    print(f"  {'Name':<18s}  Class")
    print(f"  {'-'*18}  {'-'*40}")
    for name, cls in sorted(TRACKER_REGISTRY.items()):
        print(f"  {name:<18s}  {cls.__module__}.{cls.__name__}")
    print()
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    """Print EOVOT version and system info."""
    import platform

    import numpy as np
    import cv2

    import eovot

    print(f"\nEOVOT  {eovot.__version__}")
    print(f"  Python   {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"  NumPy    {np.__version__}")
    print(f"  OpenCV   {cv2.__version__}")

    _optional("torch", "PyTorch")
    _optional("onnxruntime", "ONNX Runtime")
    _optional("psutil", "psutil")
    _optional("yaml", "PyYAML")

    from eovot.trackers.registry import TRACKER_REGISTRY
    print(f"\n  Trackers: {', '.join(sorted(TRACKER_REGISTRY))}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optional(module: str, label: str) -> None:
    try:
        import importlib
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "?")
        print(f"  {label:<12s} {ver}")
    except ImportError:
        print(f"  {label:<12s} not installed")


def _build_dataset(loader_name: str, root: str, split: str, max_sequences):
    """Instantiate a dataset; print an error and return None on failure."""
    try:
        from eovot.datasets.base import OTBDataset
        from eovot.datasets.got10k import GOT10kDataset
        from eovot.datasets.lasot import LaSOTDataset
        from eovot.datasets.synthetic import SyntheticDataset
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None

    loaders = {
        "OTBDataset": OTBDataset,
        "GOT10kDataset": GOT10kDataset,
        "LaSOTDataset": LaSOTDataset,
        "SyntheticDataset": SyntheticDataset,
    }
    cls = loaders.get(loader_name)
    if cls is None:
        print(
            f"ERROR: Unknown dataset loader '{loader_name}'.\n"
            f"Available: {list(loaders)}",
            file=sys.stderr,
        )
        return None

    try:
        if loader_name == "OTBDataset":
            return cls(root=root)
        if loader_name == "SyntheticDataset":
            return cls(num_sequences=10, num_frames=100)
        return cls(root=root, split=split, max_sequences=max_sequences)
    except Exception as exc:
        print(f"ERROR loading dataset: {exc}", file=sys.stderr)
        return None


def _print_experiment_plan(config: dict) -> None:
    """Print a dry-run summary of an experiment config."""
    exp = config.get("experiment", {})
    ds = config.get("dataset", {})
    trackers = config.get("trackers", [])

    print("\nEOVOT Experiment Plan (dry run)")
    print("=" * 48)
    print(f"  Name       : {exp.get('name', 'unnamed')}")
    print(f"  Dataset    : {ds.get('name', ds.get('loader', '?'))} @ {ds.get('root', '?')}")
    print(f"  Max seqs   : {ds.get('max_sequences', 'all')}")
    print(f"  TDP watts  : {exp.get('tdp_watts', 'disabled')}")
    print(f"  Trackers   :")
    for t in trackers:
        params = t.get("params") or {}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "defaults"
        print(f"      {t['name']:<16s} [{param_str}]")
    print("=" * 48 + "\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eovot",
        description="EOVOT — Edge-Optimized Visual Object Tracking Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m eovot list-trackers\n"
            "  python -m eovot run configs/experiments/multi_tracker.yaml\n"
            "  python -m eovot compare --trackers MOSSE KCF --dataset-root /data/OTB100 --max-sequences 5\n"
            "  python -m eovot info\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- run ----------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run a YAML experiment config",
        description="Load an experiment YAML and execute a multi-tracker benchmark run.",
    )
    p_run.add_argument("config", help="Path to experiment YAML config file")
    p_run.add_argument("--output-dir", default=None, help="Override output directory")
    p_run.add_argument("--resume", action="store_true", help="Skip trackers with existing results")
    p_run.add_argument("--quiet", action="store_true", help="Suppress per-sequence progress")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan without running")
    p_run.add_argument("--json", action="store_true", help="Print metadata as JSON on stdout")
    p_run.set_defaults(func=_cmd_run)

    # ---- compare ------------------------------------------------------------
    p_cmp = sub.add_parser(
        "compare",
        help="Compare trackers side-by-side on a dataset",
        description="Run two or more trackers on a dataset and print a Markdown comparison table.",
    )
    p_cmp.add_argument("--trackers", nargs="+", required=True, metavar="TRACKER")
    p_cmp.add_argument("--dataset-root", required=True, help="Path to dataset root directory")
    p_cmp.add_argument(
        "--dataset-loader",
        default="OTBDataset",
        choices=["OTBDataset", "GOT10kDataset", "LaSOTDataset", "SyntheticDataset"],
        help="Dataset loader class (default: OTBDataset)",
    )
    p_cmp.add_argument("--dataset-name", default=None, help="Label used in reports")
    p_cmp.add_argument("--split", default="val", help="Dataset split (GOT10k / LaSOT)")
    p_cmp.add_argument("--max-sequences", type=int, default=None, metavar="N")
    p_cmp.add_argument("--output-dir", default="results/", help="Directory for output files")
    p_cmp.add_argument("--tdp-watts", type=float, default=None, metavar="W",
                       help="Enable energy estimation with this device TDP (Watts)")
    p_cmp.add_argument("--quiet", action="store_true")
    p_cmp.set_defaults(func=_cmd_compare)

    # ---- list-trackers ------------------------------------------------------
    p_list = sub.add_parser(
        "list-trackers",
        help="List all registered trackers",
    )
    p_list.set_defaults(func=_cmd_list_trackers)

    # ---- info ---------------------------------------------------------------
    p_info = sub.add_parser(
        "info",
        help="Show EOVOT version and system info",
    )
    p_info.set_defaults(func=_cmd_info)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
