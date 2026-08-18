"""EOVOT unified command-line interface.

Provides a single ``python -m eovot`` entry point with subcommands that
replace the ad-hoc scripts in ``scripts/``.  Designed for zero-friction
experimentation: every subcommand defaults to the built-in
:class:`~eovot.datasets.synthetic.SyntheticDataset` so no dataset download
is needed.

Subcommands
-----------
``run``
    Benchmark one tracker on a dataset and print a summary.
``compare``
    Benchmark multiple trackers on the same dataset and print a ranked table.
``report``
    Generate a Markdown or JSON report from a saved result file.
``bank``
    List or compare results stored in a :class:`~eovot.results.bank.ResultsBank`.
``list``
    Print all registered tracker names.

Usage::

    python -m eovot --help
    python -m eovot list
    python -m eovot run --tracker MOSSE
    python -m eovot run --tracker KCF --sequences 3 --frames 50
    python -m eovot compare --trackers MOSSE KCF CSRT
    python -m eovot compare --trackers MOSSE KCF --output results/
    python -m eovot report --input results/KCF-synthetic.json
    python -m eovot bank --bank results/bank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(sequences: int, frames: int, motion: str):
    from eovot.datasets.synthetic import SyntheticDataset
    return SyntheticDataset(
        num_sequences=sequences,
        num_frames=frames,
        motion=motion,
    )


def _build_real_dataset(loader: str, root: str, split: str, max_seq: Optional[int]):
    from eovot.datasets.base import OTBDataset
    from eovot.datasets.got10k import GOT10kDataset
    from eovot.datasets.lasot import LaSOTDataset

    registry = {
        "OTBDataset": OTBDataset,
        "GOT10kDataset": GOT10kDataset,
        "LaSOTDataset": LaSOTDataset,
    }
    cls = registry.get(loader)
    if cls is None:
        print(f"[eovot] Unknown dataset loader '{loader}'. Choices: {list(registry)}", file=sys.stderr)
        sys.exit(1)
    if loader in ("GOT10kDataset", "LaSOTDataset"):
        return cls(root=root, split=split, max_sequences=max_seq)
    return cls(root=root)


def _build_tracker(name: str):
    from eovot.trackers.registry import build_tracker, available_trackers
    try:
        return build_tracker(name)
    except ValueError:
        print(
            f"[eovot] Unknown tracker '{name}'. "
            f"Available: {available_trackers()}",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_divider(char: str = "─", width: int = 70) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(_args: argparse.Namespace) -> None:
    """Print all registered tracker names."""
    from eovot.trackers.registry import available_trackers
    names = available_trackers()
    print(f"\nEOVOT registered trackers ({len(names)} total):\n")
    for name in names:
        print(f"  • {name}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Benchmark a single tracker and optionally save the result."""
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = _build_tracker(args.tracker)

    if args.dataset_root:
        dataset = _build_real_dataset(
            args.dataset_loader, args.dataset_root, args.split, args.sequences
        )
        dataset_name = args.dataset_name or args.dataset_loader
    else:
        dataset = _build_synthetic_dataset(args.sequences, args.frames, args.motion)
        dataset_name = f"Synthetic-{args.motion}"

    engine = BenchmarkEngine(
        verbose=not args.quiet,
        tdp_watts=args.tdp_watts,
    )
    result = engine.run(
        tracker=tracker,
        dataset=dataset,
        dataset_name=dataset_name,
        max_sequences=args.sequences if args.dataset_root else None,
    )

    _print_divider()
    print(result)
    _print_divider()

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{args.tracker}-{dataset_name}"
        saved = result.save(out_dir / run_name)
        print(f"\n[saved] {saved}")

        if args.bank:
            from eovot.results.bank import ResultsBank
            bank = ResultsBank(args.bank)
            bank.save(result)
            print(f"[bank]  stored → {args.bank}")

    summary = result.summary()
    _print_summary_table([summary])


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Benchmark multiple trackers and print a ranked comparison."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.reporting.reporter import BenchmarkReporter
    from eovot.metrics.efficiency import EfficiencyMetricsEngine

    if args.dataset_root:
        dataset_name = args.dataset_name or args.dataset_loader
    else:
        dataset_name = f"Synthetic-{args.motion}"

    engine = BenchmarkEngine(verbose=not args.quiet, tdp_watts=args.tdp_watts)
    eff_engine = EfficiencyMetricsEngine(memory_budget_mb=args.memory_budget)
    reporter = BenchmarkReporter(output_dir=args.output) if args.output else None

    results = []
    summaries = []

    for tracker_name in args.trackers:
        tracker = _build_tracker(tracker_name)

        if args.dataset_root:
            dataset = _build_real_dataset(
                args.dataset_loader, args.dataset_root, args.split, args.sequences
            )
        else:
            dataset = _build_synthetic_dataset(args.sequences, args.frames, args.motion)

        _print_divider()
        result = engine.run(
            tracker=tracker,
            dataset=dataset,
            dataset_name=dataset_name,
            max_sequences=args.sequences if args.dataset_root else None,
        )
        results.append(result)
        summaries.append(result.summary())

        if reporter:
            run_name = f"{tracker_name}-{dataset_name}"
            reporter.save_all(result.to_dict(), name=run_name)

        if args.bank:
            from eovot.results.bank import ResultsBank
            ResultsBank(args.bank).save(result)

    # Ranked comparison
    _print_divider("═")
    print("\nComparison (ranked by Edge Efficiency Score)\n")
    entries = eff_engine.rank_trackers(results)
    print(eff_engine.to_markdown_table(entries))
    print()
    _print_summary_table(summaries)

    if reporter and len(results) > 1:
        cmp_path = reporter.save_comparison(
            [r.to_dict() for r in results],
            name=f"comparison-{dataset_name}",
        )
        print(f"\n[comparison] saved → {cmp_path}")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    """Generate a Markdown or JSON report from a saved result file."""
    from eovot.benchmark.engine import BenchmarkResult

    path = Path(args.input)
    if not path.exists():
        print(f"[eovot] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    result = BenchmarkResult.load(path)
    summary = result.summary()

    _print_divider("═")
    print(f"\nBenchmark Report — {summary['tracker']} on {summary['dataset']}\n")
    _print_divider()

    _print_summary_table([summary])

    print(f"\nSequences ({summary['num_sequences']}):")
    _print_divider()
    for sr in result.sequence_results:
        iou_str = f"mIoU={sr.mean_iou:.4f}"
        fps_str = f"FPS={sr.profiling.fps:.1f}"
        mem_str = f"mem={sr.profiling.peak_memory_mb:.0f}MB"
        auc_str = ""
        if sr.accuracy_metrics:
            auc_str = f"  AUC={sr.accuracy_metrics.success_auc:.3f}"
        print(f"  {sr.sequence_name:<30s}  {iou_str}  {fps_str}  {mem_str}{auc_str}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            out.write_text(json.dumps(result.to_dict(), indent=2))
        else:
            from eovot.reporting.reporter import BenchmarkReporter
            rp = BenchmarkReporter(output_dir=str(out.parent))
            saved = rp.save_all(result.to_dict(), name=out.stem)
            for fmt, p in saved.items():
                print(f"\n[{fmt}] → {p}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: bank
# ---------------------------------------------------------------------------

def cmd_bank(args: argparse.Namespace) -> None:
    """List or compare results in a ResultsBank directory."""
    from eovot.results.bank import ResultsBank

    bank = ResultsBank(args.bank)

    if args.list:
        entries = bank.list_results(dataset=args.dataset)
        if not entries:
            print("No results found.")
            return
        print(f"\n{'Tracker':<20s}  {'Dataset':<15s}  {'mIoU':>6s}  {'FPS':>7s}  Saved At")
        _print_divider()
        for e in entries:
            print(
                f"{e['tracker']:<20s}  {e['dataset']:<15s}  "
                f"{e['mean_iou']:>6.4f}  {e['mean_fps']:>7.1f}  {e['timestamp']}"
            )
        print()
        return

    table = bank.compare(dataset=args.dataset, sort_by=args.sort_by)
    print("\n# EOVOT Benchmark Comparison\n")
    print(table)

    if args.output:
        Path(args.output).write_text("# EOVOT Benchmark Comparison\n\n" + table + "\n")
        print(f"\n[saved] {args.output}")


# ---------------------------------------------------------------------------
# Shared formatting helper
# ---------------------------------------------------------------------------

def _print_summary_table(summaries: List[dict]) -> None:
    """Print a compact aligned summary table for one or more runs."""
    if not summaries:
        return

    headers = ["Tracker", "Dataset", "mIoU", "FPS", "Mem(MB)"]
    extra_cols = []
    if any("success_auc" in s for s in summaries):
        extra_cols.append("Succ.AUC")
        headers.append("Succ.AUC")
    if any("precision_auc" in s for s in summaries):
        extra_cols.append("Prec.AUC")
        headers.append("Prec.AUC")

    col_w = [max(len(h), 8) for h in headers]
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print("\n" + header_row)
    _print_divider("─", len(header_row))
    for s in summaries:
        row = [
            str(s.get("tracker", "")).ljust(col_w[0]),
            str(s.get("dataset", "")).ljust(col_w[1]),
            f"{s.get('mean_iou', 0.0):.4f}".rjust(col_w[2]),
            f"{s.get('mean_fps', 0.0):.1f}".rjust(col_w[3]),
            f"{s.get('peak_memory_mb', 0.0):.1f}".rjust(col_w[4]),
        ]
        for col in extra_cols:
            key = col.lower().replace(".", "_").replace("succ_auc", "success_auc").replace("prec_auc", "precision_auc")
            val = s.get(key)
            row.append((f"{val:.4f}" if val is not None else "N/A").rjust(8))
        print("  ".join(row))
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eovot",
        description=(
            "EOVOT — Edge-Optimized Visual Object Tracking Benchmark Suite\n\n"
            "Zero-config quickstart (uses built-in synthetic dataset):\n"
            "  python -m eovot list\n"
            "  python -m eovot run --tracker MOSSE\n"
            "  python -m eovot compare --trackers MOSSE KCF CSRT\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ------------------------------------------------------------------ list
    sub.add_parser("list", help="Print all registered tracker names.")

    # ------------------------------------------------------------------ run
    run_p = sub.add_parser("run", help="Benchmark one tracker on a dataset.")
    _add_tracker_arg(run_p)
    _add_dataset_args(run_p)
    _add_output_args(run_p)
    _add_profiling_args(run_p)

    # --------------------------------------------------------------- compare
    cmp_p = sub.add_parser("compare", help="Benchmark multiple trackers and compare.")
    cmp_p.add_argument(
        "--trackers", "-t", nargs="+", required=True, metavar="TRACKER",
        help="Two or more tracker names (e.g. MOSSE KCF CSRT).",
    )
    cmp_p.add_argument(
        "--memory-budget", type=float, default=512.0, metavar="MB",
        help="Memory budget (MB) used in the Edge Efficiency Score denominator.",
    )
    _add_dataset_args(cmp_p)
    _add_output_args(cmp_p)
    _add_profiling_args(cmp_p)

    # --------------------------------------------------------------- report
    rpt_p = sub.add_parser("report", help="Generate report from a saved JSON result.")
    rpt_p.add_argument("--input", "-i", required=True, metavar="FILE",
                       help="Path to a JSON file saved by `run`.")
    rpt_p.add_argument("--output", "-o", metavar="PATH",
                       help="Write report to this path (optional).")
    rpt_p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Output format (default: markdown).")

    # --------------------------------------------------------------- bank
    bnk_p = sub.add_parser("bank", help="List or compare results in a ResultsBank directory.")
    bnk_p.add_argument("--bank", "-b", required=True, metavar="DIR",
                       help="Path to the ResultsBank directory.")
    bnk_p.add_argument("--list", action="store_true",
                       help="List stored results and exit.")
    bnk_p.add_argument("--dataset", "-d", metavar="NAME",
                       help="Filter to this dataset name (substring match).")
    bnk_p.add_argument("--sort-by", default="mean_iou",
                       choices=["mean_iou", "mean_fps", "success_auc", "precision_auc"],
                       help="Sort column for the comparison table.")
    bnk_p.add_argument("--output", "-o", metavar="FILE",
                       help="Write comparison table to this Markdown file.")

    return parser


def _add_tracker_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tracker", "-t", default="MOSSE", metavar="NAME",
                   help="Tracker name (default: MOSSE). Use `list` to see all.")


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    grp = p.add_argument_group("dataset")
    grp.add_argument("--sequences", "-n", type=int, default=5, metavar="N",
                     help="Number of sequences to evaluate (default: 5).")
    grp.add_argument("--frames", type=int, default=100, metavar="N",
                     help="Frames per sequence for synthetic dataset (default: 100).")
    grp.add_argument("--motion", default="linear",
                     choices=["linear", "circular", "random"],
                     help="Synthetic motion pattern (default: linear).")
    grp.add_argument("--dataset-root", metavar="PATH",
                     help="Path to a real dataset root (OTB, GOT-10k, LaSOT). "
                          "If omitted, the synthetic dataset is used.")
    grp.add_argument("--dataset-loader", default="OTBDataset",
                     choices=["OTBDataset", "GOT10kDataset", "LaSOTDataset"],
                     help="Dataset loader class (default: OTBDataset).")
    grp.add_argument("--dataset-name", metavar="NAME",
                     help="Override the dataset label used in reports.")
    grp.add_argument("--split", default="val",
                     help="Dataset split (GOT-10k: train|val|test; LaSOT: train|test|all).")


def _add_output_args(p: argparse.ArgumentParser) -> None:
    grp = p.add_argument_group("output")
    grp.add_argument("--output", "-o", metavar="DIR",
                     help="Directory to save JSON / Markdown results.")
    grp.add_argument("--bank", metavar="DIR",
                     help="ResultsBank directory; result is stored here after each run.")
    grp.add_argument("--quiet", "-q", action="store_true",
                     help="Suppress per-sequence progress output.")


def _add_profiling_args(p: argparse.ArgumentParser) -> None:
    grp = p.add_argument_group("profiling")
    grp.add_argument("--tdp-watts", type=float, metavar="W",
                     help="Enable CPU energy estimation using this TDP value (e.g. 6.0 "
                          "for Raspberry Pi 4, 15.0 for a laptop).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "list": cmd_list,
    "run": cmd_run,
    "compare": cmd_compare,
    "report": cmd_report,
    "bank": cmd_bank,
}


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    COMMANDS[args.command](args)
