"""CLI tool: per-attribute tracker performance analysis.

Reads one or more JSON result files produced by the EOVOT experiment runner,
loads the corresponding OTB attribute annotations, and generates a Markdown
attribute-breakdown report.

Usage::

    python scripts/analyze_attributes.py \\
        --results results/mosse.json results/kcf.json \\
        --output attribute_report.md \\
        --dataset-name OTB-100 \\
        [--attributes FM OCC SV]

The ``--results`` files must be in the format written by
:class:`eovot.reporting.reporter.BenchmarkReporter`.

The ``--attributes`` flag optionally restricts analysis to a subset of the
11 OTB challenge attributes.  When omitted, all attributes present in the
result files are analysed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Minimal BenchmarkResult reconstruction from saved JSON
# (avoids requiring the actual dataset files at analysis time)
# ---------------------------------------------------------------------------

class _SequenceResult:
    """Lightweight reconstruction of SequenceResult from saved JSON."""

    def __init__(self, sequence_name: str, ious: List[float]) -> None:
        self.sequence_name = sequence_name
        self.ious = np.array(ious, dtype=np.float64)


class _BenchmarkResult:
    """Lightweight reconstruction of BenchmarkResult from saved JSON."""

    def __init__(self, tracker_name: str, sequence_results: List[_SequenceResult]) -> None:
        self.tracker_name = tracker_name
        self.sequence_results = sequence_results


def _load_result(path: Path) -> _BenchmarkResult:
    """Parse a JSON result file into a lightweight BenchmarkResult."""
    with open(path) as fh:
        data = json.load(fh)

    tracker_name = data.get("tracker_name", path.stem)
    seq_results: List[_SequenceResult] = []

    for seq_data in data.get("sequence_results", []):
        name = seq_data.get("sequence_name", "unknown")
        ious = seq_data.get("ious", [])
        seq_results.append(_SequenceResult(name, ious))

    return _BenchmarkResult(tracker_name=tracker_name, sequence_results=seq_results)


# ---------------------------------------------------------------------------
# Attribute map bootstrap — works without OTBDataset
# ---------------------------------------------------------------------------

def _otb_attribute_map() -> Dict[str, FrozenSet[str]]:
    """Return the OTB-100 sequence→attribute mapping without loading image data."""
    try:
        from eovot.datasets.otb import OTB100_ATTRIBUTES
        return {seq: attrs for seq, attrs in OTB100_ATTRIBUTES.items()}
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-attribute tracker performance analysis for OTB benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results", "-r",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Path(s) to JSON result files (one per tracker).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output Markdown file path.  Defaults to stdout.",
    )
    parser.add_argument(
        "--dataset-name",
        default="OTB-100",
        help="Dataset label in the report header (default: OTB-100).",
    )
    parser.add_argument(
        "--attributes",
        nargs="+",
        default=None,
        metavar="ATTR",
        help=(
            "Restrict analysis to these attribute codes "
            "(e.g. --attributes FM OCC SV).  "
            "Default: all attributes."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a JSON summary alongside the Markdown file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Load results.
    results = []
    for path_str in args.results:
        path = Path(path_str)
        if not path.is_file():
            print(f"ERROR: result file not found: {path}", file=sys.stderr)
            return 1
        try:
            results.append(_load_result(path))
            print(f"Loaded: {path} ({results[-1].tracker_name}, "
                  f"{len(results[-1].sequence_results)} sequences)")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"ERROR: failed to parse {path}: {exc}", file=sys.stderr)
            return 1

    if not results:
        print("ERROR: no valid result files provided.", file=sys.stderr)
        return 1

    # Build attribute map from OTB annotations.
    attr_map = _otb_attribute_map()
    if not attr_map:
        print(
            "WARNING: eovot.datasets.otb not importable — "
            "falling back to attribute map built from result sequence names only.",
            file=sys.stderr,
        )
        # Build a trivial map so the analyzer still runs (no attribute filtering).
        all_seqs = {sr.sequence_name for r in results for sr in r.sequence_results}
        attr_map = {seq: frozenset() for seq in all_seqs}

    from eovot.analysis import AttributeAnalyzer
    try:
        from eovot.datasets.otb import OTB_ATTRIBUTE_NAMES
    except ImportError:
        OTB_ATTRIBUTE_NAMES = {}

    analyzer = AttributeAnalyzer(
        attribute_map=attr_map,
        attribute_names=OTB_ATTRIBUTE_NAMES,
        dataset_name=args.dataset_name,
    )

    try:
        report = analyzer.generate_report(results, attributes=args.attributes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    markdown = report.to_markdown()

    # Output.
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"\nMarkdown report saved to: {out_path}")
        if args.json:
            import json as _json
            json_path = out_path.with_suffix(".json")
            json_path.write_text(
                _json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
            print(f"JSON summary saved to:   {json_path}")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
