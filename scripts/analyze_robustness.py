#!/usr/bin/env python
"""Robustness analysis CLI for EOVOT benchmark results.

Loads per-tracker JSON result files produced by the EOVOT experiment
runner (or individual benchmark runs), extracts per-sequence IoU
sequences, and computes VOT-challenge-style robustness metrics:

  1. Per-tracker aggregates — total failures, mean EAO, survival rate,
     mean recovery lag
  2. Per-sequence breakdowns showing which sequences the tracker failed on
  3. A ranked Markdown table sorted by mean EAO (descending)

The heavy lifting is delegated to
:class:`eovot.metrics.robustness.RobustnessAnalyzer` — this script is a
thin CLI wrapper that adds JSON loading, argument parsing, and Markdown
report formatting on top.

Usage
-----
Run on a directory of tracker result JSON files::

    python scripts/analyze_robustness.py results/my_experiment/

Run on explicitly listed files::

    python scripts/analyze_robustness.py \\
        results/MOSSE-OTB100.json \\
        results/KCF-OTB100.json \\
        results/CSRT-OTB100.json

Options::

    --failure-threshold FLOAT    IoU below which a frame counts as a failure
                                  (default: 0.1 — standard VOT threshold).
    --recovery-threshold FLOAT   IoU above which the tracker is deemed
                                  recovered (default: 0.1 — no hysteresis).
    --burn-in INT                Frames to skip at the start of each
                                  sequence before looking for failures
                                  (default: 5).
    --top-failures INT           Include the top-N worst per-sequence
                                  failure breakdowns per tracker in the
                                  report (default: 5; use 0 to omit).
    --out PATH                   Write the Markdown report to this file
                                  (default: stdout only).

Result JSON format
------------------
Each input file must contain the schema produced by
:meth:`eovot.benchmark.engine.BenchmarkResult.to_dict` — i.e. a dict with
keys ``"summary"`` and ``"sequences"``, where each sequence entry has a
per-frame ``"ious"`` array.  Older files that only have per-sequence
``mean_iou`` scalars are still loaded, but degenerate to trivial single-
frame sequences and are called out with a warning.

Example
-------
End-to-end from a fresh experiment run::

    python scripts/run_experiment.py --config configs/experiments/multi_tracker.yaml
    python scripts/analyze_robustness.py results/experiments/classical-tracker-comparison/ \\
        --failure-threshold 0.2 --top-failures 3 --out reports/robustness.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.metrics.robustness import RobustnessAnalyzer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_result_file(path: Path) -> Optional[Tuple[str, Dict[str, np.ndarray]]]:
    """Parse one JSON result file and extract per-sequence IoU arrays.

    Args:
        path: Path to the JSON file produced by
            :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

    Returns:
        ``(tracker_name, {sequence_name: ious_array})`` on success;
        ``None`` if the file is malformed or contains no usable sequences.

    Notes:
        * Prefers per-frame ``"ious"`` arrays (present when the result was
          written by a recent EOVOT run).
        * Falls back to a length-1 array with ``mean_iou`` for legacy
          files, which produces coarse but still ranking-consistent EAO
          numbers.  A warning is emitted so users know the data is
          degraded.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
        return None

    tracker_name = (
        data.get("summary", {}).get("tracker")
        or data.get("summary", {}).get("tracker_name")
        or path.stem.split("-")[0]
    )

    sequences = data.get("sequences", [])
    if not sequences:
        print(f"[WARN] Skipping {path.name}: no 'sequences' key.", file=sys.stderr)
        return None

    seq_ious: Dict[str, np.ndarray] = {}
    degraded = False
    for seq in sequences:
        seq_name = seq.get("sequence_name", f"seq_{len(seq_ious)}")
        ious_list = seq.get("ious")
        if ious_list:
            arr = np.asarray(ious_list, dtype=np.float64)
        else:
            degraded = True
            arr = np.asarray([float(seq.get("mean_iou", 0.0))], dtype=np.float64)
        seq_ious[seq_name] = arr

    if not seq_ious:
        print(f"[WARN] Skipping {path.name}: no usable sequences.", file=sys.stderr)
        return None

    if degraded:
        print(
            f"[WARN] {path.name}: some sequences lack per-frame 'ious'; "
            "falling back to mean_iou (robustness numbers will be coarse).",
            file=sys.stderr,
        )

    return tracker_name, seq_ious


def _collect_inputs(paths: List[str]) -> List[Path]:
    """Expand directories to JSON files; return plain JSON paths as-is."""
    result: List[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            found = sorted(p.glob("*.json"))
            if not found:
                print(f"[WARN] No JSON files found in directory: {p}", file=sys.stderr)
            result.extend(found)
        elif p.is_file():
            result.append(p)
        else:
            print(f"[WARN] Path not found: {p}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_top_failures(
    tracker_name: str, per_sequence, top_n: int
) -> List[str]:
    """Return Markdown lines listing the ``top_n`` worst sequences for one tracker.

    "Worst" is defined as the largest number of detected failures, breaking
    ties by lowest EAO — a natural ordering for edge-deployment failure
    triage.

    Args:
        tracker_name: Tracker identifier (used in the section heading).
        per_sequence: Mapping of ``{sequence_name: RobustnessResult}`` from
            :meth:`~eovot.metrics.robustness.RobustnessAnalyzer.analyze_benchmark`.
        top_n: Maximum number of sequences to include.  ``0`` disables
            the section entirely.

    Returns:
        List of Markdown-formatted strings, or an empty list when
        ``top_n == 0`` or no failures were detected.
    """
    if top_n <= 0:
        return []

    ranked = sorted(
        per_sequence.values(),
        key=lambda r: (-r.num_failures, r.eao),
    )
    ranked = [r for r in ranked if r.num_failures > 0][:top_n]
    if not ranked:
        return []

    lines: List[str] = [f"### {tracker_name} — worst {len(ranked)} sequences\n"]
    lines.append(
        "| Sequence | Failures | EAO | Survival | Mean Recovery Lag |"
    )
    lines.append(
        "|----------|---------:|----:|---------:|------------------:|"
    )
    for r in ranked:
        lines.append(
            f"| {r.sequence_name} | {r.num_failures} "
            f"| {r.eao:.4f} | {r.survival_rate:.4f} "
            f"| {r.mean_recovery_lag:.1f} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Robustness analysis for EOVOT tracker benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON result files or directories containing them.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.1,
        dest="failure_threshold",
        help="IoU below which a frame counts as a failure (default: 0.1).",
    )
    parser.add_argument(
        "--recovery-threshold",
        type=float,
        default=0.1,
        dest="recovery_threshold",
        help="IoU above which the tracker is deemed recovered (default: 0.1).",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=5,
        dest="burn_in",
        help="Frames to skip at the start of each sequence before looking "
        "for failures (default: 5).",
    )
    parser.add_argument(
        "--top-failures",
        type=int,
        default=5,
        dest="top_failures",
        help="Include the top-N worst per-sequence failures per tracker "
        "in the report (default: 5; use 0 to omit).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the Markdown report to this file (default: stdout only).",
    )
    args = parser.parse_args(argv)

    json_files = _collect_inputs(args.inputs)
    if not json_files:
        print("[ERROR] No JSON result files found.", file=sys.stderr)
        return 1

    analyzer = RobustnessAnalyzer(
        failure_threshold=args.failure_threshold,
        recovery_threshold=args.recovery_threshold,
        burn_in_frames=args.burn_in,
    )

    aggregates: List[Dict] = []
    per_tracker_seqs: Dict[str, Dict] = {}
    for path in json_files:
        parsed = _load_result_file(path)
        if parsed is None:
            continue
        tracker_name, seq_ious = parsed
        analysis = analyzer.analyze_benchmark(seq_ious, tracker_name=tracker_name)
        aggregates.append(analysis["aggregate"])
        per_tracker_seqs[tracker_name] = analysis["per_sequence"]

    if not aggregates:
        print("[ERROR] No usable tracker results loaded.", file=sys.stderr)
        return 1

    lines: List[str] = [
        "# EOVOT Robustness Analysis Report\n",
        f"**Failure threshold:** IoU < {args.failure_threshold}  ",
        f"**Recovery threshold:** IoU ≥ {args.recovery_threshold}  ",
        f"**Burn-in frames:** {args.burn_in}  ",
        f"**Trackers analysed:** {', '.join(sorted(per_tracker_seqs))}  \n",
        "## Aggregate Leaderboard (ranked by mean EAO)\n",
        analyzer.to_markdown_table(aggregates),
        "",
    ]

    if args.top_failures > 0:
        header_added = False
        for tracker_name in sorted(per_tracker_seqs):
            block = _format_top_failures(
                tracker_name,
                per_tracker_seqs[tracker_name],
                args.top_failures,
            )
            if block and not header_added:
                lines.append("## Worst Failure Sequences\n")
                header_added = True
            lines.extend(block)
        if not header_added:
            lines.append("_No failures detected in any sequence._\n")

    report = "\n".join(lines)
    print(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
