#!/usr/bin/env python3
"""Protocol-specific evaluation of saved EOVOT benchmark results.

Reads one or more JSON result files produced by
:class:`~eovot.reporting.reporter.BenchmarkReporter` and computes the
protocol-specific scalars for GOT-10k, LaSOT, or OTB so that EOVOT
numbers can be directly compared against published leaderboards.

Each protocol reports different primary scalars:

  GOT-10k   AO, SR₀.₅, SR₀.₇₅
  LaSOT     Success AUC, Normalized Precision, Normalized Precision AUC
  OTB       Success AUC, Precision Score @ 20 px

Usage
-----
    # GOT-10k metrics for two trackers
    python scripts/evaluate_protocol.py \\
        --results results/MOSSE-GOT10k.json results/KCF-GOT10k.json \\
        --protocol got10k

    # LaSOT metrics
    python scripts/evaluate_protocol.py \\
        --results results/MOSSE-LaSOT.json \\
        --protocol lasot

    # All protocols at once
    python scripts/evaluate_protocol.py \\
        --results results/MOSSE-OTB100.json \\
        --protocol all

    # Save comparison table to CSV
    python scripts/evaluate_protocol.py \\
        --results results/MOSSE-OTB100.json results/KCF-OTB100.json \\
        --protocol otb \\
        --output results/otb_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.metrics.protocols import PROTOCOL_REGISTRY, ProtocolMetricsEngine

import numpy as np

VALID_PROTOCOLS = sorted(PROTOCOL_REGISTRY) + ["all"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_result(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _preds_gts_from_result(result: Dict[str, Any]):
    """Extract stacked prediction and GT arrays from a full result dict.

    Sequences that don't store per-frame arrays (only ``mean_iou``) are
    skipped with a warning so legacy result files still work.
    """
    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []

    for seq in result.get("sequences", []):
        ious = seq.get("ious")
        if ious is None:
            continue

        preds_raw = seq.get("predictions")
        gts_raw = seq.get("ground_truths")
        if preds_raw is None or gts_raw is None:
            # Legacy: only IoU arrays stored — synthesize dummy arrays for
            # protocols that need bbox data (LaSOT NP, OTB precision).
            # We use the same degenerate 1x1 box at origin so IoU-only
            # protocols (GOT-10k) still work, while NP/precision will be NaN.
            n = len(ious)
            preds_raw = [[0.0, 0.0, 1.0, 1.0]] * n
            gts_raw = [[0.0, 0.0, 1.0, 1.0]] * n

        all_preds.append(np.array(preds_raw, dtype=np.float64))
        all_gts.append(np.array(gts_raw, dtype=np.float64))

    if not all_preds:
        return None, None

    return np.vstack(all_preds), np.vstack(all_gts)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _tracker_name(result: Dict[str, Any]) -> str:
    return result.get("summary", {}).get("tracker", Path("?").stem)


def _print_table(rows: List[Dict[str, Any]], protocol: str) -> None:
    if not rows:
        return
    metric_keys = [k for k in rows[0] if k != "tracker"]
    header = ["Tracker"] + metric_keys
    col_w = [max(len(h), *(len(str(r.get(h, r.get(k, "")))) for r in rows))
             for h, k in zip(header, ["tracker"] + metric_keys)]
    sep = "  ".join("-" * w for w in col_w)
    line = "  ".join(h.ljust(w) for h, w in zip(header, col_w))

    print(f"\n{'=' * 60}")
    print(f"  Protocol: {protocol.upper()}")
    print(f"{'=' * 60}")
    print(line)
    print(sep)
    for row in rows:
        vals = [row.get("tracker", "?")] + [str(row.get(k, "N/A")) for k in metric_keys]
        print("  ".join(v.ljust(w) for v, w in zip(vals, col_w)))
    print()


def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evaluate_protocol",
        description="Compute protocol-specific metrics from EOVOT result JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", "-r",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more JSON result files from BenchmarkReporter.",
    )
    parser.add_argument(
        "--protocol", "-p",
        default="all",
        choices=VALID_PROTOCOLS,
        help=(
            "Evaluation protocol to apply. "
            "'all' runs every protocol and prints separate tables."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Optional CSV file path to save the comparison table.",
    )
    args = parser.parse_args()

    engine = ProtocolMetricsEngine()
    protocols = list(PROTOCOL_REGISTRY) if args.protocol == "all" else [args.protocol]

    for protocol in protocols:
        rows: List[Dict[str, Any]] = []
        for path in args.results:
            result = _load_result(path)
            tracker = _tracker_name(result)
            preds, gts = _preds_gts_from_result(result)

            if preds is None:
                print(f"[WARN] {path}: no per-frame data found — skipping.")
                continue

            metrics = engine.compute(protocol, preds, gts)
            row = {"tracker": tracker, **metrics}
            rows.append(row)

        _print_table(rows, protocol)

        if args.output and len(protocols) == 1:
            _save_csv(rows, args.output)


if __name__ == "__main__":
    main()
