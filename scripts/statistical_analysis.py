#!/usr/bin/env python
"""Statistical significance analysis for EOVOT benchmark results.

Loads per-tracker JSON result files produced by the EOVOT experiment runner
(or individual benchmark runs), extracts per-sequence IoU scores, and performs:

  1. Bootstrap confidence intervals for each tracker's mean IoU
  2. All-pairs Wilcoxon signed-rank tests with Bonferroni correction
  3. Printed Markdown report ready for copy-paste into papers or GitHub issues

Usage
-----
Run on a directory of tracker result JSON files::

    python scripts/statistical_analysis.py results/my_experiment/

Run on explicitly listed files::

    python scripts/statistical_analysis.py \\
        results/MOSSE-synthetic.json \\
        results/KCF-synthetic.json  \\
        results/CSRT-synthetic.json

Options::

    --alpha FLOAT       Significance level (default: 0.05)
    --bootstrap INT     Bootstrap resamples (default: 10000)
    --seed INT          Random seed (default: 42)
    --metric {iou,success_auc,precision_auc}
                        Per-sequence metric to test (default: iou)
    --out PATH          Write Markdown report to this file

Result JSON format
------------------
Each file must contain a dict with key ``"sequences"``, where each element
is a dict with at least one of:

    mean_iou        (float)
    success_auc     (float)
    precision_auc   (float)

The tracker name is taken from ``summary.tracker_name`` if present, or
inferred from the file stem.

Example::

    python scripts/statistical_analysis.py results/experiment/ --metric iou --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eovot.metrics.statistical import StatisticalTestEngine


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_METRIC_KEY: Dict[str, str] = {
    "iou": "mean_iou",
    "success_auc": "success_auc",
    "precision_auc": "precision_auc",
}


def _load_result_file(path: Path, metric_key: str) -> Optional[tuple[str, List[float]]]:
    """Parse one JSON result file and extract per-sequence scores.

    Args:
        path: Path to the JSON file.
        metric_key: Which per-sequence field to extract (e.g. ``"mean_iou"``).

    Returns:
        ``(tracker_name, scores)`` on success; ``None`` if file is malformed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Skipping {path.name}: {exc}", file=sys.stderr)
        return None

    # Tracker name
    tracker_name = (
        data.get("summary", {}).get("tracker_name")
        or path.stem.split("-")[0]
    )

    sequences = data.get("sequences", [])
    if not sequences:
        print(f"[WARN] Skipping {path.name}: no 'sequences' key.", file=sys.stderr)
        return None

    scores: List[float] = []
    for seq in sequences:
        val = seq.get(metric_key)
        if val is not None:
            scores.append(float(val))

    if not scores:
        print(
            f"[WARN] Skipping {path.name}: metric '{metric_key}' not found "
            f"in any sequence entry.",
            file=sys.stderr,
        )
        return None

    return tracker_name, scores


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
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Statistical significance analysis for EOVOT tracker comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON result files or directories containing them.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        dest="n_bootstrap",
        help="Number of bootstrap resamples for CI (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible bootstrap (default: 42).",
    )
    parser.add_argument(
        "--metric",
        choices=list(_METRIC_KEY),
        default="iou",
        help="Per-sequence metric to analyse (default: iou).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write Markdown report to this file (default: stdout only).",
    )
    args = parser.parse_args(argv)

    metric_key = _METRIC_KEY[args.metric]
    json_files = _collect_inputs(args.inputs)

    if not json_files:
        print("[ERROR] No JSON result files found.", file=sys.stderr)
        return 1

    # Load per-tracker scores
    tracker_scores: Dict[str, List[float]] = {}
    for path in json_files:
        parsed = _load_result_file(path, metric_key)
        if parsed is None:
            continue
        name, scores = parsed
        if name in tracker_scores:
            print(
                f"[WARN] Duplicate tracker name '{name}' — using last seen file.",
                file=sys.stderr,
            )
        tracker_scores[name] = scores

    if len(tracker_scores) < 2:
        print(
            f"[ERROR] Need at least 2 trackers for comparison, "
            f"loaded {len(tracker_scores)}: {list(tracker_scores)}",
            file=sys.stderr,
        )
        return 1

    engine = StatisticalTestEngine(
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    lines: List[str] = [
        "# EOVOT Statistical Analysis Report\n",
        f"**Metric:** `{args.metric}` (`{metric_key}`)  ",
        f"**Significance level:** α = {args.alpha}  ",
        f"**Bootstrap resamples:** {args.n_bootstrap}  ",
        f"**Trackers analysed:** {', '.join(tracker_scores)}  \n",
    ]

    # --- Confidence intervals ---
    lines.append("## Bootstrap Confidence Intervals\n")
    ci_table = engine.ci_table(tracker_scores, metric_name=args.metric)
    lines.append(ci_table)
    lines.append("")

    # --- Pairwise significance ---
    lines.append("## Pairwise Wilcoxon Signed-Rank Tests\n")
    summary = engine.pairwise_report(tracker_scores)
    lines.append(summary.to_markdown())
    lines.append("")

    # --- Warnings ---
    warnings = [c.warning for c in summary.comparisons if c.warning]
    if warnings:
        lines.append("## Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written to: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
