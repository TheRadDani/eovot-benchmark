#!/usr/bin/env python3
"""Confidence calibration comparison for EOVOT trackers.

Reads saved ``BenchmarkResult`` JSON files alongside per-tracker PSR sidecar
files and produces a unified Markdown report that ranks trackers on three
complementary axes:

1. **Confidence calibration** — how well the tracker's own PSR signal
   predicts its failures (Pearson r, AUROC, optimal F1 threshold).
2. **Robustness** — VOT-style failure count and Expected Average Overlap.
3. **Efficiency** — Edge Efficiency Score (EES), FPS, memory footprint.

PSR sidecar format
------------------
A PSR sidecar is a JSON file produced by the AdaptiveKCFTracker (or any
tracker that records per-frame PSR values).  It must contain a dict keyed by
sequence name, each value being a list of floats::

    {
        "seq1": [8.3, 7.9, 5.1, ...],
        "seq2": [12.0, 11.4, ...],
        ...
    }

Usage examples
--------------
Run a demo with the built-in synthetic dataset (no files needed)::

    python scripts/compare_confidence.py --demo

Load real results + PSR sidecars::

    python scripts/compare_confidence.py \\
        --results results/AdaptiveKCF-OTB100.json results/KCF-OTB100.json \\
        --psr-dir results/psr/

Save the report to a file::

    python scripts/compare_confidence.py --demo --output report.md

Control the IoU failure threshold used for calibration analysis::

    python scripts/compare_confidence.py --demo --failure-threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure the repo root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.benchmark.engine import BenchmarkResult
from eovot.metrics.confidence import ConfidenceAnalyzer, ConfidenceResult
from eovot.metrics.robustness import RobustnessAnalyzer
from eovot.metrics.efficiency import EfficiencyMetricsEngine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_confidence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results",
        nargs="*",
        metavar="JSON_FILE",
        default=[],
        help="BenchmarkResult JSON files (one per tracker/dataset combination).",
    )
    p.add_argument(
        "--psr-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory containing PSR sidecar JSON files.  Each file must be "
            "named <tracker_name>.json and contain a dict {sequence_name: [psr...]}."
        ),
    )
    p.add_argument(
        "--failure-threshold",
        type=float,
        default=0.1,
        metavar="IOU",
        help="IoU below which a frame is counted as a tracking failure. Default: 0.1.",
    )
    p.add_argument(
        "--memory-budget",
        type=float,
        default=512.0,
        metavar="MB",
        help="Memory budget for EES denominator (MB). Default: 512.",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write Markdown report to FILE instead of (or in addition to) stdout.",
    )
    p.add_argument(
        "--no-stdout",
        action="store_true",
        help="Suppress stdout output (useful when --output is given).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a self-contained demo using the built-in synthetic dataset.  "
            "Benchmarks KCF and AdaptiveKCF on 3 synthetic sequences."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo(
    failure_threshold: float,
    memory_budget_mb: float,
) -> Tuple[List[BenchmarkResult], Dict[str, Dict[str, List[float]]]]:
    """Benchmark KCF and AdaptiveKCF on the synthetic dataset and collect PSR histories."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.adaptive_kcf import AdaptiveKCFTracker
    from eovot.trackers.kcf import KCFTracker

    print("Running demo benchmark (KCF vs AdaptiveKCF on synthetic sequences)…")
    dataset = SyntheticDataset(num_sequences=3, num_frames=40, motion="linear")
    engine = BenchmarkEngine(verbose=False)

    results: List[BenchmarkResult] = []
    psr_sidecars: Dict[str, Dict[str, List[float]]] = {}

    # KCF — no PSR signal available
    kcf = KCFTracker()
    kcf_result = engine.run(kcf, dataset, dataset_name="Synthetic")
    results.append(kcf_result)

    # AdaptiveKCF — yields PSR history
    adaptive = AdaptiveKCFTracker(min_psr=4.0)
    adaptive_result = engine.run(adaptive, dataset, dataset_name="Synthetic")
    results.append(adaptive_result)

    # Collect per-sequence PSR histories from the tracker object.
    # After engine.run() the tracker has been run over all sequences; the
    # psr_history property holds the full concatenated history.  We need per-
    # sequence lengths to split it by sequence.
    per_seq_psr: Dict[str, List[float]] = {}
    flat_psr = adaptive.psr_history
    offset = 0
    for seq_result in adaptive_result.sequence_results:
        n = len(seq_result.ious)
        per_seq_psr[seq_result.sequence_name] = flat_psr[offset: offset + n]
        offset += n
    psr_sidecars["AdaptiveKCF"] = per_seq_psr

    print("Demo benchmark complete.\n")
    return results, psr_sidecars


# ---------------------------------------------------------------------------
# PSR sidecar loading
# ---------------------------------------------------------------------------

def _load_psr_sidecars(psr_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """Load all <tracker_name>.json files from *psr_dir* into a nested dict."""
    sidecars: Dict[str, Dict[str, List[float]]] = {}
    for path in sorted(Path(psr_dir).glob("*.json")):
        tracker_name = path.stem
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            print(f"  Warning: {path} is not a dict — skipped.", file=sys.stderr)
            continue
        sidecars[tracker_name] = data
    return sidecars


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyze_tracker(
    result: BenchmarkResult,
    psr_by_seq: Optional[Dict[str, List[float]]],
    failure_threshold: float,
    memory_budget_mb: float,
) -> dict:
    """Return a summary dict for one tracker across all its sequences."""
    conf_analyzer = ConfidenceAnalyzer(failure_iou_threshold=failure_threshold)
    rob_analyzer = RobustnessAnalyzer(failure_threshold=failure_threshold)
    eff_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)

    # Collect per-sequence IoUs
    iou_by_seq: Dict[str, List[float]] = {
        r.sequence_name: r.ious.tolist() for r in result.sequence_results
    }

    # Robustness aggregate
    rob_agg = rob_analyzer.analyze_benchmark(
        {k: np.asarray(v) for k, v in iou_by_seq.items()},
        tracker_name=result.tracker_name,
    )
    rob_summary = rob_agg.get("aggregate", {})

    # Efficiency
    ees = eff_engine.edge_efficiency_score(
        mean_iou=result.mean_iou,
        fps=result.mean_fps,
        peak_memory_mb=result.peak_memory_mb,
    )

    # Confidence calibration (only when PSR sidecar is available)
    conf_result: Optional[ConfidenceResult] = None
    if psr_by_seq is not None:
        # Align PSR lists with IoU lists (skip the first IoU = init frame)
        conf_by_seq: Dict[str, List[float]] = {}
        iou_skip1_by_seq: Dict[str, List[float]] = {}
        for seq_name, psr_list in psr_by_seq.items():
            if seq_name not in iou_by_seq:
                continue
            iou_list = iou_by_seq[seq_name]
            # psr_history records one entry per update call (skipping frame 0)
            # ious has one entry per frame including init (frame 0 IoU = 1.0)
            aligned_ious = iou_list[1:] if len(iou_list) > len(psr_list) else iou_list
            if len(psr_list) >= 2 and len(aligned_ious) >= 2:
                conf_by_seq[seq_name] = psr_list
                iou_skip1_by_seq[seq_name] = aligned_ious

        if conf_by_seq:
            agg_conf = conf_analyzer.analyze_benchmark(
                conf_by_seq, iou_skip1_by_seq, tracker_name=result.tracker_name
            )
            agg = agg_conf.get("aggregate", {})
            conf_result = ConfidenceResult(
                tracker_name=result.tracker_name,
                sequence_name="aggregate",
                pearson_r=agg.get("mean_pearson_r", float("nan")),
                auroc=agg.get("mean_auroc", float("nan")),
                best_threshold=agg.get("median_best_threshold", float("nan")),
                best_f1=agg.get("mean_best_f1", float("nan")),
                n_frames=agg.get("total_frames", 0),
                n_failures=agg.get("total_failures", 0),
            )

    return {
        "tracker_name": result.tracker_name,
        "dataset_name": result.dataset_name,
        "mean_iou": result.mean_iou,
        "mean_fps": result.mean_fps,
        "peak_memory_mb": result.peak_memory_mb,
        "ees": ees,
        "mean_eao": rob_summary.get("mean_eao", float("nan")),
        "mean_failures": rob_summary.get("mean_failures", float("nan")),
        "has_confidence": conf_result is not None,
        "pearson_r": conf_result.pearson_r if conf_result else float("nan"),
        "auroc": conf_result.auroc if conf_result else float("nan"),
        "best_threshold": conf_result.best_threshold if conf_result else float("nan"),
        "best_f1": conf_result.best_f1 if conf_result else float("nan"),
        "n_frames": conf_result.n_frames if conf_result else 0,
        "n_failures": conf_result.n_failures if conf_result else 0,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(value: float, fmt: str = ".4f") -> str:
    if not np.isfinite(value):
        return "—"
    return format(value, fmt)


def _render_report(
    rows: List[dict],
    failure_threshold: float,
    memory_budget_mb: float,
) -> str:
    lines: List[str] = []

    lines.append("# EOVOT Confidence Calibration Report\n")
    lines.append(
        f"_Failure threshold: IoU < {failure_threshold}  "
        f"| Memory budget: {memory_budget_mb:.0f} MB_\n"
    )

    # --- Section 1: Confidence calibration ---
    lines.append("## 1. Confidence Calibration\n")
    lines.append(
        "Measures how well each tracker's own confidence signal (PSR) predicts "
        "tracking failures.  Trackers without a PSR sidecar show `—`.\n"
    )
    lines.append(
        "| Tracker | Dataset | Pearson r | AUROC | Best PSR thr | Best F1 | "
        "Frames | Failures |"
    )
    lines.append(
        "|---------|---------|----------:|------:|-------------:|--------:|"
        "-------:|---------:|"
    )
    for r in sorted(rows, key=lambda x: x["auroc"] if np.isfinite(x["auroc"]) else -1, reverse=True):
        lines.append(
            f"| {r['tracker_name']} | {r['dataset_name']} "
            f"| {_fmt(r['pearson_r'], '.3f')} "
            f"| {_fmt(r['auroc'], '.3f')} "
            f"| {_fmt(r['best_threshold'], '.2f')} "
            f"| {_fmt(r['best_f1'], '.3f')} "
            f"| {r['n_frames']} "
            f"| {r['n_failures']} |"
        )

    lines.append("")
    lines.append("> **AUROC > 0.7** — reliable failure predictor.  "
                 "**0.5** — chance (PSR is uninformative).")

    # --- Section 2: Robustness ---
    lines.append("\n## 2. Robustness\n")
    lines.append(
        "| Tracker | Dataset | mIoU | EAO | Mean failures / seq |"
    )
    lines.append("|---------|---------|-----:|----:|--------------------:|")
    for r in sorted(rows, key=lambda x: x["mean_eao"] if np.isfinite(x["mean_eao"]) else -1, reverse=True):
        lines.append(
            f"| {r['tracker_name']} | {r['dataset_name']} "
            f"| {_fmt(r['mean_iou'], '.4f')} "
            f"| {_fmt(r['mean_eao'], '.4f')} "
            f"| {_fmt(r['mean_failures'], '.1f')} |"
        )

    # --- Section 3: Efficiency ---
    lines.append("\n## 3. Edge Efficiency\n")
    lines.append(
        f"EES = mIoU × log1p(FPS) / (1 + mem / {memory_budget_mb:.0f} MB)\n"
    )
    lines.append(
        "| Rank | Tracker | Dataset | EES | mIoU | FPS | Mem (MB) |"
    )
    lines.append("|-----:|---------|---------|----:|-----:|----:|---------:|")
    sorted_eff = sorted(rows, key=lambda x: x["ees"], reverse=True)
    for rank, r in enumerate(sorted_eff, start=1):
        lines.append(
            f"| {rank} | {r['tracker_name']} | {r['dataset_name']} "
            f"| {_fmt(r['ees'], '.4f')} "
            f"| {_fmt(r['mean_iou'], '.4f')} "
            f"| {_fmt(r['mean_fps'], '.1f')} "
            f"| {_fmt(r['peak_memory_mb'], '.1f')} |"
        )

    # --- Section 4: Interpretation notes ---
    lines.append("\n## Notes\n")
    lines.append(
        "- **PSR (Peak-to-Sidelobe Ratio)**: filter response quality metric.  "
        "PSR > 7 typically indicates reliable tracking; < 5 suggests failure.\n"
        "- **AUROC**: area under the receiver-operating-characteristic curve "
        "treating PSR as a binary failure predictor (failure = IoU < threshold).  "
        "An uninformative signal scores 0.5.\n"
        "- **EES**: Edge Efficiency Score.  Combines accuracy and speed with a "
        "soft memory penalty, favouring trackers that are both accurate and fast "
        "within the memory budget.\n"
        "- Rows marked `—` have no PSR sidecar and cannot be evaluated for "
        "confidence calibration.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.demo and not args.results:
        parser.error("Provide --results files or use --demo.")

    # Load or generate benchmark results and PSR sidecars
    if args.demo:
        bench_results, psr_sidecars = _run_demo(
            failure_threshold=args.failure_threshold,
            memory_budget_mb=args.memory_budget,
        )
    else:
        bench_results = [BenchmarkResult.load(f) for f in args.results]
        psr_sidecars: Dict[str, Dict[str, List[float]]] = {}
        if args.psr_dir:
            psr_sidecars = _load_psr_sidecars(args.psr_dir)

    # Analyse each tracker
    rows = []
    for result in bench_results:
        psr_by_seq = psr_sidecars.get(result.tracker_name)
        row = _analyze_tracker(
            result=result,
            psr_by_seq=psr_by_seq,
            failure_threshold=args.failure_threshold,
            memory_budget_mb=args.memory_budget,
        )
        rows.append(row)

    report = _render_report(
        rows=rows,
        failure_threshold=args.failure_threshold,
        memory_budget_mb=args.memory_budget,
    )

    if not args.no_stdout:
        print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
