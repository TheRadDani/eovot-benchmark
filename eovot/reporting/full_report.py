"""Unified multi-metric benchmark report for EOVOT.

Aggregates all EOVOT metric dimensions — accuracy, robustness, temporal
consistency, and edge efficiency — into a single publishable comparison table
from a list of :class:`~eovot.benchmark.engine.BenchmarkResult` objects.

The :class:`FullBenchmarkReport` class is the intended entry point for
experiments that need a complete side-by-side tracker comparison without
manually chaining multiple metric engines.

Metric dimensions
~~~~~~~~~~~~~~~~~

**Accuracy** (from benchmark engine)
    - Mean IoU, Success AUC, Precision AUC

**Robustness** (:class:`~eovot.metrics.robustness.RobustnessAnalyzer`)
    - Expected Average Overlap (EAO)
    - Total failures and mean failure rate per sequence

**Temporal consistency** (:class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`)
    - Composite smoothness score ``S ∈ (0, 1]``
    - Position jitter, scale jitter

**Edge efficiency** (:class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`)
    - Edge Efficiency Score (EES)
    - FPS, peak memory, Pareto-front membership

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker
    from eovot.reporting.full_report import FullBenchmarkReport

    ds      = SyntheticDataset(num_sequences=10, num_frames=200)
    engine  = BenchmarkEngine(verbose=False)
    results = [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(),   ds, dataset_name="Synthetic"),
    ]

    report = FullBenchmarkReport(results)
    print(report.to_markdown())
    report.to_csv("comparison.csv")
    report.to_json("comparison.json")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..benchmark.engine import BenchmarkResult
from ..metrics.efficiency import EfficiencyMetricsEngine
from ..metrics.robustness import RobustnessAnalyzer
from ..metrics.temporal import TemporalConsistencyAnalyzer


# ---------------------------------------------------------------------------
# Per-tracker summary row
# ---------------------------------------------------------------------------

@dataclass
class TrackerRow:
    """All metrics for one tracker, computed by :class:`FullBenchmarkReport`.

    Every numeric field is ``None`` when the underlying data needed to
    compute it was absent (e.g. ``predictions`` not stored).
    """

    tracker_name: str
    dataset_name: str

    # --- accuracy ---
    mean_iou: float
    success_auc: Optional[float]
    precision_auc: Optional[float]

    # --- efficiency ---
    mean_fps: float
    peak_memory_mb: float
    ees: float
    on_pareto_front: bool

    # --- robustness ---
    eao: float
    total_failures: int
    mean_failures_per_seq: float
    mean_survival_rate: float

    # --- temporal consistency ---
    smoothness_score: Optional[float]
    position_jitter: Optional[float]
    scale_jitter: Optional[float]

    def to_dict(self) -> Dict:
        """Flat dict — every key is a valid CSV / JSON field name."""
        return {
            "tracker": self.tracker_name,
            "dataset": self.dataset_name,
            "mean_iou": _fmt(self.mean_iou),
            "success_auc": _fmt(self.success_auc),
            "precision_auc": _fmt(self.precision_auc),
            "mean_fps": _fmt(self.mean_fps, 1),
            "peak_memory_mb": _fmt(self.peak_memory_mb, 1),
            "ees": _fmt(self.ees),
            "on_pareto_front": self.on_pareto_front,
            "eao": _fmt(self.eao),
            "total_failures": self.total_failures,
            "mean_failures_per_seq": _fmt(self.mean_failures_per_seq, 2),
            "mean_survival_rate": _fmt(self.mean_survival_rate),
            "smoothness_score": _fmt(self.smoothness_score),
            "position_jitter": _fmt(self.position_jitter, 5),
            "scale_jitter": _fmt(self.scale_jitter, 5),
        }


def _fmt(val, decimals: int = 4):
    """Round a float value or return None unchanged."""
    if val is None:
        return None
    return round(float(val), decimals)


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------

class FullBenchmarkReport:
    """Compute and render a comprehensive multi-metric tracker comparison.

    Args:
        results:            One :class:`~eovot.benchmark.engine.BenchmarkResult`
                            per tracker.
        memory_budget_mb:   Memory budget for EES computation (default 512 MB).
        failure_threshold:  IoU below which a frame is a failure (default 0.1).

    Example::

        report = FullBenchmarkReport(results)
        print(report.to_markdown())
        report.to_csv("report.csv")
        report.to_json("report.json")
    """

    def __init__(
        self,
        results: List[BenchmarkResult],
        memory_budget_mb: float = 512.0,
        failure_threshold: float = 0.1,
    ) -> None:
        if not results:
            raise ValueError("FullBenchmarkReport requires at least one BenchmarkResult.")
        self._results = results
        self._robustness = RobustnessAnalyzer(failure_threshold=failure_threshold)
        self._temporal = TemporalConsistencyAnalyzer()
        self._efficiency = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        self._rows: Optional[List[TrackerRow]] = None

    # ------------------------------------------------------------------
    # Row computation
    # ------------------------------------------------------------------

    @property
    def rows(self) -> List[TrackerRow]:
        """Compute and cache all metric rows (one per result)."""
        if self._rows is None:
            self._rows = self._compute_rows()
        return self._rows

    def _compute_rows(self) -> List[TrackerRow]:
        eff_entries = self._efficiency.rank_trackers(self._results)
        eff_by_name = {e.tracker_name: e for e in eff_entries}

        rows: List[TrackerRow] = []
        for r in self._results:
            # Robustness
            seq_ious = {sr.sequence_name: sr.ious for sr in r.sequence_results}
            rob = self._robustness.analyze_benchmark(seq_ious, tracker_name=r.tracker_name)
            rob_agg = rob["aggregate"]

            # Temporal consistency — requires stored predictions
            smooth_score = pos_jitter = scale_jitter = None
            preds_available = [
                sr for sr in r.sequence_results if sr.predictions is not None
            ]
            if preds_available:
                seq_preds = {sr.sequence_name: sr.predictions for sr in preds_available}
                temp = self._temporal.analyze_benchmark(seq_preds, tracker_name=r.tracker_name)
                agg = temp.get("aggregate", {})
                smooth_score = agg.get("mean_smoothness_score")
                pos_jitter = agg.get("mean_position_jitter")
                scale_jitter = agg.get("mean_scale_jitter")

            eff = eff_by_name.get(r.tracker_name)

            rows.append(TrackerRow(
                tracker_name=r.tracker_name,
                dataset_name=r.dataset_name,
                mean_iou=r.mean_iou,
                success_auc=r.mean_success_auc,
                precision_auc=r.mean_precision_auc,
                mean_fps=r.mean_fps,
                peak_memory_mb=r.peak_memory_mb,
                ees=eff.ees if eff else 0.0,
                on_pareto_front=eff.on_pareto_front if eff else False,
                eao=rob_agg["mean_eao"],
                total_failures=rob_agg["total_failures"],
                mean_failures_per_seq=rob_agg["mean_failures_per_sequence"],
                mean_survival_rate=rob_agg["mean_survival_rate"],
                smoothness_score=smooth_score,
                position_jitter=pos_jitter,
                scale_jitter=scale_jitter,
            ))

        # Sort by EES descending (best edge-deployable tracker first)
        rows.sort(key=lambda row: row.ees, reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Markdown output
    # ------------------------------------------------------------------

    def to_markdown(self, title: str = "Full Multi-Metric Benchmark Comparison") -> str:
        """Render a comprehensive Markdown report with four metric sections.

        Args:
            title: Heading for the report.

        Returns:
            Multi-section Markdown string ready for README or PR comments.
        """
        rows = self.rows
        lines: List[str] = [f"## {title}", ""]

        # ---- Accuracy table ----
        lines += [
            "### Accuracy",
            "",
            "| Rank | Tracker | mIoU | Succ. AUC | Prec. AUC |",
            "|:----:|:--------|-----:|----------:|----------:|",
        ]
        best_iou = max(r.mean_iou for r in rows)
        for i, row in enumerate(rows, 1):
            iou_s = _bold(f"{row.mean_iou:.4f}", row.mean_iou == best_iou)
            sauc  = f"{row.success_auc:.4f}" if row.success_auc is not None else "—"
            pauc  = f"{row.precision_auc:.4f}" if row.precision_auc is not None else "—"
            lines.append(f"| {i} | {row.tracker_name} | {iou_s} | {sauc} | {pauc} |")

        # ---- Robustness table ----
        lines += [
            "",
            "### Robustness",
            "",
            "| Rank | Tracker | EAO | Failures | Survival Rate |",
            "|:----:|:--------|----:|---------:|:-------------:|",
        ]
        best_eao = max(r.eao for r in rows)
        for i, row in enumerate(rows, 1):
            eao_s = _bold(f"{row.eao:.4f}", row.eao == best_eao)
            lines.append(
                f"| {i} | {row.tracker_name} | {eao_s} "
                f"| {row.total_failures} | {row.mean_survival_rate:.4f} |"
            )

        # ---- Temporal consistency table ----
        has_smooth = any(r.smoothness_score is not None for r in rows)
        if has_smooth:
            lines += [
                "",
                "### Temporal Consistency",
                "",
                "| Rank | Tracker | Smoothness ↑ | Pos Jitter ↓ | Scale Jitter ↓ |",
                "|:----:|:--------|:------------:|:------------:|:--------------:|",
            ]
            valid_smooth = [r.smoothness_score for r in rows if r.smoothness_score is not None]
            best_smooth = max(valid_smooth) if valid_smooth else None
            for i, row in enumerate(rows, 1):
                s_s = "—" if row.smoothness_score is None else _bold(
                    f"{row.smoothness_score:.4f}",
                    best_smooth is not None and row.smoothness_score == best_smooth,
                )
                pj_s = "—" if row.position_jitter is None else f"{row.position_jitter:.5f}"
                sj_s = "—" if row.scale_jitter is None else f"{row.scale_jitter:.5f}"
                lines.append(f"| {i} | {row.tracker_name} | {s_s} | {pj_s} | {sj_s} |")

        # ---- Edge efficiency table ----
        lines += [
            "",
            "### Edge Efficiency",
            "",
            "| Rank | Tracker | EES ↑ | FPS | Mem (MiB) | Pareto |",
            "|:----:|:--------|------:|----:|----------:|:------:|",
        ]
        best_ees = max(r.ees for r in rows)
        for i, row in enumerate(rows, 1):
            ees_s   = _bold(f"{row.ees:.4f}", row.ees == best_ees)
            pareto  = "✓" if row.on_pareto_front else ""
            lines.append(
                f"| {i} | {row.tracker_name} | {ees_s} "
                f"| {row.mean_fps:.1f} | {row.peak_memory_mb:.1f} | {pareto} |"
            )

        lines.append("")
        lines.append(
            f"*{len(rows)} tracker(s) evaluated on {rows[0].dataset_name}. "
            "EES = Edge Efficiency Score. Higher is better unless ↓ noted.*"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write all metrics to a CSV file (one row per tracker).

        Args:
            path: Destination file path.

        Returns:
            Resolved :class:`pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".csv")
        p.parent.mkdir(parents=True, exist_ok=True)

        if not self.rows:
            return p

        fieldnames = list(self.rows[0].to_dict().keys())
        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.to_dict())
        return p

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def to_json(self, path: Union[str, Path]) -> Path:
        """Write all metrics to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            Resolved :class:`pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dataset": self.rows[0].dataset_name if self.rows else "unknown",
            "trackers": [row.to_dict() for row in self.rows],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return p


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bold(s: str, condition: bool) -> str:
    return f"**{s}**" if condition else s
