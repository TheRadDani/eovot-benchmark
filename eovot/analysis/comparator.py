"""Benchmark run comparison engine for regression detection and ablation studies.

:class:`BenchmarkComparator` takes two :class:`~eovot.benchmark.engine.BenchmarkResult`
objects — a *baseline* (the reference run) and a *candidate* (the run under
evaluation) — and produces a structured comparison across every tracked metric:
accuracy, efficiency, and sequence-level breakdowns.

Motivation
----------
The existing leaderboard tooling (:mod:`eovot.reporting.leaderboard`,
:class:`~eovot.results.bank.ResultsBank`) is designed for **multi-tracker
ranking** on the same dataset.  It cannot answer the question researchers and
CI systems most often need:

    "Did this change make the tracker better or worse, and on which metrics?"

:class:`BenchmarkComparator` fills that gap.  Typical use cases:

- **Regression detection in CI** — fail the pipeline if mean IoU drops more
  than 1 % vs. the last known-good run.
- **Ablation studies** — compare ``KCF(lr=0.075)`` vs. ``KCF(lr=0.125)`` on
  the same synthetic dataset.
- **Latency-accuracy tradeoffs** — surface whether an efficiency improvement
  (resolution downscale, frame-skip) carries an accuracy penalty.

Output formats
--------------
- :class:`ComparisonResult` — typed dataclass with per-metric deltas.
- :meth:`BenchmarkComparator.to_markdown_table` — compact Markdown diff table
  ready for PR descriptions or GitHub comments.
- :meth:`BenchmarkComparator.to_csv` — machine-readable CSV row for CI
  artifact storage.
- :meth:`BenchmarkComparator.to_dict` — JSON-serialisable dict for automated
  pipelines.

Typical usage::

    from eovot.benchmark.engine import BenchmarkResult
    from eovot.analysis.comparator import BenchmarkComparator

    baseline  = BenchmarkResult.load("results/kcf_baseline.json")
    candidate = BenchmarkResult.load("results/kcf_candidate.json")

    cmp = BenchmarkComparator(regression_threshold=0.01)
    result = cmp.compare(baseline, candidate)

    print(result.summary())
    # {'mean_iou': {'baseline': 0.412, 'candidate': 0.433, 'delta': 0.021,
    #               'pct_change': 5.1, 'improved': True, 'regressed': False}, ...}

    print(cmp.to_markdown_table(result))
    print(cmp.to_csv(result))

CLI (via scripts/compare_runs.py)::

    python scripts/compare_runs.py baseline.json candidate.json
    python scripts/compare_runs.py baseline.json candidate.json --threshold 0.02 --csv out.csv
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# Metrics to compare, in display order.
# Each tuple: (attribute_name, display_label, higher_is_better)
_METRIC_SPEC: List[Tuple[str, str, bool]] = [
    ("mean_iou",                     "Mean IoU",             True),
    ("mean_fps",                     "FPS",                  True),
    ("peak_memory_mb",               "Peak Mem (MB)",        False),
    ("mean_success_auc",             "Success AUC",          True),
    ("mean_precision_auc",           "Precision AUC",        True),
    ("mean_normalized_precision_auc","Norm. Prec. AUC",      True),
    ("total_energy_j",               "Total Energy (J)",     False),
    ("mean_energy_per_frame_mj",     "Energy/frame (mJ)",    False),
    ("mean_center_distance",         "Mean CtrDist (px)",    False),
]


@dataclass
class MetricDelta:
    """Delta information for a single metric.

    Attributes:
        metric:        Internal attribute name on :class:`~eovot.benchmark.engine.BenchmarkResult`.
        label:         Human-readable label for display.
        baseline:      Value in the baseline run; ``None`` if not available.
        candidate:     Value in the candidate run; ``None`` if not available.
        delta:         ``candidate - baseline``; ``None`` if either is missing.
        pct_change:    ``delta / |baseline| * 100``; ``None`` if baseline is 0 or missing.
        higher_is_better: Direction of improvement for this metric.
        improved:      ``True`` when the candidate is meaningfully better.
        regressed:     ``True`` when the candidate is meaningfully worse.
    """
    metric: str
    label: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    pct_change: Optional[float]
    higher_is_better: bool
    improved: bool
    regressed: bool

    def direction_symbol(self) -> str:
        """Return a one-character symbol showing direction vs. baseline."""
        if self.improved:
            return "↑"
        if self.regressed:
            return "↓"
        if self.delta is not None and not math.isclose(self.delta, 0.0, abs_tol=1e-9):
            return "~"
        return "="

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "label": self.label,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "pct_change": self.pct_change,
            "higher_is_better": self.higher_is_better,
            "improved": self.improved,
            "regressed": self.regressed,
        }


@dataclass
class SequenceDelta:
    """Per-sequence IoU delta between baseline and candidate."""
    name: str
    baseline_iou: Optional[float]
    candidate_iou: Optional[float]
    delta_iou: Optional[float]

    @property
    def improved(self) -> bool:
        return self.delta_iou is not None and self.delta_iou > 0

    @property
    def regressed(self) -> bool:
        return self.delta_iou is not None and self.delta_iou < 0


@dataclass
class ComparisonResult:
    """Full comparison between a baseline and a candidate benchmark run.

    Attributes:
        baseline_tracker:   Tracker name in the baseline run.
        candidate_tracker:  Tracker name in the candidate run.
        dataset:            Dataset name (from baseline or candidate).
        metric_deltas:      Per-metric :class:`MetricDelta` list.
        sequence_deltas:    Per-sequence IoU deltas (may be empty if sequences
                            differ between runs).
        has_regressions:    ``True`` when any metric regressed beyond the threshold.
        has_improvements:   ``True`` when any metric improved beyond the threshold.
    """
    baseline_tracker: str
    candidate_tracker: str
    dataset: str
    metric_deltas: List[MetricDelta] = field(default_factory=list)
    sequence_deltas: List[SequenceDelta] = field(default_factory=list)
    has_regressions: bool = False
    has_improvements: bool = False

    def summary(self) -> Dict[str, Dict]:
        """Return a nested dict keyed by metric name, suitable for JSON export."""
        return {d.metric: d.to_dict() for d in self.metric_deltas}

    def regression_report(self) -> str:
        """Return a human-readable report of regressed metrics."""
        regs = [d for d in self.metric_deltas if d.regressed]
        if not regs:
            return "No regressions detected."
        lines = [f"Regressions ({self.baseline_tracker} → {self.candidate_tracker}):"]
        for d in regs:
            pct = f" ({d.pct_change:+.1f}%)" if d.pct_change is not None else ""
            lines.append(
                f"  {d.label}: {d.baseline:.4f} → {d.candidate:.4f} "
                f"[Δ={d.delta:+.4f}{pct}]"
            )
        return "\n".join(lines)


class BenchmarkComparator:
    """Compare two benchmark runs and surface per-metric regressions/improvements.

    Args:
        regression_threshold: Minimum absolute relative change (fraction, not %)
            needed to declare a metric regressed or improved.  For example,
            ``0.01`` means a 1 % relative change.  Default: ``0.01``.
        sequence_match_by_name: When ``True``, per-sequence IoU deltas are
            computed only for sequences present in **both** runs (matched by
            name).  When ``False``, per-sequence deltas are skipped.
            Default: ``True``.
    """

    def __init__(
        self,
        regression_threshold: float = 0.01,
        sequence_match_by_name: bool = True,
    ) -> None:
        if regression_threshold < 0:
            raise ValueError(
                f"regression_threshold must be >= 0, got {regression_threshold}"
            )
        self.regression_threshold = regression_threshold
        self.sequence_match_by_name = sequence_match_by_name

    # ------------------------------------------------------------------
    # Main comparison entry point
    # ------------------------------------------------------------------

    def compare(
        self,
        baseline: "BenchmarkResult",
        candidate: "BenchmarkResult",
    ) -> ComparisonResult:
        """Compare *candidate* against *baseline* across all tracked metrics.

        Args:
            baseline:  The reference run (e.g. last known-good or prior config).
            candidate: The run under evaluation (e.g. new algorithm version).

        Returns:
            :class:`ComparisonResult` with per-metric deltas, regression flags,
            and (when sequences match) per-sequence IoU deltas.
        """
        metric_deltas = self._compute_metric_deltas(baseline, candidate)
        has_reg = any(d.regressed for d in metric_deltas)
        has_imp = any(d.improved for d in metric_deltas)

        seq_deltas: List[SequenceDelta] = []
        if self.sequence_match_by_name:
            seq_deltas = self._compute_sequence_deltas(baseline, candidate)

        return ComparisonResult(
            baseline_tracker=baseline.tracker_name,
            candidate_tracker=candidate.tracker_name,
            dataset=baseline.dataset_name or candidate.dataset_name,
            metric_deltas=metric_deltas,
            sequence_deltas=seq_deltas,
            has_regressions=has_reg,
            has_improvements=has_imp,
        )

    # ------------------------------------------------------------------
    # Metric-level comparison
    # ------------------------------------------------------------------

    def _compute_metric_deltas(
        self,
        baseline: "BenchmarkResult",
        candidate: "BenchmarkResult",
    ) -> List[MetricDelta]:
        deltas: List[MetricDelta] = []
        for attr, label, higher_is_better in _METRIC_SPEC:
            b_val = _safe_get(baseline, attr)
            c_val = _safe_get(candidate, attr)

            if b_val is None and c_val is None:
                continue  # metric not available in either run — skip

            delta: Optional[float] = None
            pct: Optional[float] = None
            if b_val is not None and c_val is not None:
                delta = c_val - b_val
                if abs(b_val) > 1e-12:
                    pct = delta / abs(b_val) * 100.0

            improved, regressed = self._classify_delta(delta, pct, higher_is_better)
            deltas.append(
                MetricDelta(
                    metric=attr,
                    label=label,
                    baseline=b_val,
                    candidate=c_val,
                    delta=delta,
                    pct_change=pct,
                    higher_is_better=higher_is_better,
                    improved=improved,
                    regressed=regressed,
                )
            )
        return deltas

    def _classify_delta(
        self,
        delta: Optional[float],
        pct: Optional[float],
        higher_is_better: bool,
    ) -> Tuple[bool, bool]:
        """Return (improved, regressed) booleans for a metric delta."""
        if delta is None:
            return False, False
        # Use relative change when available, absolute delta as fallback.
        magnitude = abs(pct / 100.0) if pct is not None else abs(delta)
        if magnitude < self.regression_threshold:
            return False, False

        # Sign of improvement depends on metric direction.
        positive_change = delta > 0
        is_improvement = positive_change if higher_is_better else not positive_change
        return is_improvement, not is_improvement

    # ------------------------------------------------------------------
    # Sequence-level comparison
    # ------------------------------------------------------------------

    def _compute_sequence_deltas(
        self,
        baseline: "BenchmarkResult",
        candidate: "BenchmarkResult",
    ) -> List[SequenceDelta]:
        b_map = {sr.sequence_name: sr.mean_iou for sr in baseline.sequence_results}
        c_map = {sr.sequence_name: sr.mean_iou for sr in candidate.sequence_results}

        common = sorted(set(b_map) & set(c_map))
        deltas: List[SequenceDelta] = []
        for name in common:
            b_iou = b_map[name]
            c_iou = c_map[name]
            deltas.append(
                SequenceDelta(
                    name=name,
                    baseline_iou=b_iou,
                    candidate_iou=c_iou,
                    delta_iou=c_iou - b_iou,
                )
            )
        return deltas

    # ------------------------------------------------------------------
    # Output formatters
    # ------------------------------------------------------------------

    def to_markdown_table(self, result: ComparisonResult) -> str:
        """Format the comparison as a Markdown table.

        Each row shows the baseline value, candidate value, absolute delta,
        relative change, and a direction symbol (↑ improved / ↓ regressed / = neutral).

        Args:
            result: Output of :meth:`compare`.

        Returns:
            Multi-line Markdown string suitable for PR descriptions and reports.
        """
        header = (
            f"## Benchmark Comparison: `{result.baseline_tracker}` vs "
            f"`{result.candidate_tracker}` on *{result.dataset}*\n\n"
            "| Metric | Baseline | Candidate | Δ | Δ% | Status |\n"
            "|--------|:--------:|:---------:|:-:|:--:|:------:|"
        )
        rows = [header]
        for d in result.metric_deltas:
            b_str = f"{d.baseline:.4f}" if d.baseline is not None else "—"
            c_str = f"{d.candidate:.4f}" if d.candidate is not None else "—"
            delta_str = f"{d.delta:+.4f}" if d.delta is not None else "—"
            pct_str = f"{d.pct_change:+.1f}%" if d.pct_change is not None else "—"
            sym = d.direction_symbol()
            rows.append(
                f"| {d.label} | {b_str} | {c_str} | {delta_str} | {pct_str} | {sym} |"
            )

        # Append summary status
        status_parts = []
        if result.has_regressions:
            status_parts.append("⚠️ regressions detected")
        if result.has_improvements:
            status_parts.append("✅ improvements detected")
        if not result.has_regressions and not result.has_improvements:
            status_parts.append("✅ no significant change")

        rows.append(f"\n*{' · '.join(status_parts)}*")

        if result.sequence_deltas:
            worst = sorted(result.sequence_deltas, key=lambda s: s.delta_iou or 0.0)[:5]
            best = sorted(result.sequence_deltas, key=lambda s: -(s.delta_iou or 0.0))[:5]
            rows.append("\n### Biggest IoU drops (sequences)")
            rows.append("| Sequence | Baseline IoU | Candidate IoU | Δ IoU |")
            rows.append("|----------|:------------:|:-------------:|:-----:|")
            for s in worst:
                rows.append(
                    f"| {s.name} | {s.baseline_iou:.4f} | "
                    f"{s.candidate_iou:.4f} | {s.delta_iou:+.4f} |"
                )
            rows.append("\n### Biggest IoU gains (sequences)")
            rows.append("| Sequence | Baseline IoU | Candidate IoU | Δ IoU |")
            rows.append("|----------|:------------:|:-------------:|:-----:|")
            for s in best:
                rows.append(
                    f"| {s.name} | {s.baseline_iou:.4f} | "
                    f"{s.candidate_iou:.4f} | {s.delta_iou:+.4f} |"
                )

        return "\n".join(rows)

    def to_csv(self, result: ComparisonResult) -> str:
        """Serialise the metric-level comparison as a CSV string.

        Suitable for CI artifact storage or further analysis in pandas.

        Args:
            result: Output of :meth:`compare`.

        Returns:
            CSV-formatted string with a header row followed by one row per metric.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "baseline_tracker", "candidate_tracker", "dataset",
            "metric", "label", "baseline", "candidate",
            "delta", "pct_change", "higher_is_better", "improved", "regressed",
        ])
        for d in result.metric_deltas:
            writer.writerow([
                result.baseline_tracker,
                result.candidate_tracker,
                result.dataset,
                d.metric,
                d.label,
                "" if d.baseline is None else round(d.baseline, 6),
                "" if d.candidate is None else round(d.candidate, 6),
                "" if d.delta is None else round(d.delta, 6),
                "" if d.pct_change is None else round(d.pct_change, 4),
                d.higher_is_better,
                d.improved,
                d.regressed,
            ])
        return buf.getvalue()

    def to_dict(self, result: ComparisonResult) -> Dict[str, Any]:
        """Serialise the full comparison result to a JSON-compatible dict.

        Args:
            result: Output of :meth:`compare`.

        Returns:
            Nested dict suitable for ``json.dumps``.
        """
        return {
            "baseline_tracker": result.baseline_tracker,
            "candidate_tracker": result.candidate_tracker,
            "dataset": result.dataset,
            "has_regressions": result.has_regressions,
            "has_improvements": result.has_improvements,
            "metric_deltas": [d.to_dict() for d in result.metric_deltas],
            "sequence_deltas": [
                {
                    "name": s.name,
                    "baseline_iou": s.baseline_iou,
                    "candidate_iou": s.candidate_iou,
                    "delta_iou": s.delta_iou,
                }
                for s in result.sequence_deltas
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_get(obj: Any, attr: str) -> Optional[float]:
    """Return getattr(obj, attr) as float, or None if absent/None/NaN/error."""
    try:
        val = getattr(obj, attr, None)
    except Exception:
        return None
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
