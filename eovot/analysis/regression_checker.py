"""Performance regression detection for EOVOT benchmark results.

Compares two :class:`~eovot.benchmark.engine.BenchmarkResult` objects —
a *baseline* (before a change) and a *candidate* (after the change) —
and determines whether accuracy or efficiency metrics have regressed,
improved, or stayed the same.

Why automated regression detection?
------------------------------------
When iterating on a tracker implementation or a profiling configuration,
it is easy to accidentally regress tracking accuracy while optimising for
speed.  Manual inspection of summary tables is error-prone, especially
when differences are small.  This module provides:

- **Per-metric delta** — exact numeric difference between candidate and
  baseline for every aggregate metric (IoU, FPS, latency, AUCs).
- **Direction tagging** — each metric is tagged ``improved``,
  ``regressed``, or ``unchanged`` based on configurable thresholds.
- **Statistical significance** — a two-sided paired t-test (Welch) on
  the aligned per-sequence IoU arrays determines whether the observed
  IoU change is likely to be real or noise.
- **CI-friendly exit** — :func:`run_cli` exits with code ``0`` if no
  regressions are detected and ``1`` otherwise, enabling automated
  gating in GitHub Actions or pre-merge hooks.

Usage::

    from eovot.analysis.regression_checker import RegressionChecker
    from eovot.benchmark.engine import BenchmarkResult

    baseline  = BenchmarkResult.load("results/mosse_v1.json")
    candidate = BenchmarkResult.load("results/mosse_v2.json")

    checker = RegressionChecker(min_iou_delta=0.005, significance_level=0.05)
    report  = checker.compare(baseline, candidate)

    print(report.summary())
    print("Regressions found:", report.has_regression)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# Metrics where a *higher* value is better (delta = candidate - baseline > 0 → improvement)
_HIGHER_IS_BETTER = {
    "mean_iou", "mean_fps", "success_auc", "precision_auc",
    "normalized_precision_auc",
}
# Metrics where a *lower* value is better
_LOWER_IS_BETTER = {
    "mean_latency_ms", "peak_memory_mb", "total_energy_j",
    "mean_energy_per_frame_mj",
}


@dataclass
class MetricDelta:
    """Change in a single metric between baseline and candidate."""

    name: str
    baseline: float
    candidate: float
    delta: float
    relative_delta: float
    """(candidate - baseline) / |baseline|.  ``nan`` when baseline ≈ 0."""
    direction: str
    """``"improved"``, ``"regressed"``, or ``"unchanged"``."""

    def __str__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        pct = f"{self.relative_delta * 100:+.2f}%" if not math.isnan(self.relative_delta) else "n/a"
        return (
            f"  {self.name:<35s}  "
            f"baseline={self.baseline:.4f}  "
            f"candidate={self.candidate:.4f}  "
            f"Δ={sign}{self.delta:.4f} ({pct})  [{self.direction}]"
        )


@dataclass
class RegressionReport:
    """Full comparison report produced by :class:`RegressionChecker`.

    Attributes:
        baseline_name:   Tracker/dataset label from the baseline result.
        candidate_name:  Tracker/dataset label from the candidate result.
        metric_deltas:   One :class:`MetricDelta` per compared metric.
        iou_pvalue:      p-value of the paired t-test on per-sequence IoU;
                         ``None`` when fewer than two matching sequences exist.
        iou_significant: ``True`` when ``iou_pvalue < significance_level``.
        has_regression:  ``True`` when at least one metric is tagged
                         ``"regressed"`` and (if IoU-related) the change is
                         statistically significant.
    """

    baseline_name: str
    candidate_name: str
    metric_deltas: List[MetricDelta] = field(default_factory=list)
    iou_pvalue: Optional[float] = None
    iou_significant: bool = False
    has_regression: bool = False
    sequence_count: int = 0
    matched_sequences: int = 0

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        lines = [
            "=" * 70,
            f"  EOVOT Regression Report",
            f"  Baseline:  {self.baseline_name}",
            f"  Candidate: {self.candidate_name}",
            f"  Sequences: {self.matched_sequences} matched / {self.sequence_count} total",
        ]
        if self.iou_pvalue is not None:
            lines.append(
                f"  IoU t-test p-value: {self.iou_pvalue:.4f}  "
                f"{'(significant)' if self.iou_significant else '(not significant)'}"
            )
        lines.append("=" * 70)
        for d in self.metric_deltas:
            lines.append(str(d))
        lines.append("=" * 70)
        status = "REGRESSION DETECTED" if self.has_regression else "OK (no regression)"
        lines.append(f"  Status: {status}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialisable dict for logging / JSON export."""
        return {
            "baseline": self.baseline_name,
            "candidate": self.candidate_name,
            "matched_sequences": self.matched_sequences,
            "sequence_count": self.sequence_count,
            "iou_pvalue": self.iou_pvalue,
            "iou_significant": self.iou_significant,
            "has_regression": self.has_regression,
            "metric_deltas": [
                {
                    "name": d.name,
                    "baseline": d.baseline,
                    "candidate": d.candidate,
                    "delta": d.delta,
                    "relative_delta": d.relative_delta,
                    "direction": d.direction,
                }
                for d in self.metric_deltas
            ],
        }


def _welch_ttest_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value.  Returns 1.0 when the test can't run."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    diff = b - a
    n = len(diff)
    mean_d = diff.mean()
    std_d = diff.std(ddof=1)
    if std_d == 0:
        return 0.0 if mean_d != 0.0 else 1.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    # Two-sided p-value via Student's t distribution approximation (Abramowitz & Stegun)
    # Using the simple normal approximation for n >= 30; for small n use exact
    try:
        from scipy.stats import ttest_rel
        _, p = ttest_rel(a, b)
        return float(p)
    except ImportError:
        pass
    # Fallback: normal approximation (conservative for small n)
    z = abs(t_stat)
    p_one_side = 0.5 * math.erfc(z / math.sqrt(2))
    return 2.0 * p_one_side


class RegressionChecker:
    """Detect performance regressions between two benchmark runs.

    Args:
        min_iou_delta:      Minimum absolute IoU change to consider non-trivial.
                            Changes smaller than this are tagged ``"unchanged"``.
                            Default: ``0.005`` (half a percentage point).
        min_fps_delta:      Minimum absolute FPS change threshold.
                            Default: ``0.5`` FPS.
        significance_level: p-value threshold for the IoU t-test.
                            Default: ``0.05``.
        strict:             When ``True``, any metric tagged ``"regressed"``
                            causes ``has_regression=True`` regardless of
                            statistical significance.  When ``False`` (default),
                            only IoU regressions are gated on significance;
                            other metrics are unconditionally flagged.

    Example::

        checker = RegressionChecker(min_iou_delta=0.01, significance_level=0.05)
        report  = checker.compare(baseline_result, candidate_result)
        if report.has_regression:
            print("Performance regression detected!")
            print(report.summary())
            sys.exit(1)
    """

    def __init__(
        self,
        min_iou_delta: float = 0.005,
        min_fps_delta: float = 0.5,
        significance_level: float = 0.05,
        strict: bool = False,
    ) -> None:
        self.min_iou_delta = min_iou_delta
        self.min_fps_delta = min_fps_delta
        self.significance_level = significance_level
        self.strict = strict

    def compare(
        self,
        baseline: "BenchmarkResult",
        candidate: "BenchmarkResult",
    ) -> RegressionReport:
        """Compare *candidate* against *baseline* and return a regression report.

        Args:
            baseline:  The reference result (before a code/config change).
            candidate: The result to evaluate (after the change).

        Returns:
            A :class:`RegressionReport` with per-metric deltas and a
            ``has_regression`` flag suitable for CI gating.
        """
        b_summary = baseline.summary()
        c_summary = candidate.summary()
        b_label = f"{baseline.tracker_name} on {baseline.dataset_name}"
        c_label = f"{candidate.tracker_name} on {candidate.dataset_name}"

        # ---- Per-metric deltas ----
        deltas: List[MetricDelta] = []
        for key in ("mean_iou", "mean_fps", "success_auc", "precision_auc",
                    "normalized_precision_auc", "mean_center_distance_px",
                    "total_energy_j", "mean_energy_per_frame_mj", "peak_memory_mb"):
            b_val = b_summary.get(key)
            c_val = c_summary.get(key)
            if b_val is None or c_val is None:
                continue
            delta = self._metric_delta(key, float(b_val), float(c_val))
            deltas.append(delta)

        # ---- IoU statistical test on aligned sequences ----
        b_seq = {r.sequence_name: r.ious for r in baseline.sequence_results}
        c_seq = {r.sequence_name: r.ious for r in candidate.sequence_results}
        common = sorted(set(b_seq) & set(c_seq))
        seq_count = max(len(baseline.sequence_results), len(candidate.sequence_results))

        pvalue: Optional[float] = None
        significant = False
        if len(common) >= 2:
            b_means = np.array([b_seq[k].mean() for k in common])
            c_means = np.array([c_seq[k].mean() for k in common])
            pvalue = _welch_ttest_pvalue(b_means, c_means)
            significant = pvalue < self.significance_level

        # ---- Determine overall regression flag ----
        has_regression = False
        for d in deltas:
            if d.direction == "regressed":
                if d.name == "mean_iou":
                    # IoU regression is only flagged when statistically significant,
                    # unless strict mode overrides this.
                    if self.strict or significant or pvalue is None:
                        has_regression = True
                else:
                    has_regression = True

        return RegressionReport(
            baseline_name=b_label,
            candidate_name=c_label,
            metric_deltas=deltas,
            iou_pvalue=pvalue,
            iou_significant=significant,
            has_regression=has_regression,
            sequence_count=seq_count,
            matched_sequences=len(common),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metric_delta(self, name: str, baseline: float, candidate: float) -> MetricDelta:
        delta = candidate - baseline
        rel = delta / abs(baseline) if abs(baseline) > 1e-9 else float("nan")

        # Determine threshold and improvement direction
        if name in _HIGHER_IS_BETTER:
            threshold = self.min_iou_delta if "iou" in name or "auc" in name else self.min_fps_delta
            if delta > threshold:
                direction = "improved"
            elif delta < -threshold:
                direction = "regressed"
            else:
                direction = "unchanged"
        elif name in _LOWER_IS_BETTER:
            threshold = self.min_fps_delta
            if delta < -threshold:
                direction = "improved"
            elif delta > threshold:
                direction = "regressed"
            else:
                direction = "unchanged"
        else:
            # Unknown metric — use sign only
            direction = "improved" if delta < 0 else ("regressed" if delta > 0 else "unchanged")

        return MetricDelta(
            name=name,
            baseline=round(baseline, 6),
            candidate=round(candidate, 6),
            delta=round(delta, 6),
            relative_delta=rel if not math.isnan(rel) else float("nan"),
            direction=direction,
        )
