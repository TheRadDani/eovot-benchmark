"""Benchmark regression detector for EOVOT CI/CD integration.

Compares two benchmark result summaries and classifies each metric as
IMPROVED / PASS / WARN / FAIL based on configurable percentage thresholds.

Motivation
----------
As the EOVOT tracker portfolio grows, it becomes easy to inadvertently
regress accuracy or speed when refactoring the benchmark engine, tweaking
tracker defaults, or upgrading dependencies.  A regression detector that
runs as a CI gate catches these changes before they land on ``main``.

Regression rule
---------------
For each scalar metric present in both results::

    delta          = current_value − baseline_value
    relative_pct   = delta / max(|baseline_value|, ε) × 100

Direction sensitivity:

* ``mean_iou``, ``success_auc``, ``precision_auc``, ``mean_fps``
  → higher is better.  A *negative* relative change is a regression.
* ``peak_memory_mb``, ``mean_latency_ms``, ``mean_center_distance_px``,
  ``mean_energy_per_frame_mj``
  → lower is better.  A *positive* relative change is a regression.

Status thresholds (applied to the *effective change*, sign-adjusted for
directionality):

+------------------+-----------------------------------+
| Status           | Effective change                  |
+==================+===================================+
| IMPROVED         | > +warn_pct                       |
+------------------+-----------------------------------+
| PASS             | in [−warn_pct, +warn_pct]         |
+------------------+-----------------------------------+
| WARN             | in [−fail_pct, −warn_pct)         |
+------------------+-----------------------------------+
| FAIL             | < −fail_pct                       |
+------------------+-----------------------------------+

Defaults: ``warn_pct = 2.0 %``, ``fail_pct = 5.0 %``.

Exit codes (for CI integration)
---------------------------------
* ``0`` — all metrics PASS or IMPROVED
* ``1`` — at least one WARN, no FAILs
* ``2`` — at least one FAIL

Typical usage
-------------
::

    # Python API
    from eovot.analysis.regression import RegressionDetector, load_scalars_from_file

    baseline = load_scalars_from_file("results/mosse_baseline.json")
    current  = load_scalars_from_file("results/mosse_current.json")

    detector = RegressionDetector(warn_pct=2.0, fail_pct=5.0)
    report   = detector.compare(baseline, current,
                                tracker_name="MOSSE", dataset_name="Synthetic")
    print(report.summary())
    raise SystemExit(report.exit_code)

    # CLI wrapper: scripts/diff_results.py
    python scripts/diff_results.py baseline.json current.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Metric directionality tables
# ---------------------------------------------------------------------------

#: Scalar metrics where a larger value is better.
HIGHER_IS_BETTER: frozenset = frozenset({
    "mean_iou",
    "success_auc",
    "precision_auc",
    "mean_fps",
})

#: Scalar metrics where a smaller value is better.
LOWER_IS_BETTER: frozenset = frozenset({
    "peak_memory_mb",
    "mean_latency_ms",
    "mean_center_distance_px",
    "mean_energy_per_frame_mj",
    "total_energy_j",
})

#: Union of all tracked scalar metrics.
ALL_TRACKED_METRICS: frozenset = HIGHER_IS_BETTER | LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class MetricStatus(str, Enum):
    """Per-metric regression status."""

    IMPROVED = "IMPROVED"
    """Metric improved beyond the warn threshold — good."""

    PASS = "PASS"
    """Metric changed within the acceptable noise band."""

    WARN = "WARN"
    """Metric regressed between warn and fail thresholds."""

    FAIL = "FAIL"
    """Metric regressed beyond the fail threshold."""

    MISSING = "MISSING"
    """Metric is absent from baseline or current results."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricDelta:
    """Per-metric comparison result.

    Attributes:
        name: Metric key (e.g. ``"mean_iou"``).
        baseline: Value in the reference run, or ``None`` if absent.
        current: Value in the current run, or ``None`` if absent.
        delta: ``current − baseline``, or ``None`` when either is missing.
        relative_pct: ``delta / |baseline| × 100``, or ``None``.
        status: Regression classification.
        higher_is_better: Directionality flag for this metric.
    """

    name: str
    baseline: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    relative_pct: Optional[float]
    status: MetricStatus
    higher_is_better: bool

    def __str__(self) -> str:
        b = f"{self.baseline:.4f}" if self.baseline is not None else "N/A"
        c = f"{self.current:.4f}" if self.current is not None else "N/A"
        p = f"{self.relative_pct:+.1f}%" if self.relative_pct is not None else "N/A"
        return f"{self.name}: {b} → {c} ({p}) [{self.status.value}]"


@dataclass
class RegressionReport:
    """Full comparison report between a baseline and a current benchmark run.

    Attributes:
        tracker_name: Tracker identifier (inferred from JSON or supplied).
        dataset_name: Dataset identifier.
        deltas: Per-metric :class:`MetricDelta` objects.
        warn_pct: Warn threshold used to produce this report.
        fail_pct: Fail threshold used to produce this report.
    """

    tracker_name: str
    dataset_name: str
    deltas: List[MetricDelta] = field(default_factory=list)
    warn_pct: float = 2.0
    fail_pct: float = 5.0

    # ------------------------------------------------------------------ #
    # Status queries                                                       #
    # ------------------------------------------------------------------ #

    @property
    def has_failures(self) -> bool:
        """``True`` when at least one metric has ``FAIL`` status."""
        return any(d.status == MetricStatus.FAIL for d in self.deltas)

    @property
    def has_warnings(self) -> bool:
        """``True`` when at least one metric has ``WARN`` status."""
        return any(d.status == MetricStatus.WARN for d in self.deltas)

    @property
    def exit_code(self) -> int:
        """CI-compatible exit code: 0 = pass, 1 = warn, 2 = fail."""
        if self.has_failures:
            return 2
        if self.has_warnings:
            return 1
        return 0

    @property
    def overall_status(self) -> str:
        """Human-readable overall verdict: ``"PASS"``, ``"WARN"``, or ``"FAIL"``."""
        if self.has_failures:
            return "FAIL"
        if self.has_warnings:
            return "WARN"
        return "PASS"

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def summary(self, color: bool = True) -> str:
        """Render a terminal-friendly comparison table.

        Args:
            color: Use ANSI colour codes for IMPROVED / PASS / WARN / FAIL.
                   Set ``False`` for plain CI log files.

        Returns:
            Multi-line string ready to ``print()``.
        """
        _COLORS = {
            MetricStatus.IMPROVED: "\033[32m",
            MetricStatus.PASS:     "\033[37m",
            MetricStatus.WARN:     "\033[33m",
            MetricStatus.FAIL:     "\033[31m",
            MetricStatus.MISSING:  "\033[35m",
        }
        _RESET = "\033[0m"
        _W = 68

        lines: List[str] = [
            f"\nEOVOT Regression Report — {self.tracker_name} on {self.dataset_name}",
            f"Thresholds: warn={self.warn_pct:.1f}%  fail={self.fail_pct:.1f}%",
            "─" * _W,
            f"{'Metric':<32} {'Baseline':>9} {'Current':>9} {'Δ %':>8}  {'Status'}",
            "─" * _W,
        ]

        for d in self.deltas:
            b_str = f"{d.baseline:.4f}" if d.baseline is not None else "N/A"
            c_str = f"{d.current:.4f}"  if d.current  is not None else "N/A"
            p_str = f"{d.relative_pct:+.1f}%" if d.relative_pct is not None else "N/A"
            row = f"{d.name:<32} {b_str:>9} {c_str:>9} {p_str:>8}  {d.status.value}"
            if color:
                row = _COLORS.get(d.status, "") + row + _RESET
            lines.append(row)

        lines.append("─" * _W)
        verdict = self.overall_status
        if color:
            verdict_color = (
                "\033[31m" if self.has_failures else
                "\033[33m" if self.has_warnings else
                "\033[32m"
            )
            lines.append(f"Overall: {verdict_color}{verdict}{_RESET}  (exit {self.exit_code})")
        else:
            lines.append(f"Overall: {verdict}  (exit {self.exit_code})")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialise the report to a plain dict suitable for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "dataset_name": self.dataset_name,
            "overall": self.overall_status.lower(),
            "exit_code": self.exit_code,
            "warn_pct": self.warn_pct,
            "fail_pct": self.fail_pct,
            "metrics": [
                {
                    "name": d.name,
                    "baseline": d.baseline,
                    "current": d.current,
                    "delta": d.delta,
                    "relative_pct": d.relative_pct,
                    "status": d.status.value,
                    "higher_is_better": d.higher_is_better,
                }
                for d in self.deltas
            ],
        }

    def to_json(self, path: Union[str, Path]) -> Path:
        """Write the report dict to a JSON file.

        Args:
            path: Destination file path.  Parent directories are created
                automatically.  A ``.json`` suffix is appended when missing.

        Returns:
            Resolved :class:`~pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return p


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class RegressionDetector:
    """Compare two scalar metric dicts and classify regressions.

    Args:
        warn_pct: Relative change (%) that triggers a WARN.  Default ``2.0``.
        fail_pct: Relative change (%) that triggers a FAIL.  Default ``5.0``.
        epsilon:  Guard value to avoid division by near-zero baselines.
            Default ``1e-6``.

    Raises:
        ValueError: If ``warn_pct ≤ 0`` or ``fail_pct ≤ warn_pct``.

    Example::

        detector = RegressionDetector(warn_pct=1.0, fail_pct=3.0)
        report = detector.compare(baseline_scalars, current_scalars,
                                  tracker_name="KCF", dataset_name="OTB100")
        print(report.summary())
        sys.exit(report.exit_code)
    """

    def __init__(
        self,
        warn_pct: float = 2.0,
        fail_pct: float = 5.0,
        epsilon: float = 1e-6,
    ) -> None:
        if warn_pct <= 0:
            raise ValueError(f"warn_pct must be positive, got {warn_pct}")
        if fail_pct <= warn_pct:
            raise ValueError(
                f"fail_pct ({fail_pct}) must be greater than warn_pct ({warn_pct})"
            )
        self.warn_pct = warn_pct
        self.fail_pct = fail_pct
        self.epsilon = epsilon

    def compare(
        self,
        baseline: Dict[str, float],
        current: Dict[str, float],
        tracker_name: str = "unknown",
        dataset_name: str = "unknown",
    ) -> RegressionReport:
        """Compare baseline vs current scalar metrics.

        Only metrics listed in :data:`ALL_TRACKED_METRICS` are evaluated.
        Metrics present in only one dict receive ``MISSING`` status.

        Args:
            baseline: Scalar metrics from the reference run.
            current: Scalar metrics from the run under evaluation.
            tracker_name: Label for the report header.
            dataset_name: Label for the report header.

        Returns:
            :class:`RegressionReport` with per-metric deltas and overall status.
        """
        report = RegressionReport(
            tracker_name=tracker_name,
            dataset_name=dataset_name,
            warn_pct=self.warn_pct,
            fail_pct=self.fail_pct,
        )

        # Evaluate all tracked metrics that appear in at least one dict
        keys = sorted(ALL_TRACKED_METRICS & (set(baseline) | set(current)))
        for key in keys:
            hib = key in HIGHER_IS_BETTER
            b_val = baseline.get(key)
            c_val = current.get(key)

            if b_val is None or c_val is None:
                report.deltas.append(MetricDelta(
                    name=key,
                    baseline=b_val,
                    current=c_val,
                    delta=None,
                    relative_pct=None,
                    status=MetricStatus.MISSING,
                    higher_is_better=hib,
                ))
                continue

            delta = c_val - b_val
            rel_pct = (delta / max(abs(b_val), self.epsilon)) * 100.0

            # Flip sign for lower-is-better metrics so the threshold logic
            # is uniform: positive effective_change = good, negative = bad.
            effective = rel_pct if hib else -rel_pct

            if effective > self.warn_pct:
                status = MetricStatus.IMPROVED
            elif effective >= -self.warn_pct:
                status = MetricStatus.PASS
            elif effective >= -self.fail_pct:
                status = MetricStatus.WARN
            else:
                status = MetricStatus.FAIL

            report.deltas.append(MetricDelta(
                name=key,
                baseline=b_val,
                current=c_val,
                delta=round(delta, 6),
                relative_pct=round(rel_pct, 3),
                status=status,
                higher_is_better=hib,
            ))

        return report

    def compare_results(
        self,
        baseline: "BenchmarkResult",  # noqa: F821  — forward ref
        current: "BenchmarkResult",   # noqa: F821
    ) -> RegressionReport:
        """Convenience wrapper: compare two ``BenchmarkResult`` objects directly.

        Args:
            baseline: Reference :class:`~eovot.benchmark.engine.BenchmarkResult`.
            current: Current  :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Returns:
            :class:`RegressionReport` for the pair.
        """
        b_scalars = _scalars_from_result(baseline)
        c_scalars = _scalars_from_result(current)
        return self.compare(
            b_scalars,
            c_scalars,
            tracker_name=baseline.tracker_name,
            dataset_name=baseline.dataset_name,
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_scalars_from_file(path: Union[str, Path]) -> Dict[str, float]:
    """Load tracked scalar metrics from a BenchmarkResult JSON file.

    Accepts both the nested ``{"summary": {...}, "sequences": [...]}`` format
    produced by :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict` and
    a flat ``{"mean_iou": ..., ...}`` dict (e.g. a bare summary).

    Args:
        path: Path to a JSON file written by
            :meth:`~eovot.benchmark.engine.BenchmarkResult.save`.

    Returns:
        Dict ``{metric_name: float}`` for every tracked metric found in the
        file.  Metrics not present in the file are silently omitted.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both {"summary": {...}} and flat dicts
    summary = data.get("summary", data)
    return {
        k: float(v)
        for k, v in summary.items()
        if k in ALL_TRACKED_METRICS and v is not None
    }


def _scalars_from_result(result: "BenchmarkResult") -> Dict[str, float]:  # noqa: F821
    """Extract tracked scalar metrics from a live BenchmarkResult object."""
    d: Dict[str, float] = {}
    summary = result.summary()
    for key in ALL_TRACKED_METRICS:
        if key in summary and summary[key] is not None:
            d[key] = float(summary[key])
    return d
