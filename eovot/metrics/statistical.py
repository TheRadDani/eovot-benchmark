"""Statistical analysis for research-grade tracker comparison.

Implements significance testing, effect size estimation, and bootstrap
confidence intervals to support statistically sound claims in tracker
evaluation studies.

Methods implemented:

- **Bootstrap percentile CI** — non-parametric confidence intervals for
  any scalar metric, requiring no distributional assumptions.
- **Wilcoxon signed-rank test** — paired non-parametric test for
  per-sequence metric distributions (paired because sequences are shared).
- **Cohen's d** — standardised mean difference effect size.
- **Vargha–Delaney A12** — probability of superiority; interpretation-free
  stochastic dominance measure.

These tools enable claims such as:
    "MOSSE achieves significantly lower IoU than KCF
     (p=0.003, Cohen's d=−0.71, A12=0.23,
      95% CI KCF: [0.52, 0.61] vs MOSSE: [0.38, 0.46])"

Example::

    from eovot.metrics.statistical import compare_trackers, rank_trackers

    cmp = compare_trackers(mosse_result, kcf_result, metric="mean_iou", seed=42)
    print(cmp)

    rankings = rank_trackers([mosse_result, kcf_result, csrt_result],
                             metric="mean_iou", seed=42)
    for r in rankings:
        print(r)

References:
    Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets."
    JMLR 7 (2006) — standard for multi-classifier significance testing.

    Vargha & Delaney, "A Critique and Improvement of the CL Common Language
    Effect Size Statistics of McGraw and Wong." JSS 25 (2000).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """Bootstrap percentile confidence interval for a scalar metric.

    Attributes:
        metric_name:     Name of the metric (e.g. ``"mean_iou"``).
        point_estimate:  Observed value on the original data.
        lower:           Lower bound of the confidence interval.
        upper:           Upper bound of the confidence interval.
        confidence_level: Nominal confidence level (e.g. ``0.95``).
        n_bootstrap:     Number of bootstrap resamples used.
    """

    metric_name: str
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 10_000

    def __str__(self) -> str:
        pct = int(self.confidence_level * 100)
        return (
            f"{self.metric_name}={self.point_estimate:.4f} "
            f"({pct}% CI [{self.lower:.4f}, {self.upper:.4f}])"
        )


@dataclass
class PairwiseComparison:
    """Result of a statistical comparison between two trackers.

    Attributes:
        tracker_a:      Name of tracker A.
        tracker_b:      Name of tracker B.
        metric_name:    Metric that was compared.
        mean_a:         Mean metric value for tracker A.
        mean_b:         Mean metric value for tracker B.
        p_value:        Two-sided p-value from Wilcoxon signed-rank test.
                        ``None`` if scipy is unavailable or n < 5.
        statistic:      Wilcoxon test statistic.  ``None`` if unavailable.
        effect_size_d:  Cohen's d (positive = A > B).
        a12:            Vargha–Delaney A12 (>0.71 = large effect for A).
        significant:    ``True`` if ``p_value < alpha``.
        alpha:          Significance threshold.
        ci_a:           Bootstrap CI for tracker A.
        ci_b:           Bootstrap CI for tracker B.
    """

    tracker_a: str
    tracker_b: str
    metric_name: str
    mean_a: float
    mean_b: float
    p_value: Optional[float]
    statistic: Optional[float]
    effect_size_d: float
    a12: float
    significant: bool
    alpha: float = 0.05
    ci_a: Optional[BootstrapCI] = None
    ci_b: Optional[BootstrapCI] = None

    @property
    def winner(self) -> str:
        """Tracker with the higher mean metric value."""
        return self.tracker_a if self.mean_a >= self.mean_b else self.tracker_b

    def __str__(self) -> str:
        p_str = f"p={self.p_value:.4f}" if self.p_value is not None else "p=N/A"
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"[{self.tracker_a} vs {self.tracker_b}] {self.metric_name}: "
            f"{self.mean_a:.4f} vs {self.mean_b:.4f} | "
            f"{p_str} | d={self.effect_size_d:.3f} | A12={self.a12:.3f} | {sig_str}"
        )


@dataclass
class TrackerRanking:
    """Leaderboard entry for a single tracker.

    Attributes:
        rank:           Position (1 = best).
        tracker_name:   Tracker identifier.
        metric_name:    Metric used for ranking.
        mean_value:     Mean metric value.
        ci:             Bootstrap CI for the mean.
        n_sequences:    Number of evaluated sequences.
    """

    rank: int
    tracker_name: str
    metric_name: str
    mean_value: float
    ci: BootstrapCI
    n_sequences: int

    def __str__(self) -> str:
        return (
            f"#{self.rank:>2}  {self.tracker_name:<25s}  "
            f"{self.metric_name}={self.mean_value:.4f}  {self.ci}  "
            f"n={self.n_sequences}"
        )

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "tracker_name": self.tracker_name,
            "metric_name": self.metric_name,
            "mean_value": round(self.mean_value, 4),
            "ci_lower": round(self.ci.lower, 4),
            "ci_upper": round(self.ci.upper, 4),
            "confidence_level": self.ci.confidence_level,
            "n_sequences": self.n_sequences,
        }


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    metric_name: str = "metric",
    statistic: Callable = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapCI:
    """Compute a bootstrap percentile confidence interval.

    Args:
        values:      1-D array of per-sequence metric values.
        metric_name: Label embedded in the returned object.
        statistic:   Aggregation function (default: ``np.mean``).
        confidence:  Desired confidence level (default: ``0.95``).
        n_bootstrap: Number of resamples (default: 10 000).
        rng:         NumPy random generator for reproducibility.

    Returns:
        :class:`BootstrapCI` with lower/upper bounds.

    Raises:
        ValueError: If *values* is empty.
    """
    if rng is None:
        rng = np.random.default_rng()
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot compute CI on an empty sequence.")

    point = float(statistic(values))
    boot = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    boot_stats = np.apply_along_axis(statistic, 1, boot)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100.0 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2)))

    return BootstrapCI(
        metric_name=metric_name,
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
    )


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation).

    Args:
        a: Sample from group A (per-sequence metric values).
        b: Sample from group B.

    Returns:
        Cohen's d — positive when A > B, negative when A < B.
        Returns ``0.0`` when pooled std is zero.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a.mean() - b.mean()
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    return float(diff / pooled) if pooled > 0 else 0.0


def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    """Vargha–Delaney A12 stochastic dominance measure.

    ``A12 = P(A > B) + 0.5 * P(A == B)``

    Interpretation:
        - 0.50 → no difference
        - 0.56 → small effect
        - 0.64 → medium effect
        - 0.71 → large effect (tracker A clearly dominates)

    Args:
        a: Per-sequence metric values for tracker A.
        b: Per-sequence metric values for tracker B.

    Returns:
        A12 in ``[0, 1]``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.5
    wins = sum(1 for ai in a for bi in b if ai > bi)
    ties = sum(1 for ai in a for bi in b if ai == bi)
    return float((wins + 0.5 * ties) / (m * n))


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float], bool]:
    """Paired Wilcoxon signed-rank test (two-sided).

    Appropriate when the same sequences are evaluated by both trackers,
    since per-sequence scores are paired by sequence identity.

    Args:
        a:     Per-sequence values for tracker A.
        b:     Per-sequence values for tracker B (same order).
        alpha: Significance threshold (default: ``0.05``).

    Returns:
        ``(statistic, p_value, significant)``.
        ``statistic`` and ``p_value`` are ``None`` when scipy is
        unavailable or when fewer than 5 paired samples are available.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))

    if not _SCIPY_AVAILABLE:
        warnings.warn(
            "scipy is not installed — Wilcoxon test skipped. "
            "Install: pip install scipy",
            stacklevel=2,
        )
        return None, None, False

    if n < 5:
        warnings.warn(
            f"Wilcoxon test requires ≥5 paired samples; got {n}. Test skipped.",
            stacklevel=2,
        )
        return None, None, False

    diff = a[:n] - b[:n]
    if np.all(diff == 0):
        return 0.0, 1.0, False

    stat, pval = _scipy_stats.wilcoxon(a[:n], b[:n], alternative="two-sided")
    return float(stat), float(pval), bool(pval < alpha)


# ---------------------------------------------------------------------------
# High-level comparison API
# ---------------------------------------------------------------------------

def _extract_per_sequence_metric(benchmark_result, metric: str) -> np.ndarray:
    """Extract a 1-D per-sequence metric array from a BenchmarkResult."""
    _VALID = ("mean_iou", "fps", "peak_memory_mb", "mean_latency_ms", "energy_per_frame_mj")
    if metric not in _VALID:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {_VALID}")

    values: List[float] = []
    for sr in benchmark_result.sequence_results:
        if metric == "mean_iou":
            values.append(float(sr.mean_iou))
        elif metric == "fps":
            values.append(float(sr.profiling.fps))
        elif metric == "peak_memory_mb":
            values.append(float(sr.profiling.peak_memory_mb))
        elif metric == "mean_latency_ms":
            values.append(float(sr.profiling.latency_mean_ms))
        elif metric == "energy_per_frame_mj":
            if getattr(sr, "energy", None) is not None:
                values.append(float(sr.energy.energy_per_frame_mj))
    return np.array(values, dtype=np.float64)


def compare_trackers(
    result_a,
    result_b,
    metric: str = "mean_iou",
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    seed: Optional[int] = None,
) -> PairwiseComparison:
    """Compare two BenchmarkResult objects with statistical rigor.

    Performs:
    - Wilcoxon signed-rank test on paired per-sequence values
    - Cohen's d and Vargha–Delaney A12 effect sizes
    - Bootstrap 95% CIs for both trackers

    Args:
        result_a:    BenchmarkResult for tracker A.
        result_b:    BenchmarkResult for tracker B.
        metric:      One of ``"mean_iou"``, ``"fps"``, ``"peak_memory_mb"``,
                     ``"mean_latency_ms"``, ``"energy_per_frame_mj"``.
        alpha:       Significance level (default: ``0.05``).
        n_bootstrap: Bootstrap resamples for CIs (default: 10 000).
        seed:        Random seed for reproducibility.

    Returns:
        :class:`PairwiseComparison` with all test results populated.

    Raises:
        ValueError: If either result has zero sequences.
    """
    rng = np.random.default_rng(seed)
    vals_a = _extract_per_sequence_metric(result_a, metric)
    vals_b = _extract_per_sequence_metric(result_b, metric)

    if len(vals_a) == 0 or len(vals_b) == 0:
        raise ValueError(
            f"Cannot compare trackers with zero sequences "
            f"(got {len(vals_a)} for '{result_a.tracker_name}', "
            f"{len(vals_b)} for '{result_b.tracker_name}')."
        )

    stat, pval, sig = wilcoxon_test(vals_a, vals_b, alpha=alpha)
    d = cohens_d(vals_a, vals_b)
    a12 = vargha_delaney_a12(vals_a, vals_b)
    ci_a = bootstrap_ci(vals_a, metric_name=metric, n_bootstrap=n_bootstrap, rng=rng)
    ci_b = bootstrap_ci(vals_b, metric_name=metric, n_bootstrap=n_bootstrap, rng=rng)

    return PairwiseComparison(
        tracker_a=result_a.tracker_name,
        tracker_b=result_b.tracker_name,
        metric_name=metric,
        mean_a=float(vals_a.mean()),
        mean_b=float(vals_b.mean()),
        p_value=pval,
        statistic=stat,
        effect_size_d=d,
        a12=a12,
        significant=sig,
        alpha=alpha,
        ci_a=ci_a,
        ci_b=ci_b,
    )


def rank_trackers(
    benchmark_results: list,
    metric: str = "mean_iou",
    higher_is_better: bool = True,
    n_bootstrap: int = 10_000,
    seed: Optional[int] = None,
) -> List[TrackerRanking]:
    """Rank a list of BenchmarkResult objects by a metric with CIs.

    Trackers with zero sequences for the requested metric are silently
    omitted from the ranking.

    Args:
        benchmark_results: List of BenchmarkResult objects.
        metric:            Metric to rank by.
        higher_is_better:  Use ``False`` for latency / memory metrics.
        n_bootstrap:       Bootstrap resamples for CIs.
        seed:              Random seed.

    Returns:
        List of :class:`TrackerRanking` sorted best → worst.
    """
    rng = np.random.default_rng(seed)
    entries: List[Tuple] = []
    for br in benchmark_results:
        vals = _extract_per_sequence_metric(br, metric)
        if len(vals) == 0:
            continue
        ci = bootstrap_ci(vals, metric_name=metric, n_bootstrap=n_bootstrap, rng=rng)
        entries.append((br.tracker_name, float(vals.mean()), ci, len(vals)))

    entries.sort(key=lambda x: x[1], reverse=higher_is_better)

    return [
        TrackerRanking(
            rank=i + 1,
            tracker_name=name,
            metric_name=metric,
            mean_value=mean,
            ci=ci,
            n_sequences=n,
        )
        for i, (name, mean, ci, n) in enumerate(entries)
    ]
