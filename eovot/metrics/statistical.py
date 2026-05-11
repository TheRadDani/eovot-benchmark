"""Statistical comparison framework for tracker benchmarking.

Provides two complementary tools for rigorous tracker comparison:

1. **Wilcoxon Signed-Rank Test** — non-parametric pairwise significance test
   for comparing two trackers across sequences.  Preferred over the paired
   t-test because per-sequence IoU scores are bounded and non-Gaussian.

2. **Bootstrap Confidence Intervals** — distribution-free interval estimation
   that quantifies uncertainty in aggregate metrics (mIoU, AUC, FPS) without
   assuming a particular distribution.

Both are implemented in pure NumPy / stdlib to avoid adding SciPy as a hard
dependency.

Usage example::

    from eovot.metrics.statistical import StatisticalComparison

    cmp = StatisticalComparison()
    scores = {
        "MOSSE": np.array([0.31, 0.28, 0.35, 0.29]),
        "KCF":   np.array([0.44, 0.41, 0.48, 0.43]),
    }
    summary = cmp.analyze(scores, metric="mean_iou")
    print(summary.to_markdown())

References:
    Wilcoxon (1945), "Individual Comparisons by Ranking Methods,"
      Biometrics Bulletin 1(6), 80–83.
    Efron & Tibshirani (1993), "An Introduction to the Bootstrap."
    Kristan et al. (2016), "The Visual Object Tracking VOT2016 Challenge Results."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a scalar statistic.

    Attributes:
        estimate:   Point estimate (sample mean).
        ci_lower:   Lower bound of the confidence interval.
        ci_upper:   Upper bound of the confidence interval.
        confidence: Nominal confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples used.
        n_samples:  Number of per-sequence values used.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float = 0.95
    n_bootstrap: int = 1000
    n_samples: int = 0

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return (
            f"{self.estimate:.4f} "
            f"[{pct}% CI: {self.ci_lower:.4f} – {self.ci_upper:.4f}, "
            f"n={self.n_samples}]"
        )

    def to_dict(self) -> Dict:
        return {
            "estimate": round(self.estimate, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
        }


@dataclass
class WilcoxonResult:
    """Result of a Wilcoxon signed-rank test between two trackers.

    Attributes:
        tracker_a:   Name of the first tracker.
        tracker_b:   Name of the second tracker.
        metric:      Metric being compared (e.g. ``"mean_iou"``).
        n_sequences: Number of sequences (pairs) used in the test.
        n_effective: Pairs remaining after zero-difference removal.
        statistic:   W+ statistic (sum of positive ranks).
        z_score:     Normal-approximation z-score.
        p_value:     Two-tailed p-value.
        significant: Whether ``p_value < alpha``.
        alpha:       Significance threshold used.
        direction:   Human-readable conclusion string.
    """

    tracker_a: str
    tracker_b: str
    metric: str
    n_sequences: int
    n_effective: int
    statistic: float
    z_score: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    direction: str = ""

    def __post_init__(self) -> None:
        if not self.direction:
            self.direction = "no significant difference"

    def __str__(self) -> str:
        sig_tag = f"** p={self.p_value:.4f} **" if self.significant else f"p={self.p_value:.4f}"
        return (
            f"Wilcoxon [{self.metric}] "
            f"{self.tracker_a} vs {self.tracker_b}: "
            f"W+={self.statistic:.1f}, z={self.z_score:.3f}, {sig_tag}  "
            f"→ {self.direction}"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_a": self.tracker_a,
            "tracker_b": self.tracker_b,
            "metric": self.metric,
            "n_sequences": self.n_sequences,
            "n_effective": self.n_effective,
            "statistic": round(self.statistic, 4),
            "z_score": round(self.z_score, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
            "alpha": self.alpha,
            "direction": self.direction,
        }


@dataclass
class StatisticalSummary:
    """Complete statistical analysis for a set of tracker comparisons.

    Attributes:
        metric:         The metric that was analysed (e.g. ``"mean_iou"``).
        tracker_cis:    Mapping from tracker name to its bootstrap CI.
        pairwise_tests: All pairwise Wilcoxon test results.
    """

    metric: str
    tracker_cis: Dict[str, BootstrapCI] = field(default_factory=dict)
    pairwise_tests: List[WilcoxonResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def significant_pairs(self) -> List[WilcoxonResult]:
        """Return only the pairs for which the test is significant."""
        return [t for t in self.pairwise_tests if t.significant]

    def ranking(self) -> List[Tuple[str, float]]:
        """Rank trackers by bootstrap point estimate (descending)."""
        return sorted(
            [(name, ci.estimate) for name, ci in self.tracker_cis.items()],
            key=lambda x: x[1],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "tracker_cis": {k: v.to_dict() for k, v in self.tracker_cis.items()},
            "pairwise_tests": [t.to_dict() for t in self.pairwise_tests],
            "ranking": self.ranking(),
        }

    def to_markdown(self) -> str:
        """Render the analysis as a Markdown string."""
        lines: List[str] = [f"## Statistical Analysis: `{self.metric}`\n"]

        # Confidence interval table
        lines.append("### Bootstrap 95% Confidence Intervals\n")
        lines.append("| Rank | Tracker | Estimate | CI Lower | CI Upper |")
        lines.append("|------|---------|----------|----------|----------|")
        for rank, (name, _) in enumerate(self.ranking(), 1):
            ci = self.tracker_cis[name]
            lines.append(
                f"| {rank} | {name} | {ci.estimate:.4f} "
                f"| {ci.ci_lower:.4f} | {ci.ci_upper:.4f} |"
            )

        # Pairwise test table
        lines.append("\n### Pairwise Wilcoxon Signed-Rank Tests (α=0.05)\n")
        lines.append(
            "| Tracker A | Tracker B | n | W+ | z | p-value | Sig | Direction |"
        )
        lines.append(
            "|-----------|-----------|---|----|---|---------|-----|-----------|"
        )
        for t in self.pairwise_tests:
            sig = "✓" if t.significant else ""
            lines.append(
                f"| {t.tracker_a} | {t.tracker_b} | {t.n_effective} "
                f"| {t.statistic:.1f} | {t.z_score:.3f} "
                f"| {t.p_value:.4f} | {sig} | {t.direction} |"
            )

        # Significant findings summary
        sig_pairs = self.significant_pairs()
        if sig_pairs:
            lines.append("\n### Significant Differences\n")
            for t in sig_pairs:
                lines.append(f"- **{t.direction}** (p={t.p_value:.4f})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core statistical routines
# ---------------------------------------------------------------------------


def _standard_normal_sf(z: float) -> float:
    """Survival function (1 - CDF) of the standard normal at *z*."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _average_ranks(abs_diffs: np.ndarray) -> np.ndarray:
    """Assign average ranks to |differences|, handling ties correctly."""
    n = len(abs_diffs)
    order = np.argsort(abs_diffs, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Average ranks for tied values
    unique_vals = np.unique(abs_diffs)
    for val in unique_vals:
        mask = abs_diffs == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def wilcoxon_signed_rank(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    tracker_a: str = "A",
    tracker_b: str = "B",
    metric: str = "metric",
) -> WilcoxonResult:
    """Wilcoxon signed-rank test comparing two per-sequence score arrays.

    Implements the normal approximation (n ≥ 10 recommended).  Handles ties
    in the differences via average-rank assignment.

    Args:
        a:         Per-sequence scores for tracker A — shape ``(N,)``.
        b:         Per-sequence scores for tracker B — shape ``(N,)``.
        alpha:     Significance level for the ``significant`` flag.
        tracker_a: Name label for tracker A.
        tracker_b: Name label for tracker B.
        metric:    Metric name embedded in the result.

    Returns:
        :class:`WilcoxonResult` with test statistics and conclusion.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"Score arrays must have equal length; got {len(a)} vs {len(b)}"
        )
    n_total = len(a)
    diffs = a - b

    # Remove zero differences (tied sequences)
    nonzero = diffs != 0.0
    diffs_nz = diffs[nonzero]
    n = len(diffs_nz)

    if n == 0:
        return WilcoxonResult(
            tracker_a=tracker_a,
            tracker_b=tracker_b,
            metric=metric,
            n_sequences=n_total,
            n_effective=0,
            statistic=0.0,
            z_score=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            direction="no significant difference (all sequences tied)",
        )

    abs_diffs = np.abs(diffs_nz)
    ranks = _average_ranks(abs_diffs)

    w_plus = float(ranks[diffs_nz > 0].sum())

    # Tie-corrected variance
    unique_abs, tie_counts = np.unique(abs_diffs, return_counts=True)
    tie_correction = float(np.sum(tie_counts ** 3 - tie_counts)) / 48.0
    variance = float(n * (n + 1) * (2 * n + 1)) / 24.0 - tie_correction
    std = math.sqrt(max(variance, 1e-12))

    expected = n * (n + 1) / 4.0
    # Continuity correction toward zero
    diff_from_mean = w_plus - expected
    corrected = diff_from_mean - 0.5 if diff_from_mean > 0 else diff_from_mean + 0.5
    z = corrected / std

    p_value = min(1.0, 2.0 * _standard_normal_sf(abs(z)))
    significant = p_value < alpha

    mean_diff = float(diffs_nz.mean())
    if significant:
        direction = (
            f"{tracker_a} > {tracker_b}"
            if mean_diff > 0
            else f"{tracker_b} > {tracker_a}"
        )
    else:
        direction = "no significant difference"

    return WilcoxonResult(
        tracker_a=tracker_a,
        tracker_b=tracker_b,
        metric=metric,
        n_sequences=n_total,
        n_effective=n,
        statistic=w_plus,
        z_score=z,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        direction=direction,
    )


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapCI:
    """Compute a bootstrap confidence interval for the mean of *values*.

    Args:
        values:     1-D array of per-sequence metric values.
        confidence: Desired confidence level (default ``0.95``).
        n_bootstrap: Number of bootstrap resamples (default ``1000``).
        rng:        Optional NumPy random generator for reproducibility.

    Returns:
        :class:`BootstrapCI` with point estimate and interval bounds.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return BootstrapCI(
            estimate=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            n_samples=0,
        )

    if rng is None:
        rng = np.random.default_rng()

    n = len(values)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        estimate=float(values.mean()),
        ci_lower=lo,
        ci_upper=hi,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# High-level analyser
# ---------------------------------------------------------------------------


class StatisticalComparison:
    """High-level interface for comparing multiple trackers statistically.

    Args:
        alpha:       Significance level for Wilcoxon tests (default ``0.05``).
        n_bootstrap: Bootstrap resamples for CI estimation (default ``1000``).
        seed:        Random seed for reproducible bootstrap results.

    Example::

        cmp = StatisticalComparison(seed=42)
        per_seq_ious = {
            "MOSSE": np.array([0.31, 0.28, 0.35, 0.29, 0.33]),
            "KCF":   np.array([0.44, 0.41, 0.48, 0.43, 0.45]),
            "CSRT":  np.array([0.56, 0.52, 0.59, 0.54, 0.57]),
        }
        summary = cmp.analyze(per_seq_ious, metric="mean_iou")
        print(summary.to_markdown())
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(seed)

    def bootstrap_ci(self, values: np.ndarray) -> BootstrapCI:
        """Compute bootstrap CI for a single tracker's per-sequence scores."""
        return bootstrap_ci(
            values,
            confidence=1.0 - self.alpha,
            n_bootstrap=self.n_bootstrap,
            rng=self._rng,
        )

    def wilcoxon(
        self,
        a: np.ndarray,
        b: np.ndarray,
        tracker_a: str = "A",
        tracker_b: str = "B",
        metric: str = "metric",
    ) -> WilcoxonResult:
        """Run a Wilcoxon signed-rank test for a single tracker pair."""
        return wilcoxon_signed_rank(
            a, b, alpha=self.alpha,
            tracker_a=tracker_a, tracker_b=tracker_b, metric=metric,
        )

    def analyze(
        self,
        per_sequence_scores: Dict[str, np.ndarray],
        metric: str = "metric",
    ) -> StatisticalSummary:
        """Run full statistical analysis across all tracker pairs.

        Computes bootstrap CIs for every tracker and performs all pairwise
        Wilcoxon signed-rank tests.

        Args:
            per_sequence_scores: Mapping from tracker name to a 1-D array of
                per-sequence metric values (e.g. per-sequence mean IoU).
                All arrays must have equal length (same sequences).
            metric: Human-readable metric name embedded in the output.

        Returns:
            :class:`StatisticalSummary` with CIs and pairwise test results.
        """
        tracker_names = list(per_sequence_scores.keys())

        # Validate sequence counts match
        lengths = {name: len(v) for name, v in per_sequence_scores.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"All trackers must have the same number of sequences. "
                f"Got: {lengths}"
            )

        # Bootstrap CIs
        tracker_cis: Dict[str, BootstrapCI] = {}
        for name, scores in per_sequence_scores.items():
            tracker_cis[name] = self.bootstrap_ci(np.asarray(scores, dtype=float))

        # All-pairs Wilcoxon tests
        pairwise: List[WilcoxonResult] = []
        for name_a, name_b in combinations(tracker_names, 2):
            result = self.wilcoxon(
                np.asarray(per_sequence_scores[name_a], dtype=float),
                np.asarray(per_sequence_scores[name_b], dtype=float),
                tracker_a=name_a,
                tracker_b=name_b,
                metric=metric,
            )
            pairwise.append(result)

        return StatisticalSummary(
            metric=metric,
            tracker_cis=tracker_cis,
            pairwise_tests=pairwise,
        )

    def analyze_from_benchmark_results(
        self,
        results: List,  # List[BenchmarkResult] — avoid circular import
        metric: str = "mean_iou",
    ) -> StatisticalSummary:
        """Build per-sequence score arrays from a list of BenchmarkResults.

        Sequences are matched by name.  Only sequences that appear in **all**
        results are included (intersection).

        Args:
            results: List of :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects, one per tracker.
            metric: One of ``"mean_iou"``, ``"fps"``, ``"peak_memory_mb"``.

        Returns:
            :class:`StatisticalSummary`.
        """
        _SUPPORTED = {"mean_iou", "fps", "peak_memory_mb"}
        if metric not in _SUPPORTED:
            raise ValueError(f"metric must be one of {_SUPPORTED}, got {metric!r}")

        # Build per-tracker dicts: seq_name -> scalar
        tracker_seq_scores: Dict[str, Dict[str, float]] = {}
        for br in results:
            seq_dict: Dict[str, float] = {}
            for sr in br.sequence_results:
                if metric == "mean_iou":
                    seq_dict[sr.sequence_name] = sr.mean_iou
                elif metric == "fps":
                    seq_dict[sr.sequence_name] = sr.profiling.fps
                else:
                    seq_dict[sr.sequence_name] = sr.profiling.peak_memory_mb
            tracker_seq_scores[br.tracker_name] = seq_dict

        # Intersect sequence names
        all_seq_sets = [set(d.keys()) for d in tracker_seq_scores.values()]
        common_seqs = sorted(set.intersection(*all_seq_sets))

        if not common_seqs:
            raise ValueError("No common sequences found across all benchmark results.")

        per_sequence_scores: Dict[str, np.ndarray] = {
            tracker: np.array([scores[seq] for seq in common_seqs], dtype=float)
            for tracker, scores in tracker_seq_scores.items()
        }

        return self.analyze(per_sequence_scores, metric=metric)
