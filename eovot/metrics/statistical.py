"""Statistical significance testing for VOT tracker comparison.

Rigorous tracker comparison requires more than mean performance: confidence
intervals quantify measurement uncertainty, and significance tests establish
whether observed differences are reproducible or due to chance.  This module
provides publication-ready statistical analysis with no scipy dependency —
all tests are implemented using NumPy and the standard library.

Components
----------
BootstrapCI
    Percentile bootstrap confidence interval for any scalar metric
    (success AUC, mean IoU, precision AUC, …).

WilcoxonResult
    Two-sided Wilcoxon signed-rank test for a pair of trackers evaluated on
    the same set of sequences.  Non-parametric — makes no normality
    assumption.  Uses a normal approximation with continuity correction and
    tie correction for the variance term.

PairwiseSummary
    Full all-pairs comparison table with Bonferroni-corrected significance
    and rank-biserial effect sizes, formatted as a Markdown table for
    direct embedding in paper appendices.

Example::

    import numpy as np
    from eovot.metrics.statistical import StatisticalTestEngine

    rng = np.random.default_rng(0)
    mosse_ious = rng.uniform(0.25, 0.55, size=50).tolist()
    kcf_ious   = rng.uniform(0.30, 0.60, size=50).tolist()
    csrt_ious  = rng.uniform(0.45, 0.75, size=50).tolist()

    engine = StatisticalTestEngine(alpha=0.05, n_bootstrap=10_000, seed=42)

    # Confidence interval for a single tracker
    ci = engine.bootstrap_ci(mosse_ious, metric_name="success_AUC")
    print(ci)

    # Pairwise significance
    result = engine.wilcoxon_test(mosse_ious, kcf_ious, "MOSSE", "KCF")
    print(result)

    # Full comparison table
    summary = engine.pairwise_report(
        {"MOSSE": mosse_ious, "KCF": kcf_ious, "CSRT": csrt_ious}
    )
    print(summary.to_markdown())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_with_ties(arr: np.ndarray) -> np.ndarray:
    """Assign average ranks to equal elements (standard competition ranking).

    Args:
        arr: 1-D non-negative float array to rank.

    Returns:
        Float array of the same shape; tied elements receive their mean rank.
    """
    n = len(arr)
    idx = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[idx] = np.arange(1, n + 1, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[idx[j]] == arr[idx[j + 1]]:
            j += 1
        if j > i:
            avg = float(ranks[idx[i]] + ranks[idx[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _normal_cdf_upper(z: float) -> float:
    """P(Z > z) for Z ~ N(0, 1) using math.erfc.

    Accurate to ~15 significant figures; no scipy required.
    """
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _effect_size_label(r: float) -> str:
    """Descriptive label for rank-biserial correlation effect size."""
    abs_r = abs(r)
    if abs_r < 0.10:
        return "negligible"
    if abs_r < 0.30:
        return "small"
    if abs_r < 0.50:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    """Percentile bootstrap confidence interval for a scalar metric.

    Attributes:
        metric_name: Name of the metric being estimated.
        observed: Point estimate (mean of the original sample).
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        confidence_level: Nominal confidence level, e.g. ``0.95`` for 95 %.
        n_bootstrap: Number of bootstrap resamples used.
        n_samples: Number of observations in the original sample.
    """

    metric_name: str
    observed: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int
    n_samples: int

    def __str__(self) -> str:
        pct = int(round(self.confidence_level * 100))
        return (
            f"BootstrapCI({self.metric_name}={self.observed:.4f}  "
            f"{pct}% CI [{self.lower:.4f}, {self.upper:.4f}]  "
            f"n={self.n_samples}, B={self.n_bootstrap})"
        )


@dataclass
class WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test result for one tracker pair.

    Attributes:
        tracker_a: Name of the first tracker (treated as "A" in effect size).
        tracker_b: Name of the second tracker.
        w_plus: Sum of ranks for sequences where A outperforms B.
        w_minus: Sum of ranks for sequences where B outperforms A.
        z_statistic: Normal approximation z-score (with continuity correction).
        p_value: Two-tailed p-value (reliable for n ≥ 10 pairs).
        alpha: Significance threshold used to set ``significant``.
        significant: ``True`` if ``p_value < alpha``.
        effect_size: Rank-biserial correlation in ``[-1, 1]``.
            ``+1``: A wins every comparison; ``-1``: B wins every comparison.
        effect_label: Descriptive label: ``"negligible"``, ``"small"``,
            ``"medium"``, or ``"large"``.
        n_pairs: Number of paired observations (zero-difference pairs excluded).
        direction: Human-readable winner: ``"A > B"``, ``"B > A"``, or
            ``"no difference"``.
        warning: Optional note when assumptions may not hold (e.g. n < 10).
    """

    tracker_a: str
    tracker_b: str
    w_plus: float
    w_minus: float
    z_statistic: float
    p_value: float
    alpha: float
    significant: bool
    effect_size: float
    effect_label: str
    n_pairs: int
    direction: str
    warning: Optional[str] = None

    def __str__(self) -> str:
        sig = "✓" if self.significant else "✗"
        return (
            f"Wilcoxon({self.tracker_a} vs {self.tracker_b})  "
            f"z={self.z_statistic:.3f}  p={self.p_value:.4f}  "
            f"sig={sig}  r={self.effect_size:.3f} ({self.effect_label})  "
            f"direction={self.direction}  n={self.n_pairs}"
        )


@dataclass
class PairwiseSummary:
    """Full all-pairs significance comparison for a set of trackers.

    Attributes:
        comparisons: All :class:`WilcoxonResult` objects, one per ordered pair.
        trackers: Ordered list of tracker names.
        n_comparisons: Number of pairwise tests performed.
        alpha_raw: Unadjusted significance level.
        alpha_bonferroni: Bonferroni-corrected threshold (α / n_comparisons).
    """

    comparisons: List[WilcoxonResult]
    trackers: List[str]
    n_comparisons: int
    alpha_raw: float
    alpha_bonferroni: float

    def significant_pairs(self) -> List[WilcoxonResult]:
        """Return comparisons significant at the Bonferroni-corrected level."""
        return [c for c in self.comparisons if c.p_value < self.alpha_bonferroni]

    def to_markdown(self) -> str:
        """Format the pairwise comparison as a Markdown table.

        Returns:
            A multi-line Markdown string with one row per tracker pair,
            suitable for embedding in a paper appendix or README.

        Example output::

            | Tracker A | Tracker B | W+   | W-   | z     | p-value | Sig? | r     | Effect  | Direction |
            |-----------|-----------|-----:|-----:|------:|--------:|:----:|------:|---------|-----------|
            | CSRT      | MOSSE     | 1225 | 25   | 4.312 | <0.0001 |  ✓  | 0.96  | large   | A > B     |
        """
        lines = [
            f"*Bonferroni-corrected α = {self.alpha_bonferroni:.4f} "
            f"({self.n_comparisons} comparisons)*\n",
            "| Tracker A | Tracker B | W+ | W- | z | p-value | Sig (corrected)? | r | Effect | Direction |",
            "|-----------|-----------|---:|---:|--:|--------:|:----------------:|--:|--------|-----------|",
        ]
        for c in self.comparisons:
            p_fmt = f"{c.p_value:.4f}" if c.p_value >= 0.0001 else "<0.0001"
            sig = "✓" if c.p_value < self.alpha_bonferroni else "✗"
            lines.append(
                f"| {c.tracker_a} | {c.tracker_b} "
                f"| {c.w_plus:.0f} | {c.w_minus:.0f} "
                f"| {c.z_statistic:.3f} | {p_fmt} "
                f"| {sig} "
                f"| {c.effect_size:.3f} | {c.effect_label} "
                f"| {c.direction} |"
            )
        if self.significant_pairs():
            lines.append(
                f"\n**{len(self.significant_pairs())} of {self.n_comparisons} "
                f"pairs significant at Bonferroni-corrected α.**"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class StatisticalTestEngine:
    """Bootstrap confidence intervals and Wilcoxon significance tests for VOT.

    Args:
        alpha: Significance level for hypothesis testing (default ``0.05``).
        n_bootstrap: Number of bootstrap resamples for CI estimation
            (default ``10_000``). Higher values reduce Monte Carlo variance.
        seed: Random seed for reproducible bootstrap results.

    Example::

        engine = StatisticalTestEngine(alpha=0.05, n_bootstrap=10_000, seed=42)
        ci = engine.bootstrap_ci(per_sequence_ious, "mean_IoU")
        summary = engine.pairwise_report({"MOSSE": mosse_ious, "KCF": kcf_ious})
        print(summary.to_markdown())
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 10_000,
        seed: int = 42,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100, got {n_bootstrap}.")
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        scores: Sequence[float],
        metric_name: str = "metric",
    ) -> BootstrapCI:
        """Compute a percentile bootstrap confidence interval for the mean.

        Resamples *scores* with replacement ``n_bootstrap`` times, computes
        the mean of each resample, and returns the ``(alpha/2, 1-alpha/2)``
        percentiles as the CI bounds.

        Args:
            scores: Per-sequence scalar scores (e.g. success AUC values).
                At least 2 observations are required.
            metric_name: Label to embed in the returned :class:`BootstrapCI`.

        Returns:
            :class:`BootstrapCI` with observed mean and CI bounds.

        Raises:
            ValueError: If fewer than 2 observations are provided.
        """
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1 or len(arr) < 2:
            raise ValueError(
                f"bootstrap_ci requires a 1-D array with ≥ 2 elements, "
                f"got shape {arr.shape}."
            )

        rng = np.random.default_rng(self.seed)
        n = len(arr)
        boot_means = np.empty(self.n_bootstrap, dtype=np.float64)
        for i in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_means[i] = arr[idx].mean()

        lo_pct = 100.0 * self.alpha / 2.0
        hi_pct = 100.0 * (1.0 - self.alpha / 2.0)

        return BootstrapCI(
            metric_name=metric_name,
            observed=float(arr.mean()),
            lower=float(np.percentile(boot_means, lo_pct)),
            upper=float(np.percentile(boot_means, hi_pct)),
            confidence_level=1.0 - self.alpha,
            n_bootstrap=self.n_bootstrap,
            n_samples=n,
        )

    # ------------------------------------------------------------------
    # Wilcoxon signed-rank test
    # ------------------------------------------------------------------

    def wilcoxon_test(
        self,
        scores_a: Sequence[float],
        scores_b: Sequence[float],
        tracker_a: str = "A",
        tracker_b: str = "B",
    ) -> WilcoxonResult:
        """Two-sided Wilcoxon signed-rank test for paired per-sequence scores.

        Tests H₀: the median difference between tracker A and tracker B is
        zero.  Appropriate when scores come from the same sequences evaluated
        by both trackers (paired design).  No normality assumption.

        Uses a normal approximation with continuity correction, valid for
        n ≥ 10 non-zero-difference pairs.  A ``warning`` is attached to the
        result when n < 10.

        Args:
            scores_a: Per-sequence scores for tracker A (e.g. mean IoU per
                sequence).
            scores_b: Per-sequence scores for tracker B.  Length may differ
                from *scores_a*; only the first ``min(len_a, len_b)`` pairs
                are used.
            tracker_a: Human-readable name for tracker A.
            tracker_b: Human-readable name for tracker B.

        Returns:
            :class:`WilcoxonResult` with test statistic, p-value, and effect.

        Raises:
            ValueError: If fewer than 2 paired observations are provided.
        """
        a = np.asarray(scores_a, dtype=np.float64).ravel()
        b = np.asarray(scores_b, dtype=np.float64).ravel()
        n_total = min(len(a), len(b))
        if n_total < 2:
            raise ValueError(
                f"wilcoxon_test requires at least 2 paired observations, "
                f"got {n_total}."
            )
        a, b = a[:n_total], b[:n_total]

        d = a - b
        nonzero = d != 0.0
        d_nz = d[nonzero]
        n = len(d_nz)

        warning: Optional[str] = None

        if n == 0:
            # All differences are zero — cannot reject H0
            return WilcoxonResult(
                tracker_a=tracker_a, tracker_b=tracker_b,
                w_plus=0.0, w_minus=0.0,
                z_statistic=0.0, p_value=1.0, alpha=self.alpha,
                significant=False, effect_size=0.0, effect_label="negligible",
                n_pairs=0, direction="no difference",
                warning="All paired differences are zero; cannot reject H₀.",
            )

        if n < 10:
            warning = (
                f"Only {n} non-zero pairs — normal approximation may be "
                f"unreliable for n < 10. Interpret p-value with caution."
            )

        abs_d = np.abs(d_nz)
        ranks = _rank_with_ties(abs_d)

        w_plus = float(np.sum(ranks[d_nz > 0]))
        w_minus = float(np.sum(ranks[d_nz < 0]))
        w_total = w_plus + w_minus  # = n*(n+1)/2

        # Variance with tie correction
        expected = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0

        # Tie correction: subtract Σ(t³ - t)/48 for each tie group of size t
        unique_abs, counts = np.unique(abs_d, return_counts=True)
        tie_sum = float(np.sum(counts ** 3 - counts))
        variance -= tie_sum / 48.0

        if variance <= 0.0:
            variance = 1e-12  # degenerate case; p will be pushed toward 1

        # Continuity correction: shift toward expected by 0.5
        z = (w_plus - 0.5 - expected) / math.sqrt(variance)
        p_value = 2.0 * _normal_cdf_upper(abs(z))
        p_value = min(p_value, 1.0)

        # Rank-biserial correlation: (W+ - W-) / W_total
        effect_size = (w_plus - w_minus) / w_total if w_total > 0 else 0.0

        if abs(effect_size) < 1e-9:
            direction = "no difference"
        elif effect_size > 0:
            direction = f"{tracker_a} > {tracker_b}"
        else:
            direction = f"{tracker_b} > {tracker_a}"

        return WilcoxonResult(
            tracker_a=tracker_a,
            tracker_b=tracker_b,
            w_plus=w_plus,
            w_minus=w_minus,
            z_statistic=z,
            p_value=p_value,
            alpha=self.alpha,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            effect_label=_effect_size_label(effect_size),
            n_pairs=n,
            direction=direction,
            warning=warning,
        )

    # ------------------------------------------------------------------
    # Pairwise comparison
    # ------------------------------------------------------------------

    def pairwise_report(
        self,
        tracker_scores: Dict[str, Sequence[float]],
    ) -> PairwiseSummary:
        """Run all pairwise Wilcoxon tests with Bonferroni correction.

        Tests every ordered pair (A, B) where A != B.  Applies a Bonferroni
        correction to the raw α so that the family-wise error rate stays at
        ``self.alpha`` across all comparisons.

        Args:
            tracker_scores: Mapping from tracker name to per-sequence scores.
                All score lists should correspond to the same sequences.
                At least 2 trackers are required.

        Returns:
            :class:`PairwiseSummary` with all test results and the corrected
            significance threshold.

        Raises:
            ValueError: If fewer than 2 trackers are provided.
        """
        trackers = list(tracker_scores)
        if len(trackers) < 2:
            raise ValueError(
                f"pairwise_report requires at least 2 trackers, got {len(trackers)}."
            )

        # Generate all unordered pairs (i < j) to avoid duplication
        pairs: List[Tuple[str, str]] = []
        for i, a in enumerate(trackers):
            for b in trackers[i + 1:]:
                pairs.append((a, b))

        n_comparisons = len(pairs)
        alpha_bonferroni = self.alpha / n_comparisons if n_comparisons > 0 else self.alpha

        comparisons = [
            self.wilcoxon_test(
                tracker_scores[a], tracker_scores[b],
                tracker_a=a, tracker_b=b,
            )
            for a, b in pairs
        ]

        return PairwiseSummary(
            comparisons=comparisons,
            trackers=trackers,
            n_comparisons=n_comparisons,
            alpha_raw=self.alpha,
            alpha_bonferroni=alpha_bonferroni,
        )

    # ------------------------------------------------------------------
    # Convenience: CI table for multiple trackers
    # ------------------------------------------------------------------

    def ci_table(
        self,
        tracker_scores: Dict[str, Sequence[float]],
        metric_name: str = "metric",
    ) -> str:
        """Return a Markdown table of bootstrap CIs for every tracker.

        Args:
            tracker_scores: Mapping from tracker name to per-sequence scores.
            metric_name: Metric label displayed in the table header.

        Returns:
            Multi-line Markdown string with one row per tracker.
        """
        pct = int(round((1.0 - self.alpha) * 100))
        lines = [
            f"| Tracker | Mean {metric_name} | {pct}% CI Lower | {pct}% CI Upper | n |",
            f"|---------|------------|-----------|-----------|---|",
        ]
        for name, scores in tracker_scores.items():
            ci = self.bootstrap_ci(scores, metric_name=metric_name)
            lines.append(
                f"| {name} | {ci.observed:.4f} "
                f"| {ci.lower:.4f} | {ci.upper:.4f} "
                f"| {ci.n_samples} |"
            )
        return "\n".join(lines)
