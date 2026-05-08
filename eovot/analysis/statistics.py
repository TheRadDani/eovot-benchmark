"""Statistical significance testing for tracker comparison.

Implements a distribution-free pairwise significance test
(Mann-Whitney U) with bootstrap confidence intervals and Cohen's d
effect size.  No external dependencies beyond the Python standard
library and NumPy are required.

Typical usage::

    from eovot.analysis.statistics import TrackerStatistics

    stats = TrackerStatistics(alpha=0.05, n_bootstrap=2000)
    result = stats.compare(mosse_result_dict, kcf_result_dict)
    print(f"p={result.p_value:.4f}  significant={result.significant}")

    comparisons = stats.pairwise_comparison([r1, r2, r3])
    print(stats.significance_table(comparisons))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Map user-facing metric names to keys in per-sequence result dicts.
_METRIC_KEY: Dict[str, str] = {
    "iou": "mean_iou",
    "fps": "fps",
    "latency_ms": "mean_latency_ms",
    "memory_mb": "peak_memory_mb",
    "energy_j": "energy_j",
}


@dataclass
class ComparisonResult:
    """Outcome of a pairwise statistical test between two trackers.

    Attributes:
        tracker_a:    Name of the first tracker.
        tracker_b:    Name of the second tracker.
        metric:       Metric that was compared (e.g. ``"iou"``).
        u_statistic:  Mann-Whitney U statistic.
        p_value:      Two-sided p-value from the normal approximation.
        effect_size:  Cohen's d (positive means tracker_a > tracker_b).
        mean_a:       Mean per-sequence score for tracker A.
        mean_b:       Mean per-sequence score for tracker B.
        ci_lower:     Lower bound of the 95% bootstrap CI for mean_a.
        ci_upper:     Upper bound of the 95% bootstrap CI for mean_a.
        significant:  ``True`` when ``p_value < alpha``.
    """

    tracker_a: str
    tracker_b: str
    metric: str
    u_statistic: float
    p_value: float
    effect_size: float
    mean_a: float
    mean_b: float
    ci_lower: float
    ci_upper: float
    significant: bool

    @property
    def winner(self) -> Optional[str]:
        """Tracker with the higher mean, or ``None`` when not significant."""
        if not self.significant:
            return None
        return self.tracker_a if self.mean_a > self.mean_b else self.tracker_b


class TrackerStatistics:
    """Publication-quality statistical comparison between tracker results.

    Uses the Mann-Whitney U test (non-parametric, distribution-free) with
    a normal approximation for the p-value.  Bootstrap resampling provides
    confidence intervals on the mean score, and Cohen's d quantifies the
    practical effect size.

    All methods accept the dict format produced by
    ``BenchmarkResult.to_dict()``::

        {
            "summary": {"tracker": "MOSSE", "mean_iou": 0.52, ...},
            "sequences": [
                {"sequence_name": "Basketball", "mean_iou": 0.48, "fps": 320.0, ...},
                ...
            ]
        }

    Args:
        alpha:       Significance level.  Default: ``0.05``.
        n_bootstrap: Number of bootstrap resamples for CI estimation.
                     Default: ``2000``.
        seed:        Random seed for reproducible bootstrap sampling.
                     Default: ``0``.

    Example::

        stats = TrackerStatistics()
        result = stats.compare(mosse_result, kcf_result, metric="iou")
        print(stats.significance_table([result]))
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 2000,
        seed: int = 0,
    ) -> None:
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        result_a: Dict,
        result_b: Dict,
        metric: str = "iou",
    ) -> ComparisonResult:
        """Pairwise Mann-Whitney U test between two tracker result dicts.

        Args:
            result_a: ``BenchmarkResult.to_dict()`` output for tracker A.
            result_b: ``BenchmarkResult.to_dict()`` output for tracker B.
            metric:   Per-sequence metric to compare.  Supported values:
                      ``"iou"`` (default), ``"fps"``, ``"latency_ms"``,
                      ``"memory_mb"``, ``"energy_j"``.

        Returns:
            :class:`ComparisonResult` with test statistic, p-value, effect
            size, and bootstrap CI.

        Raises:
            ValueError: If the requested metric is absent from the results.
        """
        name_a = result_a.get("summary", {}).get("tracker", "A")
        name_b = result_b.get("summary", {}).get("tracker", "B")
        scores_a = self._extract_scores(result_a, metric)
        scores_b = self._extract_scores(result_b, metric)

        u_stat, p_value = self._mannwhitneyu(scores_a, scores_b)
        effect = self._cohens_d(scores_a, scores_b)
        ci_lo, ci_hi = self._bootstrap_ci(scores_a)

        return ComparisonResult(
            tracker_a=name_a,
            tracker_b=name_b,
            metric=metric,
            u_statistic=float(u_stat),
            p_value=float(p_value),
            effect_size=float(effect),
            mean_a=float(scores_a.mean()),
            mean_b=float(scores_b.mean()),
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            significant=bool(p_value < self.alpha),
        )

    def pairwise_comparison(
        self,
        results: List[Dict],
        metric: str = "iou",
    ) -> List[ComparisonResult]:
        """All-pairs pairwise comparison for a list of tracker result dicts.

        Args:
            results: List of dicts from ``BenchmarkResult.to_dict()``.
            metric:  Metric to compare.  Default: ``"iou"``.

        Returns:
            List of :class:`ComparisonResult`, one per unique ordered pair.
        """
        comparisons: List[ComparisonResult] = []
        for i, ra in enumerate(results):
            for rb in results[i + 1 :]:
                comparisons.append(self.compare(ra, rb, metric))
        return comparisons

    def significance_table(self, comparisons: List[ComparisonResult]) -> str:
        """Render a Markdown significance table.

        Args:
            comparisons: Output of :meth:`pairwise_comparison` or a manual
                list of :meth:`compare` calls.

        Returns:
            Multi-line Markdown string suitable for GitHub or a ``.md`` file.
        """
        header = (
            "| Tracker A | Tracker B | Metric | U-stat | p-value "
            "| Cohen's d | Significant |"
        )
        sep = (
            "|-----------|-----------|--------|-------:|--------:"
            "|----------:|:-----------:|"
        )
        lines = [header, sep]
        for c in comparisons:
            sig = "Yes" if c.significant else "No"
            lines.append(
                f"| {c.tracker_a} | {c.tracker_b} | {c.metric} "
                f"| {c.u_statistic:.1f} | {c.p_value:.4f} "
                f"| {c.effect_size:.3f} | {sig} |"
            )
        return "\n".join(lines)

    def latex_table(self, comparisons: List[ComparisonResult]) -> str:
        """Render a LaTeX booktabs significance table.

        Requires ``\\usepackage{booktabs}`` in the document preamble.

        Args:
            comparisons: Output of :meth:`pairwise_comparison`.

        Returns:
            Multi-line LaTeX string ready to paste into a paper.
        """
        rows = []
        for c in comparisons:
            sig = r"\checkmark" if c.significant else "--"
            rows.append(
                f"  {c.tracker_a} & {c.tracker_b} & {c.metric} "
                f"& {c.u_statistic:.1f} & {c.p_value:.4f} "
                f"& {c.effect_size:.3f} & {sig} \\\\"
            )
        header = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\begin{tabular}{llcrrrr}\n"
            "\\toprule\n"
            "Tracker A & Tracker B & Metric & U & p-value & Cohen's d & Sig.\\\\\n"
            "\\midrule\n"
        )
        footer = "\\bottomrule\n\\end{tabular}\n\\end{table}"
        return header + "\n".join(rows) + "\n" + footer

    def auc_confidence_interval(
        self,
        result: Dict,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean IoU across sequences.

        Args:
            result: Dict from ``BenchmarkResult.to_dict()``.
            ci:     Confidence level.  Default: ``0.95``.

        Returns:
            ``(lower, upper)`` bounds on mean IoU.
        """
        scores = self._extract_scores(result, "iou")
        return self._bootstrap_ci(scores, ci=ci)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scores(result: Dict, metric: str) -> np.ndarray:
        """Pull per-sequence scalar values from a result dict."""
        key = _METRIC_KEY.get(metric, metric)
        sequences = result.get("sequences", [])
        values = [float(s[key]) for s in sequences if key in s]
        if not values:
            available = list(sequences[0]) if sequences else []
            raise ValueError(
                f"No per-sequence data for metric '{metric}' (key '{key}'). "
                f"Available keys: {available}"
            )
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _rankdata(x: np.ndarray) -> np.ndarray:
        """Rank array elements using the average method for ties."""
        n = len(x)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j < n - 1 and x[order[j]] == x[order[j + 1]]:
                j += 1
            avg = (i + j + 2) / 2.0  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    @staticmethod
    def _normal_sf(z: float) -> float:
        """Upper-tail survival function of the standard normal."""
        return 0.5 * math.erfc(z / math.sqrt(2))

    def _mannwhitneyu(
        self, a: np.ndarray, b: np.ndarray
    ) -> Tuple[float, float]:
        """Two-sided Mann-Whitney U test using the normal approximation.

        Suitable for n > 20; for very small samples the approximation is
        less precise, but avoids any look-up table dependency.
        """
        n1, n2 = len(a), len(b)
        if n1 == 0 or n2 == 0:
            raise ValueError("Both score arrays must be non-empty.")
        combined = np.concatenate([a, b])
        ranks = self._rankdata(combined)
        r1 = ranks[:n1].sum()
        u1 = r1 - n1 * (n1 + 1) / 2.0
        u2 = n1 * n2 - u1
        u = min(u1, u2)
        mu = n1 * n2 / 2.0
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u - mu) / sigma if sigma > 0 else 0.0
        p = 2.0 * self._normal_sf(abs(z))
        return float(u), float(min(p, 1.0))

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Pooled-variance Cohen's d effect size."""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        pooled_var = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
        return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled_var + 1e-12))

    def _bootstrap_ci(
        self,
        scores: np.ndarray,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """Bootstrap CI for the mean of ``scores``."""
        n = len(scores)
        boot_means = np.fromiter(
            (
                self._rng.choice(scores, size=n, replace=True).mean()
                for _ in range(self.n_bootstrap)
            ),
            dtype=np.float64,
            count=self.n_bootstrap,
        )
        lo = (1.0 - ci) / 2.0
        hi = 1.0 - lo
        return float(np.quantile(boot_means, lo)), float(np.quantile(boot_means, hi))
