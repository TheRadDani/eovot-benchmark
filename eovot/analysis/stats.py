"""Statistical analysis utilities for EOVOT tracker comparisons.

Implements the standard statistical toolkit used in computer vision
benchmarking papers:

1. **Wilcoxon Signed-Rank Test** — non-parametric pairwise significance
   test on per-sequence IoU scores.  Preferred over a t-test because
   tracking IoU distributions are rarely Gaussian.

2. **Cohen's d** — standardised effect-size measure.  Complements the
   p-value by quantifying *how much* better one tracker is.

3. **Bootstrap Confidence Intervals** — distribution-free CI for any
   scalar metric derived from a sample of sequence results.

4. **Ranking Table** — multi-tracker comparison ranked by mean IoU,
   annotated with pairwise significance symbols (*p<0.05, **p<0.01).

References
----------
Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
Data Sets. JMLR 7, 1–30.

Kristan, M. et al. (2016). The Visual Object Tracking VOT2016 Challenge
Results. ECCVW. — use of statistical significance in VOT evaluations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class TrackerComparison:
    """Result of a pairwise statistical test between two trackers.

    Attributes:
        tracker_a:       Name of the first tracker.
        tracker_b:       Name of the second tracker.
        metric:          Metric that was compared (e.g. ``"iou"``).
        n_sequences:     Number of paired sequences used in the test.
        mean_a:          Mean metric value for tracker A.
        mean_b:          Mean metric value for tracker B.
        delta:           ``mean_a - mean_b`` (positive = A is better).
        p_value:         Two-tailed p-value from the Wilcoxon signed-rank test.
        cohens_d:        Effect size (Cohen's d).
        significant:     ``True`` when ``p_value < alpha``.
        alpha:           Significance threshold used (default: 0.05).
        ci_a:            Bootstrap 95% CI for mean metric of tracker A,
                         as ``(lower, upper)``.
        ci_b:            Bootstrap 95% CI for mean metric of tracker B.
    """

    tracker_a: str
    tracker_b: str
    metric: str
    n_sequences: int
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    cohens_d: float
    significant: bool
    alpha: float = 0.05
    ci_a: Tuple[float, float] = (0.0, 0.0)
    ci_b: Tuple[float, float] = (0.0, 0.0)

    def summary_line(self) -> str:
        sig_str = "*" if self.significant else " "
        return (
            f"{self.tracker_a} vs {self.tracker_b} [{self.metric}]{sig_str}  "
            f"Δ={self.delta:+.4f}  p={self.p_value:.4f}  d={self.cohens_d:.3f}  "
            f"n={self.n_sequences}"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_a": self.tracker_a,
            "tracker_b": self.tracker_b,
            "metric": self.metric,
            "n_sequences": self.n_sequences,
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "delta": round(self.delta, 6),
            "p_value": round(self.p_value, 6),
            "cohens_d": round(self.cohens_d, 4),
            "significant": self.significant,
            "alpha": self.alpha,
            "ci_a": [round(v, 6) for v in self.ci_a],
            "ci_b": [round(v, 6) for v in self.ci_b],
        }


@dataclass
class RankingTable:
    """Multi-tracker ranking annotated with pairwise significance.

    Attributes:
        metric:       Metric used for ranking (e.g. ``"iou"``).
        rows:         List of ``(rank, tracker_name, mean, ci_lower, ci_upper)``
                      sorted best-first.
        comparisons:  All pairwise :class:`TrackerComparison` objects.
    """

    metric: str
    rows: List[Tuple[int, str, float, float, float]] = field(default_factory=list)
    comparisons: List[TrackerComparison] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Return the ranking as a GitHub-flavoured Markdown table."""
        lines = [
            f"## Tracker Ranking — {self.metric.upper()}",
            "",
            "| Rank | Tracker | Mean | 95% CI |",
            "| ---: | :------ | ---: | :----- |",
        ]
        for rank, name, mean, lo, hi in self.rows:
            lines.append(
                f"| {rank} | {name} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |"
            )

        if self.comparisons:
            lines += ["", "### Pairwise Significance (Wilcoxon, α=0.05)", ""]
            lines += ["| Comparison | Δ mean | p-value | d | sig |"]
            lines += ["| :--------- | -----: | ------: | -: | :-: |"]
            for cmp in self.comparisons:
                sig = "✓" if cmp.significant else "–"
                lines.append(
                    f"| {cmp.tracker_a} vs {cmp.tracker_b} "
                    f"| {cmp.delta:+.4f} | {cmp.p_value:.4f} "
                    f"| {cmp.cohens_d:.3f} | {sig} |"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "ranking": [
                {
                    "rank": r,
                    "tracker": name,
                    "mean": round(mean, 6),
                    "ci_lower": round(lo, 6),
                    "ci_upper": round(hi, 6),
                }
                for r, name, mean, lo, hi in self.rows
            ],
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------


def _wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> float:
    """Two-tailed Wilcoxon signed-rank p-value for paired samples.

    Uses the normal approximation with continuity correction, valid for
    n ≥ 10.  Returns 1.0 for degenerate inputs (all differences zero or
    n < 2).

    Args:
        x: Per-sequence scores for tracker A, shape ``(n,)``.
        y: Per-sequence scores for tracker B, shape ``(n,)``.

    Returns:
        Two-tailed p-value in ``[0, 1]``.
    """
    diffs = x - y
    diffs = diffs[diffs != 0.0]  # remove zero differences
    n = len(diffs)
    if n < 2:
        return 1.0

    ranks = _rank_data(np.abs(diffs))
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    w_stat = min(w_plus, w_minus)

    # Normal approximation with continuity correction
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return 1.0
    z = (w_stat - mean_w + 0.5) / std_w  # continuity correction
    p_value = 2.0 * _normal_sf(abs(z))
    return float(p_value)


def _rank_data(arr: np.ndarray) -> np.ndarray:
    """Assign average ranks to elements of *arr* (handles ties)."""
    n = len(arr)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average rank
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _normal_sf(z: float) -> float:
    """Survival function of the standard normal (upper tail probability)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two paired samples.

    Uses the pooled standard deviation of the *differences* array,
    which is the paired-samples variant recommended for tracking
    benchmarks where each sequence is matched across trackers.

    Args:
        x: Per-sequence scores for tracker A.
        y: Per-sequence scores for tracker B.

    Returns:
        Cohen's d (signed; positive means A > B).
    """
    diffs = x - y
    n = len(diffs)
    if n < 2:
        return 0.0
    std = float(np.std(diffs, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(diffs) / std)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval for the mean.

    Args:
        values:      1-D array of observed values (e.g. per-sequence IoU).
        n_bootstrap: Number of bootstrap resamples (default: 2000).
        confidence:  Coverage probability (default: 0.95).
        rng_seed:    Random seed for reproducibility (default: 42).

    Returns:
        ``(lower, upper)`` confidence interval bounds.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def compare_trackers(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str = "tracker_a",
    name_b: str = "tracker_b",
    metric: str = "iou",
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
) -> TrackerComparison:
    """Perform a full pairwise statistical comparison between two trackers.

    Args:
        scores_a:    Per-sequence metric values for tracker A, shape ``(n,)``.
        scores_b:    Per-sequence metric values for tracker B, shape ``(n,)``.
        name_a:      Human-readable name for tracker A.
        name_b:      Human-readable name for tracker B.
        metric:      Name of the metric (used in reports).
        alpha:       Significance level (default: 0.05).
        n_bootstrap: Resamples for CI estimation (default: 2000).

    Returns:
        A :class:`TrackerComparison` with p-value, effect size, and CIs.

    Example::

        iou_mosse = np.array([0.72, 0.68, 0.81, ...])  # one value per sequence
        iou_kcf   = np.array([0.75, 0.71, 0.79, ...])
        result = compare_trackers(iou_mosse, iou_kcf, "MOSSE", "KCF")
        print(result.summary_line())
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    p = _wilcoxon_signed_rank(a, b)
    d = cohens_d(a, b)
    ci_a = bootstrap_ci(a, n_bootstrap=n_bootstrap)
    ci_b = bootstrap_ci(b, n_bootstrap=n_bootstrap)

    return TrackerComparison(
        tracker_a=name_a,
        tracker_b=name_b,
        metric=metric,
        n_sequences=n,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        delta=float(a.mean() - b.mean()),
        p_value=p,
        cohens_d=d,
        significant=p < alpha,
        alpha=alpha,
        ci_a=ci_a,
        ci_b=ci_b,
    )


def rank_trackers(
    tracker_scores: Dict[str, np.ndarray],
    metric: str = "iou",
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
) -> RankingTable:
    """Rank trackers by mean metric and compute all pairwise comparisons.

    Args:
        tracker_scores: Mapping of tracker name → per-sequence metric array.
        metric:         Metric name for display (default: ``"iou"``).
        alpha:          Significance level for pairwise tests (default: 0.05).
        n_bootstrap:    Bootstrap resamples for CI estimation.

    Returns:
        A :class:`RankingTable` with ranked rows and pairwise comparisons.

    Example::

        scores = {
            "MOSSE":  np.array([0.72, 0.68, 0.81]),
            "KCF":    np.array([0.75, 0.71, 0.79]),
            "CSRT":   np.array([0.80, 0.78, 0.85]),
        }
        table = rank_trackers(scores, metric="iou")
        print(table.to_markdown())
    """
    names = list(tracker_scores.keys())
    means: Dict[str, float] = {}
    cis: Dict[str, Tuple[float, float]] = {}
    for name, arr in tracker_scores.items():
        arr_f = np.asarray(arr, dtype=np.float64)
        means[name] = float(arr_f.mean()) if len(arr_f) else 0.0
        cis[name] = bootstrap_ci(arr_f, n_bootstrap=n_bootstrap)

    sorted_names = sorted(names, key=lambda n: means[n], reverse=True)
    rows = [
        (rank + 1, name, means[name], cis[name][0], cis[name][1])
        for rank, name in enumerate(sorted_names)
    ]

    comparisons: List[TrackerComparison] = []
    for i in range(len(sorted_names)):
        for j in range(i + 1, len(sorted_names)):
            na, nb = sorted_names[i], sorted_names[j]
            cmp = compare_trackers(
                tracker_scores[na],
                tracker_scores[nb],
                name_a=na,
                name_b=nb,
                metric=metric,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
            )
            comparisons.append(cmp)

    return RankingTable(metric=metric, rows=rows, comparisons=comparisons)


# ---------------------------------------------------------------------------
# Helpers for loading EOVOT JSON result files
# ---------------------------------------------------------------------------


def load_sequence_ious(result_path: str) -> Tuple[str, np.ndarray]:
    """Load per-sequence mean IoU values from an EOVOT JSON result file.

    Args:
        result_path: Path to a ``*.json`` file produced by
            :class:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.

    Returns:
        ``(tracker_name, ious)`` where *ious* is a ``(n_sequences,)`` array.

    Raises:
        KeyError: If the JSON file is missing expected fields.
        FileNotFoundError: If the path does not exist.
    """
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracker_name: str = data["summary"]["tracker"]
    ious = np.array(
        [seq["mean_iou"] for seq in data["sequences"]],
        dtype=np.float64,
    )
    return tracker_name, ious


def load_sequence_metric(
    result_path: str,
    metric_key: str = "mean_iou",
) -> Tuple[str, np.ndarray]:
    """Load any per-sequence scalar metric from an EOVOT JSON result file.

    Args:
        result_path: Path to the JSON file.
        metric_key:  Key present in each sequence dict (e.g. ``"fps"``,
                     ``"peak_memory_mb"``, ``"energy_per_frame_mj"``).

    Returns:
        ``(tracker_name, values)`` where *values* is a ``(n_sequences,)`` array.
    """
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracker_name: str = data["summary"]["tracker"]
    values = np.array(
        [seq[metric_key] for seq in data["sequences"] if metric_key in seq],
        dtype=np.float64,
    )
    return tracker_name, values
