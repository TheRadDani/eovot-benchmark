"""Statistical analysis sub-package for EOVOT benchmark comparisons.

Provides:
- :func:`~eovot.analysis.stats.compare_trackers` — pairwise Wilcoxon
  signed-rank test between two sets of per-sequence IoU scores.
- :func:`~eovot.analysis.stats.cohens_d` — effect-size calculation.
- :func:`~eovot.analysis.stats.bootstrap_ci` — bootstrap confidence
  intervals for any scalar metric.
- :class:`~eovot.analysis.stats.RankingTable` — produce a ranked summary
  table across N trackers with significance annotations.
"""

from .stats import (
    TrackerComparison,
    RankingTable,
    bootstrap_ci,
    cohens_d,
    compare_trackers,
    rank_trackers,
)

__all__ = [
    "TrackerComparison",
    "RankingTable",
    "bootstrap_ci",
    "cohens_d",
    "compare_trackers",
    "rank_trackers",
]
