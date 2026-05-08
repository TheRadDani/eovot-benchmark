"""Statistical analysis sub-package for EOVOT.

Provides publication-quality significance testing and Pareto frontier
analysis for comparing tracker results.

Typical usage::

    from eovot.analysis.statistics import TrackerStatistics
    from eovot.analysis.pareto import ParetoAnalyzer

    stats = TrackerStatistics()
    comparisons = stats.pairwise_comparison([mosse_result, kcf_result])
    print(stats.significance_table(comparisons))

    analyzer = ParetoAnalyzer()
    points = analyzer.analyze({"MOSSE": mosse_result, "KCF": kcf_result})
    print(analyzer.to_markdown(points))
"""

from .statistics import ComparisonResult, TrackerStatistics
from .pareto import ParetoAnalyzer, ParetoPoint

__all__ = [
    "TrackerStatistics",
    "ComparisonResult",
    "ParetoAnalyzer",
    "ParetoPoint",
]
