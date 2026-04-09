"""Analysis sub-package — Pareto frontier and edge deployment decision support."""

from .pareto import (
    TrackerProfile,
    ParetoResult,
    ParetoAnalyzer,
    compute_pareto_front,
    compute_edge_score,
    rank_trackers,
)

__all__ = [
    "TrackerProfile",
    "ParetoResult",
    "ParetoAnalyzer",
    "compute_pareto_front",
    "compute_edge_score",
    "rank_trackers",
]
