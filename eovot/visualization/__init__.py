"""Visualization utilities for EOVOT benchmark results.

Provides publication-quality plots for tracker evaluation:

- :func:`~eovot.visualization.plots.plot_success_curves` — IoU success curves
- :func:`~eovot.visualization.plots.plot_precision_curves` — centre-distance
  precision curves
- :func:`~eovot.visualization.plots.plot_tracker_comparison` — bar-chart
  comparison of multiple trackers across key metrics
- :class:`~eovot.visualization.radar.RadarChartBuilder` — multi-metric radar
  chart comparing trackers across accuracy, speed, robustness, smoothness, and
  efficiency simultaneously

All plot functions require ``matplotlib`` (not listed as a core dependency to
keep the base install lightweight; install it with ``pip install matplotlib``).
"""

from .plots import plot_success_curves, plot_precision_curves, plot_tracker_comparison
from .radar import RadarChartBuilder, RadarEntry

__all__ = [
    "plot_success_curves",
    "plot_precision_curves",
    "plot_tracker_comparison",
    "RadarChartBuilder",
    "RadarEntry",
]
