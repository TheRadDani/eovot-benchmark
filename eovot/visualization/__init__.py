"""Visualization utilities for EOVOT benchmark results.

Provides publication-quality plots for tracker evaluation:

- :func:`~eovot.visualization.plots.plot_success_curves` — IoU success curves
- :func:`~eovot.visualization.plots.plot_precision_curves` — centre-distance
  precision curves
- :func:`~eovot.visualization.plots.plot_tracker_comparison` — bar-chart
  comparison of multiple trackers across key metrics

Efficiency frontier and deployment plots:

- :func:`~eovot.visualization.frontier.plot_efficiency_frontier` — Pareto
  frontier scatter in (FPS, mIoU) space with memory-encoded bubble size
- :func:`~eovot.visualization.frontier.plot_device_projection` — heatmap of
  estimated performance across trackers and edge device targets
- :func:`~eovot.visualization.frontier.plot_frame_skip_degradation` — line
  chart of accuracy degradation vs. temporal skip rate

All functions require ``matplotlib`` (``pip install matplotlib``).
"""

from .frontier import (
    plot_device_projection,
    plot_efficiency_frontier,
    plot_frame_skip_degradation,
)
from .plots import plot_precision_curves, plot_success_curves, plot_tracker_comparison

__all__ = [
    "plot_success_curves",
    "plot_precision_curves",
    "plot_tracker_comparison",
    "plot_efficiency_frontier",
    "plot_device_projection",
    "plot_frame_skip_degradation",
]
