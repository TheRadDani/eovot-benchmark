"""Visualization utilities for EOVOT benchmark results.

Provides publication-quality plots of standard VOT evaluation curves:

* **Success curve** — fraction of frames with IoU above a threshold (0 → 1).
* **Precision curve** — fraction of frames with centre-distance below a
  threshold (0 → 50 px).
* **Performance bar chart** — side-by-side comparison of scalar metrics
  (mIoU, FPS, latency) across multiple trackers.

All plotting functions accept raw numpy arrays so they integrate naturally
with :class:`~eovot.metrics.accuracy.MetricsEngine` and
:class:`~eovot.benchmark.engine.BenchmarkResult`.

Requires ``matplotlib`` (optional dependency)::

    pip install matplotlib

Example::

    from eovot.visualization import TrackingPlotter
    plotter = TrackingPlotter()
    plotter.plot_success_curve(
        {"MOSSE": mosse_ious, "KCF": kcf_ious},
        save_path="results/success_curve.png",
    )
"""

from .plots import TrackingPlotter

__all__ = ["TrackingPlotter"]
