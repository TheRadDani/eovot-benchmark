"""Publication-quality tracking performance plots for EOVOT.

Standard VOT benchmarks (OTB, GOT-10k, LaSOT) report results as:
- **Success curves**: IoU threshold vs. fraction of frames above it.
- **Precision curves**: centre-distance threshold vs. fraction of frames below it.

These two curves are the community standard for comparing trackers and are
expected in any research paper or benchmark report.

A **bar chart** helper is also provided for quick scalar comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib is an optional dependency — import lazily so the rest of EOVOT
# works even without it installed.
_MPL_ERR = (
    "matplotlib is required for visualization.\n"
    "Install it with:  pip install matplotlib"
)


def _require_mpl():
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(_MPL_ERR) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _success_curve(
    ious: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of frames with IoU strictly above each threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rates = np.array([(ious > t).mean() for t in thresholds])
    return thresholds, rates


def _precision_curve(
    preds: np.ndarray,
    gts: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of frames with centre distance below each threshold."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 50.0, 51)
    px = preds[:, 0] + preds[:, 2] / 2.0
    py = preds[:, 1] + preds[:, 3] / 2.0
    gx = gts[:, 0] + gts[:, 2] / 2.0
    gy = gts[:, 1] + gts[:, 3] / 2.0
    dists = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    rates = np.array([(dists < t).mean() for t in thresholds])
    return thresholds, rates


def _auc(thresholds: np.ndarray, rates: np.ndarray) -> float:
    """Normalised area under a curve (compatible with NumPy 1.x and 2.x)."""
    span = thresholds[-1] - thresholds[0]
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(rates, thresholds) / span) if span > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TrackingPlotter:
    """Generate publication-quality VOT evaluation plots.

    Args:
        figsize: Default figure size ``(width_inches, height_inches)``.
        dpi: Dots per inch for saved figures.
        style: matplotlib style sheet name.  ``"seaborn-v0_8-paper"`` is
            used when available, otherwise ``"default"``.

    Example::

        from eovot.visualization import TrackingPlotter
        import numpy as np

        plotter = TrackingPlotter()

        # Per-tracker IoU arrays, shape (N,)
        ious = {
            "MOSSE": np.random.rand(500),
            "KCF":   np.random.rand(500) + 0.05,
        }
        plotter.plot_success_curve(ious, save_path="results/success.png")
    """

    # Colour cycle consistent with common VOT benchmark papers.
    _COLOURS = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # grey
    ]

    def __init__(
        self,
        figsize: Tuple[float, float] = (7.0, 5.0),
        dpi: int = 150,
        style: str = "default",
    ) -> None:
        _require_mpl()
        import matplotlib.pyplot as plt

        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")

    # ------------------------------------------------------------------
    # Success curve
    # ------------------------------------------------------------------

    def plot_success_curve(
        self,
        tracker_ious: Dict[str, np.ndarray],
        thresholds: Optional[np.ndarray] = None,
        title: str = "Success Plot",
        save_path: Optional[str] = None,
    ):
        """Plot one success curve per tracker, labelled with its AUC.

        Args:
            tracker_ious: Mapping from tracker name to a 1-D array of
                per-frame IoU values (shape ``(N,)``).
            thresholds: IoU thresholds to evaluate (default: 0.0 → 1.0 in
                101 steps).
            title: Plot title.
            save_path: If given, save the figure to this path (PNG/PDF/…).
                The parent directory is created automatically.

        Returns:
            The ``matplotlib.figure.Figure`` object so callers can embed it
            in notebooks or further customise it.

        Example::

            plotter.plot_success_curve(
                {"MOSSE": mosse_ious, "KCF": kcf_ious},
                save_path="results/success_curve.png",
            )
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (name, ious) in enumerate(tracker_ious.items()):
            ious = np.asarray(ious, dtype=np.float64)
            thr, rates = _success_curve(ious, thresholds)
            auc = _auc(thr, rates)
            colour = self._COLOURS[i % len(self._COLOURS)]
            ax.plot(thr, rates, label=f"{name} [{auc:.3f}]", color=colour, linewidth=2)

        ax.set_xlabel("Overlap threshold", fontsize=12)
        ax.set_ylabel("Fraction of frames", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Precision curve
    # ------------------------------------------------------------------

    def plot_precision_curve(
        self,
        tracker_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        thresholds: Optional[np.ndarray] = None,
        title: str = "Precision Plot",
        save_path: Optional[str] = None,
    ):
        """Plot one precision curve per tracker, labelled with its AUC@20px.

        Args:
            tracker_data: Mapping from tracker name to ``(preds, gts)`` where
                both arrays have shape ``(N, 4)`` in ``(x, y, w, h)`` format.
            thresholds: Distance thresholds in pixels (default: 0 → 50 px).
            title: Plot title.
            save_path: If given, save the figure to this path.

        Returns:
            The ``matplotlib.figure.Figure`` object.

        Example::

            plotter.plot_precision_curve(
                {"MOSSE": (mosse_preds, gt_boxes), "KCF": (kcf_preds, gt_boxes)},
                save_path="results/precision_curve.png",
            )
        """
        import matplotlib.pyplot as plt

        if thresholds is None:
            thresholds = np.linspace(0.0, 50.0, 51)

        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (name, (preds, gts)) in enumerate(tracker_data.items()):
            preds = np.asarray(preds, dtype=np.float64)
            gts = np.asarray(gts, dtype=np.float64)
            thr, rates = _precision_curve(preds, gts, thresholds)
            # Canonical scalar: precision at 20-px threshold
            idx20 = int(np.argmin(np.abs(thr - 20.0)))
            prec20 = float(rates[idx20])
            colour = self._COLOURS[i % len(self._COLOURS)]
            ax.plot(thr, rates, label=f"{name} [{prec20:.3f}]", color=colour, linewidth=2)

        ax.axvline(20.0, color="grey", linestyle=":", linewidth=1.0, label="20 px")
        ax.set_xlabel("Centre distance threshold (px)", fontsize=12)
        ax.set_ylabel("Fraction of frames", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0.0, float(thresholds[-1]))
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Comparison bar chart
    # ------------------------------------------------------------------

    def plot_comparison_bars(
        self,
        summaries: List[Dict],
        metrics: Optional[List[str]] = None,
        title: str = "Tracker Comparison",
        save_path: Optional[str] = None,
    ):
        """Bar chart comparing scalar metrics across multiple trackers.

        Args:
            summaries: List of summary dicts, one per tracker.  Each dict
                must have a ``"tracker"`` key and whichever metric keys are
                listed in *metrics*.  The format matches the output of
                :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`.
            metrics: Metric keys to plot.  Defaults to
                ``["mean_iou", "mean_fps", "peak_memory_mb"]``.
            title: Overall figure title.
            save_path: If given, save the figure to this path.

        Returns:
            The ``matplotlib.figure.Figure`` object.

        Example::

            summaries = [mosse_result.summary(), kcf_result.summary()]
            plotter.plot_comparison_bars(
                summaries,
                metrics=["mean_iou", "mean_fps"],
                save_path="results/comparison.png",
            )
        """
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = ["mean_iou", "mean_fps", "peak_memory_mb"]

        tracker_names = [s.get("tracker", f"tracker_{i}") for i, s in enumerate(summaries)]
        n_trackers = len(tracker_names)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(
            1, n_metrics, figsize=(max(4 * n_metrics, self.figsize[0]), self.figsize[1])
        )
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = [float(s.get(metric, 0.0)) for s in summaries]
            colours = [self._COLOURS[i % len(self._COLOURS)] for i in range(n_trackers)]
            bars = ax.bar(tracker_names, values, color=colours, edgecolor="white", linewidth=0.8)

            # Annotate bar tops with the numeric value.
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            label_map = {
                "mean_iou": "Mean IoU",
                "mean_fps": "Throughput (FPS)",
                "peak_memory_mb": "Peak Memory (MB)",
                "mean_latency_ms": "Mean Latency (ms)",
            }
            ax.set_title(label_map.get(metric, metric), fontsize=11)
            ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Convenience: plot all standard curves at once
    # ------------------------------------------------------------------

    def plot_all(
        self,
        tracker_ious: Dict[str, np.ndarray],
        summaries: List[Dict],
        output_dir: str = "results/plots",
        prefix: str = "",
    ) -> Dict[str, str]:
        """Generate success curve, and comparison bar chart in one call.

        Args:
            tracker_ious: ``{tracker_name: ious_array}`` for success curve.
            summaries: List of :meth:`~eovot.benchmark.engine.BenchmarkResult.summary`
                dicts for the bar chart.
            output_dir: Directory where all PNG files are written.
            prefix: Optional filename prefix (e.g. ``"otb100-"``).

        Returns:
            Dict mapping plot type to saved file path.
        """
        paths: Dict[str, str] = {}
        pfx = f"{prefix}-" if prefix else ""

        sc_path = str(Path(output_dir) / f"{pfx}success_curve.png")
        self.plot_success_curve(tracker_ious, save_path=sc_path)
        paths["success_curve"] = sc_path

        cmp_path = str(Path(output_dir) / f"{pfx}comparison_bars.png")
        self.plot_comparison_bars(summaries, save_path=cmp_path)
        paths["comparison_bars"] = cmp_path

        return paths
