"""Benchmark result visualisation for EOVOT.

Generates publication-ready plots from :class:`~eovot.benchmark.engine.BenchmarkResult`
objects using matplotlib.  All plots are saved to disk as PNG files and
optionally displayed interactively.

Supported plots:

- **Success curve** — fraction of frames with IoU > threshold (0→1), one line
  per tracker.  AUC is shown in the legend.
- **Precision curve** — fraction of frames with centre-distance < threshold
  (0→50 px), one line per tracker.  Score at 20 px threshold shown in legend.
- **FPS comparison** — horizontal bar chart comparing mean FPS across trackers.
- **Per-sequence IoU** — bar chart of per-sequence mean IoU for a single result.

Usage::

    from eovot.reporting.visualizer import BenchmarkVisualizer
    from eovot.benchmark.engine import BenchmarkEngine

    engine = BenchmarkEngine()
    results = [engine.run(tracker, dataset) for tracker in trackers]

    viz = BenchmarkVisualizer(output_dir="results/plots")
    viz.plot_success_curves(results, filename="success_curves.png")
    viz.plot_precision_curves(results, filename="precision_curves.png")
    viz.plot_fps_comparison(results, filename="fps_comparison.png")
    viz.plot_sequence_ious(results[0], filename="sequence_ious.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

if TYPE_CHECKING:
    from ..benchmark.engine import BenchmarkResult

# ---------------------------------------------------------------------------
# Colour palette (colour-blind-friendly, matches common VOT paper style)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
]


def _require_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        )


class BenchmarkVisualizer:
    """Generate and save benchmark plots from :class:`BenchmarkResult` objects.

    Args:
        output_dir: Directory where plots are saved.  Created automatically if
            it does not exist.  Default: ``"results/plots"``.
        dpi: Resolution of saved PNG files.  Default: ``150``.
        figsize: ``(width, height)`` in inches.  Default: ``(8, 5)``.
    """

    def __init__(
        self,
        output_dir: str = "results/plots",
        dpi: int = 150,
        figsize: tuple = (8, 5),
    ) -> None:
        _require_matplotlib()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def plot_success_curves(
        self,
        results: List["BenchmarkResult"],
        filename: str = "success_curves.png",
        title: Optional[str] = None,
    ) -> Path:
        """Plot success curves for one or more trackers.

        Each curve shows the fraction of frames whose IoU exceeds a threshold
        swept from 0 to 1.  The Area Under the Curve (AUC) is shown in the
        legend as the canonical VOT success score.

        Args:
            results: List of :class:`BenchmarkResult` objects (one per tracker).
            filename: Output filename relative to ``output_dir``.
            title: Optional plot title; defaults to ``"Success Curves — <dataset>"``.

        Returns:
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        thresholds = np.linspace(0.0, 1.0, 101)

        for i, result in enumerate(results):
            all_ious = np.concatenate([sr.ious for sr in result.sequence_results])
            rates = np.array([(all_ious > t).mean() for t in thresholds])
            auc = float(np.trapz(rates, thresholds))
            color = _PALETTE[i % len(_PALETTE)]
            ax.plot(
                thresholds, rates,
                color=color,
                linewidth=2,
                label=f"{result.tracker_name} [AUC={auc:.3f}]",
            )

        ax.set_xlabel("IoU Threshold", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title or f"Success Curves — {results[0].dataset_name}", fontsize=13)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def plot_precision_curves(
        self,
        results: List["BenchmarkResult"],
        filename: str = "precision_curves.png",
        title: Optional[str] = None,
    ) -> Path:
        """Plot precision curves for one or more trackers.

        Each curve shows the fraction of frames whose predicted-centre to
        ground-truth-centre distance is below a pixel threshold swept from
        0 to 50 px.  The score at 20 px is the standard OTB precision metric.

        Args:
            results: List of :class:`BenchmarkResult` objects.
            filename: Output filename relative to ``output_dir``.
            title: Optional plot title.

        Returns:
            Path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        thresholds = np.linspace(0.0, 50.0, 51)
        ref_thresh = 20.0  # canonical OTB precision score threshold

        for i, result in enumerate(results):
            # Use stored per-frame centre-distances when available (engine >= v0.2).
            # Fall back to an IoU-derived approximation for older result objects.
            has_real_dists = all(
                sr.center_distances is not None for sr in result.sequence_results
            )
            if has_real_dists:
                all_dists = np.concatenate(
                    [sr.center_distances for sr in result.sequence_results]
                )
            else:
                # Heuristic: dist ≈ (1 - IoU) × 50 px — retained for backward
                # compatibility with results produced by older engine versions.
                all_ious = np.concatenate([sr.ious for sr in result.sequence_results])
                all_dists = (1.0 - all_ious) * 50.0

            rates = np.array([(all_dists < t).mean() for t in thresholds])
            score_at_20 = float((all_dists < ref_thresh).mean())
            color = _PALETTE[i % len(_PALETTE)]
            label_suffix = "" if has_real_dists else " (approx)"
            ax.plot(
                thresholds, rates,
                color=color,
                linewidth=2,
                label=f"{result.tracker_name} [@20px={score_at_20:.3f}]{label_suffix}",
            )

        ax.axvline(x=ref_thresh, color="gray", linestyle=":", linewidth=1.2,
                   label="20 px threshold")
        ax.set_xlabel("Centre-Distance Threshold (px)", fontsize=12)
        ax.set_ylabel("Precision Rate", fontsize=12)
        ax.set_xlim(0.0, 50.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title or f"Precision Curves — {results[0].dataset_name}", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def plot_fps_comparison(
        self,
        results: List["BenchmarkResult"],
        filename: str = "fps_comparison.png",
        title: Optional[str] = None,
    ) -> Path:
        """Horizontal bar chart comparing mean FPS across trackers.

        Args:
            results: List of :class:`BenchmarkResult` objects.
            filename: Output filename relative to ``output_dir``.
            title: Optional plot title.

        Returns:
            Path to the saved PNG file.
        """
        names = [r.tracker_name for r in results]
        fps_vals = [r.mean_fps for r in results]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(results))]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(names, fps_vals, color=colors, edgecolor="white", height=0.5)

        # Annotate bars with numeric values
        for bar, val in zip(bars, fps_vals):
            ax.text(
                bar.get_width() + max(fps_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.1f}",
                va="center", ha="left", fontsize=10,
            )

        ax.set_xlabel("Mean FPS", fontsize=12)
        ax.set_title(title or "Tracker Throughput Comparison", fontsize=13)
        ax.set_xlim(0, max(fps_vals) * 1.18)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax.invert_yaxis()

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def plot_sequence_ious(
        self,
        result: "BenchmarkResult",
        filename: str = "sequence_ious.png",
        title: Optional[str] = None,
        max_sequences: int = 30,
    ) -> Path:
        """Bar chart of per-sequence mean IoU for a single tracker result.

        Args:
            result: A single :class:`BenchmarkResult`.
            filename: Output filename relative to ``output_dir``.
            title: Optional plot title.
            max_sequences: Cap on sequences shown to keep the chart readable.
                Sequences are sorted by mean IoU descending.

        Returns:
            Path to the saved PNG file.
        """
        seq_results = sorted(result.sequence_results, key=lambda s: s.mean_iou, reverse=True)
        if len(seq_results) > max_sequences:
            seq_results = seq_results[:max_sequences]

        names = [sr.sequence_name for sr in seq_results]
        ious = [sr.mean_iou for sr in seq_results]
        colors = [
            "#2ca02c" if v >= 0.5 else "#ff7f0e" if v >= 0.3 else "#d62728"
            for v in ious
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))
        ax.bar(range(len(names)), ious, color=colors, edgecolor="white")
        ax.axhline(y=0.5, color="green", linestyle="--", linewidth=1.0,
                   label="IoU=0.5 threshold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean IoU", fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            title or f"{result.tracker_name} per-sequence IoU — {result.dataset_name}",
            fontsize=13,
        )
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def plot_efficiency_accuracy(
        self,
        results: List["BenchmarkResult"],
        filename: str = "efficiency_accuracy.png",
        title: Optional[str] = None,
        memory_budget_mb: float = 512.0,
    ) -> Path:
        """Scatter plot of accuracy (mIoU) vs throughput (FPS), bubble area ∝ memory.

        EOVOT's signature visualisation — the efficiency-accuracy trade-off
        plot that differentiates this benchmark from accuracy-only evaluations.
        Each bubble represents one tracker:

        * X-axis: mean FPS (efficiency proxy)
        * Y-axis: mean IoU (accuracy)
        * Bubble area: proportional to peak memory footprint
        * Bold black outline: Pareto-optimal trackers

        Args:
            results: List of :class:`BenchmarkResult` objects (one per tracker).
            filename: Output filename relative to ``output_dir``.
            title: Optional plot title.
            memory_budget_mb: Reference memory ceiling for EES computation
                and bubble scaling.  Default: ``512.0`` MB.

        Returns:
            Path to the saved PNG file.
        """
        from ..metrics.efficiency import EfficiencyMetricsEngine

        eff_engine = EfficiencyMetricsEngine(memory_budget_mb=memory_budget_mb)
        ranking = eff_engine.rank_trackers(results)

        fig, ax = plt.subplots(figsize=self.figsize)

        mem_vals = [e.peak_memory_mb for e in ranking]
        max_mem = max(mem_vals) if mem_vals else 1.0

        for i, entry in enumerate(ranking):
            color = _PALETTE[i % len(_PALETTE)]
            # Bubble area in points² — minimum 80 so tiny-memory trackers are visible.
            size = max(80.0, (entry.peak_memory_mb / max_mem) * 1200.0)
            edgecolor = "black" if entry.on_pareto_front else "white"
            linewidth = 2.5 if entry.on_pareto_front else 0.8
            label = (
                f"{entry.tracker_name} "
                f"(EES={entry.ees:.3f}{'  ✓' if entry.on_pareto_front else ''})"
            )
            ax.scatter(
                entry.fps,
                entry.mean_iou,
                s=size,
                c=color,
                alpha=0.82,
                edgecolors=edgecolor,
                linewidths=linewidth,
                zorder=3,
                label=label,
            )
            ax.annotate(
                entry.tracker_name,
                (entry.fps, entry.mean_iou),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
            )

        ax.set_xlabel("Mean FPS  (higher → more efficient)", fontsize=12)
        ax.set_ylabel("Mean IoU  (higher → more accurate)", fontsize=12)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            title or "Efficiency–Accuracy Trade-off  (bubble area ∝ memory)",
            fontsize=12,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

        if any(e.on_pareto_front for e in ranking):
            ax.text(
                0.02, 0.97,
                "✓ = Pareto-optimal",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                color="gray",
            )

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def save_all(
        self,
        results: List["BenchmarkResult"],
        prefix: str = "benchmark",
    ) -> dict:
        """Save standard plots and return a mapping of name → path.

        Saves success, precision, FPS, per-sequence IoU, and (when multiple
        trackers are provided) the efficiency-accuracy trade-off scatter plot.

        Args:
            results: List of benchmark results (one or more trackers).
            prefix: Filename prefix for all saved plots.

        Returns:
            Dict mapping plot names to their :class:`pathlib.Path` objects.
        """
        paths = {}
        paths["success"] = self.plot_success_curves(
            results, filename=f"{prefix}_success.png"
        )
        paths["precision"] = self.plot_precision_curves(
            results, filename=f"{prefix}_precision.png"
        )
        paths["fps"] = self.plot_fps_comparison(
            results, filename=f"{prefix}_fps.png"
        )
        # Per-sequence IoU only makes sense for a single result
        paths["sequence_ious"] = self.plot_sequence_ious(
            results[0], filename=f"{prefix}_seq_ious.png"
        )
        if len(results) > 1:
            paths["efficiency_accuracy"] = self.plot_efficiency_accuracy(
                results, filename=f"{prefix}_efficiency_accuracy.png"
            )
        return paths
