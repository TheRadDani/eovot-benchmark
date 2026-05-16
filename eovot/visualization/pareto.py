"""Pareto frontier and radar chart visualisation for edge efficiency analysis.

Generates two complementary plots for the accuracy–efficiency trade-off that
is central to EOVOT's edge-deployment evaluation:

* **Pareto scatter plot** — accuracy (mIoU) vs composite efficiency score,
  with Pareto-optimal trackers highlighted and connected by the frontier line.
  Dominated trackers appear in grey; optimal trackers in colour.
* **Radar chart (spider plot)** — multi-axis comparison of mIoU, FPS score,
  memory score, and energy score per tracker.  Useful for understanding which
  dimension each tracker excels or fails at.

Both functions accept :class:`~eovot.metrics.efficiency.EdgeEfficiencyScore`
objects produced by :class:`~eovot.metrics.efficiency.EdgeEfficiencyAnalyzer`.

Requires ``matplotlib``.  Install with ``pip install matplotlib``.

Example::

    from eovot.metrics.efficiency import EdgeEfficiencyAnalyzer
    from eovot.visualization.pareto import plot_pareto_frontier, plot_efficiency_radar

    analyzer = EdgeEfficiencyAnalyzer(target_fps=30.0, max_memory_mb=512.0)
    scores   = analyzer.analyze([mosse_result, kcf_result, csrt_result])
    frontier = analyzer.pareto_frontier(scores)

    plot_pareto_frontier(scores, frontier, output_path="results/pareto.png")
    plot_efficiency_radar(scores, output_path="results/radar.png")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..metrics.efficiency import EdgeEfficiencyScore

# ---------------------------------------------------------------------------
# Colour palette (colour-blind-friendly)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]
_DOMINATED_COLOR = "#aaaaaa"
_FRONTIER_COLOR = "#d62728"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        return plt, matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for EOVOT visualisation. "
            "Install with: pip install matplotlib"
        ) from exc


def plot_pareto_frontier(
    scores: "List[EdgeEfficiencyScore]",
    frontier: "Optional[List[EdgeEfficiencyScore]]" = None,
    output_path: Optional[str] = None,
    title: str = "Accuracy–Efficiency Trade-off (Pareto Frontier)",
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> None:
    """Scatter plot of mIoU vs composite efficiency with the Pareto frontier.

    Pareto-optimal trackers are plotted in colour and connected by the
    frontier line.  Dominated trackers are shown in grey.  Each point is
    annotated with the tracker name.

    Args:
        scores:      All tracker efficiency scores (one per tracker).
        frontier:    Pareto-optimal subset, as returned by
                     :meth:`~eovot.metrics.efficiency.EdgeEfficiencyAnalyzer.pareto_frontier`.
                     If ``None``, all trackers are plotted uniformly.
        output_path: Save path (PNG / PDF / SVG).  Interactive display
                     when ``None``.
        title:       Figure title.
        figsize:     ``(width, height)`` in inches.
        dpi:         Resolution for saved files.
    """
    plt, _ = _require_matplotlib()

    frontier_names = {s.tracker_name for s in frontier} if frontier else set()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot dominated trackers first (background)
    for s in scores:
        if s.tracker_name not in frontier_names:
            ax.scatter(
                s.composite_score, s.mean_iou,
                color=_DOMINATED_COLOR, s=80, zorder=2, alpha=0.7,
            )
            ax.annotate(
                s.tracker_name,
                (s.composite_score, s.mean_iou),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, color=_DOMINATED_COLOR,
            )

    # Plot Pareto-optimal trackers (foreground)
    color_idx = 0
    for s in scores:
        if s.tracker_name in frontier_names:
            color = _PALETTE[color_idx % len(_PALETTE)]
            color_idx += 1
            ax.scatter(
                s.composite_score, s.mean_iou,
                color=color, s=120, zorder=3, edgecolors="white", linewidths=0.8,
            )
            ax.annotate(
                s.tracker_name,
                (s.composite_score, s.mean_iou),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, fontweight="bold", color=color,
            )

    # Draw the Pareto frontier line (sorted by efficiency score)
    if frontier and len(frontier) >= 2:
        sorted_front = sorted(frontier, key=lambda s: s.composite_score)
        fx = [s.composite_score for s in sorted_front]
        fy = [s.mean_iou for s in sorted_front]
        ax.plot(fx, fy, color=_FRONTIER_COLOR, linewidth=1.5,
                linestyle="--", zorder=1, alpha=0.6, label="Pareto frontier")
        ax.legend(fontsize=9, loc="lower right")

    ax.set_xlabel("Composite Efficiency Score", fontsize=12)
    ax.set_ylabel("Mean IoU (Accuracy)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[pareto] Scatter plot saved → {output_path}")
    else:
        plt.show()


def plot_efficiency_radar(
    scores: "List[EdgeEfficiencyScore]",
    output_path: Optional[str] = None,
    title: str = "Edge Efficiency Radar",
    figsize: tuple = (7, 7),
    dpi: int = 150,
) -> None:
    """Radar (spider) chart comparing trackers across multiple efficiency axes.

    Axes shown: mIoU, FPS Score, Memory Score, and (if available) Energy Score.
    Each tracker is drawn as a filled polygon; the further from the centre on
    each axis, the better.

    Args:
        scores:      List of :class:`~eovot.metrics.efficiency.EdgeEfficiencyScore`.
        output_path: Save path.  Interactive display when ``None``.
        title:       Figure title.
        figsize:     ``(width, height)`` in inches.
        dpi:         Resolution for saved files.
    """
    plt, _ = _require_matplotlib()

    has_energy = any(s.has_energy for s in scores)
    axes_labels = ["mIoU", "FPS\nScore", "Memory\nScore"]
    if has_energy:
        axes_labels.append("Energy\nScore")

    n_axes = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    for i, s in enumerate(scores):
        values = [s.mean_iou, s.fps_score, s.memory_score]
        if has_energy:
            values.append(s.energy_score)
        values += values[:1]  # close the polygon

        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2, label=s.tracker_name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[pareto] Radar chart saved → {output_path}")
    else:
        plt.show()
