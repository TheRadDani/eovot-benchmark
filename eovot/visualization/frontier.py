"""Efficiency frontier and multi-device deployment visualizations for EOVOT.

This module produces the two publication-essential figures that the existing
:mod:`eovot.visualization.plots` module does not cover:

1. **Efficiency Frontier Plot** — a scatter plot in (FPS, mIoU) space where
   Pareto-optimal trackers are highlighted with star markers and connected by
   the frontier line, while dominated trackers appear as dimmed circles.
   Point area encodes peak memory footprint.  This is the canonical figure for
   comparing trackers under accuracy–efficiency tradeoffs.

2. **Multi-Device Projection Heatmap** — a grid plot (trackers × devices)
   showing estimated FPS, latency, or memory across the 6 built-in edge
   targets from :class:`~eovot.profiling.device_sim.DeviceSimulator`.  An
   OOM indicator highlights memory-constrained cells.

3. **Frame-Skip Degradation Plot** — a line chart showing how mIoU (or
   success AUC) decreases as the temporal skip rate increases, with an
   optional "optimal skip rate" marker.

All functions accept plain data structures (lists, dicts) returned by the
existing engines, require only ``matplotlib``, and produce publication-quality
output at 150 dpi.

Requires ``matplotlib``.  Install with::

    pip install matplotlib

Example::

    from eovot.metrics.efficiency import EfficiencyMetricsEngine
    from eovot.profiling.device_sim import DeviceSimulator
    from eovot.visualization.frontier import (
        plot_efficiency_frontier,
        plot_device_projection,
        plot_frame_skip_degradation,
    )

    # --- Efficiency frontier ---
    engine  = EfficiencyMetricsEngine(memory_budget_mb=512.0)
    entries = engine.rank_trackers(benchmark_results)
    plot_efficiency_frontier(entries, output_path="frontier.png")

    # --- Device projection heatmap ---
    sim = DeviceSimulator()
    sim_by_tracker = {
        r.tracker_name: sim.simulate_all(r.profiling_result)
        for r in benchmark_results
    }
    plot_device_projection(sim_by_tracker, output_path="devices.png")

    # --- Frame-skip analysis ---
    from eovot.analysis.frame_skip import FrameSkipEvaluator
    analysis = FrameSkipEvaluator().evaluate(tracker, dataset, "OTB100")
    plot_frame_skip_degradation([analysis], output_path="frame_skip.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Matplotlib lazy-import helper
# ---------------------------------------------------------------------------


def _plt():
    """Import and return matplotlib.pyplot, with a clear error if absent."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for EOVOT frontier visualization.\n"
            "Install it with:  pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Plot 1: Efficiency Frontier
# ---------------------------------------------------------------------------


def plot_efficiency_frontier(
    entries: List[Any],
    output_path: Optional[str] = None,
    title: str = "Accuracy–Efficiency Frontier",
    annotate: bool = True,
    memory_legend_values: Optional[List[float]] = None,
) -> None:
    """Scatter plot of mIoU vs FPS with the Pareto-optimal trackers highlighted.

    Each point represents one tracker evaluated on one dataset.  Pareto-optimal
    trackers (those for which no other tracker dominates *both* accuracy and EES)
    are drawn with red star markers and connected by the frontier step-line.
    Dominated trackers appear as dimmed blue circles.  Point *area* encodes
    peak memory usage so the three-objective tradeoff is visible in one figure.

    This plot is produced from the output of
    :meth:`~eovot.metrics.efficiency.EfficiencyMetricsEngine.rank_trackers`,
    which already computes ``on_pareto_front`` flags and sorts by EES.

    Args:
        entries: List of :class:`~eovot.metrics.efficiency.EfficiencyEntry`
            objects from :meth:`~eovot.metrics.efficiency.EfficiencyMetricsEngine.rank_trackers`.
        output_path: Path to save the figure (PNG / PDF / SVG).  When ``None``
            the figure is shown interactively.
        title: Figure title string.
        annotate: If ``True``, annotate each point with the tracker name.
        memory_legend_values: Memory values (MB) shown in the size legend.
            Defaults to ``[50, 200, 500]``.

    Example::

        engine  = EfficiencyMetricsEngine(memory_budget_mb=512.0)
        entries = engine.rank_trackers(benchmark_results)
        plot_efficiency_frontier(entries, output_path="frontier.png")
    """
    plt = _plt()

    if not entries:
        return

    if memory_legend_values is None:
        memory_legend_values = [50, 200, 500]

    pareto = [e for e in entries if e.on_pareto_front]
    dominated = [e for e in entries if not e.on_pareto_front]

    all_mem = np.array([e.peak_memory_mb for e in entries], dtype=np.float64)
    mem_min = all_mem.min()
    mem_range = max(all_mem.max() - mem_min, 1.0)

    def _bubble_size(mem: float) -> float:
        """Map memory (MB) to matplotlib scatter marker area in [40, 320]."""
        return 40.0 + 280.0 * (mem - mem_min) / mem_range

    fig, ax = plt.subplots(figsize=(9, 6))

    # Dominated trackers — subdued blue circles
    if dominated:
        xs = [e.fps for e in dominated]
        ys = [e.mean_iou for e in dominated]
        sz = [_bubble_size(e.peak_memory_mb) for e in dominated]
        ax.scatter(xs, ys, s=sz, color="#7fb3d3", alpha=0.65,
                   zorder=3, label="sub-optimal")
        if annotate:
            for e, x, y in zip(dominated, xs, ys):
                ax.annotate(
                    e.tracker_name, (x, y),
                    fontsize=8, color="#555555",
                    xytext=(4, 4), textcoords="offset points",
                )

    # Pareto-optimal trackers — prominent red stars + frontier step-line
    if pareto:
        pareto_sorted = sorted(pareto, key=lambda e: e.fps)
        xs_p = [e.fps for e in pareto_sorted]
        ys_p = [e.mean_iou for e in pareto_sorted]
        sz_p = [_bubble_size(e.peak_memory_mb) for e in pareto_sorted]

        ax.plot(
            xs_p, ys_p,
            color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            xs_p, ys_p, s=sz_p,
            color="#e74c3c", marker="*", zorder=4,
            label="Pareto-optimal",
            edgecolors="darkred", linewidths=0.4,
        )
        if annotate:
            for e, x, y in zip(pareto_sorted, xs_p, ys_p):
                ax.annotate(
                    e.tracker_name, (x, y),
                    fontsize=9, fontweight="bold", color="#c0392b",
                    xytext=(5, 5), textcoords="offset points",
                )

    # Memory size legend entries
    for mem_val in memory_legend_values:
        if mem_min <= mem_val <= (mem_min + mem_range):
            ax.scatter([], [], s=_bubble_size(mem_val),
                       color="grey", alpha=0.5,
                       label=f"mem ≈ {mem_val} MB")

    ax.set_xlabel("Mean FPS", fontsize=13)
    ax.set_ylabel("Mean IoU", fontsize=13)
    ax.set_title(title, fontsize=15, pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[frontier] Efficiency frontier saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot 2: Multi-Device Projection Heatmap
# ---------------------------------------------------------------------------


def plot_device_projection(
    sim_results_by_tracker: Dict[str, List[Any]],
    metric: str = "estimated_fps",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    mark_oom: bool = True,
) -> None:
    """Heatmap of projected performance across trackers and edge devices.

    Rows are trackers; columns are edge devices from
    :class:`~eovot.profiling.device_sim.DeviceSimulator`.  Each cell
    shows the value of *metric* for that (tracker, device) pair.
    Out-of-memory cells are marked with hatching when ``mark_oom=True``.

    Args:
        sim_results_by_tracker: ``{tracker_name: [DeviceSimResult, …]}`` —
            typically produced by calling
            :meth:`~eovot.profiling.device_sim.DeviceSimulator.simulate_all`
            for each tracker.  All trackers must have the same device order.
        metric: Attribute name on
            :class:`~eovot.profiling.device_sim.DeviceSimResult` to visualise.
            Common choices: ``"estimated_fps"``, ``"estimated_latency_ms"``,
            ``"estimated_energy_mj_per_frame"``.
        output_path: Save path. Interactive display when ``None``.
        title: Figure title.  Defaults to ``"<metric> across Edge Devices"``.
        mark_oom: If ``True``, cross-hatch cells where ``fits_in_memory=False``.

    Example::

        sim = DeviceSimulator()
        sim_by_tracker = {
            "MOSSE": sim.simulate_all(mosse_profiling_result),
            "KCF":   sim.simulate_all(kcf_profiling_result),
        }
        plot_device_projection(sim_by_tracker, output_path="devices.png")
    """
    plt = _plt()

    if not sim_results_by_tracker:
        return

    tracker_names = list(sim_results_by_tracker.keys())
    first_results = sim_results_by_tracker[tracker_names[0]]
    device_labels = [r.display_name for r in first_results]

    n_trackers = len(tracker_names)
    n_devices = len(device_labels)

    matrix = np.zeros((n_trackers, n_devices), dtype=np.float64)
    oom_mask = np.zeros((n_trackers, n_devices), dtype=bool)

    for i, name in enumerate(tracker_names):
        dev_results = sim_results_by_tracker[name]
        for j, dr in enumerate(dev_results[:n_devices]):
            val = getattr(dr, metric, 0.0)
            matrix[i, j] = float(bool(val)) if isinstance(val, bool) else float(val)
            oom_mask[i, j] = not dr.fits_in_memory

    if title is None:
        title = f"{metric.replace('_', ' ')} — Edge Device Projection"

    fig_w = max(8, n_devices * 1.7)
    fig_h = max(3, n_trackers * 0.85 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colour direction: low-latency = green; low-FPS = red
    cmap = "YlOrRd_r" if "latency" in metric or "energy" in metric else "YlGn"
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0)

    # Cell annotations and OOM hatching
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for i in range(n_trackers):
        for j in range(n_devices):
            val = matrix[i, j]
            label = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
            color = "white" if val < 0.55 * vmax else "black"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
            if mark_oom and oom_mask[i, j]:
                ax.add_patch(
                    plt.Rectangle(  # type: ignore[attr-defined]
                        (j - 0.5, i - 0.5), 1.0, 1.0,
                        fill=False, hatch="////", edgecolor="red",
                        linewidth=0.5, alpha=0.8,
                    )
                )

    ax.set_xticks(range(n_devices))
    ax.set_xticklabels(device_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_trackers))
    ax.set_yticklabels(tracker_names, fontsize=10)
    ax.set_title(title, fontsize=13, pad=10)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(metric.replace("_", " "), fontsize=10)

    if mark_oom and oom_mask.any():
        # Add a proxy artist for the OOM hatch legend
        from matplotlib.patches import Patch  # type: ignore[import]
        oom_patch = Patch(fill=False, hatch="////",
                          edgecolor="red", label="OOM (exceeds device RAM)")
        ax.legend(handles=[oom_patch], loc="upper right", fontsize=9,
                  framealpha=0.8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[frontier] Device projection saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Plot 3: Frame-Skip Degradation Curves
# ---------------------------------------------------------------------------


def plot_frame_skip_degradation(
    analyses: List[Any],
    metric: str = "mean_iou",
    output_path: Optional[str] = None,
    title: str = "Accuracy vs. Frame Skip Rate",
    show_optimal: bool = True,
) -> None:
    """Line plot of accuracy degradation vs. temporal skip rate.

    Each line represents one :class:`~eovot.analysis.frame_skip.FrameSkipAnalysis`.
    X-axis is the skip rate (1 = every frame; k = every k-th frame).
    Y-axis is the chosen accuracy metric.  Dotted vertical lines mark
    each tracker's ``optimal_skip_rate`` when ``show_optimal=True``.

    Args:
        analyses: List of :class:`~eovot.analysis.frame_skip.FrameSkipAnalysis`
            objects from :class:`~eovot.analysis.frame_skip.FrameSkipEvaluator`.
        metric: Attribute on :class:`~eovot.analysis.frame_skip.SkipRateResult`
            to plot.  One of ``"mean_iou"``, ``"success_auc"``,
            ``"failure_rate"``.
        output_path: Save path.  Interactive display when ``None``.
        title: Figure title.
        show_optimal: If ``True``, draw a vertical dotted line at each
            tracker's optimal skip rate.

    Example::

        evaluator = FrameSkipEvaluator(skip_rates=[1, 2, 4, 8])
        analysis  = evaluator.evaluate(tracker, dataset, "OTB100")
        plot_frame_skip_degradation([analysis], output_path="frame_skip.png")
    """
    plt = _plt()

    if not analyses:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for analysis in analyses:
        if not analysis.skip_results:
            continue

        skip_rates = [r.skip_rate for r in analysis.skip_results]
        values = [getattr(r, metric, 0.0) for r in analysis.skip_results]

        line, = ax.plot(
            skip_rates, values,
            marker="o", linewidth=2,
            label=analysis.tracker_name,
        )

        if show_optimal:
            opt = analysis.optimal_skip_rate
            if opt > 1:
                ax.axvline(
                    opt,
                    color=line.get_color(),
                    linestyle=":",
                    alpha=0.55,
                    linewidth=1.2,
                )
                ax.annotate(
                    f"opt={opt}×",
                    xy=(opt, min(values)),
                    xytext=(opt + 0.1, min(values) + 0.01),
                    fontsize=8,
                    color=line.get_color(),
                )

    ax.set_xlabel("Frame Skip Rate", fontsize=12)
    ylabel_map = {
        "mean_iou": "Mean IoU",
        "success_auc": "Success AUC",
        "failure_rate": "Failure Rate",
    }
    ax.set_ylabel(ylabel_map.get(metric, metric), fontsize=12)
    ax.set_title(title, fontsize=14)

    all_skip_rates = [r.skip_rate for a in analyses for r in a.skip_results]
    if all_skip_rates:
        ax.set_xticks(sorted(set(all_skip_rates)))

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[frontier] Frame-skip degradation saved → {output_path}")
    else:
        plt.show()
