"""Publication-quality plots for EOVOT benchmark results.

All plotting functions accept result dicts produced by
:meth:`~eovot.benchmark.engine.BenchmarkEngine.run`, which have the form::

    {
        "summary": {"tracker_name": ..., "mean_iou": ..., "mean_fps": ..., ...},
        "sequences": [
            {"sequence_name": ..., "ious": [...], "fps": ..., ...},
            ...
        ]
    }

Requires ``matplotlib``.  Install with::

    pip install matplotlib

Example::

    import json
    from eovot.visualization.plots import (
        plot_success_curves,
        plot_tracker_comparison,
        plot_edge_scatter,
    )

    with open("results/MOSSE-OTB100.json") as f:
        mosse = json.load(f)
    with open("results/KCF-OTB100.json") as f:
        kcf = json.load(f)

    plot_success_curves([mosse, kcf], output_path="success_curves.png")
    plot_edge_scatter([mosse, kcf], output_path="edge_scatter.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _get_matplotlib():
    """Import matplotlib.pyplot, raising a clear error if absent."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for EOVOT visualization.\n"
            "Install it with:  pip install matplotlib"
        ) from exc


def _collect_ious(result: Dict[str, Any]) -> np.ndarray:
    """Gather all per-frame IoU values from a benchmark result dict.

    The ``"sequences"`` list may contain either raw ``"ious"`` arrays
    (written by the current engine) or only ``"mean_iou"`` scalars (legacy
    format).  In the scalar case a single-element array is returned per
    sequence so the success curve degrades gracefully.
    """
    ious_list = []
    for seq in result.get("sequences", []):
        raw = seq.get("ious")
        if raw is not None:
            ious_list.extend(raw)
        else:
            ious_list.append(seq.get("mean_iou", 0.0))
    return np.array(ious_list, dtype=np.float64)


def plot_success_curves(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Success Curves",
    thresholds: Optional[np.ndarray] = None,
) -> None:
    """Plot overlap success curves for one or more benchmark results.

    A *success curve* sweeps an IoU threshold from 0 to 1 and plots the
    fraction of frames whose predicted box exceeds that threshold.  The
    area under the curve (AUC) is the canonical single-number summary used
    in OTB, GOT-10k, and LaSOT papers.

    Args:
        results: List of dicts from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
        output_path: If given, save the figure to this path (PNG/PDF/SVG).
            When ``None`` the plot is shown interactively.
        title: Figure title string.
        thresholds: IoU threshold sweep values.  Defaults to
            ``np.linspace(0, 1, 101)``.
    """
    plt = _get_matplotlib()

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    fig, ax = plt.subplots(figsize=(8, 5))

    for result in results:
        tracker_name = result.get("summary", {}).get("tracker_name", "unknown")
        ious = _collect_ious(result)
        if len(ious) == 0:
            continue
        success_rates = np.array([(ious >= t).mean() for t in thresholds])
        auc = float(np.trapz(success_rates, thresholds))
        ax.plot(
            thresholds,
            success_rates,
            label=f"{tracker_name} (AUC={auc:.3f})",
            linewidth=2,
        )

    ax.set_xlabel("Overlap Threshold", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Success curves saved → {output_path}")
    else:
        plt.show()


def plot_precision_curves(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Precision Curves",
    thresholds: Optional[np.ndarray] = None,
) -> None:
    """Plot centre-distance precision curves for one or more benchmark results.

    A *precision curve* sweeps a distance threshold from 0 to 50 px and
    plots the fraction of frames whose predicted centre is within that
    distance of the ground-truth centre.  The precision score at 20 px is
    the canonical OTB scalar.

    Args:
        results: List of dicts from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
            Each sequence entry must include a ``"center_distances"`` list;
            sequences without it are skipped.
        output_path: Save path.  Interactive display when ``None``.
        title: Figure title string.
        thresholds: Distance thresholds in pixels.  Defaults to
            ``np.linspace(0, 50, 51)``.
    """
    plt = _get_matplotlib()

    if thresholds is None:
        thresholds = np.linspace(0.0, 50.0, 51)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0

    for result in results:
        tracker_name = result.get("summary", {}).get("tracker_name", "unknown")
        dists_list = []
        for seq in result.get("sequences", []):
            raw = seq.get("center_distances")
            if raw is not None:
                dists_list.extend(raw)

        if not dists_list:
            continue

        dists = np.array(dists_list, dtype=np.float64)
        precision_rates = np.array([(dists < t).mean() for t in thresholds])
        score_at_20 = float(np.interp(20.0, thresholds, precision_rates))
        ax.plot(
            thresholds,
            precision_rates,
            label=f"{tracker_name} (@20px={score_at_20:.3f})",
            linewidth=2,
        )
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5, 0.5,
            "No centre-distance data available.\nRe-run with a future engine version.",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )

    ax.set_xlabel("Centre-Distance Threshold (px)", fontsize=12)
    ax.set_ylabel("Precision Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Precision curves saved → {output_path}")
    else:
        plt.show()


def plot_tracker_comparison(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: str = "Tracker Comparison",
) -> None:
    """Plot a grouped bar chart comparing trackers across multiple metrics.

    Args:
        results: List of dicts from :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`.
        metrics: Metric keys to plot from each ``"summary"`` dict.  Defaults
            to ``["mean_iou", "mean_fps", "peak_memory_mb"]``.
        output_path: Save path.  Interactive display when ``None``.
        title: Figure title string.
    """
    plt = _get_matplotlib()
    import matplotlib

    if metrics is None:
        metrics = ["mean_iou", "mean_fps", "peak_memory_mb"]

    labels = [r.get("summary", {}).get("tracker_name", f"tracker_{i}")
              for i, r in enumerate(results)]
    values = {
        m: [r.get("summary", {}).get(m, 0.0) for r in results]
        for m in metrics
    }

    # Normalise each metric to [0, 1] so all bars live on the same axis.
    # We also track the raw max to annotate bars.
    n_metrics = len(metrics)
    n_trackers = len(results)
    x = np.arange(n_trackers)
    bar_width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(6, n_trackers * 2), 5))
    color_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

    for m_idx, metric in enumerate(metrics):
        raw = np.array(values[metric], dtype=np.float64)
        max_val = raw.max() if raw.max() > 0 else 1.0
        normalised = raw / max_val
        offset = (m_idx - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            normalised,
            width=bar_width * 0.9,
            label=metric,
            color=color_cycle[m_idx % len(color_cycle)],
            alpha=0.85,
        )
        # Annotate each bar with the raw value.
        for bar, raw_val in zip(bars, raw):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{raw_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Normalised Score (raw value annotated)", fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0.0, 1.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Tracker comparison saved → {output_path}")
    else:
        plt.show()


def plot_edge_scatter(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Edge Deployment Trade-off: Accuracy vs. Throughput",
    memory_budget_mb: float = 512.0,
    show_pareto: bool = True,
) -> None:
    """Bubble scatter plot of IoU vs. FPS coloured by Edge Efficiency Score (EES).

    Visualises EOVOT's core edge-deployment thesis: *both* accuracy and
    throughput must be high, and memory consumption must remain within budget.

    - **X axis** — mean IoU (accuracy)
    - **Y axis** — mean FPS (throughput, log scale)
    - **Bubble area** — peak memory in MB (larger = more memory)
    - **Colour** — Edge Efficiency Score (EES = mIoU × log1p(FPS) / memory_factor)
    - **Orange cross** — Pareto-optimal trackers

    Args:
        results: List of result dicts from
            :meth:`~eovot.benchmark.engine.BenchmarkEngine.run`, one per
            tracker / dataset combination.
        output_path: File path to save the figure (PNG/PDF/SVG).
            Shows interactively when ``None``.
        title: Figure title string.
        memory_budget_mb: Reference memory budget in MB used to compute EES
            and scale the legend annotation.  Default: ``512.0``.
        show_pareto: When ``True``, Pareto-optimal trackers are highlighted
            with an orange ring.  Default: ``True``.

    Example::

        from eovot.visualization.plots import plot_edge_scatter

        plot_edge_scatter(
            [mosse_result, kcf_result, csrt_result],
            output_path="edge_scatter.png",
        )
    """
    import math

    plt = _get_matplotlib()
    import matplotlib.pyplot as _plt
    import matplotlib.cm as cm

    if not results:
        return

    labels, ious, fpss, mems, eess = [], [], [], [], []
    for r in results:
        s = r.get("summary", {})
        tracker = s.get("tracker") or s.get("tracker_name", "?")
        miou  = float(s.get("mean_iou", 0.0))
        fps   = float(s.get("mean_fps", 1.0))
        mem   = float(s.get("peak_memory_mb", 1.0))
        ees   = miou * math.log1p(fps) / (1.0 + mem / memory_budget_mb)
        labels.append(tracker)
        ious.append(miou)
        fpss.append(max(fps, 1e-3))
        mems.append(max(mem, 1.0))
        eess.append(ees)

    ious_arr = np.array(ious)
    fpss_arr = np.array(fpss)
    mems_arr = np.array(mems)
    eess_arr = np.array(eess)

    # Bubble area scaled so the range [min_mem, max_mem] maps to [100, 2000] pt²
    mem_min, mem_max = mems_arr.min(), mems_arr.max()
    mem_range = max(mem_max - mem_min, 1.0)
    bubble_sizes = 100 + 1900 * (mems_arr - mem_min) / mem_range

    # Identify Pareto front in (mIoU, EES) space for optional highlighting
    pareto_flags = [True] * len(results)
    if show_pareto:
        for i in range(len(results)):
            for j in range(len(results)):
                if i == j:
                    continue
                if (
                    ious_arr[j] >= ious_arr[i]
                    and eess_arr[j] >= eess_arr[i]
                    and (ious_arr[j] > ious_arr[i] or eess_arr[j] > eess_arr[i])
                ):
                    pareto_flags[i] = False
                    break

    fig, ax = plt.subplots(figsize=(9, 6))

    # Colour map over EES values
    norm = _plt.Normalize(vmin=eess_arr.min(), vmax=max(eess_arr.max(), 1e-6))
    cmap = cm.viridis

    sc = ax.scatter(
        ious_arr,
        fpss_arr,
        s=bubble_sizes,
        c=eess_arr,
        cmap=cmap,
        norm=norm,
        alpha=0.80,
        edgecolors="grey",
        linewidths=0.6,
        zorder=3,
    )

    # Pareto-front ring
    if show_pareto:
        pareto_x = ious_arr[pareto_flags]
        pareto_y = fpss_arr[pareto_flags]
        pareto_s = bubble_sizes[pareto_flags]
        if len(pareto_x):
            ax.scatter(
                pareto_x,
                pareto_y,
                s=pareto_s * 1.45,
                facecolors="none",
                edgecolors="darkorange",
                linewidths=2.2,
                zorder=4,
                label="Pareto front",
            )

    # Tracker labels
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl,
            (ious_arr[i], fpss_arr[i]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            zorder=5,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Edge Efficiency Score (EES)", fontsize=10)

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Mean IoU (accuracy)", fontsize=12)
    ax.set_ylabel("Mean FPS (throughput, log scale)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.25)

    # Legend: bubble size → memory
    for mem_val in [mem_min, (mem_min + mem_max) / 2, mem_max]:
        s_val = 100 + 1900 * (mem_val - mem_min) / mem_range
        ax.scatter([], [], s=s_val, c="grey", alpha=0.5,
                   label=f"{mem_val:.0f} MB")
    ax.legend(title="Peak Memory", fontsize=9, title_fontsize=9,
              loc="lower right", framealpha=0.8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Edge scatter saved → {output_path}")
    else:
        plt.show()
