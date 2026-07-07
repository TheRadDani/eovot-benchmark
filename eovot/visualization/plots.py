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
    from eovot.visualization.plots import plot_success_curves, plot_tracker_comparison

    with open("results/MOSSE-OTB100.json") as f:
        mosse = json.load(f)
    with open("results/KCF-OTB100.json") as f:
        kcf = json.load(f)

    plot_success_curves([mosse, kcf], output_path="success_curves.png")
    plot_tracker_comparison([mosse, kcf], output_path="comparison.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# NumPy 2.0 renamed trapz → trapezoid; support both.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _tracker_label(result: Dict[str, Any]) -> str:
    """Extract tracker name from a result summary, handling both key variants."""
    s = result.get("summary", {})
    return s.get("tracker") or s.get("tracker_name", "unknown")


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
        tracker_name = _tracker_label(result)
        ious = _collect_ious(result)
        if len(ious) == 0:
            continue
        success_rates = np.array([(ious >= t).mean() for t in thresholds])
        auc = float(_trapz(success_rates, thresholds))
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
        tracker_name = _tracker_label(result)
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

    labels = [_tracker_label(r) for r in results]
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
