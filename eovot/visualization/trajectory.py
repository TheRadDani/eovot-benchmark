"""Per-sequence tracking quality visualisation for EOVOT.

Complements :mod:`eovot.visualization.plots` — which handles aggregate
success/precision curves — with per-sequence and per-run visual diagnostics:

- :func:`plot_iou_timeline` — per-frame IoU over time for one sequence, with
  shaded failure regions and configurable threshold markers.
- :func:`plot_sequence_heatmap` — compact heatmap of accuracy and profiling
  metrics across all sequences in a benchmark run.
- :func:`plot_multi_tracker_iou_timeline` — overlay IoU timelines for
  multiple trackers evaluated on the same sequence.

All functions are ``output_path``-optional: when ``None`` the figure is shown
interactively; when a path is provided the figure is saved and closed so the
function is safe to call in headless / CI environments.

Requires ``matplotlib``.  Install with::

    pip install matplotlib

Example::

    import json, numpy as np
    from eovot.visualization.trajectory import (
        plot_iou_timeline,
        plot_sequence_heatmap,
        plot_multi_tracker_iou_timeline,
    )

    with open("results/MOSSE-OTB100.json") as f:
        mosse = json.load(f)

    seq = mosse["sequences"][0]
    plot_iou_timeline(
        np.array(seq["ious"]),
        sequence_name=seq["sequence_name"],
        tracker_name="MOSSE",
        output_path="plots/basketball_iou.png",
    )

    plot_sequence_heatmap(mosse, output_path="plots/mosse_heatmap.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _get_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for EOVOT visualization.\n"
            "Install it with:  pip install matplotlib"
        ) from exc


def plot_iou_timeline(
    ious: np.ndarray,
    sequence_name: str = "sequence",
    tracker_name: str = "tracker",
    failure_threshold: float = 0.1,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plot per-frame IoU over time for a single sequence.

    Frames below *failure_threshold* are highlighted with a translucent red
    background band, making failure onset and duration immediately visible.
    Reference lines are drawn at the failure threshold and at IoU = 0.5 (the
    canonical OTB success threshold).

    Args:
        ious: Per-frame IoU values, shape ``(N,)``.
        sequence_name: Sequence identifier shown in the legend and title.
        tracker_name: Tracker identifier shown in the title.
        failure_threshold: IoU below this value is considered a tracking
            failure (default ``0.1``, matching OTB convention).
        output_path: Save path (PNG / PDF / SVG).  When ``None`` the figure
            is shown interactively.
        title: Override the auto-generated figure title.
    """
    plt = _get_matplotlib()

    ious = np.asarray(ious, dtype=np.float64)
    frames = np.arange(len(ious))
    mean_iou = float(ious.mean()) if len(ious) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 4))

    # Shade failure regions.
    in_failure = False
    f_start = 0
    for i, v in enumerate(ious):
        if v < failure_threshold and not in_failure:
            f_start = i
            in_failure = True
        elif v >= failure_threshold and in_failure:
            ax.axvspan(f_start, i, alpha=0.15, color="red", label="_nolegend_")
            in_failure = False
    if in_failure:
        ax.axvspan(f_start, len(ious), alpha=0.15, color="red", label="_nolegend_")

    ax.plot(frames, ious, color="steelblue", linewidth=1.2, label=sequence_name)
    ax.axhline(
        mean_iou, color="steelblue", linestyle="--", linewidth=1.0, alpha=0.7,
        label=f"mean IoU = {mean_iou:.3f}",
    )
    ax.axhline(
        failure_threshold, color="red", linestyle=":", linewidth=1.0, alpha=0.75,
        label=f"failure threshold ({failure_threshold})",
    )
    ax.axhline(
        0.5, color="forestgreen", linestyle=":", linewidth=1.0, alpha=0.55,
        label="IoU = 0.5",
    )

    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title(
        title or f"{tracker_name} — {sequence_name} (IoU timeline)",
        fontsize=13,
    )
    ax.set_xlim(0, max(1, len(ious) - 1))
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] IoU timeline saved → {output_path}")
    else:
        plt.show()


def plot_sequence_heatmap(
    result_dict: Dict[str, Any],
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    max_sequences: int = 50,
) -> None:
    """Plot a heatmap of per-sequence metrics for a single tracker benchmark run.

    Each row is a sequence; each column is a metric.  Colour intensity (using
    a red→yellow→green diverging map) reveals which sequences were hardest and
    which consumed the most resources, at a glance.

    Args:
        result_dict: Dict from
            :meth:`~eovot.benchmark.engine.BenchmarkResult.to_dict`.
            Must contain a ``"sequences"`` list, each with at least one of
            the keys in *metrics*.
        metrics: Metric keys to visualise from each sequence dict.  Defaults
            to ``["mean_iou", "fps", "peak_memory_mb"]``.
        output_path: Save path.  Interactive display when ``None``.
        title: Override the auto-generated figure title.
        max_sequences: Cap the number of rows shown (first N sequences) to
            keep the figure readable.  Default ``50``.
    """
    plt = _get_matplotlib()

    if metrics is None:
        metrics = ["mean_iou", "fps", "peak_memory_mb"]

    sequences = result_dict.get("sequences", [])
    if not sequences:
        print("[heatmap] No sequence data in result_dict; nothing to plot.")
        return

    if len(sequences) > max_sequences:
        sequences = sequences[:max_sequences]

    seq_names = [s.get("sequence_name", f"seq_{i}") for i, s in enumerate(sequences)]
    data = np.zeros((len(sequences), len(metrics)), dtype=np.float64)
    for i, seq in enumerate(sequences):
        for j, m in enumerate(metrics):
            data[i, j] = float(seq.get(m, 0.0))

    tracker_name = result_dict.get("summary", {}).get("tracker", "Tracker")
    fig_h = max(4, len(seq_names) * 0.30)
    fig, ax = plt.subplots(figsize=(max(5, len(metrics) * 2.6), fig_h))

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(
        [m.replace("_", " ") for m in metrics],
        fontsize=9, rotation=20, ha="right",
    )
    ax.set_yticks(range(len(seq_names)))
    ax.set_yticklabels(seq_names, fontsize=7)

    # Annotate cells with raw values.
    for i in range(len(seq_names)):
        for j in range(len(metrics)):
            val = data[i, j]
            col_max = data[:, j].max()
            brightness = val / col_max if col_max > 0 else 0.0
            text_color = "black" if 0.25 < brightness < 0.80 else "white"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=6, color=text_color,
            )

    fig.colorbar(im, ax=ax, shrink=0.6, label="Value (column-normalised colour scale)")
    ax.set_title(
        title or f"{tracker_name} — Per-Sequence Metrics Heatmap",
        fontsize=12,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Sequence heatmap saved → {output_path}")
    else:
        plt.show()


def plot_multi_tracker_iou_timeline(
    tracker_ious: Dict[str, np.ndarray],
    sequence_name: str = "sequence",
    failure_threshold: float = 0.1,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Overlay IoU timelines for multiple trackers evaluated on the same sequence.

    Each tracker gets a distinct coloured line.  The mean IoU per tracker is
    shown in the legend.  A failure-threshold reference line and an IoU = 0.5
    guide line are included.

    Args:
        tracker_ious: Mapping ``{tracker_name: ious_array}`` where each array
            has shape ``(N,)``.  Arrays of different lengths are supported
            (each is plotted against its own frame indices).
        sequence_name: Sequence identifier used in the figure title.
        failure_threshold: Reference line drawn at this IoU level.
        output_path: Save path.  Interactive display when ``None``.
        title: Override the auto-generated figure title.
    """
    plt = _get_matplotlib()

    if not tracker_ious:
        print("[plot] tracker_ious is empty; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for tracker_name, ious in tracker_ious.items():
        ious = np.asarray(ious, dtype=np.float64)
        frames = np.arange(len(ious))
        mean_iou = float(ious.mean()) if len(ious) > 0 else 0.0
        ax.plot(frames, ious, linewidth=1.5, label=f"{tracker_name} (mean={mean_iou:.3f})")

    ax.axhline(
        failure_threshold, color="red", linestyle=":", linewidth=1.0, alpha=0.65,
        label=f"failure threshold ({failure_threshold})",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.45)

    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title(title or f"IoU Timeline Comparison — {sequence_name}", fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Multi-tracker IoU timeline saved → {output_path}")
    else:
        plt.show()
