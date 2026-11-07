"""Multi-metric radar chart for publication-quality tracker comparison.

Radar charts (also called spider or web charts) are the standard visual for
comparing tracking algorithms across heterogeneous axes in a single figure.
EOVOT evaluates trackers on five complementary dimensions that cannot be
collapsed into a single scalar without losing information:

Axes (all normalised to ``[0, 1]``, higher = better)
------------------------------------------------------
1. **Accuracy** — mean IoU across all evaluated frames.
2. **Speed** — log-normalised FPS: ``log1p(fps) / log1p(fps_max)``.
   Logarithmic scaling applies diminishing returns (500→550 FPS matters less
   than 5→55 FPS on edge hardware).
3. **Robustness** — fraction of frames the tracker stayed alive (IoU ≥ 0.1).
   Computed from ``RobustnessAnalyzer`` output or approximated from mean IoU.
4. **Smoothness** — temporal consistency score from ``TemporalConsistencyAnalyzer``
   (or 1 / (1 + position_jitter) approximation when only predictions are
   available).
5. **Efficiency** — Edge Efficiency Score normalised by the maximum EES in the
   comparison set.  Rewards trackers that deliver accuracy per compute dollar.

Data input
----------
:class:`RadarChartBuilder` accepts a list of :class:`RadarEntry` objects, each
providing the five raw metric values for one tracker.  Normalisation is done
*across the comparison set* so the chart reflects relative rather than
absolute performance.  This is the correct framing for head-to-head comparison:
the best tracker on each axis scores 1.0.

Matplotlib is optional.  :meth:`RadarChartBuilder.to_dict` returns the
normalised data as a plain dict suitable for JSON export or integration into
custom plotting pipelines (D3.js, Vega-Altair, etc.).

Example::

    from eovot.visualization.radar import RadarChartBuilder, RadarEntry

    entries = [
        RadarEntry("MOSSE",   mean_iou=0.42, fps=312, survival_rate=0.78,
                   smoothness_score=0.81, ees=0.21),
        RadarEntry("KCF",     mean_iou=0.51, fps=180, survival_rate=0.83,
                   smoothness_score=0.74, ees=0.28),
        RadarEntry("CSRT",    mean_iou=0.68, fps=42,  survival_rate=0.91,
                   smoothness_score=0.69, ees=0.19),
    ]

    builder = RadarChartBuilder(entries)
    builder.plot(output_path="radar.png", title="OTB-100 Tracker Comparison")
    data = builder.to_dict()   # JSON-serialisable summary
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Axis labels displayed on the chart — order must match _normalise() output.
_AXES = ["Accuracy", "Speed", "Robustness", "Smoothness", "Efficiency"]
_N_AXES = len(_AXES)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RadarEntry:
    """Five-dimensional metric summary for one tracker.

    All raw values are in their natural units; normalisation is performed by
    :class:`RadarChartBuilder` *across the full comparison set*.

    Attributes:
        tracker_name:    Human-readable identifier used in the legend.
        mean_iou:        Mean IoU in ``[0, 1]``.
        fps:             Mean frames-per-second throughput (> 0).
        survival_rate:   Fraction of frames the tracker stayed alive ``[0, 1]``.
                         Pass ``mean_iou`` as a proxy when robustness data is
                         unavailable (conservative approximation).
        smoothness_score: Temporal smoothness in ``(0, 1]``; use
                          ``TemporalConsistencyResult.smoothness_score`` or
                          ``1.0`` if not computed.
        ees:             Edge Efficiency Score (non-negative); use
                         ``EfficiencyEntry.ees`` or ``mean_iou`` as a proxy.
    """

    tracker_name: str
    mean_iou: float
    fps: float
    survival_rate: float
    smoothness_score: float
    ees: float

    def raw_values(self) -> Tuple[float, float, float, float, float]:
        """Return ``(accuracy, fps, survival, smoothness, ees)`` as a tuple."""
        return (self.mean_iou, self.fps, self.survival_rate,
                self.smoothness_score, self.ees)

    @classmethod
    def from_benchmark_result(
        cls,
        result: "Any",
        survival_rate: float = 0.0,
        smoothness_score: float = 1.0,
        ees: float = 0.0,
    ) -> "RadarEntry":
        """Convenience constructor from a :class:`~eovot.benchmark.engine.BenchmarkResult`.

        Args:
            result:          A ``BenchmarkResult`` object.
            survival_rate:   From :class:`~eovot.metrics.robustness.RobustnessAnalyzer`;
                             defaults to 0 (use ``mean_iou`` as a proxy instead).
            smoothness_score: From :class:`~eovot.metrics.temporal.TemporalConsistencyAnalyzer`.
            ees:             From :class:`~eovot.metrics.efficiency.EfficiencyMetricsEngine`.

        Returns:
            :class:`RadarEntry` populated from the benchmark result.
        """
        return cls(
            tracker_name=result.tracker_name,
            mean_iou=result.mean_iou,
            fps=result.mean_fps,
            survival_rate=survival_rate if survival_rate > 0 else result.mean_iou,
            smoothness_score=smoothness_score,
            ees=ees if ees > 0 else result.mean_iou,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class RadarChartBuilder:
    """Build and optionally render a multi-metric radar chart.

    Args:
        entries:    List of :class:`RadarEntry` objects — one per tracker.
                    At least two entries are recommended for a meaningful
                    comparison.
        fill_alpha: Transparency of the filled polygon (``0`` = no fill).
                    Default ``0.15``.

    Raises:
        ValueError: If ``entries`` is empty.

    Example::

        builder = RadarChartBuilder([entry_mosse, entry_kcf, entry_csrt])
        builder.plot("chart.png")
        print(builder.to_dict())
    """

    def __init__(
        self,
        entries: List[RadarEntry],
        fill_alpha: float = 0.15,
    ) -> None:
        if not entries:
            raise ValueError("entries must contain at least one RadarEntry")
        self.entries = entries
        self.fill_alpha = fill_alpha
        self._normalised: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalise(self) -> np.ndarray:
        """Compute ``(n_trackers, 5)`` array of scores normalised to ``[0, 1]``.

        Each axis is normalised independently across the comparison set:
        the tracker with the highest raw value scores ``1.0`` on that axis.

        Speed is log-compressed before normalisation::

            log_fps = log1p(fps)
            norm_speed = log_fps / max(log_fps)

        All other axes are linearly normalised by their maximum.

        Returns:
            Float array of shape ``(n_trackers, 5)``.
        """
        raw = np.array(
            [e.raw_values() for e in self.entries], dtype=np.float64
        )  # (N, 5)

        normed = raw.copy()
        # Axis 1 (speed): log-compress first
        normed[:, 1] = np.log1p(np.maximum(raw[:, 1], 0.0))

        # Normalise each axis by its maximum (guard against all-zero axes)
        for col in range(_N_AXES):
            col_max = normed[:, col].max()
            if col_max > 1e-9:
                normed[:, col] /= col_max

        self._normalised = normed
        return normed

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict with normalised scores and metadata.

        Returns:
            Dict with keys:

            * ``"axes"`` — list of axis names.
            * ``"trackers"`` — list of dicts, one per tracker, each with
              ``"name"`` and per-axis normalised scores.
            * ``"raw"`` — same structure with raw (un-normalised) values.
        """
        normed = self.normalise()

        trackers_normed = []
        trackers_raw = []
        for i, entry in enumerate(self.entries):
            row_n = {_AXES[j]: round(float(normed[i, j]), 4) for j in range(_N_AXES)}
            row_n["name"] = entry.tracker_name
            trackers_normed.append(row_n)

            raw_vals = entry.raw_values()
            row_r = {_AXES[j]: round(float(raw_vals[j]), 4) for j in range(_N_AXES)}
            row_r["name"] = entry.tracker_name
            trackers_raw.append(row_r)

        return {
            "axes": list(_AXES),
            "trackers": trackers_normed,
            "raw": trackers_raw,
        }

    # ------------------------------------------------------------------
    # Matplotlib rendering
    # ------------------------------------------------------------------

    def plot(
        self,
        output_path: Optional[str] = None,
        title: str = "Tracker Comparison",
        figsize: Tuple[float, float] = (7.0, 7.0),
        colors: Optional[List[str]] = None,
        legend_loc: str = "upper right",
    ) -> None:
        """Render the radar chart with matplotlib.

        Args:
            output_path: File path to save the figure (``".png"``, ``".pdf"``,
                         ``".svg"``).  When ``None``, the figure is displayed
                         interactively with ``plt.show()``.
            title:       Chart title.
            figsize:     Matplotlib figure size in inches.
            colors:      List of colour strings, one per tracker.  Defaults to
                         the ``tab10`` palette.
            legend_loc:  Matplotlib legend location string.

        Raises:
            ImportError: If ``matplotlib`` is not installed.
        """
        plt = _get_matplotlib()
        normed = self.normalise()

        # Compute angles for each axis (evenly spaced, closed polygon)
        angles = np.linspace(0, 2 * math.pi, _N_AXES, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": "polar"},
        )

        default_colors = plt.cm.tab10(np.linspace(0, 0.9, len(self.entries)))
        palette = colors or [
            f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            for c in default_colors
        ]

        for i, entry in enumerate(self.entries):
            values = normed[i].tolist()
            values += values[:1]  # close the loop
            color = palette[i % len(palette)]
            ax.plot(angles, values, linewidth=1.8, linestyle="solid",
                    label=entry.tracker_name, color=color)
            ax.fill(angles, values, alpha=self.fill_alpha, color=color)

        # Axis labels
        ax.set_thetagrids(
            [a * 180 / math.pi for a in angles[:-1]],
            _AXES,
            fontsize=11,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")
        ax.set_title(title, pad=20, fontsize=13, fontweight="bold")
        ax.legend(loc=legend_loc, bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Return a Markdown table of normalised per-axis scores.

        Useful for embedding in paper appendices or README files.

        Returns:
            Multi-line Markdown string with a header row, separator, and one
            data row per tracker.
        """
        normed = self.normalise()
        col_w = max(len(a) for a in _AXES) + 2
        name_w = max(len(e.tracker_name) for e in self.entries) + 2

        header = f"| {'Tracker':<{name_w}} | " + " | ".join(
            f"{a:^{col_w}}" for a in _AXES
        ) + " |"
        sep = f"| {'-' * name_w} | " + " | ".join("-" * col_w for _ in _AXES) + " |"

        rows = [header, sep]
        for i, entry in enumerate(self.entries):
            cells = [f"{normed[i, j]:^{col_w}.3f}" for j in range(_N_AXES)]
            rows.append(
                f"| {entry.tracker_name:<{name_w}} | " + " | ".join(cells) + " |"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_matplotlib():
    """Import matplotlib.pyplot, raising a clear error if absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for headless environments
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for radar chart rendering.\n"
            "Install it with:  pip install matplotlib"
        ) from exc
