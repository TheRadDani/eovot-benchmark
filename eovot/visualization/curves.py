"""Success and precision curve visualisation for EOVOT benchmarks.

Generates publication-ready plots from
:class:`~eovot.benchmark.engine.BenchmarkResult` objects following the
standard VOT evaluation protocol used in OTB, LaSOT, and GOT-10k papers:

* **Success curve** — fraction of frames with IoU above a threshold,
  swept from 0 → 1.  Legend score = AUC (higher is better).
* **Precision curve** — fraction of frames with predicted-centre distance
  below a pixel threshold, swept from 0 → 50 px.  Legend score = precision
  at the canonical 20 px threshold (higher is better).

Requires ``matplotlib``::

    pip install matplotlib

Typical usage::

    from eovot.visualization.curves import CurvePlotter

    plotter = CurvePlotter("results/plots")
    plotter.plot_success([result_mosse, result_kcf], title="OTB100")
    plotter.plot_precision([result_mosse, result_kcf], title="OTB100")

    # Or generate both in one call:
    paths = plotter.plot_all([result_mosse, result_kcf], title="OTB100")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe; must be set before importing pyplot
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from ..benchmark.engine import BenchmarkResult
from ..metrics.accuracy import MetricsEngine

# Colour-blind-friendly palette (Okabe–Ito, 8 colours)
_COLOURS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for curve visualisation. "
            "Install it with:  pip install matplotlib"
        )


def _safe_stem(title: str) -> str:
    """Convert a plot title to a filesystem-safe filename stem."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in title.lower()).strip("_")


class CurvePlotter:
    """Generate and save success / precision curve plots.

    Args:
        output_dir: Directory where PNG files are written.
            Created automatically when absent.
        dpi: Figure resolution in dots-per-inch. Default: ``150``.
        figsize: Figure size in inches ``(width, height)``. Default: ``(7, 5)``.
    """

    def __init__(
        self,
        output_dir: str = "results/plots",
        dpi: int = 150,
        figsize: tuple = (7, 5),
    ) -> None:
        _require_matplotlib()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self._engine = MetricsEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_success(
        self,
        results: List[BenchmarkResult],
        title: str = "Success Plot",
        filename: Optional[str] = None,
    ) -> Path:
        """Plot the IoU-threshold success curve for one or more trackers.

        The success rate at each threshold is the fraction of frames whose
        predicted-box IoU exceeds that threshold.  The AUC (area under the
        curve) is shown in each legend entry.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker / configuration to compare.
            title: Plot title; also used as the filename stem when *filename*
                is ``None``.
            filename: Override the output filename stem (no extension).

        Returns:
            Absolute :class:`~pathlib.Path` of the saved PNG.
        """
        _require_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, result in enumerate(results):
            all_ious = np.concatenate(
                [sr.ious for sr in result.sequence_results]
            )
            thresholds, rates = self._engine.success_curve(all_ious)
            auc = float(np.trapz(rates, thresholds))

            colour = _COLOURS[i % len(_COLOURS)]
            marker = _MARKERS[i % len(_MARKERS)]
            ax.plot(
                thresholds,
                rates,
                color=colour,
                marker=marker,
                markevery=10,
                linewidth=1.8,
                markersize=5,
                label=f"{result.tracker_name} [AUC={auc:.3f}]",
            )

        ax.set_xlabel("Overlap Threshold", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        stem = filename or _safe_stem(title)
        out_path = self.output_dir / f"{stem}_success.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_precision(
        self,
        results: List[BenchmarkResult],
        title: str = "Precision Plot",
        filename: Optional[str] = None,
    ) -> Path:
        """Plot the centre-distance precision curve for one or more trackers.

        The precision at each threshold is the fraction of frames whose
        predicted-box centre is within that many pixels of the ground-truth
        centre.  The legend shows the precision at the canonical 20 px
        threshold.

        Args:
            results: One :class:`~eovot.benchmark.engine.BenchmarkResult`
                per tracker / configuration to compare.
            title: Plot title.
            filename: Override the output filename stem.

        Returns:
            Absolute :class:`~pathlib.Path` of the saved PNG.

        Note:
            Requires ``SequenceResult.preds`` and ``SequenceResult.gts`` to be
            populated — set automatically by
            :class:`~eovot.benchmark.engine.BenchmarkEngine` (v0.2+).
            Sequences that lack stored predictions are silently skipped.
        """
        _require_matplotlib()
        fig, ax = plt.subplots(figsize=self.figsize)
        thresholds = np.linspace(0.0, 50.0, 51)

        for i, result in enumerate(results):
            per_seq_rates = []
            for sr in result.sequence_results:
                if sr.preds.size == 0 or sr.gts.size == 0:
                    continue
                _, rates = self._engine.precision_curve(sr.preds, sr.gts, thresholds)
                per_seq_rates.append(rates)

            if not per_seq_rates:
                continue  # tracker has no stored predictions — skip

            mean_rates = np.mean(per_seq_rates, axis=0)
            prec_at_20 = float(np.interp(20.0, thresholds, mean_rates))

            colour = _COLOURS[i % len(_COLOURS)]
            marker = _MARKERS[i % len(_MARKERS)]
            ax.plot(
                thresholds,
                mean_rates,
                color=colour,
                marker=marker,
                markevery=5,
                linewidth=1.8,
                markersize=5,
                label=f"{result.tracker_name} [P@20px={prec_at_20:.3f}]",
            )

        ax.axvline(x=20, color="gray", linestyle=":", linewidth=1.2, label="20 px")
        ax.set_xlabel("Location Error Threshold (px)", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 50.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        stem = filename or _safe_stem(title)
        out_path = self.output_dir / f"{stem}_precision.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_all(
        self,
        results: List[BenchmarkResult],
        title: str = "EOVOT Results",
        filename: Optional[str] = None,
    ) -> List[Path]:
        """Plot both success and precision curves and return their paths.

        Args:
            results: Benchmark results to compare (one per tracker).
            title: Base title; ``" \u2014 Success"`` and ``" \u2014 Precision"``
                are appended to the two plots respectively.
            filename: Override filename stem for both plots.

        Returns:
            ``[success_path, precision_path]``.
        """
        success_path = self.plot_success(
            results,
            title=f"{title} \u2014 Success",
            filename=filename,
        )
        prec_path = self.plot_precision(
            results,
            title=f"{title} \u2014 Precision",
            filename=filename,
        )
        return [success_path, prec_path]
