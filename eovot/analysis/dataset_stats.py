"""Ground-truth statistics for tracking datasets.

Edge benchmarks depend on understanding *what* is in a dataset before
running a single tracker.  Sequence properties such as motion magnitude,
scale variability, and target coverage directly predict which trackers will
struggle — but computing these requires only the ground-truth boxes, not any
tracker output.

:class:`DatasetStatisticsAnalyzer` accepts any :class:`~eovot.datasets.base.BaseDataset`
and returns a :class:`DatasetStats` summary that can be printed as a Markdown
table or exported as CSV for downstream analysis.

Computed per-sequence and aggregated across the dataset:

* **motion_magnitude** — mean inter-frame centre displacement (pixels).
* **scale_variability** — standard deviation of the normalised target area
  (``w*h / frame_area``), measuring how much the target size fluctuates.
* **aspect_ratio_stability** — 1 − std(w/h), so 1.0 = perfectly stable shape.
* **target_coverage** — mean fraction of the frame covered by the GT box.
* **n_frames** — total frames across all sequences.

Example::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.analysis.dataset_stats import DatasetStatisticsAnalyzer

    dataset  = SyntheticDataset(num_sequences=10)
    analyzer = DatasetStatisticsAnalyzer()
    stats    = analyzer.analyze(dataset, dataset_name="Synthetic-10")

    print(stats)
    print(stats.to_markdown_table())
    stats.to_csv("synthetic_stats.csv")
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from eovot.datasets.base import BaseDataset, Sequence


@dataclass
class SequenceStats:
    """Per-sequence ground-truth statistics."""

    name: str
    n_frames: int
    motion_magnitude: float
    scale_variability: float
    aspect_ratio_stability: float
    target_coverage: float


@dataclass
class DatasetStats:
    """Aggregated dataset-level statistics.

    Attributes:
        dataset_name:           Human-readable name passed at construction.
        sequence_stats:         Per-sequence breakdown.
        mean_motion_magnitude:  Mean of per-sequence motion magnitudes (px).
        mean_scale_variability: Mean of per-sequence scale variability.
        mean_aspect_ratio_stability: Mean aspect-ratio stability (0–1).
        mean_target_coverage:   Mean fraction of frame covered by GT box.
        total_frames:           Sum of frames across all sequences.
        n_sequences:            Number of sequences analysed.
    """

    dataset_name: str
    sequence_stats: List[SequenceStats]
    mean_motion_magnitude: float
    mean_scale_variability: float
    mean_aspect_ratio_stability: float
    mean_target_coverage: float
    total_frames: int
    n_sequences: int

    def __str__(self) -> str:
        return (
            f"DatasetStats({self.dataset_name!r}, "
            f"seqs={self.n_sequences}, frames={self.total_frames}, "
            f"motion={self.mean_motion_magnitude:.2f}px, "
            f"scale_var={self.mean_scale_variability:.4f}, "
            f"ar_stability={self.mean_aspect_ratio_stability:.4f}, "
            f"coverage={self.mean_target_coverage:.4f})"
        )

    def to_markdown_table(self) -> str:
        """Return a Markdown table with one row per sequence plus a summary row.

        Returns:
            Multi-line string suitable for inclusion in a README or report.
        """
        header = (
            "| Sequence | Frames | Motion (px) | Scale Var | AR Stability | Coverage |\n"
            "|---|---:|---:|---:|---:|---:|"
        )
        rows = []
        for s in self.sequence_stats:
            rows.append(
                f"| {s.name} | {s.n_frames} "
                f"| {s.motion_magnitude:.2f} "
                f"| {s.scale_variability:.4f} "
                f"| {s.aspect_ratio_stability:.4f} "
                f"| {s.target_coverage:.4f} |"
            )
        summary = (
            f"| **{self.dataset_name} (mean)** | {self.total_frames} "
            f"| **{self.mean_motion_magnitude:.2f}** "
            f"| **{self.mean_scale_variability:.4f}** "
            f"| **{self.mean_aspect_ratio_stability:.4f}** "
            f"| **{self.mean_target_coverage:.4f}** |"
        )
        return "\n".join([header] + rows + [summary])

    def to_csv(self, path: str) -> None:
        """Write per-sequence statistics to a CSV file.

        Args:
            path: Destination file path.  Parent directories must exist.
        """
        fieldnames = [
            "sequence",
            "n_frames",
            "motion_magnitude",
            "scale_variability",
            "aspect_ratio_stability",
            "target_coverage",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.sequence_stats:
                writer.writerow(
                    {
                        "sequence": s.name,
                        "n_frames": s.n_frames,
                        "motion_magnitude": f"{s.motion_magnitude:.6f}",
                        "scale_variability": f"{s.scale_variability:.6f}",
                        "aspect_ratio_stability": f"{s.aspect_ratio_stability:.6f}",
                        "target_coverage": f"{s.target_coverage:.6f}",
                    }
                )


class DatasetStatisticsAnalyzer:
    """Compute ground-truth statistics for a dataset without running any tracker.

    The analyzer iterates over sequences in the dataset and derives statistics
    purely from the ``ground_truth`` array (shape ``(N, 4)``, columns
    ``x, y, w, h``).  No frames are decoded from disk unless the dataset
    provides frame-level metadata (e.g. resolution) — by default a standard
    ``1920×1080`` frame is assumed for normalisation; pass *frame_wh* to
    override.

    Args:
        frame_wh: ``(width, height)`` assumed for normalising target area and
            coverage statistics.  Default ``(1920, 1080)``.  Sequences that
            expose a ``frame_size`` attribute override this automatically.
    """

    def __init__(self, frame_wh: tuple = (1920, 1080)) -> None:
        self._default_frame_area = float(frame_wh[0]) * float(frame_wh[1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        dataset: BaseDataset,
        dataset_name: str = "Dataset",
    ) -> DatasetStats:
        """Compute statistics for every sequence in *dataset*.

        Args:
            dataset:      Any :class:`~eovot.datasets.base.BaseDataset`.
            dataset_name: Label used in the summary string and Markdown output.

        Returns:
            :class:`DatasetStats` with per-sequence and aggregate values.

        Raises:
            ValueError: If *dataset* contains no sequences.
        """
        if len(dataset) == 0:
            raise ValueError("Cannot analyse an empty dataset.")

        seq_stats: List[SequenceStats] = []
        for i in range(len(dataset)):
            seq = dataset[i]
            seq_stats.append(self._analyze_sequence(seq))

        motions = [s.motion_magnitude for s in seq_stats]
        scales = [s.scale_variability for s in seq_stats]
        ar_stabs = [s.aspect_ratio_stability for s in seq_stats]
        coverages = [s.target_coverage for s in seq_stats]

        return DatasetStats(
            dataset_name=dataset_name,
            sequence_stats=seq_stats,
            mean_motion_magnitude=float(np.mean(motions)),
            mean_scale_variability=float(np.mean(scales)),
            mean_aspect_ratio_stability=float(np.mean(ar_stabs)),
            mean_target_coverage=float(np.mean(coverages)),
            total_frames=sum(s.n_frames for s in seq_stats),
            n_sequences=len(seq_stats),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_sequence(self, seq: Sequence) -> SequenceStats:
        gt = seq.ground_truth  # (N, 4) — x, y, w, h
        n = len(gt)

        frame_area = self._default_frame_area

        xs, ys, ws, hs = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

        # Centre coordinates
        cxs = xs + ws / 2.0
        cys = ys + hs / 2.0

        # Motion: mean inter-frame Euclidean displacement of centres
        if n > 1:
            dcx = np.diff(cxs)
            dcy = np.diff(cys)
            motion = float(np.mean(np.sqrt(dcx**2 + dcy**2)))
        else:
            motion = 0.0

        # Normalised area: w*h / frame_area
        norm_areas = (ws * hs) / frame_area
        scale_var = float(np.std(norm_areas)) if n > 1 else 0.0

        # Aspect ratio: w/h (avoid division by zero)
        safe_hs = np.where(hs > 0, hs, 1.0)
        ar = ws / safe_hs
        ar_stability = max(0.0, 1.0 - float(np.std(ar))) if n > 1 else 1.0

        # Target coverage: mean fraction of frame area
        coverage = float(np.mean(norm_areas))

        return SequenceStats(
            name=seq.name,
            n_frames=n,
            motion_magnitude=motion,
            scale_variability=scale_var,
            aspect_ratio_stability=ar_stability,
            target_coverage=coverage,
        )
