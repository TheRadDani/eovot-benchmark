"""Per-frame IoU distribution analysis for EOVOT benchmark results.

Standard VOT benchmarks report the **mean** IoU — a single scalar that collapses
the entire tracking quality distribution into one number.  This hides a critical
distinction between trackers:

- **Consistent tracker**: IoU hovers near a stable value throughout the sequence.
  Mean IoU = 0.55 may represent frames that are all in [0.45, 0.65].
- **Oscillating tracker**: IoU alternates between near-perfect tracking and
  near-total failure.  Mean IoU = 0.55 may represent a mix of IoU ≈ 1.0 and
  IoU ≈ 0.1 frames.

For edge deployment, the oscillating tracker is far more problematic even with
the same mean score: downstream consumers (robot controllers, video analytics)
cannot tolerate unpredictable localization quality.

This module provides distribution-level diagnostics:

Bimodality Coefficient (BC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sarle's (1990) bimodality coefficient measures how much a distribution deviates
from unimodal:

    BC = (γ₁² + 1) / (γ₂ + 3)

where γ₁ is the sample skewness and γ₂ is the excess kurtosis.

Interpretation:
    - BC ≤ 5/9 ≈ 0.555: unimodal distribution (consistent tracker).
    - BC > 5/9: bimodal or multimodal distribution (oscillating tracker).
    - BC = 1.0: maximum bimodality (Bernoulli-like: only two distinct values).

Failure Ratio
~~~~~~~~~~~~~
Fraction of frames with IoU < 0.1 — the standard VOT failure threshold.
A failure ratio above 0.10 indicates persistent re-detection issues.

Excellent Ratio
~~~~~~~~~~~~~~~
Fraction of frames with IoU > 0.5 — the threshold used in success-curve AUC.
A tracker needs excellent_ratio > 0.6 to achieve a respectable success AUC.

Quartiles
~~~~~~~~~
Q1, Q2 (median), Q3 of the IoU distribution.  A low Q1 with high Q3 indicates
high variance even when the median appears acceptable.

Histogram
~~~~~~~~~
Raw frequency counts across 10 bins in [0, 1], enabling downstream plotting
of per-tracker IoU distributions without re-running the benchmark.

Typical usage::

    from eovot.analysis.iou_distribution import IoUDistributionAnalyzer

    analyzer = IoUDistributionAnalyzer()

    # Per-sequence analysis
    result = analyzer.analyze_sequence(ious, "KCF", "car1")
    print(result)
    # IoUDist[KCF on car1] mean=0.612 BC=0.421 fail=0.031 excel=0.734

    # Aggregate over a full benchmark run
    seq_ious = {r.sequence_name: r.ious for r in benchmark_result.sequence_results}
    agg = analyzer.analyze_benchmark(seq_ious, tracker_name="KCF")
    print(agg["aggregate"]["mean_bimodality_coefficient"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Sarle's threshold: distributions with BC > 5/9 are considered bimodal.
BIMODALITY_THRESHOLD: float = 5.0 / 9.0  # ≈ 0.5556


@dataclass
class IoUDistributionResult:
    """Per-sequence IoU distribution summary.

    Attributes:
        tracker_name: Human-readable tracker identifier.
        sequence_name: Sequence identifier.
        num_frames: Total frames analysed.
        mean_iou: Arithmetic mean of per-frame IoU values.
        median_iou: Median (Q2) of per-frame IoU values.
        q1_iou: First quartile (25th percentile) of per-frame IoU.
        q3_iou: Third quartile (75th percentile) of per-frame IoU.
        std_iou: Standard deviation of per-frame IoU values.
        failure_ratio: Fraction of frames with IoU < ``failure_threshold``.
        excellent_ratio: Fraction of frames with IoU > ``excellent_threshold``.
        bimodality_coefficient: Sarle's BC in ``(0, 1]``.  Values above
            ``BIMODALITY_THRESHOLD`` (≈ 0.556) indicate an oscillating tracker.
        is_bimodal: ``True`` when BC exceeds ``BIMODALITY_THRESHOLD``.
        histogram_counts: Frequency counts across 10 equal-width bins in [0, 1].
        histogram_edges: Bin edges (length 11) matching ``histogram_counts``.
    """

    tracker_name: str
    sequence_name: str
    num_frames: int
    mean_iou: float
    median_iou: float
    q1_iou: float
    q3_iou: float
    std_iou: float
    failure_ratio: float
    excellent_ratio: float
    bimodality_coefficient: float
    is_bimodal: bool
    histogram_counts: List[int] = field(default_factory=list)
    histogram_edges: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        bimodal_tag = " [BIMODAL]" if self.is_bimodal else ""
        return (
            f"IoUDist[{self.tracker_name} on {self.sequence_name}]{bimodal_tag} "
            f"mean={self.mean_iou:.3f}  median={self.median_iou:.3f}  "
            f"BC={self.bimodality_coefficient:.3f}  "
            f"fail={self.failure_ratio:.3f}  excel={self.excellent_ratio:.3f}  "
            f"IQR=[{self.q1_iou:.3f}, {self.q3_iou:.3f}]"
        )

    def to_dict(self) -> Dict:
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "mean_iou": round(self.mean_iou, 4),
            "median_iou": round(self.median_iou, 4),
            "q1_iou": round(self.q1_iou, 4),
            "q3_iou": round(self.q3_iou, 4),
            "std_iou": round(self.std_iou, 4),
            "failure_ratio": round(self.failure_ratio, 4),
            "excellent_ratio": round(self.excellent_ratio, 4),
            "bimodality_coefficient": round(self.bimodality_coefficient, 4),
            "is_bimodal": self.is_bimodal,
            "histogram_counts": self.histogram_counts,
            "histogram_edges": [round(e, 3) for e in self.histogram_edges],
        }


class IoUDistributionAnalyzer:
    """Analyse the full distribution of per-frame IoU values for a tracker.

    Args:
        failure_threshold: IoU below which a frame counts as a tracking failure.
            Default: ``0.1`` (standard VOT threshold).
        excellent_threshold: IoU above which a frame counts as excellent tracking.
            Default: ``0.5`` (the standard success-curve threshold).
        n_bins: Number of histogram bins across ``[0, 1]``.  Default: ``10``.

    Example::

        analyzer = IoUDistributionAnalyzer()

        # From a BenchmarkResult
        for seq_result in benchmark_result.sequence_results:
            result = analyzer.analyze_sequence(
                seq_result.ious,
                tracker_name=benchmark_result.tracker_name,
                sequence_name=seq_result.sequence_name,
            )
            if result.is_bimodal:
                print(f"WARNING: oscillating tracker on {result.sequence_name}")
    """

    def __init__(
        self,
        failure_threshold: float = 0.1,
        excellent_threshold: float = 0.5,
        n_bins: int = 10,
    ) -> None:
        if not (0.0 <= failure_threshold < excellent_threshold <= 1.0):
            raise ValueError(
                f"Require 0 <= failure_threshold ({failure_threshold}) "
                f"< excellent_threshold ({excellent_threshold}) <= 1"
            )
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.failure_threshold = failure_threshold
        self.excellent_threshold = excellent_threshold
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Core metric computations
    # ------------------------------------------------------------------

    def bimodality_coefficient(self, ious: np.ndarray) -> float:
        """Compute Sarle's bimodality coefficient for an IoU array.

        Uses the sample (unbiased) central moments.  Returns 0.0 for
        degenerate inputs (< 3 values or zero variance).

        Args:
            ious: 1-D float array of IoU values in ``[0, 1]``.

        Returns:
            BC in ``(0, 1]``.  Higher values indicate more bimodal distributions.
        """
        n = len(ious)
        if n < 3:
            return 0.0

        mean = ious.mean()
        deviations = ious - mean
        m2 = float(np.mean(deviations ** 2))
        if m2 < 1e-12:
            return 0.0  # all values identical -> perfectly unimodal

        m3 = float(np.mean(deviations ** 3))
        m4 = float(np.mean(deviations ** 4))

        skewness = m3 / (m2 ** 1.5)
        # Excess kurtosis (Fisher's definition: 0 for a normal distribution)
        excess_kurtosis = m4 / (m2 ** 2) - 3.0

        # Denominator is the non-excess kurtosis = excess_kurtosis + 3.
        # By Pearson's inequality, non-excess kurtosis >= 1 for any distribution,
        # so the denominator is always >= 1 and the division is always safe.
        bc = (skewness ** 2 + 1.0) / (excess_kurtosis + 3.0)
        return float(bc)

    def compute_histogram(
        self, ious: np.ndarray
    ) -> Tuple[List[int], List[float]]:
        """Compute a frequency histogram of IoU values.

        Args:
            ious: 1-D float array of IoU values.

        Returns:
            ``(counts, edges)`` where ``counts`` has ``n_bins`` elements and
            ``edges`` has ``n_bins + 1`` elements (the bin boundaries).
        """
        if len(ious) == 0:
            edges = list(np.linspace(0.0, 1.0, self.n_bins + 1))
            return [0] * self.n_bins, edges
        counts_arr, edges_arr = np.histogram(ious, bins=self.n_bins, range=(0.0, 1.0))
        return counts_arr.tolist(), edges_arr.tolist()

    # ------------------------------------------------------------------
    # High-level analysis entry points
    # ------------------------------------------------------------------

    def analyze_sequence(
        self,
        ious: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> IoUDistributionResult:
        """Run full distribution analysis on a single sequence.

        Args:
            ious:          Per-frame IoU array, shape ``(N,)``.
            tracker_name:  Identifier stored in the result.
            sequence_name: Identifier stored in the result.

        Returns:
            :class:`IoUDistributionResult` with all statistics populated.
        """
        ious = np.asarray(ious, dtype=np.float64).ravel()
        n = len(ious)

        if n == 0:
            counts, edges = self.compute_histogram(ious)
            return IoUDistributionResult(
                tracker_name=tracker_name,
                sequence_name=sequence_name,
                num_frames=0,
                mean_iou=0.0,
                median_iou=0.0,
                q1_iou=0.0,
                q3_iou=0.0,
                std_iou=0.0,
                failure_ratio=0.0,
                excellent_ratio=0.0,
                bimodality_coefficient=0.0,
                is_bimodal=False,
                histogram_counts=counts,
                histogram_edges=edges,
            )

        q1 = float(np.percentile(ious, 25))
        median = float(np.median(ious))
        q3 = float(np.percentile(ious, 75))
        failure_ratio = float((ious < self.failure_threshold).mean())
        excellent_ratio = float((ious > self.excellent_threshold).mean())
        bc = self.bimodality_coefficient(ious)
        counts, edges = self.compute_histogram(ious)

        return IoUDistributionResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            num_frames=n,
            mean_iou=float(ious.mean()),
            median_iou=median,
            q1_iou=q1,
            q3_iou=q3,
            std_iou=float(ious.std()),
            failure_ratio=failure_ratio,
            excellent_ratio=excellent_ratio,
            bimodality_coefficient=bc,
            is_bimodal=bc > BIMODALITY_THRESHOLD,
            histogram_counts=counts,
            histogram_edges=edges,
        )

    def analyze_benchmark(
        self,
        sequence_ious: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> Dict:
        """Aggregate IoU distribution analysis across all sequences in a run.

        Args:
            sequence_ious: Mapping ``{sequence_name: ious_array}``.
                Build from a benchmark result with::

                    {r.sequence_name: r.ious for r in result.sequence_results}

            tracker_name: Identifier for the tracker under evaluation.

        Returns:
            Dict with two keys:

            * ``"per_sequence"`` — ``{seq_name: IoUDistributionResult}``
            * ``"aggregate"`` — scalar summaries across all sequences, including
              ``"mean_bimodality_coefficient"``, ``"bimodal_sequence_fraction"``,
              ``"mean_failure_ratio"``, and ``"mean_excellent_ratio"``.
        """
        per_seq: Dict[str, IoUDistributionResult] = {}
        for seq_name, ious in sequence_ious.items():
            per_seq[seq_name] = self.analyze_sequence(
                ious, tracker_name=tracker_name, sequence_name=seq_name
            )

        if not per_seq:
            return {"per_sequence": {}, "aggregate": {}}

        results = list(per_seq.values())
        n = len(results)
        n_bimodal = sum(1 for r in results if r.is_bimodal)

        aggregate = {
            "tracker_name": tracker_name,
            "num_sequences": n,
            "mean_iou": round(float(np.mean([r.mean_iou for r in results])), 4),
            "mean_median_iou": round(float(np.mean([r.median_iou for r in results])), 4),
            "mean_std_iou": round(float(np.mean([r.std_iou for r in results])), 4),
            "mean_failure_ratio": round(float(np.mean([r.failure_ratio for r in results])), 4),
            "mean_excellent_ratio": round(float(np.mean([r.excellent_ratio for r in results])), 4),
            "mean_bimodality_coefficient": round(
                float(np.mean([r.bimodality_coefficient for r in results])), 4
            ),
            "bimodal_sequence_count": n_bimodal,
            "bimodal_sequence_fraction": round(n_bimodal / n, 4) if n > 0 else 0.0,
        }

        return {"per_sequence": per_seq, "aggregate": aggregate}

    def compare_trackers(
        self,
        tracker_benchmarks: Dict[str, Dict[str, np.ndarray]],
    ) -> List[Dict]:
        """Compare multiple trackers by their aggregate IoU distribution stats.

        Args:
            tracker_benchmarks: Mapping ``{tracker_name: {seq_name: ious}}``.
                Build from multiple :class:`~eovot.benchmark.engine.BenchmarkResult`
                objects::

                    {
                        r.tracker_name: {
                            seq.sequence_name: seq.ious
                            for seq in r.sequence_results
                        }
                        for r in results
                    }

        Returns:
            List of aggregate dicts (one per tracker), sorted by
            ``mean_bimodality_coefficient`` ascending (most consistent first).
        """
        rows = []
        for tracker_name, seq_ious in tracker_benchmarks.items():
            agg = self.analyze_benchmark(seq_ious, tracker_name=tracker_name)
            rows.append(agg["aggregate"])
        rows.sort(key=lambda x: x.get("mean_bimodality_coefficient", 1.0))
        return rows

    def to_markdown_table(self, rows: List[Dict]) -> str:
        """Format a comparison table as a Markdown string.

        Args:
            rows: Output of :meth:`compare_trackers`.

        Returns:
            Multi-line Markdown table string.
        """
        header = "| Tracker | mIoU | Median | Std | Fail% | Excel% | BC | Bimodal% |"
        sep = "|---------|-----:|-------:|----:|------:|-------:|---:|---------:|"
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"| {r.get('tracker_name', '?')} "
                f"| {r.get('mean_iou', 0):.4f} "
                f"| {r.get('mean_median_iou', 0):.4f} "
                f"| {r.get('mean_std_iou', 0):.4f} "
                f"| {r.get('mean_failure_ratio', 0) * 100:.1f} "
                f"| {r.get('mean_excellent_ratio', 0) * 100:.1f} "
                f"| {r.get('mean_bimodality_coefficient', 0):.3f} "
                f"| {r.get('bimodal_sequence_fraction', 0) * 100:.0f} |"
            )
        return "\n".join(lines)
