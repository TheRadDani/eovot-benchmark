"""Temporal drift analysis for visual object tracking evaluation.

Measures how a tracker's IoU degrades over the course of a sequence —
a dimension that per-sequence averages completely hide.  A tracker with
mean IoU 0.6 could maintain steady accuracy throughout the sequence, or
it could start at 0.9 and crash to 0.2 after the first few seconds.
For edge deployment these are radically different failure modes.

Key concepts
------------
**Temporal IoU curve**
    The sequence's per-frame IoU resampled to a fixed number of time bins
    by averaging within each bin.  Allows comparing trackers across
    sequences of different lengths on a common normalized time axis [0, 1].

**Drift rate**
    The slope of a linear fit to the temporal IoU curve (IoU/time unit).
    Negative → tracker degrades; positive → tracker improves (rare).
    Computed via ordinary least squares on the bin midpoints.

**IoU half-life**
    The normalized time at which the smoothed IoU curve first drops to
    half its initial value.  ``None`` when the tracker never degrades that
    far.  Useful for comparing long-term stability of edge-deployed trackers.

**Stability index**
    Fraction of non-burn-in frames with IoU ≥ ``stability_threshold``
    (default 0.5).  A single scalar summarising long-term reliability.

**Comparative drift table**
    :meth:`TemporalDriftAnalyzer.compare` accepts a dict mapping tracker
    names to per-sequence IoU arrays and returns a ranked Markdown table,
    making it easy to include in research papers or GitHub README files.

Typical usage::

    from eovot.metrics.temporal import TemporalDriftAnalyzer

    analyzer = TemporalDriftAnalyzer(n_bins=20)

    # Analyze one tracker on one sequence
    result = analyzer.analyze_sequence(ious, tracker_name="KCF", sequence_name="car1")
    print(result.drift_rate, result.stability_index)

    # Compare multiple trackers across a whole benchmark run
    tracker_ious = {
        "MOSSE": [seq_result.ious for seq_result in mosse_result.sequence_results],
        "KCF":   [seq_result.ious for seq_result in kcf_result.sequence_results],
    }
    table = analyzer.compare(tracker_ious)
    print(table)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TemporalDriftResult:
    """Temporal drift statistics for one tracker on one sequence.

    Attributes:
        tracker_name:      Human-readable tracker identifier.
        sequence_name:     Sequence identifier.
        n_frames:          Total number of frames in the sequence.
        time_bins:         ``(B,)`` array of normalized time bin centres in [0, 1].
        binned_iou:        ``(B,)`` array of mean IoU per bin.
        drift_rate:        Linear drift slope (IoU units per normalized time unit).
                           Negative means IoU degrades; positive means it improves.
        iou_half_life:     Normalized time at which smoothed IoU first falls to half
                           its initial value, or ``None`` when never reached.
        stability_index:   Fraction of frames (after burn-in) with IoU ≥ threshold.
        initial_iou:       Mean IoU over the first ``n_bins // 5`` bins.
        final_iou:         Mean IoU over the last ``n_bins // 5`` bins.
        iou_drop:          ``initial_iou - final_iou``; positive = degradation.
    """

    tracker_name: str
    sequence_name: str
    n_frames: int
    time_bins: np.ndarray
    binned_iou: np.ndarray
    drift_rate: float
    iou_half_life: Optional[float]
    stability_index: float
    initial_iou: float
    final_iou: float
    iou_drop: float

    def __str__(self) -> str:
        half_life = f"{self.iou_half_life:.3f}" if self.iou_half_life is not None else "N/A"
        return (
            f"TemporalDriftResult[{self.tracker_name} on {self.sequence_name}] "
            f"drift={self.drift_rate:+.4f}/t  "
            f"stability={self.stability_index:.3f}  "
            f"half_life={half_life}  "
            f"drop={self.iou_drop:+.4f}"
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (arrays converted to lists)."""
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "n_frames": self.n_frames,
            "drift_rate": round(self.drift_rate, 6),
            "iou_half_life": round(self.iou_half_life, 4) if self.iou_half_life is not None else None,
            "stability_index": round(self.stability_index, 4),
            "initial_iou": round(self.initial_iou, 4),
            "final_iou": round(self.final_iou, 4),
            "iou_drop": round(self.iou_drop, 4),
            "time_bins": self.time_bins.tolist(),
            "binned_iou": self.binned_iou.tolist(),
        }


@dataclass
class TrackerDriftSummary:
    """Aggregate temporal drift statistics for one tracker across all sequences.

    Attributes:
        tracker_name:         Human-readable tracker identifier.
        num_sequences:        Number of sequences analysed.
        mean_drift_rate:      Mean per-sequence drift rate (IoU/time).
        mean_stability_index: Mean fraction of frames with IoU ≥ threshold.
        mean_iou_drop:        Mean IoU drop from first to last segment.
        mean_half_life:       Mean IoU half-life (only sequences where it was reached).
        pct_degrading:        Percentage of sequences with negative drift rate.
        mean_binned_iou:      Mean temporal IoU curve across all sequences ``(B,)``.
    """

    tracker_name: str
    num_sequences: int
    mean_drift_rate: float
    mean_stability_index: float
    mean_iou_drop: float
    mean_half_life: Optional[float]
    pct_degrading: float
    mean_binned_iou: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        hl = f"{self.mean_half_life:.3f}" if self.mean_half_life is not None else "N/A"
        return (
            f"TrackerDriftSummary[{self.tracker_name}] "
            f"drift={self.mean_drift_rate:+.4f}/t  "
            f"stability={self.mean_stability_index:.3f}  "
            f"half_life={hl}  "
            f"drop={self.mean_iou_drop:+.4f}  "
            f"degrading={self.pct_degrading:.1f}%  "
            f"({self.num_sequences} sequences)"
        )


class TemporalDriftAnalyzer:
    """Analyze how tracker IoU evolves over normalized sequence time.

    Sequences are divided into equal-width time bins.  Per-frame IoU values
    are averaged within each bin to produce a smooth temporal IoU curve that
    can be compared across sequences of different lengths.

    Args:
        n_bins: Number of equal-width time bins.  Default: ``20`` (5% steps).
        stability_threshold: IoU threshold for stability-index computation.
            Default: ``0.5`` (standard half-success criterion).
        burn_in_frac: Fraction of the sequence skipped at the start before
            computing drift and stability metrics.  Default: ``0.05``
            (first 5% of frames, roughly matching the initialization window).
        min_frames: Sequences shorter than this are skipped in aggregate
            analysis.  Default: ``10``.

    Example::

        analyzer = TemporalDriftAnalyzer(n_bins=20)

        result = analyzer.analyze_sequence(
            ious=np.array([0.9, 0.85, 0.8, ...]),
            tracker_name="MOSSE",
            sequence_name="car1",
        )
        print(result.drift_rate)    # negative → degrading
        print(result.stability_index)

        # Aggregate across a full benchmark run
        all_seq_ious = {r.sequence_name: r.ious
                        for r in benchmark_result.sequence_results}
        summary = analyzer.analyze_tracker(all_seq_ious, tracker_name="MOSSE")
        print(summary.mean_drift_rate)
    """

    def __init__(
        self,
        n_bins: int = 20,
        stability_threshold: float = 0.5,
        burn_in_frac: float = 0.05,
        min_frames: int = 10,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if not 0.0 <= stability_threshold <= 1.0:
            raise ValueError(f"stability_threshold must be in [0, 1], got {stability_threshold}")
        if not 0.0 <= burn_in_frac < 1.0:
            raise ValueError(f"burn_in_frac must be in [0, 1), got {burn_in_frac}")
        self.n_bins = n_bins
        self.stability_threshold = stability_threshold
        self.burn_in_frac = burn_in_frac
        self.min_frames = min_frames

    # ------------------------------------------------------------------
    # Core per-sequence analysis
    # ------------------------------------------------------------------

    def analyze_sequence(
        self,
        ious: np.ndarray,
        tracker_name: str = "",
        sequence_name: str = "",
    ) -> TemporalDriftResult:
        """Compute temporal drift statistics for a single sequence.

        Args:
            ious:          Per-frame IoU array, shape ``(N,)``.
            tracker_name:  Stored in the result for identification.
            sequence_name: Stored in the result for identification.

        Returns:
            :class:`TemporalDriftResult` with all temporal statistics.
        """
        ious = np.asarray(ious, dtype=np.float64)
        n = len(ious)

        time_bins, binned_iou = self._bin_ious(ious)
        drift_rate = self._linear_drift_rate(time_bins, binned_iou)
        half_life = self._iou_half_life(time_bins, binned_iou)
        stability = self._stability_index(ious)

        # Initial / final IoU over the first and last 1/5 of bins
        seg = max(1, self.n_bins // 5)
        initial_iou = float(binned_iou[:seg].mean()) if len(binned_iou) >= seg else float(binned_iou.mean())
        final_iou = float(binned_iou[-seg:].mean()) if len(binned_iou) >= seg else float(binned_iou.mean())

        return TemporalDriftResult(
            tracker_name=tracker_name,
            sequence_name=sequence_name,
            n_frames=n,
            time_bins=time_bins,
            binned_iou=binned_iou,
            drift_rate=drift_rate,
            iou_half_life=half_life,
            stability_index=stability,
            initial_iou=initial_iou,
            final_iou=final_iou,
            iou_drop=initial_iou - final_iou,
        )

    def analyze_tracker(
        self,
        sequence_ious: Dict[str, np.ndarray],
        tracker_name: str = "",
    ) -> TrackerDriftSummary:
        """Aggregate temporal drift across all sequences in a benchmark run.

        Args:
            sequence_ious: ``{sequence_name: ious_array}`` mapping.
            tracker_name:  Stored in the result.

        Returns:
            :class:`TrackerDriftSummary` with mean statistics across sequences.
        """
        results: List[TemporalDriftResult] = []
        for seq_name, ious in sequence_ious.items():
            if len(ious) < self.min_frames:
                continue
            results.append(self.analyze_sequence(
                ious, tracker_name=tracker_name, sequence_name=seq_name,
            ))

        if not results:
            return TrackerDriftSummary(
                tracker_name=tracker_name,
                num_sequences=0,
                mean_drift_rate=0.0,
                mean_stability_index=0.0,
                mean_iou_drop=0.0,
                mean_half_life=None,
                pct_degrading=0.0,
                mean_binned_iou=np.zeros(self.n_bins),
            )

        drift_rates = np.array([r.drift_rate for r in results])
        stabilities = np.array([r.stability_index for r in results])
        drops = np.array([r.iou_drop for r in results])
        half_lives = [r.iou_half_life for r in results if r.iou_half_life is not None]
        mean_half_life = float(np.mean(half_lives)) if half_lives else None

        # Mean temporal curve: stack and average (all curves share the same bins)
        all_curves = np.stack([r.binned_iou for r in results], axis=0)
        mean_curve = all_curves.mean(axis=0)

        pct_degrading = float(100.0 * np.mean(drift_rates < 0))

        return TrackerDriftSummary(
            tracker_name=tracker_name,
            num_sequences=len(results),
            mean_drift_rate=float(drift_rates.mean()),
            mean_stability_index=float(stabilities.mean()),
            mean_iou_drop=float(drops.mean()),
            mean_half_life=mean_half_life,
            pct_degrading=pct_degrading,
            mean_binned_iou=mean_curve,
        )

    def compare(
        self,
        tracker_seq_ious: Dict[str, List[np.ndarray]],
    ) -> str:
        """Build a ranked Markdown table comparing drift statistics across trackers.

        Args:
            tracker_seq_ious: ``{tracker_name: [ious_seq_0, ious_seq_1, ...]}``
                where each value is a list of per-frame IoU arrays (one per
                sequence), as produced by :attr:`~eovot.benchmark.engine.BenchmarkResult.sequence_results`.

        Returns:
            Multi-line Markdown table string sorted by mean drift rate
            (least degrading first), suitable for direct inclusion in papers
            or READMEs.

        Example::

            table = analyzer.compare({
                "MOSSE": [r.ious for r in mosse_result.sequence_results],
                "KCF":   [r.ious for r in kcf_result.sequence_results],
            })
            print(table)
        """
        summaries: List[TrackerDriftSummary] = []
        for tracker_name, seq_ious_list in tracker_seq_ious.items():
            seq_dict = {f"seq_{i}": arr for i, arr in enumerate(seq_ious_list)}
            summary = self.analyze_tracker(seq_dict, tracker_name=tracker_name)
            summaries.append(summary)

        # Sort by mean_drift_rate descending (least degrading = largest = first)
        summaries.sort(key=lambda s: s.mean_drift_rate, reverse=True)

        lines = [
            "## Temporal Drift Analysis\n",
            "| Rank | Tracker | Drift Rate | Stability | IoU Drop | Half-Life | Degrading% | Sequences |",
            "|------|---------|:----------:|:---------:|:--------:|:---------:|:----------:|:---------:|",
        ]
        for rank, s in enumerate(summaries, start=1):
            hl = f"{s.mean_half_life:.3f}" if s.mean_half_life is not None else "—"
            lines.append(
                f"| {rank} | {s.tracker_name} "
                f"| {s.mean_drift_rate:+.4f} "
                f"| {s.mean_stability_index:.3f} "
                f"| {s.mean_iou_drop:+.4f} "
                f"| {hl} "
                f"| {s.pct_degrading:.1f}% "
                f"| {s.num_sequences} |"
            )
        lines.append("\n_Drift Rate: IoU change per normalized time unit. Negative = degrading._")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bin_ious(self, ious: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resample IoU array into ``n_bins`` equal-width temporal bins.

        Each bin averages the IoU values whose frame index falls within it.
        Returns ``(bin_centres, binned_means)`` both shape ``(n_bins,)``.

        Short sequences (< n_bins frames) use fewer bins equal to len(ious).
        """
        n = len(ious)
        if n == 0:
            bins = np.linspace(0.5 / self.n_bins, 1 - 0.5 / self.n_bins, self.n_bins)
            return bins, np.zeros(self.n_bins)

        effective_bins = min(self.n_bins, n)
        edges = np.linspace(0, n, effective_bins + 1)
        binned = np.zeros(effective_bins)
        for b in range(effective_bins):
            lo = int(math.floor(edges[b]))
            hi = int(math.ceil(edges[b + 1]))
            hi = min(hi, n)
            if hi > lo:
                binned[b] = float(ious[lo:hi].mean())

        # Bin centres on [0, 1]
        centres = (edges[:-1] + edges[1:]) / 2.0 / n

        # Pad to n_bins if sequence was short
        if effective_bins < self.n_bins:
            pad = self.n_bins - effective_bins
            binned = np.concatenate([binned, np.full(pad, binned[-1])])
            extra = np.linspace(centres[-1], 1.0, pad + 2)[1:-1]
            centres = np.concatenate([centres, extra])

        return centres, binned

    def _linear_drift_rate(self, time_bins: np.ndarray, binned_iou: np.ndarray) -> float:
        """Fit a line to the temporal IoU curve and return its slope.

        Slope is in units of IoU per normalized time step.  A slope of
        -0.3 means IoU is expected to drop by 0.3 over the full sequence.
        """
        if len(time_bins) < 2:
            return 0.0
        # Ordinary least squares: slope = cov(t, iou) / var(t)
        t = time_bins - time_bins.mean()
        y = binned_iou - binned_iou.mean()
        var_t = float(np.dot(t, t))
        if var_t < 1e-12:
            return 0.0
        return float(np.dot(t, y) / var_t)

    def _iou_half_life(
        self, time_bins: np.ndarray, binned_iou: np.ndarray
    ) -> Optional[float]:
        """Return normalized time when IoU first drops to half its initial value.

        The initial value is taken from the first non-burn-in bin.
        Returns ``None`` if the IoU never reaches the half-value threshold.
        """
        if len(binned_iou) < 2:
            return None
        burn_bins = max(1, int(round(self.burn_in_frac * len(binned_iou))))
        if burn_bins >= len(binned_iou):
            return None
        initial = float(binned_iou[burn_bins])
        if initial <= 0:
            return None
        target = initial / 2.0
        for i in range(burn_bins + 1, len(binned_iou)):
            if binned_iou[i] <= target:
                return float(time_bins[i])
        return None

    def _stability_index(self, ious: np.ndarray) -> float:
        """Fraction of post-burn-in frames with IoU ≥ stability_threshold."""
        n = len(ious)
        if n == 0:
            return 0.0
        burn = max(0, int(math.ceil(n * self.burn_in_frac)))
        post = ious[burn:]
        if len(post) == 0:
            return 0.0
        return float(np.mean(post >= self.stability_threshold))
