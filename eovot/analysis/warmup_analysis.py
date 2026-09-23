"""Tracker warm-up and cold-start analysis for edge deployment.

Edge trackers often exhibit a two-phase latency profile:

1. **Cold start** — the first N frames are slower because CPU caches are
   cold, JIT paths warm up, and internal ring buffers fill.
2. **Steady state** — latency stabilises to its long-run per-frame value.

For real-time edge deployment this distinction matters: a system that
achieves 60 FPS steady-state but takes 500 ms to reach it will miss its
first few frames.  This module quantifies that transition.

Typical usage::

    from eovot.analysis.warmup_analysis import WarmupAnalyzer
    from eovot.trackers.registry import build_tracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset  = SyntheticDataset(num_sequences=3, num_frames=200)
    tracker  = build_tracker("KCF")
    analyzer = WarmupAnalyzer()

    # Single-sequence characterisation
    result = analyzer.analyze_sequence(tracker, dataset[0])
    print(result)

    # Multi-sequence aggregate
    report = analyzer.analyze_dataset(tracker, dataset)
    print(report.to_markdown())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..trackers.base import BaseTracker
    from ..datasets.base import BaseDataset, Sequence


@dataclass
class WarmupResult:
    """Cold-start and warm-up characterisation for one tracker on one sequence."""

    tracker_name: str
    sequence_name: str
    frame_count: int

    init_ms: float
    """Wall-clock time for ``tracker.initialize()`` in milliseconds."""

    frame_latencies_ms: List[float] = field(default_factory=list)
    """Per-frame latency (ms) for update frames 1 … N-1 (frame 0 is the init call)."""

    warmup_end_frame: int = 0
    """Index of the first update frame whose EMA latency enters the stable band.
    0 means the tracker was already stable from the first update."""

    cold_start_fps: float = 0.0
    """Mean FPS over the cold phase (frames 0 … warmup_end_frame-1)."""

    steady_state_fps: float = 0.0
    """Mean FPS from warmup_end_frame onward (warm phase)."""

    fps_improvement_ratio: float = 1.0
    """``steady_state_fps / cold_start_fps``.  1.0 when the cold phase is empty."""

    total_warmup_ms: float = 0.0
    """Cumulative latency of the cold phase in milliseconds (excludes init_ms)."""

    def __str__(self) -> str:
        return (
            f"WarmupResult[{self.tracker_name} @ {self.sequence_name}]  "
            f"init={self.init_ms:.1f} ms  "
            f"warmup_frames={self.warmup_end_frame}  "
            f"warmup_wall={self.total_warmup_ms:.1f} ms  "
            f"cold_fps={self.cold_start_fps:.1f}  "
            f"steady_fps={self.steady_state_fps:.1f}  "
            f"speedup={self.fps_improvement_ratio:.2f}x"
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "frame_count": self.frame_count,
            "init_ms": self.init_ms,
            "warmup_end_frame": self.warmup_end_frame,
            "cold_start_fps": self.cold_start_fps,
            "steady_state_fps": self.steady_state_fps,
            "fps_improvement_ratio": self.fps_improvement_ratio,
            "total_warmup_ms": self.total_warmup_ms,
        }


class WarmupAnalyzer:
    """Measure and characterise tracker warm-up behaviour.

    The analyser:

    1. Pre-loads all frames into memory (to exclude disk I/O from timing).
    2. Times ``tracker.initialize()`` on the first frame.
    3. Times each subsequent ``tracker.update()`` call individually.
    4. Applies an exponential moving average (EMA) to the latency series.
    5. Finds the first frame at which the EMA enters a band of width
       ``±tolerance × steady_state_mean`` around the final-window mean,
       marking the end of the cold-start phase.

    Args:
        ema_alpha: EMA smoothing factor in ``(0, 1]``.  Higher values react
            faster but are noisier.  Default 0.1.
        tolerance: Band half-width relative to steady-state mean.  Default 0.10
            (±10 %).
        min_steady_frames: Minimum frames in the steady-state estimation window.
            Default 10.

    Example::

        analyzer = WarmupAnalyzer(ema_alpha=0.15, tolerance=0.05)
        result   = analyzer.analyze_sequence(tracker, sequence)
        print(result.warmup_end_frame, result.fps_improvement_ratio)
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        tolerance: float = 0.10,
        min_steady_frames: int = 10,
    ) -> None:
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1].")
        if not 0 < tolerance < 1:
            raise ValueError("tolerance must be in (0, 1).")
        if min_steady_frames < 1:
            raise ValueError("min_steady_frames must be >= 1.")
        self.ema_alpha = ema_alpha
        self.tolerance = tolerance
        self.min_steady_frames = min_steady_frames

    def analyze_sequence(
        self,
        tracker: "BaseTracker",
        sequence: "Sequence",
        max_frames: Optional[int] = None,
    ) -> WarmupResult:
        """Run *tracker* on *sequence* and return the warm-up characterisation.

        Frames are pre-loaded into memory before any timing begins so that
        disk I/O is not included in the latency measurements.

        Args:
            tracker:    Tracker instance.  ``initialize()`` is called
                        internally; pass a fresh or reset instance.
            sequence:   :class:`~eovot.datasets.base.Sequence` with at least
                        ``min_steady_frames + 2`` frames.
            max_frames: Cap on the number of frames evaluated.

        Returns:
            :class:`WarmupResult` with cold/warm FPS statistics.

        Raises:
            ValueError: If the sequence is too short for analysis.
        """
        all_frames = list(sequence)
        gt = sequence.ground_truth

        if max_frames is not None:
            all_frames = all_frames[:max_frames]
            gt = gt[:max_frames]

        n = len(all_frames)
        min_required = self.min_steady_frames + 2
        if n < min_required:
            raise ValueError(
                f"Sequence '{getattr(sequence, 'name', '?')}' has only {n} "
                f"frames; need at least {min_required} for warmup analysis."
            )

        # --- Initialisation timing ---
        init_bbox = tuple(int(v) for v in gt[0])
        t0 = time.perf_counter()
        tracker.initialize(all_frames[0], init_bbox)
        init_ms = (time.perf_counter() - t0) * 1_000.0

        # --- Per-frame update timing ---
        latencies: List[float] = []
        for frame in all_frames[1:]:
            t0 = time.perf_counter()
            tracker.update(frame)
            latencies.append((time.perf_counter() - t0) * 1_000.0)

        warmup_end, cold_fps, steady_fps, warmup_ms = self._detect_warmup(latencies)
        ratio = steady_fps / cold_fps if cold_fps > 0 else 1.0

        return WarmupResult(
            tracker_name=tracker.name,
            sequence_name=getattr(sequence, "name", "unknown"),
            frame_count=n,
            init_ms=init_ms,
            frame_latencies_ms=latencies,
            warmup_end_frame=warmup_end,
            cold_start_fps=cold_fps,
            steady_state_fps=steady_fps,
            fps_improvement_ratio=ratio,
            total_warmup_ms=warmup_ms,
        )

    def analyze_dataset(
        self,
        tracker: "BaseTracker",
        dataset: "BaseDataset",
        max_sequences: Optional[int] = None,
        max_frames_per_seq: Optional[int] = None,
    ) -> "WarmupReport":
        """Analyse warm-up behaviour across multiple sequences.

        Sequences that are too short for the configured ``min_steady_frames``
        are silently skipped.

        Args:
            tracker:             Tracker instance (re-initialised per sequence).
            dataset:             Dataset providing :class:`~eovot.datasets.base.Sequence`
                                 objects.
            max_sequences:       Cap on the number of sequences.
            max_frames_per_seq:  Cap on frames per sequence.

        Returns:
            :class:`WarmupReport` with per-sequence results and aggregate stats.
        """
        n_seq = (
            len(dataset)
            if max_sequences is None
            else min(max_sequences, len(dataset))
        )
        results: List[WarmupResult] = []
        for i in range(n_seq):
            try:
                r = self.analyze_sequence(
                    tracker, dataset[i], max_frames=max_frames_per_seq
                )
                results.append(r)
            except (ValueError, RuntimeError):
                continue

        return WarmupReport(results=results)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _detect_warmup(self, latencies: List[float]) -> tuple:
        """Detect the warmup boundary and compute phase statistics.

        Returns:
            ``(warmup_end_frame, cold_fps, steady_fps, warmup_wall_ms)``
        """
        arr = np.array(latencies, dtype=float)
        n = len(arr)

        window = max(self.min_steady_frames, n // 4)
        steady_mean = float(arr[-window:].mean())

        ema = np.empty(n)
        ema[0] = arr[0]
        for i in range(1, n):
            ema[i] = self.ema_alpha * arr[i] + (1.0 - self.ema_alpha) * ema[i - 1]

        lo = steady_mean * (1.0 - self.tolerance)
        hi = steady_mean * (1.0 + self.tolerance)

        warmup_end = n
        search_limit = max(0, n - self.min_steady_frames)
        for i in range(search_limit):
            if lo <= ema[i] <= hi:
                warmup_end = i
                break

        if warmup_end > 0:
            cold_lat = float(arr[:warmup_end].mean())
            cold_fps = 1_000.0 / cold_lat if cold_lat > 0 else 0.0
            warmup_wall = float(arr[:warmup_end].sum())
        else:
            cold_fps = 1_000.0 / float(arr[0]) if arr[0] > 0 else 0.0
            warmup_wall = float(arr[0])

        if warmup_end < n:
            steady_lat = float(arr[warmup_end:].mean())
        else:
            steady_lat = steady_mean
        steady_fps = 1_000.0 / steady_lat if steady_lat > 0 else 0.0

        return warmup_end, cold_fps, steady_fps, warmup_wall


@dataclass
class WarmupReport:
    """Aggregate warm-up results across multiple sequences."""

    results: List[WarmupResult]

    @property
    def tracker_name(self) -> str:
        return self.results[0].tracker_name if self.results else "unknown"

    @property
    def mean_init_ms(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.init_ms for r in self.results]))

    @property
    def mean_warmup_frames(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.warmup_end_frame for r in self.results]))

    @property
    def mean_steady_state_fps(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.steady_state_fps for r in self.results]))

    @property
    def mean_fps_improvement(self) -> float:
        if not self.results:
            return 1.0
        return float(np.mean([r.fps_improvement_ratio for r in self.results]))

    def to_markdown(self) -> str:
        """Render a Markdown table of per-sequence results plus a mean row."""
        lines = [
            f"## Warm-up Analysis: {self.tracker_name}",
            "",
            "| Sequence | Init (ms) | Warmup Frames | Cold FPS | Steady FPS | Speedup |",
            "|----------|-----------|---------------|----------|------------|---------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.sequence_name} "
                f"| {r.init_ms:.1f} "
                f"| {r.warmup_end_frame} "
                f"| {r.cold_start_fps:.1f} "
                f"| {r.steady_state_fps:.1f} "
                f"| {r.fps_improvement_ratio:.2f}x |"
            )
        lines += [
            "|----------|-----------|---------------|----------|------------|---------|",
            f"| **Mean** "
            f"| **{self.mean_init_ms:.1f}** "
            f"| **{self.mean_warmup_frames:.1f}** "
            f"| — "
            f"| **{self.mean_steady_state_fps:.1f}** "
            f"| **{self.mean_fps_improvement:.2f}x** |",
        ]
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        """Return a JSON-serialisable summary dict."""
        return {
            "tracker_name": self.tracker_name,
            "num_sequences": len(self.results),
            "mean_init_ms": self.mean_init_ms,
            "mean_warmup_frames": self.mean_warmup_frames,
            "mean_steady_state_fps": self.mean_steady_state_fps,
            "mean_fps_improvement_ratio": self.mean_fps_improvement,
            "sequences": [r.to_dict() for r in self.results],
        }
