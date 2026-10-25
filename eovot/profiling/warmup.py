"""Cold-start latency profiler for edge deployment analysis.

Edge devices often face a **cold-start problem**: the first inference call
(and the subsequent warmup period) is substantially slower than steady-state
because of:

- JIT compilation (PyTorch, TensorFlow)
- OS page faults on first memory access
- Cache warming (L1/L2 for correlation filters storing the FFT of prior frames)
- OpenCV lazy-initialisation of algorithm state

For real-time tracking on constrained hardware this matters:
a robot arm starting a grasp sequence or a drone initiating object-following
pays a one-time latency spike that can exceed the control loop's deadline.

This module provides :class:`ColdStartProfiler` to measure and decompose
cold-start cost and :class:`ColdStartResult` to report the breakdown.

Typical usage::

    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.profiling.warmup import ColdStartProfiler

    dataset = SyntheticDataset(num_sequences=1, num_frames=120)
    seq = dataset[0]

    profiler = ColdStartProfiler(warmup_window=20)
    result = profiler.profile(MOSSETracker(), seq, max_frames=100)
    print(result)
    # ColdStartResult[MOSSE]  init=0.82ms  warmup=12fr  warmup_mean=1.43ms
    #   steady_state=0.94ms  startup_overhead=6.74ms
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..trackers.base import BaseTracker
from ..datasets.base import Sequence


@dataclass
class ColdStartResult:
    """Cold-start latency breakdown for a single tracker on a single sequence.

    Attributes:
        tracker_name: Identifier of the profiled tracker.
        sequence_name: Name of the sequence used for profiling.
        init_latency_ms: Wall-clock time (ms) to execute ``tracker.initialize()``.
        per_frame_latencies_ms: Raw per-frame latency list (ms) for the update
            calls that followed initialization, up to ``max_frames``.
        warmup_frames: Index of the first frame where latency is considered
            stable (steady-state begins).  Zero means no warmup period was
            detected.
        warmup_mean_ms: Mean per-frame latency during the warmup window
            (frames 0 … ``warmup_frames - 1``).  Zero if ``warmup_frames == 0``.
        steady_state_ms: Mean per-frame latency after the warmup window
            (frames ``warmup_frames`` … end).
        startup_overhead_ms: Total extra latency cost from init + warmup
            relative to running at steady state::

                overhead = init_ms + warmup_frames * max(0, warmup_mean - steady_ms)
    """

    tracker_name: str
    sequence_name: str
    init_latency_ms: float
    per_frame_latencies_ms: List[float] = field(default_factory=list)
    warmup_frames: int = 0
    warmup_mean_ms: float = 0.0
    steady_state_ms: float = 0.0
    startup_overhead_ms: float = 0.0

    def __str__(self) -> str:
        return (
            f"ColdStartResult[{self.tracker_name}]  "
            f"init={self.init_latency_ms:.2f}ms  "
            f"warmup={self.warmup_frames}fr  "
            f"warmup_mean={self.warmup_mean_ms:.2f}ms  "
            f"steady_state={self.steady_state_ms:.2f}ms  "
            f"startup_overhead={self.startup_overhead_ms:.2f}ms"
        )

    def to_dict(self) -> dict:
        return {
            "tracker_name": self.tracker_name,
            "sequence_name": self.sequence_name,
            "init_latency_ms": round(self.init_latency_ms, 3),
            "warmup_frames": self.warmup_frames,
            "warmup_mean_ms": round(self.warmup_mean_ms, 3),
            "steady_state_ms": round(self.steady_state_ms, 3),
            "startup_overhead_ms": round(self.startup_overhead_ms, 3),
        }


class ColdStartProfiler:
    """Measure and decompose cold-start latency for visual trackers.

    The profiler separates tracker execution into three phases:

    1. **Initialization** — ``tracker.initialize(frame0, bbox)``
    2. **Warmup** — the first ``warmup_frames`` update calls where latency
       is above the steady-state baseline
    3. **Steady state** — remaining frames at stable per-frame cost

    Warmup end is detected using a sliding-window stability criterion:
    latency is considered stable when the mean in a window of
    ``warmup_window`` frames is within ``stability_threshold`` of the
    long-run mean and the coefficient of variation is below 0.3.

    Args:
        warmup_window: Number of consecutive frames used to test stability.
            Default: 20.
        stability_threshold: Maximum relative deviation of the window mean
            from the tail mean to declare stability.  Default: 0.15 (15 %).
    """

    def __init__(
        self,
        warmup_window: int = 20,
        stability_threshold: float = 0.15,
    ) -> None:
        if warmup_window < 1:
            raise ValueError(f"warmup_window must be ≥ 1, got {warmup_window}")
        if not 0.0 < stability_threshold < 1.0:
            raise ValueError(f"stability_threshold must be in (0, 1), got {stability_threshold}")
        self.warmup_window = warmup_window
        self.stability_threshold = stability_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        tracker: BaseTracker,
        sequence: Sequence,
        max_frames: int = 100,
    ) -> ColdStartResult:
        """Profile cold-start latency of *tracker* on *sequence*.

        Runs the tracker on the first ``max_frames`` frames of the sequence
        and decomposes the resulting latency series into warmup vs. steady
        state.

        Args:
            tracker:    A :class:`~eovot.trackers.base.BaseTracker` to profile.
                        The tracker is **re-initialized** at the start of this
                        call, so its state is reset.
            sequence:   A :class:`~eovot.datasets.base.Sequence` to use.
            max_frames: Maximum number of frames (including frame 0) to
                        evaluate.  Default: 100.

        Returns:
            :class:`ColdStartResult` with full latency breakdown.
        """
        frames = list(sequence)[:max_frames]
        gt = sequence.ground_truth

        if len(frames) < 2:
            raise ValueError("Sequence must have at least 2 frames for cold-start profiling.")

        # ---- Phase 1: Initialization --------------------------------
        init_bbox = tuple(gt[0])
        t0 = time.perf_counter()
        tracker.initialize(frames[0], init_bbox)  # type: ignore[arg-type]
        init_ms = (time.perf_counter() - t0) * 1_000.0

        # ---- Phase 2+3: Per-frame update latencies ------------------
        latencies: List[float] = []
        for frame in frames[1:]:
            t_start = time.perf_counter()
            tracker.update(frame)
            latencies.append((time.perf_counter() - t_start) * 1_000.0)

        if not latencies:
            return ColdStartResult(
                tracker_name=tracker.name,
                sequence_name=sequence.name,
                init_latency_ms=init_ms,
            )

        warmup_end = self._find_warmup_end(latencies)
        arr = np.array(latencies, dtype=np.float64)

        warmup_mean = float(arr[:warmup_end].mean()) if warmup_end > 0 else 0.0
        steady_arr = arr[warmup_end:] if warmup_end < len(arr) else arr
        steady_ms = float(steady_arr.mean())

        overhead = init_ms + warmup_end * max(0.0, warmup_mean - steady_ms)

        return ColdStartResult(
            tracker_name=tracker.name,
            sequence_name=sequence.name,
            init_latency_ms=init_ms,
            per_frame_latencies_ms=latencies,
            warmup_frames=warmup_end,
            warmup_mean_ms=warmup_mean,
            steady_state_ms=steady_ms,
            startup_overhead_ms=overhead,
        )

    def profile_trackers(
        self,
        trackers: List[BaseTracker],
        sequence: Sequence,
        max_frames: int = 100,
    ) -> List[ColdStartResult]:
        """Profile multiple trackers on the same sequence.

        Args:
            trackers:   List of trackers to profile.
            sequence:   Sequence to use for all trackers.
            max_frames: Frame budget per tracker.

        Returns:
            List of :class:`ColdStartResult` in input order.
        """
        return [self.profile(t, sequence, max_frames=max_frames) for t in trackers]

    def to_markdown_table(self, results: List[ColdStartResult]) -> str:
        """Format a list of results as a Markdown table.

        Args:
            results: Output of :meth:`profile_trackers`.

        Returns:
            Markdown table string with tracker cold-start breakdown.
        """
        lines = [
            "| Tracker | Init (ms) | Warmup Frames | Warmup Mean (ms) | Steady State (ms) | Overhead (ms) |",
            "|---------|----------:|:-------------:|-----------------:|------------------:|--------------:|",
        ]
        for r in results:
            lines.append(
                f"| {r.tracker_name} "
                f"| {r.init_latency_ms:.2f} "
                f"| {r.warmup_frames} "
                f"| {r.warmup_mean_ms:.2f} "
                f"| {r.steady_state_ms:.2f} "
                f"| {r.startup_overhead_ms:.2f} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_warmup_end(self, latencies: List[float]) -> int:
        """Return the index after which latency is considered stable.

        Scans the latency series from the beginning and looks for the first
        window of ``warmup_window`` frames whose mean is within
        ``stability_threshold`` of the tail mean and whose coefficient of
        variation is below 0.3.

        Returns:
            Frame index at which steady state begins (0 = no warmup).
        """
        arr = np.array(latencies, dtype=np.float64)
        n = len(arr)
        if n < self.warmup_window:
            return 0

        # Tail mean: average of the last ``warmup_window`` frames.
        tail_mean = float(arr[-self.warmup_window :].mean())
        if tail_mean <= 0:
            return 0

        for i in range(n - self.warmup_window + 1):
            window = arr[i : i + self.warmup_window]
            w_mean = float(window.mean())
            if w_mean <= 0:
                continue
            w_cv = float(window.std() / w_mean)
            relative_diff = abs(w_mean - tail_mean) / tail_mean
            if relative_diff < self.stability_threshold and w_cv < 0.3:
                return i

        return 0
