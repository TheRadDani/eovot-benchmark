"""AdaptiveSkipTracker — confidence-guided dynamic frame-skip wrapper.

:class:`FrameSkipTracker` (see :mod:`eovot.trackers.frame_skip`) applies a
*fixed* skip rate: every k-th frame is processed and the rest are propagated.
This is simple and predictable but wasteful on hard sequences (where skipping
hurts accuracy) and conservative on easy ones (where more frames could safely
be skipped without quality loss).

:class:`AdaptiveSkipTracker` replaces the fixed schedule with a **PSR-driven
bang-bang controller**:

* After each full tracker update the PSR is inspected.
* If PSR ≥ ``high_threshold`` → the tracker is confident; skip the next frame.
* If PSR < ``low_threshold`` → the tracker is uncertain; process the next frame.
* Between the two thresholds (hysteresis band) → carry the current skip state.
* ``max_consecutive_skips`` caps the maximum run of skipped frames so the
  tracker cannot fall arbitrarily far behind a fast-moving target.

Compared to :class:`~eovot.trackers.frame_skip.FrameSkipTracker`:

+---------------------------+------------------+------------------+
| Property                  | FrameSkipTracker | AdaptiveSkip     |
+===========================+==================+==================+
| Skip decision             | fixed interval   | PSR-driven       |
| Easy sequences            | may over-process | skips more       |
| Hard sequences            | skips regardless | skips less       |
| Requires ConfidenceTracker| no               | yes              |
| Throughput predictability | deterministic    | sequence-adaptive|
+---------------------------+------------------+------------------+

Skip modes
~~~~~~~~~~
``"repeat"`` (default)
    Propagate the last computed bounding box on skipped frames.

``"linear"``
    Linearly extrapolate the bounding-box centre from the last two active
    predictions — identical to the mode supported by
    :class:`~eovot.trackers.frame_skip.FrameSkipTracker`.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.adaptive_skip import AdaptiveSkipTracker
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=5)
    engine  = BenchmarkEngine(verbose=False)

    # Baseline: MOSSE at full rate
    baseline = engine.run(MOSSETracker(), dataset, dataset_name="Syn")

    # Adaptive: skip frames when PSR >= 7.0, always process when PSR < 4.0
    adaptive = engine.run(
        AdaptiveSkipTracker(MOSSETracker(), high_threshold=7.0, low_threshold=4.0),
        dataset,
        dataset_name="Syn",
    )

    print(f"FPS gain : {adaptive.mean_fps / baseline.mean_fps:.2f}x")
    print(f"IoU cost : {baseline.mean_iou - adaptive.mean_iou:.4f}")
    print(f"Skip rate: {adaptive.sequence_results[0].profiling.frame_count} frames processed")
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .base import BBox, ConfidenceTracker

SkipMode = Literal["repeat", "linear"]

# PSR thresholds from Bolme et al. (CVPR 2010):
#   PSR ≥ 7   → confident tracking
#   PSR < 4   → likely failure
_DEFAULT_HIGH = 7.0
_DEFAULT_LOW = 4.0


class AdaptiveSkipTracker(ConfidenceTracker):
    """PSR-driven adaptive frame-skip wrapper for :class:`ConfidenceTracker`.

    Dynamically decides whether to skip the current frame based on the PSR
    confidence score reported by the wrapped tracker after its last active
    update.  This delivers the throughput benefits of frame-skipping on easy
    sequences while maintaining accuracy on hard ones.

    Args:
        tracker: Any :class:`ConfidenceTracker` (e.g.
            :class:`~eovot.trackers.mosse.MOSSETracker` or
            :class:`~eovot.trackers.kcf.KCFTracker`).
        high_threshold: PSR above which the next frame is elided.
            Default: 7.0 (confident).
        low_threshold: PSR below which the next frame is always processed.
            Default: 4.0 (uncertain / possible failure).
        max_consecutive_skips: Hard cap on the number of frames that may be
            skipped in a row.  Prevents long skip runs on sequences where the
            target moves slowly but non-linearly.  Default: 4.
        mode: Propagation strategy for skipped frames.
            ``"repeat"`` returns the last active box unchanged;
            ``"linear"`` linearly extrapolates from the velocity of the last
            two active predictions.  Default: ``"repeat"``.

    Raises:
        ValueError: If *high_threshold* ≤ *low_threshold* or
            *max_consecutive_skips* < 1.
        TypeError: If *tracker* does not implement :class:`ConfidenceTracker`.
    """

    def __init__(
        self,
        tracker: ConfidenceTracker,
        high_threshold: float = _DEFAULT_HIGH,
        low_threshold: float = _DEFAULT_LOW,
        max_consecutive_skips: int = 4,
        mode: SkipMode = "repeat",
    ) -> None:
        if not isinstance(tracker, ConfidenceTracker):
            raise TypeError(
                f"AdaptiveSkipTracker requires a ConfidenceTracker, "
                f"got {type(tracker).__name__}. "
                f"Use FrameSkipTracker for trackers without confidence scoring."
            )
        if high_threshold <= low_threshold:
            raise ValueError(
                f"high_threshold ({high_threshold}) must exceed "
                f"low_threshold ({low_threshold})."
            )
        if max_consecutive_skips < 1:
            raise ValueError(
                f"max_consecutive_skips must be >= 1, got {max_consecutive_skips}."
            )

        super().__init__(name=f"{tracker.name}_adaptive")
        self._tracker = tracker
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.mode: SkipMode = mode

        # Runtime state reset in initialize()
        self._last_bbox: Optional[BBox] = None
        self._prev_bbox: Optional[BBox] = None
        self._consecutive_skips: int = 0
        self._skip_next: bool = False   # carry-forward skip decision
        self._total_frames: int = 0
        self._active_frames: int = 0
        self._last_psr: float = 0.0

    # ------------------------------------------------------------------
    # ConfidenceTracker interface
    # ------------------------------------------------------------------

    @property
    def last_confidence(self) -> float:
        """PSR from the last active update of the wrapped tracker.

        Returns ``0.0`` before the first :meth:`update` call.
        On skipped frames returns the PSR from the most recent active frame —
        this is the value that *drove* the skip decision, so it remains the
        most relevant confidence estimate for the caller.
        """
        return self._last_psr

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and reset all adaptive-skip state.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._prev_bbox = bbox
        self._consecutive_skips = 0
        self._skip_next = False
        self._total_frames = 0
        self._active_frames = 0
        self._last_psr = 0.0

    def update(self, frame: np.ndarray) -> BBox:
        """Return a bounding-box prediction, possibly from a cached frame.

        The skip decision is made at the *end* of each active frame based on
        the PSR reported by the wrapped tracker.  This means the very first
        frame after ``initialize()`` is always active (the tracker has no PSR
        yet), and subsequent decisions are data-driven.

        Skip cascade:

        1. If ``_skip_next`` is True **and** ``_consecutive_skips < max_consecutive_skips``
           → passive frame: propagate last bbox (repeat or linear).
        2. Otherwise → active frame: call the wrapped tracker, update PSR,
           set ``_skip_next`` according to the new PSR and hysteresis.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._last_bbox is None:
            raise RuntimeError(
                "AdaptiveSkipTracker is not initialised. Call initialize() first."
            )

        self._total_frames += 1

        should_skip = (
            self._skip_next
            and self._consecutive_skips < self.max_consecutive_skips
        )

        if should_skip:
            # Passive frame: propagate without updating the tracker
            self._consecutive_skips += 1
            if self.mode == "linear":
                self._last_bbox = self._extrapolate()
        else:
            # Active frame: delegate to the wrapped tracker
            prev = self._last_bbox
            self._last_bbox = self._tracker.update(frame)
            self._prev_bbox = prev
            self._consecutive_skips = 0
            self._active_frames += 1

            # Read the PSR from the wrapped tracker
            psr = self._tracker.last_confidence
            self._last_psr = psr

            # Update skip decision via hysteresis
            if psr >= self.high_threshold:
                self._skip_next = True
            elif psr < self.low_threshold:
                self._skip_next = False
            # In the hysteresis band: keep current decision

        return self._last_bbox

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def underlying_tracker(self) -> ConfidenceTracker:
        """The wrapped :class:`ConfidenceTracker` instance."""
        return self._tracker

    @property
    def active_frame_count(self) -> int:
        """Number of frames on which the underlying tracker actually ran."""
        return self._active_frames

    @property
    def skipped_frame_count(self) -> int:
        """Number of frames propagated without a tracker update."""
        return self._total_frames - self._active_frames

    @property
    def effective_skip_fraction(self) -> float:
        """Fraction of frames skipped so far (0.0 = no skipping, 1.0 = all)."""
        if self._total_frames == 0:
            return 0.0
        return self.skipped_frame_count / self._total_frames

    @property
    def effective_speedup(self) -> float:
        """Approximate throughput multiplier vs. running the wrapped tracker at full rate.

        Computed as ``total_frames / active_frames``.  A value of 2.0 means
        the tracker ran on half the frames, giving roughly 2× the effective
        FPS for the same real-time clock.
        """
        if self._active_frames == 0:
            return 1.0
        return self._total_frames / self._active_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extrapolate(self) -> BBox:
        """Linearly extrapolate from the velocity of the last two active frames."""
        if self._prev_bbox is None or self._last_bbox is None:
            return self._last_bbox or (0.0, 0.0, 0.0, 0.0)

        px, py, pw, ph = self._prev_bbox
        lx, ly, lw, lh = self._last_bbox

        vcx = (lx + lw / 2.0) - (px + pw / 2.0)
        vcy = (ly + lh / 2.0) - (py + ph / 2.0)

        new_cx = (lx + lw / 2.0) + vcx
        new_cy = (ly + lh / 2.0) + vcy

        return (new_cx - lw / 2.0, new_cy - lh / 2.0, lw, lh)
