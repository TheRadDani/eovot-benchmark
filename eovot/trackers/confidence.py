"""Appearance-based confidence estimation wrapper for any BaseTracker.

Wraps an existing tracker and produces a per-frame confidence score in
[0, 1] by measuring Normalised Cross-Correlation (NCC) between the
appearance template captured at initialisation and the appearance patch
at the current predicted bounding box.

NCC is translation- and scale-invariant under the patch extraction
convention used here (both patches resized to the same canonical size),
making it a lightweight proxy for template drift that requires no
additional model training.

Usage::

    from eovot.trackers.confidence import AppearanceConfidenceTracker
    from eovot.trackers.mosse import MOSSETracker

    base = MOSSETracker()
    tracker = AppearanceConfidenceTracker(base, template_update_thresh=0.60)

    # Standard BaseTracker API — fully compatible with BenchmarkEngine
    tracker.initialize(frame0, bbox0)
    bbox = tracker.update(frame1)           # returns BBox

    # Extended API
    tracked = tracker.update_with_confidence(frame2)   # TrackedFrame
    print(tracked.bbox, tracked.confidence)

    # Inspect history
    print(tracker.confidence_history)       # List[float] per frame
    print(tracker.low_confidence_frames)    # List[int] frame indices
    print(tracker.mean_confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

_PATCH_SIZE: Tuple[int, int] = (64, 64)


@dataclass
class TrackedFrame:
    """Bounding box prediction annotated with an appearance confidence score.

    Attributes:
        bbox: Predicted bounding box ``(x, y, w, h)``.
        confidence: NCC-based appearance similarity in [0, 1] between the
            init template and the current patch.  Values near 1.0 indicate
            high appearance consistency; values near 0.0 signal drift or
            occlusion.
        frame_idx: Zero-based frame index within the current sequence.
    """

    bbox: BBox
    confidence: float
    frame_idx: int

    def __str__(self) -> str:
        x, y, w, h = self.bbox
        return (
            f"Frame {self.frame_idx}: bbox=({x:.1f},{y:.1f},{w:.1f},{h:.1f})  "
            f"conf={self.confidence:.4f}"
        )


@dataclass
class ConfidenceReport:
    """Summary of confidence statistics over a complete sequence.

    Attributes:
        tracker_name: Name of the wrapped tracker.
        total_frames: Total number of update calls (initialisation excluded).
        mean_confidence: Mean NCC score across all frames.
        min_confidence: Minimum observed confidence.
        low_confidence_count: Frames below the drift threshold.
        low_confidence_ratio: low_confidence_count / total_frames.
        drift_detected: True when low_confidence_ratio exceeds
            ``drift_ratio_thresh``.
    """

    tracker_name: str
    total_frames: int
    mean_confidence: float
    min_confidence: float
    low_confidence_count: int
    low_confidence_ratio: float
    drift_detected: bool

    def __str__(self) -> str:
        return (
            f"ConfidenceReport[{self.tracker_name}]  frames={self.total_frames}  "
            f"mean_conf={self.mean_confidence:.4f}  min={self.min_confidence:.4f}  "
            f"low={self.low_confidence_count} ({self.low_confidence_ratio:.1%})  "
            f"drift={'YES' if self.drift_detected else 'no'}"
        )

    def to_dict(self) -> dict:
        return {
            "tracker": self.tracker_name,
            "total_frames": self.total_frames,
            "mean_confidence": round(self.mean_confidence, 6),
            "min_confidence": round(self.min_confidence, 6),
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_ratio": round(self.low_confidence_ratio, 6),
            "drift_detected": self.drift_detected,
        }


class AppearanceConfidenceTracker(BaseTracker):
    """Wraps any :class:`BaseTracker` and adds per-frame NCC confidence scoring.

    The confidence score is the normalised cross-correlation between a fixed
    appearance template (captured on ``initialize``) and the appearance patch
    extracted from the predicted bounding box on each ``update`` call.

    An optional *template update* mechanism refreshes the stored template when
    confidence exceeds ``template_update_thresh``, preventing drift accumulation
    over long sequences without triggering false positives during occlusion.

    Args:
        tracker: Any :class:`BaseTracker` instance to wrap.
        template_update_thresh: NCC threshold above which the stored template
            is refreshed with the current patch.  Set to ``1.0`` to disable
            template updates (pure one-shot matching).  Default: ``0.70``.
        low_confidence_thresh: NCC value below which a frame is counted as
            "low confidence".  Default: ``0.40``.
        drift_ratio_thresh: Fraction of low-confidence frames above which
            :meth:`confidence_report` reports drift.  Default: ``0.30``.

    Example::

        base = KCFTracker()
        tracker = AppearanceConfidenceTracker(base, template_update_thresh=0.65)
        tracker.initialize(frame0, bbox0)
        for frame in rest_of_sequence:
            tf = tracker.update_with_confidence(frame)
            print(tf.confidence)
    """

    def __init__(
        self,
        tracker: BaseTracker,
        template_update_thresh: float = 0.70,
        low_confidence_thresh: float = 0.40,
        drift_ratio_thresh: float = 0.30,
    ) -> None:
        if not isinstance(tracker, BaseTracker):
            raise TypeError(f"tracker must be a BaseTracker, got {type(tracker)}")
        if not 0.0 <= template_update_thresh <= 1.0:
            raise ValueError(f"template_update_thresh must be in [0, 1], got {template_update_thresh}")
        if not 0.0 <= low_confidence_thresh <= 1.0:
            raise ValueError(f"low_confidence_thresh must be in [0, 1], got {low_confidence_thresh}")

        super().__init__(name=f"Confidence({tracker.name})")
        self._tracker = tracker
        self.template_update_thresh = template_update_thresh
        self.low_confidence_thresh = low_confidence_thresh
        self.drift_ratio_thresh = drift_ratio_thresh

        self._template: Optional[np.ndarray] = None
        self._frame_idx: int = 0
        self._confidence_history: List[float] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the wrapped tracker and capture the appearance template.

        Args:
            frame: BGR ``(H, W, 3)`` uint8 array.
            bbox: Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._template = self._extract_patch(frame, bbox)
        self._frame_idx = 0
        self._confidence_history = []

    def update(self, frame: np.ndarray) -> BBox:
        """Return the predicted bounding box (standard :class:`BaseTracker` API).

        Confidence is computed and appended to :attr:`confidence_history` as
        a side-effect.

        Args:
            frame: BGR ``(H, W, 3)`` uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)``.
        """
        return self.update_with_confidence(frame).bbox

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    def update_with_confidence(self, frame: np.ndarray) -> TrackedFrame:
        """Predict the bounding box and compute the appearance confidence.

        Args:
            frame: BGR ``(H, W, 3)`` uint8 array.

        Returns:
            :class:`TrackedFrame` with ``bbox`` and ``confidence`` fields.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._template is None:
            raise RuntimeError("initialize() must be called before update_with_confidence()")

        self._frame_idx += 1
        bbox = self._tracker.update(frame)

        current_patch = self._extract_patch(frame, bbox)
        confidence = self._ncc_score(self._template, current_patch)
        self._confidence_history.append(confidence)

        if confidence >= self.template_update_thresh:
            self._template = current_patch

        return TrackedFrame(bbox=bbox, confidence=confidence, frame_idx=self._frame_idx)

    @property
    def confidence_history(self) -> List[float]:
        """Per-frame confidence scores since the last :meth:`initialize` call."""
        return list(self._confidence_history)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence over all update calls so far, or 0.0 if no updates."""
        return float(np.mean(self._confidence_history)) if self._confidence_history else 0.0

    @property
    def low_confidence_frames(self) -> List[int]:
        """Zero-based frame indices where confidence was below the low threshold."""
        return [
            i + 1  # frame_idx starts at 1
            for i, c in enumerate(self._confidence_history)
            if c < self.low_confidence_thresh
        ]

    def confidence_report(self) -> ConfidenceReport:
        """Return a summary of confidence statistics for the current sequence."""
        total = len(self._confidence_history)
        if total == 0:
            return ConfidenceReport(
                tracker_name=self._tracker.name,
                total_frames=0,
                mean_confidence=0.0,
                min_confidence=0.0,
                low_confidence_count=0,
                low_confidence_ratio=0.0,
                drift_detected=False,
            )
        arr = np.array(self._confidence_history)
        low_count = int((arr < self.low_confidence_thresh).sum())
        low_ratio = low_count / total
        return ConfidenceReport(
            tracker_name=self._tracker.name,
            total_frames=total,
            mean_confidence=float(arr.mean()),
            min_confidence=float(arr.min()),
            low_confidence_count=low_count,
            low_confidence_ratio=low_ratio,
            drift_detected=(low_ratio > self.drift_ratio_thresh),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_patch(frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Extract and normalise a fixed-size grayscale patch from *frame*.

        The patch is clipped to image boundaries and resized to
        ``_PATCH_SIZE``.  Returns a zero patch if the bounding box has
        zero or negative area.
        """
        x, y, w, h = bbox
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(frame.shape[1], int(round(x + w)))
        y2 = min(frame.shape[0], int(round(y + h)))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(_PATCH_SIZE, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(crop, _PATCH_SIZE, interpolation=cv2.INTER_LINEAR)
        return patch.astype(np.float32)

    @staticmethod
    def _ncc_score(template: np.ndarray, patch: np.ndarray) -> float:
        """Normalised Cross-Correlation score in [0, 1].

        NCC is computed as::

            ncc = (sum((T - mean_T)(P - mean_P))) / (n * std_T * std_P)

        The raw NCC is in [-1, 1]; it is mapped to [0, 1] via
        ``(ncc + 1) / 2`` so the output is a proper similarity score.
        Values near 1.0 indicate high appearance similarity; 0.5 is
        uncorrelated; values near 0.0 are anti-correlated (unlikely for
        a well-functioning tracker).
        """
        t = template - template.mean()
        p = patch - patch.mean()
        std_t = float(t.std())
        std_p = float(p.std())
        if std_t < 1e-6 or std_p < 1e-6:
            return 0.5  # degenerate patch — signal neutral confidence
        n = template.size
        raw_ncc = float(np.sum(t * p)) / (n * std_t * std_p)
        raw_ncc = float(np.clip(raw_ncc, -1.0, 1.0))
        return (raw_ncc + 1.0) / 2.0
