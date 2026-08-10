"""Confidence-adaptive tracker wrapper for EOVOT.

Wraps any :class:`~eovot.trackers.base.BaseTracker` with an NCC-based
confidence estimator.  On every frame, the predicted ROI is cropped,
resized to a fixed template size, and compared against a stored appearance
template using Normalized Cross-Correlation (NCC).  When NCC drops below
``loss_threshold`` the tracker is flagged as *lost*; it still returns the
last known bounding box so that the benchmark loop continues uninterrupted,
but the loss flag is recorded in :attr:`confidence_history` and
:attr:`is_lost`.

Template updates prevent drift: whenever NCC exceeds ``update_threshold``
on a frame that is a multiple of ``update_interval``, the stored template
is blended toward the current appearance.

Edge deployment motivation
--------------------------
Classical trackers (KCF, MOSSE) and cheap CV-based methods have no built-in
confidence signal; they drift silently and keep returning wrong boxes.  This
wrapper gives any such tracker a lightweight quality signal without adding a
heavy re-detection network — a common pattern in embedded tracking pipelines
where a re-detection sub-network runs only when the main tracker flags a loss.

Example::

    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.confidence import ConfidenceAdaptiveTracker
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine

    tracker = ConfidenceAdaptiveTracker(MOSSETracker(), loss_threshold=0.25)
    dataset = SyntheticDataset(num_sequences=3, num_frames=80)
    engine  = BenchmarkEngine(verbose=False)
    result  = engine.run(tracker, dataset, dataset_name="Synthetic")

    print(f"Frames flagged as lost: {tracker.total_lost_frames}")
    print(f"Mean confidence: {sum(tracker.confidence_history) / len(tracker.confidence_history):.3f}")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseTracker, BBox

# Default patch size for NCC template matching.
_DEFAULT_TEMPLATE_SIZE: Tuple[int, int] = (64, 64)


def _ncc(patch: np.ndarray, template: np.ndarray) -> float:
    """Normalized Cross-Correlation between two same-sized grayscale patches.

    Returns a value in ``[-1, 1]``.  Returns 0.0 when either patch has
    near-zero standard deviation (flat region — uninformative).

    Args:
        patch:    Float32 grayscale array, shape ``(H, W)``.
        template: Float32 grayscale array, same shape as *patch*.

    Returns:
        NCC scalar in ``[-1, 1]``.
    """
    p = patch.astype(np.float32)
    t = template.astype(np.float32)

    p -= p.mean()
    t -= t.mean()

    norm_p = np.linalg.norm(p)
    norm_t = np.linalg.norm(t)

    if norm_p < 1e-6 or norm_t < 1e-6:
        return 0.0

    return float(np.sum(p * t) / (norm_p * norm_t))


def _extract_patch(frame: np.ndarray, bbox: BBox, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Crop the region defined by *bbox* from *frame* and resize to *size*.

    Handles partial out-of-bounds boxes by clamping to frame boundaries;
    returns ``None`` when the box is fully outside the frame or has
    non-positive dimensions.

    Args:
        frame: BGR uint8 array ``(H, W, 3)`` or grayscale ``(H, W)``.
        bbox:  Bounding box ``(x, y, w, h)`` in pixel coordinates.
        size:  Target ``(width, height)`` after resizing.

    Returns:
        Float32 grayscale patch of shape ``(height, width)``, or ``None``.
    """
    x, y, w, h = bbox
    H, W = frame.shape[:2]

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    resized = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


class ConfidenceAdaptiveTracker(BaseTracker):
    """Wraps any :class:`BaseTracker` with NCC-based confidence estimation.

    On each frame after the first, extracts the predicted ROI, resizes it to
    ``template_size``, and computes NCC against a stored grayscale template.
    The result is exposed as :attr:`confidence` and appended to
    :attr:`confidence_history`.

    Template maintenance
    ~~~~~~~~~~~~~~~~~~~~
    The initial template is created from the ground-truth bounding box on the
    first frame.  On subsequent frames, when NCC exceeds ``update_threshold``
    and the frame index is a multiple of ``update_interval``, the stored
    template is updated via:

    .. code-block:: text

        template = (1 - alpha) * template + alpha * current_patch

    where ``alpha`` equals ``template_alpha``.  This lets the tracker adapt
    to gradual appearance changes (illumination, slight rotation) without
    drifting to a wrong target.

    Args:
        tracker:          Any :class:`BaseTracker` to wrap.
        template_size:    ``(width, height)`` of the appearance template in
                          pixels.  Larger patches carry more information but
                          cost more CPU.  Default: ``(64, 64)``.
        loss_threshold:   NCC below this value marks the frame as *lost*.
                          Typical range 0.15–0.40.  Default: ``0.25``.
        update_threshold: NCC above this value allows a template update.
                          Default: ``0.65``.
        update_interval:  Update the template at most every this many frames.
                          Default: ``10``.
        template_alpha:   EMA weight for template update (0 = no update,
                          1 = replace).  Default: ``0.10``.

    Raises:
        ValueError: If ``loss_threshold >= update_threshold``.
    """

    def __init__(
        self,
        tracker: BaseTracker,
        template_size: Tuple[int, int] = _DEFAULT_TEMPLATE_SIZE,
        loss_threshold: float = 0.25,
        update_threshold: float = 0.65,
        update_interval: int = 10,
        template_alpha: float = 0.10,
    ) -> None:
        if loss_threshold >= update_threshold:
            raise ValueError(
                f"loss_threshold ({loss_threshold}) must be strictly less than "
                f"update_threshold ({update_threshold})."
            )
        super().__init__(name=f"{tracker.name}_conf")
        self._tracker = tracker
        self._template_size = template_size
        self.loss_threshold = loss_threshold
        self.update_threshold = update_threshold
        self.update_interval = update_interval
        self.template_alpha = template_alpha

        self._template: Optional[np.ndarray] = None
        self._last_bbox: Optional[BBox] = None
        self._frame_idx: int = 0
        self._confidence: float = 1.0
        self._is_lost: bool = False
        self._confidence_history: List[float] = []
        self._lost_frame_indices: List[int] = []

    # ------------------------------------------------------------------
    # BaseTracker interface
    # ------------------------------------------------------------------

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise the underlying tracker and create the appearance template.

        Args:
            frame: First frame as a ``(H, W, 3)`` BGR uint8 array.
            bbox:  Ground-truth bounding box ``(x, y, w, h)``.
        """
        self._tracker.initialize(frame, bbox)
        self._last_bbox = bbox
        self._frame_idx = 0
        self._confidence = 1.0
        self._is_lost = False
        self._confidence_history = []
        self._lost_frame_indices = []

        patch = _extract_patch(frame, bbox, self._template_size)
        self._template = patch

    def update(self, frame: np.ndarray) -> BBox:
        """Run the underlying tracker, then estimate confidence via NCC.

        When confidence falls below :attr:`loss_threshold`, the frame is
        flagged as lost and its index is recorded in
        :attr:`lost_frame_indices`.  The returned bounding box always comes
        from the underlying tracker regardless of loss status — callers can
        inspect :attr:`is_lost` to decide whether to trust the prediction.

        Args:
            frame: Current frame as a ``(H, W, 3)`` BGR uint8 array.

        Returns:
            Predicted bounding box ``(x, y, w, h)`` from the wrapped tracker.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if self._template is None or self._last_bbox is None:
            raise RuntimeError(
                "ConfidenceAdaptiveTracker has not been initialised. "
                "Call initialize() first."
            )

        self._frame_idx += 1
        bbox = self._tracker.update(frame)
        self._last_bbox = bbox

        patch = _extract_patch(frame, bbox, self._template_size)
        if patch is not None:
            ncc = _ncc(patch, self._template)
            self._confidence = ncc
            self._is_lost = ncc < self.loss_threshold

            if not self._is_lost and (self._frame_idx % self.update_interval == 0):
                if ncc >= self.update_threshold:
                    self._template = (
                        (1.0 - self.template_alpha) * self._template
                        + self.template_alpha * patch
                    )
        else:
            # Patch is out of bounds — treat as lost without changing template.
            self._confidence = 0.0
            self._is_lost = True

        self._confidence_history.append(self._confidence)
        if self._is_lost:
            self._lost_frame_indices.append(self._frame_idx)

        return bbox

    # ------------------------------------------------------------------
    # Confidence interface
    # ------------------------------------------------------------------

    @property
    def confidence(self) -> float:
        """NCC confidence score for the most recent frame (``-1`` to ``1``).

        Returns ``1.0`` before any :meth:`update` calls (perfect on init).
        """
        return self._confidence

    @property
    def is_lost(self) -> bool:
        """``True`` when the most recent frame was flagged as a tracking loss."""
        return self._is_lost

    @property
    def confidence_history(self) -> List[float]:
        """Per-frame NCC confidence scores since the last :meth:`initialize` call.

        The list has one entry per :meth:`update` call, in order.  Length
        equals the number of frames processed since the last initialization
        (i.e. ``self._frame_idx``).
        """
        return list(self._confidence_history)

    @property
    def lost_frame_indices(self) -> List[int]:
        """1-indexed frame indices where a loss was detected.

        Empty when no losses occurred.
        """
        return list(self._lost_frame_indices)

    @property
    def total_lost_frames(self) -> int:
        """Total count of frames flagged as lost since the last initialization."""
        return len(self._lost_frame_indices)

    @property
    def loss_rate(self) -> float:
        """Fraction of processed frames that were flagged as lost (0.0–1.0).

        Returns ``0.0`` before any frames are processed.
        """
        if not self._confidence_history:
            return 0.0
        return self.total_lost_frames / len(self._confidence_history)

    @property
    def mean_confidence(self) -> float:
        """Mean NCC confidence across all processed frames.

        Returns ``1.0`` before any frames are processed.
        """
        if not self._confidence_history:
            return 1.0
        return float(np.mean(self._confidence_history))

    @property
    def underlying_tracker(self) -> BaseTracker:
        """The wrapped :class:`BaseTracker` instance."""
        return self._tracker
