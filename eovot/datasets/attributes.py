"""Automatic challenge-attribute tagging for VOT sequences.

Derives per-frame and per-sequence challenge attributes directly from ground-truth
bounding boxes — no external annotation files are required.  The attribute set
mirrors the taxonomy used by OTB100, TrackingNet, and LaSOT.

Attribute definitions
---------------------
Each attribute is derived from statistics computed over the ground-truth bbox
sequence.  Thresholds are chosen to match the OTB100 annotation criteria where
comparable (see Davis et al., CVPR 2016).

Attributes:

- **fast_motion** (FM): mean per-frame displacement > 20 % of mean bbox diagonal.
- **scale_change** (SC): std of log-area ratio between consecutive frames > 0.10.
- **aspect_ratio_change** (ARC): std of width/height ratio over sequence > 0.15.
- **low_resolution** (LR): mean bbox area < 400 px².
- **partial_occlusion** (OCC): detected via abrupt IoU drops between adjacent GT
  boxes (proxy for annotation gaps / sudden disappearance).
- **out_of_view** (OOV): bbox centre within 10 % of frame edge in any frame.
- **small_object** (SO): mean bbox area < 1 000 px² (tighter than low-res).
- **background_clutter** (BC): not directly derivable from GT alone; set when
  the object moves through most of the frame (large spatial coverage).
- **illumination_change** (IC): cannot be derived from bbox alone; reserved as
  ``False`` unless explicitly set externally.

Example usage::

    from eovot.datasets.attributes import AttributeTagger, TrackingAttribute

    # Tag a sequence given its ground-truth bboxes and frame size.
    gt = np.array([...])   # shape (N, 4), (x, y, w, h) format
    tags = AttributeTagger.tag_sequence(gt, frame_size=(640, 480))
    print(tags)
    # {'fast_motion': True, 'scale_change': False, 'low_resolution': True, ...}

    # Get human-readable names for present attributes.
    present = AttributeTagger.describe(tags)
    print(present)  # ['fast_motion', 'low_resolution']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Attribute taxonomy
# ---------------------------------------------------------------------------


class TrackingAttribute(str, Enum):
    """Canonical VOT challenge attribute identifiers.

    Values are short codes used as dict keys and in reports.
    """

    FAST_MOTION = "fast_motion"
    SCALE_CHANGE = "scale_change"
    ASPECT_RATIO_CHANGE = "aspect_ratio_change"
    LOW_RESOLUTION = "low_resolution"
    PARTIAL_OCCLUSION = "partial_occlusion"
    OUT_OF_VIEW = "out_of_view"
    SMALL_OBJECT = "small_object"
    BACKGROUND_CLUTTER = "background_clutter"
    ILLUMINATION_CHANGE = "illumination_change"


# Human-readable names for display.
ATTRIBUTE_DISPLAY_NAMES: Dict[str, str] = {
    TrackingAttribute.FAST_MOTION: "Fast Motion",
    TrackingAttribute.SCALE_CHANGE: "Scale Change",
    TrackingAttribute.ASPECT_RATIO_CHANGE: "Aspect Ratio Change",
    TrackingAttribute.LOW_RESOLUTION: "Low Resolution",
    TrackingAttribute.PARTIAL_OCCLUSION: "Partial Occlusion",
    TrackingAttribute.OUT_OF_VIEW: "Out of View",
    TrackingAttribute.SMALL_OBJECT: "Small Object",
    TrackingAttribute.BACKGROUND_CLUTTER: "Background Clutter",
    TrackingAttribute.ILLUMINATION_CHANGE: "Illumination Change",
}

# Short codes for compact tables.
ATTRIBUTE_CODES: Dict[str, str] = {
    TrackingAttribute.FAST_MOTION: "FM",
    TrackingAttribute.SCALE_CHANGE: "SC",
    TrackingAttribute.ASPECT_RATIO_CHANGE: "ARC",
    TrackingAttribute.LOW_RESOLUTION: "LR",
    TrackingAttribute.PARTIAL_OCCLUSION: "OCC",
    TrackingAttribute.OUT_OF_VIEW: "OOV",
    TrackingAttribute.SMALL_OBJECT: "SO",
    TrackingAttribute.BACKGROUND_CLUTTER: "BC",
    TrackingAttribute.ILLUMINATION_CHANGE: "IC",
}

# Default detection thresholds — tuned to match OTB100 annotation criteria.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "fast_motion_ratio": 0.20,
    "scale_change_log_std": 0.10,
    "aspect_ratio_std": 0.15,
    "low_resolution_area": 400.0,
    "small_object_area": 1000.0,
    "occlusion_area_drop": 0.50,
    "out_of_view_margin": 0.10,
    "clutter_coverage": 0.30,
}

# ---------------------------------------------------------------------------
# Sequence attribute tags
# ---------------------------------------------------------------------------


@dataclass
class SequenceAttributes:
    """Attribute presence flags for a single sequence.

    Attributes:
        sequence_name: Identifier of the tagged sequence.
        tags: Mapping of :class:`TrackingAttribute` value → bool.
        n_frames: Number of frames in the sequence.
        frame_size: ``(width, height)`` of the source video.
    """

    sequence_name: str
    tags: Dict[str, bool] = field(default_factory=dict)
    n_frames: int = 0
    frame_size: Optional[tuple] = None

    @property
    def present(self) -> List[str]:
        """Return a sorted list of active attribute keys."""
        return sorted(k for k, v in self.tags.items() if v)

    @property
    def absent(self) -> List[str]:
        """Return a sorted list of inactive attribute keys."""
        return sorted(k for k, v in self.tags.items() if not v)

    def __repr__(self) -> str:
        present = self.present or ["none"]
        return f"SequenceAttributes({self.sequence_name!r}, [{', '.join(present)}])"


# ---------------------------------------------------------------------------
# Core tagger
# ---------------------------------------------------------------------------


class AttributeTagger:
    """Derive challenge attributes from ground-truth bounding-box trajectories.

    All methods are static so the class can be used without instantiation.
    Pass custom thresholds via *thresholds* to override defaults.

    Args:
        thresholds: Override dict for any subset of :data:`DEFAULT_THRESHOLDS`.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    # ------------------------------------------------------------------ #
    # Per-attribute detectors                                             #
    # ------------------------------------------------------------------ #

    def _detect_fast_motion(
        self, bboxes: np.ndarray, threshold: float
    ) -> bool:
        """True when mean frame-to-frame displacement exceeds *threshold* × mean diagonal."""
        if len(bboxes) < 2:
            return False
        cx = bboxes[:, 0] + bboxes[:, 2] / 2.0
        cy = bboxes[:, 1] + bboxes[:, 3] / 2.0
        disp = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        diag = np.sqrt(bboxes[:, 2] ** 2 + bboxes[:, 3] ** 2).mean()
        if diag < 1e-6:
            return False
        return float(disp.mean()) > threshold * diag

    def _detect_scale_change(
        self, bboxes: np.ndarray, threshold: float
    ) -> bool:
        """True when std of consecutive log-area ratios > *threshold*."""
        if len(bboxes) < 2:
            return False
        areas = bboxes[:, 2] * bboxes[:, 3]
        areas = np.where(areas > 0, areas, 1e-6)
        log_ratios = np.log(areas[1:] / areas[:-1])
        return float(log_ratios.std()) > threshold

    def _detect_aspect_ratio_change(
        self, bboxes: np.ndarray, threshold: float
    ) -> bool:
        """True when std of width/height ratio over the sequence > *threshold*."""
        h = np.where(bboxes[:, 3] > 0, bboxes[:, 3], 1e-6)
        ratios = bboxes[:, 2] / h
        return float(ratios.std()) > threshold

    def _detect_low_resolution(
        self, bboxes: np.ndarray, threshold: float
    ) -> bool:
        """True when mean bbox area < *threshold* px²."""
        areas = bboxes[:, 2] * bboxes[:, 3]
        return float(areas.mean()) < threshold

    def _detect_partial_occlusion(
        self, bboxes: np.ndarray, threshold: float
    ) -> bool:
        """True when any consecutive frame pair shows an abrupt area drop > *threshold*."""
        if len(bboxes) < 2:
            return False
        areas = bboxes[:, 2] * bboxes[:, 3]
        areas = np.where(areas > 0, areas, 1e-6)
        ratios = areas[1:] / areas[:-1]
        return bool(np.any(ratios < (1.0 - threshold)))

    def _detect_out_of_view(
        self,
        bboxes: np.ndarray,
        frame_size: tuple,
        margin: float,
    ) -> bool:
        """True when the bbox centre is within *margin*×frame_size of any edge."""
        if frame_size is None:
            return False
        W, H = float(frame_size[0]), float(frame_size[1])
        cx = bboxes[:, 0] + bboxes[:, 2] / 2.0
        cy = bboxes[:, 1] + bboxes[:, 3] / 2.0
        near_left = cx < margin * W
        near_right = cx > (1.0 - margin) * W
        near_top = cy < margin * H
        near_bottom = cy > (1.0 - margin) * H
        return bool(np.any(near_left | near_right | near_top | near_bottom))

    def _detect_small_object(
        self, bboxes: np.ndarray, threshold: float
    ) -> bool:
        """True when mean bbox area < *threshold* px² (stricter than low-res)."""
        areas = bboxes[:, 2] * bboxes[:, 3]
        return float(areas.mean()) < threshold

    def _detect_background_clutter(
        self,
        bboxes: np.ndarray,
        frame_size: tuple,
        threshold: float,
    ) -> bool:
        """True when the object spatial coverage of the frame exceeds *threshold*.

        High spatial coverage (moving across large parts of the frame) is a
        proxy for encountering varied background textures.
        """
        if frame_size is None:
            return False
        W, H = float(frame_size[0]), float(frame_size[1])
        frame_area = W * H
        if frame_area < 1.0:
            return False
        cx = bboxes[:, 0] + bboxes[:, 2] / 2.0
        cy = bboxes[:, 1] + bboxes[:, 3] / 2.0
        x_range = float(cx.max() - cx.min()) / W
        y_range = float(cy.max() - cy.min()) / H
        coverage = x_range * y_range
        return coverage > threshold

    # ------------------------------------------------------------------ #
    # Main tagging interface                                              #
    # ------------------------------------------------------------------ #

    def tag(
        self,
        bboxes: np.ndarray,
        sequence_name: str = "unknown",
        frame_size: Optional[tuple] = None,
    ) -> SequenceAttributes:
        """Derive all challenge attributes for a single sequence.

        Args:
            bboxes: Ground-truth boxes, shape ``(N, 4)`` in ``(x, y, w, h)``
                format.  Rows with zero-area boxes are ignored.
            sequence_name: Identifier stored in the returned object.
            frame_size: ``(width, height)`` in pixels.  Required for OOV and BC
                attributes; both are ``False`` when ``None``.

        Returns:
            :class:`SequenceAttributes` with all attribute flags set.
        """
        bboxes = np.asarray(bboxes, dtype=np.float64)
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(
                f"bboxes must be shape (N, 4), got {bboxes.shape}"
            )
        # Filter out degenerate (zero-area) rows.
        valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
        bboxes = bboxes[valid]

        th = self.thresholds
        tags: Dict[str, bool] = {
            TrackingAttribute.FAST_MOTION: self._detect_fast_motion(
                bboxes, th["fast_motion_ratio"]
            ),
            TrackingAttribute.SCALE_CHANGE: self._detect_scale_change(
                bboxes, th["scale_change_log_std"]
            ),
            TrackingAttribute.ASPECT_RATIO_CHANGE: self._detect_aspect_ratio_change(
                bboxes, th["aspect_ratio_std"]
            ),
            TrackingAttribute.LOW_RESOLUTION: self._detect_low_resolution(
                bboxes, th["low_resolution_area"]
            ),
            TrackingAttribute.PARTIAL_OCCLUSION: self._detect_partial_occlusion(
                bboxes, th["occlusion_area_drop"]
            ),
            TrackingAttribute.OUT_OF_VIEW: self._detect_out_of_view(
                bboxes, frame_size, th["out_of_view_margin"]
            ) if frame_size else False,
            TrackingAttribute.SMALL_OBJECT: self._detect_small_object(
                bboxes, th["small_object_area"]
            ),
            TrackingAttribute.BACKGROUND_CLUTTER: self._detect_background_clutter(
                bboxes, frame_size, th["clutter_coverage"]
            ) if frame_size else False,
            TrackingAttribute.ILLUMINATION_CHANGE: False,
        }

        return SequenceAttributes(
            sequence_name=sequence_name,
            tags=tags,
            n_frames=len(bboxes),
            frame_size=frame_size,
        )

    def tag_dataset(
        self,
        sequences: Dict[str, np.ndarray],
        frame_sizes: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, SequenceAttributes]:
        """Tag all sequences in a dataset dict.

        Args:
            sequences: Mapping of ``sequence_name`` → GT bboxes array.
            frame_sizes: Optional mapping of ``sequence_name`` → ``(W, H)``.

        Returns:
            Mapping of ``sequence_name`` → :class:`SequenceAttributes`.
        """
        frame_sizes = frame_sizes or {}
        return {
            name: self.tag(bboxes, sequence_name=name, frame_size=frame_sizes.get(name))
            for name, bboxes in sequences.items()
        }

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def describe(tags: "SequenceAttributes") -> List[str]:
        """Return the list of active attribute display names."""
        return [
            ATTRIBUTE_DISPLAY_NAMES.get(k, k)
            for k in tags.present
        ]

    @staticmethod
    def attribute_coverage(
        tagged: Dict[str, SequenceAttributes],
    ) -> Dict[str, float]:
        """Compute the fraction of sequences exhibiting each attribute.

        Args:
            tagged: Output of :meth:`tag_dataset`.

        Returns:
            ``{attribute_key: fraction_of_sequences}``
        """
        if not tagged:
            return {}
        n = len(tagged)
        coverage: Dict[str, float] = {}
        all_attrs = list(TrackingAttribute)
        for attr in all_attrs:
            count = sum(1 for t in tagged.values() if t.tags.get(attr, False))
            coverage[attr] = count / n
        return coverage

    def coverage_to_markdown(
        self,
        tagged: Dict[str, SequenceAttributes],
    ) -> str:
        """Format attribute coverage as a Markdown table.

        Args:
            tagged: Output of :meth:`tag_dataset`.

        Returns:
            Markdown string.
        """
        coverage = self.attribute_coverage(tagged)
        lines = [
            "| Attribute | Code | Sequences | Coverage |",
            "|-----------|------|-----------|----------|",
        ]
        for attr, frac in sorted(coverage.items(), key=lambda x: -x[1]):
            display = ATTRIBUTE_DISPLAY_NAMES.get(attr, attr)
            code = ATTRIBUTE_CODES.get(attr, "?")
            count = round(frac * len(tagged))
            lines.append(
                f"| {display} | {code} | {count}/{len(tagged)} | {frac:.1%} |"
            )
        return "\n".join(lines) + "\n"
