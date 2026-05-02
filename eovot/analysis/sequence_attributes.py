"""Sequence difficulty attribute analysis for VOT benchmarking.

Computes per-sequence difficulty attributes from ground-truth bounding boxes,
mirroring the attribute taxonomy used by OTB-100, LaSOT, and GOT-10k to enable
fine-grained, per-attribute tracker comparisons.

Attributes computed
-------------------
SV  — Scale Variation:     max/min box area ratio exceeds threshold.
ARC — Aspect Ratio Change: width/height ratio varies significantly.
FM  — Fast Motion:         per-frame centre displacement > box-size fraction.
LR  — Low Resolution:      target bounding-box area falls below pixel threshold.
PO  — Partial Occlusion:   abrupt box-area drop mid-sequence (proxy for occlusion).
DEF — Deformation:         large intra-sequence aspect-ratio variance.

Example::

    from eovot.datasets.base import OTBDataset
    from eovot.analysis.sequence_attributes import tag_sequences

    dataset = OTBDataset("/data/OTB100")
    tagged = tag_sequences(dataset)
    for seq_name, attrs in tagged.items():
        print(seq_name, attrs.active_flags())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence as TypingSequence

import numpy as np

from ..datasets.base import BaseDataset


# ---------------------------------------------------------------------------
# Thresholds (matching OTB / GOT-10k annotation conventions)
# ---------------------------------------------------------------------------

_SV_RATIO_THRESHOLD: float = 2.0      # max/min area ratio for scale variation
_ARC_RATIO_THRESHOLD: float = 0.5     # |max_ratio - min_ratio| for aspect-ratio change
_FM_DISPLACEMENT_FRACTION: float = 0.2  # centre-displacement / sqrt(area) threshold
_LR_AREA_THRESHOLD_PX: float = 400.0  # < 20×20 px considered low resolution
_PO_AREA_DROP_FRACTION: float = 0.4   # area drops ≥ 40% in a single frame = occlusion proxy
_DEF_AR_STD_THRESHOLD: float = 0.25   # std-dev of aspect-ratio sequence for deformation


@dataclass
class AttributeFlags:
    """Boolean difficulty flags for a single tracking sequence.

    Each flag is ``True`` when the corresponding attribute is present in the
    sequence according to fixed thresholds (see module-level constants).

    Attributes:
        scale_variation:     Target area changes significantly across the sequence.
        aspect_ratio_change: Width-to-height ratio changes significantly.
        fast_motion:         Target moves faster than a fraction of its own size.
        low_resolution:      Target bounding box is very small (< ~20×20 px).
        partial_occlusion:   Abrupt mid-sequence area drop (proxy for occlusion).
        deformation:         Aspect-ratio variance indicates non-rigid deformation.
    """

    scale_variation: bool = False
    aspect_ratio_change: bool = False
    fast_motion: bool = False
    low_resolution: bool = False
    partial_occlusion: bool = False
    deformation: bool = False

    def active_flags(self) -> List[str]:
        """Return short abbreviations for all ``True`` flags."""
        mapping = [
            ("scale_variation", "SV"),
            ("aspect_ratio_change", "ARC"),
            ("fast_motion", "FM"),
            ("low_resolution", "LR"),
            ("partial_occlusion", "PO"),
            ("deformation", "DEF"),
        ]
        return [abbr for attr, abbr in mapping if getattr(self, attr)]

    def to_dict(self) -> Dict[str, bool]:
        return {
            "SV": self.scale_variation,
            "ARC": self.aspect_ratio_change,
            "FM": self.fast_motion,
            "LR": self.low_resolution,
            "PO": self.partial_occlusion,
            "DEF": self.deformation,
        }

    def __str__(self) -> str:
        active = self.active_flags()
        return f"[{', '.join(active)}]" if active else "[none]"


@dataclass
class SequenceAttributes:
    """Full attribute analysis result for a single sequence.

    Attributes:
        sequence_name:   Name of the analysed sequence.
        num_frames:      Total frame count.
        flags:           Per-attribute difficulty flags.
        mean_area_px:    Mean target bounding-box area (pixels²).
        mean_aspect_ratio: Mean width/height ratio across the sequence.
        mean_fps_motion: Mean per-frame centre displacement normalised by
            sqrt(area) — a dimensionless speed indicator.
        scale_ratio:     Observed max/min area ratio (≥ 1.0).
    """

    sequence_name: str
    num_frames: int
    flags: AttributeFlags
    mean_area_px: float
    mean_aspect_ratio: float
    mean_fps_motion: float
    scale_ratio: float

    def to_dict(self) -> Dict:
        return {
            "sequence_name": self.sequence_name,
            "num_frames": self.num_frames,
            "mean_area_px": round(self.mean_area_px, 2),
            "mean_aspect_ratio": round(self.mean_aspect_ratio, 4),
            "mean_fps_motion": round(self.mean_fps_motion, 4),
            "scale_ratio": round(self.scale_ratio, 4),
            "flags": self.flags.to_dict(),
            "active_flags": self.flags.active_flags(),
        }

    def __str__(self) -> str:
        return (
            f"SequenceAttributes[{self.sequence_name}] "
            f"frames={self.num_frames} "
            f"area={self.mean_area_px:.0f}px² "
            f"flags={self.flags}"
        )


def compute_sequence_attributes(
    ground_truth: np.ndarray,
    sequence_name: str = "unknown",
    sv_threshold: float = _SV_RATIO_THRESHOLD,
    arc_threshold: float = _ARC_RATIO_THRESHOLD,
    fm_fraction: float = _FM_DISPLACEMENT_FRACTION,
    lr_area_px: float = _LR_AREA_THRESHOLD_PX,
    po_drop_fraction: float = _PO_AREA_DROP_FRACTION,
    def_std_threshold: float = _DEF_AR_STD_THRESHOLD,
) -> SequenceAttributes:
    """Compute difficulty attribute flags from a ground-truth bounding-box array.

    Args:
        ground_truth: ``(N, 4)`` float array in ``(x, y, w, h)`` format.
        sequence_name: Identifier embedded in the returned object.
        sv_threshold:    Min max/min area ratio to trigger SV flag.
        arc_threshold:   Min |max–min| aspect-ratio change to trigger ARC.
        fm_fraction:     Centre-displacement / sqrt(area) fraction for FM.
        lr_area_px:      Pixel-area threshold below which LR is flagged.
        po_drop_fraction: Fractional area drop in a single frame to flag PO.
        def_std_threshold: Aspect-ratio std-dev threshold for DEF.

    Returns:
        :class:`SequenceAttributes` with all computed metrics and flags.

    Raises:
        ValueError: If ``ground_truth`` is empty or has wrong shape.
    """
    if ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
        raise ValueError(
            f"ground_truth must be shape (N, 4), got {ground_truth.shape}"
        )
    if len(ground_truth) == 0:
        raise ValueError("ground_truth must not be empty.")

    xs, ys, ws, hs = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], ground_truth[:, 3]

    # Guard against degenerate boxes.
    ws = np.maximum(ws, 1e-6)
    hs = np.maximum(hs, 1e-6)

    areas = ws * hs
    aspect_ratios = ws / hs

    # Centre coordinates
    cx = xs + ws / 2.0
    cy = ys + hs / 2.0

    # --- Scale Variation (SV) ---
    min_area = float(areas.min())
    max_area = float(areas.max())
    scale_ratio = max_area / max(min_area, 1e-6)
    sv_flag = scale_ratio >= sv_threshold

    # --- Aspect Ratio Change (ARC) ---
    arc_range = float(aspect_ratios.max() - aspect_ratios.min())
    arc_flag = arc_range >= arc_threshold

    # --- Fast Motion (FM) ---
    if len(cx) > 1:
        displacements = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        sqrt_areas = np.sqrt(areas[:-1])
        normalised = displacements / np.maximum(sqrt_areas, 1e-6)
        mean_motion = float(normalised.mean())
        fm_flag = mean_motion >= fm_fraction
    else:
        mean_motion = 0.0
        fm_flag = False

    # --- Low Resolution (LR) ---
    lr_flag = bool(float(areas.min()) < lr_area_px)

    # --- Partial Occlusion (PO) — proxy via abrupt area drop ---
    if len(areas) > 1:
        area_drops = (areas[:-1] - areas[1:]) / np.maximum(areas[:-1], 1e-6)
        po_flag = bool(float(area_drops.max()) >= po_drop_fraction)
    else:
        po_flag = False

    # --- Deformation (DEF) — aspect-ratio std-dev ---
    ar_std = float(aspect_ratios.std())
    def_flag = ar_std >= def_std_threshold

    flags = AttributeFlags(
        scale_variation=sv_flag,
        aspect_ratio_change=arc_flag,
        fast_motion=fm_flag,
        low_resolution=lr_flag,
        partial_occlusion=po_flag,
        deformation=def_flag,
    )

    return SequenceAttributes(
        sequence_name=sequence_name,
        num_frames=len(ground_truth),
        flags=flags,
        mean_area_px=float(areas.mean()),
        mean_aspect_ratio=float(aspect_ratios.mean()),
        mean_fps_motion=mean_motion,
        scale_ratio=scale_ratio,
    )


def tag_sequences(
    dataset: BaseDataset,
    **threshold_kwargs,
) -> Dict[str, SequenceAttributes]:
    """Compute attribute flags for every sequence in a dataset.

    Iterates over the dataset using its :class:`~eovot.datasets.base.Sequence`
    objects and computes :class:`SequenceAttributes` from each sequence's
    ground-truth boxes.  Only the GT array is read — no image frames are
    loaded.

    Args:
        dataset: Any :class:`~eovot.datasets.base.BaseDataset` instance.
        **threshold_kwargs: Forwarded verbatim to
            :func:`compute_sequence_attributes` for threshold overrides.

    Returns:
        Dict mapping sequence name → :class:`SequenceAttributes`.

    Example::

        from eovot.datasets.base import OTBDataset
        from eovot.analysis.sequence_attributes import tag_sequences

        dataset = OTBDataset("/data/OTB100")
        tagged = tag_sequences(dataset)
        fast_motion = [n for n, a in tagged.items() if a.flags.fast_motion]
        print(f"{len(fast_motion)} sequences with fast motion")
    """
    results: Dict[str, SequenceAttributes] = {}
    for seq in dataset:
        attrs = compute_sequence_attributes(
            seq.ground_truth,
            sequence_name=seq.name,
            **threshold_kwargs,
        )
        results[seq.name] = attrs
    return results
