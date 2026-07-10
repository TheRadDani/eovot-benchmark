"""Challenge-attributed synthetic sequences for EOVOT diagnostics.

Extends :mod:`eovot.datasets.synthetic` with six standard tracking challenge
types derived from the OTB / LaSOT attribute taxonomy.  Each challenge type
generates sequences that deliberately stress a specific tracker weakness,
enabling per-attribute failure analysis without requiring real annotated data.

Challenge Types
---------------
``OCCLUSION``
    The target is temporarily occluded by a solid rectangle overlaid on the
    frame.  Models partial and full occlusions from foreground objects.

``SCALE_CHANGE``
    The target bounding box grows or shrinks by up to 3× over the sequence,
    keeping the centre constant.  Tests scale-adaptive mechanisms.

``FAST_MOTION``
    High-velocity linear motion so that frame-to-frame displacement often
    exceeds the target dimension.  Stresses search region sizing.

``ILLUMINATION_CHANGE``
    The entire frame brightness varies sinusoidally between dark and bright
    across the sequence.  Challenges appearance models sensitive to illumination.

``LOW_RESOLUTION``
    The target is rendered small (≤ 20×20 px) relative to the frame, then
    the frame is upsampled to full resolution via nearest-neighbour interpolation.
    Models challenging distant/small targets.

``BACKGROUND_CLUTTER``
    The background contains multiple coloured "distractor" rectangles with
    colours close to the target.  Tests discriminative foreground/background
    separation.

Usage::

    from eovot.datasets.synthetic_challenges import ChallengeAttribute, ChallengeDataset

    ds = ChallengeDataset(
        num_sequences=4,
        num_frames=80,
        challenges=[ChallengeAttribute.OCCLUSION, ChallengeAttribute.FAST_MOTION],
    )
    for seq in ds:
        print(seq.name, seq.attributes)  # [ChallengeAttribute.OCCLUSION]

    # Use with the standard BenchmarkEngine:
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="ChallengeSet")
"""

from __future__ import annotations

import math
import random
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence


# ---------------------------------------------------------------------------
# Challenge attribute enum
# ---------------------------------------------------------------------------

class ChallengeAttribute(Enum):
    """Standard tracking challenge categories (OTB/LaSOT taxonomy)."""

    OCCLUSION = auto()
    SCALE_CHANGE = auto()
    FAST_MOTION = auto()
    ILLUMINATION_CHANGE = auto()
    LOW_RESOLUTION = auto()
    BACKGROUND_CLUTTER = auto()

    def label(self) -> str:
        """Short lowercase label used in report tables and filenames."""
        return self.name.lower()

    def description(self) -> str:
        """One-line human-readable description."""
        return {
            ChallengeAttribute.OCCLUSION: "Target is temporarily occluded",
            ChallengeAttribute.SCALE_CHANGE: "Target undergoes significant scale variation",
            ChallengeAttribute.FAST_MOTION: "Per-frame displacement exceeds target size",
            ChallengeAttribute.ILLUMINATION_CHANGE: "Frame-wide brightness varies over time",
            ChallengeAttribute.LOW_RESOLUTION: "Target occupies fewer than 400 pixels",
            ChallengeAttribute.BACKGROUND_CLUTTER: "Distractors near the target colour exist",
        }[self]


# ---------------------------------------------------------------------------
# Attributed sequence
# ---------------------------------------------------------------------------

class AttributedSequence(Sequence):
    """A :class:`~eovot.datasets.base.Sequence` with a list of challenge tags.

    Attributes:
        attributes: Challenge categories present in this sequence.
    """

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        ground_truth: np.ndarray,
        attributes: List[ChallengeAttribute],
    ) -> None:
        self._frames = frames
        self.attributes: List[ChallengeAttribute] = attributes
        super().__init__(
            name=name,
            frame_paths=["<memory>"] * len(frames),
            ground_truth=ground_truth,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames

    def __repr__(self) -> str:
        attrs = ", ".join(a.label() for a in self.attributes)
        return f"AttributedSequence(name={self.name!r}, frames={len(self)}, attrs=[{attrs}])"


# ---------------------------------------------------------------------------
# Generator helpers — one per challenge type
# ---------------------------------------------------------------------------

def _build_occlusion_sequence(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    name: str,
) -> AttributedSequence:
    """Sequence with temporary target occlusion in the middle third of frames."""
    W, H = frame_size
    bw, bh = bbox_size
    occ_start = num_frames // 3
    occ_end = 2 * num_frames // 3

    # Slow linear motion so the challenge is the occlusion, not motion
    cx = float(rng.integers(bw, W - bw))
    cy = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.3, 1.0)) * float(rng.choice([-1, 1]))

    half_bw, half_bh = bw / 2.0, bh / 2.0
    colour = tuple(int(c) for c in rng.integers(160, 256, 3))
    occ_colour = tuple(int(c) for c in rng.integers(80, 150, 3))  # occluder colour
    background = rng.integers(40, 80, (H, W, 3), dtype=np.uint8)

    occ_w, occ_h = bw + 10, bh + 10  # occluder slightly larger than target

    frames: List[np.ndarray] = []
    gt_boxes: List[Tuple[float, float, float, float]] = []

    for i in range(num_frames):
        frame = background.copy()
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        # Draw target
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
        frame[y1c:y2c, x1c:x2c] = colour
        # Draw occluder during occlusion window
        if occ_start <= i < occ_end:
            ox1 = max(0, x1c - 5)
            oy1 = max(0, y1c - 5)
            ox2 = min(W, ox1 + occ_w)
            oy2 = min(H, oy1 + occ_h)
            frame[oy1:oy2, ox1:ox2] = occ_colour
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))
        # Update position
        cx += vx
        cy += vy
        if cx < half_bw or cx > W - half_bw:
            vx = -vx
            cx = float(np.clip(cx, half_bw, W - half_bw))
        if cy < half_bh or cy > H - half_bh:
            vy = -vy
            cy = float(np.clip(cy, half_bh, H - half_bh))

    return AttributedSequence(
        name=name,
        frames=frames,
        ground_truth=np.array(gt_boxes, dtype=np.float64),
        attributes=[ChallengeAttribute.OCCLUSION],
    )


def _build_scale_change_sequence(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    name: str,
) -> AttributedSequence:
    """Sequence where the target box grows from 0.5× to 2× base size."""
    W, H = frame_size
    base_bw, base_bh = bbox_size
    cx = W / 2.0
    cy = H / 2.0
    colour = tuple(int(c) for c in rng.integers(160, 256, 3))
    background = rng.integers(40, 80, (H, W, 3), dtype=np.uint8)

    frames: List[np.ndarray] = []
    gt_boxes: List[Tuple[float, float, float, float]] = []

    for i in range(num_frames):
        # Scale oscillates: 0.5× → 2× → 0.5× using a cosine curve
        t = i / max(num_frames - 1, 1)
        scale = 0.5 + 1.5 * (1 - math.cos(2 * math.pi * t)) / 2.0  # in [0.5, 2.0]
        bw = max(4, int(round(base_bw * scale)))
        bh = max(4, int(round(base_bh * scale)))
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        frame = background.copy()
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
        frame[y1c:y2c, x1c:x2c] = colour
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))

    return AttributedSequence(
        name=name,
        frames=frames,
        ground_truth=np.array(gt_boxes, dtype=np.float64),
        attributes=[ChallengeAttribute.SCALE_CHANGE],
    )


def _build_fast_motion_sequence(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    name: str,
) -> AttributedSequence:
    """Sequence where per-frame displacement is 1.5–2× the target width."""
    W, H = frame_size
    bw, bh = bbox_size
    half_bw, half_bh = bw / 2.0, bh / 2.0

    cx = float(rng.integers(bw, W - bw))
    cy = float(rng.integers(bh, H - bh))
    # Speed ≈ 1.5× target width per frame
    speed = bw * 1.5
    angle = float(rng.uniform(0, 2 * math.pi))
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)

    colour = tuple(int(c) for c in rng.integers(160, 256, 3))
    background = rng.integers(40, 80, (H, W, 3), dtype=np.uint8)

    frames: List[np.ndarray] = []
    gt_boxes: List[Tuple[float, float, float, float]] = []

    for _ in range(num_frames):
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        frame = background.copy()
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
        frame[y1c:y2c, x1c:x2c] = colour
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))
        cx += vx
        cy += vy
        if cx < half_bw or cx > W - half_bw:
            vx = -vx
            cx = float(np.clip(cx, half_bw, W - half_bw))
        if cy < half_bh or cy > H - half_bh:
            vy = -vy
            cy = float(np.clip(cy, half_bh, H - half_bh))

    return AttributedSequence(
        name=name,
        frames=frames,
        ground_truth=np.array(gt_boxes, dtype=np.float64),
        attributes=[ChallengeAttribute.FAST_MOTION],
    )


def _build_illumination_change_sequence(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    name: str,
) -> AttributedSequence:
    """Sequence with sinusoidal frame-wide brightness change."""
    W, H = frame_size
    bw, bh = bbox_size
    half_bw, half_bh = bw / 2.0, bh / 2.0

    cx = float(rng.integers(bw, W - bw))
    cy = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.3, 1.0)) * float(rng.choice([-1, 1]))

    colour = tuple(int(c) for c in rng.integers(160, 256, 3))
    background = rng.integers(60, 100, (H, W, 3), dtype=np.uint8)

    frames: List[np.ndarray] = []
    gt_boxes: List[Tuple[float, float, float, float]] = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        # Brightness factor oscillates between 0.2× (dark) and 1.0× (bright)
        brightness = 0.2 + 0.8 * (1 - math.cos(2 * math.pi * t)) / 2.0
        frame = (background.astype(np.float32) * brightness).astype(np.uint8)
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
        tgt = tuple(int(c * brightness) for c in colour)
        frame[y1c:y2c, x1c:x2c] = tgt
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))
        cx += vx
        cy += vy
        if cx < half_bw or cx > W - half_bw:
            vx = -vx
            cx = float(np.clip(cx, half_bw, W - half_bw))
        if cy < half_bh or cy > H - half_bh:
            vy = -vy
            cy = float(np.clip(cy, half_bh, H - half_bh))

    return AttributedSequence(
        name=name,
        frames=frames,
        ground_truth=np.array(gt_boxes, dtype=np.float64),
        attributes=[ChallengeAttribute.ILLUMINATION_CHANGE],
    )


def _build_low_resolution_sequence(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    _bbox_size: Tuple[int, int],
    name: str,
) -> AttributedSequence:
    """Sequence with a tiny 10×10 px target — stresses low-resolution tracking."""
    W, H = frame_size
    bw, bh = 10, 10  # force small target regardless of bbox_size
    half_bw, half_bh = bw / 2.0, bh / 2.0

    cx = float(rng.integers(bw, W - bw))
    cy = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(0.5, 1.0)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.3, 0.8)) * float(rng.choice([-1, 1]))

    colour = tuple(int(c) for c in rng.integers(160, 256, 3))
    background = rng.integers(40, 80, (H, W, 3), dtype=np.uint8)

    frames: List[np.ndarray] = []
    gt_boxes: List[Tuple[float, float, float, float]] = []

    for _ in range(num_frames):
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        frame = background.copy()
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
        frame[y1c:y2c, x1c:x2c] = colour
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))
        cx += vx
        cy += vy
        if cx < half_bw or cx > W - half_bw:
            vx = -vx
            cx = float(np.clip(cx, half_bw, W - half_bw))
        if cy < half_bh or cy > H - half_bh:
            vy = -vy
            cy = float(np.clip(cy, half_bh, H - half_bh))

    return AttributedSequence(
        name=name,
        frames=frames,
        ground_truth=np.array(gt_boxes, dtype=np.float64),
        attributes=[ChallengeAttribute.LOW_RESOLUTION],
    )


def _build_background_clutter_sequence(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    name: str,
    n_distractors: int = 5,
) -> AttributedSequence:
    """Sequence with colour-similar distractor rectangles in the background."""
    W, H = frame_size
    bw, bh = bbox_size
    half_bw, half_bh = bw / 2.0, bh / 2.0

    cx = float(rng.integers(bw, W - bw))
    cy = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(0.5, 1.5)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.3, 1.0)) * float(rng.choice([-1, 1]))

    # Target colour
    colour = np.array([int(c) for c in rng.integers(150, 220, 3)], dtype=np.uint8)
    background = rng.integers(30, 70, (H, W, 3), dtype=np.uint8)

    # Distractor colours: similar hue to target
    distractors = []
    for _ in range(n_distractors):
        noise = rng.integers(-30, 30, 3)
        dist_colour = tuple(int(np.clip(int(c) + int(n), 0, 255)) for c, n in zip(colour, noise))
        dist_x = int(rng.integers(0, W - bw))
        dist_y = int(rng.integers(0, H - bh))
        distractors.append((dist_colour, dist_x, dist_y))

    frames: List[np.ndarray] = []
    gt_boxes: List[Tuple[float, float, float, float]] = []

    for _ in range(num_frames):
        frame = background.copy()
        # Draw distractors first (behind target)
        for (dc, dx, dy) in distractors:
            dxe, dye = min(W, dx + bw), min(H, dy + bh)
            frame[dy:dye, dx:dxe] = dc
        # Draw target on top
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x1 + bw), min(H, y1 + bh)
        frame[y1c:y2c, x1c:x2c] = tuple(int(c) for c in colour)
        frames.append(frame)
        gt_boxes.append((float(x1), float(y1), float(bw), float(bh)))
        cx += vx
        cy += vy
        if cx < half_bw or cx > W - half_bw:
            vx = -vx
            cx = float(np.clip(cx, half_bw, W - half_bw))
        if cy < half_bh or cy > H - half_bh:
            vy = -vy
            cy = float(np.clip(cy, half_bh, H - half_bh))

    return AttributedSequence(
        name=name,
        frames=frames,
        ground_truth=np.array(gt_boxes, dtype=np.float64),
        attributes=[ChallengeAttribute.BACKGROUND_CLUTTER],
    )


# ---------------------------------------------------------------------------
# Map challenge → generator function
# ---------------------------------------------------------------------------

_GENERATORS = {
    ChallengeAttribute.OCCLUSION: _build_occlusion_sequence,
    ChallengeAttribute.SCALE_CHANGE: _build_scale_change_sequence,
    ChallengeAttribute.FAST_MOTION: _build_fast_motion_sequence,
    ChallengeAttribute.ILLUMINATION_CHANGE: _build_illumination_change_sequence,
    ChallengeAttribute.LOW_RESOLUTION: _build_low_resolution_sequence,
    ChallengeAttribute.BACKGROUND_CLUTTER: _build_background_clutter_sequence,
}


# ---------------------------------------------------------------------------
# ChallengeDataset
# ---------------------------------------------------------------------------

class ChallengeDataset(BaseDataset):
    """A synthetic dataset where each sequence exercises a specific tracking challenge.

    Sequences are distributed round-robin across the requested challenge types
    so that every challenge is represented (roughly) equally.

    Args:
        num_sequences: Total number of sequences.  Default: ``12``.
        num_frames: Frames per sequence.  Default: ``100``.
        frame_size: ``(width, height)`` in pixels.  Default: ``(320, 240)``.
        bbox_size: ``(width, height)`` of the target rectangle.
            Default: ``(40, 40)``.
        challenges: List of :class:`ChallengeAttribute` values to include.
            Defaults to all six challenge types.
        seed: Base RNG seed for reproducibility.  Default: ``42``.

    Example::

        ds = ChallengeDataset(
            num_sequences=6,
            num_frames=80,
            challenges=[ChallengeAttribute.OCCLUSION, ChallengeAttribute.FAST_MOTION],
        )
        print(len(ds))           # 6
        print(ds[0].attributes)  # [ChallengeAttribute.OCCLUSION]
        print(ds[1].attributes)  # [ChallengeAttribute.FAST_MOTION]
    """

    def __init__(
        self,
        num_sequences: int = 12,
        num_frames: int = 100,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        challenges: Optional[List[ChallengeAttribute]] = None,
        seed: int = 42,
    ) -> None:
        if challenges is None:
            challenges = list(ChallengeAttribute)
        if not challenges:
            raise ValueError("challenges list must not be empty.")
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.bbox_size = bbox_size
        self.challenges = challenges
        self.seed = seed
        self._cache: List[Optional[AttributedSequence]] = [None] * num_sequences

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> AttributedSequence:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Index {idx} out of range [0, {self.num_sequences})"
            )
        if self._cache[idx] is None:
            self._cache[idx] = self._build(idx)
        return self._cache[idx]  # type: ignore[return-value]

    def __repr__(self) -> str:
        challenge_names = ", ".join(c.label() for c in self.challenges)
        return (
            f"ChallengeDataset(sequences={self.num_sequences}, "
            f"frames={self.num_frames}, "
            f"challenges=[{challenge_names}])"
        )

    def attribute_index(self) -> Dict[ChallengeAttribute, List[int]]:
        """Return mapping from challenge attribute to sequence indices.

        Builds the full dataset on first access.  Used by
        :class:`~eovot.metrics.attributes.AttributeMetricsAggregator`.

        Returns:
            Dict mapping each :class:`ChallengeAttribute` to the list of
            sequence indices that carry that attribute.
        """
        idx: Dict[ChallengeAttribute, List[int]] = {c: [] for c in self.challenges}
        for i in range(self.num_sequences):
            seq = self[i]
            for attr in seq.attributes:
                if attr in idx:
                    idx[attr].append(i)
        return idx

    def _build(self, idx: int) -> AttributedSequence:
        """Generate sequence *idx* using the round-robin challenge assignment."""
        challenge = self.challenges[idx % len(self.challenges)]
        generator = _GENERATORS[challenge]
        rng = np.random.default_rng(self.seed + idx)
        name = f"challenge_{challenge.label()}_{idx:03d}"
        return generator(rng, self.num_frames, self.frame_size, self.bbox_size, name)
