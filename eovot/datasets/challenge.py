"""VOT-standard challenge scenario dataset for EOVOT evaluation.

Extends :class:`~eovot.datasets.synthetic.SyntheticDataset` with six
challenge attributes drawn from the official VOT challenge taxonomy:

* **occlusion** — target passes behind a static occluder; frames where
  the target is hidden are tagged with attribute ``"OCC"``.
* **scale_variation** — target dimensions oscillate sinusoidally (±50 %).
  Tagged ``"SV"``.
* **fast_motion** — target velocity 3–5× normal.  Tagged ``"FM"``.
* **illumination_variation** — per-frame global brightness shift and
  per-frame contrast scaling, simulating lighting changes.  Tagged ``"IV"``.
* **distractors** — N additional coloured rectangles move around the
  scene; their colour is deliberately similar to the target, forcing
  the tracker to resolve ambiguity.  Tagged ``"DIS"``.
* **motion_blur** — artificial Gaussian blur applied proportional to
  target velocity, mimicking camera motion.  Tagged ``"MB"``.

Each generated :class:`~eovot.datasets.base.Sequence` exposes:

* ``sequence.attributes`` — a :class:`set` of active challenge tags
  (``{"OCC"}``, ``{"SV", "FM"}``, etc.)
* ``sequence.per_frame_attributes`` — a list of :class:`frozenset` of
  per-frame tags; frame *i* is challenging if its set is non-empty.

This enables attribute-sliced evaluation::

    from eovot.datasets.challenge import ChallengeDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    ds = ChallengeDataset(challenge="occlusion", num_sequences=5)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="synth-occlusion")

    # Attribute-sliced mIoU
    sliced = result.attribute_sliced_iou()
    print(sliced)  # {"OCC": 0.18, "all": 0.61}

No external data required — all frames are rendered in memory.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np

from .base import BaseDataset, Sequence


# ---------------------------------------------------------------------------
# Challenge-aware Sequence
# ---------------------------------------------------------------------------

class ChallengeSequence(Sequence):
    """A :class:`~.base.Sequence` augmented with VOT challenge attribute tags.

    Attributes:
        attributes: Set of challenge tags active for this sequence.
        per_frame_attributes: Per-frame tag sets; empty frozenset on
            frames with no active challenge.
    """

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        ground_truth: np.ndarray,
        attributes: Set[str],
        per_frame_attributes: List[FrozenSet[str]],
    ) -> None:
        self._frames = frames
        self.attributes: Set[str] = attributes
        self.per_frame_attributes: List[FrozenSet[str]] = per_frame_attributes
        super().__init__(
            name=name,
            frame_paths=["<memory>"] * len(frames),
            ground_truth=ground_truth,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames

    def challenging_frame_indices(self) -> List[int]:
        """Return indices of frames with at least one active challenge tag."""
        return [i for i, tags in enumerate(self.per_frame_attributes) if tags]

    def iou_by_attribute(
        self, ious: np.ndarray, attribute: str
    ) -> Optional[float]:
        """Mean IoU restricted to frames tagged with *attribute*.

        Args:
            ious: Per-frame IoU array from the benchmark engine.
            attribute: Challenge tag, e.g. ``"OCC"``.

        Returns:
            Mean IoU on tagged frames, or ``None`` if no tagged frames exist.
        """
        tagged = [
            i for i, tags in enumerate(self.per_frame_attributes)
            if attribute in tags and i < len(ious)
        ]
        if not tagged:
            return None
        return float(np.mean([ious[i] for i in tagged]))


# ---------------------------------------------------------------------------
# Challenge generators (internal)
# ---------------------------------------------------------------------------

_VALID_CHALLENGES = frozenset([
    "occlusion",
    "scale_variation",
    "fast_motion",
    "illumination_variation",
    "distractors",
    "motion_blur",
])

# Canonical VOT tag abbreviations
_TAG: Dict[str, str] = {
    "occlusion": "OCC",
    "scale_variation": "SV",
    "fast_motion": "FM",
    "illumination_variation": "IV",
    "distractors": "DIS",
    "motion_blur": "MB",
}


def _linear_positions(
    cx0: float, cy0: float,
    vx: float, vy: float,
    num_frames: int,
    W: int, H: int,
    half_bw: float, half_bh: float,
) -> List[Tuple[float, float]]:
    """Bounce-constrained linear motion trajectory."""
    positions = []
    cx, cy = cx0, cy0
    for _ in range(num_frames):
        positions.append((cx, cy))
        cx += vx
        cy += vy
        if cx < half_bw or cx > W - half_bw:
            vx = -vx
            cx = float(np.clip(cx, half_bw, W - half_bw))
        if cy < half_bh or cy > H - half_bh:
            vy = -vy
            cy = float(np.clip(cy, half_bh, H - half_bh))
    return positions


def _build_background(rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    """Low-frequency Perlin-like background — more realistic than pure noise."""
    bg = rng.integers(30, 90, (H, W, 3), dtype=np.uint8)
    # Add large-scale gradient blob
    cx, cy = rng.integers(W // 4, 3 * W // 4), rng.integers(H // 4, 3 * H // 4)
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    blob = np.clip((1.0 - dist / max(W, H)) * 80, 0, 60).astype(np.uint8)
    bg[:, :, 0] = np.clip(bg[:, :, 0].astype(int) + blob, 0, 255).astype(np.uint8)
    return bg


# ---------------------------------------------------------------------------
# Per-challenge frame generators
# ---------------------------------------------------------------------------

def _gen_occlusion(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[FrozenSet[str]]]:
    """Generate frames with an occluder rectangle."""
    W, H = frame_size
    bw, bh = bbox_size

    cx0 = float(rng.integers(bw, W - bw))
    cy0 = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(1.5, 3.0)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(1.0, 2.0)) * float(rng.choice([-1, 1]))
    positions = _linear_positions(cx0, cy0, vx, vy, num_frames, W, H, bw / 2, bh / 2)

    # Static occluder: 40% of frame width, vertical bar on a random side
    occ_w = int(W * 0.30)
    occ_x = int(rng.integers(W // 3, W - occ_w))
    occ_col = tuple(int(c) for c in rng.integers(20, 70, 3))
    background = _build_background(rng, H, W)
    target_col = tuple(int(c) for c in rng.integers(160, 256, 3))

    frames, gt, pfa = [], [], []
    for i, (cx, cy) in enumerate(positions):
        frame = background.copy()
        # Occluder layer (drawn first so target overlaps unless behind it)
        frame[:, occ_x: occ_x + occ_w] = occ_col
        # Determine if target is "behind" the occluder (centre inside occluder)
        is_occluded = occ_x <= cx <= occ_x + occ_w
        if not is_occluded:
            x1 = max(0, int(round(cx - bw / 2)))
            y1 = max(0, int(round(cy - bh / 2)))
            frame[y1: y1 + bh, x1: x1 + bw] = target_col
        gt_x, gt_y = cx - bw / 2, cy - bh / 2
        gt.append((gt_x, gt_y, float(bw), float(bh)))
        pfa.append(frozenset({"OCC"}) if is_occluded else frozenset())
        frames.append(frame)

    return frames, np.array(gt, dtype=np.float64), pfa


def _gen_scale_variation(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[FrozenSet[str]]]:
    """Generate frames where the target box oscillates in size."""
    W, H = frame_size
    bw0, bh0 = bbox_size

    cx0 = float(rng.integers(bw0, W - bw0))
    cy0 = float(rng.integers(bh0, H - bh0))
    vx = float(rng.uniform(1.0, 2.5)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.8, 1.8)) * float(rng.choice([-1, 1]))
    positions = _linear_positions(cx0, cy0, vx, vy, num_frames, W, H, bw0, bh0)

    background = _build_background(rng, H, W)
    target_col = tuple(int(c) for c in rng.integers(160, 256, 3))
    # Scale oscillates ±50 % with period = num_frames // 3
    period = max(num_frames // 3, 10)

    frames, gt, pfa = [], [], []
    for i, (cx, cy) in enumerate(positions):
        scale = 1.0 + 0.5 * math.sin(2 * math.pi * i / period)
        bw = max(8, int(round(bw0 * scale)))
        bh = max(8, int(round(bh0 * scale)))
        cx_c = float(np.clip(cx, bw / 2, W - bw / 2))
        cy_c = float(np.clip(cy, bh / 2, H - bh / 2))
        frame = background.copy()
        x1 = max(0, int(round(cx_c - bw / 2)))
        y1 = max(0, int(round(cy_c - bh / 2)))
        frame[y1: y1 + bh, x1: x1 + bw] = target_col
        gt.append((cx_c - bw / 2, cy_c - bh / 2, float(bw), float(bh)))
        changing = abs(scale - 1.0) > 0.15
        pfa.append(frozenset({"SV"}) if changing else frozenset())
        frames.append(frame)

    return frames, np.array(gt, dtype=np.float64), pfa


def _gen_fast_motion(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[FrozenSet[str]]]:
    """Generate frames where the target moves at 3–5× normal speed."""
    W, H = frame_size
    bw, bh = bbox_size

    cx0 = float(rng.integers(bw, W - bw))
    cy0 = float(rng.integers(bh, H - bh))
    # Fast velocity: 5–10 px/frame
    speed = float(rng.uniform(5.0, 10.0))
    angle = float(rng.uniform(0, 2 * math.pi))
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    positions = _linear_positions(cx0, cy0, vx, vy, num_frames, W, H, bw / 2, bh / 2)

    background = _build_background(rng, H, W)
    target_col = tuple(int(c) for c in rng.integers(160, 256, 3))

    frames, gt, pfa = [], [], []
    prev_cx, prev_cy = positions[0]
    for i, (cx, cy) in enumerate(positions):
        speed_now = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) if i > 0 else 0.0
        frame = background.copy()
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        frame[y1: y1 + bh, x1: x1 + bw] = target_col
        gt.append((cx - bw / 2, cy - bh / 2, float(bw), float(bh)))
        pfa.append(frozenset({"FM"}) if speed_now > 4.0 else frozenset())
        frames.append(frame)
        prev_cx, prev_cy = cx, cy

    return frames, np.array(gt, dtype=np.float64), pfa


def _gen_illumination_variation(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[FrozenSet[str]]]:
    """Generate frames with global illumination shifts."""
    W, H = frame_size
    bw, bh = bbox_size

    cx0 = float(rng.integers(bw, W - bw))
    cy0 = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(1.0, 2.5)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.8, 1.5)) * float(rng.choice([-1, 1]))
    positions = _linear_positions(cx0, cy0, vx, vy, num_frames, W, H, bw / 2, bh / 2)

    background = _build_background(rng, H, W)
    target_col = tuple(int(c) for c in rng.integers(160, 256, 3))
    # Brightness oscillates: slow sinusoid + high-freq flicker
    period = max(num_frames // 4, 10)

    frames, gt, pfa = [], [], []
    for i, (cx, cy) in enumerate(positions):
        bright_shift = int(60 * math.sin(2 * math.pi * i / period))
        contrast = 0.7 + 0.5 * abs(math.sin(math.pi * i / period))
        frame = background.copy().astype(np.float32)
        frame = frame * contrast + bright_shift
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        frame[y1: y1 + bh, x1: x1 + bw] = target_col
        gt.append((cx - bw / 2, cy - bh / 2, float(bw), float(bh)))
        is_iv = abs(bright_shift) > 20 or abs(contrast - 1.0) > 0.15
        pfa.append(frozenset({"IV"}) if is_iv else frozenset())
        frames.append(frame)

    return frames, np.array(gt, dtype=np.float64), pfa


def _gen_distractors(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    seed: int,
    n_distractors: int = 3,
) -> Tuple[List[np.ndarray], np.ndarray, List[FrozenSet[str]]]:
    """Generate frames with similar-coloured distractor objects."""
    W, H = frame_size
    bw, bh = bbox_size

    cx0 = float(rng.integers(bw, W - bw))
    cy0 = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(1.0, 2.5)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(0.8, 1.5)) * float(rng.choice([-1, 1]))
    positions = _linear_positions(cx0, cy0, vx, vy, num_frames, W, H, bw / 2, bh / 2)

    background = _build_background(rng, H, W)
    # Target colour — a distinctive hue
    base_hue = rng.integers(160, 220, 3)
    target_col = tuple(int(c) for c in base_hue)

    # Distractor colours: perturbed versions of target colour
    distractor_cols = []
    for _ in range(n_distractors):
        perturb = rng.integers(-30, 30, 3)
        col = np.clip(base_hue + perturb, 80, 255)
        distractor_cols.append(tuple(int(c) for c in col))

    # Distractor initial positions and velocities
    dist_cx = [float(rng.integers(bw, W - bw)) for _ in range(n_distractors)]
    dist_cy = [float(rng.integers(bh, H - bh)) for _ in range(n_distractors)]
    dist_vx = [float(rng.uniform(1.0, 3.0)) * float(rng.choice([-1, 1])) for _ in range(n_distractors)]
    dist_vy = [float(rng.uniform(0.8, 2.5)) * float(rng.choice([-1, 1])) for _ in range(n_distractors)]

    frames, gt, pfa = [], [], []
    for i, (cx, cy) in enumerate(positions):
        frame = background.copy()
        # Draw distractors first
        for j in range(n_distractors):
            dx = max(0, int(round(dist_cx[j] - bw / 2)))
            dy = max(0, int(round(dist_cy[j] - bh / 2)))
            frame[dy: dy + bh, dx: dx + bw] = distractor_cols[j]
            # Move distractor
            dist_cx[j] += dist_vx[j]
            dist_cy[j] += dist_vy[j]
            if dist_cx[j] < bw / 2 or dist_cx[j] > W - bw / 2:
                dist_vx[j] = -dist_vx[j]
                dist_cx[j] = float(np.clip(dist_cx[j], bw / 2, W - bw / 2))
            if dist_cy[j] < bh / 2 or dist_cy[j] > H - bh / 2:
                dist_vy[j] = -dist_vy[j]
                dist_cy[j] = float(np.clip(dist_cy[j], bh / 2, H - bh / 2))

        # Draw target on top
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        frame[y1: y1 + bh, x1: x1 + bw] = target_col
        gt.append((cx - bw / 2, cy - bh / 2, float(bw), float(bh)))
        # Check proximity: is any distractor within 2× target bbox of the target?
        min_dist = min(
            math.sqrt((dist_cx[j] - cx) ** 2 + (dist_cy[j] - cy) ** 2)
            for j in range(n_distractors)
        )
        near_distractor = min_dist < max(bw, bh) * 2.5
        pfa.append(frozenset({"DIS"}) if near_distractor else frozenset())
        frames.append(frame)

    return frames, np.array(gt, dtype=np.float64), pfa


def _gen_motion_blur(
    rng: np.random.Generator,
    num_frames: int,
    frame_size: Tuple[int, int],
    bbox_size: Tuple[int, int],
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray, List[FrozenSet[str]]]:
    """Generate frames with velocity-proportional Gaussian blur."""
    W, H = frame_size
    bw, bh = bbox_size

    cx0 = float(rng.integers(bw, W - bw))
    cy0 = float(rng.integers(bh, H - bh))
    vx = float(rng.uniform(2.0, 5.0)) * float(rng.choice([-1, 1]))
    vy = float(rng.uniform(1.5, 4.0)) * float(rng.choice([-1, 1]))
    positions = _linear_positions(cx0, cy0, vx, vy, num_frames, W, H, bw / 2, bh / 2)

    background = _build_background(rng, H, W)
    target_col = tuple(int(c) for c in rng.integers(160, 256, 3))

    frames, gt, pfa = [], [], []
    prev_cx, prev_cy = positions[0]
    for i, (cx, cy) in enumerate(positions):
        speed = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) if i > 0 else 0.0
        frame = background.copy()
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        frame[y1: y1 + bh, x1: x1 + bw] = target_col
        # Apply blur proportional to speed
        if speed > 2.0:
            k = min(int(speed * 1.5) | 1, 15)  # odd kernel, max 15
            k = k + 1 if k % 2 == 0 else k
            frame = cv2.GaussianBlur(frame, (k, k), 0)
        gt.append((cx - bw / 2, cy - bh / 2, float(bw), float(bh)))
        pfa.append(frozenset({"MB"}) if speed > 3.0 else frozenset())
        frames.append(frame)
        prev_cx, prev_cy = cx, cy

    return frames, np.array(gt, dtype=np.float64), pfa


_CHALLENGE_GENERATORS = {
    "occlusion": _gen_occlusion,
    "scale_variation": _gen_scale_variation,
    "fast_motion": _gen_fast_motion,
    "illumination_variation": _gen_illumination_variation,
    "distractors": _gen_distractors,
    "motion_blur": _gen_motion_blur,
}


# ---------------------------------------------------------------------------
# ChallengeDataset
# ---------------------------------------------------------------------------

class ChallengeDataset(BaseDataset):
    """Synthetic dataset with VOT-standard challenge scenarios.

    Each sequence is generated procedurally with a specific challenge pattern,
    tagged with the appropriate attribute abbreviations so downstream analysis
    can slice metrics by challenge type.

    Args:
        challenge: One of ``"occlusion"``, ``"scale_variation"``,
            ``"fast_motion"``, ``"illumination_variation"``,
            ``"distractors"``, ``"motion_blur"``.
        num_sequences: Number of sequences to generate.  Default: ``10``.
        num_frames: Frames per sequence.  Default: ``150``.
        frame_size: ``(width, height)`` in pixels.  Default: ``(320, 240)``.
        bbox_size: ``(width, height)`` of the target.  Default: ``(40, 40)``.
        seed: Base RNG seed; sequence *i* uses ``seed + i``.  Default: ``42``.
        n_distractors: Number of distractor objects for the ``"distractors"``
            challenge.  Ignored for other challenges.  Default: ``3``.

    Raises:
        ValueError: If *challenge* is not a recognised challenge type.

    Example::

        ds = ChallengeDataset("occlusion", num_sequences=3, num_frames=100)
        for seq in ds:
            print(seq.name, seq.attributes)
            # synth_occlusion_000  {'OCC'}

        # Attribute-sliced IoU after running the benchmark engine:
        # for each sequence result, call seq.iou_by_attribute(ious, "OCC")
    """

    def __init__(
        self,
        challenge: str,
        num_sequences: int = 10,
        num_frames: int = 150,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        seed: int = 42,
        n_distractors: int = 3,
    ) -> None:
        if challenge not in _VALID_CHALLENGES:
            raise ValueError(
                f"Unknown challenge '{challenge}'. "
                f"Valid choices: {sorted(_VALID_CHALLENGES)}"
            )
        self.challenge = challenge
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.bbox_size = bbox_size
        self.seed = seed
        self.n_distractors = n_distractors
        self._cache: List[Optional[ChallengeSequence]] = [None] * num_sequences

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> ChallengeSequence:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Sequence index {idx} out of range [0, {self.num_sequences})"
            )
        if self._cache[idx] is None:
            self._cache[idx] = self._build_sequence(idx)
        return self._cache[idx]  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"ChallengeDataset(challenge={self.challenge!r}, "
            f"sequences={self.num_sequences}, frames={self.num_frames}, "
            f"frame_size={self.frame_size})"
        )

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> ChallengeSequence:
        rng = np.random.default_rng(self.seed + idx)
        gen = _CHALLENGE_GENERATORS[self.challenge]

        if self.challenge == "distractors":
            frames, gt, pfa = gen(
                rng, self.num_frames, self.frame_size, self.bbox_size,
                self.seed + idx, self.n_distractors,
            )
        else:
            frames, gt, pfa = gen(
                rng, self.num_frames, self.frame_size, self.bbox_size,
                self.seed + idx,
            )

        tag = _TAG[self.challenge]
        # Sequence-level attribute: the challenge tag is always present
        attrs: Set[str] = {tag}

        name = f"synth_{self.challenge.replace('_', '-')}_{idx:03d}"
        return ChallengeSequence(
            name=name,
            frames=frames,
            ground_truth=gt,
            attributes=attrs,
            per_frame_attributes=pfa,
        )

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def all_challenges(cls) -> List[str]:
        """Return the sorted list of supported challenge names."""
        return sorted(_VALID_CHALLENGES)

    @classmethod
    def tag_for(cls, challenge: str) -> str:
        """Return the VOT tag abbreviation for *challenge*.

        Args:
            challenge: Challenge name (e.g. ``"occlusion"``).

        Returns:
            VOT tag (e.g. ``"OCC"``).

        Raises:
            KeyError: If *challenge* is not recognised.
        """
        return _TAG[challenge]
