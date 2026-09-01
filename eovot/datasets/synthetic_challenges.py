"""Synthetic tracking sequences with controlled VOT-attribute challenges.

Extends :class:`~eovot.datasets.synthetic.SyntheticDataset` with
programmatically-generated sequences that isolate specific visual
tracking challenges from the standard VOT attribute taxonomy:

* **Occlusion** — a second solid rectangle periodically moves over the
  target, simulating partial or full occlusion events.
* **Illumination change** — sinusoidal brightness modulation that dims
  and brightens the entire frame, stressing appearance-model updates.
* **Scale change** — the target rectangle grows and shrinks between a
  minimum and maximum factor, testing scale-adaptive tracking.
* **Background clutter** — multiple distractor rectangles of similar
  color to the target move around the frame, creating false positives.

These sequences are invaluable for:

* **CI testing** without downloading real datasets.
* **Ablation studies** — isolate which challenge type breaks a tracker.
* **Reproducible regression testing** — fixed seed ensures identical
  sequences between runs.
* **Failure mode documentation** for papers and reports.

Usage::

    from eovot.datasets.synthetic_challenges import (
        ChallengeSyntheticDataset,
        ChallengeType,
    )
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.kcf import KCFTracker

    for challenge in ChallengeType:
        ds = ChallengeSyntheticDataset(
            challenge=challenge,
            num_sequences=5,
            num_frames=200,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), ds, dataset_name=f"Synth-{challenge.value}")
        print(f"{challenge.value:25s}  mIoU={result.mean_iou:.4f}")
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, Sequence

# Bounding box: (x, y, width, height)
_BBox = Tuple[float, float, float, float]


class ChallengeType(str, Enum):
    """Supported tracking challenge types."""

    OCCLUSION = "occlusion"
    ILLUMINATION_CHANGE = "illumination_change"
    SCALE_CHANGE = "scale_change"
    BACKGROUND_CLUTTER = "background_clutter"


class _ChallengeSequence(Sequence):
    """In-memory sequence whose frames are pre-rendered numpy arrays."""

    def __init__(
        self,
        name: str,
        frames: List[np.ndarray],
        ground_truth: np.ndarray,
    ) -> None:
        self._frames = frames
        super().__init__(
            name=name,
            frame_paths=["<memory>"] * len(frames),
            ground_truth=ground_truth,
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames


class ChallengeSyntheticDataset(BaseDataset):
    """Dataset of synthetic tracking sequences with a single controlled challenge.

    Each sequence renders a brightly-coloured target rectangle moving in a
    linear bounce pattern against a textured noise background.  A specific
    challenge (occlusion, illumination change, scale change, or background
    clutter) is then layered on top.

    Args:
        challenge: The :class:`ChallengeType` to apply to every sequence.
        num_sequences: Number of sequences to generate.  Default: ``10``.
        num_frames: Frames per sequence.  Default: ``150``.
        frame_size: ``(width, height)`` of each frame in pixels.
            Default: ``(320, 240)``.
        bbox_size: ``(width, height)`` of the target rectangle in pixels.
            Default: ``(40, 40)``.
        seed: Base RNG seed; sequence *i* uses ``seed + i``.  Default: ``42``.
        occlusion_period: For OCCLUSION — frames between occlusion events.
            Default: ``30``.
        occlusion_duration: For OCCLUSION — frames the occluder covers the
            target.  Default: ``15``.
        illumination_amplitude: For ILLUMINATION_CHANGE — peak brightness
            swing in [0, 1]; 0.4 means ±40% of baseline.  Default: ``0.4``.
        scale_min: For SCALE_CHANGE — minimum scale factor.  Default: ``0.5``.
        scale_max: For SCALE_CHANGE — maximum scale factor.  Default: ``2.0``.
        num_distractors: For BACKGROUND_CLUTTER — number of distractor
            rectangles.  Default: ``5``.

    Example::

        ds = ChallengeSyntheticDataset(
            challenge=ChallengeType.OCCLUSION,
            num_sequences=5,
            num_frames=100,
        )
        for seq in ds:
            print(seq.name, seq.ground_truth.shape)
    """

    def __init__(
        self,
        challenge: ChallengeType = ChallengeType.OCCLUSION,
        num_sequences: int = 10,
        num_frames: int = 150,
        frame_size: Tuple[int, int] = (320, 240),
        bbox_size: Tuple[int, int] = (40, 40),
        seed: int = 42,
        # Challenge-specific knobs
        occlusion_period: int = 30,
        occlusion_duration: int = 15,
        illumination_amplitude: float = 0.4,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        num_distractors: int = 5,
    ) -> None:
        if not isinstance(challenge, ChallengeType):
            challenge = ChallengeType(challenge)
        self.challenge = challenge
        self.num_sequences = num_sequences
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.bbox_size = bbox_size
        self.seed = seed
        self.occlusion_period = occlusion_period
        self.occlusion_duration = occlusion_duration
        self.illumination_amplitude = illumination_amplitude
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.num_distractors = num_distractors
        self._cache: List[Optional[_ChallengeSequence]] = [None] * num_sequences

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Sequence:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Sequence index {idx} out of range [0, {self.num_sequences})"
            )
        if self._cache[idx] is None:
            self._cache[idx] = self._build_sequence(idx)
        return self._cache[idx]  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"ChallengeSyntheticDataset(challenge={self.challenge.value!r}, "
            f"sequences={self.num_sequences}, frames={self.num_frames}, "
            f"frame_size={self.frame_size})"
        )

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _build_sequence(self, idx: int) -> _ChallengeSequence:
        rng = np.random.default_rng(self.seed + idx)
        W, H = self.frame_size
        bw, bh = self.bbox_size
        half_bw, half_bh = bw / 2.0, bh / 2.0

        # Random initial centre.
        cx0 = float(rng.integers(bw, W - bw))
        cy0 = float(rng.integers(bh, H - bh))

        # Linear bounce motion for the target.
        vx = float(rng.uniform(1.5, 3.5)) * float(rng.choice([-1, 1]))
        vy = float(rng.uniform(0.8, 2.5)) * float(rng.choice([-1, 1]))
        centres = self._bounce(cx0, cy0, vx, vy, half_bw, half_bh, W, H)

        # Static textured background.
        bg = rng.integers(40, 100, (H, W, 3), dtype=np.uint8)
        # Target colour: bright and distinct.
        colour = tuple(int(c) for c in rng.integers(160, 256, 3))

        dispatch = {
            ChallengeType.OCCLUSION: self._render_occlusion,
            ChallengeType.ILLUMINATION_CHANGE: self._render_illumination,
            ChallengeType.SCALE_CHANGE: self._render_scale_change,
            ChallengeType.BACKGROUND_CLUTTER: self._render_clutter,
        }
        frames, gt = dispatch[self.challenge](bg, colour, centres, bw, bh, rng)

        name = f"synth_{self.challenge.value}_{idx:03d}"
        return _ChallengeSequence(name=name, frames=frames, ground_truth=gt)

    def _bounce(
        self,
        cx0: float,
        cy0: float,
        vx: float,
        vy: float,
        hw: float,
        hh: float,
        W: int,
        H: int,
    ) -> List[Tuple[float, float]]:
        """Generate target centre positions with wall-bounce."""
        positions = []
        cx, cy = cx0, cy0
        for _ in range(self.num_frames):
            positions.append((cx, cy))
            cx += vx
            cy += vy
            if cx < hw or cx > W - hw:
                vx = -vx
                cx = float(np.clip(cx, hw, W - hw))
            if cy < hh or cy > H - hh:
                vy = -vy
                cy = float(np.clip(cy, hh, H - hh))
        return positions

    # ------------------------------------------------------------------
    # Challenge renderers
    # ------------------------------------------------------------------

    def _draw_target(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        bw: float,
        bh: float,
        colour: tuple,
    ) -> _BBox:
        """Draw the target rectangle and return its ground-truth (x, y, w, h)."""
        W, H = frame.shape[1], frame.shape[0]
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c = min(W, int(round(cx + bw / 2)))
        y2c = min(H, int(round(cy + bh / 2)))
        frame[y1c:y2c, x1c:x2c] = colour
        return (float(x1), float(y1), float(bw), float(bh))

    def _render_occlusion(
        self,
        bg: np.ndarray,
        colour: tuple,
        centres: List[Tuple[float, float]],
        bw: float,
        bh: float,
        rng: np.random.Generator,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Render occlusion challenge: a second rectangle periodically covers target.

        Ground-truth boxes reflect the true target position even when occluded,
        consistent with the VOT protocol (GT is always the object location, not
        the visible area).
        """
        W, H = bg.shape[1], bg.shape[0]
        # Occluder: larger rectangle of a neutral grey colour.
        occ_w, occ_h = int(bw * 1.5), int(bh * 1.5)
        occ_colour = (128, 128, 128)

        frames = []
        gt_boxes = []

        for i, (cx, cy) in enumerate(centres):
            frame = bg.copy()
            gt = self._draw_target(frame, cx, cy, bw, bh, colour)

            # Apply occluder during active windows.
            phase = i % self.occlusion_period
            if phase < self.occlusion_duration:
                ocx = float(np.clip(cx, occ_w / 2, W - occ_w / 2))
                ocy = float(np.clip(cy, occ_h / 2, H - occ_h / 2))
                ox1 = max(0, int(round(ocx - occ_w / 2)))
                oy1 = max(0, int(round(ocy - occ_h / 2)))
                ox2 = min(W, ox1 + occ_w)
                oy2 = min(H, oy1 + occ_h)
                frame[oy1:oy2, ox1:ox2] = occ_colour

            frames.append(frame)
            gt_boxes.append(gt)

        return frames, np.array(gt_boxes, dtype=np.float64)

    def _render_illumination(
        self,
        bg: np.ndarray,
        colour: tuple,
        centres: List[Tuple[float, float]],
        bw: float,
        bh: float,
        rng: np.random.Generator,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Render illumination-change challenge: sinusoidal brightness modulation.

        The entire frame (background + target) is multiplied by a brightness
        factor that oscillates between ``1 - amplitude`` and ``1 + amplitude``
        over 3 full cycles, then clipped to [0, 255].
        """
        frames = []
        gt_boxes = []
        cycles = 3
        amplitude = self.illumination_amplitude

        for i, (cx, cy) in enumerate(centres):
            frame = bg.copy()
            gt = self._draw_target(frame, cx, cy, bw, bh, colour)

            t = i / max(self.num_frames - 1, 1)
            brightness = 1.0 + amplitude * math.sin(2 * math.pi * cycles * t)
            frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            frames.append(frame)
            gt_boxes.append(gt)

        return frames, np.array(gt_boxes, dtype=np.float64)

    def _render_scale_change(
        self,
        bg: np.ndarray,
        colour: tuple,
        centres: List[Tuple[float, float]],
        bw: float,
        bh: float,
        rng: np.random.Generator,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Render scale-change challenge: target size oscillates between min/max factor.

        Scale follows a sinusoidal pattern over the sequence, starting at
        the nominal size (factor 1.0) and visiting both ``scale_min`` and
        ``scale_max`` uniformly.  Ground-truth boxes reflect the true scaled
        size at each frame.
        """
        frames = []
        gt_boxes = []
        s_min, s_max = self.scale_min, self.scale_max
        W, H = bg.shape[1], bg.shape[0]

        for i, (cx, cy) in enumerate(centres):
            frame = bg.copy()
            t = i / max(self.num_frames - 1, 1)
            # Scale factor oscillates: s_min → s_max → s_min (2 cycles)
            scale = s_min + (s_max - s_min) * (0.5 + 0.5 * math.sin(2 * math.pi * 2 * t))
            cur_w = bw * scale
            cur_h = bh * scale

            x1 = int(round(cx - cur_w / 2))
            y1 = int(round(cy - cur_h / 2))
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(W, int(round(cx + cur_w / 2)))
            y2c = min(H, int(round(cy + cur_h / 2)))
            frame[y1c:y2c, x1c:x2c] = colour

            frames.append(frame)
            gt_boxes.append((float(x1), float(y1), float(cur_w), float(cur_h)))

        return frames, np.array(gt_boxes, dtype=np.float64)

    def _render_clutter(
        self,
        bg: np.ndarray,
        colour: tuple,
        centres: List[Tuple[float, float]],
        bw: float,
        bh: float,
        rng: np.random.Generator,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Render background-clutter challenge: moving distractors of similar colour.

        Each distractor starts at a random position and moves with a random
        linear-bounce velocity.  Distractor colours are sampled near the target
        colour (within ±30 per channel) to maximise confusion for appearance-
        based trackers.
        """
        W, H = bg.shape[1], bg.shape[0]
        n = self.num_distractors

        # Distractor initial positions and velocities
        d_cx = rng.uniform(bw, W - bw, n)
        d_cy = rng.uniform(bh, H - bh, n)
        d_vx = rng.uniform(1.0, 3.0, n) * rng.choice([-1, 1], n)
        d_vy = rng.uniform(0.5, 2.5, n) * rng.choice([-1, 1], n)
        # Similar colour: clamp to [0, 255]
        d_colours = [
            tuple(
                int(np.clip(c + rng.integers(-30, 31), 0, 255))
                for c in colour
            )
            for _ in range(n)
        ]
        d_w = bw * 0.8
        d_h = bh * 0.8

        frames = []
        gt_boxes = []

        for i, (cx, cy) in enumerate(centres):
            frame = bg.copy()

            # Draw distractors first so target is always on top.
            for k in range(n):
                dx = int(round(d_cx[k] - d_w / 2))
                dy = int(round(d_cy[k] - d_h / 2))
                dx2 = dx + int(d_w)
                dy2 = dy + int(d_h)
                dx_c, dy_c = max(0, dx), max(0, dy)
                frame[dy_c:min(H, dy2), dx_c:min(W, dx2)] = d_colours[k]

                # Bounce distractors
                d_cx[k] += d_vx[k]
                d_cy[k] += d_vy[k]
                if d_cx[k] < d_w / 2 or d_cx[k] > W - d_w / 2:
                    d_vx[k] = -d_vx[k]
                    d_cx[k] = float(np.clip(d_cx[k], d_w / 2, W - d_w / 2))
                if d_cy[k] < d_h / 2 or d_cy[k] > H - d_h / 2:
                    d_vy[k] = -d_vy[k]
                    d_cy[k] = float(np.clip(d_cy[k], d_h / 2, H - d_h / 2))

            gt = self._draw_target(frame, cx, cy, bw, bh, colour)
            frames.append(frame)
            gt_boxes.append(gt)

        return frames, np.array(gt_boxes, dtype=np.float64)
