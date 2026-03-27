"""Shared pytest fixtures for the EOVOT test suite.

All fixtures work entirely with synthetic in-memory or tmp_path data —
no real dataset downloads are required to run the test suite.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Primitive fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_frame() -> np.ndarray:
    """64×64 BGR uint8 frame with a visible square target."""
    rng = np.random.default_rng(42)
    frame = rng.integers(50, 150, (64, 64, 3), dtype=np.uint8)
    # draw a bright 20×20 square at (10, 10) to simulate a target
    frame[10:30, 10:30] = [200, 100, 50]
    return frame


@pytest.fixture
def bbox() -> Tuple[float, float, float, float]:
    """Bounding box (x, y, w, h) matching the target in *small_frame*."""
    return (10.0, 10.0, 20.0, 20.0)


@pytest.fixture
def pred_boxes() -> np.ndarray:
    """(10, 4) array of predicted boxes with slight jitter."""
    rng = np.random.default_rng(0)
    base = np.array([[10.0, 10.0, 20.0, 20.0]] * 10)
    base[:, :2] += rng.uniform(-2, 2, (10, 2))
    return base


@pytest.fixture
def gt_boxes() -> np.ndarray:
    """(10, 4) array of ground-truth boxes (static target)."""
    return np.tile([10.0, 10.0, 20.0, 20.0], (10, 1)).astype(np.float64)


# ---------------------------------------------------------------------------
# Synthetic OTB-style dataset on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_otb_root(tmp_path) -> str:
    """Create a minimal valid OTBDataset directory with two sequences.

    The target moves by 1 px/frame in x to simulate realistic tracking.
    """
    for seq_name, offset in [("seq_A", 0), ("seq_B", 5)]:
        seq_dir = tmp_path / seq_name
        img_dir = seq_dir / "img"
        img_dir.mkdir(parents=True)

        gt_rows: List[str] = []
        for i in range(5):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            x, y = 10 + offset + i, 10
            frame[y : y + 20, x : x + 20] = [200, 100, 50]
            cv2.imwrite(str(img_dir / f"{i + 1:04d}.jpg"), frame)
            gt_rows.append(f"{float(x)},{float(y)},20.0,20.0")

        (seq_dir / "groundtruth_rect.txt").write_text("\n".join(gt_rows) + "\n")

    return str(tmp_path)


@pytest.fixture
def static_otb_root(tmp_path) -> str:
    """Create a minimal valid OTBDataset with a static (non-moving) target.

    All GT boxes are identical across frames — a tracker that returns the
    init bbox every update() call should achieve IoU == 1.0.
    """
    root = tmp_path / "static_ds"
    seq_dir = root / "seq_static"
    img_dir = seq_dir / "img"
    img_dir.mkdir(parents=True)

    gt_rows: List[str] = []
    for i in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[10:30, 10:30] = [200, 100, 50]
        cv2.imwrite(str(img_dir / f"{i + 1:04d}.jpg"), frame)
        gt_rows.append("10.0,10.0,20.0,20.0")

    (seq_dir / "groundtruth_rect.txt").write_text("\n".join(gt_rows) + "\n")
    return str(root)


# ---------------------------------------------------------------------------
# Minimal mock tracker
# ---------------------------------------------------------------------------

class _PassthroughTracker:
    """Tracker that always returns the initialisation bbox (useful for tests)."""

    name = "Passthrough"

    def initialize(self, frame: np.ndarray, bbox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray):
        return self._bbox


@pytest.fixture
def passthrough_tracker() -> _PassthroughTracker:
    return _PassthroughTracker()
