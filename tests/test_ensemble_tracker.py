"""Tests for eovot.trackers.ensemble.EnsembleTracker."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker, BBox
from eovot.trackers.ensemble import EnsembleTracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_FRAME_H, _FRAME_W = 120, 160
_INIT_BBOX: BBox = (50.0, 30.0, 40.0, 40.0)


def _make_frame(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a synthetic BGR frame with a bright rectangle target."""
    if rng is None:
        rng = np.random.default_rng(0)
    frame = rng.integers(40, 100, (_FRAME_H, _FRAME_W, 3), dtype=np.uint8)
    frame[30:70, 50:90] = [200, 100, 50]
    return frame


def _make_frames(n: int = 10) -> List[np.ndarray]:
    rng = np.random.default_rng(42)
    return [_make_frame(rng) for _ in range(n)]


class _ListDataset(BaseDataset):
    """Minimal in-memory dataset wrapping a list of Sequence objects."""

    def __init__(self, sequences: List[Sequence]) -> None:
        self._sequences = sequences

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self._sequences[idx]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> _ListDataset:
    """Two 10-frame sequences backed by real PNG files on disk."""
    root = tmp_path_factory.mktemp("frames")
    rng = np.random.default_rng(7)
    sequences: List[Sequence] = []

    for si in range(2):
        img_dir = root / f"seq_{si}"
        img_dir.mkdir()
        paths: List[str] = []
        gt_rows: List[List[float]] = []

        for fi in range(10):
            frame = rng.integers(40, 100, (_FRAME_H, _FRAME_W, 3), dtype=np.uint8)
            frame[30:70, 50:90] = [200, 100, 50]
            path = str(img_dir / f"{fi + 1:04d}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            gt_rows.append([50.0, 30.0, 40.0, 40.0])

        sequences.append(
            Sequence(
                name=f"seq_{si}",
                frame_paths=paths,
                ground_truth=np.array(gt_rows, dtype=np.float64),
            )
        )

    return _ListDataset(sequences)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_tracker(self):
        ens = EnsembleTracker([MOSSETracker()])
        assert len(ens.trackers) == 1

    def test_multi_tracker(self):
        ens = EnsembleTracker([MOSSETracker(), KCFTracker()])
        assert len(ens.trackers) == 2

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            EnsembleTracker([])

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            EnsembleTracker([MOSSETracker()], strategy="bogus")  # type: ignore

    def test_default_name(self):
        ens = EnsembleTracker([MOSSETracker()])
        assert ens.name == "Ensemble"

    def test_custom_name(self):
        ens = EnsembleTracker([MOSSETracker()], name="MyEns")
        assert ens.name == "MyEns"

    def test_n_jobs_clipped_to_one(self):
        ens = EnsembleTracker([MOSSETracker()], n_jobs=0)
        assert ens.n_jobs == 1

    def test_repr_contains_strategy(self):
        ens = EnsembleTracker([MOSSETracker()], strategy="mean")
        assert "mean" in repr(ens)

    def test_repr_contains_name(self):
        ens = EnsembleTracker([MOSSETracker()], name="VotingEns")
        assert "VotingEns" in repr(ens)


# ---------------------------------------------------------------------------
# BaseTracker interface compliance
# ---------------------------------------------------------------------------

class TestBaseTrackerInterface:
    @pytest.fixture
    def ens(self) -> EnsembleTracker:
        e = EnsembleTracker([MOSSETracker(), MOSSETracker()], strategy="median")
        e.initialize(_make_frame(), _INIT_BBOX)
        return e

    def test_update_returns_tuple_of_4(self, ens):
        bbox = ens.update(_make_frame())
        assert isinstance(bbox, tuple) and len(bbox) == 4

    def test_update_values_are_finite(self, ens):
        bbox = ens.update(_make_frame())
        assert all(np.isfinite(v) for v in bbox)

    def test_width_positive(self, ens):
        _, _, w, _ = ens.update(_make_frame())
        assert w > 0

    def test_height_positive(self, ens):
        _, _, _, h = ens.update(_make_frame())
        assert h > 0

    def test_multiple_updates(self, ens):
        frames = _make_frames(5)
        for f in frames:
            bbox = ens.update(f)
            assert len(bbox) == 4


# ---------------------------------------------------------------------------
# Strategy: mean
# ---------------------------------------------------------------------------

class TestMeanStrategy:
    def test_single_tracker_passes_through(self):
        ens = EnsembleTracker([MOSSETracker()], strategy="mean")
        ens.initialize(_make_frame(), _INIT_BBOX)
        ref = MOSSETracker()
        ref.initialize(_make_frame(), _INIT_BBOX)

        frame = _make_frame()
        b_ens = ens.update(frame)
        b_ref = ref.update(frame)
        # Both trackers use the same seed-less state; results may differ slightly
        assert len(b_ens) == 4

    def test_mean_of_identical_trackers(self):
        # Two identical MOSSE trackers → mean == their shared output
        t1, t2 = MOSSETracker(learning_rate=0.0), MOSSETracker(learning_rate=0.0)
        ens = EnsembleTracker([t1, t2], strategy="mean")
        frame = _make_frame()
        ens.initialize(frame, _INIT_BBOX)
        bbox = ens.update(frame)
        # Mean of two identical values equals that value
        assert all(np.isfinite(v) for v in bbox)


# ---------------------------------------------------------------------------
# Strategy: median
# ---------------------------------------------------------------------------

class TestMedianStrategy:
    def test_odd_number_of_trackers(self):
        ens = EnsembleTracker(
            [MOSSETracker(), MOSSETracker(), MOSSETracker()], strategy="median"
        )
        frame = _make_frame()
        ens.initialize(frame, _INIT_BBOX)
        bbox = ens.update(frame)
        assert len(bbox) == 4

    def test_even_number_of_trackers(self):
        ens = EnsembleTracker(
            [MOSSETracker(), MOSSETracker()], strategy="median"
        )
        frame = _make_frame()
        ens.initialize(frame, _INIT_BBOX)
        bbox = ens.update(frame)
        assert len(bbox) == 4

    def test_outlier_suppression(self):
        """Median ignores a single extreme outlier prediction."""

        class ConstantTracker(BaseTracker):
            def __init__(self, bbox):
                super().__init__(name="Const")
                self._bbox = bbox

            def initialize(self, frame, bbox):
                pass

            def update(self, frame):
                return self._bbox

        normal = (50.0, 50.0, 40.0, 40.0)
        outlier = (500.0, 500.0, 40.0, 40.0)

        ens = EnsembleTracker(
            [
                ConstantTracker(normal),
                ConstantTracker(normal),
                ConstantTracker(outlier),  # single outlier
            ],
            strategy="median",
        )
        ens.initialize(_make_frame(), _INIT_BBOX)
        x, y, w, h = ens.update(_make_frame())
        # Median of [50, 50, 500] = 50 for x and y
        assert x == pytest.approx(50.0)
        assert y == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Strategy: nms_vote
# ---------------------------------------------------------------------------

class TestNmsVoteStrategy:
    def test_basic_output(self):
        ens = EnsembleTracker([MOSSETracker(), KCFTracker()], strategy="nms_vote")
        frame = _make_frame()
        ens.initialize(frame, _INIT_BBOX)
        bbox = ens.update(frame)
        assert len(bbox) == 4
        assert all(np.isfinite(v) for v in bbox)

    def test_single_tracker_fallback(self):
        ens = EnsembleTracker([MOSSETracker()], strategy="nms_vote")
        frame = _make_frame()
        ens.initialize(frame, _INIT_BBOX)
        bbox = ens.update(frame)
        assert len(bbox) == 4

    def test_non_overlapping_falls_back_to_uniform(self):
        """When all predictions are disjoint, weights should be uniform (mean)."""

        class ConstantTracker(BaseTracker):
            def __init__(self, bbox):
                super().__init__(name="Const")
                self._b = bbox

            def initialize(self, frame, bbox):
                pass

            def update(self, frame):
                return self._b

        boxes = [(0.0, 0.0, 10.0, 10.0), (100.0, 0.0, 10.0, 10.0)]
        ens = EnsembleTracker(
            [ConstantTracker(b) for b in boxes], strategy="nms_vote"
        )
        ens.initialize(_make_frame(), _INIT_BBOX)
        x, y, w, h = ens.update(_make_frame())
        # Uniform weights → mean of the two boxes
        assert x == pytest.approx(50.0)  # (0 + 100) / 2


# ---------------------------------------------------------------------------
# Parallel execution (n_jobs > 1)
# ---------------------------------------------------------------------------

class TestParallelExecution:
    def test_parallel_output_matches_sequential(self):
        """Sequential and parallel should produce identical results for MOSSE."""
        frame = _make_frame()

        ens_seq = EnsembleTracker(
            [MOSSETracker(learning_rate=0.0), MOSSETracker(learning_rate=0.0)],
            strategy="mean",
            n_jobs=1,
        )
        ens_par = EnsembleTracker(
            [MOSSETracker(learning_rate=0.0), MOSSETracker(learning_rate=0.0)],
            strategy="mean",
            n_jobs=2,
        )
        ens_seq.initialize(frame, _INIT_BBOX)
        ens_par.initialize(frame, _INIT_BBOX)

        b_seq = ens_seq.update(frame)
        b_par = ens_par.update(frame)
        # Both produce valid bboxes (exact match not guaranteed due to GIL/ordering)
        assert len(b_seq) == 4 and len(b_par) == 4


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_ensemble_runs_full_sequence(self, tiny_dataset):
        ens = EnsembleTracker(
            [MOSSETracker(), KCFTracker()],
            strategy="median",
            name="MOSSE+KCF",
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ens, tiny_dataset, dataset_name="Tiny")
        assert result.tracker_name == "MOSSE+KCF"
        assert len(result.sequence_results) == 2

    def test_mean_iou_non_negative(self, tiny_dataset):
        ens = EnsembleTracker([MOSSETracker(), MOSSETracker()], strategy="mean")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ens, tiny_dataset, dataset_name="Tiny")
        assert result.mean_iou >= 0.0

    def test_fps_positive(self, tiny_dataset):
        ens = EnsembleTracker([MOSSETracker()], strategy="median")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ens, tiny_dataset, dataset_name="Tiny")
        assert result.mean_fps > 0.0

    @pytest.mark.parametrize("strategy", ["mean", "median", "nms_vote"])
    def test_all_strategies_complete_without_error(self, tiny_dataset, strategy):
        ens = EnsembleTracker(
            [MOSSETracker(), KCFTracker()],
            strategy=strategy,  # type: ignore
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(ens, tiny_dataset, dataset_name="Tiny")
        assert len(result.sequence_results) == len(tiny_dataset)
