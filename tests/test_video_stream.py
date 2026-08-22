"""Tests for eovot.datasets.stream — VideoStreamSequence and VideoStreamDataset.

Tests use synthetic video files written to a temp directory so they run
without any external dataset files.  No webcam is required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

from eovot.datasets.stream import VideoStreamDataset, VideoStreamSequence, _load_gt


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_video(path: str, n_frames: int = 20, size: Tuple[int, int] = (320, 240)) -> str:
    """Write a synthetic BGR video file and return the path."""
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_gt_file(path: str, n_frames: int = 20) -> str:
    """Write a simple ground-truth file and return the path."""
    gt = np.tile([50.0, 30.0, 80.0, 60.0], (n_frames, 1))
    np.savetxt(path, gt, delimiter=",", fmt="%.1f")
    return path


# ---------------------------------------------------------------------------
# VideoStreamSequence tests
# ---------------------------------------------------------------------------


class TestVideoStreamSequence:
    def test_init_no_error(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"))
        seq = VideoStreamSequence(
            name="test", source=vpath, init_bbox=(50.0, 30.0, 80.0, 60.0)
        )
        assert seq.name == "test"

    def test_repr(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"))
        seq = VideoStreamSequence(
            name="test", source=vpath, init_bbox=(50.0, 30.0, 80.0, 60.0)
        )
        r = repr(seq)
        assert "VideoStreamSequence" in r
        assert "test" in r

    def test_init_bbox(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"))
        bbox = (10.0, 20.0, 30.0, 40.0)
        seq = VideoStreamSequence(name="s", source=vpath, init_bbox=bbox)
        assert seq.init_bbox == pytest.approx(bbox)

    def test_iteration_yields_frames(self, tmp_path):
        n = 15
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=n)
        seq = VideoStreamSequence(name="s", source=vpath, init_bbox=(0, 0, 10, 10))
        frames = list(seq)
        assert len(frames) == n
        assert frames[0].ndim == 3
        assert frames[0].shape[2] == 3

    def test_iteration_twice(self, tmp_path):
        """Sequence can be iterated more than once (new VideoCapture each time)."""
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=10)
        seq = VideoStreamSequence(name="s", source=vpath, init_bbox=(0, 0, 10, 10))
        first = len(list(seq))
        second = len(list(seq))
        assert first == second == 10

    def test_max_frames_cap(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=20)
        seq = VideoStreamSequence(
            name="s", source=vpath, init_bbox=(0, 0, 10, 10), max_frames=7
        )
        frames = list(seq)
        assert len(frames) == 7

    def test_invalid_source_raises(self, tmp_path):
        seq = VideoStreamSequence(
            name="bad", source="/nonexistent/path.mp4", init_bbox=(0, 0, 10, 10)
        )
        with pytest.raises(RuntimeError, match="Cannot open"):
            list(seq)

    def test_gt_shape_with_provided_gt(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=10)
        gt = np.tile([5.0, 5.0, 20.0, 20.0], (10, 1))
        seq = VideoStreamSequence(
            name="s", source=vpath, init_bbox=(5, 5, 20, 20), ground_truth=gt
        )
        assert seq.ground_truth.shape == (10, 4)
        assert seq.ground_truth[0, 0] == pytest.approx(5.0)

    def test_gt_nan_when_no_gt(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=10)
        seq = VideoStreamSequence(name="s", source=vpath, init_bbox=(0, 0, 10, 10))
        assert np.all(np.isnan(seq.ground_truth))

    def test_len_with_max_frames(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=20)
        seq = VideoStreamSequence(
            name="s", source=vpath, init_bbox=(0, 0, 10, 10), max_frames=12
        )
        assert len(seq) == 12


# ---------------------------------------------------------------------------
# VideoStreamDataset tests
# ---------------------------------------------------------------------------


class TestVideoStreamDataset:
    def test_from_video_single_sequence(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"))
        ds = VideoStreamDataset.from_video(
            path=str(tmp_path / "v.mp4"), init_bbox=(0, 0, 10, 10)
        )
        assert len(ds) == 1
        assert isinstance(ds[0], VideoStreamSequence)

    def test_from_video_default_name(self, tmp_path):
        vpath = _make_video(str(tmp_path / "highway.mp4"))
        ds = VideoStreamDataset.from_video(path=vpath, init_bbox=(0, 0, 10, 10))
        assert ds[0].name == "highway"

    def test_from_video_custom_name(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"))
        ds = VideoStreamDataset.from_video(
            path=vpath, init_bbox=(0, 0, 10, 10), name="my_seq"
        )
        assert ds[0].name == "my_seq"

    def test_from_video_with_gt(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=15)
        gtpath = _make_gt_file(str(tmp_path / "gt.txt"), n_frames=15)
        ds = VideoStreamDataset.from_video(
            path=vpath, init_bbox=(50, 30, 80, 60), gt_path=gtpath
        )
        assert not np.any(np.isnan(ds[0].ground_truth))
        assert ds[0].ground_truth.shape[1] == 4

    def test_from_webcam_returns_dataset(self):
        ds = VideoStreamDataset.from_webcam(device_index=0, init_bbox=(0, 0, 50, 50))
        assert len(ds) == 1
        assert ds[0].name == "webcam_0"

    def test_from_webcam_max_frames(self):
        ds = VideoStreamDataset.from_webcam(
            device_index=0, init_bbox=(0, 0, 50, 50), max_frames=150
        )
        # len() may return max_frames for unknown-length sources.
        assert len(ds[0]) == 150

    def test_from_sources_multiple(self, tmp_path):
        vpath1 = _make_video(str(tmp_path / "v1.mp4"), n_frames=10)
        vpath2 = _make_video(str(tmp_path / "v2.mp4"), n_frames=8)
        ds = VideoStreamDataset.from_sources([
            {"source": vpath1, "init_bbox": (0, 0, 10, 10), "name": "clip1"},
            {"source": vpath2, "init_bbox": (5, 5, 15, 15), "name": "clip2"},
        ])
        assert len(ds) == 2
        assert ds[0].name == "clip1"
        assert ds[1].name == "clip2"

    def test_from_sources_auto_name(self, tmp_path):
        vpath = _make_video(str(tmp_path / "scene_01.mp4"))
        ds = VideoStreamDataset.from_sources([
            {"source": vpath, "init_bbox": (0, 0, 10, 10)}
        ])
        assert ds[0].name == "scene_01"

    def test_empty_sources_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            VideoStreamDataset([])

    def test_iter_over_dataset(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=5)
        ds = VideoStreamDataset.from_video(path=vpath, init_bbox=(0, 0, 10, 10))
        seqs = list(ds)
        assert len(seqs) == 1

    def test_repr(self, tmp_path):
        vpath = _make_video(str(tmp_path / "v.mp4"))
        ds = VideoStreamDataset.from_video(path=vpath, init_bbox=(0, 0, 10, 10))
        assert "VideoStreamDataset" in repr(ds)


# ---------------------------------------------------------------------------
# _load_gt helper tests
# ---------------------------------------------------------------------------


class TestLoadGT:
    def test_comma_separated(self, tmp_path):
        gtpath = tmp_path / "gt.txt"
        gtpath.write_text("10,20,30,40\n11,21,31,41\n")
        gt = _load_gt(gtpath)
        assert gt.shape == (2, 4)
        assert gt[0, 0] == pytest.approx(10.0)

    def test_space_separated(self, tmp_path):
        gtpath = tmp_path / "gt.txt"
        gtpath.write_text("10 20 30 40\n11 21 31 41\n")
        gt = _load_gt(gtpath)
        assert gt.shape == (2, 4)

    def test_single_row(self, tmp_path):
        gtpath = tmp_path / "gt.txt"
        gtpath.write_text("5,10,15,20\n")
        gt = _load_gt(gtpath)
        assert gt.shape == (1, 4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_gt(tmp_path / "nonexistent.txt")

    def test_wrong_column_count_raises(self, tmp_path):
        gtpath = tmp_path / "gt.txt"
        gtpath.write_text("1,2,3\n4,5,6\n")
        with pytest.raises(ValueError, match="4 columns"):
            _load_gt(gtpath)


# ---------------------------------------------------------------------------
# BenchmarkEngine integration
# ---------------------------------------------------------------------------


class TestBenchmarkEngineIntegration:
    def test_run_on_video_stream_dataset(self, tmp_path):
        """BenchmarkEngine.run() works with VideoStreamDataset without GT."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=20)
        ds = VideoStreamDataset.from_video(
            path=vpath,
            init_bbox=(50.0, 30.0, 80.0, 60.0),
            max_frames=20,
        )
        engine = BenchmarkEngine(verbose=False)
        tracker = MOSSETracker()
        result = engine.run(tracker, ds, dataset_name="stream_test")

        assert len(result.sequence_results) == 1
        sr = result.sequence_results[0]
        assert sr.profiling.frame_count > 0
        assert sr.profiling.fps > 0

    def test_run_on_video_stream_dataset_with_gt(self, tmp_path):
        """Accuracy metrics are valid when GT is provided."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        n = 20
        vpath = _make_video(str(tmp_path / "v.mp4"), n_frames=n)
        gtpath = _make_gt_file(str(tmp_path / "gt.txt"), n_frames=n)
        ds = VideoStreamDataset.from_video(
            path=vpath,
            init_bbox=(50.0, 30.0, 80.0, 60.0),
            gt_path=gtpath,
            max_frames=n,
        )
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="stream_gt")

        sr = result.sequence_results[0]
        # IoU may be low on random frames but must be a real number (not NaN).
        assert not np.any(np.isnan(sr.ious))
