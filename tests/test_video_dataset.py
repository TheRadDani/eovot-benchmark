"""Tests for eovot.datasets.video_dataset."""

import numpy as np
import pytest

from eovot.datasets.base import BBox
from eovot.datasets.video_dataset import VideoFileDataset, VideoFileSequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_video(path, num_frames=20, width=160, height=120, fps=30):
    """Write a minimal synthetic video to *path* using cv2.VideoWriter."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    rng = np.random.default_rng(0)
    for _ in range(num_frames):
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


_BBOX: BBox = (10.0, 10.0, 30.0, 30.0)


# ---------------------------------------------------------------------------
# VideoFileSequence tests
# ---------------------------------------------------------------------------

class TestVideoFileSequence:
    def test_name_defaults_to_stem(self, tmp_path):
        p = tmp_path / "myclip.mp4"
        _write_video(p)
        assert VideoFileSequence(p, _BBOX).name == "myclip"

    def test_name_custom(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p)
        assert VideoFileSequence(p, _BBOX, name="custom").name == "custom"

    def test_init_bbox(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p)
        assert VideoFileSequence(p, _BBOX).init_bbox == _BBOX

    def test_len_matches_frame_count(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=20)
        assert len(VideoFileSequence(p, _BBOX)) == 20

    def test_len_respects_max_frames(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=20)
        assert len(VideoFileSequence(p, _BBOX, max_frames=10)) == 10

    def test_iter_yields_numpy_arrays(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=5)
        frames = list(VideoFileSequence(p, _BBOX))
        assert len(frames) == 5
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_iter_frame_shape(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=5, width=160, height=120)
        for f in VideoFileSequence(p, _BBOX):
            assert f.ndim == 3 and f.shape[2] == 3

    def test_iter_stops_at_max_frames(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=20)
        assert len(list(VideoFileSequence(p, _BBOX, max_frames=7))) == 7

    def test_iter_stops_at_ground_truth_length(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=20)
        gt = np.zeros((8, 4))
        assert len(list(VideoFileSequence(p, _BBOX, ground_truth=gt))) == 8

    def test_ground_truth_stored(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p)
        gt = np.ones((5, 4))
        seq = VideoFileSequence(p, _BBOX, ground_truth=gt)
        assert seq.ground_truth is gt

    def test_ground_truth_placeholder_when_none(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=10)
        seq = VideoFileSequence(p, _BBOX)
        gt = seq.ground_truth
        assert gt.shape == (10, 4)
        # every row should equal init_bbox
        expected = np.array(_BBOX)
        assert np.allclose(gt, expected)

    def test_video_path_property(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p)
        assert VideoFileSequence(p, _BBOX).video_path == p

    def test_missing_video_raises_on_iter(self, tmp_path):
        seq = VideoFileSequence(tmp_path / "nonexistent.mp4", _BBOX)
        with pytest.raises(IOError):
            list(seq)

    def test_iterable_multiple_times(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=5)
        seq = VideoFileSequence(p, _BBOX)
        assert len(list(seq)) == 5
        assert len(list(seq)) == 5


# ---------------------------------------------------------------------------
# VideoFileDataset tests
# ---------------------------------------------------------------------------

class TestVideoFileDataset:
    def test_len(self, tmp_path):
        paths = [tmp_path / f"clip{i}.mp4" for i in range(3)]
        for p in paths:
            _write_video(p)
        ds = VideoFileDataset([VideoFileSequence(p, _BBOX) for p in paths])
        assert len(ds) == 3

    def test_getitem(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p)
        seq = VideoFileSequence(p, _BBOX)
        assert VideoFileDataset([seq])[0] is seq

    def test_from_files_two_tuple(self, tmp_path):
        paths = [tmp_path / f"clip{i}.mp4" for i in range(2)]
        for p in paths:
            _write_video(p)
        ds = VideoFileDataset.from_files([(p, _BBOX) for p in paths])
        assert len(ds) == 2

    def test_from_files_three_tuple_with_gt(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=10)
        gt = np.zeros((10, 4))
        ds = VideoFileDataset.from_files([(p, _BBOX, gt)])
        assert ds[0].ground_truth is gt

    def test_from_files_max_frames(self, tmp_path):
        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=20)
        ds = VideoFileDataset.from_files([(p, _BBOX)], max_frames=5)
        assert len(list(ds[0])) == 5

    def test_from_directory_finds_mp4(self, tmp_path):
        for i in range(3):
            _write_video(tmp_path / f"v{i}.mp4")
        assert len(VideoFileDataset.from_directory(tmp_path, _BBOX)) == 3

    def test_from_directory_filters_non_video(self, tmp_path):
        _write_video(tmp_path / "v1.mp4")
        (tmp_path / "readme.txt").write_text("ignore")
        assert len(VideoFileDataset.from_directory(tmp_path, _BBOX)) == 1

    def test_from_directory_custom_extensions(self, tmp_path):
        _write_video(tmp_path / "v1.mp4")
        _write_video(tmp_path / "v2.avi")
        assert len(
            VideoFileDataset.from_directory(tmp_path, _BBOX, extensions=["mp4"])
        ) == 1

    def test_from_directory_empty(self, tmp_path):
        assert len(VideoFileDataset.from_directory(tmp_path, _BBOX)) == 0

    def test_from_directory_max_frames(self, tmp_path):
        _write_video(tmp_path / "v1.mp4", num_frames=20)
        ds = VideoFileDataset.from_directory(tmp_path, _BBOX, max_frames=5)
        assert len(list(ds[0])) == 5

    def test_iter_over_dataset(self, tmp_path):
        for i in range(3):
            _write_video(tmp_path / f"clip{i}.mp4", num_frames=5)
        ds = VideoFileDataset.from_directory(tmp_path, _BBOX)
        seqs = list(ds)
        assert len(seqs) == 3
        assert all(isinstance(s, VideoFileSequence) for s in seqs)


# ---------------------------------------------------------------------------
# Integration: VideoFileDataset with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestBenchmarkEngineIntegration:
    def test_engine_runs_on_video_dataset(self, tmp_path):
        """BenchmarkEngine can evaluate a trivial tracker on VideoFileDataset."""
        import cv2
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.base import BaseTracker

        # Trivial tracker that always returns the init_bbox
        class StaticTracker(BaseTracker):
            def initialize(self, frame, bbox):
                self._bbox = bbox

            def update(self, frame):
                return self._bbox

        p = tmp_path / "clip.mp4"
        _write_video(p, num_frames=10, width=80, height=60)
        ds = VideoFileDataset.from_files([(p, _BBOX)])

        engine = BenchmarkEngine(verbose=False)
        result = engine.run(StaticTracker("static"), ds, dataset_name="video_test")

        assert result.tracker_name == "static"
        assert len(result.sequence_results) == 1
        assert result.mean_fps > 0
        # IoU of predicted (init_bbox) vs placeholder GT (also init_bbox) = 1.0
        assert abs(result.mean_iou - 1.0) < 1e-6
