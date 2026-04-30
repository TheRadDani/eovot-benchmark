"""Unit and integration tests for eovot.datasets.synthetic."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from eovot.datasets.base import BaseDataset, Sequence
from eovot.datasets.synthetic import (
    SyntheticDataset,
    SyntheticSequenceConfig,
    _compute_ground_truth,
    _render_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg(**kwargs) -> SyntheticSequenceConfig:
    """Return a minimal valid config, overridable via kwargs."""
    base = dict(
        name="test_seq",
        n_frames=20,
        frame_size=(120, 160),
        init_bbox=(40.0, 30.0, 40.0, 30.0),
        motion="linear",
        velocity=(1.0, 0.5),
        seed=0,
    )
    base.update(kwargs)
    return SyntheticSequenceConfig(**base)


# ---------------------------------------------------------------------------
# SyntheticSequenceConfig
# ---------------------------------------------------------------------------

class TestSyntheticSequenceConfig:
    def test_resolve_init_bbox_explicit(self):
        cfg = _default_cfg(init_bbox=(10.0, 20.0, 50.0, 40.0))
        assert cfg.resolve_init_bbox() == (10.0, 20.0, 50.0, 40.0)

    def test_resolve_init_bbox_default_is_centred(self):
        cfg = SyntheticSequenceConfig(frame_size=(200, 400))
        x, y, w, h = cfg.resolve_init_bbox()
        # Centre of default box should be near frame centre
        cx, cy = x + w / 2, y + h / 2
        assert abs(cx - 200) < 5 and abs(cy - 100) < 5

    def test_default_motion_is_linear(self):
        cfg = SyntheticSequenceConfig()
        assert cfg.motion == "linear"


# ---------------------------------------------------------------------------
# _compute_ground_truth
# ---------------------------------------------------------------------------

class TestComputeGroundTruth:
    def test_shape(self):
        cfg = _default_cfg(n_frames=30)
        gt = _compute_ground_truth(cfg)
        assert gt.shape == (30, 4)

    def test_first_frame_matches_init_bbox(self):
        cfg = _default_cfg(
            init_bbox=(40.0, 30.0, 40.0, 30.0),
            motion="linear",
            velocity=(0.0, 0.0),
        )
        gt = _compute_ground_truth(cfg)
        np.testing.assert_allclose(gt[0], [40.0, 30.0, 40.0, 30.0], atol=1e-6)

    def test_linear_motion_advances_position(self):
        cfg = _default_cfg(
            motion="linear",
            velocity=(3.0, 2.0),
            scale_factor=1.0,
        )
        gt = _compute_ground_truth(cfg)
        # Centre at frame i should advance by velocity relative to frame 0
        cx0 = gt[0, 0] + gt[0, 2] / 2
        cy0 = gt[0, 1] + gt[0, 3] / 2
        cx1 = gt[1, 0] + gt[1, 2] / 2
        cy1 = gt[1, 1] + gt[1, 3] / 2
        assert abs((cx1 - cx0) - 3.0) < 0.5
        assert abs((cy1 - cy0) - 2.0) < 0.5

    def test_sinusoidal_motion_is_oscillating(self):
        cfg = _default_cfg(
            motion="sinusoidal",
            amplitude=(50.0, 30.0),
            frequency=0.1,
            n_frames=50,
        )
        gt = _compute_ground_truth(cfg)
        cx = gt[:, 0] + gt[:, 2] / 2
        # Oscillating — centre should vary significantly
        assert cx.max() - cx.min() > 10.0

    def test_random_walk_is_reproducible_with_seed(self):
        cfg = _default_cfg(motion="random_walk", seed=42, n_frames=40)
        gt1 = _compute_ground_truth(cfg)
        gt2 = _compute_ground_truth(cfg)
        np.testing.assert_array_equal(gt1, gt2)

    def test_random_walk_different_seeds_differ(self):
        cfg1 = _default_cfg(motion="random_walk", seed=1, n_frames=40)
        cfg2 = _default_cfg(motion="random_walk", seed=2, n_frames=40)
        gt1 = _compute_ground_truth(cfg1)
        gt2 = _compute_ground_truth(cfg2)
        assert not np.allclose(gt1, gt2)

    def test_boxes_stay_within_frame(self):
        cfg = _default_cfg(
            n_frames=80,
            frame_size=(120, 160),
            motion="linear",
            velocity=(10.0, 8.0),
        )
        gt = _compute_ground_truth(cfg)
        H, W = cfg.frame_size
        x, y, w, h = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
        assert np.all(x >= 0), "x must be non-negative"
        assert np.all(y >= 0), "y must be non-negative"
        assert np.all(x + w <= W), f"box right edge exceeded frame width"
        assert np.all(y + h <= H), "box bottom edge exceeded frame height"

    def test_scale_factor_grows_target(self):
        cfg = _default_cfg(scale_factor=1.02, n_frames=30)
        gt = _compute_ground_truth(cfg)
        # Target width should increase monotonically
        assert gt[-1, 2] > gt[0, 2]

    def test_scale_factor_shrinks_target(self):
        cfg = _default_cfg(scale_factor=0.98, n_frames=30)
        gt = _compute_ground_truth(cfg)
        assert gt[-1, 2] < gt[0, 2]

    def test_invalid_motion_raises(self):
        cfg = _default_cfg()
        cfg.motion = "invalid_motion"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown motion model"):
            _compute_ground_truth(cfg)


# ---------------------------------------------------------------------------
# _render_frame
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_output_shape_and_dtype(self):
        cfg = _default_cfg()
        bbox = (40.0, 30.0, 40.0, 30.0)
        frame = _render_frame(cfg, bbox, None)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_solid_appearance_paints_target_color(self):
        cfg = _default_cfg(appearance="solid", target_color=(0, 0, 255))
        bbox = (40.0, 30.0, 40.0, 30.0)
        frame = _render_frame(cfg, bbox, None)
        # Sample a pixel inside the bounding box
        px = frame[30 + 5, 40 + 5]
        np.testing.assert_array_equal(px, [0, 0, 255])

    def test_background_color_applied(self):
        cfg = _default_cfg(bg_color=(200, 200, 200))
        bbox = (40.0, 30.0, 40.0, 30.0)
        frame = _render_frame(cfg, bbox, None)
        # Pixel far from bounding box should have background color
        px = frame[0, 0]
        np.testing.assert_array_equal(px, [200, 200, 200])

    def test_checkerboard_appearance_valid(self):
        cfg = _default_cfg(appearance="checkerboard")
        bbox = (20.0, 20.0, 60.0, 40.0)
        frame = _render_frame(cfg, bbox, None)
        patch = frame[20:60, 20:80]
        # Checkerboard must have at least two distinct colors
        unique_colors = set(map(tuple, patch.reshape(-1, 3).tolist()))
        assert len(unique_colors) >= 2

    def test_gradient_appearance_valid(self):
        cfg = _default_cfg(appearance="gradient")
        bbox = (20.0, 20.0, 60.0, 40.0)
        frame = _render_frame(cfg, bbox, None)
        assert frame.dtype == np.uint8

    def test_noise_is_added(self):
        cfg = _default_cfg(add_noise=True, noise_sigma=20.0)
        bbox = (40.0, 30.0, 40.0, 30.0)
        rng = np.random.default_rng(0)
        noisy = _render_frame(cfg, bbox, rng)
        # Noise means the frame is not a perfectly uniform color
        assert noisy.std() > 0.0

    def test_zero_area_bbox_returns_background(self):
        cfg = _default_cfg(bg_color=(100, 100, 100))
        bbox = (10.0, 10.0, 0.0, 0.0)  # zero width and height
        frame = _render_frame(cfg, bbox, None)
        assert frame[10, 10, 0] == 100  # background, not target


# ---------------------------------------------------------------------------
# _SyntheticSequence (accessed via SyntheticDataset)
# ---------------------------------------------------------------------------

class TestSyntheticSequence:
    def setup_method(self):
        self.cfg = _default_cfg(n_frames=15)
        self.dataset = SyntheticDataset([self.cfg])
        self.seq = self.dataset[0]

    def test_is_sequence_instance(self):
        assert isinstance(self.seq, Sequence)

    def test_len_matches_n_frames(self):
        assert len(self.seq) == 15

    def test_ground_truth_shape(self):
        assert self.seq.ground_truth.shape == (15, 4)

    def test_iter_yields_correct_count(self):
        frames = list(self.seq)
        assert len(frames) == 15

    def test_frames_have_correct_shape(self):
        for frame in self.seq:
            assert frame.shape == (120, 160, 3)
            assert frame.dtype == np.uint8

    def test_init_bbox_matches_gt_frame0(self):
        np.testing.assert_allclose(
            self.seq.init_bbox,
            tuple(self.seq.ground_truth[0]),
            atol=1e-6,
        )

    def test_name_propagated(self):
        assert self.seq.name == "test_seq"

    def test_repr(self):
        s = repr(self.seq)
        assert "test_seq" in s


# ---------------------------------------------------------------------------
# SyntheticDataset
# ---------------------------------------------------------------------------

class TestSyntheticDataset:
    def test_is_base_dataset(self):
        ds = SyntheticDataset([_default_cfg()])
        assert isinstance(ds, BaseDataset)

    def test_len_matches_configs(self):
        cfgs = [_default_cfg(name=f"s{i}") for i in range(4)]
        ds = SyntheticDataset(cfgs)
        assert len(ds) == 4

    def test_getitem_returns_sequence(self):
        ds = SyntheticDataset([_default_cfg()])
        assert isinstance(ds[0], Sequence)

    def test_iter_yields_all_sequences(self):
        cfgs = [_default_cfg(name=f"s{i}") for i in range(3)]
        ds = SyntheticDataset(cfgs)
        seqs = list(ds)
        assert len(seqs) == 3

    def test_repr(self):
        ds = SyntheticDataset([_default_cfg()])
        assert "SyntheticDataset" in repr(ds)


# ---------------------------------------------------------------------------
# SyntheticDataset factory methods
# ---------------------------------------------------------------------------

class TestQuickFactory:
    def test_returns_correct_length(self):
        ds = SyntheticDataset.quick(n_sequences=5)
        assert len(ds) == 5

    def test_frames_per_sequence(self):
        ds = SyntheticDataset.quick(n_sequences=3, n_frames=25)
        for seq in ds:
            assert len(seq) == 25

    def test_frame_size_respected(self):
        ds = SyntheticDataset.quick(n_sequences=2, frame_size=(100, 150))
        for seq in ds:
            for frame in seq:
                assert frame.shape == (100, 150, 3)
                break  # only need to check first frame

    def test_reproducible_with_seed(self):
        ds1 = SyntheticDataset.quick(seed=7)
        ds2 = SyntheticDataset.quick(seed=7)
        for s1, s2 in zip(ds1, ds2):
            np.testing.assert_array_equal(s1.ground_truth, s2.ground_truth)

    def test_different_seeds_differ(self):
        ds1 = SyntheticDataset.quick(n_sequences=3, seed=1)
        ds2 = SyntheticDataset.quick(n_sequences=3, seed=2)
        gts_equal = [
            np.allclose(s1.ground_truth, s2.ground_truth)
            for s1, s2 in zip(ds1, ds2)
        ]
        assert not all(gts_equal)

    def test_sinusoidal_motion_variant(self):
        ds = SyntheticDataset.quick(motion="sinusoidal", n_sequences=2)
        assert len(ds) == 2

    def test_random_walk_motion_variant(self):
        ds = SyntheticDataset.quick(motion="random_walk", n_sequences=2, seed=0)
        assert len(ds) == 2

    def test_gt_boxes_are_within_frame(self):
        ds = SyntheticDataset.quick(n_sequences=3, frame_size=(120, 160))
        H, W = 120, 160
        for seq in ds:
            gt = seq.ground_truth
            assert np.all(gt[:, 0] >= 0)
            assert np.all(gt[:, 1] >= 0)
            assert np.all(gt[:, 0] + gt[:, 2] <= W)
            assert np.all(gt[:, 1] + gt[:, 3] <= H)


class TestStressTestFactory:
    def test_returns_correct_length(self):
        ds = SyntheticDataset.stress_test(n_sequences=6)
        assert len(ds) == 6

    def test_mixed_motions_in_names(self):
        ds = SyntheticDataset.stress_test(n_sequences=9)
        names = [ds[i].name for i in range(9)]
        assert any("linear" in n for n in names)
        assert any("sinusoidal" in n for n in names)
        assert any("random_walk" in n for n in names)

    def test_frames_are_iterable(self):
        ds = SyntheticDataset.stress_test(n_sequences=2, n_frames=10)
        for seq in ds:
            frames = list(seq)
            assert len(frames) == 10


class TestScaleChallengeFactory:
    def test_returns_correct_length(self):
        ds = SyntheticDataset.scale_challenge(n_sequences=4)
        assert len(ds) == 4

    def test_growing_sequence_increases_size(self):
        ds = SyntheticDataset.scale_challenge(n_sequences=2, n_frames=50)
        grow_seq = ds[0]   # even index → growing
        gt = grow_seq.ground_truth
        assert gt[-1, 2] > gt[0, 2], "growing sequence should have larger w at end"

    def test_shrinking_sequence_decreases_size(self):
        ds = SyntheticDataset.scale_challenge(n_sequences=2, n_frames=50)
        shrink_seq = ds[1]  # odd index → shrinking
        gt = shrink_seq.ground_truth
        assert gt[-1, 2] < gt[0, 2], "shrinking sequence should have smaller w at end"


# ---------------------------------------------------------------------------
# Integration with BenchmarkEngine
# ---------------------------------------------------------------------------

class TestIntegrationWithBenchmarkEngine:
    def test_engine_runs_without_real_data(self):
        """SyntheticDataset must be accepted by BenchmarkEngine without any
        dataset files on disk."""
        try:
            from eovot.benchmark.engine import BenchmarkEngine
        except ImportError:
            pytest.skip("BenchmarkEngine not importable")

        from eovot.trackers.base import BaseTracker

        class DummyTracker(BaseTracker):
            def __init__(self):
                super().__init__(name="Dummy")
                self._bbox = None

            def initialize(self, frame, bbox):
                self._bbox = bbox

            def update(self, frame):
                return self._bbox

        dataset = SyntheticDataset.quick(n_sequences=3, n_frames=10, seed=0)
        engine = BenchmarkEngine(verbose=False)
        tracker = DummyTracker()
        result = engine.run(tracker, dataset, dataset_name="Synthetic")

        assert len(result.sequence_results) == 3
        assert result.mean_fps > 0.0
        # A perfect tracker (returns init bbox which equals frame-0 GT) should
        # have mIoU=1.0 only if GT doesn't change; here GT moves so mIoU<1 is ok.
        assert 0.0 <= result.mean_iou <= 1.0

    def test_known_gt_motion_gives_expected_iou(self):
        """A tracker that perfectly follows the ground truth should achieve mIoU=1.0."""
        try:
            from eovot.benchmark.engine import BenchmarkEngine
        except ImportError:
            pytest.skip("BenchmarkEngine not importable")

        from eovot.trackers.base import BaseTracker

        class OracleTracker(BaseTracker):
            """Returns the exact GT box — simulates perfect tracking."""

            def __init__(self):
                super().__init__(name="Oracle")
                self._gt: list = []
                self._frame_idx = 0

            def initialize(self, frame, bbox):
                self._frame_idx = 1  # will be called for frames 1..N

            def _set_gt(self, gt):
                self._gt = gt

            def update(self, frame):
                idx = self._frame_idx
                self._frame_idx += 1
                return tuple(self._gt[idx])

        cfg = _default_cfg(n_frames=15, motion="linear", velocity=(0.0, 0.0))
        dataset = SyntheticDataset([cfg])
        engine = BenchmarkEngine(verbose=False)

        tracker = OracleTracker()
        tracker._set_gt(dataset[0].ground_truth)

        result = engine.run(tracker, dataset, dataset_name="Synthetic")
        assert result.mean_iou == pytest.approx(1.0, abs=0.01)
