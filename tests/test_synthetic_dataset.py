"""Unit and integration tests for eovot.datasets.synthetic."""

from __future__ import annotations

import numpy as np
import pytest

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

    def test_resolve_init_bbox_default_centres_target(self):
        cfg = SyntheticSequenceConfig(frame_size=(200, 400))
        x, y, w, h = cfg.resolve_init_bbox()
        cx, cy = x + w / 2, y + h / 2
        assert abs(cx - 200) < 5 and abs(cy - 100) < 5

    def test_default_motion_is_linear(self):
        assert SyntheticSequenceConfig().motion == "linear"

    def test_default_appearance_is_solid(self):
        assert SyntheticSequenceConfig().appearance == "solid"


# ---------------------------------------------------------------------------
# _compute_ground_truth
# ---------------------------------------------------------------------------

class TestComputeGroundTruth:
    def test_shape(self):
        gt = _compute_ground_truth(_default_cfg(n_frames=30))
        assert gt.shape == (30, 4)

    def test_first_frame_matches_init_bbox(self):
        cfg = _default_cfg(init_bbox=(40.0, 30.0, 40.0, 30.0), velocity=(0.0, 0.0))
        gt = _compute_ground_truth(cfg)
        np.testing.assert_allclose(gt[0], [40.0, 30.0, 40.0, 30.0], atol=1e-6)

    def test_linear_motion_advances_position(self):
        cfg = _default_cfg(motion="linear", velocity=(3.0, 2.0))
        gt = _compute_ground_truth(cfg)
        cx0 = gt[0, 0] + gt[0, 2] / 2
        cy0 = gt[0, 1] + gt[0, 3] / 2
        cx1 = gt[1, 0] + gt[1, 2] / 2
        cy1 = gt[1, 1] + gt[1, 3] / 2
        assert abs((cx1 - cx0) - 3.0) < 1.0   # may bounce
        assert abs((cy1 - cy0) - 2.0) < 1.0

    def test_sinusoidal_motion_oscillates(self):
        cfg = _default_cfg(motion="sinusoidal", amplitude=(50.0, 30.0),
                           frequency=0.1, n_frames=50)
        gt = _compute_ground_truth(cfg)
        cx = gt[:, 0] + gt[:, 2] / 2
        assert cx.max() - cx.min() > 10.0

    def test_random_walk_reproducible_with_seed(self):
        cfg = _default_cfg(motion="random_walk", seed=42, n_frames=40)
        gt1 = _compute_ground_truth(cfg)
        gt2 = _compute_ground_truth(cfg)
        np.testing.assert_array_equal(gt1, gt2)

    def test_random_walk_different_seeds_differ(self):
        cfg1 = _default_cfg(motion="random_walk", seed=1, n_frames=40)
        cfg2 = _default_cfg(motion="random_walk", seed=2, n_frames=40)
        assert not np.allclose(_compute_ground_truth(cfg1), _compute_ground_truth(cfg2))

    def test_boxes_stay_within_frame(self):
        cfg = _default_cfg(n_frames=80, frame_size=(120, 160),
                           motion="linear", velocity=(10.0, 8.0))
        gt = _compute_ground_truth(cfg)
        H, W = cfg.frame_size
        x, y, w, h = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
        assert np.all(x >= 0)
        assert np.all(y >= 0)
        assert np.all(x + w <= W)
        assert np.all(y + h <= H)

    def test_scale_factor_grows_target(self):
        cfg = _default_cfg(scale_factor=1.02, n_frames=30)
        gt = _compute_ground_truth(cfg)
        assert gt[-1, 2] > gt[0, 2]

    def test_scale_factor_shrinks_target(self):
        cfg = _default_cfg(scale_factor=0.98, n_frames=30)
        gt = _compute_ground_truth(cfg)
        assert gt[-1, 2] < gt[0, 2]

    def test_all_widths_positive(self):
        for motion in ("linear", "sinusoidal", "random_walk"):
            cfg = _default_cfg(motion=motion, n_frames=30)
            gt = _compute_ground_truth(cfg)
            assert np.all(gt[:, 2] > 0), f"zero width with motion={motion}"

    def test_invalid_motion_raises(self):
        cfg = _default_cfg()
        cfg.motion = "teleport"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown motion model"):
            _compute_ground_truth(cfg)


# ---------------------------------------------------------------------------
# _render_frame
# ---------------------------------------------------------------------------

class TestRenderFrame:
    def test_output_shape_and_dtype(self):
        cfg = _default_cfg()
        frame = _render_frame(cfg, (40.0, 30.0, 40.0, 30.0), None)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8

    def test_solid_appearance_paints_target_color(self):
        cfg = _default_cfg(appearance="solid", target_color=(0, 0, 255))
        frame = _render_frame(cfg, (40.0, 30.0, 40.0, 30.0), None)
        # Interior pixel must be the target colour.
        np.testing.assert_array_equal(frame[35, 55], [0, 0, 255])

    def test_background_color_applied(self):
        cfg = _default_cfg(bg_color=(200, 200, 200))
        frame = _render_frame(cfg, (40.0, 30.0, 40.0, 30.0), None)
        np.testing.assert_array_equal(frame[0, 0], [200, 200, 200])

    def test_checkerboard_has_multiple_colors(self):
        cfg = _default_cfg(appearance="checkerboard")
        frame = _render_frame(cfg, (20.0, 20.0, 60.0, 40.0), None)
        patch = frame[20:60, 20:80]
        unique = set(map(tuple, patch.reshape(-1, 3).tolist()))
        assert len(unique) >= 2

    def test_gradient_appearance_valid(self):
        cfg = _default_cfg(appearance="gradient")
        frame = _render_frame(cfg, (20.0, 20.0, 60.0, 40.0), None)
        assert frame.dtype == np.uint8

    def test_noise_added_when_rng_provided(self):
        cfg = _default_cfg(add_noise=True, noise_sigma=20.0)
        rng = np.random.default_rng(0)
        noisy = _render_frame(cfg, (40.0, 30.0, 40.0, 30.0), rng)
        assert noisy.std() > 0.0

    def test_zero_area_bbox_does_not_crash(self):
        # _render_frame clamps w/h to a minimum of 1 px so it renders a 1×1 target;
        # what matters is the function returns a valid frame without raising.
        cfg = _default_cfg(bg_color=(100, 100, 100))
        frame = _render_frame(cfg, (10.0, 10.0, 0.0, 0.0), None)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# In-memory sequence (_SyntheticSequence via SyntheticDataset)
# ---------------------------------------------------------------------------

class TestSyntheticSequence:
    def _make_seq(self, **kwargs):
        cfg = _default_cfg(**kwargs)
        return SyntheticDataset([cfg])[0]

    def test_len_matches_n_frames(self):
        seq = self._make_seq(n_frames=25)
        assert len(seq) == 25

    def test_ground_truth_shape(self):
        seq = self._make_seq(n_frames=30)
        assert seq.ground_truth.shape == (30, 4)

    def test_iter_yields_correct_count(self):
        seq = self._make_seq(n_frames=10)
        assert sum(1 for _ in seq) == 10

    def test_frames_have_correct_shape(self):
        seq = self._make_seq(n_frames=5)
        for frame in seq:
            assert frame.shape == (120, 160, 3)
            assert frame.dtype == np.uint8

    def test_init_bbox_matches_gt_frame0(self):
        seq = self._make_seq()
        assert seq.init_bbox == pytest.approx(tuple(seq.ground_truth[0]))

    def test_name_propagated(self):
        seq = self._make_seq()
        assert seq.name == "test_seq"


# ---------------------------------------------------------------------------
# SyntheticDataset
# ---------------------------------------------------------------------------

class TestSyntheticDataset:
    def test_len_matches_configs(self):
        configs = [_default_cfg(name=f"s{i}") for i in range(4)]
        ds = SyntheticDataset(configs)
        assert len(ds) == 4

    def test_getitem_returns_sequence(self):
        ds = SyntheticDataset([_default_cfg()])
        from eovot.datasets.base import Sequence
        assert isinstance(ds[0], Sequence)

    def test_iter_yields_all_sequences(self):
        configs = [_default_cfg(name=f"s{i}") for i in range(3)]
        ds = SyntheticDataset(configs)
        assert len(list(ds)) == 3

    def test_repr(self):
        ds = SyntheticDataset([_default_cfg()])
        assert "SyntheticDataset" in repr(ds)
        assert "sequences=1" in repr(ds)

    def test_is_base_dataset(self):
        from eovot.datasets.base import BaseDataset
        ds = SyntheticDataset([_default_cfg()])
        assert isinstance(ds, BaseDataset)

    def test_empty_dataset(self):
        ds = SyntheticDataset([])
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

class TestQuickFactory:
    def test_returns_correct_length(self):
        ds = SyntheticDataset.quick(n_sequences=4)
        assert len(ds) == 4

    def test_frames_per_sequence(self):
        ds = SyntheticDataset.quick(n_sequences=2, n_frames=30)
        for seq in ds:
            assert len(seq) == 30

    def test_frame_size_respected(self):
        ds = SyntheticDataset.quick(n_sequences=1, frame_size=(120, 160))
        frame = next(iter(ds[0]))
        assert frame.shape == (120, 160, 3)

    def test_reproducible_with_seed(self):
        ds1 = SyntheticDataset.quick(n_sequences=3, seed=7)
        ds2 = SyntheticDataset.quick(n_sequences=3, seed=7)
        for seq1, seq2 in zip(ds1, ds2):
            np.testing.assert_array_equal(seq1.ground_truth, seq2.ground_truth)

    def test_different_seeds_differ(self):
        ds1 = SyntheticDataset.quick(n_sequences=3, seed=1)
        ds2 = SyntheticDataset.quick(n_sequences=3, seed=99)
        any_differ = any(
            not np.allclose(s1.ground_truth, s2.ground_truth)
            for s1, s2 in zip(ds1, ds2)
        )
        assert any_differ

    def test_gt_boxes_within_frame(self):
        ds = SyntheticDataset.quick(n_sequences=3, frame_size=(120, 160), seed=0)
        H, W = 120, 160
        for seq in ds:
            gt = seq.ground_truth
            assert np.all(gt[:, 0] >= 0)
            assert np.all(gt[:, 1] >= 0)
            assert np.all(gt[:, 0] + gt[:, 2] <= W)
            assert np.all(gt[:, 1] + gt[:, 3] <= H)

    def test_sinusoidal_motion_variant(self):
        ds = SyntheticDataset.quick(n_sequences=2, motion="sinusoidal", n_frames=40)
        assert len(ds) == 2

    def test_random_walk_motion_variant(self):
        ds = SyntheticDataset.quick(n_sequences=2, motion="random_walk", seed=5)
        for seq in ds:
            assert seq.ground_truth.shape[0] == 60  # default n_frames


class TestStressTestFactory:
    def test_returns_correct_length(self):
        assert len(SyntheticDataset.stress_test(n_sequences=6)) == 6

    def test_mixed_motions_in_names(self):
        ds = SyntheticDataset.stress_test(n_sequences=9)
        names = [ds[i].name for i in range(len(ds))]
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
        assert len(SyntheticDataset.scale_challenge(n_sequences=4)) == 4

    def test_growing_sequence_increases_size(self):
        ds = SyntheticDataset.scale_challenge(n_sequences=2)
        grow_seq = ds[0]  # even index → growing
        gt = grow_seq.ground_truth
        assert gt[-1, 2] > gt[0, 2], "target width should grow over time"

    def test_shrinking_sequence_decreases_size(self):
        ds = SyntheticDataset.scale_challenge(n_sequences=2)
        shrink_seq = ds[1]  # odd index → shrinking
        gt = shrink_seq.ground_truth
        assert gt[-1, 2] < gt[0, 2], "target width should shrink over time"


# ---------------------------------------------------------------------------
# Integration: full benchmark pipeline
# ---------------------------------------------------------------------------

class TestIntegrationWithBenchmarkEngine:
    def test_engine_runs_without_real_data(self):
        from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset.quick(n_sequences=2, n_frames=20,
                                    frame_size=(120, 160), seed=0)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")

        assert isinstance(result, BenchmarkResult)
        assert len(result.sequence_results) == 2
        assert result.mean_fps > 0.0
        assert 0.0 <= result.mean_iou <= 1.0

    def test_known_gt_motion_gives_expected_iou(self):
        """A static tracker predicting the init bbox on a static sequence scores IoU=1."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.base import BaseTracker

        FIXED: tuple = (40.0, 30.0, 40.0, 30.0)

        class StaticTracker(BaseTracker):
            def __init__(self):
                super().__init__("static")
            def initialize(self, frame, bbox): pass
            def update(self, frame): return FIXED

        cfg = SyntheticSequenceConfig(
            name="static_seq", n_frames=20, frame_size=(120, 160),
            init_bbox=FIXED, motion="linear", velocity=(0.0, 0.0),
        )
        ds = SyntheticDataset([cfg])
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(StaticTracker(), ds, dataset_name="Synthetic")
        assert result.mean_iou == pytest.approx(1.0)

    def test_kcf_achieves_nonzero_iou(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.kcf import KCFTracker

        ds = SyntheticDataset.quick(n_sequences=1, n_frames=25,
                                    frame_size=(120, 160),
                                    motion="sinusoidal", seed=42)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(KCFTracker(), ds, dataset_name="Synthetic")
        assert result.mean_iou > 0.0

    def test_results_are_reproducible(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.trackers.mosse import MOSSETracker

        def run():
            ds = SyntheticDataset.quick(n_sequences=1, n_frames=15, seed=7)
            return BenchmarkEngine(verbose=False).run(
                MOSSETracker(), ds, dataset_name="Synthetic"
            ).mean_iou

        assert run() == pytest.approx(run(), abs=1e-6)
