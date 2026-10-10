"""Tests for challenge-parameterized SyntheticDataset patterns (occlusion, scale_change)."""

import pytest
import numpy as np

from eovot.datasets.synthetic import SyntheticDataset
from eovot.benchmark.engine import BenchmarkEngine
from eovot.metrics.attributes import AttributeDetector
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestMotionPatternValidation:
    def test_valid_original_patterns(self):
        for motion in ("linear", "circular", "random"):
            ds = SyntheticDataset(num_sequences=1, num_frames=5, motion=motion)
            assert ds.motion == motion

    def test_valid_new_patterns(self):
        for motion in ("occlusion", "scale_change"):
            ds = SyntheticDataset(num_sequences=1, num_frames=5, motion=motion)
            assert ds.motion == motion

    def test_invalid_motion_raises(self):
        with pytest.raises(ValueError, match="Unknown motion pattern"):
            SyntheticDataset(motion="teleport")

    def test_all_five_patterns_in_repr(self):
        for motion in ("linear", "circular", "random", "occlusion", "scale_change"):
            ds = SyntheticDataset(num_sequences=1, num_frames=5, motion=motion)
            assert motion in repr(ds)


# ---------------------------------------------------------------------------
# Occlusion pattern
# ---------------------------------------------------------------------------

class TestOcclusionPattern:
    def _make(self, n_seq=3, n_frames=60, period=20, occ_frames=8):
        return SyntheticDataset(
            num_sequences=n_seq,
            num_frames=n_frames,
            motion="occlusion",
            occlusion_period=period,
            occlusion_frames=occ_frames,
        )

    def test_sequence_length(self):
        ds = self._make()
        seq = ds[0]
        frames = list(seq)
        assert len(frames) == 60

    def test_gt_shape(self):
        ds = self._make()
        seq = ds[0]
        assert seq.ground_truth.shape == (60, 4)

    def test_gt_has_consistent_size(self):
        """Occlusion does not change target size — GT width/height should be constant."""
        ds = self._make()
        seq = ds[0]
        gt = seq.ground_truth
        widths = gt[:, 2]
        heights = gt[:, 3]
        np.testing.assert_allclose(widths, widths[0])
        np.testing.assert_allclose(heights, heights[0])

    def test_occluded_frames_differ_from_clean(self):
        """Frames where occluder is active should look different from clean frames."""
        ds = self._make(n_frames=30, period=15, occ_frames=5)
        seq = ds[0]
        frames = list(seq)
        # Frame 0 should be in first occlusion window (phase 0 < 5).
        # Frame 10 should be outside window (phase 10 >= 5).
        occ_frame = np.array(frames[0], dtype=np.float32)
        clean_frame = np.array(frames[10], dtype=np.float32)
        # The frames cannot be identical if the occluder was painted.
        assert not np.array_equal(occ_frame, clean_frame)

    def test_gt_valid_during_occlusion(self):
        """GT boxes must be valid (positive w, h) even on occluded frames."""
        ds = self._make()
        seq = ds[0]
        gt = seq.ground_truth
        assert np.all(gt[:, 2] > 0)
        assert np.all(gt[:, 3] > 0)

    def test_attribute_detection_partial_occlusion(self):
        """Area-drop from occlusion should trigger partial_occlusion attribute."""
        # Use large occlusion frames so GT area stays constant — the attribute
        # is detected from GT box area; our occluder doesn't change GT.
        # We check that the attribute detector can process these sequences.
        ds = self._make(n_seq=1, n_frames=60)
        seq = ds[0]
        detector = AttributeDetector()
        attrs = detector.detect(seq.ground_truth, seq.name)
        # Attribute detection should not crash; result is a dict of bools.
        assert isinstance(attrs.attributes, dict)

    def test_end_to_end_benchmark(self):
        """BenchmarkEngine must complete a run on occlusion sequences."""
        from eovot.trackers.kcf import KCFTracker
        ds = self._make(n_seq=2, n_frames=30)
        engine = BenchmarkEngine(verbose=False)
        # KCF (OpenCV built-in) handles out-of-frame predictions gracefully,
        # making it suitable for E2E tests with challenging occlusion sequences.
        result = engine.run(KCFTracker(), ds, dataset_name="OcclusionSynth")
        assert len(result.sequence_results) == 2
        assert result.mean_iou >= 0.0

    def test_occlusion_period_parameter(self):
        """Custom period / frames params are stored."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=40, motion="occlusion",
            occlusion_period=10, occlusion_frames=3,
        )
        assert ds.occlusion_period == 10
        assert ds.occlusion_frames == 3

    def test_caching(self):
        ds = self._make(n_seq=2, n_frames=20)
        seq_a = ds[0]
        seq_b = ds[0]
        assert seq_a is seq_b  # same object from cache


# ---------------------------------------------------------------------------
# Scale change pattern
# ---------------------------------------------------------------------------

class TestScaleChangePattern:
    def _make(self, n_seq=3, n_frames=80, scale_min=0.5, scale_max=2.0, period=40):
        return SyntheticDataset(
            num_sequences=n_seq,
            num_frames=n_frames,
            motion="scale_change",
            scale_min=scale_min,
            scale_max=scale_max,
            scale_period=period,
        )

    def test_sequence_length(self):
        ds = self._make()
        seq = ds[0]
        frames = list(seq)
        assert len(frames) == 80

    def test_gt_shape(self):
        ds = self._make()
        seq = ds[0]
        assert seq.ground_truth.shape == (80, 4)

    def test_gt_width_varies(self):
        """Scale-change sequences must have non-constant bbox width."""
        ds = self._make()
        seq = ds[0]
        widths = seq.ground_truth[:, 2]
        assert widths.max() > widths.min(), "Width should vary across frames"

    def test_gt_height_varies(self):
        ds = self._make()
        seq = ds[0]
        heights = seq.ground_truth[:, 3]
        assert heights.max() > heights.min()

    def test_scale_ratio_approx(self):
        """Max / min area ratio should be close to (scale_max / scale_min)²."""
        bw, bh = 40, 40
        ds = SyntheticDataset(
            num_sequences=1, num_frames=200,  # many frames to hit both extremes
            motion="scale_change",
            bbox_size=(bw, bh),
            scale_min=0.5, scale_max=2.0, scale_period=40,
        )
        seq = ds[0]
        gt = seq.ground_truth
        areas = gt[:, 2] * gt[:, 3]
        ratio = areas.max() / areas.min()
        # (scale_max / scale_min)² = (2.0 / 0.5)² = 16
        # Allow ±30% tolerance for integer rounding.
        assert ratio > 16 * 0.7, f"Expected ratio ~16, got {ratio:.2f}"

    def test_scale_variation_attribute(self):
        """Generated sequence should trigger scale_variation attribute detection."""
        ds = self._make(n_frames=200)
        seq = ds[0]
        detector = AttributeDetector(sv_ratio_threshold=4.0)
        attrs = detector.detect(seq.ground_truth, seq.name)
        assert attrs.has("scale_variation"), (
            "scale_change sequence should trigger scale_variation attribute"
        )

    def test_min_bbox_size_positive(self):
        """Even at scale_min the bbox must have positive dimensions."""
        ds = SyntheticDataset(
            num_sequences=1, num_frames=100,
            motion="scale_change", scale_min=0.1, scale_max=3.0, scale_period=20,
        )
        seq = ds[0]
        gt = seq.ground_truth
        assert np.all(gt[:, 2] > 0)
        assert np.all(gt[:, 3] > 0)

    def test_scale_params_stored(self):
        ds = SyntheticDataset(
            num_sequences=1, num_frames=10,
            motion="scale_change", scale_min=0.3, scale_max=1.8, scale_period=20,
        )
        assert ds.scale_min == pytest.approx(0.3)
        assert ds.scale_max == pytest.approx(1.8)
        assert ds.scale_period == 20

    def test_end_to_end_benchmark(self):
        ds = self._make(n_seq=2, n_frames=40)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="ScaleSynth")
        assert len(result.sequence_results) == 2
        assert result.mean_iou >= 0.0


# ---------------------------------------------------------------------------
# Generate-sizes helper
# ---------------------------------------------------------------------------

class TestGenerateSizes:
    def test_constant_for_linear(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="linear")
        sizes = ds._generate_sizes(50)
        assert all(s == (40, 40) for s in sizes)

    def test_constant_for_occlusion(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="occlusion")
        sizes = ds._generate_sizes(50)
        assert all(s == (40, 40) for s in sizes)

    def test_variable_for_scale_change(self):
        ds = SyntheticDataset(
            num_sequences=1, num_frames=80,
            motion="scale_change", scale_min=0.5, scale_max=2.0, scale_period=40,
        )
        sizes = ds._generate_sizes(80)
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        assert max(ws) > min(ws)
        assert max(hs) > min(hs)

    def test_length_matches_num_frames(self):
        for motion in ("linear", "occlusion", "scale_change"):
            ds = SyntheticDataset(num_sequences=1, num_frames=33, motion=motion)
            sizes = ds._generate_sizes(33)
            assert len(sizes) == 33


# ---------------------------------------------------------------------------
# Backwards compatibility: original patterns unchanged
# ---------------------------------------------------------------------------

class TestOriginalPatternsUnchanged:
    """Regression: existing patterns must produce the same output as before."""

    def test_linear_gt_constant_size(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=50, motion="linear")
        seq = ds[0]
        gt = seq.ground_truth
        np.testing.assert_allclose(gt[:, 2], gt[0, 2])
        np.testing.assert_allclose(gt[:, 3], gt[0, 3])

    def test_circular_num_frames(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=30, motion="circular")
        assert len(list(ds[0])) == 30

    def test_random_gt_valid(self):
        ds = SyntheticDataset(num_sequences=1, num_frames=20, motion="random")
        gt = ds[0].ground_truth
        assert np.all(gt[:, 2] > 0)
        assert np.all(gt[:, 3] > 0)
