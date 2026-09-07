"""Tests for DegradedDataset — controllable frame degradations for edge robustness evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.datasets.degraded import DegradationConfig, DegradedDataset, DegradedSequence
from eovot.datasets.base import BaseDataset, Sequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_ds():
    return SyntheticDataset(num_sequences=3, num_frames=10, motion="linear", seed=11)


@pytest.fixture()
def clean_frame(base_ds):
    seq = base_ds[0]
    return next(iter(seq))


# ---------------------------------------------------------------------------
# DegradationConfig validation
# ---------------------------------------------------------------------------

def test_default_config_is_identity():
    cfg = DegradationConfig()
    assert cfg.is_identity


def test_noise_config_not_identity():
    cfg = DegradationConfig(noise_std=10.0)
    assert not cfg.is_identity


def test_invalid_noise_std():
    with pytest.raises(ValueError, match="noise_std"):
        DegradationConfig(noise_std=-1.0)


def test_invalid_blur_even_kernel():
    with pytest.raises(ValueError, match="blur_kernel_size"):
        DegradationConfig(blur_kernel_size=4)


def test_invalid_blur_small_kernel():
    with pytest.raises(ValueError, match="blur_kernel_size"):
        DegradationConfig(blur_kernel_size=1)


def test_invalid_occlusion_fraction():
    with pytest.raises(ValueError, match="occlusion_fraction"):
        DegradationConfig(occlusion_fraction=1.0)


def test_invalid_brightness_alpha():
    with pytest.raises(ValueError, match="brightness_alpha"):
        DegradationConfig(brightness_alpha=-0.5)


def test_valid_configs_do_not_raise():
    DegradationConfig(noise_std=15.0)
    DegradationConfig(blur_kernel_size=5)
    DegradationConfig(brightness_alpha=0.5, brightness_beta=-30.0)
    DegradationConfig(occlusion_fraction=0.1)


# ---------------------------------------------------------------------------
# DegradedDataset interface
# ---------------------------------------------------------------------------

def test_is_base_dataset(base_ds):
    dd = DegradedDataset(base_ds)
    assert isinstance(dd, BaseDataset)


def test_length_preserved(base_ds):
    dd = DegradedDataset(base_ds, DegradationConfig(noise_std=5.0))
    assert len(dd) == len(base_ds)


def test_getitem_returns_degraded_sequence(base_ds):
    dd = DegradedDataset(base_ds, DegradationConfig(noise_std=5.0))
    seq = dd[0]
    assert isinstance(seq, DegradedSequence)


def test_ground_truth_unchanged(base_ds):
    cfg = DegradationConfig(noise_std=20.0)
    dd = DegradedDataset(base_ds, cfg)
    orig_gt = base_ds[0].ground_truth.copy()
    deg_gt = dd[0].ground_truth
    np.testing.assert_array_equal(orig_gt, deg_gt)


def test_sequence_length_preserved(base_ds):
    dd = DegradedDataset(base_ds, DegradationConfig(blur_kernel_size=3))
    for i in range(len(base_ds)):
        assert len(dd[i]) == len(base_ds[i])


def test_frame_shape_preserved(base_ds):
    dd = DegradedDataset(base_ds, DegradationConfig(noise_std=10.0))
    orig_frame = next(iter(base_ds[0]))
    deg_frame = next(iter(dd[0]))
    assert orig_frame.shape == deg_frame.shape


def test_frame_dtype_is_uint8(base_ds):
    dd = DegradedDataset(base_ds, DegradationConfig(noise_std=25.0))
    deg_frame = next(iter(dd[0]))
    assert deg_frame.dtype == np.uint8


def test_identity_config_returns_identical_frames(base_ds):
    dd = DegradedDataset(base_ds, DegradationConfig())
    for orig, deg in zip(iter(base_ds[0]), iter(dd[0])):
        np.testing.assert_array_equal(orig, deg)


# ---------------------------------------------------------------------------
# Degradation effects — noise
# ---------------------------------------------------------------------------

def test_noise_changes_pixels(clean_frame, base_ds):
    cfg = DegradationConfig(noise_std=30.0)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert not np.array_equal(clean_frame, deg_frame)


def test_noise_output_in_valid_range(clean_frame, base_ds):
    cfg = DegradationConfig(noise_std=50.0)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert deg_frame.min() >= 0
    assert deg_frame.max() <= 255


def test_noise_reproducible(base_ds):
    cfg = DegradationConfig(noise_std=20.0, seed=42)
    dd1 = DegradedDataset(base_ds, cfg)
    dd2 = DegradedDataset(base_ds, cfg)
    frame1 = next(iter(dd1[0]))
    frame2 = next(iter(dd2[0]))
    np.testing.assert_array_equal(frame1, frame2)


def test_noise_differs_across_frames(base_ds):
    cfg = DegradationConfig(noise_std=20.0, seed=7)
    dd = DegradedDataset(base_ds, cfg)
    frames = list(iter(dd[0]))
    # Adjacent frames must differ in at least one pixel (different RNG seeds)
    assert not np.array_equal(frames[0], frames[1])


# ---------------------------------------------------------------------------
# Degradation effects — motion blur
# ---------------------------------------------------------------------------

def test_blur_changes_pixels(clean_frame, base_ds):
    cfg = DegradationConfig(blur_kernel_size=7)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert not np.array_equal(clean_frame, deg_frame)


def test_blur_output_in_valid_range(base_ds):
    cfg = DegradationConfig(blur_kernel_size=5)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert deg_frame.min() >= 0
    assert deg_frame.max() <= 255


# ---------------------------------------------------------------------------
# Degradation effects — brightness / contrast
# ---------------------------------------------------------------------------

def test_low_brightness_reduces_mean(clean_frame, base_ds):
    cfg = DegradationConfig(brightness_alpha=0.3, brightness_beta=-20.0)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert float(deg_frame.mean()) < float(clean_frame.mean())


def test_high_brightness_increases_mean(clean_frame, base_ds):
    cfg = DegradationConfig(brightness_alpha=1.5, brightness_beta=50.0)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert float(deg_frame.mean()) > float(clean_frame.mean())


# ---------------------------------------------------------------------------
# Degradation effects — occlusion
# ---------------------------------------------------------------------------

def test_occlusion_changes_pixels(clean_frame, base_ds):
    cfg = DegradationConfig(occlusion_fraction=0.1)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert not np.array_equal(clean_frame, deg_frame)


def test_occlusion_output_in_valid_range(base_ds):
    cfg = DegradationConfig(occlusion_fraction=0.15)
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert deg_frame.min() >= 0
    assert deg_frame.max() <= 255


def test_occlusion_reproducible(base_ds):
    cfg = DegradationConfig(occlusion_fraction=0.1, seed=99)
    dd1 = DegradedDataset(base_ds, cfg)
    dd2 = DegradedDataset(base_ds, cfg)
    frame1 = next(iter(dd1[0]))
    frame2 = next(iter(dd2[0]))
    np.testing.assert_array_equal(frame1, frame2)


# ---------------------------------------------------------------------------
# Stacked degradations
# ---------------------------------------------------------------------------

def test_stacked_degradations_all_apply(clean_frame, base_ds):
    cfg = DegradationConfig(
        noise_std=10.0,
        blur_kernel_size=3,
        brightness_alpha=0.8,
        brightness_beta=-10.0,
        occlusion_fraction=0.05,
    )
    dd = DegradedDataset(base_ds, cfg)
    deg_frame = next(iter(dd[0]))
    assert deg_frame.dtype == np.uint8
    assert deg_frame.shape == clean_frame.shape
    assert not np.array_equal(clean_frame, deg_frame)


# ---------------------------------------------------------------------------
# Integration: full benchmark run
# ---------------------------------------------------------------------------

def test_benchmark_run_with_noise():
    """DegradedDataset must integrate with BenchmarkEngine end-to-end."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    base = SyntheticDataset(num_sequences=2, num_frames=15, motion="linear", seed=3)
    ds = DegradedDataset(base, DegradationConfig(noise_std=10.0))
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="Degraded-Noise")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0
    assert len(result.sequence_results) == 2


def test_benchmark_run_with_blur():
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    base = SyntheticDataset(num_sequences=2, num_frames=15, motion="linear", seed=5)
    ds = DegradedDataset(base, DegradationConfig(blur_kernel_size=5))
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="Degraded-Blur")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0


def test_benchmark_run_with_occlusion():
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    base = SyntheticDataset(num_sequences=2, num_frames=15, motion="linear", seed=6)
    ds = DegradedDataset(base, DegradationConfig(occlusion_fraction=0.08))
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="Degraded-Occlusion")

    assert result.mean_iou >= 0.0
    assert result.mean_fps > 0.0
