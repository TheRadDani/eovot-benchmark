"""Unit tests for eovot.datasets.degradation."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.degradation import (
    DegradationConfig,
    DegradationType,
    DegradedSequence,
    DegradationSweep,
    ImageDegrader,
    SweepPoint,
)
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gray_frame(h: int = 64, w: int = 64, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def _seq():
    return SyntheticDataset(num_sequences=1, num_frames=5)[0]


# ---------------------------------------------------------------------------
# DegradationConfig
# ---------------------------------------------------------------------------


class TestDegradationConfig:
    def test_valid_severity_zero(self):
        c = DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.0)
        assert c.severity == 0.0

    def test_valid_severity_one(self):
        c = DegradationConfig(DegradationType.MOTION_BLUR, severity=1.0)
        assert c.severity == 1.0

    def test_invalid_severity_negative(self):
        with pytest.raises(ValueError, match="severity"):
            DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=-0.1)

    def test_invalid_severity_above_one(self):
        with pytest.raises(ValueError, match="severity"):
            DegradationConfig(DegradationType.MOTION_BLUR, severity=1.5)

    def test_disabled_config(self):
        c = DegradationConfig(DegradationType.BRIGHTNESS, severity=1.0, enabled=False)
        assert not c.enabled

    def test_default_severity(self):
        c = DegradationConfig(DegradationType.SALT_PEPPER)
        assert c.severity == 0.5
        assert c.enabled is True


# ---------------------------------------------------------------------------
# ImageDegrader — shape, dtype, and value invariants
# ---------------------------------------------------------------------------


class TestImageDegrader:
    @pytest.mark.parametrize("deg_type", list(DegradationType))
    def test_preserves_shape(self, deg_type):
        frame = _gray_frame()
        cfg = DegradationConfig(deg_type, severity=0.5)
        degrader = ImageDegrader([cfg], rng_seed=0)
        out = degrader.apply(frame)
        assert out.shape == frame.shape

    @pytest.mark.parametrize("deg_type", list(DegradationType))
    def test_output_dtype_uint8(self, deg_type):
        frame = _gray_frame()
        cfg = DegradationConfig(deg_type, severity=0.5)
        out = ImageDegrader([cfg], rng_seed=0).apply(frame)
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("deg_type", list(DegradationType))
    def test_output_in_valid_range(self, deg_type):
        frame = _gray_frame()
        cfg = DegradationConfig(deg_type, severity=0.9)
        out = ImageDegrader([cfg], rng_seed=0).apply(frame)
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_zero_severity_near_identity_for_noise_types(self):
        frame = _gray_frame(val=128)
        for t in [
            DegradationType.GAUSSIAN_NOISE,
            DegradationType.MOTION_BLUR,
            DegradationType.SALT_PEPPER,
            DegradationType.PIXELATION,
        ]:
            cfg = DegradationConfig(t, severity=0.0)
            out = ImageDegrader([cfg], rng_seed=0).apply(frame)
            diff = np.abs(out.astype(int) - frame.astype(int))
            assert diff.max() <= 2, f"{t} at severity=0 changed pixels by {diff.max()}"

    def test_brightness_neutral_at_half(self):
        frame = _gray_frame(val=100)
        cfg = DegradationConfig(DegradationType.BRIGHTNESS, severity=0.5)
        out = ImageDegrader([cfg], rng_seed=0).apply(frame)
        assert abs(float(out.mean()) - 100.0) <= 5.0

    def test_jpeg_high_severity_changes_frame(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        cfg = DegradationConfig(DegradationType.JPEG_COMPRESSION, severity=1.0)
        out = ImageDegrader([cfg], rng_seed=0).apply(frame)
        assert not np.array_equal(out, frame)

    def test_gaussian_noise_reproducible_with_seed(self):
        frame = _gray_frame()
        cfg = DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.5)
        out1 = ImageDegrader([cfg], rng_seed=7).apply(frame)
        out2 = ImageDegrader([cfg], rng_seed=7).apply(frame)
        assert np.array_equal(out1, out2)

    def test_gaussian_noise_varies_with_different_seed(self):
        frame = _gray_frame()
        cfg = DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.5)
        out1 = ImageDegrader([cfg], rng_seed=1).apply(frame)
        out2 = ImageDegrader([cfg], rng_seed=2).apply(frame)
        assert not np.array_equal(out1, out2)

    def test_disabled_config_skipped(self):
        frame = _gray_frame(val=128)
        cfg = DegradationConfig(
            DegradationType.GAUSSIAN_NOISE, severity=1.0, enabled=False
        )
        out = ImageDegrader([cfg], rng_seed=0).apply(frame)
        assert np.array_equal(out, frame)

    def test_empty_configs_is_identity(self):
        frame = _gray_frame(val=200)
        out = ImageDegrader([], rng_seed=0).apply(frame)
        assert np.array_equal(out, frame)

    def test_chained_degradations_preserve_shape(self):
        frame = _gray_frame()
        cfgs = [
            DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.3),
            DegradationConfig(DegradationType.MOTION_BLUR, severity=0.4),
        ]
        out = ImageDegrader(cfgs, rng_seed=0).apply(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_pixelation_max_severity_valid_output(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        cfg = DegradationConfig(DegradationType.PIXELATION, severity=1.0)
        out = ImageDegrader([cfg], rng_seed=0).apply(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# DegradedSequence
# ---------------------------------------------------------------------------


class TestDegradedSequence:
    def test_name_contains_degradation_type(self):
        seq = _seq()
        cfg = DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.5)
        ds = DegradedSequence(seq, ImageDegrader([cfg], rng_seed=0))
        assert "gaussian_noise" in ds.name

    def test_name_contains_source_name(self):
        seq = _seq()
        ds = DegradedSequence(seq, ImageDegrader([], rng_seed=0))
        assert seq.name in ds.name

    def test_len_matches_source(self):
        seq = _seq()
        ds = DegradedSequence(seq, ImageDegrader([], rng_seed=0))
        assert len(ds) == len(seq)

    def test_ground_truth_unchanged(self):
        seq = _seq()
        cfg = DegradationConfig(DegradationType.BRIGHTNESS, severity=1.0)
        ds = DegradedSequence(seq, ImageDegrader([cfg], rng_seed=0))
        np.testing.assert_array_equal(ds.ground_truth, seq.ground_truth)

    def test_init_bbox_unchanged(self):
        seq = _seq()
        cfg = DegradationConfig(DegradationType.MOTION_BLUR, severity=0.5)
        ds = DegradedSequence(seq, ImageDegrader([cfg], rng_seed=0))
        assert ds.init_bbox == seq.init_bbox

    def test_frames_are_uint8(self):
        seq = _seq()
        cfg = DegradationConfig(DegradationType.GAUSSIAN_NOISE, severity=0.5)
        ds = DegradedSequence(seq, ImageDegrader([cfg], rng_seed=0))
        for frame in ds:
            assert frame.dtype == np.uint8
            break

    def test_frame_count_matches_len(self):
        seq = _seq()
        ds = DegradedSequence(seq, ImageDegrader([], rng_seed=0))
        assert len(list(ds)) == len(seq)

    def test_repr_contains_name(self):
        seq = _seq()
        ds = DegradedSequence(seq, ImageDegrader([], rng_seed=0))
        assert seq.name in repr(ds)


# ---------------------------------------------------------------------------
# DegradationSweep
# ---------------------------------------------------------------------------


class TestDegradationSweep:
    @staticmethod
    def _trivial_eval(ds: DegradedSequence):
        return 0.5, 100.0

    def test_sweep_returns_correct_count(self):
        seq = _seq()
        sweep = DegradationSweep(
            severities=[0.0, 0.5, 1.0],
            types=[DegradationType.GAUSSIAN_NOISE, DegradationType.MOTION_BLUR],
        )
        results = sweep.run(seq, self._trivial_eval)
        assert len(results) == 6  # 2 types x 3 severities

    def test_sweep_point_degradation_type_field(self):
        seq = _seq()
        sweep = DegradationSweep(
            severities=[0.5],
            types=[DegradationType.BRIGHTNESS],
        )
        pts = sweep.run(seq, self._trivial_eval)
        assert pts[0].degradation_type == DegradationType.BRIGHTNESS.value

    def test_sweep_point_severity_matches(self):
        seq = _seq()
        sweep = DegradationSweep(
            severities=[0.25],
            types=[DegradationType.SALT_PEPPER],
        )
        pts = sweep.run(seq, self._trivial_eval)
        assert pts[0].severity == 0.25

    def test_to_markdown_has_header(self):
        pts = [SweepPoint("gaussian_noise", 0.5, 0.7, 120.0)]
        md = DegradationSweep.to_markdown(pts)
        assert "Degradation" in md
        assert "Severity" in md
        assert "mIoU" in md
        assert "FPS" in md
        assert "gaussian_noise" in md

    def test_sweep_point_to_dict_keys(self):
        pt = SweepPoint("motion_blur", 0.25, 0.65, 90.5)
        d = pt.to_dict()
        assert set(d.keys()) == {"degradation_type", "severity", "mean_iou", "mean_fps"}

    def test_default_five_severities(self):
        sweep = DegradationSweep()
        assert len(sweep.severities) == 5
        assert 0.0 in sweep.severities
        assert 1.0 in sweep.severities

    def test_all_six_types_by_default(self):
        sweep = DegradationSweep()
        assert set(sweep.types) == set(DegradationType)

    def test_eval_fn_receives_degraded_sequence(self):
        seq = _seq()
        received = []

        def capture_fn(ds):
            received.append(type(ds).__name__)
            return 0.0, 1.0

        DegradationSweep(
            severities=[0.5], types=[DegradationType.PIXELATION]
        ).run(seq, capture_fn)
        assert received == ["DegradedSequence"]
