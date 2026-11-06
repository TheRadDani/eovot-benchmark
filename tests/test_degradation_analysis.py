"""Tests for the image degradation robustness analysis module.

All tests run entirely in memory using synthetic sequences — no real dataset
download required, and no GPU or file I/O beyond what pytest itself needs.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pytest

from eovot.analysis.degradation_analysis import (
    DegradationReport,
    DegradationRobustnessAnalyzer,
    DegradationSweepResult,
    DegradationType,
    ImageDegradationPipeline,
)
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker

BBox = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic tracker and dataset
# ---------------------------------------------------------------------------

FIXED_BOX: BBox = (10.0, 10.0, 60.0, 60.0)
NUM_FRAMES = 15


class ConstantTracker(BaseTracker):
    """Always returns FIXED_BOX regardless of input."""

    def __init__(self, box: BBox = FIXED_BOX) -> None:
        self._box = box

    @property
    def name(self) -> str:
        return "ConstantTracker"

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: np.ndarray) -> BBox:
        return self._box


class _InMemorySequence(Sequence):
    def __init__(self, name: str, n_frames: int, gt_box: BBox) -> None:
        gt = np.tile(np.array(gt_box), (n_frames, 1))
        super().__init__(
            name=name,
            frame_paths=["<mem>"] * n_frames,
            ground_truth=gt,
        )
        self._frames = [
            np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(n_frames)
        ]

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self._frames


class SmallDataset(BaseDataset):
    def __init__(self, n: int = 2) -> None:
        self._seqs = [_InMemorySequence(f"s{i}", NUM_FRAMES, FIXED_BOX) for i in range(n)]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# ImageDegradationPipeline tests
# ---------------------------------------------------------------------------

class TestImageDegradationPipeline:
    def setup_method(self):
        self.pipeline = ImageDegradationPipeline(rng_seed=0)
        self.frame = np.full((60, 80, 3), 200, dtype=np.uint8)

    def test_zero_severity_returns_copy(self):
        out = self.pipeline.apply(self.frame, DegradationType.GAUSSIAN_NOISE, 0.0)
        np.testing.assert_array_equal(out, self.frame)

    def test_output_shape_preserved(self):
        for dt in DegradationType:
            for s in [0.0, 0.5, 1.0]:
                out = self.pipeline.apply(self.frame, dt, s)
                assert out.shape == self.frame.shape, f"{dt} severity={s}"

    def test_output_dtype_uint8(self):
        for dt in DegradationType:
            out = self.pipeline.apply(self.frame, dt, 0.5)
            assert out.dtype == np.uint8, f"{dt} changed dtype"

    def test_gaussian_noise_changes_pixels(self):
        out = self.pipeline.apply(self.frame, DegradationType.GAUSSIAN_NOISE, 0.5)
        assert not np.array_equal(out, self.frame)

    def test_brightness_reduction_darkens(self):
        out = self.pipeline.apply(self.frame, DegradationType.BRIGHTNESS_REDUCTION, 0.5)
        assert float(out.mean()) < float(self.frame.mean())

    def test_jpeg_max_severity_degrades(self):
        out = self.pipeline.apply(self.frame, DegradationType.JPEG_COMPRESSION, 1.0)
        # JPEG at quality=10 must produce a different image than the original
        assert not np.array_equal(out, self.frame)

    def test_severity_clipped_to_range(self):
        """Values outside [0,1] should not crash — they are clamped internally."""
        out = self.pipeline.apply(self.frame, DegradationType.GAUSSIAN_NOISE, 1.5)
        assert out.shape == self.frame.shape


# ---------------------------------------------------------------------------
# DegradationRobustnessAnalyzer tests
# ---------------------------------------------------------------------------

class TestDegradationRobustnessAnalyzer:
    def setup_method(self):
        self.dataset = SmallDataset(n=2)
        self.tracker = ConstantTracker()
        # Use a small severity sweep and only one degradation type for speed
        self.analyzer = DegradationRobustnessAnalyzer(
            severities=[0.0, 0.5, 1.0],
            degradation_types=[DegradationType.GAUSSIAN_NOISE],
        )

    def test_analyze_returns_report(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        assert isinstance(report, DegradationReport)

    def test_report_tracker_name(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        assert report.tracker_name == "ConstantTracker"

    def test_report_has_one_sweep_per_type(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        assert len(report.sweeps) == 1  # only GAUSSIAN_NOISE configured

    def test_sweep_has_correct_severity_count(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        sweep = report.sweeps[0]
        assert len(sweep.severities) == 3
        assert len(sweep.mean_ious) == 3

    def test_clean_iou_is_first_severity(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        sweep = report.sweeps[0]
        assert sweep.severities[0] == pytest.approx(0.0)
        assert sweep.clean_iou == pytest.approx(sweep.mean_ious[0])

    def test_dts_in_unit_interval(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        for sweep in report.sweeps:
            assert 0.0 <= sweep.dts <= 1.0

    def test_mean_dts_is_average_of_sweeps(self):
        analyzer = DegradationRobustnessAnalyzer(
            severities=[0.0, 0.5, 1.0],
            degradation_types=[
                DegradationType.GAUSSIAN_NOISE,
                DegradationType.GAUSSIAN_BLUR,
            ],
        )
        report = analyzer.analyze(self.tracker, self.dataset)
        expected = np.mean([s.dts for s in report.sweeps])
        assert report.mean_dts == pytest.approx(expected)

    def test_perfect_tracker_dts_is_one(self):
        """A tracker that always returns the exact GT box has DTS=1 at any severity."""
        # At zero severity the pipeline returns a clean copy, so IoU stays at 1.0
        # for all evaluated severities (noise doesn't affect the constant GT prediction).
        report = self.analyzer.analyze(self.tracker, self.dataset)
        for sweep in report.sweeps:
            assert sweep.dts == pytest.approx(1.0, abs=0.01)

    def test_compare_returns_sorted_reports(self):
        analyzer = DegradationRobustnessAnalyzer(
            severities=[0.0, 1.0],
            degradation_types=[DegradationType.GAUSSIAN_NOISE],
        )
        # Both trackers are ConstantTracker (perfect on this GT), so DTS≈1 for both
        reports = analyzer.compare(
            [ConstantTracker(), ConstantTracker()], self.dataset
        )
        assert len(reports) == 2
        assert reports[0].mean_dts >= reports[1].mean_dts

    def test_to_markdown_table(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        table = self.analyzer.to_markdown_table([report])
        assert "Tracker" in table
        assert "Gaussian Noise" in table
        assert "Mean DTS" in table
        assert "ConstantTracker" in table

    def test_half_life_none_for_robust_tracker(self):
        """A perfect tracker whose IoU never halves should have half_life=None."""
        report = self.analyzer.analyze(self.tracker, self.dataset)
        for sweep in report.sweeps:
            # IoU is 1.0 at all severities → never drops below 0.5 × 1.0 = 0.5
            assert sweep.half_life is None

    def test_all_degradation_types_run(self):
        """Analyzer with all degradation types must produce four sweeps."""
        analyzer = DegradationRobustnessAnalyzer(
            severities=[0.0, 1.0],
            degradation_types=list(DegradationType),
        )
        report = analyzer.analyze(self.tracker, self.dataset)
        assert len(report.sweeps) == len(DegradationType)
        types_seen = {s.degradation_type for s in report.sweeps}
        assert types_seen == set(DegradationType)

    def test_str_representation(self):
        report = self.analyzer.analyze(self.tracker, self.dataset)
        s = str(report)
        assert "DegradationReport" in s
        assert "ConstantTracker" in s


# ---------------------------------------------------------------------------
# DTS computation edge cases
# ---------------------------------------------------------------------------

class TestDTSComputation:
    def test_zero_clean_iou_gives_zero_dts(self):
        dts = DegradationRobustnessAnalyzer._compute_dts([0.0, 1.0], [0.0, 0.0], 0.0)
        assert dts == pytest.approx(0.0)

    def test_constant_full_iou_gives_one_dts(self):
        dts = DegradationRobustnessAnalyzer._compute_dts(
            [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], 1.0
        )
        assert dts == pytest.approx(1.0)

    def test_half_life_interpolated_correctly(self):
        # IoU goes from 1.0 at severity 0 to 0.0 at severity 1 — linearly
        # half-life should be at severity 0.5
        hl = DegradationRobustnessAnalyzer._find_half_life(
            [0.0, 1.0], [1.0, 0.0], 1.0
        )
        assert hl == pytest.approx(0.5)

    def test_half_life_none_when_never_reaches_threshold(self):
        hl = DegradationRobustnessAnalyzer._find_half_life(
            [0.0, 0.5, 1.0], [1.0, 0.8, 0.6], 1.0
        )
        # 0.6 > 0.5, so never crosses half; half_life must be None
        assert hl is None
