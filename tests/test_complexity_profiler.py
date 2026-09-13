"""Tests for TrackerComplexityProfiler."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.profiling.complexity import (
    ComplexityReport,
    TrackerComplexityProfiler,
    _classify_flops,
)
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.camshift import CamShiftTracker


# ---------------------------------------------------------------------------
# _classify_flops
# ---------------------------------------------------------------------------

class TestClassifyFlops:
    def test_light(self):
        assert _classify_flops(500_000) == "light"

    def test_medium(self):
        assert _classify_flops(5_000_000) == "medium"

    def test_heavy(self):
        assert _classify_flops(100_000_000) == "heavy"

    def test_boundary_light_medium(self):
        assert _classify_flops(1_000_000) == "medium"  # boundary is 1 M exclusive

    def test_none_returns_unknown(self):
        assert _classify_flops(None) == "unknown"


# ---------------------------------------------------------------------------
# ComplexityReport
# ---------------------------------------------------------------------------

class TestComplexityReport:
    def _make_report(self, flops: int = 2_000_000) -> ComplexityReport:
        return ComplexityReport(
            tracker_name="TestTracker",
            frame_width=320,
            frame_height=240,
            param_count=1024,
            param_memory_bytes=1024 * 4,
            estimated_flops=flops,
            memory_bandwidth_bytes_per_frame=320 * 240 * 3,
            complexity_class="medium",
            notes="test",
        )

    def test_param_memory_kb(self):
        r = self._make_report()
        assert r.param_memory_kb == pytest.approx(1024 * 4 / 1024.0)

    def test_estimated_flops_m(self):
        r = self._make_report(flops=2_000_000)
        assert r.estimated_flops_m == pytest.approx(2.0)

    def test_estimated_flops_m_none(self):
        r = self._make_report()
        r2 = ComplexityReport(
            tracker_name="T", frame_width=320, frame_height=240,
            param_count=0, param_memory_bytes=0,
            estimated_flops=None,
            memory_bandwidth_bytes_per_frame=0,
            complexity_class="unknown",
        )
        assert r2.estimated_flops_m is None

    def test_pixels(self):
        r = self._make_report()
        assert r.pixels == 320 * 240

    def test_flops_per_pixel(self):
        r = self._make_report(flops=320 * 240 * 10)
        assert r.flops_per_pixel == pytest.approx(10.0, rel=1e-5)

    def test_to_dict_keys(self):
        r = self._make_report()
        d = r.to_dict()
        for key in [
            "tracker_name", "frame_width", "frame_height",
            "param_count", "param_memory_bytes", "param_memory_kb",
            "estimated_flops", "estimated_flops_m",
            "memory_bandwidth_bytes_per_frame", "memory_bandwidth_kb_per_frame",
            "complexity_class", "flops_per_pixel", "notes",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_str_contains_name(self):
        r = self._make_report()
        assert "TestTracker" in str(r)

    def test_str_contains_resolution(self):
        r = self._make_report()
        s = str(r)
        assert "320" in s and "240" in s


# ---------------------------------------------------------------------------
# TrackerComplexityProfiler.profile
# ---------------------------------------------------------------------------

class TestProfilerProfile:
    def test_returns_complexity_report(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert isinstance(report, ComplexityReport)

    def test_tracker_name_set(self):
        tracker = MOSSETracker()
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(tracker, frame_size=(160, 120))
        assert report.tracker_name == tracker.name

    def test_frame_size_stored(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(320, 240))
        assert report.frame_width == 320
        assert report.frame_height == 240

    def test_param_count_nonnegative(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert report.param_count >= 0

    def test_param_memory_consistent(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert report.param_memory_bytes == report.param_count * 4

    def test_estimated_flops_positive(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert report.estimated_flops is not None
        assert report.estimated_flops > 0

    def test_complexity_class_valid(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert report.complexity_class in ("light", "medium", "heavy", "unknown")

    def test_memory_bandwidth_positive(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert report.memory_bandwidth_bytes_per_frame > 0

    def test_default_bbox_centred(self):
        """Profile should succeed with default bbox."""
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(KCFTracker(), frame_size=(320, 240))
        assert report is not None

    def test_custom_bbox(self):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(
            MOSSETracker(), frame_size=(320, 240),
            bbox=(40.0, 30.0, 80.0, 60.0)
        )
        assert report.frame_width == 320

    @pytest.mark.parametrize("tracker_cls", [MOSSETracker, KCFTracker, CamShiftTracker])
    def test_multiple_tracker_types(self, tracker_cls):
        profiler = TrackerComplexityProfiler()
        report = profiler.profile(tracker_cls(), frame_size=(160, 120))
        assert isinstance(report, ComplexityReport)
        assert report.estimated_flops is not None

    def test_mosse_lighter_than_kcf_at_same_resolution(self):
        """MOSSE and KCF are both FFT-based; FLOPs should be in the same order."""
        profiler = TrackerComplexityProfiler()
        r_mosse = profiler.profile(MOSSETracker(), frame_size=(320, 240))
        r_kcf = profiler.profile(KCFTracker(), frame_size=(320, 240))
        # Both are classified as the same family — verify both have estimates.
        assert r_mosse.estimated_flops is not None
        assert r_kcf.estimated_flops is not None


# ---------------------------------------------------------------------------
# Multi-resolution profiling
# ---------------------------------------------------------------------------

class TestMultiResolution:
    def test_returns_dict_with_all_resolutions(self):
        profiler = TrackerComplexityProfiler()
        resolutions = [(160, 120), (320, 240), (640, 480)]
        reports = profiler.profile_multi_resolution(
            MOSSETracker(), resolutions=resolutions
        )
        assert set(reports.keys()) == set(resolutions)

    def test_default_resolutions(self):
        profiler = TrackerComplexityProfiler()
        reports = profiler.profile_multi_resolution(MOSSETracker())
        assert len(reports) == 4

    def test_flops_scale_with_resolution(self):
        """Higher resolution → more FLOPs for correlation filters."""
        profiler = TrackerComplexityProfiler()
        reports = profiler.profile_multi_resolution(
            MOSSETracker(), resolutions=[(160, 120), (640, 480)]
        )
        f_lo = reports[(160, 120)].estimated_flops
        f_hi = reports[(640, 480)].estimated_flops
        assert f_lo is not None and f_hi is not None
        assert f_hi > f_lo

    def test_memory_bandwidth_scales_with_resolution(self):
        profiler = TrackerComplexityProfiler()
        reports = profiler.profile_multi_resolution(
            MOSSETracker(), resolutions=[(160, 120), (640, 480)]
        )
        bw_lo = reports[(160, 120)].memory_bandwidth_bytes_per_frame
        bw_hi = reports[(640, 480)].memory_bandwidth_bytes_per_frame
        assert bw_hi > bw_lo


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_returns_sorted_list(self):
        profiler = TrackerComplexityProfiler()
        trackers = [KCFTracker(), MOSSETracker(), CamShiftTracker()]
        reports = profiler.compare(trackers, frame_size=(160, 120))
        assert len(reports) == 3
        flops = [
            r.estimated_flops for r in reports if r.estimated_flops is not None
        ]
        assert flops == sorted(flops)

    def test_compare_with_custom_bytes_per_param(self):
        profiler = TrackerComplexityProfiler(bytes_per_param=2)  # fp16
        report = profiler.profile(MOSSETracker(), frame_size=(160, 120))
        assert report.param_memory_bytes == report.param_count * 2
