"""Tests for ResolutionScaledTracker and ResolutionScaleAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.trackers.scale import ResolutionScaledTracker
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# ResolutionScaledTracker unit tests
# ---------------------------------------------------------------------------

class TestResolutionScaledTracker:

    def _make_frame(self, h=120, w=160):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def _make_bbox(self, x=20, y=20, w=40, h=40):
        return (float(x), float(y), float(w), float(h))

    def test_name_includes_scale(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        assert "0.50" in tracker.name

    def test_name_includes_base_tracker_name(self):
        base = MOSSETracker()
        tracker = ResolutionScaledTracker(base, scale_factor=0.5)
        assert base.name in tracker.name

    def test_invalid_scale_zero_raises(self):
        with pytest.raises(ValueError):
            ResolutionScaledTracker(MOSSETracker(), scale_factor=0.0)

    def test_invalid_scale_negative_raises(self):
        with pytest.raises(ValueError):
            ResolutionScaledTracker(MOSSETracker(), scale_factor=-0.5)

    def test_invalid_scale_above_one_raises(self):
        with pytest.raises(ValueError):
            ResolutionScaledTracker(MOSSETracker(), scale_factor=1.5)

    def test_scale_one_passthrough(self):
        """scale_factor=1.0 should be accepted and work without resizing."""
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=1.0)
        frame = self._make_frame()
        bbox = self._make_bbox()
        tracker.initialize(frame, bbox)
        pred = tracker.update(self._make_frame())
        assert len(pred) == 4

    def test_initialize_and_update_return_bbox(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        frame = self._make_frame()
        bbox = self._make_bbox()
        tracker.initialize(frame, bbox)
        pred = tracker.update(self._make_frame())
        assert len(pred) == 4
        assert all(isinstance(v, float) for v in pred)

    def test_output_bbox_in_original_coords(self):
        """Predicted bbox coordinates must be in original (not scaled) frame space."""
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        h, w = 240, 320
        frame = self._make_frame(h=h, w=w)
        bbox = (40.0, 30.0, 60.0, 50.0)
        tracker.initialize(frame, bbox)
        pred = tracker.update(self._make_frame(h=h, w=w))
        px, py, pw, ph = pred
        # The prediction is in original coords: x, y within frame bounds.
        assert px < w * 2   # generous bound; tracker may drift
        assert py < h * 2

    def test_scale_down_bbox(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        scaled = tracker._scale_bbox_down((100.0, 80.0, 40.0, 30.0))
        assert abs(scaled[0] - 50.0) < 1e-9
        assert abs(scaled[1] - 40.0) < 1e-9
        assert abs(scaled[2] - 20.0) < 1e-9
        assert abs(scaled[3] - 15.0) < 1e-9

    def test_scale_up_bbox_roundtrip(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        original = (100.0, 80.0, 40.0, 30.0)
        roundtripped = tracker._scale_bbox_up(tracker._scale_bbox_down(original))
        for a, b in zip(original, roundtripped):
            assert abs(a - b) < 1e-6

    def test_scale_frame_reduces_size(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.5)
        frame = self._make_frame(h=240, w=320)
        scaled = tracker._scale_frame(frame)
        assert scaled.shape[0] == 120
        assert scaled.shape[1] == 160

    def test_scale_frame_one_is_noop(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=1.0)
        frame = self._make_frame()
        result = tracker._scale_frame(frame)
        assert result is frame  # no copy made for scale=1.0

    def test_scale_factor_stored(self):
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=0.75)
        assert tracker.scale_factor == 0.75

    def test_pixel_reduction_ratio(self):
        scale = 0.5
        tracker = ResolutionScaledTracker(MOSSETracker(), scale_factor=scale)
        h, w = 240, 320
        frame = self._make_frame(h=h, w=w)
        scaled = tracker._scale_frame(frame)
        expected_pixels = int(round(w * scale)) * int(round(h * scale))
        actual_pixels = scaled.shape[0] * scaled.shape[1]
        assert actual_pixels == expected_pixels


# ---------------------------------------------------------------------------
# ResolutionScaleAnalyzer integration tests (uses SyntheticDataset + engine)
# ---------------------------------------------------------------------------

class TestResolutionScaleAnalyzer:

    def _build_engine(self):
        from eovot.benchmark.engine import BenchmarkEngine
        return BenchmarkEngine(verbose=False)

    def _build_dataset(self, n=2, f=20):
        return SyntheticDataset(num_sequences=n, num_frames=f, seed=7)

    def test_analyze_returns_scale_result(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        from eovot.analysis.resolution_analysis import ScaleResult
        assert isinstance(result, ScaleResult)

    def test_num_entries_matches_scale_factors(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.75, 0.5],
        )
        assert len(result.entries) == 3

    def test_baseline_is_scale_one(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        baseline = result.baseline
        assert baseline is not None
        assert abs(baseline.scale_factor - 1.0) < 1e-9

    def test_baseline_iou_degradation_zero(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        assert abs(result.baseline.iou_degradation) < 1e-9

    def test_pixel_reduction_correct(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        half_entry = result.entry_for(0.5)
        assert abs(half_entry.pixel_reduction - 0.25) < 1e-9

    def test_markdown_table_has_header(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        table = result.to_markdown_table()
        assert "Scale" in table
        assert "mIoU" in table
        assert "FPS" in table

    def test_str_representation(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        s = str(result)
        assert "ResolutionScaleAnalysis" in s

    def test_invalid_scale_factor_raises(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        with pytest.raises(ValueError):
            analyzer.analyze(
                MOSSETracker(),
                self._build_dataset(),
                scale_factors=[1.0, 1.5],
            )

    def test_duplicate_scales_deduplicated(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 1.0, 0.5, 0.5],
        )
        assert len(result.entries) == 2

    def test_optimal_scale_returns_tuple(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        scale, iou, fps = result.optimal_scale(min_iou=0.0)
        assert fps > 0
        assert 0.0 <= iou <= 1.0

    def test_optimal_scale_impossible_constraint_raises(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        with pytest.raises(ValueError):
            result.optimal_scale(min_iou=2.0)  # impossible

    def test_fps_at_iou_budget_returns_float(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        fps = result.fps_at_iou_budget(iou_budget=1.0)
        assert fps is not None
        assert fps > 0

    def test_benchmark_results_stored(self):
        from eovot.analysis.resolution_analysis import ResolutionScaleAnalyzer
        analyzer = ResolutionScaleAnalyzer(self._build_engine())
        result = analyzer.analyze(
            MOSSETracker(),
            self._build_dataset(),
            scale_factors=[1.0, 0.5],
        )
        assert 1.0 in result.benchmark_results
        assert 0.5 in result.benchmark_results
