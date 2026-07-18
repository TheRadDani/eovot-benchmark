"""Tests for eovot.benchmark.resolution_sweep."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.resolution_sweep import (
    ResolutionSweepEvaluator,
    ResolutionSweepResult,
    ResolutionWrapper,
    ScalePoint,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# ResolutionWrapper unit tests
# ---------------------------------------------------------------------------

class TestResolutionWrapper:
    def _make_frame(self, h: int = 120, w: int = 160) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_name_includes_scale(self):
        tracker = MOSSETracker()
        wrapper = ResolutionWrapper(tracker, scale=0.5)
        assert "0.50" in wrapper.name
        assert "MOSSE" in wrapper.name

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            ResolutionWrapper(MOSSETracker(), scale=0.0)
        with pytest.raises(ValueError):
            ResolutionWrapper(MOSSETracker(), scale=3.0)
        with pytest.raises(ValueError):
            ResolutionWrapper(MOSSETracker(), scale=-0.5)

    def test_bbox_roundtrip_half_scale(self):
        tracker = MOSSETracker()
        wrapper = ResolutionWrapper(tracker, scale=0.5)
        bbox = (40.0, 30.0, 20.0, 15.0)
        scaled = wrapper._bbox_to_scaled(bbox)
        recovered = wrapper._bbox_to_original(scaled)
        assert recovered == pytest.approx(bbox, rel=1e-6)

    def test_bbox_roundtrip_full_scale(self):
        tracker = MOSSETracker()
        wrapper = ResolutionWrapper(tracker, scale=1.0)
        bbox = (10.0, 20.0, 30.0, 40.0)
        assert wrapper._bbox_to_scaled(bbox) == pytest.approx(bbox)
        assert wrapper._bbox_to_original(bbox) == pytest.approx(bbox)

    def test_frame_rescaling_reduces_size(self):
        tracker = MOSSETracker()
        wrapper = ResolutionWrapper(tracker, scale=0.5)
        frame = self._make_frame(h=120, w=160)
        small = wrapper._scale_frame(frame)
        assert small.shape == (60, 80, 3)

    def test_full_scale_returns_same_frame(self):
        tracker = MOSSETracker()
        wrapper = ResolutionWrapper(tracker, scale=1.0)
        frame = self._make_frame()
        out = wrapper._scale_frame(frame)
        assert out is frame  # identity — no copy

    def test_initialize_and_update(self):
        dataset = SyntheticDataset(num_sequences=1, num_frames=5, seed=0)
        seq = dataset[0]
        frames = list(seq)
        wrapper = ResolutionWrapper(MOSSETracker(), scale=0.5)
        wrapper.initialize(frames[0], seq.init_bbox)
        bbox = wrapper.update(frames[1])
        x, y, w, h = bbox
        assert w > 0 and h > 0, "update() must return a positive-size box"

    def test_predicted_bbox_in_original_coords(self):
        """Predictions must be in original-frame coordinate space."""
        dataset = SyntheticDataset(num_sequences=1, num_frames=5, seed=1)
        seq = dataset[0]
        frames = list(seq)
        wrapper = ResolutionWrapper(MOSSETracker(), scale=0.5)
        wrapper.initialize(frames[0], seq.init_bbox)
        pred = wrapper.update(frames[1])
        # Box should be in the ballpark of the original-frame target position
        gt = seq.ground_truth[1]
        gt_cx = gt[0] + gt[2] / 2
        pred_cx = pred[0] + pred[2] / 2
        # Loose check: centroid must not be compressed by the scale factor
        assert pred_cx > 5.0, f"Centroid suspiciously small: {pred_cx}"


# ---------------------------------------------------------------------------
# ScalePoint tests
# ---------------------------------------------------------------------------

class TestScalePoint:
    def _dummy_result(self):
        from eovot.benchmark.engine import BenchmarkResult
        return BenchmarkResult(tracker_name="T", dataset_name="D")

    def test_pareto_flag_default_false(self):
        sp = ScalePoint(
            scale=0.5,
            mean_iou=0.4,
            mean_fps=100.0,
            peak_memory_mb=50.0,
            success_auc=None,
            precision_auc=None,
            benchmark_result=self._dummy_result(),
        )
        assert not sp.on_pareto_front

    def test_pareto_flag_settable(self):
        sp = ScalePoint(
            scale=1.0,
            mean_iou=0.6,
            mean_fps=50.0,
            peak_memory_mb=60.0,
            success_auc=0.5,
            precision_auc=0.4,
            benchmark_result=self._dummy_result(),
        )
        sp.on_pareto_front = True
        assert sp.on_pareto_front


# ---------------------------------------------------------------------------
# ResolutionSweepEvaluator integration tests
# ---------------------------------------------------------------------------

class TestResolutionSweepEvaluator:
    def _small_dataset(self) -> SyntheticDataset:
        return SyntheticDataset(num_sequences=2, num_frames=15, seed=42)

    def test_evaluate_returns_correct_number_of_points(self):
        scales = [0.5, 1.0]
        evaluator = ResolutionSweepEvaluator(scales=scales, verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset(), dataset_name="Syn")
        assert len(result.points) == 2

    def test_points_sorted_by_scale(self):
        scales = [1.0, 0.25, 0.75]
        evaluator = ResolutionSweepEvaluator(scales=scales, verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        assert result.points[0].scale < result.points[1].scale < result.points[2].scale

    def test_pareto_flags_set(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        pareto_count = sum(1 for p in result.points if p.on_pareto_front)
        assert pareto_count >= 1, "At least one point must be Pareto-optimal"

    def test_metrics_are_positive(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        for p in result.points:
            assert p.mean_fps > 0
            assert 0.0 <= p.mean_iou <= 1.0
            assert p.peak_memory_mb > 0

    def test_best_for_fps_target_returns_none_when_impossible(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        best = evaluator.best_for_fps_target(result, fps_target=1e9)
        assert best is None

    def test_best_for_fps_target_returns_point(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        best_fps = max(p.mean_fps for p in result.points)
        best = evaluator.best_for_fps_target(result, fps_target=1.0)
        assert best is not None
        assert best.mean_fps >= 1.0

    def test_iou_retention_keys_match_scales(self):
        scales = [0.5, 1.0]
        evaluator = ResolutionSweepEvaluator(scales=scales, verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        retention = evaluator.iou_retention(result, reference_scale=1.0)
        assert set(retention.keys()) == {0.5, 1.0}
        assert retention[1.0] == pytest.approx(1.0, abs=1e-4)

    def test_iou_retention_returns_empty_for_missing_reference(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        retention = evaluator.iou_retention(result, reference_scale=0.9)
        assert retention == {}

    def test_markdown_table_contains_tracker_name(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset(), dataset_name="Test")
        table = evaluator.to_markdown_table(result)
        assert "MOSSE" in table
        assert "Test" in table
        assert "0.50" in table
        assert "1.00" in table

    def test_summary_dict_structure(self):
        evaluator = ResolutionSweepEvaluator(scales=[0.5, 1.0], verbose=False)
        result = evaluator.evaluate(MOSSETracker, self._small_dataset())
        d = evaluator.to_summary_dict(result)
        assert "tracker_name" in d
        assert "dataset_name" in d
        assert isinstance(d["scales"], list)
        assert len(d["scales"]) == 2
        for entry in d["scales"]:
            assert "scale" in entry
            assert "mean_iou" in entry
            assert "mean_fps" in entry
            assert "on_pareto_front" in entry

    def test_max_sequences_limits_evaluation(self):
        dataset = SyntheticDataset(num_sequences=5, num_frames=10, seed=0)
        evaluator = ResolutionSweepEvaluator(
            scales=[1.0], max_sequences=2, verbose=False
        )
        result = evaluator.evaluate(MOSSETracker, dataset)
        assert len(result.points[0].benchmark_result.sequence_results) == 2


# ---------------------------------------------------------------------------
# ResolutionSweepResult helpers
# ---------------------------------------------------------------------------

class TestResolutionSweepResult:
    def _make_result(self) -> ResolutionSweepResult:
        from eovot.benchmark.engine import BenchmarkResult

        def sp(scale, iou, fps):
            return ScalePoint(
                scale=scale,
                mean_iou=iou,
                mean_fps=fps,
                peak_memory_mb=100.0,
                success_auc=None,
                precision_auc=None,
                benchmark_result=BenchmarkResult("T", "D"),
            )

        return ResolutionSweepResult(
            tracker_name="T",
            dataset_name="D",
            points=[sp(0.25, 0.30, 300.0), sp(0.5, 0.45, 150.0), sp(1.0, 0.55, 50.0)],
        )

    def test_best_iou_returns_highest(self):
        r = self._make_result()
        assert r.best_iou().scale == 1.0

    def test_best_fps_returns_highest(self):
        r = self._make_result()
        assert r.best_fps().scale == 0.25

    def test_at_scale_finds_point(self):
        r = self._make_result()
        p = r.at_scale(0.5)
        assert p is not None
        assert p.mean_fps == 150.0

    def test_at_scale_returns_none_for_missing(self):
        r = self._make_result()
        assert r.at_scale(0.9) is None

    def test_pareto_front_property(self):
        r = self._make_result()
        ResolutionSweepEvaluator._mark_pareto_front(r.points)
        pf = r.pareto_front
        assert len(pf) >= 1
