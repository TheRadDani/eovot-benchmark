"""Tests for sequence attribute analysis and challenge-conditioned metrics."""

import numpy as np
import pytest

from eovot.datasets.attributes import SequenceAttributeAnalyzer, SequenceAttributes
from eovot.metrics.attribute_metrics import AttributeMetricsEngine, AttributePerformance


# ---------------------------------------------------------------------------
# SequenceAttributes
# ---------------------------------------------------------------------------


class TestSequenceAttributes:
    def test_defaults_all_false(self):
        attr = SequenceAttributes()
        assert attr.active_attributes() == []
        assert not any(attr.attribute_vector().values())

    def test_active_attributes_lists_true_flags(self):
        attr = SequenceAttributes(fast_motion=True, scale_variation=True)
        assert set(attr.active_attributes()) == {"fast_motion", "scale_variation"}

    def test_attribute_vector_keys(self):
        attr = SequenceAttributes()
        keys = set(attr.attribute_vector().keys())
        assert keys == {
            "fast_motion",
            "scale_variation",
            "low_resolution",
            "aspect_ratio_change",
            "out_of_view",
            "partial_occlusion",
        }

    def test_to_dict_json_compatible(self):
        attr = SequenceAttributes(fast_motion=True, mean_displacement_px=25.3)
        d = attr.to_dict()
        import json
        json.dumps(d)  # must not raise
        assert d["fast_motion"] is True
        assert d["mean_displacement_px"] == pytest.approx(25.3, rel=1e-3)

    def test_repr_shows_active(self):
        attr = SequenceAttributes(out_of_view=True)
        assert "out_of_view" in repr(attr)


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — edge cases
# ---------------------------------------------------------------------------


class TestSequenceAttributeAnalyzerEdgeCases:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer()

    def test_empty_sequence_returns_defaults(self):
        boxes = np.empty((0, 4), dtype=np.float64)
        attr = self.analyzer.analyze(boxes)
        assert not attr.fast_motion
        assert attr.min_bbox_area_px2 == 0.0

    def test_single_frame_no_motion_attrs(self):
        boxes = np.array([[10.0, 10.0, 40.0, 40.0]])
        attr = self.analyzer.analyze(boxes)
        assert not attr.fast_motion
        assert not attr.scale_variation

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            self.analyzer.analyze(np.ones((10, 3)))

    def test_invalid_1d_raises(self):
        with pytest.raises(ValueError):
            self.analyzer.analyze(np.ones(10))


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — fast motion
# ---------------------------------------------------------------------------


class TestFastMotion:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer(fast_motion_px=20.0)

    def _boxes_with_velocity(self, vx: float, vy: float, n: int = 50) -> np.ndarray:
        """Generate N boxes moving at constant velocity (vx, vy)."""
        x0, y0, w, h = 100.0, 100.0, 40.0, 40.0
        rows = []
        for i in range(n):
            rows.append([x0 + i * vx, y0 + i * vy, w, h])
        return np.array(rows)

    def test_slow_motion_not_flagged(self):
        boxes = self._boxes_with_velocity(1.0, 1.0)
        attr = self.analyzer.analyze(boxes)
        assert not attr.fast_motion

    def test_fast_motion_flagged(self):
        # displacement per frame ≈ sqrt(20² + 20²) ≈ 28 px > threshold 20
        boxes = self._boxes_with_velocity(20.0, 20.0)
        attr = self.analyzer.analyze(boxes)
        assert attr.fast_motion

    def test_mean_displacement_computed(self):
        boxes = self._boxes_with_velocity(3.0, 4.0)  # 5 px/frame
        attr = self.analyzer.analyze(boxes)
        assert attr.mean_displacement_px == pytest.approx(5.0, rel=1e-6)

    def test_max_displacement_computed(self):
        boxes = self._boxes_with_velocity(3.0, 4.0)
        attr = self.analyzer.analyze(boxes)
        assert attr.max_displacement_px == pytest.approx(5.0, rel=1e-6)


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — scale variation
# ---------------------------------------------------------------------------


class TestScaleVariation:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer(scale_var_ratio=0.25)

    def test_constant_scale_not_flagged(self):
        boxes = np.tile([10.0, 10.0, 40.0, 40.0], (30, 1)).astype(float)
        attr = self.analyzer.analyze(boxes)
        assert not attr.scale_variation

    def test_large_scale_jump_flagged(self):
        # box doubles in size — relative change = 1.0 >> 0.25
        boxes = np.array([
            [10.0, 10.0, 40.0, 40.0],
            [10.0, 10.0, 80.0, 80.0],
        ])
        attr = self.analyzer.analyze(boxes)
        assert attr.scale_variation

    def test_small_scale_change_not_flagged(self):
        boxes = np.array([
            [10.0, 10.0, 40.0, 40.0],
            [10.0, 10.0, 42.0, 42.0],  # ~5% change
        ])
        attr = self.analyzer.analyze(boxes)
        assert not attr.scale_variation

    def test_max_scale_ratio_computed(self):
        boxes = np.array([
            [10.0, 10.0, 10.0, 10.0],  # area 100
            [10.0, 10.0, 20.0, 20.0],  # area 400
        ])
        attr = self.analyzer.analyze(boxes)
        assert attr.max_scale_ratio == pytest.approx(4.0, rel=1e-6)

    def test_min_bbox_area_computed(self):
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],   # area 100
            [0.0, 0.0, 30.0, 30.0],   # area 900
        ])
        attr = self.analyzer.analyze(boxes)
        assert attr.min_bbox_area_px2 == pytest.approx(100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — low resolution
# ---------------------------------------------------------------------------


class TestLowResolution:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer(low_res_area_px2=1000.0)

    def test_large_box_not_flagged(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])  # area 2500
        attr = self.analyzer.analyze(boxes)
        assert not attr.low_resolution

    def test_small_box_flagged(self):
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]])  # area 400 < 1000
        attr = self.analyzer.analyze(boxes)
        assert attr.low_resolution


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — aspect ratio change
# ---------------------------------------------------------------------------


class TestAspectRatioChange:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer(aspect_ratio_tol=0.25)

    def test_constant_ar_not_flagged(self):
        boxes = np.tile([0.0, 0.0, 40.0, 40.0], (20, 1)).astype(float)
        attr = self.analyzer.analyze(boxes)
        assert not attr.aspect_ratio_change

    def test_large_ar_change_flagged(self):
        # AR changes from 1.0 to 2.0 → relative change = 1.0 >> 0.25
        boxes = np.array([
            [0.0, 0.0, 40.0, 40.0],   # AR = 1.0
            [0.0, 0.0, 80.0, 40.0],   # AR = 2.0
        ])
        attr = self.analyzer.analyze(boxes)
        assert attr.aspect_ratio_change


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — out of view
# ---------------------------------------------------------------------------


class TestOutOfView:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer(oov_margin=0.05)

    def test_central_box_not_oov(self):
        # Centre at (160, 120) in 320×240 frame — well inside
        boxes = np.array([[140.0, 100.0, 40.0, 40.0]])
        attr = self.analyzer.analyze(boxes, frame_size=(320, 240))
        assert not attr.out_of_view

    def test_box_near_left_edge_is_oov(self):
        # Centre at x=4 (< 5% of 320 = 16)
        boxes = np.array([[0.0, 100.0, 8.0, 40.0]])
        attr = self.analyzer.analyze(boxes, frame_size=(320, 240))
        assert attr.out_of_view
        assert attr.out_of_view_frame_count == 1

    def test_no_frame_size_skips_oov(self):
        boxes = np.array([[0.0, 0.0, 5.0, 5.0]])
        attr = self.analyzer.analyze(boxes, frame_size=None)
        assert not attr.out_of_view

    def test_out_of_view_frame_count(self):
        # 3 boxes near the edge, 1 in the centre
        boxes = np.array([
            [0.0, 0.0, 8.0, 8.0],    # OOV
            [0.0, 0.0, 8.0, 8.0],    # OOV
            [0.0, 0.0, 8.0, 8.0],    # OOV
            [140.0, 100.0, 40.0, 40.0],  # fine
        ])
        attr = self.analyzer.analyze(boxes, frame_size=(320, 240))
        assert attr.out_of_view_frame_count == 3


# ---------------------------------------------------------------------------
# SequenceAttributeAnalyzer — partial occlusion
# ---------------------------------------------------------------------------


class TestPartialOcclusion:
    def setup_method(self):
        self.analyzer = SequenceAttributeAnalyzer(
            occlusion_low_iou=0.3,
            occlusion_recovery_iou=0.5,
        )
        self.gt = np.tile([10.0, 10.0, 40.0, 40.0], (20, 1)).astype(float)

    def test_no_predictions_no_occlusion(self):
        attr = self.analyzer.analyze(self.gt)
        assert not attr.partial_occlusion

    def test_high_iou_throughout_no_occlusion(self):
        # Predictions perfectly match GT → high IoU always
        attr = self.analyzer.analyze(self.gt, predicted_boxes=self.gt.copy())
        assert not attr.partial_occlusion

    def test_drop_and_recovery_detected(self):
        # Misaligned predictions at frames 5-10 then back to perfect
        preds = self.gt.copy()
        preds[5:10] = [200.0, 200.0, 10.0, 10.0]  # far from GT → low IoU
        attr = self.analyzer.analyze(self.gt, predicted_boxes=preds)
        assert attr.partial_occlusion

    def test_drop_without_recovery_not_flagged(self):
        # Predictions drift off permanently at the end — no recovery
        preds = self.gt.copy()
        preds[15:] = [200.0, 200.0, 10.0, 10.0]
        attr = self.analyzer.analyze(self.gt, predicted_boxes=preds)
        assert not attr.partial_occlusion


# ---------------------------------------------------------------------------
# AttributeMetricsEngine
# ---------------------------------------------------------------------------


class TestAttributeMetricsEngine:
    def _make_inputs(self):
        """Two sequences: seq0 has fast_motion, seq1 is normal."""
        from eovot.datasets.attributes import SequenceAttributes

        attrs = {
            "seq0": SequenceAttributes(fast_motion=True, scale_variation=True),
            "seq1": SequenceAttributes(fast_motion=False, low_resolution=True),
        }
        ious = {
            "seq0": np.array([0.8, 0.75, 0.7]),
            "seq1": np.array([0.6, 0.65, 0.55]),
        }
        fps = {"seq0": 120.0, "seq1": 80.0}
        return attrs, ious, fps

    def test_compute_returns_dict(self):
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        assert isinstance(result, dict)
        # fast_motion present in seq0
        assert "fast_motion" in result

    def test_fast_motion_only_from_seq0(self):
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        fm = result["fast_motion"]
        assert fm.sequence_count == 1
        assert fm.mean_iou == pytest.approx(np.mean([0.8, 0.75, 0.7]), rel=1e-4)
        assert fm.mean_fps == pytest.approx(120.0, rel=1e-4)

    def test_low_resolution_only_from_seq1(self):
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        lr = result["low_resolution"]
        assert lr.sequence_count == 1
        assert lr.mean_iou == pytest.approx(np.mean([0.6, 0.65, 0.55]), rel=1e-4)

    def test_missing_attribute_absent_from_result(self):
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        assert "out_of_view" not in result

    def test_empty_sequence_list(self):
        engine = AttributeMetricsEngine()
        result = engine.compute([], {}, {}, {})
        assert result == {}

    def test_success_auc_in_range(self):
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        for perf in result.values():
            assert 0.0 <= perf.success_auc <= 1.0

    def test_to_markdown_table_format(self):
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        table = engine.to_markdown_table(result, tracker_name="MOSSE")
        assert "MOSSE" in table
        assert "Fast Motion" in table
        assert "# Seqs" in table
        # Markdown table rows should start with '|'
        rows = [l for l in table.splitlines() if l.startswith("|")]
        assert len(rows) >= 3  # header + separator + at least 1 data row

    def test_to_dict_json_serialisable(self):
        import json
        attrs, ious, fps = self._make_inputs()
        engine = AttributeMetricsEngine()
        result = engine.compute(["seq0", "seq1"], attrs, ious, fps)
        for perf in result.values():
            json.dumps(perf.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Integration: BenchmarkEngine populates attributes
# ---------------------------------------------------------------------------


class TestBenchmarkEngineAttributes:
    def test_sequence_results_have_attributes(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False, compute_attributes=True)
        result = engine.run(MOSSETracker(), ds, "Synthetic")

        for sr in result.sequence_results:
            assert sr.attributes is not None
            assert isinstance(sr.attributes, SequenceAttributes)

    def test_attributes_disabled_gives_none(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=2, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False, compute_attributes=False)
        result = engine.run(MOSSETracker(), ds, "Synthetic")

        for sr in result.sequence_results:
            assert sr.attributes is None

    def test_attribute_breakdown_via_reporter(self):
        import tempfile
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.reporting.reporter import BenchmarkReporter
        from eovot.trackers.mosse import MOSSETracker

        ds = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear")
        engine = BenchmarkEngine(verbose=False, compute_attributes=True)
        result = engine.run(MOSSETracker(), ds, "Synthetic")

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            table = reporter.attribute_breakdown_table(result, output_name="test")
            assert isinstance(table, str)
            assert len(table) > 0
