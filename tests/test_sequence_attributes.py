"""Tests for eovot.analysis — sequence attribute computation and reporting."""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pytest

from eovot.analysis.sequence_attributes import (
    AttributeFlags,
    SequenceAttributes,
    compute_sequence_attributes,
    tag_sequences,
)
from eovot.analysis.attribute_report import (
    AttributeReport,
    generate_attribute_report,
    TrackerAttributeSlice,
)
from eovot.datasets.base import BaseDataset, Sequence
from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.profiling.profiler import Profiler, ProfilingResult
from eovot.trackers.base import BaseTracker

BBox = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_FRAMES = 30
FIXED_BOX: BBox = (10.0, 10.0, 50.0, 50.0)


def make_gt(boxes: List[BBox]) -> np.ndarray:
    return np.array(boxes, dtype=np.float64)


def static_gt(n: int, box: BBox = (10.0, 10.0, 50.0, 50.0)) -> np.ndarray:
    return np.tile(np.array(box), (n, 1)).astype(np.float64)


class SyntheticSequence(Sequence):
    def __init__(self, name: str, gt: np.ndarray) -> None:
        super().__init__(
            name=name,
            frame_paths=[f"frame_{i:04d}.jpg" for i in range(len(gt))],
            ground_truth=gt,
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(len(self.ground_truth)):
            yield frame


class SyntheticDataset(BaseDataset):
    def __init__(self, seqs: List[Sequence]) -> None:
        self._seqs = seqs

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


class ConstantTracker(BaseTracker):
    @property
    def name(self) -> str:
        return "ConstantTracker"

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._box = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._box


def _make_profiling_result(fps: float = 100.0) -> ProfilingResult:
    return ProfilingResult(
        tracker_name="Test",
        frame_count=10,
        fps=fps,
        latency_mean_ms=1_000 / fps,
        latency_std_ms=0.0,
        latency_p95_ms=1_000 / fps,
        peak_memory_mb=50.0,
    )


def _make_seq_result(name: str, ious: np.ndarray, fps: float = 100.0) -> SequenceResult:
    return SequenceResult(
        sequence_name=name,
        ious=ious,
        profiling=_make_profiling_result(fps),
    )


def _make_bench_result(tracker_name: str, seq_results: List[SequenceResult]) -> BenchmarkResult:
    r = BenchmarkResult(tracker_name=tracker_name, dataset_name="Synthetic")
    r.sequence_results = seq_results
    return r


# ---------------------------------------------------------------------------
# AttributeFlags tests
# ---------------------------------------------------------------------------

class TestAttributeFlags:
    def test_all_false_by_default(self):
        f = AttributeFlags()
        assert not f.scale_variation
        assert not f.aspect_ratio_change
        assert not f.fast_motion
        assert not f.low_resolution
        assert not f.partial_occlusion
        assert not f.deformation

    def test_active_flags_empty_when_none(self):
        f = AttributeFlags()
        assert f.active_flags() == []

    def test_active_flags_returns_abbreviations(self):
        f = AttributeFlags(scale_variation=True, fast_motion=True)
        active = f.active_flags()
        assert "SV" in active
        assert "FM" in active
        assert "ARC" not in active

    def test_to_dict_has_all_keys(self):
        f = AttributeFlags(low_resolution=True)
        d = f.to_dict()
        assert set(d.keys()) == {"SV", "ARC", "FM", "LR", "PO", "DEF"}
        assert d["LR"] is True
        assert d["SV"] is False

    def test_str_shows_active_flags(self):
        f = AttributeFlags(fast_motion=True)
        assert "FM" in str(f)

    def test_str_none_when_no_flags(self):
        f = AttributeFlags()
        assert "none" in str(f)


# ---------------------------------------------------------------------------
# compute_sequence_attributes tests
# ---------------------------------------------------------------------------

class TestComputeSequenceAttributes:
    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_sequence_attributes(np.zeros((10, 3)), "bad")

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_sequence_attributes(np.zeros((0, 4)), "empty")

    def test_static_sequence_no_flags(self):
        gt = static_gt(30)
        attrs = compute_sequence_attributes(gt, "static")
        assert not attrs.flags.scale_variation
        assert not attrs.flags.fast_motion
        assert not attrs.flags.partial_occlusion

    def test_scale_variation_detected(self):
        # Box grows 4× in area → ratio=4 ≥ threshold (2)
        boxes = [(10.0, 10.0, 20.0, 20.0)] * 10 + [(10.0, 10.0, 40.0, 40.0)] * 10
        gt = make_gt(boxes)
        attrs = compute_sequence_attributes(gt, "sv_seq")
        assert attrs.flags.scale_variation

    def test_no_scale_variation_when_stable(self):
        gt = static_gt(20, (10.0, 10.0, 30.0, 30.0))
        attrs = compute_sequence_attributes(gt, "stable")
        assert not attrs.flags.scale_variation

    def test_fast_motion_detected(self):
        # Object moves 200px per frame on a 40×40 box → 200/40 = 5 >> 0.2
        boxes = [(float(i * 200), 10.0, 40.0, 40.0) for i in range(20)]
        gt = make_gt(boxes)
        attrs = compute_sequence_attributes(gt, "fast")
        assert attrs.flags.fast_motion

    def test_slow_motion_not_flagged(self):
        # Object moves 1px per frame on a 100×100 box → negligible
        boxes = [(float(i), 10.0, 100.0, 100.0) for i in range(20)]
        gt = make_gt(boxes)
        attrs = compute_sequence_attributes(gt, "slow")
        assert not attrs.flags.fast_motion

    def test_low_resolution_detected(self):
        # 10×10 box = 100 px² < 400 threshold
        gt = static_gt(20, (10.0, 10.0, 10.0, 10.0))
        attrs = compute_sequence_attributes(gt, "lr")
        assert attrs.flags.low_resolution

    def test_high_resolution_not_flagged(self):
        # 100×100 box = 10000 px²
        gt = static_gt(20, (10.0, 10.0, 100.0, 100.0))
        attrs = compute_sequence_attributes(gt, "hr")
        assert not attrs.flags.low_resolution

    def test_aspect_ratio_change_detected(self):
        # Switches from wide (4:1) to tall (1:4) aspect ratio
        boxes = [(10.0, 10.0, 80.0, 20.0)] * 10 + [(10.0, 10.0, 20.0, 80.0)] * 10
        gt = make_gt(boxes)
        attrs = compute_sequence_attributes(gt, "arc")
        assert attrs.flags.aspect_ratio_change

    def test_partial_occlusion_detected(self):
        # Large box then very small box — abrupt area drop
        boxes = [(10.0, 10.0, 80.0, 80.0)] * 10 + [(10.0, 10.0, 10.0, 10.0)] * 10
        gt = make_gt(boxes)
        attrs = compute_sequence_attributes(gt, "po")
        assert attrs.flags.partial_occlusion

    def test_scale_ratio_reported(self):
        boxes = [(0.0, 0.0, 10.0, 10.0)] + [(0.0, 0.0, 30.0, 30.0)] * 9
        gt = make_gt(boxes)
        attrs = compute_sequence_attributes(gt, "ratio")
        assert attrs.scale_ratio == pytest.approx(9.0, rel=0.01)

    def test_mean_area_computed(self):
        gt = static_gt(10, (0.0, 0.0, 10.0, 10.0))  # area = 100
        attrs = compute_sequence_attributes(gt, "area")
        assert attrs.mean_area_px == pytest.approx(100.0)

    def test_num_frames_matches(self):
        gt = static_gt(42)
        attrs = compute_sequence_attributes(gt, "count")
        assert attrs.num_frames == 42

    def test_to_dict_keys(self):
        gt = static_gt(10)
        attrs = compute_sequence_attributes(gt, "dict_test")
        d = attrs.to_dict()
        assert "sequence_name" in d
        assert "num_frames" in d
        assert "flags" in d
        assert "active_flags" in d

    def test_custom_threshold_override(self):
        # Only flag SV when ratio >= 10 (very high threshold)
        boxes = [(0.0, 0.0, 10.0, 10.0)] * 5 + [(0.0, 0.0, 30.0, 30.0)] * 5
        gt = make_gt(boxes)
        # ratio = 9 → should NOT trigger SV with threshold=10
        attrs = compute_sequence_attributes(gt, "high_thresh", sv_threshold=10.0)
        assert not attrs.flags.scale_variation
        # But should trigger with default threshold (2.0)
        attrs2 = compute_sequence_attributes(gt, "default_thresh")
        assert attrs2.flags.scale_variation


# ---------------------------------------------------------------------------
# tag_sequences tests
# ---------------------------------------------------------------------------

class TestTagSequences:
    def _make_dataset(self):
        seqs = [
            SyntheticSequence("static", static_gt(20)),
            SyntheticSequence("fast", make_gt(
                [(float(i * 200), 10.0, 40.0, 40.0) for i in range(20)]
            )),
        ]
        return SyntheticDataset(seqs)

    def test_returns_dict_keyed_by_name(self):
        ds = self._make_dataset()
        tagged = tag_sequences(ds)
        assert set(tagged.keys()) == {"static", "fast"}

    def test_each_value_is_sequence_attributes(self):
        ds = self._make_dataset()
        tagged = tag_sequences(ds)
        for v in tagged.values():
            assert isinstance(v, SequenceAttributes)

    def test_fast_sequence_flagged(self):
        ds = self._make_dataset()
        tagged = tag_sequences(ds)
        assert tagged["fast"].flags.fast_motion

    def test_static_sequence_not_flagged_for_motion(self):
        ds = self._make_dataset()
        tagged = tag_sequences(ds)
        assert not tagged["static"].flags.fast_motion


# ---------------------------------------------------------------------------
# AttributeReport tests
# ---------------------------------------------------------------------------

class TestGenerateAttributeReport:
    def _build_inputs(self):
        # Two sequences: 'fast' has FM, 'static' doesn't
        tagged_fast = make_gt([(float(i * 200), 10.0, 40.0, 40.0) for i in range(20)])
        tagged_static = static_gt(20)

        seqs = [
            SyntheticSequence("fast_seq", tagged_fast),
            SyntheticSequence("static_seq", tagged_static),
        ]
        ds = SyntheticDataset(seqs)
        seq_attrs = tag_sequences(ds)

        # Fake benchmark results for two trackers
        r_mosse = _make_bench_result("MOSSE", [
            _make_seq_result("fast_seq", np.full(20, 0.5)),
            _make_seq_result("static_seq", np.full(20, 0.9)),
        ])
        r_kcf = _make_bench_result("KCF", [
            _make_seq_result("fast_seq", np.full(20, 0.6)),
            _make_seq_result("static_seq", np.full(20, 0.85)),
        ])
        return {"MOSSE": r_mosse, "KCF": r_kcf}, seq_attrs

    def test_report_has_correct_trackers(self):
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        assert set(report.tracker_names) == {"MOSSE", "KCF"}

    def test_overall_slice_populated(self):
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        assert "MOSSE" in report.overall
        assert report.overall["MOSSE"].num_sequences == 2

    def test_attribute_slices_all_flags(self):
        from eovot.analysis.attribute_report import _ALL_FLAGS
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        for flag in _ALL_FLAGS:
            assert flag in report.attribute_slices

    def test_fm_slice_has_fast_sequence(self):
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        fm_mosse = report.attribute_slices["FM"]["MOSSE"]
        assert fm_mosse.num_sequences >= 1

    def test_overall_mean_iou_correct(self):
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        # MOSSE: mean IoU over fast_seq (0.5) and static_seq (0.9) = 0.7
        assert report.overall["MOSSE"].mean_iou == pytest.approx(0.7, rel=0.01)

    def test_markdown_table_contains_trackers(self):
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        table = report.markdown_table()
        assert "MOSSE" in table
        assert "KCF" in table

    def test_markdown_table_contains_attributes(self):
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        table = report.markdown_table()
        assert "Fast Motion" in table
        assert "Scale Variation" in table

    def test_to_dict_serialisable(self):
        import json
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        d = report.to_dict()
        # Should be JSON-serialisable (no numpy types)
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_save_json(self, tmp_path):
        import json
        results, attrs = self._build_inputs()
        report = generate_attribute_report(results, attrs)
        out = tmp_path / "report.json"
        report.save_json(str(out))
        assert out.exists()
        with open(out) as fh:
            loaded = json.load(fh)
        assert "trackers" in loaded
        assert "per_attribute" in loaded
