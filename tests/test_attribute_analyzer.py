"""Tests for eovot.analysis.attributes — AttributeAnalyzer and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.attributes import (
    AttributeAnalyzer,
    AttributePerformance,
    SequenceAttribute,
)
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_profiling(fps: float = 30.0, tracker_name: str = "TestTracker") -> ProfilingResult:
    mean_ms = 1000.0 / fps
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=10,
        fps=fps,
        latency_mean_ms=mean_ms,
        latency_std_ms=0.0,
        latency_p95_ms=mean_ms,
        peak_memory_mb=50.0,
    )


def _make_sequence_result(
    name: str,
    mean_iou: float,
    fps: float = 30.0,
    success_auc: float = 0.5,
    precision_auc: float = 0.4,
    tracker_name: str = "TestTracker",
) -> SequenceResult:
    acc = AccuracyMetrics(
        mean_iou=mean_iou,
        success_auc=success_auc,
        precision_auc=precision_auc,
    )
    prof = _make_profiling(fps, tracker_name)
    return SequenceResult(
        sequence_name=name,
        ious=np.array([mean_iou]),
        profiling=prof,
        accuracy_metrics=acc,
    )


def _make_benchmark_result(
    seq_names_ious: list,
    tracker_name: str = "TestTracker",
) -> BenchmarkResult:
    br = BenchmarkResult(tracker_name=tracker_name, dataset_name="TestDataset")
    for name, iou in seq_names_ious:
        br.sequence_results.append(
            _make_sequence_result(name, iou, tracker_name=tracker_name)
        )
    return br


def _attr_map(*pairs) -> dict:
    """Build attribute_map from (name, {attr, ...}) pairs."""
    return {name: attrs for name, attrs in pairs}


# ---------------------------------------------------------------------------
# SequenceAttribute
# ---------------------------------------------------------------------------

class TestSequenceAttribute:
    def test_all_members_have_string_values(self):
        for attr in SequenceAttribute:
            assert isinstance(attr.value, str)
            assert len(attr.value) > 0

    def test_expected_members_exist(self):
        names = {a.name for a in SequenceAttribute}
        assert "OCCLUSION" in names
        assert "FAST_MOTION" in names
        assert "SCALE_CHANGE" in names
        assert "ROTATION" in names
        assert "BACKGROUND_CLUTTER" in names
        assert "LOW_RESOLUTION" in names
        assert "OUT_OF_VIEW" in names

    def test_eleven_members(self):
        assert len(list(SequenceAttribute)) == 11


# ---------------------------------------------------------------------------
# AttributePerformance
# ---------------------------------------------------------------------------

class TestAttributePerformance:
    def _make(self, iou=0.6, sauc=0.5, pauc=0.4, fps=25.0, n=3):
        return AttributePerformance(
            attribute=SequenceAttribute.OCCLUSION,
            num_sequences=n,
            mean_iou=iou,
            mean_success_auc=sauc,
            mean_precision_auc=pauc,
            mean_fps=fps,
            tracker_name="KCF",
        )

    def test_str_contains_attribute_and_iou(self):
        p = self._make()
        s = str(p)
        assert "occlusion" in s
        assert "0.6000" in s

    def test_str_with_none_sauc(self):
        p = AttributePerformance(
            attribute=SequenceAttribute.FAST_MOTION,
            num_sequences=2,
            mean_iou=0.5,
            mean_success_auc=None,
            mean_precision_auc=None,
            mean_fps=10.0,
        )
        assert "N/A" in str(p)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        assert set(d.keys()) == {
            "attribute", "num_sequences", "mean_iou",
            "mean_success_auc", "mean_precision_auc", "mean_fps",
        }

    def test_to_dict_values_rounded(self):
        d = self._make(iou=0.123456).to_dict()
        assert d["mean_iou"] == 0.1235

    def test_to_dict_none_auc(self):
        p = AttributePerformance(
            attribute=SequenceAttribute.OCCLUSION,
            num_sequences=2,
            mean_iou=0.5,
            mean_success_auc=None,
            mean_precision_auc=None,
            mean_fps=10.0,
        )
        assert p.to_dict()["mean_success_auc"] is None
        assert p.to_dict()["mean_precision_auc"] is None


# ---------------------------------------------------------------------------
# AttributeAnalyzer — constructor
# ---------------------------------------------------------------------------

class TestAttributeAnalyzerConstructor:
    def test_default_min_sequences(self):
        a = AttributeAnalyzer()
        assert a.min_sequences == 2

    def test_custom_min_sequences(self):
        a = AttributeAnalyzer(min_sequences=5)
        assert a.min_sequences == 5

    def test_min_sequences_zero_raises(self):
        with pytest.raises(ValueError):
            AttributeAnalyzer(min_sequences=0)

    def test_min_sequences_negative_raises(self):
        with pytest.raises(ValueError):
            AttributeAnalyzer(min_sequences=-3)

    def test_min_sequences_one_allowed(self):
        a = AttributeAnalyzer(min_sequences=1)
        assert a.min_sequences == 1


# ---------------------------------------------------------------------------
# AttributeAnalyzer.analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_empty_attribute_map_returns_empty(self):
        br = _make_benchmark_result([("seq1", 0.7), ("seq2", 0.6)])
        analyzer = AttributeAnalyzer(min_sequences=1)
        result = analyzer.analyze(br, {})
        assert result == {}

    def test_sequences_not_in_map_are_skipped(self):
        br = _make_benchmark_result([("seq1", 0.7), ("seq2", 0.6)])
        attr_map = {"seq99": {SequenceAttribute.OCCLUSION}}
        analyzer = AttributeAnalyzer(min_sequences=1)
        result = analyzer.analyze(br, attr_map)
        assert result == {}

    def test_single_attribute_above_threshold(self):
        br = _make_benchmark_result([("s1", 0.8), ("s2", 0.6), ("s3", 0.4)])
        attr_map = {
            "s1": {SequenceAttribute.OCCLUSION},
            "s2": {SequenceAttribute.OCCLUSION},
        }
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert SequenceAttribute.OCCLUSION in result
        perf = result[SequenceAttribute.OCCLUSION]
        assert perf.num_sequences == 2
        assert pytest.approx(perf.mean_iou, abs=1e-4) == 0.7

    def test_attribute_below_min_sequences_excluded(self):
        br = _make_benchmark_result([("s1", 0.8), ("s2", 0.6)])
        attr_map = {
            "s1": {SequenceAttribute.OCCLUSION},  # only 1 sequence
        }
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert SequenceAttribute.OCCLUSION not in result

    def test_sequence_can_have_multiple_attributes(self):
        br = _make_benchmark_result([("s1", 0.9), ("s2", 0.7)])
        attr_map = {
            "s1": {SequenceAttribute.FAST_MOTION, SequenceAttribute.ROTATION},
            "s2": {SequenceAttribute.FAST_MOTION, SequenceAttribute.ROTATION},
        }
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert SequenceAttribute.FAST_MOTION in result
        assert SequenceAttribute.ROTATION in result
        assert result[SequenceAttribute.FAST_MOTION].num_sequences == 2
        assert result[SequenceAttribute.ROTATION].num_sequences == 2

    def test_mean_iou_correctness(self):
        br = _make_benchmark_result([("s1", 0.4), ("s2", 0.6), ("s3", 0.8)])
        attr_map = {k: {SequenceAttribute.SCALE_CHANGE} for k in ["s1", "s2", "s3"]}
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert pytest.approx(result[SequenceAttribute.SCALE_CHANGE].mean_iou, abs=1e-6) == 0.6

    def test_tracker_name_is_set(self):
        br = _make_benchmark_result([("s1", 0.5), ("s2", 0.5)], tracker_name="KCF")
        attr_map = {"s1": {SequenceAttribute.OCCLUSION}, "s2": {SequenceAttribute.OCCLUSION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert result[SequenceAttribute.OCCLUSION].tracker_name == "KCF"

    def test_only_sequenceattribute_members_counted(self):
        """Non-SequenceAttribute objects in attr sets are silently ignored."""
        br = _make_benchmark_result([("s1", 0.5), ("s2", 0.5)])
        attr_map = {
            "s1": {"some_string_attribute", SequenceAttribute.OCCLUSION},
            "s2": {"some_string_attribute", SequenceAttribute.OCCLUSION},
        }
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert SequenceAttribute.OCCLUSION in result
        assert len(result) == 1

    def test_fps_averaged(self):
        br = BenchmarkResult(tracker_name="T", dataset_name="D")
        for fps in [20.0, 40.0]:
            sr = _make_sequence_result(f"s{fps}", 0.5, fps=fps)
            br.sequence_results.append(sr)
        attr_map = {"s20.0": {SequenceAttribute.FAST_MOTION}, "s40.0": {SequenceAttribute.FAST_MOTION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze(br, attr_map)
        assert pytest.approx(result[SequenceAttribute.FAST_MOTION].mean_fps, abs=1e-4) == 30.0


# ---------------------------------------------------------------------------
# AttributeAnalyzer.analyze_dataset
# ---------------------------------------------------------------------------

class TestAnalyzeDataset:
    def test_uses_sequence_attributes_field(self):
        """analyze_dataset should pull attributes from dataset.seq.attributes."""
        from eovot.datasets.synthetic import SyntheticDataset

        ds = SyntheticDataset(num_sequences=4, num_frames=2, motion="linear", seed=0)
        br = BenchmarkResult(tracker_name="KCF", dataset_name="Synthetic")
        for seq in ds:
            br.sequence_results.append(_make_sequence_result(seq.name, 0.7))

        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze_dataset(br, ds)
        # linear motion tags FAST_MOTION + OUT_OF_VIEW — at least one should appear
        assert len(result) >= 1

    def test_sequences_without_attributes_skipped(self):
        from eovot.datasets.base import Sequence, BaseDataset

        class FakeDataset(BaseDataset):
            def __len__(self): return 2
            def __getitem__(self, idx):
                return Sequence(
                    name=f"seq{idx}",
                    frame_paths=[],
                    ground_truth=np.zeros((1, 4)),
                    attributes=None,
                )

        ds = FakeDataset()
        br = _make_benchmark_result([("seq0", 0.5), ("seq1", 0.5)])
        analyzer = AttributeAnalyzer(min_sequences=1)
        result = analyzer.analyze_dataset(br, ds)
        assert result == {}


# ---------------------------------------------------------------------------
# AttributeAnalyzer.compare_trackers
# ---------------------------------------------------------------------------

class TestCompareTrackers:
    def test_single_tracker_comparison(self):
        br = _make_benchmark_result([("s1", 0.7), ("s2", 0.5)], tracker_name="A")
        attr_map = {"s1": {SequenceAttribute.OCCLUSION}, "s2": {SequenceAttribute.OCCLUSION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br], attr_map)
        assert SequenceAttribute.OCCLUSION in comp
        assert len(comp[SequenceAttribute.OCCLUSION]) == 1

    def test_two_trackers_both_present(self):
        seqs = [("s1", 0.8), ("s2", 0.6)]
        br_a = _make_benchmark_result(seqs, tracker_name="A")
        br_b = _make_benchmark_result([("s1", 0.5), ("s2", 0.3)], tracker_name="B")
        attr_map = {"s1": {SequenceAttribute.FAST_MOTION}, "s2": {SequenceAttribute.FAST_MOTION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br_a, br_b], attr_map)
        perfs = comp[SequenceAttribute.FAST_MOTION]
        assert len(perfs) == 2
        tracker_names = [p.tracker_name for p in perfs]
        assert "A" in tracker_names
        assert "B" in tracker_names

    def test_tracker_order_preserved(self):
        br_a = _make_benchmark_result([("s1", 0.9), ("s2", 0.9)], tracker_name="A")
        br_b = _make_benchmark_result([("s1", 0.5), ("s2", 0.5)], tracker_name="B")
        attr_map = {"s1": {SequenceAttribute.ROTATION}, "s2": {SequenceAttribute.ROTATION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br_a, br_b], attr_map)
        perfs = comp[SequenceAttribute.ROTATION]
        assert perfs[0].tracker_name == "A"
        assert perfs[1].tracker_name == "B"

    def test_attributes_sorted_alphabetically(self):
        br = _make_benchmark_result(
            [("s1", 0.5), ("s2", 0.5), ("s3", 0.5), ("s4", 0.5)], tracker_name="T"
        )
        attr_map = {
            "s1": {SequenceAttribute.ROTATION},
            "s2": {SequenceAttribute.ROTATION},
            "s3": {SequenceAttribute.FAST_MOTION},
            "s4": {SequenceAttribute.FAST_MOTION},
        }
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br], attr_map)
        keys = list(comp.keys())
        assert keys == sorted(keys, key=lambda a: a.value)

    def test_empty_results_returns_empty(self):
        analyzer = AttributeAnalyzer(min_sequences=1)
        comp = analyzer.compare_trackers([], {})
        assert comp == {}


# ---------------------------------------------------------------------------
# AttributeAnalyzer.to_markdown_table
# ---------------------------------------------------------------------------

class TestToMarkdownTable:
    def _analyze_two_attrs(self):
        br = _make_benchmark_result(
            [("s1", 0.9), ("s2", 0.7), ("s3", 0.5), ("s4", 0.3)], tracker_name="KCF"
        )
        attr_map = {
            "s1": {SequenceAttribute.FAST_MOTION},
            "s2": {SequenceAttribute.FAST_MOTION},
            "s3": {SequenceAttribute.OCCLUSION},
            "s4": {SequenceAttribute.OCCLUSION},
        }
        analyzer = AttributeAnalyzer(min_sequences=2)
        return analyzer.analyze(br, attr_map), "KCF"

    def test_empty_analysis_returns_placeholder(self):
        analyzer = AttributeAnalyzer()
        md = analyzer.to_markdown_table({})
        assert "No attribute data" in md

    def test_contains_tracker_name_in_title(self):
        analysis, name = self._analyze_two_attrs()
        md = AttributeAnalyzer().to_markdown_table(analysis, tracker_name=name)
        assert "KCF" in md

    def test_header_row_present(self):
        analysis, _ = self._analyze_two_attrs()
        md = AttributeAnalyzer().to_markdown_table(analysis)
        assert "mIoU" in md
        assert "# Seqs" in md

    def test_all_attributes_in_output(self):
        analysis, _ = self._analyze_two_attrs()
        md = AttributeAnalyzer().to_markdown_table(analysis)
        assert "fast_motion" in md
        assert "occlusion" in md

    def test_rows_sorted_by_iou_descending(self):
        analysis, _ = self._analyze_two_attrs()
        md = AttributeAnalyzer().to_markdown_table(analysis)
        fast_pos = md.index("fast_motion")
        occ_pos = md.index("occlusion")
        # fast_motion has higher mean_iou (0.8) so should appear first
        assert fast_pos < occ_pos


# ---------------------------------------------------------------------------
# AttributeAnalyzer.to_multi_tracker_table
# ---------------------------------------------------------------------------

class TestToMultiTrackerTable:
    def test_empty_returns_placeholder(self):
        md = AttributeAnalyzer().to_multi_tracker_table({})
        assert "No attribute comparison" in md

    def test_tracker_names_in_header(self):
        br_a = _make_benchmark_result([("s1", 0.8), ("s2", 0.6)], tracker_name="MOSSE")
        br_b = _make_benchmark_result([("s1", 0.5), ("s2", 0.3)], tracker_name="KCF")
        attr_map = {"s1": {SequenceAttribute.FAST_MOTION}, "s2": {SequenceAttribute.FAST_MOTION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br_a, br_b], attr_map)
        md = analyzer.to_multi_tracker_table(comp, tracker_names=["MOSSE", "KCF"])
        assert "MOSSE" in md
        assert "KCF" in md

    def test_iou_values_present(self):
        br_a = _make_benchmark_result([("s1", 0.8), ("s2", 0.6)], tracker_name="A")
        attr_map = {"s1": {SequenceAttribute.ROTATION}, "s2": {SequenceAttribute.ROTATION}}
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br_a], attr_map)
        md = analyzer.to_multi_tracker_table(comp, tracker_names=["A"])
        assert "0.7000" in md  # mean of 0.8 and 0.6


# ---------------------------------------------------------------------------
# AttributeAnalyzer.degradation_report
# ---------------------------------------------------------------------------

class TestDegradationReport:
    def test_empty_analysis_returns_placeholder(self):
        br = _make_benchmark_result([("s1", 0.5)])
        md = AttributeAnalyzer().degradation_report(br, {})
        assert "No attribute data" in md

    def test_contains_tracker_name(self):
        br = _make_benchmark_result([("s1", 0.8), ("s2", 0.6)], tracker_name="CSRT")
        attr_map = {"s1": {SequenceAttribute.OCCLUSION}, "s2": {SequenceAttribute.OCCLUSION}}
        md = AttributeAnalyzer(min_sequences=2).degradation_report(br, attr_map)
        assert "CSRT" in md

    def test_negative_delta_shows_down_arrow(self):
        # Overall = mean(0.8, 0.2) = 0.5; occlusion mean = 0.2 → delta = -0.3
        br = _make_benchmark_result([("s1", 0.8), ("s2", 0.2)], tracker_name="T")
        attr_map = {"s1": {SequenceAttribute.OCCLUSION}, "s2": {SequenceAttribute.OCCLUSION}}
        br_occl = _make_benchmark_result([("s1", 0.2), ("s2", 0.2)], tracker_name="T2")
        # Build a result where occlusion sequences score worse than overall
        br2 = BenchmarkResult(tracker_name="T", dataset_name="D")
        # seq1 and seq2 are high; seq3 and seq4 (occlusion) are low
        for name, iou in [("s1", 0.9), ("s2", 0.9), ("s3", 0.1), ("s4", 0.1)]:
            br2.sequence_results.append(_make_sequence_result(name, iou))
        attr_map2 = {"s3": {SequenceAttribute.OCCLUSION}, "s4": {SequenceAttribute.OCCLUSION}}
        md = AttributeAnalyzer(min_sequences=2).degradation_report(br2, attr_map2)
        assert "▼" in md

    def test_positive_delta_shows_up_arrow(self):
        # Sequences s3/s4 have high iou, overall avg is 0.5
        br = BenchmarkResult(tracker_name="T", dataset_name="D")
        for name, iou in [("s1", 0.1), ("s2", 0.1), ("s3", 0.9), ("s4", 0.9)]:
            br.sequence_results.append(_make_sequence_result(name, iou))
        attr_map = {"s3": {SequenceAttribute.FAST_MOTION}, "s4": {SequenceAttribute.FAST_MOTION}}
        md = AttributeAnalyzer(min_sequences=2).degradation_report(br, attr_map)
        assert "▲" in md

    def test_overall_iou_in_report(self):
        br = _make_benchmark_result([("s1", 0.6), ("s2", 0.4)], tracker_name="T")
        attr_map = {"s1": {SequenceAttribute.ROTATION}, "s2": {SequenceAttribute.ROTATION}}
        md = AttributeAnalyzer(min_sequences=2).degradation_report(br, attr_map)
        assert "0.5000" in md  # overall mIoU = mean(0.6, 0.4)


# ---------------------------------------------------------------------------
# SyntheticDataset attribute tagging
# ---------------------------------------------------------------------------

class TestSyntheticDatasetAttributes:
    def test_linear_sequences_have_attributes(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=3, num_frames=2, motion="linear", seed=7)
        for seq in ds:
            assert seq.attributes is not None
            assert len(seq.attributes) > 0

    def test_linear_has_fast_motion(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=3, num_frames=2, motion="linear", seed=7)
        for seq in ds:
            assert SequenceAttribute.FAST_MOTION in seq.attributes

    def test_linear_has_out_of_view(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=2, num_frames=2, motion="linear", seed=7)
        for seq in ds:
            assert SequenceAttribute.OUT_OF_VIEW in seq.attributes

    def test_circular_has_rotation(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=2, num_frames=2, motion="circular", seed=1)
        for seq in ds:
            assert SequenceAttribute.ROTATION in seq.attributes

    def test_circular_has_fast_motion(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=2, num_frames=2, motion="circular", seed=1)
        for seq in ds:
            assert SequenceAttribute.FAST_MOTION in seq.attributes

    def test_random_has_background_clutter(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=2, num_frames=2, motion="random", seed=2)
        for seq in ds:
            assert SequenceAttribute.BACKGROUND_CLUTTER in seq.attributes

    def test_attributes_are_sequenceattribute_instances(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=3, num_frames=2, motion="linear", seed=5)
        for seq in ds:
            for attr in seq.attributes:
                assert isinstance(attr, SequenceAttribute)


# ---------------------------------------------------------------------------
# End-to-end: SyntheticDataset → AttributeAnalyzer
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_analyze_dataset_from_synthetic(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=5, num_frames=2, motion="linear", seed=42)
        br = BenchmarkResult(tracker_name="KCF", dataset_name="Synthetic")
        for seq in ds:
            br.sequence_results.append(_make_sequence_result(seq.name, 0.65))
        analyzer = AttributeAnalyzer(min_sequences=2)
        result = analyzer.analyze_dataset(br, ds)
        assert len(result) >= 1
        for attr, perf in result.items():
            assert isinstance(attr, SequenceAttribute)
            assert isinstance(perf, AttributePerformance)

    def test_degradation_report_runs(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=4, num_frames=2, motion="circular", seed=0)
        br = BenchmarkResult(tracker_name="MOSSE", dataset_name="Synthetic")
        for seq in ds:
            br.sequence_results.append(_make_sequence_result(seq.name, 0.55))
        attr_map = {seq.name: seq.attributes for seq in ds if seq.attributes}
        md = AttributeAnalyzer(min_sequences=2).degradation_report(br, attr_map)
        assert "MOSSE" in md
        assert "|" in md

    def test_compare_two_trackers(self):
        from eovot.datasets.synthetic import SyntheticDataset
        ds = SyntheticDataset(num_sequences=4, num_frames=2, motion="random", seed=3)
        attr_map = {seq.name: seq.attributes for seq in ds if seq.attributes}
        br_a = BenchmarkResult(tracker_name="A", dataset_name="D")
        br_b = BenchmarkResult(tracker_name="B", dataset_name="D")
        for seq in ds:
            br_a.sequence_results.append(_make_sequence_result(seq.name, 0.7, tracker_name="A"))
            br_b.sequence_results.append(_make_sequence_result(seq.name, 0.4, tracker_name="B"))
        analyzer = AttributeAnalyzer(min_sequences=2)
        comp = analyzer.compare_trackers([br_a, br_b], attr_map)
        md = analyzer.to_multi_tracker_table(comp, tracker_names=["A", "B"])
        assert "A" in md
        assert "B" in md
        assert "|" in md
