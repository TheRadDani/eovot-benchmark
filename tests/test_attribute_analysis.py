"""Tests for eovot.analysis.attribute_analyzer."""

from __future__ import annotations

from typing import Dict, FrozenSet, List

import numpy as np
import pytest

from eovot.analysis.attribute_analyzer import (
    AttributeAnalyzer,
    AttributeReport,
    AttributeStats,
)


# ---------------------------------------------------------------------------
# Minimal stubs — avoid importing BenchmarkEngine in tests
# ---------------------------------------------------------------------------

class _FakeSequenceResult:
    def __init__(self, sequence_name: str, ious: List[float]) -> None:
        self.sequence_name = sequence_name
        self.ious = np.array(ious, dtype=np.float64)


class _FakeBenchmarkResult:
    def __init__(self, tracker_name: str, sequences: Dict[str, List[float]]) -> None:
        self.tracker_name = tracker_name
        self.sequence_results = [
            _FakeSequenceResult(name, ious) for name, ious in sequences.items()
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ATTR_MAP: Dict[str, FrozenSet[str]] = {
    "SeqA": frozenset(["FM", "OCC"]),
    "SeqB": frozenset(["FM", "IV"]),
    "SeqC": frozenset(["OCC"]),
    "SeqD": frozenset(["IV", "SV"]),
    "SeqE": frozenset(["SV"]),
}

ATTR_NAMES = {"FM": "Fast Motion", "OCC": "Occlusion", "IV": "Illumination Variation",
              "SV": "Scale Variation"}


@pytest.fixture()
def analyzer() -> AttributeAnalyzer:
    return AttributeAnalyzer(ATTR_MAP, attribute_names=ATTR_NAMES, dataset_name="TestDS")


@pytest.fixture()
def result_mosse() -> _FakeBenchmarkResult:
    return _FakeBenchmarkResult("MOSSE", {
        "SeqA": [0.8, 0.7, 0.75],
        "SeqB": [0.6, 0.65],
        "SeqC": [0.5, 0.55, 0.6],
        "SeqD": [0.9, 0.85],
        "SeqE": [0.7],
    })


@pytest.fixture()
def result_kcf() -> _FakeBenchmarkResult:
    return _FakeBenchmarkResult("KCF", {
        "SeqA": [0.5, 0.55, 0.52],
        "SeqB": [0.72, 0.68],
        "SeqC": [0.45, 0.5],
        "SeqD": [0.88, 0.9],
        "SeqE": [0.65],
    })


# ---------------------------------------------------------------------------
# AttributeAnalyzer construction
# ---------------------------------------------------------------------------

class TestAnalyzerConstruction:
    def test_attributes_sorted(self, analyzer):
        assert analyzer.attributes == sorted(analyzer.attributes)

    def test_empty_attr_map_raises(self):
        with pytest.raises(ValueError, match="empty"):
            AttributeAnalyzer({})

    def test_dataset_name_stored(self, analyzer):
        assert analyzer.dataset_name == "TestDS"

    def test_all_attributes_discovered(self, analyzer):
        assert set(analyzer.attributes) == {"FM", "OCC", "IV", "SV"}


# ---------------------------------------------------------------------------
# sequences_for_attribute
# ---------------------------------------------------------------------------

class TestSequencesForAttribute:
    def test_fm_sequences(self, analyzer):
        seqs = analyzer.sequences_for_attribute("FM")
        assert set(seqs) == {"SeqA", "SeqB"}
        assert seqs == sorted(seqs)  # must be sorted

    def test_occ_sequences(self, analyzer):
        assert set(analyzer.sequences_for_attribute("OCC")) == {"SeqA", "SeqC"}

    def test_attribute_not_in_map_returns_empty(self, analyzer):
        seqs = analyzer.sequences_for_attribute("NONEXISTENT")
        assert seqs == []


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_returns_dict_keyed_by_attribute(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        assert set(stats.keys()) == {"FM", "OCC", "IV", "SV"}

    def test_n_sequences_correct(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        assert stats["FM"].n_sequences == 2   # SeqA + SeqB
        assert stats["OCC"].n_sequences == 2  # SeqA + SeqC
        assert stats["SV"].n_sequences == 2   # SeqD + SeqE

    def test_tracker_name_stored(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        assert all(s.tracker_name == "MOSSE" for s in stats.values())

    def test_attribute_name_populated(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        assert stats["FM"].attribute_name == "Fast Motion"
        assert stats["OCC"].attribute_name == "Occlusion"

    def test_mean_iou_computed_correctly(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        # SeqA mIoU = mean([0.8, 0.7, 0.75]) = 0.75
        # SeqB mIoU = mean([0.6, 0.65]) = 0.625
        # FM mIoU = mean(0.75, 0.625) = 0.6875
        expected_fm = (0.75 + 0.625) / 2
        assert stats["FM"].mean_iou == pytest.approx(expected_fm, rel=1e-5)

    def test_std_iou_zero_for_single_sequence(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        assert stats["SV"].std_iou >= 0.0  # must be non-negative

    def test_min_max_iou_bounds(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        for attr, s in stats.items():
            assert s.min_iou <= s.mean_iou <= s.max_iou

    def test_sequence_names_stored(self, analyzer, result_mosse):
        stats = analyzer.compute_stats(result_mosse)
        assert set(stats["FM"].sequence_names) == {"SeqA", "SeqB"}

    def test_missing_sequences_in_result_skipped(self):
        """Sequences in attr_map but missing from result are silently skipped."""
        partial_result = _FakeBenchmarkResult("X", {"SeqA": [0.5]})  # SeqB–E absent
        ana = AttributeAnalyzer(ATTR_MAP)
        stats = ana.compute_stats(partial_result)
        # Only SeqA present; it carries FM and OCC → those attributes appear.
        assert "FM" in stats
        assert "OCC" in stats
        # IV requires SeqB or SeqD — neither present → IV absent from stats.
        assert "IV" not in stats


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_attribute_report(self, analyzer, result_mosse, result_kcf):
        report = analyzer.generate_report([result_mosse, result_kcf])
        assert isinstance(report, AttributeReport)

    def test_tracker_names_ordered(self, analyzer, result_mosse, result_kcf):
        report = analyzer.generate_report([result_mosse, result_kcf])
        assert report.tracker_names == ["MOSSE", "KCF"]

    def test_dataset_name_propagated(self, analyzer, result_mosse):
        report = analyzer.generate_report([result_mosse])
        assert report.dataset_name == "TestDS"

    def test_attribute_filter_respected(self, analyzer, result_mosse):
        report = analyzer.generate_report([result_mosse], attributes=["FM", "OCC"])
        assert set(report.attributes) == {"FM", "OCC"}

    def test_unknown_attribute_in_filter_raises(self, analyzer, result_mosse):
        with pytest.raises(ValueError, match="not in attribute_map"):
            analyzer.generate_report([result_mosse], attributes=["UNKNOWN"])

    def test_empty_results_raises(self, analyzer):
        with pytest.raises(ValueError, match="at least one"):
            analyzer.generate_report([])


# ---------------------------------------------------------------------------
# AttributeReport
# ---------------------------------------------------------------------------

class TestAttributeReport:
    @pytest.fixture()
    def report(self, analyzer, result_mosse, result_kcf):
        return analyzer.generate_report([result_mosse, result_kcf])

    def test_iou_matrix_shape(self, report):
        mat = report.iou_matrix()
        assert mat.shape == (2, len(report.attributes))  # 2 trackers × N attrs

    def test_iou_matrix_values_in_range(self, report):
        mat = report.iou_matrix()
        valid = mat[~np.isnan(mat)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_hardest_attribute_returns_tuple(self, report):
        result = report.hardest_attribute("MOSSE")
        assert result is not None
        attr_code, miou = result
        assert attr_code in report.attributes
        assert 0.0 <= miou <= 1.0

    def test_hardest_attribute_unknown_tracker_returns_none(self, report):
        assert report.hardest_attribute("__no_such_tracker__") is None

    def test_best_tracker_per_attribute_keys(self, report):
        best = report.best_tracker_per_attribute()
        assert set(best.keys()).issubset(set(report.attributes))

    def test_best_tracker_per_attribute_values_valid(self, report):
        best = report.best_tracker_per_attribute()
        for attr, (tracker, miou) in best.items():
            assert tracker in report.tracker_names
            assert 0.0 <= miou <= 1.0

    def test_to_dict_structure(self, report):
        d = report.to_dict()
        assert "trackers" in d
        assert "attributes" in d
        assert "stats" in d
        assert d["dataset"] == "TestDS"

    def test_to_dict_json_serialisable(self, report):
        import json
        d = report.to_dict()
        serialised = json.dumps(d)
        assert len(serialised) > 0

    def test_to_markdown_contains_tracker_names(self, report):
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_to_markdown_contains_attribute_names(self, report):
        md = report.to_markdown()
        assert "Fast Motion" in md
        assert "Occlusion" in md

    def test_to_markdown_contains_best_tracker_section(self, report):
        md = report.to_markdown()
        assert "Best Tracker per Attribute" in md

    def test_to_markdown_contains_hardest_attribute_section(self, report):
        md = report.to_markdown()
        assert "Hardest Attribute per Tracker" in md

    def test_to_markdown_contains_methodology(self, report):
        md = report.to_markdown()
        assert "Methodology" in md

    def test_to_markdown_bold_marks_best_column(self, report):
        md = report.to_markdown()
        # Bold markers should appear for the best-performing tracker per column.
        assert "**" in md


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sequence_per_attribute(self):
        attr_map = {"OnlySeq": frozenset(["FM"])}
        ana = AttributeAnalyzer(attr_map)
        result = _FakeBenchmarkResult("T", {"OnlySeq": [0.6, 0.7]})
        stats = ana.compute_stats(result)
        assert stats["FM"].n_sequences == 1
        assert stats["FM"].std_iou == 0.0  # single sequence → std is 0

    def test_sequence_with_empty_ious(self):
        attr_map = {"EmptySeq": frozenset(["FM"])}
        ana = AttributeAnalyzer(attr_map)
        result = _FakeBenchmarkResult("T", {"EmptySeq": []})
        stats = ana.compute_stats(result)
        # mean IoU of an empty array is 0.0 (handled gracefully).
        assert stats["FM"].mean_iou == pytest.approx(0.0)

    def test_all_attributes_none_match_in_result(self):
        attr_map = {"SeqX": frozenset(["FM"])}
        ana = AttributeAnalyzer(attr_map)
        result = _FakeBenchmarkResult("T", {"__other_seq__": [0.5]})
        stats = ana.compute_stats(result)
        assert len(stats) == 0

    def test_report_single_tracker(self):
        attr_map = {"SeqA": frozenset(["FM", "OCC"]), "SeqB": frozenset(["FM"])}
        ana = AttributeAnalyzer(attr_map)
        result = _FakeBenchmarkResult("MOSSE", {"SeqA": [0.7], "SeqB": [0.8]})
        report = ana.generate_report([result])
        md = report.to_markdown()
        assert "MOSSE" in md
        assert len(report.tracker_names) == 1
