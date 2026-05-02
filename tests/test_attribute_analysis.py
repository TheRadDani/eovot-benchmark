"""Tests for eovot.analysis.attribute — AttributeAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.analysis.attribute import (
    STANDARD_ATTRIBUTES,
    AttributeAnalysis,
    AttributeAnalyzer,
    AttributeResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def attr_map() -> dict:
    return {
        "seq_a": ["fast_motion", "occlusion"],
        "seq_b": ["occlusion", "scale_variation"],
        "seq_c": ["fast_motion", "motion_blur"],
        "seq_d": ["illumination_variation"],
    }


@pytest.fixture
def analyzer(attr_map) -> AttributeAnalyzer:
    return AttributeAnalyzer(attribute_map=attr_map)


@pytest.fixture
def sequence_ious() -> dict:
    rng = np.random.default_rng(42)
    return {
        "seq_a": rng.uniform(0.3, 0.8, 50),
        "seq_b": rng.uniform(0.5, 0.9, 60),
        "seq_c": rng.uniform(0.1, 0.6, 40),
        "seq_d": rng.uniform(0.6, 1.0, 70),
    }


@pytest.fixture
def sequence_distances() -> dict:
    rng = np.random.default_rng(7)
    return {
        "seq_a": rng.uniform(0.0, 30.0, 50),
        "seq_b": rng.uniform(0.0, 20.0, 60),
        "seq_c": rng.uniform(5.0, 40.0, 40),
        "seq_d": rng.uniform(0.0, 15.0, 70),
    }


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TestStandardAttributes:
    def test_contains_core_challenges(self):
        for attr in ("fast_motion", "occlusion", "scale_variation", "motion_blur"):
            assert attr in STANDARD_ATTRIBUTES

    def test_minimum_count(self):
        assert len(STANDARD_ATTRIBUTES) >= 10

    def test_all_lowercase(self):
        for attr in STANDARD_ATTRIBUTES:
            assert attr == attr.lower()


# ---------------------------------------------------------------------------
# AttributeAnalyzer construction and registration
# ---------------------------------------------------------------------------


class TestAttributeAnalyzerConstruction:
    def test_default_construction(self):
        az = AttributeAnalyzer()
        assert az._attribute_map == {}

    def test_init_with_map(self, attr_map):
        az = AttributeAnalyzer(attribute_map=attr_map)
        assert "seq_a" in az._attribute_map
        assert "fast_motion" in az._attribute_map["seq_a"]

    def test_attributes_normalised_to_lowercase(self):
        az = AttributeAnalyzer(attribute_map={"seq_x": ["Fast_Motion", "OCCLUSION"]})
        assert az._attribute_map["seq_x"] == ["fast_motion", "occlusion"]

    def test_register_sequence(self, sequence_ious):
        az = AttributeAnalyzer()
        az.register_sequence("seq_a", ["fast_motion", "occlusion"])
        result = az.analyze("T", sequence_ious)
        assert "fast_motion" in result.attribute_results

    def test_register_normalises_case(self):
        az = AttributeAnalyzer()
        az.register_sequence("seq_x", ["FastMotion", "OCCLUSION"])
        assert az._attribute_map["seq_x"] == ["fastmotion", "occlusion"]

    def test_load_otb_attributes(self, tmp_path, sequence_ious):
        f = tmp_path / "attrs.txt"
        f.write_text(
            "# comment\n"
            "seq_a, fast_motion, occlusion\n"
            "seq_b, scale_variation\n"
            "\n"
            "seq_c, motion_blur\n"
        )
        az = AttributeAnalyzer()
        az.load_otb_attributes(str(f))
        assert "fast_motion" in az._attribute_map["seq_a"]
        assert "scale_variation" in az._attribute_map["seq_b"]
        result = az.analyze("T", sequence_ious)
        assert "fast_motion" in result.attribute_results

    def test_load_otb_attributes_file_not_found(self):
        az = AttributeAnalyzer()
        with pytest.raises(FileNotFoundError):
            az.load_otb_attributes("/nonexistent/path/attrs.txt")

    def test_load_otb_attributes_skips_short_lines(self, tmp_path, sequence_ious):
        f = tmp_path / "attrs.txt"
        # A line with only the sequence name (no attributes) should be skipped.
        f.write_text("seq_a\nseq_b, occlusion\n")
        az = AttributeAnalyzer()
        az.load_otb_attributes(str(f))
        assert "seq_a" not in az._attribute_map
        assert "seq_b" in az._attribute_map


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_attribute_analysis(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        assert isinstance(result, AttributeAnalysis)
        assert result.tracker_name == "MOSSE"

    def test_expected_attributes_present(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        for attr in ("fast_motion", "occlusion", "scale_variation", "illumination_variation"):
            assert attr in result.attribute_results

    def test_attribute_sequence_counts(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        # fast_motion: seq_a + seq_c
        assert result.attribute_results["fast_motion"].num_sequences == 2
        # occlusion: seq_a + seq_b
        assert result.attribute_results["occlusion"].num_sequences == 2
        # illumination_variation: seq_d only
        assert result.attribute_results["illumination_variation"].num_sequences == 1

    def test_mean_iou_in_range(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        for ar in result.attribute_results.values():
            assert 0.0 <= ar.mean_iou <= 1.0

    def test_overall_matches_global_mean(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        expected = float(np.concatenate(list(sequence_ious.values())).mean())
        assert result.overall.mean_iou == pytest.approx(expected, abs=1e-6)

    def test_success_auc_in_range(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        for ar in result.attribute_results.values():
            assert 0.0 <= ar.success_auc <= 1.0

    def test_precision_auc_zero_without_distances(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious, sequence_distances=None)
        for ar in result.attribute_results.values():
            assert ar.precision_auc == pytest.approx(0.0)

    def test_precision_auc_nonzero_with_distances(self, analyzer, sequence_ious, sequence_distances):
        result = analyzer.analyze("MOSSE", sequence_ious, sequence_distances=sequence_distances)
        # At least some attributes should have non-trivial precision AUC
        aucs = [ar.precision_auc for ar in result.attribute_results.values()]
        assert max(aucs) > 0.0

    def test_sequences_not_in_attribute_map_skipped(self, sequence_ious):
        az = AttributeAnalyzer(attribute_map={"seq_a": ["occlusion"]})
        result = az.analyze("T", sequence_ious)
        # Only seq_a contributes to attribute data
        assert "occlusion" in result.attribute_results
        assert result.attribute_results["occlusion"].num_sequences == 1
        assert result.attribute_results["occlusion"].sequence_names == ["seq_a"]

    def test_missing_sequences_excluded_from_overall(self):
        partial_ious = {"seq_a": np.array([0.5, 0.6, 0.7])}
        az = AttributeAnalyzer(attribute_map={"seq_a": ["fast_motion"]})
        result = az.analyze("T", partial_ious)
        expected = float(np.array([0.5, 0.6, 0.7]).mean())
        assert result.overall.mean_iou == pytest.approx(expected, abs=1e-6)

    def test_empty_distances_dict(self, analyzer, sequence_ious):
        result = analyzer.analyze("T", sequence_ious, sequence_distances={})
        for ar in result.attribute_results.values():
            assert ar.precision_auc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AttributeAnalysis helpers
# ---------------------------------------------------------------------------


class TestAttributeAnalysisHelpers:
    def test_sorted_by_difficulty(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        sorted_attrs = result.sorted_by_difficulty
        ious = [r.mean_iou for r in sorted_attrs]
        assert ious == sorted(ious), "sorted_by_difficulty should be ascending"

    def test_hardest_attribute_is_first(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        hardest = result.hardest_attribute
        assert hardest is not None
        all_ious = [r.mean_iou for r in result.attribute_results.values()]
        assert hardest.mean_iou == pytest.approx(min(all_ious), abs=1e-8)

    def test_easiest_attribute_is_last(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        easiest = result.easiest_attribute
        assert easiest is not None
        all_ious = [r.mean_iou for r in result.attribute_results.values()]
        assert easiest.mean_iou == pytest.approx(max(all_ious), abs=1e-8)

    def test_no_attributes_returns_none_hardest(self, sequence_ious):
        az = AttributeAnalyzer()  # empty map
        result = az.analyze("T", sequence_ious)
        assert result.hardest_attribute is None
        assert result.easiest_attribute is None


# ---------------------------------------------------------------------------
# Multi-tracker comparison
# ---------------------------------------------------------------------------


class TestCompare:
    @pytest.fixture
    def two_analyses(self, analyzer, sequence_ious):
        rng = np.random.default_rng(99)
        other_ious = {k: rng.uniform(0.2, 0.7, len(v)) for k, v in sequence_ious.items()}
        a1 = analyzer.analyze("MOSSE", sequence_ious)
        a2 = analyzer.analyze("KCF", other_ious)
        return a1, a2

    def test_compare_returns_nested_dict(self, two_analyses):
        a1, a2 = two_analyses
        analyzer = AttributeAnalyzer(
            attribute_map={
                "seq_a": ["fast_motion", "occlusion"],
                "seq_b": ["occlusion", "scale_variation"],
                "seq_c": ["fast_motion", "motion_blur"],
                "seq_d": ["illumination_variation"],
            }
        )
        table = analyzer.compare([a1, a2])
        assert "fast_motion" in table
        assert "MOSSE" in table["fast_motion"]
        assert "KCF" in table["fast_motion"]

    def test_compare_values_in_range(self, analyzer, two_analyses):
        a1, a2 = two_analyses
        table = analyzer.compare([a1, a2])
        for attr_vals in table.values():
            for val in attr_vals.values():
                assert 0.0 <= val <= 1.0

    def test_format_comparison_table_markdown(self, analyzer, sequence_ious):
        result = analyzer.analyze("MOSSE", sequence_ious)
        md = analyzer.format_comparison_table([result])
        assert "| Attribute |" in md
        assert "MOSSE" in md
        assert "**Overall**" in md

    def test_format_comparison_table_missing_entry(self, sequence_ious):
        az1 = AttributeAnalyzer(attribute_map={"seq_a": ["fast_motion"]})
        az2 = AttributeAnalyzer(attribute_map={"seq_b": ["scale_variation"]})
        rng = np.random.default_rng(0)
        iou_a = {"seq_a": rng.uniform(0.3, 0.8, 30)}
        iou_b = {"seq_b": rng.uniform(0.4, 0.9, 40)}
        a1 = az1.analyze("T1", iou_a)
        a2 = az2.analyze("T2", iou_b)
        # Combine both analyses; T1 has no scale_variation, T2 has no fast_motion
        md = az1.format_comparison_table([a1, a2])
        assert " — |" in md  # missing entry placeholder

    def test_difficulty_ranking_sorted_ascending(self, analyzer, two_analyses):
        a1, a2 = two_analyses
        ranking = analyzer.difficulty_ranking([a1, a2])
        values = [v for _, v in ranking]
        assert values == sorted(values), "Difficulty ranking should be ascending"

    def test_difficulty_ranking_contains_all_attrs(self, analyzer, two_analyses):
        a1, a2 = two_analyses
        ranking = analyzer.difficulty_ranking([a1, a2])
        all_attrs = {attr for r in (a1, a2) for attr in r.attribute_results}
        ranked_attrs = {attr for attr, _ in ranking}
        assert ranked_attrs == all_attrs


# ---------------------------------------------------------------------------
# from_benchmark_result integration
# ---------------------------------------------------------------------------


class TestFromBenchmarkResult:
    """Tests for the BenchmarkResult integration path."""

    class _FakeSequenceResult:
        def __init__(self, name, ious, dists=None):
            self.sequence_name = name
            self.ious = ious
            self.center_distances = dists

    class _FakeBenchmarkResult:
        def __init__(self, tracker_name, seq_results):
            self.tracker_name = tracker_name
            self.sequence_results = seq_results

    def test_from_benchmark_result_basic(self, analyzer, sequence_ious, sequence_distances):
        sr_list = [
            self._FakeSequenceResult(
                name, sequence_ious[name], sequence_distances.get(name)
            )
            for name in sequence_ious
        ]
        result_obj = self._FakeBenchmarkResult("KCF", sr_list)
        analysis = analyzer.from_benchmark_result(result_obj)
        assert analysis.tracker_name == "KCF"
        assert "fast_motion" in analysis.attribute_results

    def test_from_benchmark_result_without_distances(self, analyzer, sequence_ious):
        sr_list = [
            self._FakeSequenceResult(name, sequence_ious[name], None)
            for name in sequence_ious
        ]
        result_obj = self._FakeBenchmarkResult("MOSSE", sr_list)
        analysis = analyzer.from_benchmark_result(result_obj)
        for ar in analysis.attribute_results.values():
            assert ar.precision_auc == pytest.approx(0.0)
