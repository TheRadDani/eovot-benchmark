"""Tests for eovot.metrics.attribute_analysis."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.attribute_analysis import (
    AttributeAnalyzer,
    AttributeMetrics,
    AttributeReport,
    OTB_ATTRIBUTES,
    assign_synthetic_attributes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seq_data(
    seq_names: list,
    n_frames: int = 50,
    iou_level: float = 0.6,
    rng_seed: int = 0,
) -> dict:
    """Create synthetic seq_data with a controllable quality level."""
    rng = np.random.default_rng(rng_seed)
    result = {}
    for name in seq_names:
        gts = np.tile([10.0, 10.0, 40.0, 30.0], (n_frames, 1))
        noise = rng.uniform(-5.0 * (1 - iou_level), 5.0 * (1 - iou_level), size=(n_frames, 4))
        preds = np.clip(gts + noise, 0.0, None)
        preds[:, 2:] = np.maximum(preds[:, 2:], 1.0)  # ensure positive w, h
        result[name] = {"preds": preds, "gts": gts}
    return result


# ---------------------------------------------------------------------------
# assign_synthetic_attributes
# ---------------------------------------------------------------------------

class TestAssignSyntheticAttributes:
    def test_all_sequences_covered(self):
        names = ["seq1", "seq2", "seq3"]
        attrs = assign_synthetic_attributes(names, seed=0)
        assert set(attrs.keys()) == set(names)

    def test_attribute_pool_respected(self):
        names = ["a", "b", "c"]
        pool = ["FM", "OCC"]
        attrs = assign_synthetic_attributes(names, seed=1, attribute_pool=pool)
        for _, tag_set in attrs.items():
            assert tag_set <= set(pool)

    def test_reproducible_with_same_seed(self):
        names = [f"seq{i}" for i in range(10)]
        a1 = assign_synthetic_attributes(names, seed=42)
        a2 = assign_synthetic_attributes(names, seed=42)
        assert a1 == a2

    def test_different_seeds_differ(self):
        names = [f"seq{i}" for i in range(10)]
        a1 = assign_synthetic_attributes(names, seed=1)
        a2 = assign_synthetic_attributes(names, seed=2)
        # Very unlikely to be identical
        assert a1 != a2

    def test_min_max_attrs_respected(self):
        names = [f"seq{i}" for i in range(20)]
        attrs = assign_synthetic_attributes(names, seed=7, min_attrs=2, max_attrs=2)
        for _, tag_set in attrs.items():
            assert len(tag_set) == 2

    def test_defaults_use_otb_pool(self):
        names = ["s1", "s2", "s3"]
        attrs = assign_synthetic_attributes(names)
        for _, tag_set in attrs.items():
            assert tag_set <= set(OTB_ATTRIBUTES.keys())


# ---------------------------------------------------------------------------
# AttributeAnalyzer.analyze
# ---------------------------------------------------------------------------

class TestAttributeAnalyzerAnalyze:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def test_basic_returns_report(self):
        seq_names = ["car1", "person1", "ball1"]
        seq_data = _make_seq_data(seq_names)
        seq_attrs = {
            "car1":    {"FM", "SV"},
            "person1": {"OCC"},
            "ball1":   {"FM"},
        }
        report = self.analyzer.analyze(seq_data, seq_attrs, tracker_name="MOSSE")
        assert isinstance(report, AttributeReport)
        assert report.tracker_name == "MOSSE"

    def test_per_attribute_populated(self):
        seq_data = _make_seq_data(["a", "b", "c"])
        seq_attrs = {"a": {"FM"}, "b": {"FM", "OCC"}, "c": {"OCC"}}
        report = self.analyzer.analyze(seq_data, seq_attrs)
        assert "FM" in report.per_attribute
        assert "OCC" in report.per_attribute
        assert report.per_attribute["FM"].num_sequences == 2
        assert report.per_attribute["OCC"].num_sequences == 2

    def test_metrics_in_valid_range(self):
        seq_data = _make_seq_data(["s1", "s2"])
        seq_attrs = {"s1": {"FM"}, "s2": {"FM"}}
        report = self.analyzer.analyze(seq_data, seq_attrs)
        m = report.per_attribute["FM"]
        assert 0.0 <= m.mean_iou <= 1.0
        assert 0.0 <= m.success_auc <= 1.0
        assert 0.0 <= m.precision_auc <= 1.0

    def test_best_worst_attributes_set(self):
        seq_data = {
            "high": {"preds": np.tile([10.0, 10.0, 40.0, 30.0], (30, 1)),
                     "gts":   np.tile([10.0, 10.0, 40.0, 30.0], (30, 1))},
            "low":  {"preds": np.tile([50.0, 50.0, 10.0, 10.0], (30, 1)),
                     "gts":   np.tile([10.0, 10.0, 40.0, 30.0], (30, 1))},
        }
        seq_attrs = {"high": {"IV"}, "low": {"FM"}}
        report = self.analyzer.analyze(seq_data, seq_attrs)
        assert report.best_attribute == "IV"
        assert report.worst_attribute == "FM"

    def test_sequences_missing_from_seq_data_are_skipped(self):
        seq_data = _make_seq_data(["seq1"])
        seq_attrs = {"seq1": {"FM"}, "missing_seq": {"OCC"}}
        report = self.analyzer.analyze(seq_data, seq_attrs)
        assert "FM" in report.per_attribute
        # OCC had no data so it should be absent
        assert "OCC" not in report.per_attribute

    def test_empty_seq_attrs_returns_empty_report(self):
        seq_data = _make_seq_data(["s1"])
        report = self.analyzer.analyze(seq_data, {})
        assert len(report.per_attribute) == 0
        assert report.best_attribute is None
        assert report.worst_attribute is None

    def test_coverage_counts_correctly(self):
        seq_data = _make_seq_data(["a", "b", "c"])
        seq_attrs = {"a": {"FM", "SV"}, "b": {"FM"}, "c": {"OCC"}}
        report = self.analyzer.analyze(seq_data, seq_attrs)
        # a counted twice (FM, SV), b once, c once → total = 4
        assert report.coverage == 4

    def test_to_markdown_returns_string(self):
        seq_data = _make_seq_data(["car1"])
        seq_attrs = {"car1": {"FM"}}
        report = self.analyzer.analyze(seq_data, seq_attrs, tracker_name="KCF")
        md = report.to_markdown()
        assert "KCF" in md
        assert "FM" in md

    def test_to_dict_serialisable(self):
        seq_data = _make_seq_data(["s1", "s2"])
        seq_attrs = {"s1": {"OCC"}, "s2": {"OCC", "FM"}}
        report = self.analyzer.analyze(seq_data, seq_attrs, tracker_name="CSRT")
        d = report.to_dict()
        assert d["tracker_name"] == "CSRT"
        assert "OCC" in d["per_attribute"]
        assert "mean_iou" in d["per_attribute"]["OCC"]


# ---------------------------------------------------------------------------
# AttributeAnalyzer.compare
# ---------------------------------------------------------------------------

class TestAttributeAnalyzerCompare:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def test_compare_returns_markdown_string(self):
        seq_attrs = {"s1": {"FM"}, "s2": {"OCC"}, "s3": {"FM", "OCC"}}
        tracker_data = {
            "MOSSE": _make_seq_data(["s1", "s2", "s3"], iou_level=0.5),
            "KCF":   _make_seq_data(["s1", "s2", "s3"], iou_level=0.7),
        }
        table = self.analyzer.compare(tracker_data, seq_attrs)
        assert "MOSSE" in table
        assert "KCF" in table
        assert "FM" in table
        assert "OCC" in table

    def test_compare_includes_all_attributes(self):
        seq_attrs = {"s1": {"FM"}, "s2": {"SV"}}
        tracker_data = {
            "T1": _make_seq_data(["s1", "s2"]),
            "T2": _make_seq_data(["s1", "s2"]),
        }
        table = self.analyzer.compare(tracker_data, seq_attrs)
        assert "FM" in table
        assert "SV" in table


# ---------------------------------------------------------------------------
# AttributeAnalyzer.rank_by_attribute
# ---------------------------------------------------------------------------

class TestRankByAttribute:
    def setup_method(self):
        self.analyzer = AttributeAnalyzer()

    def test_rank_returns_all_attributes(self):
        seq_attrs = {"s1": {"FM"}, "s2": {"OCC"}}
        tracker_data = {
            "T1": _make_seq_data(["s1", "s2"], iou_level=0.5),
            "T2": _make_seq_data(["s1", "s2"], iou_level=0.8),
        }
        ranking = self.analyzer.rank_by_attribute(tracker_data, seq_attrs)
        assert "FM" in ranking
        assert "OCC" in ranking

    def test_rank_ordered_best_to_worst(self):
        perfect_data = {
            "s1": {
                "preds": np.tile([10.0, 10.0, 40.0, 30.0], (40, 1)),
                "gts":   np.tile([10.0, 10.0, 40.0, 30.0], (40, 1)),
            }
        }
        bad_data = {
            "s1": {
                "preds": np.tile([100.0, 100.0, 5.0, 5.0], (40, 1)),
                "gts":   np.tile([10.0, 10.0, 40.0, 30.0], (40, 1)),
            }
        }
        seq_attrs = {"s1": {"FM"}}
        tracker_data = {"perfect": perfect_data, "bad": bad_data}
        ranking = self.analyzer.rank_by_attribute(tracker_data, seq_attrs)
        assert ranking["FM"][0] == "perfect"
        assert ranking["FM"][-1] == "bad"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            self.analyzer.rank_by_attribute({}, {}, metric="nonexistent")


# ---------------------------------------------------------------------------
# AttributeMetrics str representation
# ---------------------------------------------------------------------------

class TestAttributeMetricsStr:
    def test_str_contains_key_fields(self):
        m = AttributeMetrics(
            attribute="FM",
            num_sequences=5,
            mean_iou=0.45,
            success_auc=0.52,
            precision_auc=0.61,
        )
        s = str(m)
        assert "FM" in s
        assert "0.45" in s
        assert "0.52" in s
