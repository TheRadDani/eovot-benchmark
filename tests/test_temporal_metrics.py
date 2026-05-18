"""Unit and integration tests for eovot.metrics.temporal.

Tests cover:
- Per-sequence analysis (analyze_sequence)
- Tracker-level aggregation (analyze_tracker)
- Comparative drift table (compare)
- Edge cases: empty arrays, single-frame, constant IoU, monotonic sequences
- Numerical properties: drift rate sign, half-life bounds, stability range
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.temporal import (
    TemporalDriftAnalyzer,
    TemporalDriftResult,
    TrackerDriftSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(**kwargs) -> TemporalDriftAnalyzer:
    return TemporalDriftAnalyzer(n_bins=10, **kwargs)


def _constant_ious(value: float, n: int = 50) -> np.ndarray:
    return np.full(n, value, dtype=np.float64)


def _linear_ious(start: float, end: float, n: int = 60) -> np.ndarray:
    return np.linspace(start, end, n)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_construction(self):
        a = TemporalDriftAnalyzer()
        assert a.n_bins == 20
        assert a.stability_threshold == 0.5
        assert a.burn_in_frac == pytest.approx(0.05)

    def test_custom_params(self):
        a = TemporalDriftAnalyzer(n_bins=10, stability_threshold=0.3, burn_in_frac=0.1)
        assert a.n_bins == 10
        assert a.stability_threshold == pytest.approx(0.3)

    def test_invalid_n_bins(self):
        with pytest.raises(ValueError):
            TemporalDriftAnalyzer(n_bins=1)

    def test_invalid_stability_threshold(self):
        with pytest.raises(ValueError):
            TemporalDriftAnalyzer(stability_threshold=1.5)

    def test_invalid_burn_in_frac(self):
        with pytest.raises(ValueError):
            TemporalDriftAnalyzer(burn_in_frac=-0.1)
        with pytest.raises(ValueError):
            TemporalDriftAnalyzer(burn_in_frac=1.0)


# ---------------------------------------------------------------------------
# analyze_sequence
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:
    def setup_method(self):
        self.analyzer = _make_analyzer()

    def test_returns_correct_type(self):
        result = self.analyzer.analyze_sequence(np.ones(30))
        assert isinstance(result, TemporalDriftResult)

    def test_perfect_tracker_no_drift(self):
        ious = _constant_ious(1.0, n=100)
        result = self.analyzer.analyze_sequence(ious, tracker_name="Perfect")
        assert result.drift_rate == pytest.approx(0.0, abs=1e-6)
        assert result.stability_index == pytest.approx(1.0)
        assert result.iou_drop == pytest.approx(0.0, abs=1e-6)

    def test_degrading_tracker_negative_drift(self):
        ious = _linear_ious(start=1.0, end=0.0, n=100)
        result = self.analyzer.analyze_sequence(ious)
        assert result.drift_rate < 0, "Decreasing IoU must yield negative drift rate"

    def test_improving_tracker_positive_drift(self):
        ious = _linear_ious(start=0.0, end=1.0, n=100)
        result = self.analyzer.analyze_sequence(ious)
        assert result.drift_rate > 0, "Increasing IoU must yield positive drift rate"

    def test_binned_iou_length_matches_n_bins(self):
        ious = np.random.default_rng(0).uniform(0, 1, 80)
        result = self.analyzer.analyze_sequence(ious)
        assert len(result.binned_iou) == self.analyzer.n_bins
        assert len(result.time_bins) == self.analyzer.n_bins

    def test_binned_iou_range(self):
        ious = np.random.default_rng(1).uniform(0, 1, 60)
        result = self.analyzer.analyze_sequence(ious)
        assert float(result.binned_iou.min()) >= 0.0
        assert float(result.binned_iou.max()) <= 1.0

    def test_time_bins_monotone_in_01(self):
        ious = np.random.default_rng(2).uniform(0, 1, 50)
        result = self.analyzer.analyze_sequence(ious)
        assert float(result.time_bins[0]) > 0.0
        assert float(result.time_bins[-1]) <= 1.0
        assert np.all(np.diff(result.time_bins) > 0), "time_bins must be strictly increasing"

    def test_stability_index_range(self):
        ious = np.random.default_rng(3).uniform(0, 1, 70)
        result = self.analyzer.analyze_sequence(ious)
        assert 0.0 <= result.stability_index <= 1.0

    def test_zero_iou_stability_is_zero(self):
        ious = _constant_ious(0.0, n=50)
        result = self.analyzer.analyze_sequence(ious)
        assert result.stability_index == pytest.approx(0.0)

    def test_high_iou_stability_is_one(self):
        ious = _constant_ious(0.9, n=50)
        analyzer = TemporalDriftAnalyzer(n_bins=10, stability_threshold=0.5)
        result = analyzer.analyze_sequence(ious)
        assert result.stability_index == pytest.approx(1.0)

    def test_iou_drop_positive_for_degrading_tracker(self):
        ious = _linear_ious(0.9, 0.1, n=80)
        result = self.analyzer.analyze_sequence(ious)
        assert result.iou_drop > 0, "initial_iou > final_iou → positive iou_drop"

    def test_n_frames_stored(self):
        ious = np.ones(77)
        result = self.analyzer.analyze_sequence(ious)
        assert result.n_frames == 77

    def test_tracker_and_sequence_names_stored(self):
        ious = np.ones(20)
        result = self.analyzer.analyze_sequence(ious, tracker_name="KCF", sequence_name="car1")
        assert result.tracker_name == "KCF"
        assert result.sequence_name == "car1"

    def test_iou_half_life_none_for_stable_tracker(self):
        ious = _constant_ious(0.9, n=100)
        result = self.analyzer.analyze_sequence(ious)
        assert result.iou_half_life is None

    def test_iou_half_life_detected_for_crashing_tracker(self):
        # IoU goes from 1.0 to 0.0 → half-life should be around 0.5
        ious = _linear_ious(1.0, 0.0, n=200)
        result = self.analyzer.analyze_sequence(ious)
        assert result.iou_half_life is not None
        assert 0.0 < result.iou_half_life < 1.0

    def test_str_representation(self):
        ious = np.ones(30) * 0.7
        result = self.analyzer.analyze_sequence(ious, tracker_name="X", sequence_name="Y")
        s = str(result)
        assert "X" in s and "Y" in s

    def test_to_dict_json_safe(self):
        import json
        ious = np.random.default_rng(9).uniform(0.2, 0.8, 40)
        result = self.analyzer.analyze_sequence(ious)
        d = result.to_dict()
        # Should not raise
        json.dumps(d)
        assert isinstance(d["binned_iou"], list)
        assert isinstance(d["time_bins"], list)


# ---------------------------------------------------------------------------
# analyze_tracker (aggregation)
# ---------------------------------------------------------------------------

class TestAnalyzeTracker:
    def setup_method(self):
        self.analyzer = _make_analyzer()

    def _make_seq_dict(self, n_seqs: int = 5, n_frames: int = 50) -> dict:
        rng = np.random.default_rng(42)
        return {f"seq_{i}": rng.uniform(0.3, 0.9, n_frames) for i in range(n_seqs)}

    def test_returns_tracker_drift_summary(self):
        seq_dict = self._make_seq_dict()
        result = self.analyzer.analyze_tracker(seq_dict, tracker_name="MOSSE")
        assert isinstance(result, TrackerDriftSummary)

    def test_correct_num_sequences(self):
        seq_dict = self._make_seq_dict(n_seqs=7)
        result = self.analyzer.analyze_tracker(seq_dict, tracker_name="T")
        assert result.num_sequences == 7

    def test_tracker_name_stored(self):
        seq_dict = self._make_seq_dict()
        result = self.analyzer.analyze_tracker(seq_dict, tracker_name="KCF")
        assert result.tracker_name == "KCF"

    def test_mean_stability_in_range(self):
        seq_dict = self._make_seq_dict()
        result = self.analyzer.analyze_tracker(seq_dict)
        assert 0.0 <= result.mean_stability_index <= 1.0

    def test_pct_degrading_in_range(self):
        seq_dict = self._make_seq_dict()
        result = self.analyzer.analyze_tracker(seq_dict)
        assert 0.0 <= result.pct_degrading <= 100.0

    def test_all_degrading_sequences(self):
        rng = np.random.default_rng(0)
        # All sequences strictly decrease
        seq_dict = {f"s{i}": np.linspace(0.9, 0.1, 60) for i in range(5)}
        result = self.analyzer.analyze_tracker(seq_dict)
        assert result.mean_drift_rate < 0
        assert result.pct_degrading == pytest.approx(100.0)

    def test_mean_binned_iou_shape(self):
        seq_dict = self._make_seq_dict(n_seqs=4)
        result = self.analyzer.analyze_tracker(seq_dict)
        assert result.mean_binned_iou.shape == (self.analyzer.n_bins,)

    def test_empty_dict_returns_empty_summary(self):
        result = self.analyzer.analyze_tracker({})
        assert result.num_sequences == 0
        assert result.mean_drift_rate == pytest.approx(0.0)

    def test_short_sequences_skipped(self):
        analyzer = TemporalDriftAnalyzer(n_bins=10, min_frames=20)
        seq_dict = {
            "long": np.ones(50) * 0.7,
            "short": np.ones(5) * 0.5,  # below min_frames=20, should be skipped
        }
        result = analyzer.analyze_tracker(seq_dict)
        assert result.num_sequences == 1

    def test_str_representation(self):
        seq_dict = self._make_seq_dict()
        result = self.analyzer.analyze_tracker(seq_dict, tracker_name="MIL")
        s = str(result)
        assert "MIL" in s


# ---------------------------------------------------------------------------
# compare (multi-tracker Markdown table)
# ---------------------------------------------------------------------------

class TestCompare:
    def setup_method(self):
        self.analyzer = _make_analyzer()

    def _make_tracker_ious(self) -> dict:
        rng = np.random.default_rng(77)
        return {
            "MOSSE": [rng.uniform(0.5, 0.9, 60) for _ in range(4)],
            "KCF": [rng.uniform(0.4, 0.85, 60) for _ in range(4)],
            "MIL": [np.linspace(0.8, 0.3, 60) for _ in range(4)],
        }

    def test_returns_markdown_string(self):
        table = self.analyzer.compare(self._make_tracker_ious())
        assert isinstance(table, str)
        assert "|" in table

    def test_table_contains_all_tracker_names(self):
        table = self.analyzer.compare(self._make_tracker_ious())
        assert "MOSSE" in table
        assert "KCF" in table
        assert "MIL" in table

    def test_most_degrading_tracker_ranked_last(self):
        tracker_ious = {
            "Stable": [np.ones(50) * 0.8 for _ in range(3)],
            "Crashing": [np.linspace(0.9, 0.0, 50) for _ in range(3)],
        }
        table = self.analyzer.compare(tracker_ious)
        stable_pos = table.find("Stable")
        crashing_pos = table.find("Crashing")
        assert stable_pos < crashing_pos, "Stable tracker should appear before crashing one"

    def test_single_tracker(self):
        table = self.analyzer.compare({"MOSSE": [np.ones(40) * 0.7]})
        assert "MOSSE" in table

    def test_empty_tracker_dict(self):
        table = self.analyzer.compare({})
        assert isinstance(table, str)


# ---------------------------------------------------------------------------
# Integration: end-to-end with synthetic IoU patterns
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_constant_vs_degrading_tracker_ranking(self):
        analyzer = TemporalDriftAnalyzer(n_bins=10)
        constant = [np.ones(60) * 0.7 for _ in range(5)]
        degrading = [np.linspace(0.9, 0.1, 60) for _ in range(5)]

        tracker_ious = {"Stable": constant, "Drifting": degrading}
        stable_sum = analyzer.analyze_tracker(
            {f"s{i}": v for i, v in enumerate(constant)}, tracker_name="Stable"
        )
        drifting_sum = analyzer.analyze_tracker(
            {f"s{i}": v for i, v in enumerate(degrading)}, tracker_name="Drifting"
        )

        assert stable_sum.mean_drift_rate > drifting_sum.mean_drift_rate
        assert stable_sum.mean_stability_index >= drifting_sum.mean_stability_index

    def test_drift_result_from_real_sequence_shape(self):
        """Simulate a 300-frame sequence as the engine would produce."""
        rng = np.random.default_rng(5)
        ious = np.clip(rng.normal(0.65, 0.1, 300), 0, 1)
        analyzer = TemporalDriftAnalyzer(n_bins=20)
        result = analyzer.analyze_sequence(ious, tracker_name="MOSSE", sequence_name="GOT-10k_val_001")
        assert result.n_frames == 300
        assert len(result.binned_iou) == 20
        assert result.stability_index >= 0.0
