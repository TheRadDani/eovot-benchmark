"""Tests for eovot.analysis.warmup_analysis."""

from __future__ import annotations

import pytest
import numpy as np

from eovot.analysis.warmup_analysis import (
    WarmupAnalyzer,
    WarmupResult,
    WarmupReport,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker_name: str = "T",
    sequence_name: str = "seq",
    frame_count: int = 100,
    init_ms: float = 5.0,
    warmup_end: int = 10,
    cold_fps: float = 50.0,
    steady_fps: float = 100.0,
    fps_ratio: float = 2.0,
    warmup_ms: float = 200.0,
) -> WarmupResult:
    return WarmupResult(
        tracker_name=tracker_name,
        sequence_name=sequence_name,
        frame_count=frame_count,
        init_ms=init_ms,
        warmup_end_frame=warmup_end,
        cold_start_fps=cold_fps,
        steady_state_fps=steady_fps,
        fps_improvement_ratio=fps_ratio,
        total_warmup_ms=warmup_ms,
    )


def _small_dataset(num_sequences: int = 3, num_frames: int = 80) -> SyntheticDataset:
    return SyntheticDataset(num_sequences=num_sequences, num_frames=num_frames, seed=42)


# ---------------------------------------------------------------------------
# WarmupAnalyzer constructor validation
# ---------------------------------------------------------------------------

class TestWarmupAnalyzerInit:
    def test_default_construction(self):
        a = WarmupAnalyzer()
        assert a.ema_alpha == pytest.approx(0.1)
        assert a.tolerance == pytest.approx(0.1)
        assert a.min_steady_frames == 10

    def test_custom_params(self):
        a = WarmupAnalyzer(ema_alpha=0.3, tolerance=0.05, min_steady_frames=20)
        assert a.ema_alpha == pytest.approx(0.3)
        assert a.tolerance == pytest.approx(0.05)
        assert a.min_steady_frames == 20

    def test_invalid_ema_alpha_zero(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            WarmupAnalyzer(ema_alpha=0.0)

    def test_invalid_ema_alpha_too_large(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            WarmupAnalyzer(ema_alpha=1.1)

    def test_valid_ema_alpha_one(self):
        a = WarmupAnalyzer(ema_alpha=1.0)
        assert a.ema_alpha == pytest.approx(1.0)

    def test_invalid_tolerance_zero(self):
        with pytest.raises(ValueError, match="tolerance"):
            WarmupAnalyzer(tolerance=0.0)

    def test_invalid_tolerance_one(self):
        with pytest.raises(ValueError, match="tolerance"):
            WarmupAnalyzer(tolerance=1.0)

    def test_invalid_min_steady_frames_zero(self):
        with pytest.raises(ValueError, match="min_steady_frames"):
            WarmupAnalyzer(min_steady_frames=0)


# ---------------------------------------------------------------------------
# _detect_warmup with synthetic latency series
# ---------------------------------------------------------------------------

class TestDetectWarmup:
    def _run(self, latencies, **kwargs):
        a = WarmupAnalyzer(**kwargs)
        return a._detect_warmup(latencies)

    def test_flat_latency_steady_fps_correct(self):
        latencies = [10.0] * 100
        _, _, steady_fps, _ = self._run(latencies)
        assert steady_fps == pytest.approx(100.0, rel=0.01)

    def test_flat_latency_early_convergence(self):
        latencies = [10.0] * 100
        warmup_end, _, _, _ = self._run(latencies)
        assert warmup_end < 15

    def test_high_to_low_steady_fps_is_higher(self):
        slow = [20.0] * 20
        fast = [5.0] * 80
        latencies = slow + fast
        _, cold_fps, steady_fps, _ = self._run(latencies, min_steady_frames=5)
        assert steady_fps > cold_fps

    def test_high_to_low_steady_fps_near_fast(self):
        slow = [20.0] * 20
        fast = [5.0] * 80
        latencies = slow + fast
        _, _, steady_fps, _ = self._run(latencies, min_steady_frames=5)
        assert steady_fps == pytest.approx(200.0, rel=0.20)

    def test_warmup_wall_ms_non_negative(self):
        latencies = [15.0] * 50
        _, _, _, warmup_ms = self._run(latencies)
        assert warmup_ms >= 0.0

    def test_fps_values_positive(self):
        latencies = [8.0] * 60
        _, cold_fps, steady_fps, _ = self._run(latencies)
        assert cold_fps > 0
        assert steady_fps > 0

    def test_fps_improvement_significant_for_step_change(self):
        slow = [100.0] * 5
        fast = [10.0] * 95
        latencies = slow + fast
        a = WarmupAnalyzer(min_steady_frames=5)
        warmup_end, cold_fps, steady_fps, _ = a._detect_warmup(latencies)
        ratio = steady_fps / cold_fps if cold_fps > 0 else 1.0
        assert ratio > 3.0


# ---------------------------------------------------------------------------
# WarmupResult str and to_dict
# ---------------------------------------------------------------------------

class TestWarmupResult:
    def test_str_contains_tracker_name(self):
        r = _make_result(tracker_name="KCF")
        assert "KCF" in str(r)

    def test_str_contains_sequence_name(self):
        r = _make_result(sequence_name="seq_001")
        assert "seq_001" in str(r)

    def test_str_contains_init_ms(self):
        r = _make_result(init_ms=12.5)
        assert "12.5" in str(r)

    def test_str_contains_speedup(self):
        r = _make_result(fps_ratio=3.5)
        s = str(r)
        assert "3.5" in s or "3.50" in s

    def test_to_dict_has_all_keys(self):
        d = _make_result().to_dict()
        expected = {
            "tracker_name", "sequence_name", "frame_count",
            "init_ms", "warmup_end_frame", "cold_start_fps",
            "steady_state_fps", "fps_improvement_ratio", "total_warmup_ms",
        }
        assert set(d.keys()) == expected

    def test_to_dict_values(self):
        r = _make_result(tracker_name="MOSSE", init_ms=3.0, warmup_end=5)
        d = r.to_dict()
        assert d["tracker_name"] == "MOSSE"
        assert d["init_ms"] == pytest.approx(3.0)
        assert d["warmup_end_frame"] == 5


# ---------------------------------------------------------------------------
# analyze_sequence — integration with SyntheticDataset
# ---------------------------------------------------------------------------

class TestAnalyzeSequence:
    def test_returns_warmup_result(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert isinstance(result, WarmupResult)

    def test_frame_count_matches_sequence(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert result.frame_count == 50

    def test_init_ms_positive(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert result.init_ms > 0

    def test_steady_fps_positive(self):
        dataset = _small_dataset(num_sequences=1, num_frames=60)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert result.steady_state_fps > 0

    def test_cold_fps_positive(self):
        dataset = _small_dataset(num_sequences=1, num_frames=60)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert result.cold_start_fps > 0

    def test_latencies_length_is_n_minus_1(self):
        n_frames = 60
        dataset = _small_dataset(num_sequences=1, num_frames=n_frames)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert len(result.frame_latencies_ms) == n_frames - 1

    def test_max_frames_cap(self):
        dataset = _small_dataset(num_sequences=1, num_frames=100)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0], max_frames=30)
        assert result.frame_count == 30

    def test_sequence_name_is_string(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert isinstance(result.sequence_name, str)
        assert len(result.sequence_name) > 0

    def test_tracker_name_mosse(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert result.tracker_name == "MOSSE"

    def test_raises_on_too_short_sequence(self):
        dataset = _small_dataset(num_sequences=1, num_frames=5)
        analyzer = WarmupAnalyzer(min_steady_frames=10)
        with pytest.raises(ValueError):
            analyzer.analyze_sequence(MOSSETracker(), dataset[0])

    def test_kcf_tracker_works(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(KCFTracker(), dataset[0])
        assert result.tracker_name == "KCF"
        assert result.steady_state_fps > 0

    def test_fps_improvement_ratio_is_float(self):
        dataset = _small_dataset(num_sequences=1, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        result = analyzer.analyze_sequence(MOSSETracker(), dataset[0])
        assert isinstance(result.fps_improvement_ratio, float)
        assert result.fps_improvement_ratio > 0


# ---------------------------------------------------------------------------
# analyze_dataset
# ---------------------------------------------------------------------------

class TestAnalyzeDataset:
    def test_returns_warmup_report(self):
        dataset = _small_dataset(num_sequences=2, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        report = analyzer.analyze_dataset(MOSSETracker(), dataset)
        assert isinstance(report, WarmupReport)

    def test_correct_number_of_results(self):
        dataset = _small_dataset(num_sequences=3, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        report = analyzer.analyze_dataset(MOSSETracker(), dataset)
        assert len(report.results) == 3

    def test_max_sequences_cap(self):
        dataset = _small_dataset(num_sequences=5, num_frames=50)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        report = analyzer.analyze_dataset(MOSSETracker(), dataset, max_sequences=2)
        assert len(report.results) == 2

    def test_max_frames_per_seq_applied(self):
        dataset = _small_dataset(num_sequences=2, num_frames=100)
        analyzer = WarmupAnalyzer(min_steady_frames=5)
        report = analyzer.analyze_dataset(
            MOSSETracker(), dataset, max_frames_per_seq=30
        )
        for r in report.results:
            assert r.frame_count == 30

    def test_skips_too_short_sequences_gracefully(self):
        dataset = _small_dataset(num_sequences=3, num_frames=5)
        analyzer = WarmupAnalyzer(min_steady_frames=20)
        report = analyzer.analyze_dataset(MOSSETracker(), dataset)
        assert len(report.results) == 0


# ---------------------------------------------------------------------------
# WarmupReport
# ---------------------------------------------------------------------------

class TestWarmupReport:
    def _make_report(self) -> WarmupReport:
        return WarmupReport(results=[
            _make_result("KCF", "seq_0", init_ms=5.0, steady_fps=120.0, fps_ratio=2.0),
            _make_result("KCF", "seq_1", init_ms=3.0, steady_fps=80.0, fps_ratio=1.5),
        ])

    def test_tracker_name_from_first_result(self):
        assert self._make_report().tracker_name == "KCF"

    def test_mean_init_ms(self):
        assert self._make_report().mean_init_ms == pytest.approx(4.0)

    def test_mean_steady_state_fps(self):
        assert self._make_report().mean_steady_state_fps == pytest.approx(100.0)

    def test_mean_fps_improvement(self):
        assert self._make_report().mean_fps_improvement == pytest.approx(1.75)

    def test_to_markdown_contains_header(self):
        md = self._make_report().to_markdown()
        assert "Warm-up Analysis" in md
        assert "KCF" in md

    def test_to_markdown_contains_sequence_names(self):
        md = self._make_report().to_markdown()
        assert "seq_0" in md
        assert "seq_1" in md

    def test_to_markdown_contains_mean_row(self):
        assert "Mean" in self._make_report().to_markdown()

    def test_summary_dict_has_all_keys(self):
        d = self._make_report().summary_dict()
        expected = {
            "tracker_name", "num_sequences", "mean_init_ms",
            "mean_warmup_frames", "mean_steady_state_fps",
            "mean_fps_improvement_ratio", "sequences",
        }
        assert set(d.keys()) == expected

    def test_summary_dict_num_sequences(self):
        assert self._make_report().summary_dict()["num_sequences"] == 2

    def test_summary_dict_sequences_list_length(self):
        assert len(self._make_report().summary_dict()["sequences"]) == 2

    def test_mean_warmup_frames(self):
        r1 = _make_result(warmup_end=10)
        r2 = _make_result(warmup_end=20)
        report = WarmupReport(results=[r1, r2])
        assert report.mean_warmup_frames == pytest.approx(15.0)

    def test_empty_report_tracker_name(self):
        assert WarmupReport(results=[]).tracker_name == "unknown"

    def test_empty_report_mean_init_ms(self):
        assert WarmupReport(results=[]).mean_init_ms == pytest.approx(0.0)

    def test_empty_report_mean_fps_improvement(self):
        assert WarmupReport(results=[]).mean_fps_improvement == pytest.approx(1.0)

    def test_empty_report_to_markdown_runs(self):
        md = WarmupReport(results=[]).to_markdown()
        assert "Warm-up Analysis" in md
