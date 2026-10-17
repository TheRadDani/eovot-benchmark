"""Tests for the edge constraint evaluation system (eovot/constraints/)."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.constraints.evaluator import ConstraintCheck, ConstraintEvaluator, ConstraintReport
from eovot.constraints.profiles import (
    EMBEDDED_MICRO,
    JETSON_NANO,
    LAPTOP_CPU,
    MOBILE_CLASS,
    PREDEFINED_PROFILES,
    RASPBERRY_PI_4,
    EdgeProfile,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profiling(fps: float = 50.0, peak_memory_mb: float = 128.0) -> ProfilingResult:
    latency_ms = 1000.0 / fps
    return ProfilingResult(
        tracker_name="test",
        frame_count=100,
        fps=fps,
        latency_mean_ms=latency_ms,
        latency_std_ms=latency_ms * 0.05,
        latency_p95_ms=latency_ms * 1.2,
        peak_memory_mb=peak_memory_mb,
    )


def _benchmark_result(
    fps: float = 50.0,
    peak_memory_mb: float = 128.0,
    tracker_name: str = "TestTracker",
) -> BenchmarkResult:
    seq = SequenceResult(
        sequence_name="seq_0",
        ious=np.full(50, 0.6),
        profiling=_profiling(fps=fps, peak_memory_mb=peak_memory_mb),
    )
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name="SyntheticDB")
    result.sequence_results.append(seq)
    return result


# ---------------------------------------------------------------------------
# EdgeProfile
# ---------------------------------------------------------------------------

class TestEdgeProfile:
    def test_all_predefined_keys_present(self):
        assert set(PREDEFINED_PROFILES) == {
            "raspberry_pi_4", "jetson_nano", "mobile", "embedded_micro", "laptop_cpu"
        }

    def test_raspberry_pi_4_attributes(self):
        p = RASPBERRY_PI_4
        assert p.min_fps == pytest.approx(10.0)
        assert p.max_memory_mb == pytest.approx(512.0)
        assert p.max_latency_ms == pytest.approx(100.0)
        assert p.max_energy_mj_per_frame is not None

    def test_laptop_cpu_has_no_energy_limit(self):
        assert LAPTOP_CPU.max_energy_mj_per_frame is None

    def test_custom_profile_created(self):
        p = EdgeProfile(name="FPGA", min_fps=30.0, max_memory_mb=128.0, max_latency_ms=33.0)
        assert p.name == "FPGA"
        assert p.max_energy_mj_per_frame is None

    def test_invalid_min_fps_raises(self):
        with pytest.raises(ValueError, match="min_fps"):
            EdgeProfile(name="Bad", min_fps=-1.0, max_memory_mb=100.0, max_latency_ms=50.0)

    def test_invalid_max_memory_raises(self):
        with pytest.raises(ValueError, match="max_memory_mb"):
            EdgeProfile(name="Bad", min_fps=10.0, max_memory_mb=0.0, max_latency_ms=50.0)

    def test_invalid_max_latency_raises(self):
        with pytest.raises(ValueError, match="max_latency_ms"):
            EdgeProfile(name="Bad", min_fps=10.0, max_memory_mb=100.0, max_latency_ms=0.0)

    def test_invalid_energy_raises(self):
        with pytest.raises(ValueError, match="max_energy_mj_per_frame"):
            EdgeProfile(
                name="Bad", min_fps=10.0, max_memory_mb=100.0,
                max_latency_ms=50.0, max_energy_mj_per_frame=-5.0
            )

    def test_predefined_profile_lookup(self):
        assert PREDEFINED_PROFILES["jetson_nano"] is JETSON_NANO
        assert PREDEFINED_PROFILES["mobile"] is MOBILE_CLASS


# ---------------------------------------------------------------------------
# ConstraintEvaluator — individual checks
# ---------------------------------------------------------------------------

class TestConstraintEvaluator:
    def setup_method(self):
        self.ev = ConstraintEvaluator()

    # FPS checks
    def test_fps_passes_when_fast_enough(self):
        result = _benchmark_result(fps=50.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)  # min_fps=10
        fps_check = next(c for c in report.checks if c.constraint == "min_fps")
        assert fps_check.passed
        assert fps_check.margin > 0

    def test_fps_fails_when_too_slow(self):
        result = _benchmark_result(fps=3.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)  # min_fps=10
        fps_check = next(c for c in report.checks if c.constraint == "min_fps")
        assert not fps_check.passed
        assert fps_check.margin < 0

    # Memory checks
    def test_memory_passes_when_within_budget(self):
        result = _benchmark_result(fps=50.0, peak_memory_mb=64.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)  # max_memory=512
        mem_check = next(c for c in report.checks if c.constraint == "max_memory_mb")
        assert mem_check.passed
        assert mem_check.margin > 0

    def test_memory_fails_when_exceeds_budget(self):
        result = _benchmark_result(fps=50.0, peak_memory_mb=1024.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)  # max_memory=512
        mem_check = next(c for c in report.checks if c.constraint == "max_memory_mb")
        assert not mem_check.passed
        assert mem_check.margin < 0

    # Latency checks
    def test_latency_passes_when_fps_is_high(self):
        # 100 FPS → 10 ms latency, well under RPi4's 100 ms limit
        result = _benchmark_result(fps=100.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        lat_check = next(c for c in report.checks if c.constraint == "max_latency_ms")
        assert lat_check.passed
        assert lat_check.measured == pytest.approx(10.0, rel=1e-6)

    def test_latency_fails_when_fps_is_very_low(self):
        # 1 FPS → 1000 ms latency, exceeds all profiles
        result = _benchmark_result(fps=1.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        lat_check = next(c for c in report.checks if c.constraint == "max_latency_ms")
        assert not lat_check.passed

    # Energy checks
    def test_energy_check_skipped_when_not_profiled(self):
        result = _benchmark_result(fps=50.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        assert not any(c.constraint == "max_energy_mj_per_frame" for c in report.checks)
        assert any("max_energy_mj_per_frame" in m for m in report.missing_data)

    def test_energy_check_skipped_when_profile_has_no_limit(self):
        result = _benchmark_result(fps=50.0)
        report = self.ev.evaluate(result, LAPTOP_CPU)
        assert not any(c.constraint == "max_energy_mj_per_frame" for c in report.checks)
        assert not report.missing_data  # no warning either

    # Overall pass/fail
    def test_overall_pass_requires_all_checks_pass(self):
        # Fast but way too much memory
        result = _benchmark_result(fps=500.0, peak_memory_mb=4096.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        assert not report.overall_pass

    def test_overall_pass_when_all_constraints_met(self):
        # 200 FPS, 64 MB — easily passes RPi4 FPS/memory/latency
        result = _benchmark_result(fps=200.0, peak_memory_mb=64.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        assert report.overall_pass

    def test_overall_fail_propagated_in_summary(self):
        result = _benchmark_result(fps=2.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        assert "NOT DEPLOYABLE" in report.summary()

    def test_overall_pass_propagated_in_summary(self):
        result = _benchmark_result(fps=200.0, peak_memory_mb=64.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        assert "DEPLOYABLE" in report.summary()
        assert "NOT DEPLOYABLE" not in report.summary()


# ---------------------------------------------------------------------------
# evaluate_many
# ---------------------------------------------------------------------------

class TestEvaluateMany:
    def setup_method(self):
        self.ev = ConstraintEvaluator()

    def test_returns_one_report_per_result(self):
        results = [
            _benchmark_result(fps=100.0, tracker_name="Fast"),
            _benchmark_result(fps=2.0, tracker_name="Slow"),
        ]
        reports = self.ev.evaluate_many(results, RASPBERRY_PI_4)
        assert len(reports) == 2

    def test_report_order_preserved(self):
        results = [
            _benchmark_result(tracker_name="A"),
            _benchmark_result(tracker_name="B"),
            _benchmark_result(tracker_name="C"),
        ]
        reports = self.ev.evaluate_many(results, JETSON_NANO)
        assert [r.tracker_name for r in reports] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestConstraintReportDict:
    def setup_method(self):
        self.ev = ConstraintEvaluator()

    def test_to_dict_keys(self):
        result = _benchmark_result()
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        d = report.to_dict()
        assert "tracker_name" in d
        assert "profile_name" in d
        assert "overall_pass" in d
        assert "checks" in d
        assert "missing_data" in d

    def test_checks_have_required_keys(self):
        result = _benchmark_result()
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        for check in report.to_dict()["checks"]:
            for key in ("constraint", "passed", "measured", "limit", "margin"):
                assert key in check

    def test_to_dict_overall_pass_matches_report(self):
        result = _benchmark_result(fps=200.0, peak_memory_mb=64.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        assert report.to_dict()["overall_pass"] == report.overall_pass


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def setup_method(self):
        self.ev = ConstraintEvaluator()

    def test_table_contains_profile_name(self):
        result = _benchmark_result()
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        table = self.ev.markdown_table([report], RASPBERRY_PI_4)
        assert "Raspberry Pi 4" in table

    def test_table_contains_tracker_names(self):
        results = [
            _benchmark_result(fps=200.0, tracker_name="FastTracker"),
            _benchmark_result(fps=2.0, tracker_name="SlowTracker"),
        ]
        reports = self.ev.evaluate_many(results, RASPBERRY_PI_4)
        table = self.ev.markdown_table(reports, RASPBERRY_PI_4)
        assert "FastTracker" in table
        assert "SlowTracker" in table

    def test_table_marks_deployable(self):
        result = _benchmark_result(fps=200.0, peak_memory_mb=64.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        table = self.ev.markdown_table([report], RASPBERRY_PI_4)
        assert "YES" in table

    def test_table_marks_not_deployable(self):
        result = _benchmark_result(fps=1.0)
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        table = self.ev.markdown_table([report], RASPBERRY_PI_4)
        assert "NO" in table

    def test_table_omits_energy_column_when_no_limit(self):
        result = _benchmark_result()
        report = self.ev.evaluate(result, LAPTOP_CPU)
        table = self.ev.markdown_table([report], LAPTOP_CPU)
        assert "Energy" not in table

    def test_table_includes_energy_column_when_limited(self):
        result = _benchmark_result()
        report = self.ev.evaluate(result, RASPBERRY_PI_4)
        table = self.ev.markdown_table([report], RASPBERRY_PI_4)
        assert "Energy" in table


# ---------------------------------------------------------------------------
# ConstraintCheck str representation
# ---------------------------------------------------------------------------

class TestConstraintCheckStr:
    def test_pass_str_contains_pass(self):
        c = ConstraintCheck("min_fps", True, 50.0, 10.0, 40.0)
        assert "PASS" in str(c)
        assert "min_fps" in str(c)

    def test_fail_str_contains_fail(self):
        c = ConstraintCheck("min_fps", False, 3.0, 10.0, -7.0)
        assert "FAIL" in str(c)
