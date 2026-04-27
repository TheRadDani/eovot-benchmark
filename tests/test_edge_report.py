"""Unit tests for EdgeDeploymentAnalyzer and EdgeDeploymentReport."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.profiling.hardware_profiles import (
    JETSON_NANO,
    LAPTOP_CPU,
    RASPBERRY_PI_4,
    HardwareProfile,
    get_profile,
)
from eovot.reporting.edge_report import (
    ConstraintScore,
    EdgeDeploymentAnalyzer,
    EdgeDeploymentReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_result(
    tracker_name: str = "MOSSE",
    dataset_name: str = "OTB100",
    mean_fps: float = 500.0,
    peak_memory_mb: float = 128.0,
    mean_energy_per_frame_mj: float | None = None,
) -> MagicMock:
    """Build a minimal mock of BenchmarkResult."""
    r = MagicMock()
    r.tracker_name = tracker_name
    r.dataset_name = dataset_name
    r.mean_fps = mean_fps
    r.peak_memory_mb = peak_memory_mb
    r.mean_energy_per_frame_mj = mean_energy_per_frame_mj
    return r


# ---------------------------------------------------------------------------
# ConstraintScore
# ---------------------------------------------------------------------------

def test_constraint_score_pass():
    s = ConstraintScore("FPS", required=30.0, measured=500.0, passed=True, margin_pct=1566.7)
    assert s.passed
    assert "PASS" in str(s)


def test_constraint_score_fail():
    s = ConstraintScore("FPS", required=30.0, measured=5.0, passed=False, margin_pct=-83.3)
    assert not s.passed
    assert "FAIL" in str(s)


# ---------------------------------------------------------------------------
# EdgeDeploymentAnalyzer.analyze — basic
# ---------------------------------------------------------------------------

def test_analyze_returns_report():
    result = _mock_result()
    analyzer = EdgeDeploymentAnalyzer()
    report = analyzer.analyze(result, JETSON_NANO)
    assert isinstance(report, EdgeDeploymentReport)


def test_analyze_tracker_and_dataset_names_propagated():
    result = _mock_result(tracker_name="KCF", dataset_name="GOT10k")
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.tracker_name == "KCF"
    assert report.dataset_name == "GOT10k"


def test_analyze_profile_stored():
    result = _mock_result()
    report = EdgeDeploymentAnalyzer().analyze(result, RASPBERRY_PI_4)
    assert report.profile is RASPBERRY_PI_4


# ---------------------------------------------------------------------------
# FPS constraint
# ---------------------------------------------------------------------------

def test_fps_passes_when_above_target(tmp_path):
    result = _mock_result(mean_fps=60.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)  # target = 30
    assert report.fps_score.passed
    assert report.fps_score.margin_pct > 0


def test_fps_fails_when_below_target():
    result = _mock_result(mean_fps=5.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)  # target = 30
    assert not report.fps_score.passed
    assert report.fps_score.margin_pct < 0


def test_fps_margin_exact_target():
    result = _mock_result(mean_fps=30.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.fps_score.passed
    assert report.fps_score.margin_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Memory constraint
# ---------------------------------------------------------------------------

def test_memory_passes_when_within_limit():
    result = _mock_result(peak_memory_mb=256.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)  # limit = 4096
    assert report.memory_score.passed
    assert report.memory_score.margin_pct > 0


def test_memory_fails_when_exceeds_limit():
    result = _mock_result(peak_memory_mb=8192.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)  # limit = 4096
    assert not report.memory_score.passed
    assert report.memory_score.margin_pct < 0


# ---------------------------------------------------------------------------
# Energy constraint
# ---------------------------------------------------------------------------

def test_energy_score_none_when_no_budget_set():
    result = _mock_result(mean_energy_per_frame_mj=0.1)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.energy_score is None


def test_energy_score_none_when_no_energy_data():
    result = _mock_result(mean_energy_per_frame_mj=None)
    report = EdgeDeploymentAnalyzer(energy_budget_mj_per_frame=1.0).analyze(
        result, JETSON_NANO
    )
    assert report.energy_score is None


def test_energy_passes_when_within_budget():
    result = _mock_result(mean_energy_per_frame_mj=0.1)
    report = EdgeDeploymentAnalyzer(energy_budget_mj_per_frame=1.0).analyze(
        result, JETSON_NANO
    )
    assert report.energy_score is not None
    assert report.energy_score.passed


def test_energy_fails_when_over_budget():
    result = _mock_result(mean_energy_per_frame_mj=2.0)
    report = EdgeDeploymentAnalyzer(energy_budget_mj_per_frame=1.0).analyze(
        result, JETSON_NANO
    )
    assert report.energy_score is not None
    assert not report.energy_score.passed


# ---------------------------------------------------------------------------
# is_deployable
# ---------------------------------------------------------------------------

def test_deployable_when_all_pass():
    result = _mock_result(mean_fps=500.0, peak_memory_mb=128.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.is_deployable


def test_not_deployable_when_fps_fails():
    result = _mock_result(mean_fps=1.0, peak_memory_mb=128.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert not report.is_deployable


def test_not_deployable_when_memory_fails():
    result = _mock_result(mean_fps=500.0, peak_memory_mb=999999.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert not report.is_deployable


# ---------------------------------------------------------------------------
# overall_grade
# ---------------------------------------------------------------------------

def test_grade_a_for_high_margin():
    result = _mock_result(mean_fps=10000.0, peak_memory_mb=1.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.overall_grade == "A"


def test_grade_f_for_multiple_failures():
    result = _mock_result(mean_fps=0.1, peak_memory_mb=999999.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.overall_grade == "F"


def test_grade_d_for_one_failure():
    result = _mock_result(mean_fps=0.1, peak_memory_mb=1.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.overall_grade == "D"


def test_grade_c_for_tight_pass():
    result = _mock_result(mean_fps=30.5, peak_memory_mb=4000.0)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert report.overall_grade == "C"


# ---------------------------------------------------------------------------
# summary dict
# ---------------------------------------------------------------------------

def test_summary_keys_present():
    result = _mock_result()
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    s = report.summary()
    for key in ("tracker", "dataset", "hardware_profile", "deployable", "grade",
                "fps_required", "fps_measured", "fps_margin_pct",
                "memory_limit_mb", "memory_measured_mb", "memory_margin_pct"):
        assert key in s, f"Missing key: {key}"


def test_summary_energy_keys_present_when_profiled():
    result = _mock_result(mean_energy_per_frame_mj=0.2)
    report = EdgeDeploymentAnalyzer(energy_budget_mj_per_frame=1.0).analyze(
        result, JETSON_NANO
    )
    s = report.summary()
    assert "energy_budget_mj_per_frame" in s
    assert "energy_measured_mj_per_frame" in s
    assert "energy_margin_pct" in s


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

def test_to_markdown_contains_tracker_name():
    result = _mock_result(tracker_name="MOSSE")
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    md = report.to_markdown()
    assert "MOSSE" in md


def test_to_markdown_contains_profile_name():
    result = _mock_result()
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    md = report.to_markdown()
    assert "Jetson Nano" in md


def test_to_markdown_contains_pass_fail():
    result = _mock_result(mean_fps=0.1)
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    md = report.to_markdown()
    assert "FAIL" in md


def test_to_markdown_is_string():
    result = _mock_result()
    report = EdgeDeploymentAnalyzer().analyze(result, JETSON_NANO)
    assert isinstance(report.to_markdown(), str)


# ---------------------------------------------------------------------------
# save to file
# ---------------------------------------------------------------------------

def test_save_writes_file(tmp_path):
    result = _mock_result()
    report = EdgeDeploymentAnalyzer().analyze(result, LAPTOP_CPU)
    out = str(tmp_path / "report.md")
    report.save(out)
    with open(out) as f:
        content = f.read()
    assert "MOSSE" in content
    assert "Laptop" in content


# ---------------------------------------------------------------------------
# compare + leaderboard
# ---------------------------------------------------------------------------

def test_compare_returns_sorted_list():
    results = [
        _mock_result("MOSSE", mean_fps=600.0, peak_memory_mb=50.0),
        _mock_result("KCF", mean_fps=200.0, peak_memory_mb=80.0),
        _mock_result("SlowTracker", mean_fps=5.0, peak_memory_mb=200.0),
    ]
    reports = EdgeDeploymentAnalyzer().compare(results, JETSON_NANO)
    assert len(reports) == 3
    grades = [r.overall_grade for r in reports]
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    assert all(
        grade_order[grades[i]] <= grade_order[grades[i + 1]]
        for i in range(len(grades) - 1)
    )


def test_leaderboard_is_string():
    results = [
        _mock_result("MOSSE", mean_fps=600.0, peak_memory_mb=50.0),
        _mock_result("KCF", mean_fps=10.0, peak_memory_mb=80.0),
    ]
    reports = EdgeDeploymentAnalyzer().compare(results, RASPBERRY_PI_4)
    lb = EdgeDeploymentAnalyzer.leaderboard(reports)
    assert isinstance(lb, str)
    assert "MOSSE" in lb
    assert "KCF" in lb


def test_leaderboard_contains_rank_column():
    results = [_mock_result("MOSSE")]
    reports = EdgeDeploymentAnalyzer().compare(results, LAPTOP_CPU)
    lb = EdgeDeploymentAnalyzer.leaderboard(reports)
    assert "Rank" in lb
