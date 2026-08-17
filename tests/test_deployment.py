"""Tests for the deployment readiness checker."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.analysis.deployment import (
    DeploymentChecker,
    DeploymentProfile,
    DeploymentReport,
    MetricAssessment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    tracker_name: str = "TestTracker",
    dataset_name: str = "Synthetic",
    mean_iou: float = 0.6,
    mean_fps: float = 100.0,
    peak_memory_mb: float = 128.0,
    latency_mean_ms: float = 10.0,
    latency_p99_ms: float = 15.0,
    success_auc: float = 0.55,
    num_sequences: int = 3,
):
    """Build a minimal BenchmarkResult with controllable aggregate values."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult
    from eovot.metrics.accuracy import AccuracyMetrics

    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    for i in range(num_sequences):
        ious = np.full(10, mean_iou)
        profiling = ProfilingResult(
            tracker_name=tracker_name,
            frame_count=10,
            fps=mean_fps,
            latency_mean_ms=latency_mean_ms,
            latency_std_ms=1.0,
            latency_p95_ms=latency_p99_ms * 0.95,
            latency_p99_ms=latency_p99_ms,
            latency_cv=0.1,
            peak_memory_mb=peak_memory_mb,
        )
        accuracy = AccuracyMetrics(
            mean_iou=mean_iou,
            success_auc=success_auc,
            precision_auc=0.6,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq{i}",
                ious=ious,
                profiling=profiling,
                accuracy_metrics=accuracy,
            )
        )
    return result


# ---------------------------------------------------------------------------
# DeploymentProfile tests
# ---------------------------------------------------------------------------

class TestDeploymentProfile:
    def test_all_fields_none_by_default(self):
        p = DeploymentProfile()
        assert p.target_fps is None
        assert p.memory_budget_mb is None
        assert p.max_latency_mean_ms is None
        assert p.max_latency_p99_ms is None
        assert p.min_iou is None
        assert p.min_success_auc is None
        assert p.device_name == "unspecified"

    def test_to_dict_round_trip(self):
        p = DeploymentProfile(
            target_fps=30.0,
            memory_budget_mb=256.0,
            device_name="Jetson Nano",
        )
        d = p.to_dict()
        restored = DeploymentProfile.from_dict(d)
        assert restored.target_fps == pytest.approx(30.0)
        assert restored.memory_budget_mb == pytest.approx(256.0)
        assert restored.device_name == "Jetson Nano"
        assert restored.min_iou is None


# ---------------------------------------------------------------------------
# MetricAssessment tests
# ---------------------------------------------------------------------------

class TestMetricAssessment:
    def test_pass_higher_is_better(self):
        a = MetricAssessment(
            name="FPS", required=30.0, actual=60.0,
            passes=True, higher_is_better=True, headroom=1.0
        )
        assert "PASS" in str(a)
        assert "+" in str(a)

    def test_fail_lower_is_better(self):
        a = MetricAssessment(
            name="Memory", required=256.0, actual=512.0,
            passes=False, higher_is_better=False, headroom=-1.0
        )
        assert "FAIL" in str(a)

    def test_to_dict_keys(self):
        a = MetricAssessment(
            name="IoU", required=0.5, actual=0.6,
            passes=True, higher_is_better=True, headroom=0.2
        )
        d = a.to_dict()
        assert set(d.keys()) == {"name", "required", "actual", "passes",
                                   "higher_is_better", "headroom"}


# ---------------------------------------------------------------------------
# DeploymentChecker — GO verdicts
# ---------------------------------------------------------------------------

class TestDeploymentCheckerGo:
    def setup_method(self):
        self.checker = DeploymentChecker()
        self.result = _make_result(
            mean_fps=100.0, peak_memory_mb=128.0,
            latency_mean_ms=10.0, latency_p99_ms=15.0,
            mean_iou=0.6, success_auc=0.55,
        )

    def test_go_when_all_requirements_met(self):
        profile = DeploymentProfile(
            target_fps=30.0, memory_budget_mb=256.0,
            max_latency_mean_ms=33.3, max_latency_p99_ms=50.0,
            min_iou=0.5, min_success_auc=0.4,
        )
        report = self.checker.check(self.result, profile)
        assert report.deployable is True
        assert report.risk_score == pytest.approx(0.0)
        assert len(report.bottlenecks) == 0

    def test_empty_profile_no_assessments(self):
        profile = DeploymentProfile()
        report = self.checker.check(self.result, profile)
        assert report.deployable is True
        assert len(report.assessments) == 0

    def test_fps_only_go(self):
        profile = DeploymentProfile(target_fps=50.0)
        report = self.checker.check(self.result, profile)
        assert report.deployable is True
        assert len(report.assessments) == 1
        assert report.assessments[0].name == "Mean FPS"

    def test_memory_only_go(self):
        profile = DeploymentProfile(memory_budget_mb=512.0)
        report = self.checker.check(self.result, profile)
        assert report.deployable is True

    def test_iou_only_go(self):
        profile = DeploymentProfile(min_iou=0.4)
        report = self.checker.check(self.result, profile)
        assert report.deployable is True


# ---------------------------------------------------------------------------
# DeploymentChecker — NO-GO verdicts
# ---------------------------------------------------------------------------

class TestDeploymentCheckerNoGo:
    def setup_method(self):
        self.checker = DeploymentChecker()
        self.result = _make_result(
            mean_fps=20.0, peak_memory_mb=600.0,
            latency_mean_ms=60.0, latency_p99_ms=90.0,
            mean_iou=0.3, success_auc=0.25,
        )

    def test_nogo_fps_insufficient(self):
        profile = DeploymentProfile(target_fps=30.0)
        report = self.checker.check(self.result, profile)
        assert report.deployable is False
        assert len(report.bottlenecks) == 1
        assert report.bottlenecks[0].name == "Mean FPS"

    def test_nogo_memory_exceeded(self):
        profile = DeploymentProfile(memory_budget_mb=256.0)
        report = self.checker.check(self.result, profile)
        assert report.deployable is False
        assert report.bottlenecks[0].name == "Peak Memory (MB)"

    def test_nogo_latency_mean_exceeded(self):
        profile = DeploymentProfile(max_latency_mean_ms=33.3)
        report = self.checker.check(self.result, profile)
        assert report.deployable is False
        assert report.bottlenecks[0].name == "Mean Latency (ms)"

    def test_nogo_latency_p99_exceeded(self):
        profile = DeploymentProfile(max_latency_p99_ms=50.0)
        report = self.checker.check(self.result, profile)
        assert report.deployable is False
        assert report.bottlenecks[0].name == "p99 Latency (ms)"

    def test_nogo_iou_too_low(self):
        profile = DeploymentProfile(min_iou=0.5)
        report = self.checker.check(self.result, profile)
        assert report.deployable is False
        assert report.bottlenecks[0].name == "Mean IoU"

    def test_nogo_success_auc_too_low(self):
        profile = DeploymentProfile(min_success_auc=0.5)
        report = self.checker.check(self.result, profile)
        assert report.deployable is False
        assert report.bottlenecks[0].name == "Success AUC"

    def test_risk_score_positive_on_failure(self):
        profile = DeploymentProfile(target_fps=30.0, min_iou=0.5)
        report = self.checker.check(self.result, profile)
        assert report.risk_score > 0.0

    def test_risk_score_capped_at_one(self):
        # Tracker at zero FPS would give infinite headroom deficit;
        # risk score should still be ≤ 1.
        result = _make_result(mean_fps=0.001, mean_iou=0.01)
        profile = DeploymentProfile(target_fps=30.0, min_iou=0.9)
        report = self.checker.check(result, profile)
        assert report.risk_score <= 1.0


# ---------------------------------------------------------------------------
# Headroom computation correctness
# ---------------------------------------------------------------------------

class TestHeadroomComputation:
    def setup_method(self):
        self.checker = DeploymentChecker()

    def test_fps_headroom_when_at_double(self):
        result = _make_result(mean_fps=60.0)
        profile = DeploymentProfile(target_fps=30.0)
        report = self.checker.check(result, profile)
        a = report.assessments[0]
        assert a.headroom == pytest.approx(1.0)  # (60 - 30) / 30

    def test_memory_headroom_when_at_half(self):
        result = _make_result(peak_memory_mb=128.0)
        profile = DeploymentProfile(memory_budget_mb=256.0)
        report = self.checker.check(result, profile)
        a = report.assessments[0]
        assert a.headroom == pytest.approx(0.5)  # (256 - 128) / 256

    def test_latency_deficit_when_double(self):
        result = _make_result(latency_mean_ms=66.6)
        profile = DeploymentProfile(max_latency_mean_ms=33.3)
        report = self.checker.check(result, profile)
        a = report.assessments[0]
        # headroom = (33.3 - 66.6) / 33.3 ≈ -1.0
        assert a.headroom < 0
        assert a.passes is False

    def test_iou_exact_boundary_passes(self):
        result = _make_result(mean_iou=0.5)
        profile = DeploymentProfile(min_iou=0.5)
        report = self.checker.check(result, profile)
        assert report.assessments[0].passes is True
        assert report.assessments[0].headroom == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

class TestDeploymentReportRendering:
    def setup_method(self):
        self.checker = DeploymentChecker()
        self.result = _make_result(
            tracker_name="KCF", dataset_name="Synthetic",
            mean_fps=100.0, peak_memory_mb=128.0, mean_iou=0.6,
        )

    def test_str_contains_verdict(self):
        profile = DeploymentProfile(target_fps=30.0)
        report = self.checker.check(self.result, profile)
        s = str(report)
        assert "GO" in s or "NO-GO" in s
        assert "KCF" in s

    def test_markdown_contains_table(self):
        profile = DeploymentProfile(target_fps=30.0, min_iou=0.5)
        report = self.checker.check(self.result, profile)
        md = report.to_markdown()
        assert "| Metric |" in md
        assert "Mean FPS" in md
        assert "Mean IoU" in md

    def test_markdown_go_icon(self):
        profile = DeploymentProfile(target_fps=30.0)
        report = self.checker.check(self.result, profile)
        md = report.to_markdown()
        assert "✅ GO" in md

    def test_markdown_nogo_icon(self):
        result = _make_result(mean_fps=10.0)
        profile = DeploymentProfile(target_fps=30.0)
        report = self.checker.check(result, profile)
        md = report.to_markdown()
        assert "❌ NO-GO" in md
        assert "Bottlenecks" in md

    def test_device_name_in_markdown(self):
        profile = DeploymentProfile(device_name="Jetson Nano", target_fps=30.0)
        report = self.checker.check(self.result, profile)
        md = report.to_markdown()
        assert "Jetson Nano" in md

    def test_to_dict_structure(self):
        profile = DeploymentProfile(target_fps=30.0, min_iou=0.5)
        report = self.checker.check(self.result, profile)
        d = report.to_dict()
        assert "tracker_name" in d
        assert "deployable" in d
        assert "risk_score" in d
        assert "assessments" in d
        assert "bottlenecks" in d
        assert "profile" in d

    def test_save_and_reload(self):
        profile = DeploymentProfile(target_fps=30.0, memory_budget_mb=512.0)
        report = self.checker.check(self.result, profile)
        with tempfile.TemporaryDirectory() as tmp:
            path = report.save(f"{tmp}/report")
            assert path.exists()
            with open(path) as f:
                d = json.load(f)
        assert d["deployable"] == report.deployable
        assert d["risk_score"] == pytest.approx(report.risk_score, rel=1e-4)


# ---------------------------------------------------------------------------
# Integration with real BenchmarkEngine
# ---------------------------------------------------------------------------

class TestDeploymentCheckerIntegration:
    def test_full_pipeline(self):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        engine = BenchmarkEngine(verbose=False)
        ds = SyntheticDataset(num_sequences=2, num_frames=20)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")

        profile = DeploymentProfile(
            target_fps=5.0,          # easy to satisfy
            memory_budget_mb=2048.0, # easy to satisfy
            min_iou=0.0,             # always passes
        )
        checker = DeploymentChecker()
        report = checker.check(result, profile)

        assert isinstance(report, DeploymentReport)
        assert report.deployable is True
        assert report.risk_score == pytest.approx(0.0)
        assert len(report.assessments) == 3

    def test_package_export(self):
        from eovot.analysis import (
            DeploymentChecker,
            DeploymentProfile,
            DeploymentReport,
            MetricAssessment,
        )
        assert DeploymentChecker is not None
        assert DeploymentProfile is not None
        assert DeploymentReport is not None
        assert MetricAssessment is not None
