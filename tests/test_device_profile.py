"""Unit tests for eovot.profiling.device_profile.

Tests cover:
- DeviceProfile construction and derived properties
- DEVICE_PRESETS registry completeness
- DeployabilityChecker constraint logic (pass/fail)
- DeployabilityReport structure and helpers
- check_all() multi-device sweep
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.profiling.device_profile import (
    DEVICE_PRESETS,
    ConstraintResult,
    DeployabilityChecker,
    DeployabilityReport,
    DeviceProfile,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight BenchmarkResult mock
# ---------------------------------------------------------------------------

def _make_profiling(fps: float, latency_ms: float, memory_mb: float):
    p = MagicMock()
    p.fps = fps
    p.latency_mean_ms = latency_ms
    p.peak_memory_mb = memory_mb
    return p


def _make_result(
    tracker_name: str = "TestTracker",
    fps: float = 100.0,
    latency_ms: float = 10.0,
    memory_mb: float = 100.0,
    energy_per_frame_mj: float = None,
    n_sequences: int = 3,
):
    """Build a minimal BenchmarkResult-like mock for checker tests."""
    result = MagicMock()
    result.tracker_name = tracker_name
    result.mean_fps = fps
    result.peak_memory_mb = memory_mb
    result.mean_energy_per_frame_mj = energy_per_frame_mj
    result.total_energy_j = (energy_per_frame_mj * 0.001 * n_sequences * 20) if energy_per_frame_mj else None

    seq = MagicMock()
    seq.profiling.latency_mean_ms = latency_ms
    seq.energy = None if energy_per_frame_mj is None else MagicMock()
    result.sequence_results = [seq] * n_sequences
    return result


# ---------------------------------------------------------------------------
# DeviceProfile
# ---------------------------------------------------------------------------

class TestDeviceProfile:
    def test_basic_construction(self):
        p = DeviceProfile(name="TestDevice", min_fps=30.0, max_memory_mb=512.0)
        assert p.name == "TestDevice"
        assert p.min_fps == 30.0
        assert p.max_memory_mb == 512.0

    def test_derived_max_latency_from_fps(self):
        p = DeviceProfile(name="X", min_fps=25.0)
        assert p.derived_max_latency_ms == pytest.approx(40.0)  # 1000/25

    def test_derived_max_latency_explicit_override(self):
        p = DeviceProfile(name="X", min_fps=25.0, max_latency_ms=50.0)
        assert p.derived_max_latency_ms == pytest.approx(50.0)

    def test_derived_max_latency_none_when_no_fps(self):
        p = DeviceProfile(name="X")
        assert p.derived_max_latency_ms is None

    def test_str_representation(self):
        p = DeviceProfile(name="RPi4", min_fps=15.0, tdp_watts=6.0)
        s = str(p)
        assert "RPi4" in s
        assert "min_fps=15.0" in s

    def test_defaults_are_none(self):
        p = DeviceProfile(name="Empty")
        assert p.min_fps is None
        assert p.max_memory_mb is None
        assert p.tdp_watts is None
        assert p.max_energy_per_frame_mj is None


# ---------------------------------------------------------------------------
# DEVICE_PRESETS registry
# ---------------------------------------------------------------------------

class TestDevicePresets:
    def test_expected_keys_present(self):
        expected = {"raspberry_pi4", "jetson_nano", "jetson_orin_nano", "laptop_cpu", "desktop_cpu", "coral_usb"}
        assert expected.issubset(set(DEVICE_PRESETS.keys()))

    def test_all_presets_are_device_profiles(self):
        for key, profile in DEVICE_PRESETS.items():
            assert isinstance(profile, DeviceProfile), f"{key} is not a DeviceProfile"

    def test_rpi4_has_tight_constraints(self):
        rpi4 = DEVICE_PRESETS["raspberry_pi4"]
        assert rpi4.min_fps <= 20.0
        assert rpi4.max_memory_mb <= 512.0
        assert rpi4.tdp_watts <= 10.0

    def test_desktop_more_relaxed_than_rpi4(self):
        desktop = DEVICE_PRESETS["desktop_cpu"]
        rpi4 = DEVICE_PRESETS["raspberry_pi4"]
        assert desktop.min_fps > rpi4.min_fps
        assert desktop.max_memory_mb > rpi4.max_memory_mb

    def test_all_presets_have_name(self):
        for profile in DEVICE_PRESETS.values():
            assert profile.name, "every preset must have a non-empty name"


# ---------------------------------------------------------------------------
# DeployabilityChecker — FPS constraint
# ---------------------------------------------------------------------------

class TestFPSConstraint:
    def setup_method(self):
        self.checker = DeployabilityChecker()
        self.device = DeviceProfile(name="TestDev", min_fps=30.0)

    def test_fps_pass(self):
        result = _make_result(fps=100.0)
        report = self.checker.check(result, self.device)
        fps_check = next(c for c in report.constraint_results if c.name == "Mean FPS")
        assert fps_check.passed

    def test_fps_fail(self):
        result = _make_result(fps=10.0)
        report = self.checker.check(result, self.device)
        fps_check = next(c for c in report.constraint_results if c.name == "Mean FPS")
        assert not fps_check.passed

    def test_fps_exactly_at_limit_passes(self):
        result = _make_result(fps=30.0)
        report = self.checker.check(result, self.device)
        fps_check = next(c for c in report.constraint_results if c.name == "Mean FPS")
        assert fps_check.passed

    def test_no_fps_constraint_skips_check(self):
        device = DeviceProfile(name="NoFPS")
        result = _make_result(fps=5.0)
        report = self.checker.check(result, device)
        fps_checks = [c for c in report.constraint_results if c.name == "Mean FPS"]
        assert len(fps_checks) == 0


# ---------------------------------------------------------------------------
# DeployabilityChecker — memory constraint
# ---------------------------------------------------------------------------

class TestMemoryConstraint:
    def setup_method(self):
        self.checker = DeployabilityChecker()
        self.device = DeviceProfile(name="MemDev", max_memory_mb=256.0)

    def test_memory_pass(self):
        result = _make_result(memory_mb=100.0)
        report = self.checker.check(result, self.device)
        mem_check = next(c for c in report.constraint_results if c.name == "Peak Memory")
        assert mem_check.passed

    def test_memory_fail(self):
        result = _make_result(memory_mb=512.0)
        report = self.checker.check(result, self.device)
        mem_check = next(c for c in report.constraint_results if c.name == "Peak Memory")
        assert not mem_check.passed

    def test_no_memory_constraint_skips(self):
        device = DeviceProfile(name="NoMem")
        result = _make_result(memory_mb=99999.0)
        report = self.checker.check(result, device)
        mem_checks = [c for c in report.constraint_results if c.name == "Peak Memory"]
        assert len(mem_checks) == 0


# ---------------------------------------------------------------------------
# DeployabilityChecker — latency constraint
# ---------------------------------------------------------------------------

class TestLatencyConstraint:
    def setup_method(self):
        self.checker = DeployabilityChecker()

    def test_latency_derived_from_fps(self):
        device = DeviceProfile(name="LatDev", min_fps=25.0)
        result = _make_result(fps=25.0, latency_ms=30.0)
        report = self.checker.check(result, device)
        lat_check = next(c for c in report.constraint_results if c.name == "Mean Latency")
        assert lat_check.limit == pytest.approx(40.0)  # 1000/25

    def test_latency_pass(self):
        device = DeviceProfile(name="LatDev", min_fps=25.0)
        result = _make_result(fps=25.0, latency_ms=10.0)
        report = self.checker.check(result, device)
        lat_check = next(c for c in report.constraint_results if c.name == "Mean Latency")
        assert lat_check.passed

    def test_latency_fail(self):
        device = DeviceProfile(name="LatDev", min_fps=25.0)
        result = _make_result(fps=10.0, latency_ms=100.0)
        report = self.checker.check(result, device)
        lat_check = next(c for c in report.constraint_results if c.name == "Mean Latency")
        assert not lat_check.passed


# ---------------------------------------------------------------------------
# DeployabilityChecker — energy constraint
# ---------------------------------------------------------------------------

class TestEnergyConstraint:
    def setup_method(self):
        self.checker = DeployabilityChecker()
        self.device = DeviceProfile(name="EnergyDev", max_energy_per_frame_mj=10.0)

    def test_energy_pass(self):
        result = _make_result(energy_per_frame_mj=5.0)
        report = self.checker.check(result, self.device)
        e_checks = [c for c in report.constraint_results if c.name == "Energy per Frame"]
        assert len(e_checks) == 1
        assert e_checks[0].passed

    def test_energy_fail(self):
        result = _make_result(energy_per_frame_mj=20.0)
        report = self.checker.check(result, self.device)
        e_checks = [c for c in report.constraint_results if c.name == "Energy per Frame"]
        assert len(e_checks) == 1
        assert not e_checks[0].passed

    def test_energy_limit_set_but_no_data_adds_note(self):
        result = _make_result(energy_per_frame_mj=None)  # no energy data
        report = self.checker.check(result, self.device)
        e_checks = [c for c in report.constraint_results if c.name == "Energy per Frame"]
        assert len(e_checks) == 0
        assert any("energy" in n.lower() for n in report.notes)


# ---------------------------------------------------------------------------
# DeployabilityReport structure
# ---------------------------------------------------------------------------

class TestDeployabilityReport:
    def test_deployable_true_when_all_pass(self):
        result = _make_result(fps=200.0, memory_mb=50.0)
        device = DEVICE_PRESETS["raspberry_pi4"]
        report = DeployabilityChecker().check(result, device)
        # High FPS and low memory → deployable
        assert report.deployable

    def test_deployable_false_when_fps_fails(self):
        result = _make_result(fps=1.0, memory_mb=50.0)
        device = DEVICE_PRESETS["raspberry_pi4"]
        report = DeployabilityChecker().check(result, device)
        assert not report.deployable

    def test_violations_list_on_failure(self):
        result = _make_result(fps=1.0, memory_mb=9999.0)
        device = DEVICE_PRESETS["raspberry_pi4"]
        report = DeployabilityChecker().check(result, device)
        assert len(report.violations) >= 2  # FPS + memory

    def test_violations_empty_when_deployable(self):
        result = _make_result(fps=500.0, memory_mb=10.0)
        device = DEVICE_PRESETS["laptop_cpu"]
        report = DeployabilityChecker().check(result, device)
        assert report.violations == []

    def test_summary_dict_keys(self):
        result = _make_result(fps=100.0, memory_mb=100.0)
        device = DEVICE_PRESETS["laptop_cpu"]
        report = DeployabilityChecker().check(result, device)
        d = report.summary_dict()
        assert "tracker" in d
        assert "device" in d
        assert "deployable" in d
        assert "constraints" in d
        assert isinstance(d["constraints"], list)

    def test_str_contains_verdict(self):
        result = _make_result(fps=500.0, memory_mb=10.0)
        device = DEVICE_PRESETS["laptop_cpu"]
        report = DeployabilityChecker().check(result, device)
        s = str(report)
        assert "DEPLOYABLE" in s or "NOT DEPLOYABLE" in s

    def test_notes_are_populated(self):
        result = _make_result(fps=500.0, memory_mb=10.0)
        device = DEVICE_PRESETS["laptop_cpu"]
        report = DeployabilityChecker().check(result, device)
        assert len(report.notes) > 0


# ---------------------------------------------------------------------------
# check_all() multi-device sweep
# ---------------------------------------------------------------------------

class TestCheckAll:
    def test_returns_dict_for_all_presets(self):
        result = _make_result(fps=200.0, memory_mb=100.0)
        reports = DeployabilityChecker().check_all(result)
        assert set(reports.keys()) == set(DEVICE_PRESETS.keys())

    def test_all_values_are_reports(self):
        result = _make_result(fps=200.0, memory_mb=100.0)
        reports = DeployabilityChecker().check_all(result)
        for key, rpt in reports.items():
            assert isinstance(rpt, DeployabilityReport), f"{key} did not return a report"

    def test_custom_device_dict(self):
        custom = {"dev_a": DeviceProfile(name="A", min_fps=10.0)}
        result = _make_result(fps=100.0)
        reports = DeployabilityChecker().check_all(result, devices=custom)
        assert "dev_a" in reports
        assert reports["dev_a"].deployable

    def test_fast_tracker_deployable_on_all_devices(self):
        result = _make_result(fps=1000.0, memory_mb=10.0)
        reports = DeployabilityChecker().check_all(result)
        for key, rpt in reports.items():
            fps_checks = [c for c in rpt.constraint_results if c.name == "Mean FPS"]
            mem_checks = [c for c in rpt.constraint_results if c.name == "Peak Memory"]
            for c in fps_checks + mem_checks:
                assert c.passed, f"Fast/tiny tracker should pass {c.name} on {key}"
