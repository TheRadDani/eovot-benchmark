"""Tests for eovot.profiling.device_profiles."""
from __future__ import annotations

import pytest
import numpy as np

from eovot.profiling.device_profiles import (
    ALL_PROFILES,
    DESKTOP_SERVER,
    INTEL_NUC,
    JETSON_NANO,
    JETSON_ORIN_NANO,
    LAPTOP_MID,
    PROFILE_REGISTRY,
    RASPBERRY_PI_4B,
    Criterion,
    DeviceProfile,
    EdgeComplianceReport,
    assess_edge_compliance,
    compliance_matrix,
    get_profile,
)


# ---------------------------------------------------------------------------
# Minimal stubs — avoids importing the full benchmark engine in unit tests
# ---------------------------------------------------------------------------

class _ProfilingStub:
    def __init__(self, fps: float = 30.0, peak_memory_mb: float = 256.0):
        self.fps = fps
        self.peak_memory_mb = peak_memory_mb
        self.latency_mean_ms = 1000.0 / fps if fps > 0 else 0.0


class _SequenceStub:
    def __init__(self, fps: float = 30.0, peak_memory_mb: float = 256.0):
        self.ious = np.array([0.6])
        self.profiling = _ProfilingStub(fps, peak_memory_mb)
        self.energy = None


class _BenchmarkResultStub:
    def __init__(
        self,
        tracker_name: str = "test_tracker",
        fps: float = 30.0,
        peak_memory_mb: float = 256.0,
        energy_per_frame_mj: float | None = None,
        n_seq: int = 5,
    ):
        self.tracker_name = tracker_name
        self.dataset_name = "test"
        self.sequence_results = [
            _SequenceStub(fps=fps, peak_memory_mb=peak_memory_mb) for _ in range(n_seq)
        ]
        self._energy_per_frame_mj = energy_per_frame_mj

    @property
    def mean_fps(self) -> float:
        return np.mean([s.profiling.fps for s in self.sequence_results])

    @property
    def peak_memory_mb(self) -> float:
        return max(s.profiling.peak_memory_mb for s in self.sequence_results)

    @property
    def mean_energy_per_frame_mj(self) -> float | None:
        return self._energy_per_frame_mj


# ---------------------------------------------------------------------------
# Tests — DeviceProfile
# ---------------------------------------------------------------------------

class TestDeviceProfile:
    def test_all_profiles_count(self):
        assert len(ALL_PROFILES) == 6

    def test_registry_keys_match_names(self):
        for name, profile in PROFILE_REGISTRY.items():
            assert name == profile.name

    def test_all_profiles_positive_tdp(self):
        for p in ALL_PROFILES:
            assert p.tdp_watts > 0, f"{p.name} has non-positive TDP"

    def test_all_profiles_positive_target_fps(self):
        for p in ALL_PROFILES:
            assert p.target_fps > 0, f"{p.name} has non-positive target_fps"

    def test_raspberry_pi_4b_values(self):
        assert RASPBERRY_PI_4B.tdp_watts == pytest.approx(6.4)
        assert RASPBERRY_PI_4B.ram_mb == 4096
        assert RASPBERRY_PI_4B.target_fps == pytest.approx(15.0)
        assert RASPBERRY_PI_4B.max_energy_per_frame_mj is not None

    def test_jetson_nano_values(self):
        assert JETSON_NANO.tdp_watts == pytest.approx(10.0)
        assert JETSON_NANO.target_fps == pytest.approx(25.0)

    def test_desktop_no_energy_budget(self):
        assert DESKTOP_SERVER.max_energy_per_frame_mj is None

    def test_device_profile_str(self):
        s = str(RASPBERRY_PI_4B)
        assert "Raspberry Pi" in s
        assert "TDP=" in s

    def test_profiles_are_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            RASPBERRY_PI_4B.tdp_watts = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests — get_profile
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_exact_match(self):
        assert get_profile("Raspberry Pi 4B") is RASPBERRY_PI_4B

    def test_substring_match_lower(self):
        assert get_profile("jetson nano") is JETSON_NANO

    def test_substring_orin(self):
        assert get_profile("Orin") is JETSON_ORIN_NANO

    def test_unknown_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent device xyz")


# ---------------------------------------------------------------------------
# Tests — assess_edge_compliance
# ---------------------------------------------------------------------------

class TestEdgeCompliance:
    def _result(self, fps: float = 50.0, mem: float = 200.0, energy_mj: float | None = None):
        return _BenchmarkResultStub(fps=fps, peak_memory_mb=mem, energy_per_frame_mj=energy_mj)

    def test_compliant_tracker(self):
        result = self._result(fps=100.0, mem=100.0)
        report = assess_edge_compliance(result, JETSON_NANO)
        assert report.compliant is True
        assert report.n_passed == len(report.criteria)

    def test_fps_failure(self):
        result = self._result(fps=5.0, mem=100.0)
        report = assess_edge_compliance(result, JETSON_NANO)
        assert report.compliant is False
        fps_crit = next(c for c in report.criteria if "FPS" in c.name)
        assert not fps_crit.passed

    def test_memory_failure(self):
        result = self._result(fps=100.0, mem=9999.0)
        report = assess_edge_compliance(result, JETSON_NANO)
        assert report.compliant is False
        mem_crit = next(c for c in report.criteria if "Memory" in c.name)
        assert not mem_crit.passed

    def test_energy_criterion_included_when_available(self):
        result = self._result(fps=100.0, mem=100.0, energy_mj=2.0)
        report = assess_edge_compliance(result, RASPBERRY_PI_4B)
        names = [c.name for c in report.criteria]
        assert "Energy/Frame" in names

    def test_energy_criterion_absent_without_energy_data(self):
        result = self._result(fps=100.0, mem=100.0, energy_mj=None)
        report = assess_edge_compliance(result, RASPBERRY_PI_4B)
        names = [c.name for c in report.criteria]
        assert "Energy/Frame" not in names

    def test_energy_criterion_absent_when_device_unconstrained(self):
        result = self._result(fps=200.0, mem=100.0, energy_mj=50.0)
        report = assess_edge_compliance(result, DESKTOP_SERVER)
        names = [c.name for c in report.criteria]
        assert "Energy/Frame" not in names

    def test_energy_failure(self):
        result = self._result(fps=100.0, mem=100.0, energy_mj=100.0)
        report = assess_edge_compliance(result, RASPBERRY_PI_4B)
        e_crit = next(c for c in report.criteria if "Energy" in c.name)
        assert e_crit.passed is False

    def test_report_str_contains_status(self):
        result = self._result(fps=50.0, mem=200.0)
        report = assess_edge_compliance(result, RASPBERRY_PI_4B)
        s = str(report)
        assert "COMPLIANT" in s or "NON-COMPLIANT" in s

    def test_report_to_dict_structure(self):
        result = self._result(fps=50.0, mem=200.0)
        report = assess_edge_compliance(result, LAPTOP_MID)
        d = report.to_dict()
        assert d["tracker_name"] == result.tracker_name
        assert d["device"] == LAPTOP_MID.name
        assert isinstance(d["compliant"], bool)
        assert isinstance(d["criteria"], list)
        assert d["n_criteria"] == len(report.criteria)


# ---------------------------------------------------------------------------
# Tests — compliance_matrix
# ---------------------------------------------------------------------------

class TestComplianceMatrix:
    def test_shape(self):
        results = [
            _BenchmarkResultStub("fast", fps=100.0, peak_memory_mb=100.0),
            _BenchmarkResultStub("slow", fps=5.0, peak_memory_mb=900.0),
        ]
        devices = [RASPBERRY_PI_4B, JETSON_NANO]
        matrix = compliance_matrix(results, devices=devices)
        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix)

    def test_default_devices(self):
        results = [_BenchmarkResultStub("tracker")]
        matrix = compliance_matrix(results)
        assert len(matrix[0]) == len(ALL_PROFILES)

    def test_report_types(self):
        results = [_BenchmarkResultStub("tracker")]
        matrix = compliance_matrix(results, devices=[JETSON_NANO])
        assert isinstance(matrix[0][0], EdgeComplianceReport)


# ---------------------------------------------------------------------------
# Tests — Criterion
# ---------------------------------------------------------------------------

class TestCriterion:
    def test_pass_str(self):
        c = Criterion("Throughput (FPS)", passed=True, measured=30.0, threshold=25.0, unit="fps")
        assert "PASS" in str(c)

    def test_fail_str(self):
        c = Criterion("Peak Memory", passed=False, measured=900.0, threshold=512.0, unit="MB")
        assert "FAIL" in str(c)
