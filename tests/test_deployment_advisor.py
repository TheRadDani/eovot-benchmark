"""Tests for eovot.profiling.hardware_profiles and deployment_advisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from eovot.profiling.deployment_advisor import (
    DeploymentAdvisor,
    DeploymentScore,
    _fps_score,
    _latency_score,
    _memory_score,
    _power_score,
)
from eovot.profiling.hardware_profiles import (
    PROFILES,
    HardwareProfile,
    get_profile,
    list_profiles,
)


# ---------------------------------------------------------------------------
# Minimal BenchmarkResult stubs — mirrors the public API used by the advisor
# ---------------------------------------------------------------------------


@dataclass
class _FakeProfiling:
    fps: float = 30.0
    latency_mean_ms: float = 33.0
    peak_memory_mb: float = 100.0


@dataclass
class _FakeEnergy:
    mean_power_w: float = 5.0


@dataclass
class _FakeSeqResult:
    sequence_name: str = "seq1"
    profiling: _FakeProfiling = field(default_factory=_FakeProfiling)
    energy: Optional[_FakeEnergy] = None


@dataclass
class _FakeBenchmarkResult:
    tracker_name: str
    dataset_name: str = "OTB100"
    sequence_results: List[_FakeSeqResult] = field(default_factory=list)

    @property
    def mean_fps(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(sum(s.profiling.fps for s in self.sequence_results) / len(self.sequence_results))

    @property
    def peak_memory_mb(self) -> float:
        if not self.sequence_results:
            return 0.0
        return float(max(s.profiling.peak_memory_mb for s in self.sequence_results))


def _make_result(
    name: str,
    fps: float = 30.0,
    latency_ms: float = 33.0,
    memory_mb: float = 100.0,
    mean_power_w: Optional[float] = None,
    n_seqs: int = 3,
) -> _FakeBenchmarkResult:
    seqs = []
    for i in range(n_seqs):
        profiling = _FakeProfiling(fps=fps, latency_mean_ms=latency_ms, peak_memory_mb=memory_mb)
        energy = _FakeEnergy(mean_power_w=mean_power_w) if mean_power_w is not None else None
        seqs.append(_FakeSeqResult(sequence_name=f"seq{i}", profiling=profiling, energy=energy))
    return _FakeBenchmarkResult(tracker_name=name, sequence_results=seqs)


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    def test_all_built_in_profiles_exist(self) -> None:
        for key in ("raspberry_pi_4", "jetson_nano", "jetson_xavier_nx", "workstation", "laptop_cpu"):
            assert key in PROFILES

    def test_profile_fields_positive(self) -> None:
        for p in PROFILES.values():
            assert p.tdp_watts > 0
            assert p.memory_mb > 0
            assert p.target_fps > 0
            assert p.latency_budget_ms > 0
            assert p.power_budget_w > 0

    def test_get_profile_known(self) -> None:
        p = get_profile("jetson_nano")
        assert p.name == "jetson_nano"

    def test_get_profile_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            get_profile("nonexistent_device_xyz")

    def test_list_profiles_sorted_by_tdp(self) -> None:
        profiles = list_profiles()
        tdps = [p.tdp_watts for p in profiles]
        assert tdps == sorted(tdps)

    def test_invalid_tdp_raises(self) -> None:
        with pytest.raises(ValueError, match="tdp_watts"):
            HardwareProfile(
                name="bad", display_name="Bad", tdp_watts=-1,
                memory_mb=1024, target_fps=30, latency_budget_ms=33,
                power_budget_w=5,
            )

    def test_to_dict_has_all_keys(self) -> None:
        p = get_profile("raspberry_pi_4")
        d = p.to_dict()
        for key in ("name", "display_name", "tdp_watts", "memory_mb", "target_fps",
                    "latency_budget_ms", "power_budget_w", "description", "category"):
            assert key in d

    def test_category_field(self) -> None:
        assert get_profile("raspberry_pi_4").category == "ultra_low"
        assert get_profile("workstation").category == "workstation"


# ---------------------------------------------------------------------------
# Constraint score helpers
# ---------------------------------------------------------------------------


class TestConstraintScoreHelpers:
    def test_fps_score_exact_target(self) -> None:
        cs = _fps_score(30.0, 30.0)
        assert cs.score == pytest.approx(1.0)
        assert cs.passed

    def test_fps_score_below_target(self) -> None:
        cs = _fps_score(15.0, 30.0)
        assert cs.score == pytest.approx(0.5)
        assert not cs.passed

    def test_fps_score_exceeds_target_capped_at_1(self) -> None:
        cs = _fps_score(120.0, 30.0)
        assert cs.score == pytest.approx(1.0)
        assert cs.passed

    def test_latency_score_within_budget(self) -> None:
        cs = _latency_score(20.0, 33.0)
        assert cs.score == pytest.approx(min(33.0 / 20.0, 1.0))
        assert cs.passed

    def test_latency_score_exceeds_budget(self) -> None:
        cs = _latency_score(50.0, 33.0)
        assert not cs.passed
        assert cs.score < 1.0

    def test_memory_score_within_limit(self) -> None:
        cs = _memory_score(256.0, 1024.0)
        assert cs.score == pytest.approx(1.0)
        assert cs.passed

    def test_memory_score_exceeds_limit(self) -> None:
        cs = _memory_score(5000.0, 4096.0)
        assert not cs.passed

    def test_power_score_none_when_no_data(self) -> None:
        assert _power_score(None, 10.0) is None

    def test_power_score_within_budget(self) -> None:
        cs = _power_score(4.0, 10.0)
        assert cs is not None
        assert cs.passed
        assert cs.score == pytest.approx(1.0)

    def test_power_score_exceeds_budget(self) -> None:
        cs = _power_score(15.0, 10.0)
        assert cs is not None
        assert not cs.passed


# ---------------------------------------------------------------------------
# DeploymentAdvisor.score
# ---------------------------------------------------------------------------


class TestDeploymentAdvisorScore:
    def _advisor(self) -> DeploymentAdvisor:
        return DeploymentAdvisor(memory_safety_factor=0.80)

    def test_fast_low_mem_tracker_deployable_on_rpi4(self) -> None:
        advisor = self._advisor()
        profile = get_profile("raspberry_pi_4")
        result = _make_result("MOSSE", fps=50.0, latency_ms=20.0, memory_mb=50.0)
        ds = advisor.score(result, profile)
        assert ds.deployable
        assert ds.overall_score > 0.8

    def test_slow_tracker_not_deployable_on_rpi4(self) -> None:
        advisor = self._advisor()
        profile = get_profile("raspberry_pi_4")
        result = _make_result("SlowTracker", fps=5.0, latency_ms=200.0, memory_mb=50.0)
        ds = advisor.score(result, profile)
        assert not ds.deployable
        assert "fps" in ds.violations or "latency" in ds.violations

    def test_high_memory_tracker_not_deployable(self) -> None:
        advisor = self._advisor()
        profile = get_profile("raspberry_pi_4")
        # 80% of 4096 MB = 3276.8 MB; use 4000 MB to exceed that
        result = _make_result("HeavyTracker", fps=60.0, latency_ms=10.0, memory_mb=4000.0)
        ds = advisor.score(result, profile)
        assert not ds.deployable
        assert "memory" in ds.violations

    def test_energy_data_contributes_to_score(self) -> None:
        advisor = self._advisor()
        profile = get_profile("laptop_cpu")
        # One tracker with energy data within budget
        r_with_energy = _make_result("GreenTracker", fps=35.0, latency_ms=28.0,
                                      memory_mb=200.0, mean_power_w=8.0)
        ds = advisor.score(r_with_energy, profile)
        has_power = ds.constraint("power") is not None
        assert has_power

    def test_score_returns_deployment_score_type(self) -> None:
        advisor = self._advisor()
        result = _make_result("T1")
        ds = advisor.score(result, get_profile("workstation"))
        assert isinstance(ds, DeploymentScore)

    def test_to_dict_structure(self) -> None:
        advisor = self._advisor()
        ds = advisor.score(_make_result("T1"), get_profile("workstation"))
        d = ds.to_dict()
        for key in ("tracker_name", "profile_name", "overall_score",
                    "deployable", "violations", "recommendation", "constraints"):
            assert key in d

    def test_recommendation_non_empty(self) -> None:
        advisor = self._advisor()
        ds = advisor.score(_make_result("T1"), get_profile("jetson_nano"))
        assert len(ds.recommendation) > 0


# ---------------------------------------------------------------------------
# DeploymentAdvisor.rank
# ---------------------------------------------------------------------------


class TestDeploymentAdvisorRank:
    def test_deployable_trackers_ranked_first(self) -> None:
        advisor = DeploymentAdvisor()
        profile = get_profile("raspberry_pi_4")
        results = [
            _make_result("SlowTracker", fps=3.0, latency_ms=333.0, memory_mb=50.0),
            _make_result("FastTracker", fps=80.0, latency_ms=12.0, memory_mb=80.0),
        ]
        ranked = advisor.rank(results, profile)
        assert ranked[0].tracker_name == "FastTracker"
        assert ranked[0].deployable

    def test_rank_returns_all_trackers(self) -> None:
        advisor = DeploymentAdvisor()
        profile = get_profile("jetson_xavier_nx")
        results = [_make_result(f"T{i}", fps=10.0 * i + 1) for i in range(4)]
        ranked = advisor.rank(results, profile)
        assert len(ranked) == 4

    def test_multi_profile_summary_keys(self) -> None:
        advisor = DeploymentAdvisor()
        profiles = [get_profile("raspberry_pi_4"), get_profile("workstation")]
        results = [_make_result("T1"), _make_result("T2")]
        summary = advisor.multi_profile_summary(results, profiles)
        assert set(summary.keys()) == {"raspberry_pi_4", "workstation"}


# ---------------------------------------------------------------------------
# DeploymentAdvisor reporting
# ---------------------------------------------------------------------------


class TestDeploymentAdvisorReporting:
    def test_report_markdown_contains_tracker_name(self) -> None:
        advisor = DeploymentAdvisor()
        profile = get_profile("jetson_nano")
        results = [_make_result("CSRT", fps=35.0, latency_ms=28.0, memory_mb=150.0)]
        ranked = advisor.rank(results, profile)
        md = advisor.report_markdown(ranked, profile)
        assert "CSRT" in md
        assert "Jetson Nano" in md

    def test_report_multi_profile_markdown_contains_devices(self) -> None:
        advisor = DeploymentAdvisor()
        profiles = [get_profile("raspberry_pi_4"), get_profile("jetson_nano")]
        results = [_make_result("KCF"), _make_result("MOSSE")]
        summary = advisor.multi_profile_summary(results, profiles)
        md = advisor.report_multi_profile_markdown(summary, profiles)
        assert "Raspberry Pi 4" in md
        assert "Jetson Nano" in md
        assert "KCF" in md
        assert "MOSSE" in md

    def test_memory_safety_factor_applied(self) -> None:
        """A tracker using 85% of device RAM should fail with 80% safety factor
        but pass with a 90% safety factor."""
        profile = get_profile("raspberry_pi_4")  # 4096 MB
        target_mb = 4096 * 0.87  # 87% of RAM
        result = _make_result("BigTracker", fps=50.0, latency_ms=20.0, memory_mb=target_mb)

        advisor_strict = DeploymentAdvisor(memory_safety_factor=0.80)
        advisor_lenient = DeploymentAdvisor(memory_safety_factor=0.90)

        ds_strict = advisor_strict.score(result, profile)
        ds_lenient = advisor_lenient.score(result, profile)

        assert not ds_strict.deployable
        assert ds_lenient.deployable

    def test_invalid_memory_safety_factor(self) -> None:
        with pytest.raises(ValueError, match="memory_safety_factor"):
            DeploymentAdvisor(memory_safety_factor=1.5)
