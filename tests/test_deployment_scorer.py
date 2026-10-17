"""Tests for DeploymentConstraints, DeploymentScore, and DeploymentSuitabilityScorer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.deployment.scorer import (
    DeploymentConstraints,
    DeploymentScore,
    DeploymentSuitabilityScorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_benchmark_result(
    tracker_name: str = "TestTracker",
    fps: float = 100.0,
    memory_mb: float = 50.0,
    mean_iou: float = 0.60,
    num_sequences: int = 3,
):
    """Construct a minimal BenchmarkResult without running the engine."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult

    def _prof(name, f=fps, m=memory_mb):
        return ProfilingResult(
            tracker_name=name,
            frame_count=100,
            fps=f,
            latency_mean_ms=1000.0 / f if f > 0 else 0.0,
            latency_std_ms=1.0,
            latency_p95_ms=1000.0 / f if f > 0 else 0.0,
            peak_memory_mb=m,
        )

    result = BenchmarkResult(tracker_name=tracker_name, dataset_name="TestDataset")
    for i in range(num_sequences):
        n = 10
        ious = np.full(n, mean_iou, dtype=np.float64)
        sr = SequenceResult(
            sequence_name=f"seq{i}",
            ious=ious,
            profiling=_prof(tracker_name),
        )
        result.sequence_results.append(sr)
    return result


def _default_scorer(device: str = "rpi4", min_fps: float = 5.0) -> DeploymentSuitabilityScorer:
    return DeploymentSuitabilityScorer(
        device=device,
        constraints=DeploymentConstraints(min_fps=min_fps, target_fps=30.0),
    )


# ===========================================================================
# TestDeploymentConstraints
# ===========================================================================

class TestDeploymentConstraints:
    def test_default_min_fps(self):
        c = DeploymentConstraints()
        assert c.min_fps == 10.0

    def test_effective_target_fps_default(self):
        c = DeploymentConstraints(min_fps=10.0)
        assert c.effective_target_fps() == 30.0

    def test_effective_target_fps_explicit(self):
        c = DeploymentConstraints(min_fps=10.0, target_fps=25.0)
        assert c.effective_target_fps() == 25.0

    def test_default_weights_keys(self):
        c = DeploymentConstraints()
        assert set(c.weights.keys()) == {"accuracy", "throughput", "memory", "energy"}

    def test_default_weights_positive(self):
        c = DeploymentConstraints()
        assert all(v > 0 for v in c.weights.values())

    def test_max_memory_defaults_to_none(self):
        c = DeploymentConstraints()
        assert c.max_memory_mb is None

    def test_custom_constraints(self):
        c = DeploymentConstraints(min_fps=20.0, target_fps=40.0, min_accuracy=0.3)
        assert c.min_fps == 20.0
        assert c.target_fps == 40.0
        assert c.min_accuracy == 0.3


# ===========================================================================
# TestDeploymentScore
# ===========================================================================

class TestDeploymentScore:
    def _sample(self, ready=True, total=0.75) -> DeploymentScore:
        return DeploymentScore(
            tracker_name="MOSSE",
            device_name="rpi4",
            total_score=total,
            deployment_ready=ready,
            disqualified=not ready,
            disqualify_reason="" if ready else "OOM: needs 5000 MB, limit is 3800 MB",
            accuracy_score=0.60,
            throughput_score=0.80,
            memory_score=0.90,
            energy_score=0.70,
            mean_accuracy=0.60,
            estimated_fps=20.0,
            estimated_memory_mb=100.0,
            estimated_energy_mj_per_frame=262.5,
            thermal_state="nominal",
        )

    def test_to_dict_has_required_keys(self):
        d = self._sample().to_dict()
        for key in ("tracker_name", "device_name", "total_score", "deployment_ready",
                    "disqualified", "accuracy_score", "throughput_score", "memory_score",
                    "energy_score", "estimated_fps", "thermal_state"):
            assert key in d

    def test_to_dict_total_score_rounded(self):
        d = self._sample(total=0.123456789).to_dict()
        assert d["total_score"] == round(0.123456789, 4)

    def test_str_contains_tracker_and_device(self):
        s = str(self._sample())
        assert "MOSSE" in s
        assert "rpi4" in s

    def test_str_ready_label(self):
        assert "READY" in str(self._sample(ready=True))

    def test_str_dq_label(self):
        assert "DQ" in str(self._sample(ready=False))


# ===========================================================================
# TestScorerDimensionMethods — unit-test static scoring methods
# ===========================================================================

class TestScorerDimensionMethods:
    def test_accuracy_score_clamps_above_one(self):
        s = DeploymentSuitabilityScorer._score_accuracy(1.5, 0.0)
        assert s == 1.0

    def test_accuracy_score_zero_below_min(self):
        s = DeploymentSuitabilityScorer._score_accuracy(0.2, 0.3)
        assert s == 0.0

    def test_accuracy_score_normal(self):
        s = DeploymentSuitabilityScorer._score_accuracy(0.7, 0.0)
        assert abs(s - 0.7) < 1e-9

    def test_throughput_zero_at_min_fps(self):
        s = DeploymentSuitabilityScorer._score_throughput(10.0, 10.0, 30.0)
        assert s == 0.0

    def test_throughput_zero_below_min_fps(self):
        s = DeploymentSuitabilityScorer._score_throughput(5.0, 10.0, 30.0)
        assert s == 0.0

    def test_throughput_one_at_target_fps(self):
        s = DeploymentSuitabilityScorer._score_throughput(30.0, 10.0, 30.0)
        assert abs(s - 1.0) < 1e-9

    def test_throughput_one_above_target_fps(self):
        s = DeploymentSuitabilityScorer._score_throughput(60.0, 10.0, 30.0)
        assert s == 1.0

    def test_throughput_half_at_midpoint(self):
        s = DeploymentSuitabilityScorer._score_throughput(20.0, 10.0, 30.0)
        assert abs(s - 0.5) < 1e-9

    def test_memory_score_zero_at_limit(self):
        s = DeploymentSuitabilityScorer._score_memory(1000.0, 1000.0)
        assert abs(s) < 1e-9

    def test_memory_score_one_when_empty(self):
        s = DeploymentSuitabilityScorer._score_memory(0.0, 1000.0)
        assert abs(s - 1.0) < 1e-9

    def test_memory_score_half_at_half(self):
        s = DeploymentSuitabilityScorer._score_memory(500.0, 1000.0)
        assert abs(s - 0.5) < 1e-9

    def test_energy_score_one_at_target_fps(self):
        # At target FPS, est_energy equals ref_energy → score = 1.0
        target_fps = 30.0
        tdp = 7.5
        util = 0.70  # DeviceSimulator.CPU_UTIL_FRACTION
        lat_ms = 1000.0 / target_fps
        est_energy = tdp * util * lat_ms
        s = DeploymentSuitabilityScorer._score_energy(est_energy, tdp, target_fps)
        assert abs(s - 1.0) < 1e-6

    def test_energy_score_decays_below_target_fps(self):
        # At 15 FPS (half of target 30), latency doubles → energy doubles → score = 0.5
        target_fps = 30.0
        tdp = 7.5
        util = 0.70
        lat_slow = 1000.0 / 15.0  # half the FPS
        est_energy_slow = tdp * util * lat_slow
        s = DeploymentSuitabilityScorer._score_energy(est_energy_slow, tdp, target_fps)
        assert abs(s - 0.5) < 1e-6


# ===========================================================================
# TestDeploymentSuitabilityScorerScore
# ===========================================================================

class TestDeploymentSuitabilityScorerScore:
    def test_score_returns_deployment_score(self):
        scorer = _default_scorer()
        result = _make_benchmark_result(fps=200.0, memory_mb=50.0)
        s = scorer.score(result)
        assert isinstance(s, DeploymentScore)

    def test_total_score_in_unit_interval(self):
        scorer = _default_scorer()
        result = _make_benchmark_result(fps=200.0, memory_mb=50.0)
        s = scorer.score(result)
        assert 0.0 <= s.total_score <= 1.0

    def test_deployment_ready_when_constraints_met(self):
        # rpi4 speed factor 0.12 → 200 host FPS * 0.12 = 24 device FPS > min_fps=5
        scorer = _default_scorer(min_fps=5.0)
        result = _make_benchmark_result(fps=200.0, memory_mb=50.0)
        s = scorer.score(result)
        assert s.deployment_ready
        assert not s.disqualified

    def test_disqualified_when_oom(self):
        # rpi4 limit is ~3800 MB; use 5000 MB tracker
        scorer = _default_scorer()
        result = _make_benchmark_result(fps=200.0, memory_mb=5000.0)
        s = scorer.score(result)
        assert s.disqualified
        assert not s.deployment_ready
        assert "OOM" in s.disqualify_reason or "MB" in s.disqualify_reason

    def test_disqualified_when_too_slow(self):
        # rpi4 speed factor 0.12 → 1 host FPS * 0.12 = 0.12 device FPS < min_fps=5
        scorer = _default_scorer(min_fps=5.0)
        result = _make_benchmark_result(fps=1.0, memory_mb=50.0)
        s = scorer.score(result)
        assert s.disqualified
        assert "FPS" in s.disqualify_reason or "fps" in s.disqualify_reason.lower()

    def test_custom_max_memory_overrides_device_limit(self):
        # rpi4 device limit ~3800 MB; set stricter 200 MB constraint
        constraints = DeploymentConstraints(min_fps=5.0, max_memory_mb=200.0)
        scorer = DeploymentSuitabilityScorer(device="rpi4", constraints=constraints)
        result = _make_benchmark_result(fps=200.0, memory_mb=300.0)  # 300 > 200 MB limit
        s = scorer.score(result)
        assert s.disqualified

    def test_tracker_name_propagated(self):
        scorer = _default_scorer()
        result = _make_benchmark_result(tracker_name="CSRT", fps=200.0)
        s = scorer.score(result)
        assert s.tracker_name == "CSRT"

    def test_device_name_propagated(self):
        scorer = _default_scorer(device="rpi4")
        result = _make_benchmark_result(fps=200.0)
        s = scorer.score(result)
        assert s.device_name == "rpi4"


# ===========================================================================
# TestRank
# ===========================================================================

class TestRank:
    def test_rank_returns_sorted_list(self):
        scorer = _default_scorer(min_fps=5.0)
        fast = _make_benchmark_result("Fast", fps=500.0, mean_iou=0.8, memory_mb=50.0)
        slow = _make_benchmark_result("Slow", fps=200.0, mean_iou=0.5, memory_mb=50.0)
        scores = scorer.rank([slow, fast])
        # fast should rank above slow
        assert scores[0].tracker_name == "Fast"

    def test_ready_trackers_before_disqualified(self):
        scorer = _default_scorer(min_fps=5.0)
        ready = _make_benchmark_result("ReadyTracker", fps=500.0, memory_mb=50.0)
        dq = _make_benchmark_result("OOMTracker", fps=500.0, memory_mb=5000.0)
        scores = scorer.rank([dq, ready])
        assert scores[0].tracker_name == "ReadyTracker"

    def test_rank_single_tracker(self):
        scorer = _default_scorer()
        result = _make_benchmark_result(fps=200.0)
        scores = scorer.rank([result])
        assert len(scores) == 1

    def test_rank_empty_list(self):
        scorer = _default_scorer()
        scores = scorer.rank([])
        assert scores == []


# ===========================================================================
# TestMarkdownTable
# ===========================================================================

class TestMarkdownTable:
    def test_header_present(self):
        scorer = _default_scorer()
        result = _make_benchmark_result(fps=200.0)
        scores = scorer.rank([result])
        table = scorer.to_markdown_table(scores)
        assert "Rank" in table
        assert "Tracker" in table
        assert "Score" in table

    def test_row_count_matches_input(self):
        scorer = _default_scorer(min_fps=5.0)
        results = [_make_benchmark_result(f"T{i}", fps=200.0 + i * 10) for i in range(4)]
        scores = scorer.rank(results)
        table = scorer.to_markdown_table(scores)
        data_rows = [l for l in table.splitlines() if l.startswith("|") and "---" not in l and "Rank" not in l]
        assert len(data_rows) == 4

    def test_dq_marker_in_table(self):
        scorer = _default_scorer(min_fps=5.0)
        dq = _make_benchmark_result("OOMTracker", fps=200.0, memory_mb=5000.0)
        scores = scorer.rank([dq])
        table = scorer.to_markdown_table(scores)
        assert "DQ" in table


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    def test_empty_benchmark_result_does_not_crash(self):
        from eovot.benchmark.engine import BenchmarkResult
        scorer = _default_scorer()
        result = BenchmarkResult(tracker_name="Empty", dataset_name="none")
        s = scorer.score(result)
        assert isinstance(s, DeploymentScore)
        assert s.disqualified  # 0 FPS < any min_fps

    def test_unknown_device_raises_key_error(self):
        scorer = DeploymentSuitabilityScorer(device="nonexistent_device_xyz")
        result = _make_benchmark_result(fps=200.0)
        with pytest.raises(KeyError):
            scorer.score(result)

    def test_all_builtin_devices_work(self):
        from eovot.profiling.device_sim import KNOWN_DEVICES
        result = _make_benchmark_result(fps=200.0, memory_mb=50.0)
        for device in KNOWN_DEVICES:
            scorer = DeploymentSuitabilityScorer(device=device)
            s = scorer.score(result)
            assert 0.0 <= s.total_score <= 1.0
