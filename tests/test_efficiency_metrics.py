"""Tests for EfficiencyMetricsEngine, DeviceSimulator, and edge leaderboard integration."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from eovot.metrics.efficiency import EfficiencyEntry, EfficiencyMetricsEngine
from eovot.profiling.device_sim import (
    KNOWN_DEVICES,
    DeviceProfile,
    DeviceSimulator,
)
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profiling_result(tracker_name: str, fps: float, mem_mb: float) -> ProfilingResult:
    lat = 1000.0 / fps if fps > 0 else 1e6
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=100,
        fps=fps,
        latency_mean_ms=lat,
        latency_std_ms=lat * 0.05,
        latency_p95_ms=lat * 1.2,
        peak_memory_mb=mem_mb,
    )


# ---------------------------------------------------------------------------
# EfficiencyMetricsEngine — EES formula
# ---------------------------------------------------------------------------


class TestEdgeEfficiencyScore:
    def test_positive_ees_for_valid_inputs(self):
        engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)
        ees = engine.edge_efficiency_score(mean_iou=0.6, fps=100.0, peak_memory_mb=200.0)
        assert ees > 0.0

    def test_zero_fps_returns_zero(self):
        engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)
        assert engine.edge_efficiency_score(0.5, fps=0.0, peak_memory_mb=100.0) == 0.0

    def test_negative_iou_returns_zero(self):
        engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)
        assert engine.edge_efficiency_score(-0.1, fps=50.0, peak_memory_mb=100.0) == 0.0

    def test_ees_formula_correctness(self):
        engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)
        miou, fps, mem = 0.7, 50.0, 256.0
        expected = (miou * math.log1p(fps)) / (1.0 + mem / 512.0)
        assert abs(engine.edge_efficiency_score(miou, fps, mem) - expected) < 1e-9

    def test_higher_fps_higher_ees(self):
        engine = EfficiencyMetricsEngine()
        ees_slow = engine.edge_efficiency_score(0.6, fps=10.0, peak_memory_mb=100.0)
        ees_fast = engine.edge_efficiency_score(0.6, fps=200.0, peak_memory_mb=100.0)
        assert ees_fast > ees_slow

    def test_higher_memory_lower_ees(self):
        engine = EfficiencyMetricsEngine(memory_budget_mb=512.0)
        ees_small = engine.edge_efficiency_score(0.6, fps=100.0, peak_memory_mb=100.0)
        ees_large = engine.edge_efficiency_score(0.6, fps=100.0, peak_memory_mb=1000.0)
        assert ees_small > ees_large

    def test_diminishing_returns_on_fps(self):
        """Going 5→50 FPS should gain more EES than 500→550 FPS."""
        engine = EfficiencyMetricsEngine()
        gain_low = (
            engine.edge_efficiency_score(0.6, fps=50.0, peak_memory_mb=100.0)
            - engine.edge_efficiency_score(0.6, fps=5.0, peak_memory_mb=100.0)
        )
        gain_high = (
            engine.edge_efficiency_score(0.6, fps=550.0, peak_memory_mb=100.0)
            - engine.edge_efficiency_score(0.6, fps=500.0, peak_memory_mb=100.0)
        )
        assert gain_low > gain_high

    def test_invalid_memory_budget_raises(self):
        with pytest.raises(ValueError):
            EfficiencyMetricsEngine(memory_budget_mb=0.0)


# ---------------------------------------------------------------------------
# EfficiencyMetricsEngine — Pareto front
# ---------------------------------------------------------------------------


class TestParetoFront:
    def _make_entries(self, specs):
        return [
            EfficiencyEntry(
                tracker_name=name,
                dataset_name="test",
                mean_iou=iou,
                fps=fps,
                peak_memory_mb=100.0,
                ees=iou * math.log1p(fps),
            )
            for name, iou, fps in specs
        ]

    def test_pareto_front_single_entry(self):
        entries = self._make_entries([("A", 0.7, 100.0)])
        engine = EfficiencyMetricsEngine()
        engine.compute_pareto_front(entries)
        assert entries[0].on_pareto_front is True

    def test_dominated_entry_not_on_front(self):
        # B dominates A in both mIoU and EES
        entries = self._make_entries([
            ("A", 0.5, 50.0),
            ("B", 0.8, 200.0),
        ])
        engine = EfficiencyMetricsEngine()
        engine.compute_pareto_front(entries)
        a = next(e for e in entries if e.tracker_name == "A")
        b = next(e for e in entries if e.tracker_name == "B")
        assert b.on_pareto_front is True
        assert a.on_pareto_front is False

    def test_trade_off_both_on_front(self):
        # A has higher IoU, B has higher EES — neither dominates
        entries = self._make_entries([
            ("A", 0.9, 5.0),   # high IoU, low EES
            ("B", 0.4, 500.0), # low IoU, high EES
        ])
        engine = EfficiencyMetricsEngine()
        engine.compute_pareto_front(entries)
        assert all(e.on_pareto_front for e in entries)

    def test_identical_entries_both_on_front(self):
        entries = self._make_entries([("A", 0.7, 100.0), ("B", 0.7, 100.0)])
        engine = EfficiencyMetricsEngine()
        engine.compute_pareto_front(entries)
        assert all(e.on_pareto_front for e in entries)


# ---------------------------------------------------------------------------
# EfficiencyMetricsEngine — rank_trackers (integration with BenchmarkResult)
# ---------------------------------------------------------------------------


class TestRankTrackers:
    def test_rank_trackers_from_benchmark_results(self):
        """Smoke test: rank_trackers works end-to-end with real BenchmarkResult objects."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker
        from eovot.trackers.kcf import KCFTracker

        dataset = SyntheticDataset(num_sequences=2, num_frames=15, seed=0)
        engine = BenchmarkEngine(verbose=False)
        results = [
            engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
            engine.run(KCFTracker(), dataset, dataset_name="Synthetic"),
        ]
        eff_engine = EfficiencyMetricsEngine()
        ranking = eff_engine.rank_trackers(results)
        assert len(ranking) == 2
        # Sorted by EES descending
        assert ranking[0].ees >= ranking[1].ees

    def test_to_markdown_table_format(self):
        engine = EfficiencyMetricsEngine()
        entries = [
            EfficiencyEntry("MOSSE", "Synthetic", 0.7, 500.0, 50.0, ees=3.14, on_pareto_front=True),
            EfficiencyEntry("KCF", "Synthetic", 0.65, 200.0, 80.0, ees=2.80),
        ]
        table = engine.to_markdown_table(entries)
        assert "MOSSE" in table
        assert "KCF" in table
        assert "EES" in table
        assert "✓" in table


# ---------------------------------------------------------------------------
# DeviceSimulator — profiles
# ---------------------------------------------------------------------------


class TestDeviceSimulatorProfiles:
    def test_known_devices_available(self):
        sim = DeviceSimulator()
        for name in ["rpi4", "rpi5", "jetson_nano", "jetson_xnx", "coral_board", "snapdragon888"]:
            assert name in sim.list_devices()

    def test_register_custom_device(self):
        sim = DeviceSimulator()
        profile = DeviceProfile(
            name="custom", display_name="Custom SBC",
            cpu_speed_factor=0.15, memory_limit_mb=1024.0, tdp_watts=5.0
        )
        sim.register_device("custom", profile)
        assert "custom" in sim.list_devices()

    def test_unknown_device_raises(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        with pytest.raises(KeyError):
            sim.simulate(prof, "unknown_board")

    def test_invalid_calibration_factor_raises(self):
        with pytest.raises(ValueError):
            DeviceSimulator(host_calibration_factor=0.0)


# ---------------------------------------------------------------------------
# DeviceSimulator — simulation logic
# ---------------------------------------------------------------------------


class TestDeviceSimulation:
    def test_estimated_fps_lower_than_host(self):
        """All edge devices should have lower FPS than host (speed_factor < 1)."""
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        for device in sim.list_devices():
            result = sim.simulate(prof, device)
            assert result.estimated_fps < result.host_fps, f"{device} should be slower than host"

    def test_memory_matches_host(self):
        """Memory is algorithm-determined — should equal host memory."""
        sim = DeviceSimulator()
        prof = _make_profiling_result("KCF", fps=200.0, mem_mb=120.0)
        result = sim.simulate(prof, "rpi4")
        assert result.estimated_memory_mb == prof.peak_memory_mb

    def test_fits_in_memory_flag(self):
        sim = DeviceSimulator()
        # MOSSE typically uses <100 MB — should fit on all devices
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        result = sim.simulate(prof, "coral_board")  # 900 MB limit
        assert result.fits_in_memory is True

    def test_oom_flag_for_large_model(self):
        sim = DeviceSimulator()
        # 950 MB should exceed coral_board's 900 MB limit
        prof = _make_profiling_result("HeavyModel", fps=5.0, mem_mb=950.0)
        result = sim.simulate(prof, "coral_board")
        assert result.fits_in_memory is False

    def test_energy_per_frame_positive(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        result = sim.simulate(prof, "rpi4")
        assert result.estimated_energy_mj_per_frame > 0.0

    def test_thermal_state_nominal_at_zero_seconds(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        result = sim.simulate(prof, "rpi4", sustained_seconds=0.0)
        assert result.thermal_state == "nominal"

    def test_thermal_state_throttled_at_long_duration(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        # rpi4: onset=40s, ramp=25s → throttled after 65s
        result = sim.simulate(prof, "rpi4", sustained_seconds=200.0)
        assert result.thermal_state == "throttled"

    def test_throttled_fps_lower_than_nominal(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=500.0, mem_mb=50.0)
        nominal = sim.simulate(prof, "rpi4", sustained_seconds=0.0)
        throttled = sim.simulate(prof, "rpi4", sustained_seconds=200.0)
        assert throttled.estimated_fps < nominal.estimated_fps

    def test_simulate_all_returns_sorted_by_fps(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("KCF", fps=200.0, mem_mb=80.0)
        results = sim.simulate_all(prof)
        fps_vals = [r.estimated_fps for r in results]
        assert fps_vals == sorted(fps_vals, reverse=True)

    def test_simulate_all_device_subset(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=400.0, mem_mb=50.0)
        results = sim.simulate_all(prof, device_names=["rpi4", "jetson_nano"])
        assert len(results) == 2
        device_names = {r.device_name for r in results}
        assert device_names == {"rpi4", "jetson_nano"}

    def test_to_markdown_table_contains_device_names(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=400.0, mem_mb=50.0)
        results = sim.simulate_all(prof, device_names=["rpi4", "rpi5"])
        table = sim.to_markdown_table(results)
        assert "Raspberry Pi 4B" in table
        assert "Raspberry Pi 5" in table

    def test_to_summary_dict_serializable(self):
        sim = DeviceSimulator()
        prof = _make_profiling_result("MOSSE", fps=400.0, mem_mb=50.0)
        results = sim.simulate_all(prof, device_names=["rpi4"])
        summary = sim.to_summary_dict(results)
        # Should be JSON-serialisable
        json.dumps(summary)
        assert summary[0]["device"] == "rpi4"


# ---------------------------------------------------------------------------
# ExperimentRunner edge leaderboard integration
# ---------------------------------------------------------------------------


class TestEdgeLeaderboardIntegration:
    def test_edge_leaderboard_generated_when_configured(self):
        from eovot.experiment.runner import ExperimentRunner

        config = {
            "experiment": {"name": "test-edge-integration", "seed": 0},
            "dataset": {
                "loader": "SyntheticDataset",
                "name": "Synthetic",
                "num_sequences": 2,
                "num_frames": 20,
                "motion": "linear",
                "seed": 0,
            },
            "trackers": [
                {"name": "MOSSE", "params": {"learning_rate": 0.125}},
            ],
            "edge_profile": {
                "devices": ["rpi4", "jetson_nano"],
                "sustained_seconds": 0.0,
                "memory_budget_mb": 512.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_dir=tmpdir, verbose=False)
            output = runner.run_from_config(config)

        assert "edge_leaderboard" in output
        assert "rpi4" in output["edge_leaderboard"].lower() or "raspberry" in output["edge_leaderboard"].lower()

    def test_leaderboard_contains_ees_column(self):
        from eovot.experiment.runner import ExperimentRunner

        config = {
            "experiment": {"name": "test-ees-column", "seed": 0},
            "dataset": {
                "loader": "SyntheticDataset",
                "name": "Synthetic",
                "num_sequences": 2,
                "num_frames": 15,
                "motion": "linear",
                "seed": 0,
            },
            "trackers": [
                {"name": "MOSSE", "params": {}},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_dir=tmpdir, verbose=False)
            output = runner.run_from_config(config)

        assert "EES" in output["leaderboard"]

    def test_no_edge_leaderboard_without_config(self):
        from eovot.experiment.runner import ExperimentRunner

        config = {
            "experiment": {"name": "test-no-edge", "seed": 0},
            "dataset": {
                "loader": "SyntheticDataset",
                "name": "Synthetic",
                "num_sequences": 2,
                "num_frames": 15,
                "motion": "linear",
                "seed": 0,
            },
            "trackers": [{"name": "MOSSE", "params": {}}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_dir=tmpdir, verbose=False)
            output = runner.run_from_config(config)

        assert "edge_leaderboard" not in output

    def test_edge_projection_json_saved(self):
        from eovot.experiment.runner import ExperimentRunner

        config = {
            "experiment": {"name": "test-projection-json", "seed": 0},
            "dataset": {
                "loader": "SyntheticDataset",
                "name": "Synthetic",
                "num_sequences": 2,
                "num_frames": 15,
                "motion": "linear",
                "seed": 0,
            },
            "trackers": [{"name": "MOSSE", "params": {}}],
            "edge_profile": {
                "devices": ["rpi4"],
                "sustained_seconds": 0.0,
                "memory_budget_mb": 512.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(output_dir=tmpdir, verbose=False)
            runner.run_from_config(config)
            proj_path = Path(tmpdir) / "test-projection-json" / "edge_projection.json"
            assert proj_path.exists()
            data = json.loads(proj_path.read_text())
            assert "MOSSE" in data
