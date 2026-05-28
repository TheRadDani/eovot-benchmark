"""Tests for the edge deployment analysis report module."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.reporting.deployment_report import (
    DeploymentReport,
    DeploymentReportEngine,
    DeviceTrackerEntry,
    _aggregate_profiling,
    _device_ees,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _mock_profiling(
    tracker_name: str = "TestTracker",
    latency_ms: float = 5.0,
    fps: float = 200.0,
    peak_mem: float = 100.0,
    frame_count: int = 99,
) -> MagicMock:
    p = MagicMock()
    p.tracker_name = tracker_name
    p.latency_mean_ms = latency_ms
    p.latency_std_ms = 0.5
    p.latency_p95_ms = latency_ms + 1.0
    p.fps = fps
    p.peak_memory_mb = peak_mem
    p.frame_count = frame_count
    return p


def _mock_sequence_result(
    tracker_name: str = "T",
    latency_ms: float = 5.0,
    peak_mem: float = 100.0,
) -> MagicMock:
    sr = MagicMock()
    sr.profiling = _mock_profiling(tracker_name, latency_ms, 1000.0 / latency_ms, peak_mem)
    return sr


def _mock_benchmark_result(
    tracker_name: str = "TestTracker",
    mean_iou: float = 0.65,
    mean_fps: float = 200.0,
    peak_memory_mb: float = 120.0,
    dataset_name: str = "Synthetic",
    n_sequences: int = 3,
) -> MagicMock:
    br = MagicMock()
    br.tracker_name = tracker_name
    br.dataset_name = dataset_name
    br.mean_iou = mean_iou
    br.mean_fps = mean_fps
    br.peak_memory_mb = peak_memory_mb
    latency_ms = 1000.0 / mean_fps
    br.sequence_results = [
        _mock_sequence_result(tracker_name, latency_ms, peak_memory_mb)
        for _ in range(n_sequences)
    ]
    return br


# ---------------------------------------------------------------------------
# _device_ees
# ---------------------------------------------------------------------------

class TestDeviceEES:
    def test_positive_values(self):
        ees = _device_ees(0.7, 100.0, 150.0, 512.0)
        assert ees > 0

    def test_zero_fps(self):
        assert _device_ees(0.7, 0.0, 150.0, 512.0) == 0.0

    def test_negative_fps(self):
        assert _device_ees(0.7, -1.0, 150.0, 512.0) == 0.0

    def test_zero_iou(self):
        assert _device_ees(0.0, 100.0, 150.0, 512.0) == 0.0

    def test_negative_iou(self):
        assert _device_ees(-0.1, 100.0, 150.0, 512.0) == 0.0

    def test_memory_penalty(self):
        ees_within = _device_ees(0.7, 100.0, 200.0, 512.0)
        ees_over = _device_ees(0.7, 100.0, 1024.0, 512.0)
        assert ees_within > ees_over

    def test_higher_fps_higher_ees(self):
        ees_low = _device_ees(0.7, 10.0, 150.0, 512.0)
        ees_high = _device_ees(0.7, 100.0, 150.0, 512.0)
        assert ees_high > ees_low

    def test_diminishing_returns(self):
        # Going from 10→100 FPS should gain more than 1000→1100 FPS
        gain_low = _device_ees(0.7, 100.0, 100.0, 512.0) - _device_ees(0.7, 10.0, 100.0, 512.0)
        gain_high = _device_ees(0.7, 1100.0, 100.0, 512.0) - _device_ees(0.7, 1000.0, 100.0, 512.0)
        assert gain_low > gain_high

    def test_formula_correctness(self):
        expected = 0.6 * math.log1p(50.0) / (1.0 + 100.0 / 512.0)
        got = _device_ees(0.6, 50.0, 100.0, 512.0)
        assert abs(got - expected) < 1e-10


# ---------------------------------------------------------------------------
# _aggregate_profiling
# ---------------------------------------------------------------------------

class TestAggregateProfilering:
    def test_basic(self):
        bench = _mock_benchmark_result("T", mean_fps=100.0, peak_memory_mb=150.0, n_sequences=4)
        result = _aggregate_profiling(bench)
        assert result.tracker_name == "T"
        assert result.fps > 0
        assert result.peak_memory_mb == 150.0
        assert result.frame_count > 0

    def test_empty_sequences_raises(self):
        bench = _mock_benchmark_result()
        bench.sequence_results = []
        with pytest.raises(ValueError):
            _aggregate_profiling(bench)

    def test_latency_mean_consistent_with_fps(self):
        bench = _mock_benchmark_result("T", mean_fps=200.0, n_sequences=2)
        result = _aggregate_profiling(bench)
        expected_latency = 1000.0 / result.fps
        assert abs(result.latency_mean_ms - expected_latency) < 0.01

    def test_grand_p95_above_mean(self):
        bench = _mock_benchmark_result("T", n_sequences=5)
        result = _aggregate_profiling(bench)
        assert result.latency_p95_ms >= result.latency_mean_ms


# ---------------------------------------------------------------------------
# DeploymentReportEngine construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_default_construction(self):
        engine = DeploymentReportEngine()
        assert engine.memory_budget_mb == 512.0
        assert engine.sustained_seconds == 0.0

    def test_invalid_memory_budget(self):
        with pytest.raises(ValueError):
            DeploymentReportEngine(memory_budget_mb=0.0)

    def test_custom_devices(self):
        engine = DeploymentReportEngine(device_names=["rpi4", "rpi5"])
        assert engine._device_names == ["rpi4", "rpi5"]

    def test_all_devices_loaded(self):
        engine = DeploymentReportEngine()
        # Should include the 6 built-in profiles
        assert len(engine._device_names) >= 6


# ---------------------------------------------------------------------------
# DeploymentReportEngine.analyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_empty_results_raises(self):
        engine = DeploymentReportEngine()
        with pytest.raises(ValueError):
            engine.analyze([])

    def test_single_tracker_single_device(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        bench = _mock_benchmark_result("MOSSE", mean_iou=0.6, mean_fps=500.0, peak_memory_mb=80.0)
        report = engine.analyze([bench])
        assert len(report.entries) == 1
        assert report.entries[0].tracker_name == "MOSSE"
        assert report.entries[0].device_name == "rpi4"

    def test_entries_count(self):
        engine = DeploymentReportEngine(device_names=["rpi4", "rpi5"])
        trackers = [
            _mock_benchmark_result("A", mean_fps=200.0),
            _mock_benchmark_result("B", mean_fps=50.0),
            _mock_benchmark_result("C", mean_fps=10.0),
        ]
        report = engine.analyze(trackers)
        # 3 trackers × 2 devices = 6 entries
        assert len(report.entries) == 6

    def test_recommendations_populated(self):
        engine = DeploymentReportEngine(device_names=["rpi4", "rpi5"])
        trackers = [
            _mock_benchmark_result("MOSSE", mean_fps=500.0, mean_iou=0.5),
            _mock_benchmark_result("CSRT", mean_fps=30.0, mean_iou=0.75),
        ]
        report = engine.analyze(trackers)
        assert "rpi4" in report.recommendations
        assert "rpi5" in report.recommendations

    def test_recommendation_is_in_entries(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        trackers = [
            _mock_benchmark_result("A", mean_fps=100.0, mean_iou=0.6),
            _mock_benchmark_result("B", mean_fps=200.0, mean_iou=0.4),
        ]
        report = engine.analyze(trackers)
        rec = report.recommendations["rpi4"]
        names = [e.tracker_name for e in report.entries_for_device("rpi4")]
        assert rec in names

    def test_pareto_flags_set(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        trackers = [
            _mock_benchmark_result("A", mean_fps=500.0, mean_iou=0.3),
            _mock_benchmark_result("B", mean_fps=30.0, mean_iou=0.8),
        ]
        report = engine.analyze(trackers)
        pareto_entries = [e for e in report.entries if e.on_pareto_front]
        assert len(pareto_entries) >= 1

    def test_exactly_one_recommendation_per_device(self):
        engine = DeploymentReportEngine(device_names=["rpi4", "jetson_nano"])
        trackers = [
            _mock_benchmark_result("X", mean_fps=200.0),
            _mock_benchmark_result("Y", mean_fps=100.0),
        ]
        report = engine.analyze(trackers)
        for device in ["rpi4", "jetson_nano"]:
            is_rec = [e.is_recommended for e in report.entries if e.device_name == device]
            assert sum(is_rec) == 1

    def test_dataset_name_propagated(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        bench = _mock_benchmark_result(dataset_name="OTB100")
        report = engine.analyze([bench])
        assert report.dataset_name == "OTB100"


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def _make_report(self):
        engine = DeploymentReportEngine(device_names=["rpi4", "rpi5"])
        trackers = [
            _mock_benchmark_result("MOSSE", mean_fps=400.0, mean_iou=0.5),
            _mock_benchmark_result("KCF", mean_fps=200.0, mean_iou=0.6),
        ]
        return engine, engine.analyze(trackers)

    def test_returns_string(self):
        engine, report = self._make_report()
        md = engine.to_markdown(report)
        assert isinstance(md, str)

    def test_contains_tracker_names(self):
        engine, report = self._make_report()
        md = engine.to_markdown(report)
        assert "MOSSE" in md
        assert "KCF" in md

    def test_contains_device_names(self):
        engine, report = self._make_report()
        md = engine.to_markdown(report)
        assert "Raspberry Pi 4" in md or "rpi4" in md

    def test_recommendation_section_present(self):
        engine, report = self._make_report()
        md = engine.to_markdown(report)
        assert "Recommended Tracker" in md

    def test_markdown_tables_present(self):
        engine, report = self._make_report()
        md = engine.to_markdown(report)
        assert "| Rank |" in md


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_structure(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        bench = _mock_benchmark_result("T")
        report = engine.analyze([bench])
        d = engine.to_dict(report)
        assert "metadata" in d
        assert "recommendations" in d
        assert "per_device" in d

    def test_metadata_fields(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        bench = _mock_benchmark_result("T", dataset_name="OTB100")
        report = engine.analyze([bench])
        meta = engine.to_dict(report)["metadata"]
        assert meta["dataset"] == "OTB100"
        assert "memory_budget_mb" in meta
        assert "devices" in meta

    def test_per_device_has_trackers(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        bench = _mock_benchmark_result("T")
        report = engine.analyze([bench])
        d = engine.to_dict(report)
        assert "rpi4" in d["per_device"]
        assert len(d["per_device"]["rpi4"]) == 1

    def test_entry_fields(self):
        engine = DeploymentReportEngine(device_names=["rpi4"])
        bench = _mock_benchmark_result("T")
        report = engine.analyze([bench])
        entry = engine.to_dict(report)["per_device"]["rpi4"][0]
        required = {
            "tracker", "mean_iou", "device_fps", "device_latency_ms",
            "fits_in_memory", "device_ees", "is_recommended",
        }
        assert required.issubset(entry.keys())


# ---------------------------------------------------------------------------
# DeploymentReport helpers
# ---------------------------------------------------------------------------

class TestDeploymentReportHelpers:
    def _make_report_obj(self):
        engine = DeploymentReportEngine(device_names=["rpi4", "rpi5"])
        t = [_mock_benchmark_result("A"), _mock_benchmark_result("B")]
        return engine.analyze(t)

    def test_entries_for_device(self):
        report = self._make_report_obj()
        rpi4 = report.entries_for_device("rpi4")
        assert all(e.device_name == "rpi4" for e in rpi4)

    def test_entries_for_device_sorted_by_ees(self):
        report = self._make_report_obj()
        rpi4 = report.entries_for_device("rpi4")
        ees_vals = [e.device_ees for e in rpi4]
        assert ees_vals == sorted(ees_vals, reverse=True)

    def test_entries_for_tracker(self):
        report = self._make_report_obj()
        entries = report.entries_for_tracker("A")
        assert all(e.tracker_name == "A" for e in entries)
        # Should appear on both devices
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Integration: importable from package
# ---------------------------------------------------------------------------

def test_importable_from_package():
    from eovot.reporting import DeploymentReportEngine as D
    assert D is DeploymentReportEngine
