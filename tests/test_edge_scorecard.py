"""Tests for EdgeScorecard."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.edge_scorecard import EdgeScorecard, ScorecardRow
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_results():
    """Run MOSSE and KCF on 3 short synthetic sequences."""
    dataset = SyntheticDataset(num_sequences=3, num_frames=30, motion="linear")
    engine = BenchmarkEngine(verbose=False)
    results = [
        engine.run(MOSSETracker(), dataset, dataset_name="Synthetic"),
        engine.run(KCFTracker(), dataset, dataset_name="Synthetic"),
    ]
    return results


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestEdgeScorecardConstruction:
    def test_default_devices_populated(self):
        sc = EdgeScorecard()
        assert len(sc._devices) > 0

    def test_explicit_devices_accepted(self):
        sc = EdgeScorecard(devices=["rpi4", "jetson_nano"])
        assert sc._devices == ["rpi4", "jetson_nano"]

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="Unknown devices"):
            EdgeScorecard(devices=["nonexistent_board"])

    def test_non_positive_target_fps_raises(self):
        with pytest.raises(ValueError):
            EdgeScorecard(target_fps=0.0)

    def test_invalid_marginal_ratio_raises(self):
        with pytest.raises(ValueError):
            EdgeScorecard(marginal_fps_ratio=1.5)

    def test_invalid_energy_weight_raises(self):
        with pytest.raises(ValueError):
            EdgeScorecard(energy_weight=-0.1)


# ---------------------------------------------------------------------------
# Evaluation correctness
# ---------------------------------------------------------------------------

class TestEdgeScorecardEvaluate:
    def test_row_count(self, benchmark_results):
        devices = ["rpi4", "jetson_nano"]
        sc = EdgeScorecard(devices=devices)
        rows = sc.evaluate(benchmark_results)
        # 2 trackers × 2 devices
        assert len(rows) == 4

    def test_rows_sorted_by_tracker_then_device(self, benchmark_results):
        devices = ["rpi4", "jetson_nano"]
        sc = EdgeScorecard(devices=devices)
        rows = sc.evaluate(benchmark_results)
        keys = [(r.tracker_name, r.device_name) for r in rows]
        assert keys == sorted(keys)

    def test_score_in_unit_interval(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        for r in rows:
            assert 0.0 <= r.deployment_score <= 1.0, (
                f"{r.tracker_name} on {r.device_name}: score={r.deployment_score}"
            )

    def test_tier_values(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4", "jetson_xnx"])
        rows = sc.evaluate(benchmark_results)
        valid_tiers = {"READY", "MARGINAL", "NOT FEASIBLE"}
        for r in rows:
            assert r.tier in valid_tiers

    def test_projected_fps_positive(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        for r in rows:
            assert r.projected_fps > 0

    def test_energy_non_negative(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        for r in rows:
            assert r.energy_wh_per_hour >= 0.0

    def test_memory_fail_gives_not_feasible(self):
        """A device with 1 MB RAM should mark all trackers NOT FEASIBLE."""
        from eovot.profiling.device_sim import DeviceProfile
        sc = EdgeScorecard(devices=["coral_board"])  # small memory
        # Override the coral profile memory limit to 1 MB
        sc._sim._profiles["coral_board"] = DeviceProfile(
            name="coral_board",
            display_name="Tiny Test Board",
            cpu_speed_factor=0.1,
            memory_limit_mb=1.0,   # impossibly small
            tdp_watts=4.0,
        )
        dataset = SyntheticDataset(num_sequences=2, num_frames=20)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(MOSSETracker(), dataset, dataset_name="Syn")]
        rows = sc.evaluate(results)
        for r in rows:
            assert not r.memory_ok
            assert r.tier == "NOT FEASIBLE"

    def test_empty_results_returns_empty(self):
        sc = EdgeScorecard(devices=["rpi4"])
        assert sc.evaluate([]) == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestEdgeScorecardExport:
    def test_markdown_contains_header(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        md = sc.to_markdown(rows)
        assert "Tracker" in md
        assert "Device" in md
        assert "Tier" in md

    def test_markdown_contains_tracker_names(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        md = sc.to_markdown(rows)
        for r in rows:
            assert r.tracker_name in md

    def test_json_roundtrip(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "scorecard.json")
            sc.to_json(rows, path)
            data = json.loads(Path(path).read_text())
        assert "scorecard" in data
        assert "config" in data
        assert len(data["scorecard"]) == len(rows)
        # Verify a few fields
        entry = data["scorecard"][0]
        assert "projected_fps" in entry
        assert "tier" in entry
        assert "deployment_score" in entry

    def test_csv_written(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "scorecard.csv")
            sc.to_csv(rows, path)
            text = Path(path).read_text()
        assert "tracker" in text
        assert "deployment_score" in text
        assert "MOSSE" in text or "KCF" in text

    def test_row_str_representation(self, benchmark_results):
        sc = EdgeScorecard(devices=["rpi4"])
        rows = sc.evaluate(benchmark_results)
        for r in rows:
            s = str(r)
            assert r.tracker_name in s
            assert r.tier in s
