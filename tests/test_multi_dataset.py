"""Tests for MultiDatasetEvaluator and MultiDatasetResult."""

from __future__ import annotations

import json
import math
import pytest

from eovot.benchmark.multi_dataset import (
    CrossDatasetEntry,
    MultiDatasetEvaluator,
    MultiDatasetResult,
)
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_datasets():
    ds1 = SyntheticDataset(num_sequences=2, num_frames=40, motion="linear")
    ds2 = SyntheticDataset(num_sequences=2, num_frames=60, motion="random")
    return [(ds1, "Syn-Linear"), (ds2, "Syn-Random")]


@pytest.fixture()
def two_trackers():
    return [MOSSETracker(), KCFTracker()]


# ---------------------------------------------------------------------------
# MultiDatasetEvaluator.run
# ---------------------------------------------------------------------------

class TestMultiDatasetEvaluatorRun:
    def test_returns_multi_dataset_result(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        assert isinstance(result, MultiDatasetResult)

    def test_result_count(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        # 2 trackers × 2 datasets = 4 BenchmarkResult objects
        assert len(result.benchmark_results) == 4

    def test_dataset_names_stored(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        names = {r.dataset_name for r in result.benchmark_results}
        assert names == {"Syn-Linear", "Syn-Random"}

    def test_tracker_names_stored(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        names = {r.tracker_name for r in result.benchmark_results}
        assert names == {"MOSSE", "KCF"}

    def test_max_sequences_respected(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets, max_sequences=1)
        for r in result.benchmark_results:
            assert len(r.sequence_results) == 1

    def test_single_tracker_single_dataset(self, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run([MOSSETracker()], [two_datasets[0]])
        assert len(result.benchmark_results) == 1


# ---------------------------------------------------------------------------
# MultiDatasetResult.aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_returns_one_entry_per_tracker(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        entries = result.aggregate()
        assert len(entries) == 2

    def test_sorted_by_iou_desc(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        entries = result.aggregate()
        ious = [e.mean_iou for e in entries]
        assert ious == sorted(ious, reverse=True)

    def test_weighted_iou_is_finite(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        for e in result.aggregate():
            assert math.isfinite(e.mean_iou)
            assert 0.0 <= e.mean_iou <= 1.0

    def test_total_frames_equals_sum(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run([MOSSETracker()], two_datasets)
        entries = result.aggregate()
        assert len(entries) == 1
        # 2 sequences × 40 frames + 2 sequences × 60 frames = 200
        assert entries[0].total_frames == 200

    def test_per_dataset_keys_present(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run([MOSSETracker()], two_datasets)
        entry = result.aggregate()[0]
        assert set(entry.per_dataset.keys()) == {"Syn-Linear", "Syn-Random"}

    def test_peak_memory_is_max(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run([MOSSETracker()], two_datasets)
        entry = result.aggregate()[0]
        from_results = [
            r.peak_memory_mb for r in result.benchmark_results
            if r.tracker_name == "MOSSE"
        ]
        assert entry.peak_memory_mb == max(from_results)


# ---------------------------------------------------------------------------
# MultiDatasetResult.cross_dataset_table
# ---------------------------------------------------------------------------

class TestCrossDatasetTable:
    def test_returns_string(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        table = result.cross_dataset_table()
        assert isinstance(table, str)

    def test_contains_tracker_names(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        table = result.cross_dataset_table()
        assert "MOSSE" in table
        assert "KCF" in table

    def test_contains_dataset_names(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        table = result.cross_dataset_table()
        assert "Syn-Linear" in table
        assert "Syn-Random" in table

    def test_markdown_header_row(self, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        table = result.cross_dataset_table()
        lines = table.splitlines()
        assert any("Rank" in line for line in lines)
        assert any("mIoU" in line for line in lines)

    def test_empty_result_returns_fallback(self):
        result = MultiDatasetResult()
        table = result.cross_dataset_table()
        assert "No results" in table


# ---------------------------------------------------------------------------
# MultiDatasetResult.save / round-trip
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_json(self, tmp_path, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        out = tmp_path / "multi_result"
        saved = result.save(out)
        assert saved.suffix == ".json"
        assert saved.exists()

    def test_saved_json_valid(self, tmp_path, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        saved = result.save(tmp_path / "out.json")
        data = json.loads(saved.read_text())
        assert "aggregate" in data
        assert "results" in data
        assert len(data["aggregate"]) == 2
        assert len(data["results"]) == 4

    def test_aggregate_keys_present(self, tmp_path, two_trackers, two_datasets):
        ev = MultiDatasetEvaluator(verbose=False)
        result = ev.run(two_trackers, two_datasets)
        saved = result.save(tmp_path / "out.json")
        data = json.loads(saved.read_text())
        for entry in data["aggregate"]:
            assert "tracker_name" in entry
            assert "mean_iou" in entry
            assert "mean_fps" in entry
            assert "total_frames" in entry
            assert "per_dataset" in entry


# ---------------------------------------------------------------------------
# MultiDatasetEvaluator.from_config
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_config_synthetic(self):
        config = {
            "experiment": {"tdp_watts": None, "max_sequences": 1},
            "datasets": [
                {
                    "loader": "synthetic",
                    "name": "Syn-A",
                    "n_sequences": 2,
                    "n_frames": 30,
                    "motion": "linear",
                },
                {
                    "loader": "synthetic",
                    "name": "Syn-B",
                    "n_sequences": 2,
                    "n_frames": 30,
                    "motion": "circular",
                },
            ],
            "trackers": [
                {"name": "MOSSE", "params": {}},
            ],
        }
        result = MultiDatasetEvaluator.from_config(config, verbose=False)
        assert isinstance(result, MultiDatasetResult)
        assert len(result.benchmark_results) == 2

    def test_from_config_unknown_loader_raises(self):
        config = {
            "experiment": {},
            "datasets": [{"loader": "NoSuchDataset", "name": "Bad"}],
            "trackers": [{"name": "MOSSE", "params": {}}],
        }
        with pytest.raises(ValueError, match="Unknown dataset loader"):
            MultiDatasetEvaluator.from_config(config, verbose=False)

    def test_from_config_unknown_tracker_raises(self):
        config = {
            "experiment": {},
            "datasets": [
                {"loader": "synthetic", "name": "S", "n_sequences": 1, "n_frames": 20}
            ],
            "trackers": [{"name": "GhostTracker", "params": {}}],
        }
        with pytest.raises(ValueError):
            MultiDatasetEvaluator.from_config(config, verbose=False)
