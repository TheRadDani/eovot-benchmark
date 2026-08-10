"""Tests for GridSearchEngine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.grid_search import GridSearchEngine, GridSearchEntry
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=20, seed=0)


@pytest.fixture()
def tiny_grid():
    """2-parameter grid → 4 combinations."""
    return {"learning_rate": [0.10, 0.125], "sigma": [1.5, 2.0]}


@pytest.fixture()
def engine(tiny_grid):
    return GridSearchEngine(tracker_cls=MOSSETracker, param_grid=tiny_grid)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_param_grid_raises():
    with pytest.raises(ValueError, match="param_grid must not be empty"):
        GridSearchEngine(MOSSETracker, {})


def test_iter_combinations_count(engine, tiny_grid):
    combos = engine._iter_combinations()
    expected = 1
    for v in tiny_grid.values():
        expected *= len(v)
    assert len(combos) == expected


def test_iter_combinations_keys(engine, tiny_grid):
    for combo in engine._iter_combinations():
        assert set(combo.keys()) == set(tiny_grid.keys())


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_returns_entries(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    assert len(entries) == 4


def test_run_sorted_descending(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    ious = [e.mean_iou for e in entries]
    assert ious == sorted(ious, reverse=True)


def test_entries_have_correct_params(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    for entry in entries:
        assert "learning_rate" in entry.params
        assert "sigma" in entry.params


def test_entry_properties_in_range(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    for entry in entries:
        assert 0.0 <= entry.mean_iou <= 1.0
        assert entry.fps > 0.0
        assert entry.peak_memory_mb >= 0.0
        assert entry.elapsed_s >= 0.0


def test_run_max_sequences(engine, small_dataset):
    entries = engine.run(
        small_dataset, dataset_name="test", max_sequences=2, verbose=False
    )
    assert len(entries) == 4  # still 4 combos, just fewer sequences each


# ---------------------------------------------------------------------------
# best_config()
# ---------------------------------------------------------------------------


def test_best_config_returns_dict(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    best = engine.best_config(entries)
    assert isinstance(best, dict)
    assert set(best.keys()) == {"learning_rate", "sigma"}


def test_best_config_by_fps(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    best_fps = engine.best_config(entries, metric="fps")
    assert isinstance(best_fps, dict)


def test_best_config_empty_raises(engine):
    with pytest.raises(ValueError, match="entries list is empty"):
        engine.best_config([])


def test_best_config_bad_metric_raises(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    with pytest.raises(ValueError, match="Unknown metric"):
        engine.best_config(entries, metric="nonexistent")


# ---------------------------------------------------------------------------
# sensitivity_report()
# ---------------------------------------------------------------------------


def test_sensitivity_report_structure(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    report = engine.sensitivity_report(entries)
    assert set(report.keys()) == {"learning_rate", "sigma"}
    for param, mapping in report.items():
        for val, score in mapping.items():
            assert isinstance(score, float)


def test_sensitivity_report_value_count(engine, small_dataset, tiny_grid):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    report = engine.sensitivity_report(entries)
    for param, mapping in report.items():
        assert len(mapping) == len(tiny_grid[param])


# ---------------------------------------------------------------------------
# to_markdown()
# ---------------------------------------------------------------------------


def test_to_markdown_contains_header(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    md = engine.to_markdown(entries)
    assert "Rank" in md
    assert "mIoU" in md
    assert "FPS" in md


def test_to_markdown_top_n(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    md = engine.to_markdown(entries, top_n=2)
    # header + sep + 2 data rows
    data_rows = [l for l in md.strip().split("\n") if l.startswith("|") and "---" not in l and "Rank" not in l]
    assert len(data_rows) == 2


def test_sensitivity_to_markdown_contains_params(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    md = engine.sensitivity_to_markdown(entries)
    assert "learning_rate" in md
    assert "sigma" in md


# ---------------------------------------------------------------------------
# save_json()
# ---------------------------------------------------------------------------


def test_save_json_creates_file(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = engine.save_json(entries, path=str(Path(tmpdir) / "results.json"))
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["tracker"] == "MOSSETracker"
        assert data["n_combinations"] == 4
        assert len(data["results"]) == 4


def test_save_json_result_keys(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = engine.save_json(entries, path=str(Path(tmpdir) / "r.json"))
        data = json.loads(path.read_text())
        row = data["results"][0]
        assert "params" in row
        assert "mean_iou" in row
        assert "fps" in row
        assert "elapsed_s" in row


# ---------------------------------------------------------------------------
# GridSearchEntry.to_dict()
# ---------------------------------------------------------------------------


def test_entry_to_dict(engine, small_dataset):
    entries = engine.run(small_dataset, dataset_name="test", verbose=False)
    d = entries[0].to_dict()
    assert "params" in d
    assert "mean_iou" in d
    assert "success_auc" in d
    assert "precision_auc" in d
    assert "fps" in d
    assert "peak_memory_mb" in d


# ---------------------------------------------------------------------------
# from_config()
# ---------------------------------------------------------------------------


def test_from_config_mosse():
    config = {
        "tracker": {
            "name": "MOSSE",
            "param_grid": {
                "learning_rate": [0.1, 0.125],
                "sigma": [2.0],
            },
        }
    }
    eng = GridSearchEngine.from_config(config)
    assert eng.tracker_cls is MOSSETracker
    assert "learning_rate" in eng.param_grid


def test_from_config_kcf():
    config = {
        "tracker": {
            "name": "KCF",
            "param_grid": {"learning_rate": [0.075, 0.10]},
        }
    }
    eng = GridSearchEngine.from_config(config)
    assert eng.tracker_cls is KCFTracker


def test_from_config_unknown_tracker_raises():
    config = {
        "tracker": {
            "name": "NonExistentTracker",
            "param_grid": {"x": [1, 2]},
        }
    }
    with pytest.raises(ValueError, match="Unknown tracker"):
        GridSearchEngine.from_config(config)


def test_from_config_empty_param_grid_raises():
    config = {
        "tracker": {
            "name": "MOSSE",
            "param_grid": {},
        }
    }
    with pytest.raises(ValueError):
        GridSearchEngine.from_config(config)


# ---------------------------------------------------------------------------
# run_from_config() — integration
# ---------------------------------------------------------------------------


def test_run_from_config_synthetic():
    config = {
        "tracker": {
            "name": "MOSSE",
            "param_grid": {"learning_rate": [0.10, 0.125]},
        },
        "dataset": {
            "loader": "SyntheticDataset",
            "name": "Synthetic",
            "params": {"num_sequences": 2, "num_frames": 15, "seed": 7},
        },
        "benchmark": {"verbose": False},
    }
    eng = GridSearchEngine.from_config(config)
    entries = eng.run_from_config(config)
    assert len(entries) == 2
    assert all(isinstance(e, GridSearchEntry) for e in entries)


def test_run_from_config_unknown_loader_raises():
    config = {
        "tracker": {
            "name": "MOSSE",
            "param_grid": {"learning_rate": [0.10]},
        },
        "dataset": {"loader": "NoSuchDataset", "params": {}},
    }
    eng = GridSearchEngine.from_config(config)
    with pytest.raises(ValueError, match="Unknown dataset loader"):
        eng.run_from_config(config)
