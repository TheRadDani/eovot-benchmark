"""Tests for the hyperparameter sweep utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.sweep import (
    HyperparameterSweeper,
    SweepEntry,
    SweepResult,
    _generate_configs,
    _params_str,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=30, motion="linear", seed=0)


# ---------------------------------------------------------------------------
# _generate_configs unit tests
# ---------------------------------------------------------------------------


def test_generate_configs_empty_grid():
    """Empty grid produces one config with no parameters."""
    configs = _generate_configs({})
    assert configs == [{}]


def test_generate_configs_single_param():
    configs = _generate_configs({"lr": [0.05, 0.1, 0.2]})
    assert len(configs) == 3
    assert all("lr" in c for c in configs)
    lrs = [c["lr"] for c in configs]
    assert set(lrs) == {0.05, 0.1, 0.2}


def test_generate_configs_cartesian_product():
    """2 × 3 grid produces 6 configs."""
    configs = _generate_configs({"a": [1, 2], "b": [10, 20, 30]})
    assert len(configs) == 6
    # All (a, b) pairs should be present.
    pairs = {(c["a"], c["b"]) for c in configs}
    assert len(pairs) == 6


def test_generate_configs_sorted_keys():
    """Keys are iterated in sorted order for reproducibility."""
    configs = _generate_configs({"z": [1, 2], "a": [10]})
    assert list(configs[0].keys()) == ["a", "z"]


# ---------------------------------------------------------------------------
# HyperparameterSweeper — basic contract
# ---------------------------------------------------------------------------


def test_sweep_empty_grid_returns_one_entry(small_dataset, tmp_path):
    """An empty param_grid evaluates the tracker once with default params."""
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="MOSSE",
        param_grid={},
        dataset=small_dataset,
        dataset_name="Test",
    )
    assert isinstance(result, SweepResult)
    assert len(result.entries) == 1
    assert result.entries[0].params == {}


def test_sweep_grid_count(small_dataset, tmp_path):
    """2 × 2 grid produces exactly 4 entries."""
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.05, 0.125], "padding": [1.0, 1.5]},
        dataset=small_dataset,
        dataset_name="Test",
        max_sequences=3,
    )
    assert len(result.entries) == 4


def test_sweep_entries_are_sweep_entry(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="MOSSE",
        param_grid={"learning_rate": [0.05, 0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    for entry in result.entries:
        assert isinstance(entry, SweepEntry)


def test_sweep_iou_bounded(small_dataset, tmp_path):
    """All mean_iou values are in [0, 1]."""
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.075, 0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    for entry in result.entries:
        assert 0.0 <= entry.mean_iou <= 1.0


def test_sweep_wall_time_positive(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="MOSSE",
        param_grid={"learning_rate": [0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    assert result.entries[0].wall_time_s > 0.0


# ---------------------------------------------------------------------------
# SweepResult — best config selection
# ---------------------------------------------------------------------------


def test_best_by_success_auc_not_none(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.05, 0.075, 0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    best = result.best_by_success_auc
    assert best is not None
    assert isinstance(best, SweepEntry)


def test_best_by_iou_not_none(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="MOSSE",
        param_grid={"learning_rate": [0.1, 0.2]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    best = result.best_by_iou
    assert best is not None
    assert all(best.mean_iou >= e.mean_iou for e in result.entries)


def test_best_none_when_no_entries():
    result = SweepResult(
        tracker_name="KCF", dataset_name="Test", param_grid={"lr": [0.1]}
    )
    assert result.best_by_success_auc is None
    assert result.best_by_iou is None


# ---------------------------------------------------------------------------
# SweepResult.to_dict
# ---------------------------------------------------------------------------


def test_to_dict_structure(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.075]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    d = result.to_dict()
    assert d["tracker"] == "KCF"
    assert d["dataset"] == "Test"
    assert d["num_configs"] == 1
    assert isinstance(d["rows"], list)
    assert "best_params" in d


def test_to_dict_rows_contain_mean_iou(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.075, 0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    d = result.to_dict()
    for row in d["rows"]:
        assert "mean_iou" in row
        assert "mean_fps" in row


# ---------------------------------------------------------------------------
# SweepResult.to_markdown_table
# ---------------------------------------------------------------------------


def test_markdown_table_contains_param_name(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.075, 0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    table = result.to_markdown_table()
    assert "learning_rate" in table
    assert "mIoU" in table
    assert "FPS" in table


def test_markdown_table_star_marks_best(small_dataset, tmp_path):
    """The best configuration row ends with ★."""
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    result = sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.05, 0.075, 0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    table = result.to_markdown_table()
    star_rows = [line for line in table.split("\n") if "★" in line]
    assert len(star_rows) == 1


def test_markdown_table_empty_result():
    result = SweepResult(
        tracker_name="KCF", dataset_name="Test", param_grid={"lr": [0.1]}
    )
    assert "No results" in result.to_markdown_table()


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def test_sweep_saves_json_and_md(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    sweeper.sweep(
        tracker_name="KCF",
        param_grid={"learning_rate": [0.075, 0.125]},
        dataset=small_dataset,
        dataset_name="MySet",
    )
    json_files = list(tmp_path.glob("*.json"))
    md_files = list(tmp_path.glob("*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1


def test_sweep_json_is_valid(small_dataset, tmp_path):
    sweeper = HyperparameterSweeper(output_dir=str(tmp_path), verbose=False)
    sweeper.sweep(
        tracker_name="MOSSE",
        param_grid={"learning_rate": [0.125]},
        dataset=small_dataset,
        dataset_name="Test",
    )
    json_path = next(tmp_path.glob("*.json"))
    with open(json_path) as fh:
        d = json.load(fh)
    assert d["tracker"] == "MOSSE"
    assert d["num_configs"] == 1


# ---------------------------------------------------------------------------
# _params_str helper
# ---------------------------------------------------------------------------


def test_params_str_empty():
    assert _params_str({}) == ""


def test_params_str_sorted():
    s = _params_str({"z": 1, "a": 2})
    assert s.index("a=") < s.index("z=")
