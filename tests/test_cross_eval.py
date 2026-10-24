"""Tests for CrossDatasetEvaluator and CrossEvalTable."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.cross_eval import (
    CrossDatasetEvaluator,
    CrossEvalTable,
    TrackerCrossResult,
)
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_ds(motion: str = "linear", seed: int = 0) -> SyntheticDataset:
    return SyntheticDataset(num_sequences=3, num_frames=30, motion=motion, seed=seed)


def _evaluator(**kwargs) -> CrossDatasetEvaluator:
    ds = {"Lin": _small_ds("linear"), "Circ": _small_ds("circular", seed=10)}
    ev = CrossDatasetEvaluator(datasets=ds, verbose=False, **kwargs)
    ev.add_tracker(MOSSETracker())
    ev.add_tracker(KCFTracker())
    return ev


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator construction
# ---------------------------------------------------------------------------

def test_empty_datasets_raises():
    with pytest.raises(ValueError, match="least one dataset"):
        CrossDatasetEvaluator(datasets={})


def test_duplicate_tracker_raises():
    ev = CrossDatasetEvaluator(datasets={"D": _small_ds()}, verbose=False)
    ev.add_tracker(MOSSETracker())
    with pytest.raises(ValueError, match="already registered"):
        ev.add_tracker(MOSSETracker())


def test_run_without_trackers_raises():
    ev = CrossDatasetEvaluator(datasets={"D": _small_ds()}, verbose=False)
    with pytest.raises(RuntimeError, match="No trackers"):
        ev.run()


def test_invalid_weight_raises():
    ev = CrossDatasetEvaluator(datasets={"D": _small_ds()}, verbose=False)
    with pytest.raises(ValueError, match="≥ 0"):
        ev.set_weight("D", -0.5)


def test_weight_unknown_dataset_raises():
    ev = CrossDatasetEvaluator(datasets={"D": _small_ds()}, verbose=False)
    with pytest.raises(ValueError, match="not registered"):
        ev.set_weight("Unknown", 1.0)


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator run
# ---------------------------------------------------------------------------

def test_run_returns_cross_eval_table():
    ev = _evaluator()
    table = ev.run()
    assert isinstance(table, CrossEvalTable)


def test_run_result_count():
    ev = _evaluator()
    table = ev.run()
    assert len(table.results) == 2


def test_run_dataset_names():
    ev = _evaluator()
    table = ev.run()
    assert set(table.dataset_names) == {"Lin", "Circ"}


def test_run_aps_in_range():
    ev = _evaluator()
    table = ev.run()
    for r in table.results:
        assert 0.0 <= r.aps <= 1.0


def test_run_sorted_by_aps_descending():
    ev = _evaluator()
    table = ev.run()
    apss = [r.aps for r in table.results]
    assert apss == sorted(apss, reverse=True)


def test_run_per_dataset_populated():
    ev = _evaluator()
    table = ev.run()
    for r in table.results:
        assert "Lin" in r.per_dataset
        assert "Circ" in r.per_dataset


def test_run_with_max_sequences():
    ev = CrossDatasetEvaluator(
        datasets={"D": SyntheticDataset(num_sequences=10, num_frames=20)},
        verbose=False,
        max_sequences=3,
    )
    ev.add_tracker(MOSSETracker())
    table = ev.run()
    for r in table.results:
        br = r.per_dataset["D"]
        assert len(br.sequence_results) == 3


# ---------------------------------------------------------------------------
# APS weighting
# ---------------------------------------------------------------------------

def test_aps_with_zero_weight_dataset():
    """A dataset with weight=0 should not contribute to APS."""
    ev = CrossDatasetEvaluator(
        datasets={"D1": _small_ds(seed=1), "D2": _small_ds(seed=2)},
        dataset_weights={"D1": 1.0, "D2": 0.0},
        verbose=False,
    )
    ev.add_tracker(MOSSETracker())
    table = ev.run()
    r = table.results[0]
    br_d1 = r.per_dataset["D1"]
    # APS should equal D1's success_auc since D2 has 0 weight
    expected = br_d1.mean_success_auc or 0.0
    assert abs(r.aps - expected) < 1e-6


def test_aps_uniform_weights_equals_mean():
    """With equal weights, APS == mean of per-dataset success_aucs."""
    ev = CrossDatasetEvaluator(
        datasets={"D1": _small_ds(seed=1), "D2": _small_ds(seed=2)},
        verbose=False,
    )
    ev.add_tracker(MOSSETracker())
    table = ev.run()
    r = table.results[0]
    aucs = [
        br.mean_success_auc
        for br in r.per_dataset.values()
        if br.mean_success_auc is not None
    ]
    expected_aps = float(np.mean(aucs))
    assert abs(r.aps - expected_aps) < 1e-5


# ---------------------------------------------------------------------------
# CrossEvalTable output formats
# ---------------------------------------------------------------------------

def test_to_markdown_nonempty():
    ev = _evaluator()
    table = ev.run()
    md = table.to_markdown()
    assert "APS" in md
    assert "MOSSE" in md or "KCF" in md


def test_to_markdown_empty_table():
    table = CrossEvalTable()
    md = table.to_markdown()
    assert "No results" in md


def test_to_csv_writes_file():
    ev = _evaluator()
    table = ev.run()
    with tempfile.TemporaryDirectory() as tmp:
        p = table.to_csv(Path(tmp) / "leaderboard.csv")
        assert p.exists()
        text = p.read_text()
        assert "aps" in text
        assert "MOSSE" in text or "KCF" in text


def test_to_dict_structure():
    ev = _evaluator()
    table = ev.run()
    d = table.to_dict()
    assert "leaderboard" in d
    assert "dataset_names" in d
    assert len(d["leaderboard"]) == 2
    first = d["leaderboard"][0]
    assert "aps" in first
    assert "tracker" in first
    assert "datasets" in first


def test_save_and_reload_json():
    ev = _evaluator()
    table = ev.run()
    with tempfile.TemporaryDirectory() as tmp:
        p = table.save(Path(tmp) / "result")
        assert p.suffix == ".json"
        with open(p) as f:
            d = json.load(f)
        assert "leaderboard" in d
        assert len(d["leaderboard"]) == 2


# ---------------------------------------------------------------------------
# Resume / prior results
# ---------------------------------------------------------------------------

def test_resume_skips_cached():
    ev = _evaluator()
    table1 = ev.run()

    # Build prior dict from first run
    prior = {}
    for r in table1.results:
        prior[r.tracker_name] = dict(r.per_dataset)

    # Second run with prior — should not re-evaluate
    ev2 = _evaluator()
    table2 = ev2.run(prior_results=prior)

    # Results should still have 2 entries with the same APS
    assert len(table2.results) == 2
    for r1, r2 in zip(
        sorted(table1.results, key=lambda x: x.tracker_name),
        sorted(table2.results, key=lambda x: x.tracker_name),
    ):
        assert abs(r1.aps - r2.aps) < 1e-8


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_package_exports():
    from eovot.experiment import CrossDatasetEvaluator, CrossEvalTable, TrackerCrossResult
    assert CrossDatasetEvaluator is not None
    assert CrossEvalTable is not None
    assert TrackerCrossResult is not None
