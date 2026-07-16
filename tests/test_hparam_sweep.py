"""Tests for eovot.experiment.hparam_sweep — SweepConfig, SweepResult, TrackerHparamSweep."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.experiment.hparam_sweep import (
    SweepConfig,
    SweepEntry,
    SweepResult,
    TrackerHparamSweep,
)
from eovot.benchmark.engine import BenchmarkResult, SequenceResult
from eovot.profiling.profiler import ProfilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_profiling_result(tracker_name: str = "TestTracker") -> ProfilingResult:
    return ProfilingResult(
        tracker_name=tracker_name,
        frame_count=50,
        fps=120.0,
        latency_mean_ms=8.3,
        latency_std_ms=0.5,
        latency_p95_ms=9.1,
        peak_memory_mb=64.0,
    )


def _mock_sequence_result(iou: float = 0.6, tracker_name: str = "T") -> SequenceResult:
    from eovot.metrics.accuracy import AccuracyMetrics
    return SequenceResult(
        sequence_name="seq1",
        ious=np.full(50, iou),
        profiling=_mock_profiling_result(tracker_name),
        predictions=np.zeros((50, 4)),
        ground_truths=np.zeros((50, 4)),
        accuracy_metrics=AccuracyMetrics(
            mean_iou=iou,
            success_auc=iou * 0.9,
            precision_auc=iou * 0.85,
        ),
    )


def _mock_benchmark_result(iou: float = 0.6, tracker_name: str = "TestTracker") -> BenchmarkResult:
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name="test_ds")
    result.sequence_results.append(_mock_sequence_result(iou, tracker_name))
    return result


def _make_sweep_entries(scores: list) -> list:
    entries = []
    for i, s in enumerate(scores):
        entries.append(SweepEntry(
            params={"learning_rate": 0.05 * (i + 1)},
            score=s,
            benchmark_result=_mock_benchmark_result(s),
        ))
    return entries


def _make_synthetic_dataset(n_seq: int = 3, n_frames: int = 15) -> object:
    from eovot.datasets.synthetic import SyntheticDataset
    return SyntheticDataset(
        num_sequences=n_seq, num_frames=n_frames,
        frame_size=(80, 60), bbox_size=(20, 15),
        motion="linear", seed=42,
    )


# ---------------------------------------------------------------------------
# SweepConfig tests
# ---------------------------------------------------------------------------

class TestSweepConfig:
    def test_defaults(self):
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={"learning_rate": [0.05, 0.10]},
        )
        assert cfg.metric == "success_auc"
        assert cfg.max_sequences is None

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be"):
            SweepConfig(
                tracker_name="KCF",
                param_grid={"lr": [0.1]},
                metric="iou_score",
            )

    def test_empty_param_grid_raises(self):
        with pytest.raises(ValueError, match="param_grid must not be empty"):
            SweepConfig(tracker_name="KCF", param_grid={})

    def test_all_valid_metrics(self):
        for m in ("mean_iou", "success_auc", "precision_auc", "mean_fps"):
            cfg = SweepConfig(tracker_name="KCF", param_grid={"lr": [0.1]}, metric=m)
            assert cfg.metric == m

    def test_max_sequences_set(self):
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={"lr": [0.1]},
            max_sequences=5,
        )
        assert cfg.max_sequences == 5


# ---------------------------------------------------------------------------
# SweepResult tests
# ---------------------------------------------------------------------------

class TestSweepResult:
    def test_to_dict_keys(self):
        result = SweepResult(
            tracker_name="KCF", metric="success_auc",
            best_params={"learning_rate": 0.075}, best_score=0.72,
        )
        d = result.to_dict()
        for k in ("tracker", "metric", "best_params", "best_score", "n_configs_evaluated"):
            assert k in d

    def test_to_dict_best_score_rounded(self):
        r = SweepResult(
            tracker_name="KCF", metric="success_auc",
            best_params={}, best_score=0.7123456789,
        )
        assert r.to_dict()["best_score"] == round(0.7123456789, 6)

    def test_to_markdown_has_table(self):
        entries = _make_sweep_entries([0.8, 0.6, 0.7])
        r = SweepResult(
            tracker_name="KCF", metric="success_auc",
            best_params=entries[0].params, best_score=0.8,
            entries=entries,
        )
        md = r.to_markdown()
        assert "KCF" in md
        assert "success_auc" in md
        assert "|" in md

    def test_to_markdown_shows_best(self):
        entries = _make_sweep_entries([0.9, 0.5])
        r = SweepResult(
            tracker_name="MOSSE", metric="mean_iou",
            best_params=entries[0].params, best_score=0.9,
            entries=entries,
        )
        md = r.to_markdown()
        assert "0.9000" in md

    def test_to_csv_creates_file(self, tmp_path):
        entries = _make_sweep_entries([0.7, 0.6])
        r = SweepResult(
            tracker_name="KCF", metric="success_auc",
            best_params={}, best_score=0.7, entries=entries,
        )
        csv_path = str(tmp_path / "sweep.csv")
        r.to_csv(csv_path)
        assert os.path.isfile(csv_path)
        content = Path(csv_path).read_text()
        assert "learning_rate" in content
        assert "success_auc" in content

    def test_save_json_creates_file(self, tmp_path):
        r = SweepResult(
            tracker_name="KCF", metric="success_auc",
            best_params={"lr": 0.1}, best_score=0.65, entries=[],
        )
        path = str(tmp_path / "sweep.json")
        r.save_json(path)
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["tracker"] == "KCF"
        assert data["best_score"] == round(0.65, 6)

    def test_empty_result_markdown(self):
        r = SweepResult(
            tracker_name="KCF", metric="mean_iou",
            best_params={}, best_score=0.0, entries=[],
        )
        md = r.to_markdown()
        assert "No results" in md

    def test_empty_to_csv_does_not_raise(self, tmp_path):
        r = SweepResult(
            tracker_name="KCF", metric="mean_iou",
            best_params={}, best_score=0.0, entries=[],
        )
        r.to_csv(str(tmp_path / "empty.csv"))  # should not raise


# ---------------------------------------------------------------------------
# TrackerHparamSweep — constructor
# ---------------------------------------------------------------------------

class TestTrackerHparamSweepConstructor:
    def test_default_mode(self):
        s = TrackerHparamSweep()
        assert s.mode == "grid"

    def test_random_mode(self):
        s = TrackerHparamSweep(mode="random", n_random=10)
        assert s.mode == "random"
        assert s.n_random == 10

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            TrackerHparamSweep(mode="bayesian")

    def test_seed_stored(self):
        s = TrackerHparamSweep(seed=42)
        assert s.seed == 42


# ---------------------------------------------------------------------------
# _get_combinations
# ---------------------------------------------------------------------------

class TestGetCombinations:
    def test_grid_count(self):
        s = TrackerHparamSweep(mode="grid")
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={
                "a": [1, 2, 3],
                "b": [10, 20],
            },
        )
        combos = s._get_combinations(cfg)
        assert len(combos) == 6  # 3 × 2

    def test_grid_contains_all_combinations(self):
        s = TrackerHparamSweep(mode="grid")
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={"lr": [0.05, 0.10], "lambda_": [1e-4, 1e-3]},
        )
        combos = s._get_combinations(cfg)
        assert {"lr": 0.05, "lambda_": 1e-4} in combos
        assert {"lr": 0.10, "lambda_": 1e-3} in combos

    def test_random_count_respected(self):
        s = TrackerHparamSweep(mode="random", n_random=3, seed=0)
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={"lr": [0.05, 0.10, 0.15, 0.20], "l": [1e-5, 1e-4, 1e-3]},
        )
        combos = s._get_combinations(cfg)
        assert len(combos) == 3

    def test_random_subset_of_grid(self):
        s = TrackerHparamSweep(mode="random", n_random=5, seed=7)
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={"lr": [0.05, 0.10, 0.15, 0.20, 0.25]},
        )
        combos = s._get_combinations(cfg)
        # All returned combos must be from the original grid
        all_lrs = {0.05, 0.10, 0.15, 0.20, 0.25}
        for c in combos:
            assert c["lr"] in all_lrs

    def test_single_param_single_value(self):
        s = TrackerHparamSweep(mode="grid")
        cfg = SweepConfig(
            tracker_name="KCF", param_grid={"lr": [0.1]},
        )
        combos = s._get_combinations(cfg)
        assert combos == [{"lr": 0.1}]


# ---------------------------------------------------------------------------
# _get_metric
# ---------------------------------------------------------------------------

class TestGetMetric:
    def test_mean_iou(self):
        s = TrackerHparamSweep()
        r = _mock_benchmark_result(iou=0.55)
        assert abs(s._get_metric(r, "mean_iou") - 0.55) < 0.01

    def test_success_auc(self):
        s = TrackerHparamSweep()
        r = _mock_benchmark_result(iou=0.6)
        score = s._get_metric(r, "success_auc")
        # success_auc = 0.6 * 0.9 = 0.54
        assert score > 0.0

    def test_mean_fps(self):
        s = TrackerHparamSweep()
        r = _mock_benchmark_result()
        fps = s._get_metric(r, "mean_fps")
        assert fps == pytest.approx(120.0, rel=0.01)


# ---------------------------------------------------------------------------
# Full integration run (small synthetic dataset)
# ---------------------------------------------------------------------------

class TestTrackerHparamSweepIntegration:
    def test_grid_sweep_mosse(self):
        """Grid sweep over MOSSE learning_rate should return correctly sorted results."""
        ds = _make_synthetic_dataset()
        cfg = SweepConfig(
            tracker_name="MOSSE",
            param_grid={"learning_rate": [0.05, 0.10, 0.15]},
            metric="mean_iou",
            max_sequences=2,
        )
        sweep = TrackerHparamSweep(mode="grid", verbose=False)
        result = sweep.run(cfg, ds, "synthetic_test")

        assert result.tracker_name == "MOSSE"
        assert len(result.entries) == 3
        # Scores should be descending
        scores = [e.score for e in result.entries]
        assert scores == sorted(scores, reverse=True)
        assert result.best_score == scores[0]
        assert "learning_rate" in result.best_params

    def test_random_sweep_kcf(self):
        """Random sweep should evaluate exactly n_random configurations."""
        ds = _make_synthetic_dataset()
        cfg = SweepConfig(
            tracker_name="KCF",
            param_grid={"learning_rate": [0.05, 0.10, 0.15, 0.20]},
            metric="mean_iou",
            max_sequences=2,
        )
        sweep = TrackerHparamSweep(mode="random", n_random=2, seed=0)
        result = sweep.run(cfg, ds, "synthetic_test")

        assert len(result.entries) == 2

    def test_sweep_result_serialisable(self, tmp_path):
        """Sweep result must be saveable to JSON."""
        ds = _make_synthetic_dataset()
        cfg = SweepConfig(
            tracker_name="MOSSE",
            param_grid={"learning_rate": [0.10, 0.15]},
            max_sequences=1,
        )
        sweep = TrackerHparamSweep(mode="grid")
        result = sweep.run(cfg, ds, "synthetic")
        path = str(tmp_path / "out.json")
        result.save_json(path)
        with open(path) as f:
            data = json.load(f)
        assert data["tracker"] == "MOSSE"
        assert len(data["all_results"]) == 2

    def test_run_multi_returns_per_tracker(self):
        """run_multi should return one SweepResult per config."""
        ds = _make_synthetic_dataset()
        configs = [
            SweepConfig("MOSSE", {"learning_rate": [0.10]}, max_sequences=1),
            SweepConfig("KCF", {"learning_rate": [0.075]}, max_sequences=1),
        ]
        sweep = TrackerHparamSweep()
        results = sweep.run_multi(configs, ds, "synthetic")

        assert "MOSSE" in results
        assert "KCF" in results

    def test_comparison_table(self):
        """comparison_table should produce Markdown with all trackers."""
        results = {
            "MOSSE": SweepResult(
                tracker_name="MOSSE", metric="success_auc",
                best_params={"learning_rate": 0.10}, best_score=0.71, entries=[],
            ),
            "KCF": SweepResult(
                tracker_name="KCF", metric="success_auc",
                best_params={"learning_rate": 0.075}, best_score=0.68, entries=[],
            ),
        }
        sweep = TrackerHparamSweep()
        table = sweep.comparison_table(results)
        assert "MOSSE" in table
        assert "KCF" in table
        assert "0.7100" in table
