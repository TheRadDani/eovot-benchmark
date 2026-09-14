"""Tests for the hyperparameter sweep engine.

Covers:
  - SweepEntry and SweepReport dataclass behaviour
  - Pareto-front marking (_mark_pareto, _dominates)
  - HyperparamSweeper: correct number of trials, param labels, ranking
  - SweepReport.best_config / best_at_fps_budget / pareto_front queries
  - to_markdown / to_csv / to_dict / save round-trip
  - CLI: scripts/sweep_tracker.py via main()
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from eovot.analysis.hyperparam_sweep import (
    HyperparamSweeper,
    SweepEntry,
    SweepReport,
    _dominates,
    _mark_pareto,
)
from eovot.benchmark.engine import BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(success_auc: float, fps: float, params: dict = None) -> SweepEntry:
    """Create a minimal SweepEntry for unit tests."""
    mock_result = MagicMock(spec=BenchmarkResult)
    return SweepEntry(
        params=params or {},
        result=mock_result,
        mean_iou=success_auc * 0.9,
        success_auc=success_auc,
        mean_fps=fps,
        peak_memory_mb=50.0,
    )


def _small_dataset(n_seq=2, n_frames=20):
    return SyntheticDataset(
        num_sequences=n_seq,
        num_frames=n_frames,
        frame_size=(160, 120),
        seed=0,
    )


# ---------------------------------------------------------------------------
# SweepEntry unit tests
# ---------------------------------------------------------------------------

class TestSweepEntry:
    def test_params_label_single(self):
        e = _make_entry(0.5, 100, {"learning_rate": 0.1})
        assert "learning_rate=0.1" in e.params_label()

    def test_params_label_multi(self):
        e = _make_entry(0.5, 100, {"a": 1, "b": 2})
        label = e.params_label()
        assert "a=1" in label
        assert "b=2" in label

    def test_params_label_empty(self):
        e = _make_entry(0.5, 100, {})
        assert e.params_label() == "(default)"

    def test_to_dict_keys(self):
        e = _make_entry(0.65, 200, {"lr": 0.1})
        d = e.to_dict()
        assert set(d.keys()) == {"params", "mean_iou", "success_auc", "mean_fps",
                                  "peak_memory_mb", "is_pareto"}
        assert d["success_auc"] == pytest.approx(0.65, rel=1e-3)


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

class TestParetoHelpers:
    def test_dominates_strictly_better_both(self):
        a = _make_entry(0.8, 300)
        b = _make_entry(0.6, 200)
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_dominates_equal_does_not_dominate(self):
        a = _make_entry(0.7, 250)
        b = _make_entry(0.7, 250)
        assert not _dominates(a, b)

    def test_dominates_better_accuracy_worse_fps(self):
        a = _make_entry(0.9, 100)
        b = _make_entry(0.6, 400)
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_mark_pareto_single_entry(self):
        entries = [_make_entry(0.7, 200)]
        _mark_pareto(entries)
        assert entries[0].is_pareto

    def test_mark_pareto_clear_domination(self):
        dominated = _make_entry(0.5, 100)
        dominant = _make_entry(0.8, 300)
        entries = [dominated, dominant]
        _mark_pareto(entries)
        assert dominant.is_pareto
        assert not dominated.is_pareto

    def test_mark_pareto_trade_off_both_pareto(self):
        # High accuracy, low fps vs low accuracy, high fps — neither dominates
        e1 = _make_entry(0.9, 100)
        e2 = _make_entry(0.5, 500)
        entries = [e1, e2]
        _mark_pareto(entries)
        assert e1.is_pareto
        assert e2.is_pareto

    def test_mark_pareto_dominated_entry_removed(self):
        # e3 dominates e1 and e2
        e1 = _make_entry(0.5, 100)
        e2 = _make_entry(0.6, 150)
        e3 = _make_entry(0.8, 400)
        entries = [e1, e2, e3]
        _mark_pareto(entries)
        assert e3.is_pareto
        assert not e1.is_pareto
        assert not e2.is_pareto


# ---------------------------------------------------------------------------
# SweepReport unit tests
# ---------------------------------------------------------------------------

class TestSweepReport:
    def _make_report(self) -> SweepReport:
        entries = [
            _make_entry(0.80, 500, {"lr": 0.05}),
            _make_entry(0.75, 600, {"lr": 0.10}),
            _make_entry(0.65, 700, {"lr": 0.15}),
        ]
        _mark_pareto(entries)
        return SweepReport(
            tracker_name="KCF",
            dataset_name="Synthetic",
            param_names=["lr"],
            entries=sorted(entries, key=lambda e: e.success_auc, reverse=True),
        )

    def test_best_config_highest_auc(self):
        r = self._make_report()
        best = r.best_config()
        assert best is not None
        assert best.success_auc == pytest.approx(0.80)

    def test_best_at_fps_budget_filters(self):
        r = self._make_report()
        # lr=0.05 has 0.80 AUC at 500 fps; should be excluded at fps_min=550
        best = r.best_at_fps_budget(fps_min=550)
        assert best is not None
        assert best.mean_fps >= 550

    def test_best_at_fps_budget_none_when_none_qualify(self):
        r = self._make_report()
        assert r.best_at_fps_budget(fps_min=9999) is None

    def test_pareto_front_returns_subset(self):
        r = self._make_report()
        front = r.pareto_front()
        assert len(front) >= 1
        for e in front:
            assert e.is_pareto

    def test_to_markdown_contains_header(self):
        r = self._make_report()
        md = r.to_markdown()
        assert "# Hyperparameter Sweep" in md
        assert "KCF" in md
        assert "Success AUC" in md

    def test_to_markdown_rows_count(self):
        r = self._make_report()
        lines = [l for l in r.to_markdown().splitlines() if l.startswith("|") and "Rank" not in l and "---" not in l]
        assert len(lines) == 3

    def test_to_csv_parseable(self):
        r = self._make_report()
        reader = csv.DictReader(io.StringIO(r.to_csv()))
        rows = list(reader)
        assert len(rows) == 3
        assert "success_auc" in rows[0]

    def test_to_dict_structure(self):
        r = self._make_report()
        d = r.to_dict()
        assert d["tracker_name"] == "KCF"
        assert "entries" in d
        assert len(d["entries"]) == 3

    def test_save_and_load(self, tmp_path):
        r = self._make_report()
        p = r.save(tmp_path / "sweep")
        assert p.exists()
        with open(p) as f:
            d = json.load(f)
        assert d["tracker_name"] == "KCF"

    def test_save_appends_json_extension(self, tmp_path):
        r = self._make_report()
        p = r.save(tmp_path / "no_ext")
        assert p.suffix == ".json"

    def test_empty_report_markdown(self):
        r = SweepReport("X", "Y", [])
        assert "No results" in r.to_markdown()

    def test_empty_report_best_config_none(self):
        r = SweepReport("X", "Y", [])
        assert r.best_config() is None

    def test_pareto_mark_in_markdown(self):
        entries = [_make_entry(0.9, 100), _make_entry(0.5, 500)]
        _mark_pareto(entries)
        r = SweepReport("T", "D", [], entries=entries)
        md = r.to_markdown()
        assert "✓" in md


# ---------------------------------------------------------------------------
# HyperparamSweeper integration tests
# ---------------------------------------------------------------------------

class TestHyperparamSweeper:
    def _sweeper(self, tracker="MOSSE"):
        ds = _small_dataset()
        return HyperparamSweeper(
            tracker_name=tracker,
            dataset=ds,
            dataset_name="Synthetic",
            verbose=False,
        )

    def test_sweep_returns_correct_trial_count(self):
        sweeper = self._sweeper("KCF")
        report = sweeper.sweep({"learning_rate": [0.05, 0.10, 0.15]}, max_sequences=1)
        assert len(report.entries) == 3

    def test_sweep_entries_sorted_by_auc_desc(self):
        sweeper = self._sweeper("KCF")
        report = sweeper.sweep({"learning_rate": [0.05, 0.10, 0.15]}, max_sequences=1)
        aucs = [e.success_auc for e in report.entries]
        assert aucs == sorted(aucs, reverse=True)

    def test_sweep_tracker_name_in_report(self):
        sweeper = self._sweeper("MOSSE")
        report = sweeper.sweep({"learning_rate": [0.1, 0.2]}, max_sequences=1)
        assert report.tracker_name == "MOSSE"

    def test_sweep_dataset_name_in_report(self):
        sweeper = self._sweeper()
        report = sweeper.sweep({"learning_rate": [0.1]}, max_sequences=1)
        assert report.dataset_name == "Synthetic"

    def test_sweep_2d_grid_correct_count(self):
        ds = _small_dataset()
        sweeper = HyperparamSweeper("KCF", ds, verbose=False)
        report = sweeper.sweep(
            {"learning_rate": [0.05, 0.10], "kernel_sigma": [0.3, 0.5]},
            max_sequences=1,
        )
        # 2 × 2 = 4 combinations
        assert len(report.entries) == 4

    def test_sweep_pareto_flags_set(self):
        sweeper = self._sweeper("KCF")
        report = sweeper.sweep({"learning_rate": [0.05, 0.10, 0.15]}, max_sequences=1)
        pareto = report.pareto_front()
        assert len(pareto) >= 1

    def test_sweep_param_names_recorded(self):
        sweeper = self._sweeper("KCF")
        report = sweeper.sweep({"learning_rate": [0.1]}, max_sequences=1)
        assert "learning_rate" in report.param_names

    def test_sweep_entries_have_positive_fps(self):
        sweeper = self._sweeper("MOSSE")
        report = sweeper.sweep({"learning_rate": [0.1]}, max_sequences=1)
        for e in report.entries:
            assert e.mean_fps > 0.0

    def test_best_at_fps_budget_works_after_sweep(self):
        sweeper = self._sweeper("KCF")
        report = sweeper.sweep({"learning_rate": [0.05, 0.10, 0.15]}, max_sequences=2)
        fps_values = [e.mean_fps for e in report.entries]
        # A budget above max FPS means no config qualifies
        assert report.best_at_fps_budget(max(fps_values) * 2) is None
        # A budget below min FPS means all configs qualify
        result = report.best_at_fps_budget(0.0)
        assert result is not None

    def test_invalid_tracker_name_produces_empty_report(self):
        """An unknown tracker name causes all trials to be skipped gracefully."""
        ds = _small_dataset()
        report = HyperparamSweeper("NONEXISTENT", ds, verbose=False).sweep(
            {"lr": [0.1]}, max_sequences=1
        )
        assert len(report.entries) == 0
        assert report.best_config() is None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_runs_with_synthetic_config(self, tmp_path):
        """CLI produces Markdown, CSV, and JSON outputs without errors."""
        import yaml

        config = {
            "tracker": {"name": "KCF", "base_params": {}},
            "sweep": {"param_grid": {"learning_rate": [0.05, 0.15]}},
            "dataset": {
                "loader": "SyntheticDataset",
                "name": "SyntheticTest",
                "num_sequences": 2,
                "num_frames": 20,
                "frame_size": [160, 120],
                "motion": "linear",
                "seed": 1,
            },
            "profiling": {"tdp_watts": None},
        }
        cfg_path = tmp_path / "sweep.yaml"
        cfg_path.write_text(yaml.dump(config), encoding="utf-8")

        from scripts.sweep_tracker import main

        ret = main([
            "--config", str(cfg_path),
            "--output-dir", str(tmp_path / "out"),
            "--quiet",
        ])
        assert ret == 0

        out_dir = tmp_path / "out" / "KCF-SyntheticTest"
        assert (out_dir / "sweep_report.md").exists()
        assert (out_dir / "sweep_results.csv").exists()
        assert (out_dir / "sweep_results.json").exists()

    def test_cli_missing_config_returns_nonzero(self, tmp_path):
        from scripts.sweep_tracker import main

        ret = main(["--config", str(tmp_path / "nonexistent.yaml"), "--quiet"])
        assert ret != 0
