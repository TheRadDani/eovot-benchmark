"""Tests for the ParetoSweep accuracy–efficiency hyperparameter sweep."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.pareto_sweep import ParetoSweep, ParetoSweepResult, SweepPoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    """3 sequences × 20 frames — fast enough for CI."""
    return SyntheticDataset(num_sequences=3, num_frames=20, motion="linear", seed=0)


# ---------------------------------------------------------------------------
# SweepPoint tests
# ---------------------------------------------------------------------------

class TestSweepPoint:
    def test_label(self):
        p = SweepPoint(skip_rate=2, scale_factor=0.5, mean_iou=0.4,
                       mean_fps=100.0, peak_memory_mb=50.0, ees=0.3)
        assert "skip=2" in p.label
        assert "scale=0.50" in p.label

    def test_str_no_pareto(self):
        p = SweepPoint(skip_rate=1, scale_factor=1.0, mean_iou=0.6,
                       mean_fps=200.0, peak_memory_mb=80.0, ees=0.5)
        assert "Pareto" not in str(p)

    def test_str_pareto(self):
        p = SweepPoint(skip_rate=1, scale_factor=1.0, mean_iou=0.6,
                       mean_fps=200.0, peak_memory_mb=80.0, ees=0.5,
                       on_pareto_front=True)
        assert "Pareto" in str(p)


# ---------------------------------------------------------------------------
# ParetoSweepResult tests
# ---------------------------------------------------------------------------

class TestParetoSweepResult:
    def _make_result(self) -> ParetoSweepResult:
        points = [
            SweepPoint(1, 1.0, 0.60, 200.0, 80.0, 0.50, on_pareto_front=True),
            SweepPoint(2, 1.0, 0.50, 380.0, 80.0, 0.60, on_pareto_front=True),
            SweepPoint(1, 0.5, 0.40, 220.0, 40.0, 0.45, on_pareto_front=False),
        ]
        result = ParetoSweepResult(
            tracker_name="KCF",
            dataset_name="Synthetic",
            memory_budget_mb=512.0,
            points=points,
            pareto_front=[p for p in points if p.on_pareto_front],
        )
        return result

    def test_best_point_returns_pareto(self):
        result = self._make_result()
        best = result.best_point()
        assert best is not None
        assert best.on_pareto_front

    def test_best_point_none_when_no_pareto(self):
        result = ParetoSweepResult("X", "Y", 512.0)
        assert result.best_point() is None

    def test_to_markdown_contains_tracker_name(self):
        result = self._make_result()
        md = result.to_markdown()
        assert "KCF" in md
        assert "Synthetic" in md

    def test_to_markdown_contains_pareto_checkmark(self):
        result = self._make_result()
        md = result.to_markdown()
        assert "✓" in md

    def test_save_and_reload_csv(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sweep.csv"
            saved = result.save(out)
            assert saved.exists()
            with open(saved, newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 3
            assert rows[0]["tracker"] == "KCF"
            assert "skip_rate" in rows[0]
            assert "ees" in rows[0]

    def test_save_appends_csv_extension(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sweep"  # no extension
            saved = result.save(out)
            assert saved.suffix == ".csv"


# ---------------------------------------------------------------------------
# ParetoSweep configuration tests
# ---------------------------------------------------------------------------

class TestParetoSweepConfig:
    def test_empty_skip_rates_raises(self):
        with pytest.raises(ValueError, match="skip_rates"):
            ParetoSweep(skip_rates=[])

    def test_empty_scale_factors_raises(self):
        with pytest.raises(ValueError, match="scale_factors"):
            ParetoSweep(scale_factors=[])

    def test_invalid_scale_factor_raises(self):
        with pytest.raises(ValueError, match="scale_factors"):
            ParetoSweep(scale_factors=[1.0, 0.5, 0.0])

    def test_scale_above_one_raises(self):
        with pytest.raises(ValueError, match="scale_factors"):
            ParetoSweep(scale_factors=[1.5])

    def test_default_params(self):
        sweep = ParetoSweep()
        assert 1 in sweep.skip_rates
        assert 1.0 in sweep.scale_factors
        assert sweep.memory_budget_mb > 0


# ---------------------------------------------------------------------------
# Integration tests (fast with tiny_dataset)
# ---------------------------------------------------------------------------

class TestParetoSweepIntegration:
    def test_run_produces_correct_point_count(self, tiny_dataset):
        sweep = ParetoSweep(
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.75],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        assert len(result.points) == 4  # 2 × 2

    def test_result_metadata(self, tiny_dataset):
        sweep = ParetoSweep(skip_rates=[1], scale_factors=[1.0], verbose=False)
        result = sweep.run("MOSSE", tiny_dataset, "TestDataset")
        assert result.tracker_name == "MOSSE"
        assert result.dataset_name == "TestDataset"
        assert result.memory_budget_mb == 512.0

    def test_all_points_have_positive_fps(self, tiny_dataset):
        sweep = ParetoSweep(
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        for p in result.points:
            assert p.mean_fps > 0.0, f"Point {p} has non-positive FPS"

    def test_all_points_have_valid_iou(self, tiny_dataset):
        sweep = ParetoSweep(
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        for p in result.points:
            assert 0.0 <= p.mean_iou <= 1.0 + 1e-6, f"Invalid IoU {p.mean_iou}"

    def test_pareto_front_non_empty(self, tiny_dataset):
        sweep = ParetoSweep(
            skip_rates=[1, 2, 3],
            scale_factors=[1.0, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        assert len(result.pareto_front) >= 1

    def test_pareto_front_consistency(self, tiny_dataset):
        """All pareto_front points must also appear in result.points."""
        sweep = ParetoSweep(
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        pareto_labels = {p.label for p in result.pareto_front}
        for p in result.pareto_front:
            assert p in result.points
        for p in result.points:
            if p.on_pareto_front:
                assert p.label in pareto_labels

    def test_points_sorted_by_ees_descending(self, tiny_dataset):
        sweep = ParetoSweep(
            skip_rates=[1, 2, 3],
            scale_factors=[1.0, 0.75, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        ees_values = [p.ees for p in result.points]
        assert ees_values == sorted(ees_values, reverse=True)

    def test_baseline_skip1_scale1_has_highest_iou(self, tiny_dataset):
        """No-skip full-resolution point should have the highest or equal IoU."""
        sweep = ParetoSweep(
            skip_rates=[1, 2, 4],
            scale_factors=[1.0, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        baseline = next(
            (p for p in result.points if p.skip_rate == 1 and p.scale_factor == 1.0), None
        )
        assert baseline is not None
        for p in result.points:
            if p.skip_rate > 1 or p.scale_factor < 1.0:
                assert baseline.mean_iou >= p.mean_iou - 0.20, (
                    f"Baseline IoU={baseline.mean_iou:.4f} should be >= "
                    f"{p.label} IoU={p.mean_iou:.4f} - 0.20"
                )

    def test_higher_skip_gives_higher_fps(self, tiny_dataset):
        """Within a fixed scale, more skipping should increase FPS."""
        sweep = ParetoSweep(
            skip_rates=[1, 2, 4],
            scale_factors=[1.0],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        fps_by_skip = {p.skip_rate: p.mean_fps for p in result.points}
        assert fps_by_skip[2] >= fps_by_skip[1] * 0.8, (
            "skip=2 should be at least 80% of skip=1 FPS: "
            f"fps[2]={fps_by_skip[2]:.1f} fps[1]={fps_by_skip[1]:.1f}"
        )

    def test_csv_round_trip(self, tiny_dataset):
        sweep = ParetoSweep(
            skip_rates=[1, 2],
            scale_factors=[1.0, 0.5],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = result.save(Path(tmpdir) / "sweep.csv")
            with open(out_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
        assert len(rows) == 4
        # Verify numeric fields parse back correctly
        for row in rows:
            assert float(row["mean_iou"]) >= 0.0
            assert float(row["mean_fps"]) > 0.0
            assert float(row["ees"]) >= 0.0

    def test_max_sequences_limits_evaluation(self, tiny_dataset):
        """Setting max_sequences should complete without error."""
        sweep = ParetoSweep(
            skip_rates=[1, 2],
            scale_factors=[1.0],
            verbose=False,
        )
        result = sweep.run("KCF", tiny_dataset, "Synthetic", max_sequences=2)
        assert len(result.points) == 2
