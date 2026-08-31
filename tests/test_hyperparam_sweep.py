"""Tests for HyperparamSweep."""

from __future__ import annotations

import pytest

from eovot.analysis.hyperparam_sweep import HyperparamSweep, SweepAxis, SweepResult
from eovot.datasets.synthetic import SyntheticDataset


# ---------------------------------------------------------------------------
# Shared fixture — fast synthetic dataset (3 sequences, 30 frames each)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast_dataset() -> SyntheticDataset:
    return SyntheticDataset(
        num_sequences=3,
        num_frames=30,
        frame_size=(320, 240),
        bbox_size=(30, 30),
        motion="linear",
        seed=42,
    )


# ---------------------------------------------------------------------------
# SweepAxis tests
# ---------------------------------------------------------------------------

class TestSweepAxis:
    def test_label_falls_back_to_param_name(self):
        ax = SweepAxis("learning_rate", [0.1, 0.2])
        assert ax.label() == "learning_rate"

    def test_display_name_overrides_param_name(self):
        ax = SweepAxis("learning_rate", [0.1, 0.2], display_name="LR")
        assert ax.label() == "LR"


# ---------------------------------------------------------------------------
# SweepResult tests
# ---------------------------------------------------------------------------

class TestSweepResult:
    def _make_result(self, aucs: list[float]) -> SweepResult:
        from eovot.analysis.hyperparam_sweep import SweepPoint
        axis = SweepAxis("lr", list(range(len(aucs))))
        points = [
            SweepPoint(
                config={"lr": i},
                mean_iou=auc,
                success_auc=auc,
                precision_auc=auc,
                fps=100.0,
                peak_memory_mb=64.0,
            )
            for i, auc in enumerate(aucs)
        ]
        return SweepResult(tracker_name="TEST", axes=[axis], points=points)

    def test_optimal_config_picks_highest_auc(self):
        result = self._make_result([0.3, 0.7, 0.5])
        assert result.optimal_config["lr"] == 1

    def test_sensitivity_zero_for_constant_auc(self):
        result = self._make_result([0.5, 0.5, 0.5])
        sens = result.sensitivity
        assert abs(sens["lr"]) < 1e-9

    def test_sensitivity_positive_for_varying_auc(self):
        result = self._make_result([0.1, 0.9])
        sens = result.sensitivity
        assert sens["lr"] > 0.0

    def test_to_markdown_1d_contains_expected_sections(self):
        result = self._make_result([0.5, 0.6, 0.7])
        md = result.to_markdown()
        assert "Sensitivity" in md
        assert "Optimal config" in md
        assert "1D Sweep" in md

    def test_empty_points_returns_zero_sensitivity(self):
        axis = SweepAxis("lr", [])
        result = SweepResult(tracker_name="T", axes=[axis], points=[])
        sens = result.sensitivity
        assert sens["lr"] == 0.0


# ---------------------------------------------------------------------------
# HyperparamSweep end-to-end tests (use fast synthetic dataset)
# ---------------------------------------------------------------------------

class TestHyperparamSweep:
    def test_1d_sweep_returns_correct_point_count(self, fast_dataset):
        values = [0.05, 0.125, 0.2]
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            dataset_name="Synthetic",
            max_sequences=2,
            verbose=False,
        )
        result = sweep.run_1d(SweepAxis("learning_rate", values))
        assert len(result.points) == len(values)

    def test_1d_sweep_optimal_config_key_present(self, fast_dataset):
        # Use KCF — it is numerically stable across the full synthetic frame boundary range.
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            dataset_name="Synthetic",
            max_sequences=2,
        )
        result = sweep.run_1d(SweepAxis("learning_rate", [0.05, 0.125, 0.2]))
        assert "learning_rate" in result.optimal_config

    def test_2d_sweep_returns_correct_point_count(self, fast_dataset):
        ax0 = SweepAxis("learning_rate", [0.05, 0.125])
        ax1 = SweepAxis("padding", [1.0, 1.5])
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            dataset_name="Synthetic",
            max_sequences=2,
        )
        result = sweep.run_2d(ax0, ax1)
        assert len(result.points) == len(ax0.values) * len(ax1.values)

    def test_2d_sweep_sensitivity_has_both_axes(self, fast_dataset):
        ax0 = SweepAxis("learning_rate", [0.05, 0.2])
        ax1 = SweepAxis("padding", [1.0, 2.0])
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            dataset_name="Synthetic",
            max_sequences=2,
        )
        result = sweep.run_2d(ax0, ax1)
        sens = result.sensitivity
        assert "learning_rate" in sens
        assert "padding" in sens

    def test_2d_sweep_markdown_contains_grid_table(self, fast_dataset):
        ax0 = SweepAxis("learning_rate", [0.05, 0.125])
        ax1 = SweepAxis("padding", [1.0, 1.5])
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            dataset_name="Synthetic",
            max_sequences=2,
        )
        result = sweep.run_2d(ax0, ax1)
        md = result.to_markdown()
        assert "2D Sweep" in md
        assert "learning_rate" in md
        assert "padding" in md

    def test_empty_axis_raises(self, fast_dataset):
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            max_sequences=2,
        )
        with pytest.raises(ValueError, match="no values"):
            sweep.run_1d(SweepAxis("learning_rate", []))

    def test_fixed_params_are_forwarded(self, fast_dataset):
        """fixed_params should not interfere with the sweep axis."""
        sweep = HyperparamSweep(
            tracker_name="KCF",
            dataset=fast_dataset,
            dataset_name="Synthetic",
            max_sequences=2,
            fixed_params={"lambda_": 1e-4},
        )
        result = sweep.run_1d(SweepAxis("learning_rate", [0.075, 0.15]))
        # Both configs should include the fixed param internally
        assert len(result.points) == 2
