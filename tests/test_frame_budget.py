"""Tests for the frame budget simulator (eovot/simulation/frame_budget.py)."""

from __future__ import annotations

from typing import Iterator, List

import numpy as np
import pytest

from eovot.datasets.base import Sequence
from eovot.metrics.accuracy import AccuracyMetrics
from eovot.simulation.frame_budget import BudgetCurve, BudgetPoint, FrameBudgetSimulator
from eovot.trackers.base import BaseTracker

# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

BBox = tuple


class IdentityTracker(BaseTracker):
    """Always predicts the initialisation box — IoU = 1.0 on static GT."""

    def __init__(self) -> None:
        super().__init__(name="IdentityTracker")
        self._bbox: BBox = (0.0, 0.0, 1.0, 1.0)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> BBox:
        return self._bbox


class ShiftedTracker(BaseTracker):
    """Predicts a fixed box regardless of initialisation — used to test degradation."""

    def __init__(self, box: BBox = (100.0, 100.0, 50.0, 50.0)) -> None:
        super().__init__(name="ShiftedTracker")
        self._box = box

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: np.ndarray) -> BBox:
        return self._box


class StaticSequence(Sequence):
    """In-memory sequence with a fixed bounding box GT and black frames."""

    def __init__(self, n_frames: int = 30, box: BBox = (10.0, 10.0, 50.0, 50.0)) -> None:
        gt = np.tile(np.array(box, dtype=np.float64), (n_frames, 1))
        super().__init__(
            name="static_seq",
            frame_paths=[f"frame_{i:04d}.jpg" for i in range(n_frames)],
            ground_truth=gt,
        )
        self._n_frames = n_frames

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(self._n_frames):
            yield frame


# ---------------------------------------------------------------------------
# FrameBudgetSimulator — construction
# ---------------------------------------------------------------------------

class TestSimulatorConstruction:
    def test_default_rates_are_set(self):
        sim = FrameBudgetSimulator()
        assert sim.budget_rates == sorted(FrameBudgetSimulator.DEFAULT_RATES, reverse=True)

    def test_custom_rates_sorted_descending(self):
        sim = FrameBudgetSimulator(budget_rates=[0.1, 1.0, 0.5])
        assert sim.budget_rates == [1.0, 0.5, 0.1]

    def test_zero_rate_raises(self):
        with pytest.raises(ValueError, match="Invalid values"):
            FrameBudgetSimulator(budget_rates=[0.0, 0.5])

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="Invalid values"):
            FrameBudgetSimulator(budget_rates=[-0.1, 1.0])

    def test_rate_above_one_raises(self):
        with pytest.raises(ValueError, match="Invalid values"):
            FrameBudgetSimulator(budget_rates=[1.5])

    def test_rate_exactly_one_is_valid(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        assert sim.budget_rates == [1.0]


# ---------------------------------------------------------------------------
# _build_process_mask
# ---------------------------------------------------------------------------

class TestBuildProcessMask:
    def test_full_rate_processes_all(self):
        mask = FrameBudgetSimulator._build_process_mask(10, 1.0)
        assert mask.sum() == 10

    def test_half_rate_processes_half(self):
        mask = FrameBudgetSimulator._build_process_mask(10, 0.5)
        assert mask.sum() == 5

    def test_tenth_rate_at_least_one_frame(self):
        mask = FrameBudgetSimulator._build_process_mask(100, 0.01)
        assert mask.sum() >= 1

    def test_mask_length_matches_n_frames(self):
        for n in [5, 20, 100]:
            mask = FrameBudgetSimulator._build_process_mask(n, 0.5)
            assert len(mask) == n

    def test_empty_sequence_returns_empty_mask(self):
        mask = FrameBudgetSimulator._build_process_mask(0, 1.0)
        assert len(mask) == 0


# ---------------------------------------------------------------------------
# simulate — BudgetCurve structure
# ---------------------------------------------------------------------------

class TestSimulate:
    def setup_method(self):
        self.rates = [1.0, 0.5, 0.25]
        self.sim = FrameBudgetSimulator(budget_rates=self.rates)

    def test_returns_budget_curve(self):
        curve = self.sim.simulate(IdentityTracker(), StaticSequence())
        assert isinstance(curve, BudgetCurve)

    def test_correct_number_of_points(self):
        curve = self.sim.simulate(IdentityTracker(), StaticSequence())
        assert len(curve.points) == len(self.rates)

    def test_tracker_name_preserved(self):
        curve = self.sim.simulate(IdentityTracker(), StaticSequence())
        assert curve.tracker_name == "IdentityTracker"

    def test_sequence_name_preserved(self):
        curve = self.sim.simulate(IdentityTracker(), StaticSequence())
        assert curve.sequence_name == "static_seq"

    def test_native_fps_from_constructor(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0], native_fps=150.0)
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        assert curve.native_fps == pytest.approx(150.0)

    def test_native_fps_overridden_per_call(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0], native_fps=100.0)
        curve = sim.simulate(IdentityTracker(), StaticSequence(), native_fps=300.0)
        assert curve.native_fps == pytest.approx(300.0)

    def test_effective_fps_computed_correctly(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5], native_fps=200.0)
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        assert curve.points[0].effective_fps == pytest.approx(200.0)
        assert curve.points[1].effective_fps == pytest.approx(100.0)

    def test_points_ordered_highest_budget_first(self):
        curve = self.sim.simulate(IdentityTracker(), StaticSequence())
        rates = [p.budget_rate for p in curve.points]
        assert rates == sorted(rates, reverse=True)

    def test_frames_total_matches_sequence_length(self):
        seq = StaticSequence(n_frames=40)
        curve = self.sim.simulate(IdentityTracker(), seq)
        for p in curve.points:
            assert p.frames_total == 40

    def test_frames_processed_bounded_by_frames_total(self):
        curve = self.sim.simulate(IdentityTracker(), StaticSequence(n_frames=30))
        for p in curve.points:
            assert p.frames_processed <= p.frames_total

    def test_full_budget_processes_all_frames(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        seq = StaticSequence(n_frames=20)
        curve = sim.simulate(IdentityTracker(), seq)
        assert curve.points[0].frames_processed == 20


# ---------------------------------------------------------------------------
# simulate — accuracy correctness
# ---------------------------------------------------------------------------

class TestSimulateAccuracy:
    def test_identity_tracker_static_gt_iou_one(self):
        """IdentityTracker on static GT must achieve mIoU ≈ 1.0 at full budget."""
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        assert curve.points[0].accuracy.mean_iou == pytest.approx(1.0, abs=1e-6)

    def test_identity_tracker_static_gt_all_budgets(self):
        """Static GT: zero-motion propagation is perfect regardless of budget."""
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5, 0.1])
        curve = sim.simulate(IdentityTracker(), StaticSequence(n_frames=50))
        for pt in curve.points:
            assert pt.accuracy.mean_iou == pytest.approx(1.0, abs=1e-6)

    def test_shifted_tracker_lower_iou_than_identity(self):
        """A misaligned tracker must have strictly lower IoU than IdentityTracker."""
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        seq = StaticSequence()
        iou_id = sim.simulate(IdentityTracker(), seq).points[0].accuracy.mean_iou
        iou_sh = sim.simulate(ShiftedTracker(), seq).points[0].accuracy.mean_iou
        assert iou_sh < iou_id

    def test_accuracy_has_required_fields(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        acc = curve.points[0].accuracy
        assert isinstance(acc, AccuracyMetrics)
        assert hasattr(acc, "mean_iou")
        assert hasattr(acc, "success_auc")
        assert hasattr(acc, "precision_auc")

    def test_success_auc_in_unit_range(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        for pt in curve.points:
            assert 0.0 <= pt.accuracy.success_auc <= 1.0

    def test_precision_auc_in_unit_range(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        for pt in curve.points:
            assert 0.0 <= pt.accuracy.precision_auc <= 1.0


# ---------------------------------------------------------------------------
# Single-frame and very short sequences
# ---------------------------------------------------------------------------

class TestEdgeCaseLengths:
    def test_single_frame_sequence(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        curve = sim.simulate(IdentityTracker(), StaticSequence(n_frames=1))
        assert len(curve.points) == 1

    def test_two_frame_sequence(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence(n_frames=2))
        assert len(curve.points) == 2


# ---------------------------------------------------------------------------
# simulate_dataset
# ---------------------------------------------------------------------------

class TestSimulateDataset:
    def test_returns_one_curve_per_sequence(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        seqs = [StaticSequence(n_frames=20) for _ in range(3)]
        curves = sim.simulate_dataset(IdentityTracker(), seqs)
        assert len(curves) == 3

    def test_curves_are_budget_curve_instances(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        curves = sim.simulate_dataset(IdentityTracker(), [StaticSequence()])
        assert all(isinstance(c, BudgetCurve) for c in curves)


# ---------------------------------------------------------------------------
# aggregate_curves
# ---------------------------------------------------------------------------

class TestAggregateCurves:
    def test_empty_input_returns_empty(self):
        assert FrameBudgetSimulator.aggregate_curves([]) == {}

    def test_single_curve_returns_same_values(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        agg = FrameBudgetSimulator.aggregate_curves([curve])
        assert set(agg.keys()) == {1.0, 0.5}
        assert agg[1.0].mean_iou == pytest.approx(
            curve.points[0].accuracy.mean_iou, abs=1e-6
        )

    def test_multiple_curves_averaged(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        seqs = [StaticSequence(n_frames=20) for _ in range(5)]
        curves = sim.simulate_dataset(IdentityTracker(), seqs)
        agg = FrameBudgetSimulator.aggregate_curves(curves)
        assert agg[1.0].mean_iou == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_has_required_keys(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        d = curve.to_dict()
        for key in ("tracker_name", "sequence_name", "native_fps", "points"):
            assert key in d

    def test_to_dict_points_have_required_keys(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        for pt in curve.to_dict()["points"]:
            for key in (
                "budget_rate", "effective_fps", "mean_iou",
                "success_auc", "precision_auc",
                "frames_processed", "frames_total",
            ):
                assert key in pt

    def test_to_dict_is_json_serialisable(self):
        import json
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        json.dumps(curve.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# BudgetPoint helpers
# ---------------------------------------------------------------------------

class TestBudgetPointHelpers:
    def test_skip_ratio_complements_budget_rate(self):
        sim = FrameBudgetSimulator(budget_rates=[0.75])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        pt = curve.points[0]
        assert pt.skip_ratio == pytest.approx(0.25, abs=1e-9)

    def test_full_budget_skip_ratio_zero(self):
        sim = FrameBudgetSimulator(budget_rates=[1.0])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        assert curve.points[0].skip_ratio == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# print_curve smoke test
# ---------------------------------------------------------------------------

class TestPrintCurve:
    def test_print_curve_does_not_raise(self, capsys):
        sim = FrameBudgetSimulator(budget_rates=[1.0, 0.5])
        curve = sim.simulate(IdentityTracker(), StaticSequence())
        FrameBudgetSimulator.print_curve(curve)
        captured = capsys.readouterr()
        assert "IdentityTracker" in captured.out
        assert "static_seq" in captured.out
