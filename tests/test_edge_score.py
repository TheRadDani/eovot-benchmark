"""Tests for EdgeScorer, EdgeScoreWeights, and EdgeScoreResult."""

from __future__ import annotations

import pytest

from eovot.profiling.hardware_profiles import HardwareProfile, PROFILES
from eovot.metrics.edge_score import EdgeScorer, EdgeScoreWeights, EdgeScoreResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scorer(profile_key: str = "jetson_nano") -> EdgeScorer:
    return EdgeScorer(profile=PROFILES[profile_key])


# ---------------------------------------------------------------------------
# EdgeScoreWeights normalisation
# ---------------------------------------------------------------------------

def test_weights_normalise_to_one():
    w = EdgeScoreWeights(accuracy=2.0, speed=1.0, memory=1.0, energy=0.0).normalised()
    total = w.accuracy + w.speed + w.memory + w.energy
    assert total == pytest.approx(1.0)


def test_weights_zero_raises():
    with pytest.raises(ValueError):
        EdgeScoreWeights(accuracy=0.0, speed=0.0, memory=0.0, energy=0.0).normalised()


def test_weights_default_sum_to_one():
    w = EdgeScoreWeights().normalised()
    total = w.accuracy + w.speed + w.memory + w.energy
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EdgeScorer basic correctness
# ---------------------------------------------------------------------------

def test_composite_in_range():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=0.55, mean_fps=30.0, peak_memory_mb=500.0)
    assert 0.0 <= result.composite <= 1.0


def test_accuracy_score_clamped_high():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=2.0, mean_fps=30.0, peak_memory_mb=100.0)
    assert result.accuracy_score == pytest.approx(1.0)


def test_accuracy_score_clamped_low():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=-0.5, mean_fps=30.0, peak_memory_mb=100.0)
    assert result.accuracy_score == pytest.approx(0.0)


def test_speed_score_capped_at_one():
    scorer = make_scorer("raspberry_pi4")  # target_fps = 10
    result = scorer.compute(mean_iou=0.5, mean_fps=1000.0, peak_memory_mb=100.0)
    assert result.speed_score == pytest.approx(1.0)


def test_speed_score_below_target():
    scorer = make_scorer("raspberry_pi4")  # target_fps = 10
    result = scorer.compute(mean_iou=0.5, mean_fps=5.0, peak_memory_mb=100.0)
    assert result.speed_score == pytest.approx(0.5)


def test_memory_score_full_budget():
    scorer = make_scorer("raspberry_pi4")  # memory_limit = 4096
    result = scorer.compute(mean_iou=0.5, mean_fps=15.0, peak_memory_mb=0.0)
    assert result.memory_score == pytest.approx(1.0)


def test_memory_score_exceeds_limit():
    scorer = make_scorer("raspberry_pi4")  # memory_limit = 4096
    result = scorer.compute(mean_iou=0.5, mean_fps=15.0, peak_memory_mb=8000.0)
    assert result.memory_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fits_on_device
# ---------------------------------------------------------------------------

def test_fits_true_when_both_ok():
    scorer = make_scorer("raspberry_pi4")  # target_fps=10, memory=4096
    result = scorer.compute(mean_iou=0.5, mean_fps=15.0, peak_memory_mb=1000.0)
    assert result.fits_on_device is True


def test_fits_false_fps_too_low():
    scorer = make_scorer("raspberry_pi4")  # target_fps=10
    result = scorer.compute(mean_iou=0.5, mean_fps=5.0, peak_memory_mb=500.0)
    assert result.fits_on_device is False


def test_fits_false_memory_too_high():
    scorer = make_scorer("raspberry_pi4")  # memory=4096
    result = scorer.compute(mean_iou=0.5, mean_fps=20.0, peak_memory_mb=5000.0)
    assert result.fits_on_device is False


def test_fits_false_both():
    scorer = make_scorer("raspberry_pi4")
    result = scorer.compute(mean_iou=0.5, mean_fps=1.0, peak_memory_mb=9999.0)
    assert result.fits_on_device is False


# ---------------------------------------------------------------------------
# Energy sub-score
# ---------------------------------------------------------------------------

def test_energy_score_zero_when_none():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=0.5, mean_fps=30.0, peak_memory_mb=200.0, energy_per_frame_mj=None)
    assert result.energy_score == pytest.approx(0.0)


def test_energy_score_in_range():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=0.5, mean_fps=30.0, peak_memory_mb=200.0, energy_per_frame_mj=2.0)
    assert 0.0 <= result.energy_score <= 1.0


def test_energy_score_high_for_efficient_tracker():
    # Very low energy → high score
    scorer = make_scorer("raspberry_pi4")  # TDP=6W, target_fps=10 → budget=600mJ/frame
    result = scorer.compute(mean_iou=0.5, mean_fps=15.0, peak_memory_mb=200.0, energy_per_frame_mj=1.0)
    assert result.energy_score > 0.99  # 1 mJ out of 600 mJ budget


def test_composite_without_energy_is_valid():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=0.5, mean_fps=30.0, peak_memory_mb=200.0)
    assert 0.0 <= result.composite <= 1.0


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

def test_accuracy_only_weights():
    weights = EdgeScoreWeights(accuracy=1.0, speed=0.0, memory=0.0, energy=0.0)
    scorer = EdgeScorer(profile=PROFILES["x86_laptop"], weights=weights)
    result = scorer.compute(mean_iou=0.7, mean_fps=200.0, peak_memory_mb=500.0)
    assert result.composite == pytest.approx(result.accuracy_score, abs=1e-6)


def test_equal_weights():
    weights = EdgeScoreWeights(accuracy=0.25, speed=0.25, memory=0.25, energy=0.25)
    scorer = EdgeScorer(profile=PROFILES["jetson_nano"], weights=weights)
    result = scorer.compute(mean_iou=0.5, mean_fps=20.0, peak_memory_mb=1000.0, energy_per_frame_mj=5.0)
    assert 0.0 <= result.composite <= 1.0


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_keys():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=0.5, mean_fps=30.0, peak_memory_mb=200.0)
    d = result.to_dict()
    assert set(d.keys()) == {
        "hardware_profile", "composite", "accuracy_score",
        "speed_score", "memory_score", "energy_score", "fits_on_device",
    }


def test_to_dict_values_rounded():
    scorer = make_scorer()
    result = scorer.compute(mean_iou=0.123456789, mean_fps=30.0, peak_memory_mb=200.0)
    d = result.to_dict()
    # Values should be rounded to 4 decimal places
    assert len(str(d["composite"]).split(".")[-1]) <= 4


def test_str_output_format():
    scorer = make_scorer("jetson_nano")
    result = scorer.compute(mean_iou=0.5, mean_fps=25.0, peak_memory_mb=300.0)
    s = str(result)
    assert "Jetson Nano" in s
    assert "composite=" in s
    assert "fits=" in s


# ---------------------------------------------------------------------------
# Profile integration
# ---------------------------------------------------------------------------

def test_scores_differ_across_profiles():
    mosse_fps = 400.0
    mosse_iou = 0.45
    mosse_mem = 150.0

    score_pi = EdgeScorer(profile=PROFILES["raspberry_pi4"]).compute(
        mean_iou=mosse_iou, mean_fps=mosse_fps, peak_memory_mb=mosse_mem
    )
    score_desktop = EdgeScorer(profile=PROFILES["x86_desktop"]).compute(
        mean_iou=mosse_iou, mean_fps=mosse_fps, peak_memory_mb=mosse_mem
    )
    # On Raspberry Pi, MOSSE exceeds target_fps heavily → max speed score.
    # On desktop the speed bar is higher (60 fps target), but MOSSE still exceeds.
    # Scores should differ due to different memory and TDP baselines.
    assert score_pi.composite != pytest.approx(score_desktop.composite)
