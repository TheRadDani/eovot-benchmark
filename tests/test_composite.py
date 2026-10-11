"""Tests for eovot.metrics.composite — Composite Deployability Score."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eovot.metrics.composite import CDSEntry, CompositeDeployabilityScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result_dict(
    tracker: str,
    dataset: str = "synth",
    mean_iou: float = 0.5,
    fps: float = 50.0,
    memory_mb: float = 200.0,
    success_auc: float = 0.5,
    num_sequences: int = 3,
    seq_ious: list | None = None,
) -> dict:
    """Build a minimal BenchmarkResult dict for testing."""
    if seq_ious is None:
        seq_ious = [[0.6, 0.5, 0.4, 0.3, 0.2]] * num_sequences
    sequences = [
        {
            "sequence_name": f"seq{i}",
            "mean_iou": float(np.mean(iou_arr)),
            "fps": fps,
            "mean_latency_ms": 1000.0 / fps,
            "latency_std_ms": 0.5,
            "latency_p95_ms": 1000.0 / fps * 1.2,
            "latency_p99_ms": 1000.0 / fps * 1.5,
            "latency_cv": 0.02,
            "peak_memory_mb": memory_mb,
            "ious": iou_arr,
            "success_auc": success_auc,
            "precision_auc": 0.45,
            "normalized_precision_auc": 0.40,
        }
        for i, iou_arr in enumerate(seq_ious)
    ]
    return {
        "summary": {
            "tracker": tracker,
            "dataset": dataset,
            "mean_iou": mean_iou,
            "mean_fps": fps,
            "peak_memory_mb": memory_mb,
            "num_sequences": num_sequences,
            "success_auc": success_auc,
            "precision_auc": 0.45,
        },
        "sequences": sequences,
    }


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="1.0"):
        CompositeDeployabilityScore(w_acc=0.5, w_rob=0.5, w_eff=0.5, w_smooth=0.1)


def test_negative_memory_budget_raises():
    with pytest.raises(ValueError, match="positive"):
        CompositeDeployabilityScore(memory_budget_mb=-1.0)


def test_default_weights_sum_to_one():
    engine = CompositeDeployabilityScore()
    total = engine.w_acc + engine.w_rob + engine.w_eff + engine.w_smooth
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# rank_from_dicts — basic correctness
# ---------------------------------------------------------------------------

def test_rank_from_dicts_returns_sorted_by_cds():
    engine = CompositeDeployabilityScore()
    low = _make_result_dict("SlowTracker", fps=5.0, mean_iou=0.2, success_auc=0.2)
    high = _make_result_dict("FastTracker", fps=100.0, mean_iou=0.8, success_auc=0.8)
    entries = engine.rank_from_dicts([low, high])
    assert len(entries) == 2
    assert entries[0].tracker_name == "FastTracker"
    assert entries[0].cds > entries[1].cds


def test_rank_from_dicts_cds_in_unit_interval():
    engine = CompositeDeployabilityScore()
    dicts = [
        _make_result_dict("A", fps=30.0, mean_iou=0.6),
        _make_result_dict("B", fps=10.0, mean_iou=0.3),
    ]
    entries = engine.rank_from_dicts(dicts)
    for e in entries:
        assert 0.0 <= e.cds <= 1.0 + 1e-9


def test_rank_from_dicts_single_tracker():
    engine = CompositeDeployabilityScore()
    entries = engine.rank_from_dicts([_make_result_dict("Solo")])
    assert len(entries) == 1
    assert entries[0].tracker_name == "Solo"
    # Only one tracker → EES normalised to 1.0 (min==max branch)
    assert entries[0].ees_normalized == pytest.approx(1.0, abs=1e-6)


def test_rank_from_dicts_empty():
    engine = CompositeDeployabilityScore()
    assert engine.rank_from_dicts([]) == []


# ---------------------------------------------------------------------------
# ees_normalized is MinMax across the comparison set
# ---------------------------------------------------------------------------

def test_ees_normalized_range():
    engine = CompositeDeployabilityScore()
    low = _make_result_dict("Slow", fps=2.0, mean_iou=0.1)
    high = _make_result_dict("Fast", fps=200.0, mean_iou=0.9)
    entries = engine.rank_from_dicts([low, high])
    ees_values = [e.ees_normalized for e in entries]
    # Highest EES tracker should have ees_normalized == 1.0
    assert max(ees_values) == pytest.approx(1.0, abs=1e-6)
    # Lowest EES tracker should have ees_normalized == 0.0
    assert min(ees_values) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Robustness integration
# ---------------------------------------------------------------------------

def test_failures_detected_from_low_iou_sequences():
    engine = CompositeDeployabilityScore(failure_threshold=0.1)
    # Sequence with lots of zeros → many failures
    bad_seq = [1.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0]
    d = _make_result_dict("BadTracker", seq_ious=[bad_seq])
    entries = engine.rank_from_dicts([d])
    assert entries[0].num_failures > 0


def test_eao_near_one_for_perfect_ious():
    engine = CompositeDeployabilityScore()
    perfect = [1.0] * 50
    d = _make_result_dict("Perfect", seq_ious=[perfect], mean_iou=1.0, success_auc=1.0)
    entries = engine.rank_from_dicts([d])
    # EAO should be close to 1.0 with uniform-1.0 IoUs
    assert entries[0].eao >= 0.95


# ---------------------------------------------------------------------------
# has_temporal flag
# ---------------------------------------------------------------------------

def test_rank_from_dicts_has_temporal_false():
    engine = CompositeDeployabilityScore()
    entries = engine.rank_from_dicts([_make_result_dict("T")])
    assert entries[0].has_temporal is False


# ---------------------------------------------------------------------------
# Weight redistribution without temporal
# ---------------------------------------------------------------------------

def test_weight_redistribution_without_temporal():
    """CDS value computed with explicit 3D weights should match rank_from_dicts."""
    w_acc, w_rob, w_eff = 0.40, 0.20, 0.30
    total = w_acc + w_rob + w_eff
    w_acc_r = w_acc / total
    w_rob_r = w_rob / total
    w_eff_r = w_eff / total

    engine = CompositeDeployabilityScore(w_acc=w_acc, w_rob=w_rob, w_eff=w_eff, w_smooth=0.10)
    dicts = [
        _make_result_dict("A", fps=50.0, mean_iou=0.7, success_auc=0.7),
        _make_result_dict("B", fps=20.0, mean_iou=0.4, success_auc=0.4),
    ]
    entries = engine.rank_from_dicts(dicts)
    # Check that CDS is derived from 3 dimensions only
    for e in entries:
        # Smoothness is 1.0 (placeholder) and w_smooth should be 0.0 in the formula
        manual = (
            w_acc_r * e.success_auc
            + w_rob_r * e.eao
            + w_eff_r * e.ees_normalized
        )
        assert e.cds == pytest.approx(manual, abs=1e-4)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def test_to_markdown_table_contains_tracker_names():
    engine = CompositeDeployabilityScore()
    dicts = [_make_result_dict("Alpha"), _make_result_dict("Beta", fps=200.0, mean_iou=0.9)]
    entries = engine.rank_from_dicts(dicts)
    table = engine.to_markdown_table(entries)
    assert "Alpha" in table
    assert "Beta" in table
    assert "CDS" in table


def test_to_markdown_table_empty():
    engine = CompositeDeployabilityScore()
    table = engine.to_markdown_table([])
    # Header rows should still be present even with no data rows
    assert "Rank" in table


# ---------------------------------------------------------------------------
# rank() with synthetic BenchmarkResult objects
# ---------------------------------------------------------------------------

def test_rank_with_benchmark_result():
    """End-to-end test using SyntheticDataset and BenchmarkEngine."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
    engine = BenchmarkEngine(verbose=False)

    tracker_a = build_tracker("MOSSE")
    tracker_b = build_tracker("KCF")

    result_a = engine.run(tracker_a, dataset, dataset_name="synth", max_sequences=2)
    result_b = engine.run(tracker_b, dataset, dataset_name="synth", max_sequences=2)

    cds_engine = CompositeDeployabilityScore()
    entries = cds_engine.rank([result_a, result_b])

    assert len(entries) == 2
    assert entries[0].cds >= entries[1].cds
    # Predictions are stored by BenchmarkEngine → temporal should be available
    assert any(e.has_temporal for e in entries)
