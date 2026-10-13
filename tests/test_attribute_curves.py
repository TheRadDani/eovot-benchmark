"""Tests for AttributeCurveAnalyzer and AttributeCurveResult."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.metrics.attribute_curves import (
    AttributeCurve,
    AttributeCurveAnalyzer,
    AttributeCurveResult,
)
from eovot.metrics.attributes import ALL_ATTRIBUTES


# ---------------------------------------------------------------------------
# Minimal stubs — avoid importing the full benchmark engine in tests
# ---------------------------------------------------------------------------

def _make_ious(n: int = 50, mean: float = 0.6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, 0.15, n), 0.0, 1.0)


def _make_boxes(n: int = 50, seed: int = 0) -> np.ndarray:
    """Return (N, 4) float64 boxes in (x, y, w, h) format."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 200, n)
    y = rng.uniform(0, 150, n)
    w = rng.uniform(20, 60, n)
    h = rng.uniform(20, 60, n)
    return np.stack([x, y, w, h], axis=1)


class _FakeProf:
    fps = 100.0
    peak_memory_mb = 0.0


class _FakeSeqResult:
    def __init__(self, name, ious, preds, gts, attrs_dict):
        self.sequence_name = name
        self.ious = ious
        self.predictions = preds
        self.ground_truths = gts
        self._attrs = attrs_dict
        self.profiling = _FakeProf()
        self.accuracy_metrics = None

    @property
    def mean_iou(self):
        return float(self.ious.mean()) if len(self.ious) else 0.0


class _FakeBenchmarkResult:
    def __init__(self, seq_results, tracker_name="KCF", dataset_name="Test"):
        self.sequence_results = seq_results
        self.tracker_name = tracker_name
        self.dataset_name = dataset_name


# ---------------------------------------------------------------------------
# Helpers to build fake sequence attributes
# ---------------------------------------------------------------------------

from eovot.metrics.attributes import SequenceAttributes


def _seq_attrs(name: str, **attrs: bool) -> SequenceAttributes:
    full = {a: False for a in ALL_ATTRIBUTES}
    full.update(attrs)
    return SequenceAttributes(sequence_name=name, attributes=full)


# ---------------------------------------------------------------------------
# Tests: AttributeCurveAnalyzer.compute
# ---------------------------------------------------------------------------

def _build_fake_result(n_seqs: int = 6):
    """Build a fake BenchmarkResult with known attribute assignments."""
    seq_results = []
    for i in range(n_seqs):
        ious = _make_ious(n=80, seed=i)
        preds = _make_boxes(n=80, seed=i)
        gts = _make_boxes(n=80, seed=i + 100)
        seq_results.append(_FakeSeqResult(f"seq_{i:02d}", ious, preds, gts, {}))
    return _FakeBenchmarkResult(seq_results)


def test_compute_returns_curve_result():
    result = _build_fake_result()
    # Two sequences exhibit fast_motion, two exhibit scale_variation
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01", fast_motion=True),
        "seq_02": _seq_attrs("seq_02", scale_variation=True),
        "seq_03": _seq_attrs("seq_03", scale_variation=True),
        "seq_04": _seq_attrs("seq_04"),  # no attributes
        "seq_05": _seq_attrs("seq_05", fast_motion=True, scale_variation=True),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    assert isinstance(curves, AttributeCurveResult)
    assert curves.tracker_name == "KCF"
    assert curves.dataset_name == "Test"
    assert "fast_motion" in curves
    assert "scale_variation" in curves


def test_compute_correct_sequence_counts():
    result = _build_fake_result()
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01", fast_motion=True),
        "seq_02": _seq_attrs("seq_02", scale_variation=True),
        "seq_03": _seq_attrs("seq_03"),
        "seq_04": _seq_attrs("seq_04"),
        "seq_05": _seq_attrs("seq_05"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    assert curves["fast_motion"].n_sequences == 2
    assert curves["scale_variation"].n_sequences == 1


def test_compute_correct_frame_counts():
    result = _build_fake_result(n_seqs=4)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01", fast_motion=True),
        "seq_02": _seq_attrs("seq_02"),
        "seq_03": _seq_attrs("seq_03"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    # Each sequence has 80 frames → two fast_motion seqs → 160 frames.
    assert curves["fast_motion"].n_frames == 160


def test_compute_absent_attribute_not_in_result():
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01", fast_motion=True),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    assert "scale_variation" not in curves
    assert "low_resolution" not in curves


def test_compute_curve_shapes():
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    c = curves["fast_motion"]
    assert c.success_thresholds.shape == c.success_rates.shape
    assert c.precision_thresholds.shape == c.precision_rates.shape
    assert len(c.success_thresholds) == 101
    assert len(c.precision_thresholds) == 51


def test_compute_success_rates_in_01():
    result = _build_fake_result(n_seqs=3)
    seq_attrs = {f"seq_0{i}": _seq_attrs(f"seq_0{i}", fast_motion=True) for i in range(3)}
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    c = curves["fast_motion"]
    assert np.all(c.success_rates >= 0.0)
    assert np.all(c.success_rates <= 1.0)
    assert np.all(c.precision_rates >= 0.0)
    assert np.all(c.precision_rates <= 1.0)


def test_compute_success_rates_monotone_non_increasing():
    result = _build_fake_result(n_seqs=3)
    seq_attrs = {f"seq_0{i}": _seq_attrs(f"seq_0{i}", fast_motion=True) for i in range(3)}
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    c = curves["fast_motion"]
    # Success rates must be non-increasing as the IoU threshold rises.
    assert np.all(np.diff(c.success_rates) <= 1e-9)


def test_compute_auc_in_01():
    result = _build_fake_result()
    seq_attrs = {f"seq_0{i}": _seq_attrs(f"seq_0{i}", fast_motion=True) for i in range(6)}
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    c = curves["fast_motion"]
    assert 0.0 <= c.success_auc <= 1.0
    assert 0.0 <= c.precision_auc <= 1.0


def test_compute_skips_missing_predictions():
    """Sequences without predictions/GTs must be silently skipped."""
    result = _build_fake_result(n_seqs=3)
    # Null out predictions on seq_01
    result.sequence_results[1].predictions = None
    result.sequence_results[1].ground_truths = None

    seq_attrs = {f"seq_0{i}": _seq_attrs(f"seq_0{i}", fast_motion=True) for i in range(3)}
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    # Only seq_00 and seq_02 should contribute.
    assert curves["fast_motion"].n_sequences == 2
    assert curves["fast_motion"].n_frames == 160


def test_compute_empty_result():
    result = _FakeBenchmarkResult([], tracker_name="X", dataset_name="Y")
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes={})
    assert len(curves) == 0


def test_iteration_and_getitem():
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01", fast_motion=True, scale_variation=True),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    attrs_found = list(curves)
    assert "fast_motion" in attrs_found
    assert "scale_variation" in attrs_found

    fm = curves["fast_motion"]
    assert isinstance(fm, AttributeCurve)


# ---------------------------------------------------------------------------
# Tests: serialisation
# ---------------------------------------------------------------------------

def test_to_dict_structure():
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    d = curves.to_dict()
    assert d["tracker_name"] == "KCF"
    assert "curves" in d
    assert "fast_motion" in d["curves"]
    fm = d["curves"]["fast_motion"]
    assert "success_curve" in fm
    assert "precision_curve" in fm
    assert isinstance(fm["success_curve"]["thresholds"], list)


def test_save_json_round_trip(tmp_path):
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    json_path = curves.save_json(tmp_path / "curves.json")
    assert json_path.exists()

    with open(json_path) as fh:
        loaded = json.load(fh)

    assert loaded["tracker_name"] == "KCF"
    assert "fast_motion" in loaded["curves"]
    # Verify round-trip fidelity for one scalar
    orig_auc = curves["fast_motion"].success_auc
    loaded_auc = loaded["curves"]["fast_motion"]["success_auc"]
    assert abs(orig_auc - loaded_auc) < 1e-4


def test_save_csv_columns(tmp_path):
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    csv_path = curves.save_csv(tmp_path / "curves.csv")
    assert csv_path.exists()

    import csv
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 1  # only fast_motion present
    assert rows[0]["attribute"] == "fast_motion"
    assert float(rows[0]["success_auc"]) >= 0.0


def test_summary_table_contains_attr():
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01"),
    }
    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    table = curves.summary_table()
    assert "fast_motion" in table
    assert "Success AUC" in table


def test_summary_table_empty():
    curves = AttributeCurveResult(tracker_name="X", dataset_name="Y")
    table = curves.summary_table()
    assert "No attribute curves" in table


# ---------------------------------------------------------------------------
# Tests: custom thresholds
# ---------------------------------------------------------------------------

def test_custom_thresholds_respected():
    result = _build_fake_result(n_seqs=2)
    seq_attrs = {
        "seq_00": _seq_attrs("seq_00", fast_motion=True),
        "seq_01": _seq_attrs("seq_01", fast_motion=True),
    }
    s_thr = np.linspace(0, 1, 11)
    p_thr = np.linspace(0, 20, 21)
    analyzer = AttributeCurveAnalyzer(
        success_thresholds=s_thr, precision_thresholds=p_thr
    )
    curves = analyzer.compute(result, sequence_attributes=seq_attrs)

    c = curves["fast_motion"]
    assert len(c.success_thresholds) == 11
    assert len(c.precision_thresholds) == 21


# ---------------------------------------------------------------------------
# Integration: auto-detect attributes from BenchmarkEngine output
# ---------------------------------------------------------------------------

def test_integration_with_synthetic_dataset():
    """Run the full pipeline on a small synthetic dataset and check output."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.registry import build_tracker

    dataset = SyntheticDataset(
        num_sequences=6,
        num_frames=80,
        frame_size=(320, 240),
        bbox_size=(40, 40),
        motion="circular",
        seed=77,
    )
    tracker = build_tracker("KCF")
    engine = BenchmarkEngine(verbose=False)
    bench_result = engine.run(tracker, dataset, dataset_name="SynTest")

    analyzer = AttributeCurveAnalyzer()
    curves = analyzer.compute(bench_result)

    # At minimum the result should be valid (may have zero or more attributes).
    assert isinstance(curves, AttributeCurveResult)
    assert curves.tracker_name == "KCF"
    for attr, curve in curves.items():
        assert curve.n_sequences >= 1
        assert curve.n_frames >= curve.n_sequences
        assert 0.0 <= curve.success_auc <= 1.0
