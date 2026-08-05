"""Tests for eovot.metrics.bias — systematic prediction bias analysis."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.bias import BiasAnalyzer, PredictionBiasResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _static_gt(n: int = 30, x: float = 50, y: float = 50,
               w: float = 40, h: float = 30) -> np.ndarray:
    """Ground-truth array with a stationary box."""
    gt = np.zeros((n, 4), dtype=np.float64)
    gt[:, 0] = x
    gt[:, 1] = y
    gt[:, 2] = w
    gt[:, 3] = h
    return gt


# ---------------------------------------------------------------------------
# Neutral (unbiased) tracker
# ---------------------------------------------------------------------------

def test_unbiased_tracker_size_bias():
    gt = _static_gt()
    result = BiasAnalyzer().analyze(gt.copy(), gt, tracker_name="P", sequence_name="s")
    assert abs(result.size_bias - 1.0) < 1e-6


def test_unbiased_tracker_width_height_bias():
    gt = _static_gt()
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert abs(result.width_bias - 1.0) < 1e-6
    assert abs(result.height_bias - 1.0) < 1e-6


def test_unbiased_tracker_position_offset():
    gt = _static_gt()
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert abs(result.x_offset_bias) < 1e-6
    assert abs(result.y_offset_bias) < 1e-6


def test_unbiased_tracker_ar_deviation():
    gt = _static_gt()
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert abs(result.aspect_ratio_deviation) < 1e-6


def test_unbiased_tracker_drift_slope():
    gt = _static_gt()
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert abs(result.temporal_drift_slope) < 1e-6


# ---------------------------------------------------------------------------
# Size bias
# ---------------------------------------------------------------------------

def test_double_width_size_bias():
    gt = _static_gt(w=40, h=30)
    pred = gt.copy()
    pred[:, 2] *= 2.0   # width doubled → area 2×
    result = BiasAnalyzer().analyze(pred, gt)
    assert abs(result.size_bias - 2.0) < 1e-4
    assert abs(result.width_bias - 2.0) < 1e-4
    assert abs(result.height_bias - 1.0) < 1e-4


def test_area_quadrupled():
    gt = _static_gt(w=40, h=30)
    pred = gt.copy()
    pred[:, 2] *= 2.0
    pred[:, 3] *= 2.0
    result = BiasAnalyzer().analyze(pred, gt)
    assert abs(result.size_bias - 4.0) < 1e-4


def test_under_sizing():
    gt = _static_gt(w=60, h=60)
    pred = gt.copy()
    pred[:, 2] *= 0.5
    pred[:, 3] *= 0.5
    result = BiasAnalyzer().analyze(pred, gt)
    assert result.size_bias < 1.0
    assert abs(result.size_bias - 0.25) < 1e-4


# ---------------------------------------------------------------------------
# Positional offset bias
# ---------------------------------------------------------------------------

def test_positive_x_drift():
    gt = _static_gt(n=20, x=100, y=50, w=40, h=40)
    pred = gt.copy()
    pred[:, 0] += 10.0   # shift right by 10 px
    result = BiasAnalyzer().analyze(pred, gt)
    assert result.x_offset_bias > 0.0
    assert abs(result.y_offset_bias) < 1e-6


def test_negative_y_drift():
    gt = _static_gt(n=20, x=100, y=100, w=40, h=40)
    pred = gt.copy()
    pred[:, 1] -= 10.0   # shift upward
    result = BiasAnalyzer().analyze(pred, gt)
    assert result.y_offset_bias < 0.0
    assert abs(result.x_offset_bias) < 1e-6


# ---------------------------------------------------------------------------
# Aspect ratio deviation
# ---------------------------------------------------------------------------

def test_stretched_predictions_ar_deviation():
    gt = _static_gt(w=40, h=40)   # AR = 1.0
    pred = gt.copy()
    pred[:, 2] = 80.0              # AR = 2.0 → deviation = 1.0
    result = BiasAnalyzer().analyze(pred, gt)
    assert result.aspect_ratio_deviation > 0.5


def test_perfect_ar_is_zero():
    gt = _static_gt()
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert result.aspect_ratio_deviation == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Temporal drift slope
# ---------------------------------------------------------------------------

def test_degrading_tracker_has_negative_slope():
    """Predictions that shift progressively rightward → IoU decreases → slope < 0."""
    n = 60
    gt = _static_gt(n=n, x=50, y=50, w=50, h=50)
    pred = gt.copy()
    for i in range(n):
        pred[i, 0] += i * 2.0   # shifts right more each frame
    result = BiasAnalyzer().analyze(pred, gt)
    assert result.temporal_drift_slope < 0.0


def test_stable_tracker_near_zero_slope():
    gt = _static_gt(n=50)
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert abs(result.temporal_drift_slope) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_arrays_return_neutral():
    result = BiasAnalyzer().analyze(
        np.empty((0, 4), dtype=np.float64),
        np.empty((0, 4), dtype=np.float64),
    )
    assert isinstance(result, PredictionBiasResult)
    assert result.num_frames == 0
    assert result.size_bias == 1.0
    assert result.x_offset_bias == 0.0


def test_single_frame():
    gt = _static_gt(n=1)
    result = BiasAnalyzer().analyze(gt.copy(), gt)
    assert result.num_frames == 1
    assert result.temporal_drift_slope == 0.0


def test_mismatched_lengths_uses_min():
    gt_10 = _static_gt(n=10)
    gt_5 = _static_gt(n=5)
    result = BiasAnalyzer().analyze(gt_10, gt_5)
    assert result.num_frames == 5


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_keys():
    gt = _static_gt(n=10)
    result = BiasAnalyzer().analyze(gt.copy(), gt, tracker_name="T", sequence_name="S")
    d = result.to_dict()
    expected = {
        "tracker_name", "sequence_name", "num_frames",
        "size_bias", "width_bias", "height_bias",
        "x_offset_bias", "y_offset_bias",
        "aspect_ratio_deviation", "temporal_drift_slope",
    }
    assert expected.issubset(d.keys())
    assert d["tracker_name"] == "T"
    assert d["sequence_name"] == "S"


def test_str_representation():
    gt = _static_gt(n=10)
    result = BiasAnalyzer().analyze(gt.copy(), gt, tracker_name="KCF", sequence_name="seq1")
    s = str(result)
    assert "KCF" in s
    assert "seq1" in s
    assert "size_bias" in s


# ---------------------------------------------------------------------------
# analyze_benchmark (integration with BenchmarkEngine)
# ---------------------------------------------------------------------------

def test_analyze_benchmark_returns_correct_keys():
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), dataset, dataset_name="syn", max_sequences=2)

    report = BiasAnalyzer().analyze_benchmark(result)

    assert "per_sequence" in report
    assert "aggregate" in report
    agg = report["aggregate"]
    assert "mean_size_bias" in agg
    assert "mean_x_offset_bias" in agg
    assert "mean_temporal_drift_slope" in agg
    assert agg["num_sequences"] == 2


def test_analyze_benchmark_per_sequence_entries():
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.kcf import KCFTracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=3, num_frames=15, seed=1)
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(KCFTracker(), dataset, dataset_name="syn", max_sequences=3)

    report = BiasAnalyzer().analyze_benchmark(result)

    assert len(report["per_sequence"]) == 3
    for seq_result in report["per_sequence"].values():
        assert isinstance(seq_result, PredictionBiasResult)
        assert seq_result.num_frames > 0


def test_analyze_benchmark_empty_predictions():
    """Sequences without predictions → empty aggregate."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult

    prof = ProfilingResult(
        tracker_name="X", frame_count=10, fps=30.0,
        latency_mean_ms=33.0, latency_std_ms=1.0,
        latency_p95_ms=35.0, peak_memory_mb=50.0,
    )
    sr = SequenceResult(sequence_name="s1", ious=np.array([0.5, 0.6]), profiling=prof)
    br = BenchmarkResult(tracker_name="X", dataset_name="test")
    br.sequence_results = [sr]

    report = BiasAnalyzer().analyze_benchmark(br)
    assert report["aggregate"] == {}
    assert report["per_sequence"] == {}


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def test_markdown_table_format():
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.datasets.synthetic import SyntheticDataset

    dataset = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
    result = BenchmarkEngine(verbose=False).run(
        MOSSETracker(), dataset, max_sequences=2
    )
    report = BiasAnalyzer().analyze_benchmark(result)
    md = BiasAnalyzer().to_markdown_table(report["aggregate"])

    assert "|" in md
    assert "Size Bias" in md
    assert "Drift Slope" in md
    # Three lines: header, separator, data row
    assert len(md.strip().splitlines()) == 3


def test_markdown_table_empty_aggregate():
    md = BiasAnalyzer().to_markdown_table({})
    assert "No bias data" in md
