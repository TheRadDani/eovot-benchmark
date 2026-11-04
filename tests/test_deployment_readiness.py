"""Tests for the Deployment-Readiness Score (DRS) metric."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.deployment import DeploymentReadinessAnalyzer, DeploymentReadinessScore


# ---------------------------------------------------------------------------
# Minimal BenchmarkResult stub for unit tests
# ---------------------------------------------------------------------------

def _make_benchmark_result(
    tracker_name: str = "MOSSE",
    dataset_name: str = "Synthetic",
    n_sequences: int = 3,
    n_frames: int = 50,
    mean_iou_val: float = 0.6,
    fps: float = 120.0,
    peak_memory_mb: float = 200.0,
    seed: int = 0,
):
    """Build a minimal BenchmarkResult-compatible object for testing."""
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult
    from eovot.metrics.accuracy import AccuracyMetrics

    rng = np.random.default_rng(seed)
    result = BenchmarkResult(tracker_name=tracker_name, dataset_name=dataset_name)
    for i in range(n_sequences):
        iou_arr = np.clip(
            rng.normal(mean_iou_val, 0.1, n_frames), 0.0, 1.0
        ).astype(np.float64)
        profiling = ProfilingResult(
            tracker_name=tracker_name,
            frame_count=n_frames,
            fps=fps,
            latency_mean_ms=1000.0 / fps,
            latency_std_ms=0.5,
            latency_p95_ms=1000.0 / fps * 1.1,
            latency_p99_ms=1000.0 / fps * 1.2,
            latency_cv=0.05,
            peak_memory_mb=peak_memory_mb,
        )
        accuracy = AccuracyMetrics(
            mean_iou=float(iou_arr.mean()),
            success_auc=float(iou_arr.mean()) * 0.95,
            precision_auc=float(iou_arr.mean()) * 0.90,
            normalized_precision_auc=float(iou_arr.mean()) * 0.85,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq_{i:02d}",
                ious=iou_arr,
                profiling=profiling,
                accuracy_metrics=accuracy,
            )
        )
    return result


# ---------------------------------------------------------------------------
# DeploymentReadinessAnalyzer construction
# ---------------------------------------------------------------------------

def test_default_constructor():
    analyzer = DeploymentReadinessAnalyzer()
    assert analyzer.memory_budget_mb == 512.0
    assert len(analyzer.weights) == 4
    assert abs(sum(analyzer.weights) - 1.0) < 1e-6


def test_custom_weights():
    analyzer = DeploymentReadinessAnalyzer(
        weight_accuracy=0.4,
        weight_efficiency=0.3,
        weight_robustness=0.2,
        weight_smoothness=0.1,
    )
    assert abs(sum(analyzer.weights) - 1.0) < 1e-6


def test_invalid_weights_raise():
    with pytest.raises(ValueError, match="sum to 1.0"):
        DeploymentReadinessAnalyzer(
            weight_accuracy=0.5,
            weight_efficiency=0.5,
            weight_robustness=0.5,
            weight_smoothness=0.5,
        )


# ---------------------------------------------------------------------------
# DeploymentReadinessScore computation
# ---------------------------------------------------------------------------

def test_compute_returns_drs_object():
    analyzer = DeploymentReadinessAnalyzer()
    result = _make_benchmark_result()
    score = analyzer.compute(result)
    assert isinstance(score, DeploymentReadinessScore)


def test_drs_in_unit_interval():
    analyzer = DeploymentReadinessAnalyzer()
    result = _make_benchmark_result(mean_iou_val=0.6)
    score = analyzer.compute(result)
    assert 0.0 <= score.drs <= 1.0, f"DRS out of range: {score.drs}"


def test_components_in_unit_interval():
    analyzer = DeploymentReadinessAnalyzer()
    result = _make_benchmark_result()
    score = analyzer.compute(result)
    assert 0.0 <= score.accuracy <= 1.0
    assert 0.0 <= score.efficiency <= 1.0
    assert 0.0 <= score.robustness <= 1.0
    assert 0.0 <= score.smoothness <= 1.0


def test_tracker_dataset_names_propagated():
    result = _make_benchmark_result(tracker_name="KCF", dataset_name="OTB100")
    score = DeploymentReadinessAnalyzer().compute(result)
    assert score.tracker_name == "KCF"
    assert score.dataset_name == "OTB100"


def test_higher_iou_gives_higher_drs():
    analyzer = DeploymentReadinessAnalyzer()
    good = _make_benchmark_result(mean_iou_val=0.8, seed=1)
    poor = _make_benchmark_result(mean_iou_val=0.2, seed=1)
    good_score = analyzer.compute(good)
    poor_score = analyzer.compute(poor)
    assert good_score.drs > poor_score.drs


def test_higher_fps_gives_higher_drs():
    analyzer = DeploymentReadinessAnalyzer()
    fast = _make_benchmark_result(fps=500.0, seed=2)
    slow = _make_benchmark_result(fps=5.0, seed=2)
    fast_score = analyzer.compute(fast)
    slow_score = analyzer.compute(slow)
    assert fast_score.drs > slow_score.drs


def test_lower_memory_gives_higher_drs():
    analyzer = DeploymentReadinessAnalyzer(memory_budget_mb=512.0)
    lean = _make_benchmark_result(peak_memory_mb=50.0, seed=3)
    heavy = _make_benchmark_result(peak_memory_mb=1000.0, seed=3)
    lean_score = analyzer.compute(lean)
    heavy_score = analyzer.compute(heavy)
    assert lean_score.drs > heavy_score.drs


def test_drs_zero_when_accuracy_zero():
    """Geometric mean: DRS must be 0 when the accuracy component is exactly 0.

    We construct a result whose success_auc and mean_iou are both 0.0 by
    patching the ious array to all-zeros so that _compute_accuracy returns 0.
    """
    from eovot.benchmark.engine import BenchmarkResult, SequenceResult
    from eovot.profiling.profiler import ProfilingResult
    from eovot.metrics.accuracy import AccuracyMetrics

    result = BenchmarkResult(tracker_name="ZERO", dataset_name="Syn")
    for i in range(2):
        ious = np.zeros(30, dtype=np.float64)
        profiling = ProfilingResult(
            tracker_name="ZERO",
            frame_count=30,
            fps=10.0,
            latency_mean_ms=100.0,
            latency_std_ms=1.0,
            latency_p95_ms=110.0,
            latency_p99_ms=115.0,
            latency_cv=0.01,
            peak_memory_mb=100.0,
        )
        accuracy = AccuracyMetrics(
            mean_iou=0.0,
            success_auc=0.0,
            precision_auc=0.0,
            normalized_precision_auc=0.0,
        )
        result.sequence_results.append(
            SequenceResult(
                sequence_name=f"seq_{i}",
                ious=ious,
                profiling=profiling,
                accuracy_metrics=accuracy,
            )
        )
    analyzer = DeploymentReadinessAnalyzer()
    score = analyzer.compute(result)
    assert score.drs == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Rank method
# ---------------------------------------------------------------------------

def test_rank_returns_sorted_list():
    analyzer = DeploymentReadinessAnalyzer()
    results = [
        _make_benchmark_result(tracker_name=f"T{i}", mean_iou_val=0.3 + i * 0.1, seed=i)
        for i in range(4)
    ]
    scores = analyzer.rank(results)
    assert len(scores) == 4
    for i in range(len(scores) - 1):
        assert scores[i].drs >= scores[i + 1].drs


def test_rank_best_tracker_first():
    analyzer = DeploymentReadinessAnalyzer()
    good = _make_benchmark_result(tracker_name="GOOD", mean_iou_val=0.85, fps=300.0, seed=0)
    bad = _make_benchmark_result(tracker_name="BAD", mean_iou_val=0.20, fps=10.0, seed=1)
    scores = analyzer.rank([bad, good])
    assert scores[0].tracker_name == "GOOD"
    assert scores[1].tracker_name == "BAD"


def test_rank_empty_list():
    scores = DeploymentReadinessAnalyzer().rank([])
    assert scores == []


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def test_markdown_table_format():
    analyzer = DeploymentReadinessAnalyzer()
    results = [_make_benchmark_result(tracker_name=f"T{i}", seed=i) for i in range(3)]
    scores = analyzer.rank(results)
    table = analyzer.to_markdown_table(scores)
    lines = table.splitlines()
    assert lines[0].startswith("| Rank ")
    assert len(lines) == 2 + len(scores)   # header + separator + data rows


def test_markdown_table_contains_tracker_names():
    analyzer = DeploymentReadinessAnalyzer()
    results = [
        _make_benchmark_result(tracker_name="MOSSE", seed=0),
        _make_benchmark_result(tracker_name="KCF", seed=1),
    ]
    scores = analyzer.rank(results)
    table = analyzer.to_markdown_table(scores)
    assert "MOSSE" in table
    assert "KCF" in table


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_roundtrip():
    analyzer = DeploymentReadinessAnalyzer()
    result = _make_benchmark_result()
    score = analyzer.compute(result)
    d = score.to_dict()
    assert "drs" in d
    assert "accuracy" in d
    assert "efficiency" in d
    assert "robustness" in d
    assert "smoothness" in d
    assert "weights" in d
    assert isinstance(d["weights"], dict)
    assert d["tracker_name"] == score.tracker_name


def test_str_representation():
    analyzer = DeploymentReadinessAnalyzer()
    result = _make_benchmark_result()
    score = analyzer.compute(result)
    s = str(score)
    assert "DeploymentReadinessScore" in s
    assert "DRS=" in s


# ---------------------------------------------------------------------------
# Integration with real benchmark pipeline
# ---------------------------------------------------------------------------

def test_integration_with_synthetic_benchmark():
    """Full pipeline: BenchmarkEngine → DeploymentReadinessAnalyzer."""
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    dataset = SyntheticDataset(num_sequences=3, num_frames=30, seed=0)
    engine = BenchmarkEngine(verbose=False)
    mosse_result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
    kcf_result = engine.run(KCFTracker(), dataset, dataset_name="Synthetic")

    analyzer = DeploymentReadinessAnalyzer()
    scores = analyzer.rank([mosse_result, kcf_result])

    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s.drs <= 1.0
        assert s.mean_fps > 0.0
        assert s.mean_iou >= 0.0

    table = analyzer.to_markdown_table(scores)
    assert "MOSSE" in table or "KCF" in table
