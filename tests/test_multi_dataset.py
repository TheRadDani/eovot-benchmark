"""Tests for MultiDatasetAggregator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.analysis.multi_dataset import (
    AGGREGATION_METRICS,
    AggregatedEntry,
    MultiDatasetAggregator,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic BenchmarkResult-like summary dicts
# ---------------------------------------------------------------------------

def _make_result(tracker: str, dataset: str, mean_iou: float, mean_fps: float,
                 peak_memory_mb: float = 100.0, success_auc: float = None,
                 precision_auc: float = None) -> dict:
    """Build a minimal summary dict accepted by MultiDatasetAggregator."""
    d = {
        "tracker": tracker,
        "dataset": dataset,
        "mean_iou": mean_iou,
        "mean_fps": mean_fps,
        "peak_memory_mb": peak_memory_mb,
    }
    if success_auc is not None:
        d["success_auc"] = success_auc
    if precision_auc is not None:
        d["precision_auc"] = precision_auc
    return d


# Three trackers on two datasets
GOT_RESULTS = [
    _make_result("MOSSE", "GOT-10k", 0.60, 500.0, 80.0),
    _make_result("KCF",   "GOT-10k", 0.55, 300.0, 90.0),
    _make_result("CSRT",  "GOT-10k", 0.70, 120.0, 95.0),
]

LASOT_RESULTS = [
    _make_result("MOSSE", "LaSOT", 0.38, 480.0, 82.0),
    _make_result("KCF",   "LaSOT", 0.42, 290.0, 91.0),
    _make_result("CSRT",  "LaSOT", 0.55, 115.0, 97.0),
]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_default_construction():
    agg = MultiDatasetAggregator()
    assert len(agg._metrics) == len(AGGREGATION_METRICS)


def test_custom_metrics():
    custom = [("mean_iou", "mIoU", True)]
    agg = MultiDatasetAggregator(metrics=custom)
    assert len(agg._metrics) == 1


def test_min_datasets_clamp():
    agg = MultiDatasetAggregator(min_datasets=0)
    assert agg._min_datasets == 1


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def test_add_result_dict():
    agg = MultiDatasetAggregator()
    agg.add_result(GOT_RESULTS[0])
    assert "MOSSE" in agg._raw
    assert "GOT-10k" in agg._raw["MOSSE"]


def test_add_results_batch():
    agg = MultiDatasetAggregator()
    agg.add_results(GOT_RESULTS)
    assert set(agg._raw.keys()) == {"MOSSE", "KCF", "CSRT"}


def test_add_result_full_dict_with_summary_key():
    """Dicts with a nested 'summary' key (to_dict() output) are supported."""
    wrapped = {"summary": GOT_RESULTS[0]}
    agg = MultiDatasetAggregator()
    agg.add_result(wrapped)
    assert "MOSSE" in agg._raw


def test_add_result_benchmark_result_object():
    """BenchmarkResult objects (with .summary() method) are supported."""
    class _FakeResult:
        def summary(self):
            return GOT_RESULTS[1].copy()

    agg = MultiDatasetAggregator()
    agg.add_result(_FakeResult())
    assert "KCF" in agg._raw


def test_add_result_unsupported_type():
    agg = MultiDatasetAggregator()
    with pytest.raises(TypeError, match="unsupported result type"):
        agg.add_result(42)


def test_rank_raises_when_empty():
    agg = MultiDatasetAggregator()
    with pytest.raises(ValueError, match="no results added"):
        agg.rank()


# ---------------------------------------------------------------------------
# Ranking correctness
# ---------------------------------------------------------------------------

def _make_agg_two_datasets() -> MultiDatasetAggregator:
    agg = MultiDatasetAggregator()
    agg.add_results(GOT_RESULTS)
    agg.add_results(LASOT_RESULTS)
    return agg


def test_rank_returns_all_trackers():
    entries = _make_agg_two_datasets().rank()
    names = [e.tracker_name for e in entries]
    assert set(names) == {"MOSSE", "KCF", "CSRT"}


def test_rank_sorted_descending():
    entries = _make_agg_two_datasets().rank()
    scores = [e.composite_score for e in entries]
    assert scores == sorted(scores, reverse=True)


def test_rank_assigns_sequential_ranks():
    entries = _make_agg_two_datasets().rank()
    ranks = [e.rank for e in entries]
    assert ranks == list(range(1, len(ranks) + 1))


def test_rank_num_datasets():
    entries = _make_agg_two_datasets().rank()
    for e in entries:
        assert e.num_datasets == 2


def test_csrt_best_iou_ranks_first():
    """CSRT has highest IoU on both datasets; with IoU-heavy weighting it should rank first."""
    agg = MultiDatasetAggregator(metric_weights={"mean_iou": 10.0, "mean_fps": 0.1, "peak_memory_mb": 0.1})
    agg.add_results(GOT_RESULTS)
    agg.add_results(LASOT_RESULTS)
    entries = agg.rank()
    assert entries[0].tracker_name == "CSRT", f"Expected CSRT first, got {entries[0].tracker_name}"


def test_mosse_best_fps_ranks_first_when_fps_dominant():
    """MOSSE has highest FPS; with FPS-dominant weighting it should rank first."""
    agg = MultiDatasetAggregator(metric_weights={"mean_iou": 0.1, "mean_fps": 10.0, "peak_memory_mb": 0.1})
    agg.add_results(GOT_RESULTS)
    agg.add_results(LASOT_RESULTS)
    entries = agg.rank()
    assert entries[0].tracker_name == "MOSSE", f"Expected MOSSE first, got {entries[0].tracker_name}"


def test_min_datasets_filters_trackers():
    """Tracker only on one dataset is excluded when min_datasets=2."""
    agg = MultiDatasetAggregator(min_datasets=2)
    # Add MOSSE on both, KCF on one
    agg.add_result(_make_result("MOSSE", "DS1", 0.6, 200.0))
    agg.add_result(_make_result("MOSSE", "DS2", 0.5, 180.0))
    agg.add_result(_make_result("KCF",   "DS1", 0.55, 150.0))
    entries = agg.rank()
    names = [e.tracker_name for e in entries]
    assert "MOSSE" in names
    assert "KCF" not in names


def test_dataset_weights_affect_ranking():
    """Heavy weight on a dataset where MOSSE leads should promote MOSSE."""
    # MOSSE best on GOT-10k by FPS; CSRT best by IoU on both
    agg_fps_mosse = MultiDatasetAggregator(
        dataset_weights={"GOT-10k": 5.0, "LaSOT": 1.0},
        metric_weights={"mean_iou": 0.1, "mean_fps": 10.0, "peak_memory_mb": 0.1},
    )
    agg_fps_mosse.add_results(GOT_RESULTS)
    agg_fps_mosse.add_results(LASOT_RESULTS)
    first = agg_fps_mosse.rank()[0].tracker_name
    assert first == "MOSSE"


def test_single_tracker_single_dataset():
    """Corner case: only one tracker, one dataset — composite score = 0 (z-score of single element)."""
    agg = MultiDatasetAggregator()
    agg.add_result(_make_result("MOSSE", "DS1", 0.6, 200.0))
    entries = agg.rank()
    assert len(entries) == 1
    assert entries[0].composite_score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Export methods
# ---------------------------------------------------------------------------

def test_to_markdown_returns_string():
    md = _make_agg_two_datasets().to_markdown()
    assert isinstance(md, str)
    assert "Tracker" in md
    assert "Score" in md
    assert "MOSSE" in md


def test_to_csv_creates_file():
    agg = _make_agg_two_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = agg.to_csv(Path(tmpdir) / "ranking")
        assert out.exists()
        content = out.read_text()
        assert "rank" in content
        assert "composite_score" in content


def test_to_json_creates_file():
    agg = _make_agg_two_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = agg.to_json(Path(tmpdir) / "ranking")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "ranking" in data
        assert len(data["ranking"]) == 3


def test_to_json_roundtrip():
    """JSON export contains all expected fields."""
    agg = _make_agg_two_datasets()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = agg.to_json(Path(tmpdir) / "r.json")
        data = json.loads(out.read_text())
    first = data["ranking"][0]
    assert "rank" in first
    assert "tracker" in first
    assert "composite_score" in first
    assert "per_dataset_score" in first
    assert isinstance(first["per_dataset_score"], dict)


def test_summary_dict():
    d = _make_agg_two_datasets().summary_dict()
    assert d["num_trackers"] == 3
    assert d["num_datasets"] == 2
    assert len(d["ranking"]) == 3


# ---------------------------------------------------------------------------
# Integration: end-to-end with BenchmarkEngine
# ---------------------------------------------------------------------------

def test_end_to_end_with_benchmark_engine():
    """Verify MultiDatasetAggregator works with real BenchmarkEngine output."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.datasets.synthetic import SyntheticDataset
    from eovot.trackers.mosse import MOSSETracker
    from eovot.trackers.kcf import KCFTracker

    engine = BenchmarkEngine(verbose=False)
    ds1 = SyntheticDataset(num_sequences=2, num_frames=15, frame_size=(160, 120), seed=3)
    ds2 = SyntheticDataset(num_sequences=2, num_frames=15, frame_size=(160, 120), seed=7)

    results1 = [
        engine.run(MOSSETracker(), ds1, dataset_name="Synth1"),
        engine.run(KCFTracker(),   ds1, dataset_name="Synth1"),
    ]
    results2 = [
        engine.run(MOSSETracker(), ds2, dataset_name="Synth2"),
        engine.run(KCFTracker(),   ds2, dataset_name="Synth2"),
    ]

    agg = MultiDatasetAggregator()
    agg.add_results(results1)
    agg.add_results(results2)

    entries = agg.rank()
    assert len(entries) == 2
    assert entries[0].num_datasets == 2
    assert all(isinstance(e, AggregatedEntry) for e in entries)
    # Markdown should render without error
    md = agg.to_markdown()
    assert "MOSSE" in md or "KCF" in md
