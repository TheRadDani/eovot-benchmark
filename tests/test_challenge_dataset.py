"""Unit tests for ChallengeDataset and AttributeMetricsAggregator."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.datasets.synthetic_challenges import (
    ChallengeAttribute,
    ChallengeDataset,
    AttributedSequence,
)
from eovot.metrics.attributes import AttributeMetricsAggregator, AttributeReport


# ---------------------------------------------------------------------------
# ChallengeAttribute enum
# ---------------------------------------------------------------------------

def test_challenge_attribute_labels():
    for attr in ChallengeAttribute:
        assert isinstance(attr.label(), str)
        assert attr.label() == attr.label().lower()


def test_challenge_attribute_descriptions():
    for attr in ChallengeAttribute:
        desc = attr.description()
        assert isinstance(desc, str) and len(desc) > 5


# ---------------------------------------------------------------------------
# ChallengeDataset construction
# ---------------------------------------------------------------------------

def test_default_construction():
    ds = ChallengeDataset(num_sequences=6, num_frames=20)
    assert len(ds) == 6


def test_single_challenge():
    ds = ChallengeDataset(
        num_sequences=3,
        num_frames=15,
        challenges=[ChallengeAttribute.OCCLUSION],
    )
    for seq in ds:
        assert ChallengeAttribute.OCCLUSION in seq.attributes


def test_round_robin_assignment():
    challenges = [ChallengeAttribute.FAST_MOTION, ChallengeAttribute.SCALE_CHANGE]
    ds = ChallengeDataset(num_sequences=4, num_frames=10, challenges=challenges)
    assert ChallengeAttribute.FAST_MOTION in ds[0].attributes
    assert ChallengeAttribute.SCALE_CHANGE in ds[1].attributes
    assert ChallengeAttribute.FAST_MOTION in ds[2].attributes
    assert ChallengeAttribute.SCALE_CHANGE in ds[3].attributes


def test_out_of_range_raises():
    ds = ChallengeDataset(num_sequences=2, num_frames=10)
    with pytest.raises(IndexError):
        _ = ds[2]


def test_empty_challenges_raises():
    with pytest.raises(ValueError):
        ChallengeDataset(num_sequences=2, num_frames=10, challenges=[])


# ---------------------------------------------------------------------------
# Sequence shape / content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", list(ChallengeAttribute))
def test_sequence_shapes(attr):
    ds = ChallengeDataset(
        num_sequences=1,
        num_frames=20,
        frame_size=(160, 120),
        bbox_size=(20, 20),
        challenges=[attr],
    )
    seq = ds[0]
    assert isinstance(seq, AttributedSequence)
    assert len(seq) == 20
    assert seq.ground_truth.shape == (20, 4)
    assert attr in seq.attributes

    for frame in seq:
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8


def test_ground_truth_non_negative_sizes():
    ds = ChallengeDataset(
        num_sequences=6, num_frames=15, frame_size=(160, 120), bbox_size=(20, 20)
    )
    for seq in ds:
        assert np.all(seq.ground_truth[:, 2] > 0), "width must be positive"
        assert np.all(seq.ground_truth[:, 3] > 0), "height must be positive"


def test_reproducibility():
    ds1 = ChallengeDataset(num_sequences=2, num_frames=10, seed=99)
    ds2 = ChallengeDataset(num_sequences=2, num_frames=10, seed=99)
    for i in range(2):
        frames1 = list(ds1[i])
        frames2 = list(ds2[i])
        for f1, f2 in zip(frames1, frames2):
            assert np.array_equal(f1, f2)


def test_different_seeds_differ():
    ds_a = ChallengeDataset(num_sequences=1, num_frames=10, seed=1)
    ds_b = ChallengeDataset(num_sequences=1, num_frames=10, seed=2)
    frames_a = list(ds_a[0])
    frames_b = list(ds_b[0])
    assert not all(np.array_equal(fa, fb) for fa, fb in zip(frames_a, frames_b))


# ---------------------------------------------------------------------------
# ChallengeDataset.attribute_index
# ---------------------------------------------------------------------------

def test_attribute_index_keys():
    challenges = [ChallengeAttribute.OCCLUSION, ChallengeAttribute.FAST_MOTION]
    ds = ChallengeDataset(num_sequences=4, num_frames=10, challenges=challenges)
    idx = ds.attribute_index()
    assert set(idx.keys()) == set(challenges)


def test_attribute_index_all_sequences_covered():
    ds = ChallengeDataset(num_sequences=6, num_frames=10)
    idx = ds.attribute_index()
    all_covered = set()
    for indices in idx.values():
        all_covered.update(indices)
    assert all_covered == set(range(6))


# ---------------------------------------------------------------------------
# AttributeMetricsAggregator
# ---------------------------------------------------------------------------

def _make_benchmark_result(n_seq: int = 6, n_frames: int = 20):
    """Run BenchmarkEngine on ChallengeDataset and return the result."""
    from eovot.benchmark.engine import BenchmarkEngine
    from eovot.trackers.mosse import MOSSETracker

    ds = ChallengeDataset(
        num_sequences=n_seq,
        num_frames=n_frames,
        frame_size=(160, 120),
        bbox_size=(20, 20),
        seed=7,
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(MOSSETracker(), ds, dataset_name="ChallengeSet-test")
    return result, ds.attribute_index()


def test_aggregator_returns_report():
    result, attr_index = _make_benchmark_result()
    agg = AttributeMetricsAggregator()
    report = agg.compute(result, attr_index)
    assert isinstance(report, AttributeReport)
    assert report.tracker_name == "MOSSE"
    assert 0.0 <= report.overall_mean_iou <= 1.0


def test_aggregator_covers_all_attributes():
    result, attr_index = _make_benchmark_result(n_seq=12)
    agg = AttributeMetricsAggregator()
    report = agg.compute(result, attr_index)
    assert set(report.per_attribute.keys()) == set(attr_index.keys())


def test_aggregator_metrics_ranges():
    result, attr_index = _make_benchmark_result()
    agg = AttributeMetricsAggregator()
    report = agg.compute(result, attr_index)
    for m in report.per_attribute.values():
        assert 0.0 <= m.mean_iou <= 1.0
        assert m.mean_fps > 0
        assert m.peak_memory_mb >= 0
        assert m.num_sequences >= 1


def test_to_markdown_contains_headers():
    result, attr_index = _make_benchmark_result()
    agg = AttributeMetricsAggregator()
    report = agg.compute(result, attr_index)
    md = agg.to_markdown(report)
    assert "Challenge" in md
    assert "mIoU" in md
    assert "FPS" in md


def test_to_json_structure():
    result, attr_index = _make_benchmark_result()
    agg = AttributeMetricsAggregator()
    report = agg.compute(result, attr_index)
    d = agg.to_json(report)
    assert "tracker" in d
    assert "per_attribute" in d
    assert isinstance(d["per_attribute"], list)
    for entry in d["per_attribute"]:
        assert "attribute" in entry
        assert "mean_iou" in entry
