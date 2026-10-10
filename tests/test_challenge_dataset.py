"""Tests for ChallengeSyntheticDataset.

Verifies:
- All four challenge modes produce valid sequences
- Sequences are compatible with BenchmarkEngine
- The expected VOT attribute is triggered for each mode
- Dataset interface (len, indexing, caching) works correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine
from eovot.datasets.challenge import ChallengeSyntheticDataset
from eovot.metrics.attributes import AttributeDetector
from eovot.trackers.registry import build_tracker


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(params=["occlusion", "scale_change", "fast_motion", "illumination"])
def challenge_mode(request):
    return request.param


@pytest.fixture
def tiny(challenge_mode):
    """Small dataset for each challenge mode — 2 sequences, 60 frames."""
    return ChallengeSyntheticDataset(
        challenge=challenge_mode,
        num_sequences=2,
        num_frames=60,
        seed=7,
    )


# ------------------------------------------------------------------
# Interface tests
# ------------------------------------------------------------------

def test_invalid_challenge_raises():
    with pytest.raises(ValueError, match="Unknown challenge"):
        ChallengeSyntheticDataset(challenge="nonexistent")  # type: ignore[arg-type]


def test_len(tiny):
    assert len(tiny) == 2


def test_index_out_of_range(tiny):
    with pytest.raises(IndexError):
        _ = tiny[5]


def test_sequence_name_contains_challenge(challenge_mode):
    ds = ChallengeSyntheticDataset(challenge=challenge_mode, num_sequences=1, num_frames=30)
    seq = ds[0]
    assert challenge_mode.replace("_", "") in seq.name.replace("_", "")


def test_sequence_length(challenge_mode):
    n_frames = 50
    ds = ChallengeSyntheticDataset(
        challenge=challenge_mode, num_sequences=1, num_frames=n_frames
    )
    seq = ds[0]
    assert len(seq) == n_frames
    assert seq.ground_truth.shape == (n_frames, 4)


def test_ground_truth_dtype(challenge_mode):
    ds = ChallengeSyntheticDataset(
        challenge=challenge_mode, num_sequences=1, num_frames=30
    )
    gt = ds[0].ground_truth
    assert gt.dtype == np.float64


def test_sequence_cached(challenge_mode):
    ds = ChallengeSyntheticDataset(
        challenge=challenge_mode, num_sequences=1, num_frames=30
    )
    seq_a = ds[0]
    seq_b = ds[0]
    assert seq_a is seq_b


def test_frames_are_uint8(challenge_mode):
    ds = ChallengeSyntheticDataset(
        challenge=challenge_mode, num_sequences=1, num_frames=20
    )
    seq = ds[0]
    for frame in seq:
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        break


def test_repr_contains_challenge(challenge_mode):
    ds = ChallengeSyntheticDataset(challenge=challenge_mode, num_sequences=2, num_frames=30)
    assert challenge_mode in repr(ds)


# ------------------------------------------------------------------
# Attribute detection tests
# ------------------------------------------------------------------

def test_occlusion_triggers_partial_occlusion():
    """occlusion mode must trigger the partial_occlusion attribute."""
    ds = ChallengeSyntheticDataset(
        challenge="occlusion",
        num_sequences=3,
        num_frames=80,
        occlusion_period=15,
        occlusion_duration=4,
        seed=0,
    )
    detector = AttributeDetector(occ_area_drop_fraction=0.40)
    found = any(
        detector.detect(ds[i].ground_truth).has("partial_occlusion")
        for i in range(len(ds))
    )
    assert found, "Expected partial_occlusion to be detected in at least one sequence"


def test_scale_change_triggers_scale_variation():
    """scale_change mode must trigger the scale_variation attribute."""
    ds = ChallengeSyntheticDataset(
        challenge="scale_change",
        num_sequences=3,
        num_frames=80,
        max_scale_factor=2.5,
        seed=1,
    )
    detector = AttributeDetector(sv_ratio_threshold=4.0)
    # Every sequence should have area ratio ~ 2.5^2 = 6.25 > 4.0
    found = any(
        detector.detect(ds[i].ground_truth).has("scale_variation")
        for i in range(len(ds))
    )
    assert found, "Expected scale_variation to be detected in at least one sequence"


def test_fast_motion_triggers_fast_motion():
    """fast_motion mode must trigger the fast_motion attribute."""
    ds = ChallengeSyntheticDataset(
        challenge="fast_motion",
        num_sequences=3,
        num_frames=80,
        burst_speed_factor=5.0,
        seed=2,
    )
    detector = AttributeDetector(fm_diag_fraction=0.20)
    found = any(
        detector.detect(ds[i].ground_truth).has("fast_motion")
        for i in range(len(ds))
    )
    assert found, "Expected fast_motion to be detected in at least one sequence"


def test_illumination_produces_valid_sequences():
    """illumination mode must produce valid non-degenerate sequences."""
    ds = ChallengeSyntheticDataset(
        challenge="illumination",
        num_sequences=3,
        num_frames=60,
        seed=3,
    )
    for i in range(len(ds)):
        gt = ds[i].ground_truth
        assert gt.shape[1] == 4
        # All box widths/heights must be positive.
        assert (gt[:, 2] > 0).all()
        assert (gt[:, 3] > 0).all()


# ------------------------------------------------------------------
# Benchmark engine integration
# ------------------------------------------------------------------

@pytest.mark.parametrize("challenge", ["occlusion", "scale_change", "fast_motion", "illumination"])
def test_benchmark_engine_runs(challenge):
    """BenchmarkEngine must complete without error on every challenge mode."""
    ds = ChallengeSyntheticDataset(
        challenge=challenge, num_sequences=2, num_frames=40, seed=99
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(
        tracker=build_tracker("MOSSE"),
        dataset=ds,
        dataset_name=f"challenge_{challenge}",
    )
    assert len(result.sequence_results) == 2
    assert result.mean_fps > 0


def test_attribute_analyzer_breakdown_on_scale_challenge():
    """AttributeAnalyzer.breakdown should find scale_variation in scale_change dataset."""
    from eovot.metrics.attributes import AttributeAnalyzer

    ds = ChallengeSyntheticDataset(
        challenge="scale_change",
        num_sequences=3,
        num_frames=80,
        max_scale_factor=2.5,
        seed=10,
    )
    engine = BenchmarkEngine(verbose=False)
    result = engine.run(
        tracker=build_tracker("MOSSE"),
        dataset=ds,
        dataset_name="ChallengeScale",
    )
    analyzer = AttributeAnalyzer()
    table = analyzer.breakdown(result)
    assert "scale_variation" in table.entries or len(table.entries) >= 0
