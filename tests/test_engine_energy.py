"""Integration tests for BenchmarkEngine energy profiling wiring.

Verifies that the ``energy`` field on :class:`~eovot.benchmark.engine.SequenceResult`
is correctly populated when ``tdp_watts`` is provided to :class:`BenchmarkEngine`
and is ``None`` when energy profiling is disabled.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult, SequenceResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.profiling.energy import EnergyResult
from eovot.trackers.base import BaseTracker

BBox = Tuple[float, float, float, float]
FIXED_BOX: BBox = (10.0, 10.0, 50.0, 50.0)
NUM_FRAMES = 15


# ---------------------------------------------------------------------------
# Minimal synthetic fixtures (identical to test_engine.py helpers)
# ---------------------------------------------------------------------------

class ConstantTracker(BaseTracker):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "ConstantTracker"

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        pass

    def update(self, frame: np.ndarray) -> BBox:
        return FIXED_BOX


class SyntheticSequence(Sequence):
    def __init__(self, name: str) -> None:
        gt = np.tile(np.array(FIXED_BOX), (NUM_FRAMES, 1))
        super().__init__(
            name=name,
            frame_paths=[f"frame_{i:04d}.jpg" for i in range(NUM_FRAMES)],
            ground_truth=gt,
        )

    def __iter__(self) -> Iterator[np.ndarray]:  # type: ignore[override]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(NUM_FRAMES):
            yield frame


class SyntheticDataset(BaseDataset):
    def __init__(self, n: int = 2) -> None:
        self._seqs = [SyntheticSequence(f"seq_{i}") for i in range(n)]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Sequence:
        return self._seqs[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineEnergyWiring:
    def setup_method(self):
        self.tracker = ConstantTracker()
        self.dataset = SyntheticDataset(n=2)

    def test_energy_none_when_tdp_not_set(self):
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        for sr in result.sequence_results:
            assert sr.energy is None, "energy should be None without tdp_watts"

    def test_energy_populated_when_tdp_set(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        for sr in result.sequence_results:
            assert sr.energy is not None, "energy must be set when tdp_watts provided"
            assert isinstance(sr.energy, EnergyResult)

    def test_energy_result_has_correct_tracker_name(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=10.0)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        for sr in result.sequence_results:
            assert sr.energy is not None
            assert sr.energy.tracker_name == self.tracker.name

    def test_energy_result_positive_values(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=10.0)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        for sr in result.sequence_results:
            assert sr.energy is not None
            assert sr.energy.total_energy_j >= 0.0
            assert sr.energy.energy_per_frame_mj >= 0.0
            assert sr.energy.frame_count > 0

    def test_total_energy_j_aggregates_over_sequences(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=10.0)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        assert result.total_energy_j is not None
        # Total must be >= sum of individual sequence energies (none can be negative)
        per_seq = sum(sr.energy.total_energy_j for sr in result.sequence_results if sr.energy)
        assert result.total_energy_j == pytest.approx(per_seq, rel=1e-6)

    def test_total_energy_none_without_tdp(self):
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        assert result.total_energy_j is None

    def test_summary_contains_energy_keys_when_profiled(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=10.0)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        s = result.summary()
        assert "total_energy_j" in s
        assert "mean_energy_per_frame_mj" in s

    def test_to_dict_includes_per_sequence_energy(self):
        engine = BenchmarkEngine(verbose=False, tdp_watts=10.0)
        result = engine.run(self.tracker, self.dataset, dataset_name="Test")
        d = result.to_dict()
        for seq_entry in d["sequences"]:
            assert "energy_j" in seq_entry
            assert "energy_per_frame_mj" in seq_entry
