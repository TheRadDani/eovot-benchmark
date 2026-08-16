"""Tests for full round-trip fidelity of ProfilingResult and EnergyResult fields.

Verifies that tail-latency statistics (p95, p99, std, CV) and energy fields
(TDP, mean power, CPU utilisation) survive the to_dict → from_dict cycle
without data loss.
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Minimal in-memory fixtures (no disk I/O)
# ---------------------------------------------------------------------------

_FIXED_BOX = (10.0, 10.0, 50.0, 50.0)
_GT_BOX = (10.0, 10.0, 50.0, 50.0)
_N_FRAMES = 30


class _SlowTracker(BaseTracker):
    """Tracker with artificially variable latency to produce non-trivial p95/p99."""

    def __init__(self):
        super().__init__(name="SlowTracker")
        self._call = 0

    def initialize(self, frame, bbox):
        self._call = 0

    def update(self, frame):
        self._call += 1
        # Every 5th frame is 3× slower — creates a realistic latency distribution.
        if self._call % 5 == 0:
            time.sleep(0.003)
        else:
            time.sleep(0.001)
        return _FIXED_BOX


class _InMemorySequence(Sequence):
    def __init__(self, name: str, n: int):
        gt = np.tile(_GT_BOX, (n, 1))
        super().__init__(name=name, frame_paths=["x"] * n, ground_truth=gt)
        self._n = n

    def __len__(self):
        return self._n

    def __iter__(self) -> Iterator[np.ndarray]:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for _ in range(self._n):
            yield frame


class _SmallDataset(BaseDataset):
    def __init__(self, n_sequences: int = 2):
        self._seqs = [_InMemorySequence(f"seq_{i}", _N_FRAMES) for i in range(n_sequences)]

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]


@pytest.fixture(scope="module")
def result_no_energy() -> BenchmarkResult:
    engine = BenchmarkEngine(verbose=False, tdp_watts=None)
    return engine.run(_SlowTracker(), _SmallDataset(), dataset_name="Synthetic")


@pytest.fixture(scope="module")
def result_with_energy() -> BenchmarkResult:
    engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
    return engine.run(_SlowTracker(), _SmallDataset(), dataset_name="Synthetic")


# ---------------------------------------------------------------------------
# Tail-latency fields — round-trip without energy
# ---------------------------------------------------------------------------

class TestProfilingRoundTrip:
    def test_latency_std_preserved(self, result_no_energy):
        d = result_no_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_no_energy.sequence_results, loaded.sequence_results):
            assert rec.profiling.latency_std_ms == pytest.approx(
                orig.profiling.latency_std_ms, abs=0.01
            ), "latency_std_ms lost in round-trip"

    def test_latency_p95_preserved(self, result_no_energy):
        d = result_no_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_no_energy.sequence_results, loaded.sequence_results):
            assert rec.profiling.latency_p95_ms == pytest.approx(
                orig.profiling.latency_p95_ms, abs=0.01
            ), "latency_p95_ms lost in round-trip"

    def test_latency_p99_preserved(self, result_no_energy):
        d = result_no_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_no_energy.sequence_results, loaded.sequence_results):
            assert rec.profiling.latency_p99_ms == pytest.approx(
                orig.profiling.latency_p99_ms, abs=0.01
            ), "latency_p99_ms lost in round-trip"

    def test_latency_cv_preserved(self, result_no_energy):
        d = result_no_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_no_energy.sequence_results, loaded.sequence_results):
            assert rec.profiling.latency_cv == pytest.approx(
                orig.profiling.latency_cv, abs=0.001
            ), "latency_cv lost in round-trip"

    def test_latency_cv_nonzero(self, result_no_energy):
        """SlowTracker has variable latency, so CV must be > 0."""
        for r in result_no_energy.sequence_results:
            assert r.profiling.latency_cv > 0.0

    def test_p99_geq_p95(self, result_no_energy):
        for r in result_no_energy.sequence_results:
            assert r.profiling.latency_p99_ms >= r.profiling.latency_p95_ms

    def test_p95_geq_mean(self, result_no_energy):
        for r in result_no_energy.sequence_results:
            assert r.profiling.latency_p95_ms >= r.profiling.latency_mean_ms

    def test_to_dict_contains_p99_key(self, result_no_energy):
        d = result_no_energy.to_dict()
        for seq in d["sequences"]:
            assert "latency_p99_ms" in seq, "latency_p99_ms missing from serialised dict"

    def test_to_dict_contains_cv_key(self, result_no_energy):
        d = result_no_energy.to_dict()
        for seq in d["sequences"]:
            assert "latency_cv" in seq, "latency_cv missing from serialised dict"

    def test_to_dict_contains_std_key(self, result_no_energy):
        d = result_no_energy.to_dict()
        for seq in d["sequences"]:
            assert "latency_std_ms" in seq, "latency_std_ms missing from serialised dict"


# ---------------------------------------------------------------------------
# Energy fields — round-trip with energy profiling enabled
# ---------------------------------------------------------------------------

class TestEnergyRoundTrip:
    def test_energy_present(self, result_with_energy):
        for r in result_with_energy.sequence_results:
            assert r.energy is not None

    def test_tdp_watts_preserved(self, result_with_energy):
        d = result_with_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_with_energy.sequence_results, loaded.sequence_results):
            assert rec.energy.tdp_watts == pytest.approx(
                orig.energy.tdp_watts, abs=0.001
            ), "tdp_watts lost in round-trip"

    def test_mean_power_preserved(self, result_with_energy):
        d = result_with_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_with_energy.sequence_results, loaded.sequence_results):
            assert rec.energy.mean_power_w == pytest.approx(
                orig.energy.mean_power_w, abs=0.01
            ), "mean_power_w lost in round-trip"

    def test_peak_cpu_preserved(self, result_with_energy):
        d = result_with_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_with_energy.sequence_results, loaded.sequence_results):
            assert rec.energy.peak_cpu_pct == pytest.approx(
                orig.energy.peak_cpu_pct, abs=0.1
            ), "peak_cpu_pct lost in round-trip"

    def test_mean_cpu_preserved(self, result_with_energy):
        d = result_with_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_with_energy.sequence_results, loaded.sequence_results):
            assert rec.energy.mean_cpu_pct == pytest.approx(
                orig.energy.mean_cpu_pct, abs=0.1
            ), "mean_cpu_pct lost in round-trip"

    def test_total_energy_preserved(self, result_with_energy):
        d = result_with_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_with_energy.sequence_results, loaded.sequence_results):
            assert rec.energy.total_energy_j == pytest.approx(
                orig.energy.total_energy_j, rel=1e-4
            ), "total_energy_j lost in round-trip"

    def test_energy_per_frame_preserved(self, result_with_energy):
        d = result_with_energy.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(result_with_energy.sequence_results, loaded.sequence_results):
            assert rec.energy.energy_per_frame_mj == pytest.approx(
                orig.energy.energy_per_frame_mj, abs=0.001
            ), "energy_per_frame_mj lost in round-trip"

    def test_to_dict_contains_energy_tdp_key(self, result_with_energy):
        d = result_with_energy.to_dict()
        for seq in d["sequences"]:
            assert "energy_tdp_watts" in seq

    def test_to_dict_contains_mean_power_key(self, result_with_energy):
        d = result_with_energy.to_dict()
        for seq in d["sequences"]:
            assert "energy_mean_power_w" in seq


# ---------------------------------------------------------------------------
# Analysis __init__ export consistency
# ---------------------------------------------------------------------------

class TestAnalysisExports:
    def test_resolution_analyzer_importable_from_package(self):
        from eovot.analysis import ResolutionScaleAnalyzer
        assert ResolutionScaleAnalyzer is not None

    def test_scale_result_importable_from_package(self):
        from eovot.analysis import ScaleResult
        assert ScaleResult is not None

    def test_scale_entry_importable_from_package(self):
        from eovot.analysis import ScaleEntry
        assert ScaleEntry is not None

    def test_frame_skip_analyzer_still_exported(self):
        from eovot.analysis import FrameSkipAnalyzer
        assert FrameSkipAnalyzer is not None

    def test_all_exports_in_dunder_all(self):
        import eovot.analysis as m
        for name in ("FrameSkipAnalyzer", "SkipRateResult",
                     "ResolutionScaleAnalyzer", "ScaleResult", "ScaleEntry"):
            assert name in m.__all__, f"{name} missing from eovot.analysis.__all__"
