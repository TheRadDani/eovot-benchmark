"""Tests for BenchmarkResult serialisation/deserialisation and ResultsBank."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.base import BaseDataset, Sequence
from eovot.results.bank import ResultsBank
from eovot.trackers.base import BaseTracker


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic tracker + dataset (no file I/O)
# ---------------------------------------------------------------------------

_FIXED_BOX = (10.0, 10.0, 50.0, 50.0)
_GT_BOX = (10.0, 10.0, 50.0, 50.0)
_N_FRAMES = 15


class _ConstantTracker(BaseTracker):
    def __init__(self):
        super().__init__(name="ConstantTracker")

    def initialize(self, frame, bbox):
        pass

    def update(self, frame):
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


@pytest.fixture
def benchmark_result() -> BenchmarkResult:
    engine = BenchmarkEngine(verbose=False)
    return engine.run(_ConstantTracker(), _SmallDataset(), dataset_name="Synthetic")


# ---------------------------------------------------------------------------
# BenchmarkResult.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestFromDict:
    def test_round_trip_tracker_and_dataset(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        assert loaded.tracker_name == benchmark_result.tracker_name
        assert loaded.dataset_name == benchmark_result.dataset_name

    def test_round_trip_num_sequences(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        assert len(loaded.sequence_results) == len(benchmark_result.sequence_results)

    def test_round_trip_mean_iou(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        assert loaded.mean_iou == pytest.approx(benchmark_result.mean_iou, abs=1e-3)

    def test_round_trip_fps(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        assert loaded.mean_fps == pytest.approx(benchmark_result.mean_fps, rel=0.01)

    def test_round_trip_per_frame_ious(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(benchmark_result.sequence_results, loaded.sequence_results):
            np.testing.assert_allclose(rec.ious, orig.ious, atol=1e-5)

    def test_round_trip_success_auc(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        assert loaded.mean_success_auc == pytest.approx(benchmark_result.mean_success_auc, abs=1e-3)

    def test_round_trip_precision_auc(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        assert loaded.mean_precision_auc == pytest.approx(
            benchmark_result.mean_precision_auc, abs=1e-3
        )

    def test_sequence_names_preserved(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        orig_names = [r.sequence_name for r in benchmark_result.sequence_results]
        loaded_names = [r.sequence_name for r in loaded.sequence_results]
        assert loaded_names == orig_names

    def test_centre_distances_preserved(self, benchmark_result):
        d = benchmark_result.to_dict()
        loaded = BenchmarkResult.from_dict(d)
        for orig, rec in zip(benchmark_result.sequence_results, loaded.sequence_results):
            if orig.center_distances is not None:
                np.testing.assert_allclose(rec.center_distances, orig.center_distances, atol=1e-3)


# ---------------------------------------------------------------------------
# BenchmarkResult.save / load (file round-trip)
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, benchmark_result, tmp_path):
        p = benchmark_result.save(tmp_path / "result")
        assert p.exists()
        assert p.suffix == ".json"

    def test_save_appends_json_suffix(self, benchmark_result, tmp_path):
        p = benchmark_result.save(tmp_path / "result")
        assert p.name == "result.json"

    def test_save_explicit_json_path(self, benchmark_result, tmp_path):
        p = benchmark_result.save(tmp_path / "my_result.json")
        assert p.exists()

    def test_load_restores_tracker_name(self, benchmark_result, tmp_path):
        p = benchmark_result.save(tmp_path / "r")
        loaded = BenchmarkResult.load(p)
        assert loaded.tracker_name == benchmark_result.tracker_name

    def test_load_restores_mean_iou(self, benchmark_result, tmp_path):
        p = benchmark_result.save(tmp_path / "r")
        loaded = BenchmarkResult.load(p)
        assert loaded.mean_iou == pytest.approx(benchmark_result.mean_iou, abs=1e-3)

    def test_save_creates_parent_dirs(self, benchmark_result, tmp_path):
        nested = tmp_path / "deep" / "nested" / "result"
        p = benchmark_result.save(nested)
        assert p.exists()

    def test_json_is_valid(self, benchmark_result, tmp_path):
        p = benchmark_result.save(tmp_path / "r")
        with open(p) as fh:
            d = json.load(fh)
        assert "summary" in d
        assert "sequences" in d


# ---------------------------------------------------------------------------
# ResultsBank
# ---------------------------------------------------------------------------

class TestResultsBank:
    def test_save_creates_json_file(self, benchmark_result, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        path = bank.save(benchmark_result)
        assert path.exists()
        assert path.suffix == ".json"

    def test_list_results_after_save(self, benchmark_result, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        bank.save(benchmark_result)
        entries = bank.list_results()
        assert len(entries) == 1
        assert entries[0]["tracker"] == benchmark_result.tracker_name

    def test_list_results_sorted_by_miou(self, tmp_path):
        engine = BenchmarkEngine(verbose=False)
        bank = ResultsBank(tmp_path / "bank")
        for _ in range(2):
            r = engine.run(_ConstantTracker(), _SmallDataset(), dataset_name="Syn")
            bank.save(r)
        entries = bank.list_results()
        ious = [e["mean_iou"] for e in entries]
        assert ious == sorted(ious, reverse=True)

    def test_load_by_pattern(self, benchmark_result, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        bank.save(benchmark_result)
        loaded = bank.load("*")
        assert loaded.tracker_name == benchmark_result.tracker_name

    def test_load_all_returns_all(self, tmp_path):
        engine = BenchmarkEngine(verbose=False)
        bank = ResultsBank(tmp_path / "bank")
        for _ in range(3):
            r = engine.run(_ConstantTracker(), _SmallDataset(), dataset_name="Syn")
            bank.save(r)
        results = bank.load_all()
        assert len(results) == 3

    def test_load_all_dataset_filter(self, tmp_path):
        engine = BenchmarkEngine(verbose=False)
        bank = ResultsBank(tmp_path / "bank")
        r1 = engine.run(_ConstantTracker(), _SmallDataset(), dataset_name="OTB")
        r2 = engine.run(_ConstantTracker(), _SmallDataset(), dataset_name="LaSOT")
        bank.save(r1)
        bank.save(r2)
        results = bank.load_all(dataset="OTB")
        assert all(r.dataset_name == "OTB" for r in results)

    def test_compare_returns_markdown(self, benchmark_result, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        bank.save(benchmark_result)
        table = bank.compare()
        assert "Tracker" in table
        assert benchmark_result.tracker_name in table

    def test_compare_empty_bank(self, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        table = bank.compare()
        assert "No results" in table

    def test_index_persists_across_instances(self, benchmark_result, tmp_path):
        bank_dir = tmp_path / "bank"
        bank1 = ResultsBank(bank_dir)
        bank1.save(benchmark_result)

        bank2 = ResultsBank(bank_dir)
        entries = bank2.list_results()
        assert len(entries) == 1

    def test_delete_removes_file_and_index_entry(self, benchmark_result, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        bank.save(benchmark_result)
        assert len(bank.list_results()) == 1
        n = bank.delete("*")
        assert n == 1
        assert len(bank.list_results()) == 0

    def test_save_with_explicit_name(self, benchmark_result, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        path = bank.save(benchmark_result, name="my_run")
        assert path.stem == "my_run"
        loaded = bank.load("my_run")
        assert loaded.tracker_name == benchmark_result.tracker_name

    def test_load_missing_raises(self, tmp_path):
        bank = ResultsBank(tmp_path / "bank")
        with pytest.raises(FileNotFoundError):
            bank.load("does_not_exist")
