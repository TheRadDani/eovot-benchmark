"""Unit tests for the experiment configuration and runner modules.

Tests cover:
- YAML loading into ExperimentConfig / TrackerConfig / DatasetConfig
- Legacy single-tracker/dataset YAML format compatibility
- CLI override semantics via direct attribute assignment
- seed_everything() sets numpy and python random seeds
- ExperimentRunner builds trackers and datasets from registry
- ExperimentRunner raises ValueError for unknown tracker / dataset
- ExperimentRunner produces result dicts with expected keys
- Full end-to-end run with synthetic dataset and MOSSE tracker
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np
import pytest

from eovot.datasets.base import BaseDataset, Sequence
from eovot.experiment.config import DatasetConfig, ExperimentConfig, TrackerConfig
from eovot.experiment.runner import (
    ExperimentRunner,
    build_dataset,
    build_tracker,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return str(p)


def _make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class SyntheticDataset(BaseDataset):
    """Minimal in-memory dataset that returns one fixed sequence."""

    def __init__(self, n_frames: int = 10) -> None:
        self.n_frames = n_frames
        self._seq = self._build_sequence()

    def _build_sequence(self) -> Sequence:
        tmp = tempfile.mkdtemp()
        paths = []
        for i in range(self.n_frames):
            p = str(Path(tmp) / f"{i:04d}.jpg")
            cv2.imwrite(p, _make_frame())
            paths.append(p)
        gt = np.tile(np.array([[10, 10, 30, 30]], dtype=np.float64), (self.n_frames, 1))
        return Sequence(name="synthetic", frame_paths=paths, ground_truth=gt)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Sequence:
        if idx != 0:
            raise IndexError(idx)
        return self._seq


# ---------------------------------------------------------------------------
# TrackerConfig tests
# ---------------------------------------------------------------------------

class TestTrackerConfig:
    def test_from_dict_minimal(self):
        cfg = TrackerConfig.from_dict({"name": "MOSSE"})
        assert cfg.name == "MOSSE"
        assert cfg.params == {}

    def test_from_dict_with_params(self):
        cfg = TrackerConfig.from_dict({"name": "KCF", "params": {"learning_rate": 0.1}})
        assert cfg.name == "KCF"
        assert cfg.params["learning_rate"] == 0.1

    def test_from_dict_params_none(self):
        cfg = TrackerConfig.from_dict({"name": "MOSSE", "params": None})
        assert cfg.params == {}


# ---------------------------------------------------------------------------
# DatasetConfig tests
# ---------------------------------------------------------------------------

class TestDatasetConfig:
    def test_from_dict_otb(self):
        cfg = DatasetConfig.from_dict({"loader": "OTBDataset", "root": "/data/OTB"})
        assert cfg.loader == "OTBDataset"
        assert cfg.root == "/data/OTB"
        assert cfg.label == "OTBDataset"  # defaults to loader name

    def test_from_dict_label_override(self):
        cfg = DatasetConfig.from_dict(
            {"loader": "OTBDataset", "root": "/data", "label": "OTB100"}
        )
        assert cfg.label == "OTB100"

    def test_from_dict_got10k(self):
        cfg = DatasetConfig.from_dict(
            {"loader": "GOT10kDataset", "root": "/data/GOT-10k", "split": "val"}
        )
        assert cfg.split == "val"

    def test_from_dict_max_sequences(self):
        cfg = DatasetConfig.from_dict(
            {"loader": "OTBDataset", "root": "/data", "max_sequences": 10}
        )
        assert cfg.max_sequences == 10

    def test_label_defaults_to_loader(self):
        cfg = DatasetConfig(loader="OTBDataset", root="/data")
        assert cfg.label == "OTBDataset"


# ---------------------------------------------------------------------------
# ExperimentConfig YAML loading
# ---------------------------------------------------------------------------

MINIMAL_YAML = """
experiment:
  name: test-experiment
  output_dir: /tmp/results/
  seed: 7

benchmark:
  verbose: false
  max_sequences: 5

trackers:
  - name: MOSSE
    params:
      learning_rate: 0.2

datasets:
  - loader: OTBDataset
    root: /data/OTB
    label: OTB100
"""

MULTI_YAML = """
experiment:
  name: multi-test
  output_dir: /tmp/multi/
  seed: 42

benchmark:
  verbose: false

trackers:
  - name: MOSSE
  - name: KCF

datasets:
  - loader: OTBDataset
    root: /data/OTB
  - loader: GOT10kDataset
    root: /data/GOT-10k
    split: val
    max_sequences: 10
"""

LEGACY_YAML = """
experiment:
  name: legacy
  seed: 99

tracker:
  name: MOSSE

dataset:
  loader: OTBDataset
  root: /data/OTB
"""


class TestExperimentConfigFromYaml:
    def test_basic_fields(self, tmp_path):
        p = _make_yaml(tmp_path, MINIMAL_YAML)
        cfg = ExperimentConfig.from_yaml(p)
        assert cfg.name == "test-experiment"
        assert cfg.output_dir == "/tmp/results/"
        assert cfg.seed == 7
        assert cfg.verbose is False
        assert cfg.max_sequences == 5

    def test_tracker_config_loaded(self, tmp_path):
        p = _make_yaml(tmp_path, MINIMAL_YAML)
        cfg = ExperimentConfig.from_yaml(p)
        assert len(cfg.trackers) == 1
        assert cfg.trackers[0].name == "MOSSE"
        assert cfg.trackers[0].params["learning_rate"] == pytest.approx(0.2)

    def test_dataset_config_loaded(self, tmp_path):
        p = _make_yaml(tmp_path, MINIMAL_YAML)
        cfg = ExperimentConfig.from_yaml(p)
        assert len(cfg.datasets) == 1
        assert cfg.datasets[0].loader == "OTBDataset"
        assert cfg.datasets[0].label == "OTB100"

    def test_multiple_trackers_and_datasets(self, tmp_path):
        p = _make_yaml(tmp_path, MULTI_YAML)
        cfg = ExperimentConfig.from_yaml(p)
        assert len(cfg.trackers) == 2
        assert len(cfg.datasets) == 2
        assert cfg.datasets[1].max_sequences == 10

    def test_legacy_single_tracker_format(self, tmp_path):
        p = _make_yaml(tmp_path, LEGACY_YAML)
        cfg = ExperimentConfig.from_yaml(p)
        assert len(cfg.trackers) == 1
        assert cfg.trackers[0].name == "MOSSE"
        assert len(cfg.datasets) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml("/nonexistent/path/cfg.yaml")

    def test_roundtrip_to_dict(self, tmp_path):
        p = _make_yaml(tmp_path, MINIMAL_YAML)
        cfg = ExperimentConfig.from_yaml(p)
        d = cfg.to_dict()
        assert d["experiment"]["name"] == "test-experiment"
        assert d["experiment"]["seed"] == 7
        assert len(d["trackers"]) == 1
        assert len(d["datasets"]) == 1

    def test_from_dict(self):
        cfg = ExperimentConfig.from_dict({
            "experiment": {"name": "from-dict", "seed": 13},
            "benchmark": {"verbose": False},
            "trackers": [{"name": "KCF"}],
            "datasets": [{"loader": "OTBDataset", "root": "/data"}],
        })
        assert cfg.name == "from-dict"
        assert cfg.seed == 13
        assert cfg.trackers[0].name == "KCF"


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_numpy_reproducible(self):
        seed_everything(42)
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible(self):
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = np.random.rand(5)
        seed_everything(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# build_tracker / build_dataset
# ---------------------------------------------------------------------------

class TestBuildTracker:
    def test_build_mosse(self):
        from eovot.trackers.mosse import MOSSETracker
        t = build_tracker(TrackerConfig(name="MOSSE"))
        assert isinstance(t, MOSSETracker)

    def test_build_kcf(self):
        from eovot.trackers.kcf import KCFTracker
        t = build_tracker(TrackerConfig(name="KCF"))
        assert isinstance(t, KCFTracker)

    def test_build_mosse_with_params(self):
        t = build_tracker(TrackerConfig(name="MOSSE", params={"learning_rate": 0.05}))
        assert t.learning_rate == pytest.approx(0.05)

    def test_unknown_tracker_raises(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            build_tracker(TrackerConfig(name="NonExistentTracker"))


class TestBuildDataset:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset loader"):
            build_dataset(DatasetConfig(loader="NoSuchDataset", root="/data"))


# ---------------------------------------------------------------------------
# ExperimentRunner end-to-end
# ---------------------------------------------------------------------------

class TestExperimentRunner:
    def _make_runner(self, tmp_path: Path) -> ExperimentRunner:
        cfg = ExperimentConfig(
            name="unit-test",
            output_dir=str(tmp_path / "results"),
            seed=42,
            verbose=False,
            trackers=[TrackerConfig(name="MOSSE")],
            datasets=[],
        )
        return ExperimentRunner(cfg)

    def test_unknown_tracker_raises_on_run(self, tmp_path):
        cfg = ExperimentConfig(
            name="err-test",
            output_dir=str(tmp_path / "results"),
            seed=0,
            verbose=False,
            trackers=[TrackerConfig(name="BadTracker")],
            datasets=[DatasetConfig(loader="OTBDataset", root="/data")],
        )
        runner = ExperimentRunner(cfg)
        # OTBDataset root doesn't exist, but tracker error comes first
        with pytest.raises(Exception):
            runner.run()

    def test_run_produces_result_dicts(self, tmp_path, monkeypatch):
        """Patch build_dataset to inject a synthetic dataset; verify output shape."""
        synth_ds = SyntheticDataset(n_frames=8)

        cfg = ExperimentConfig(
            name="patched-test",
            output_dir=str(tmp_path / "results"),
            seed=42,
            verbose=False,
            trackers=[TrackerConfig(name="MOSSE")],
            datasets=[DatasetConfig(loader="OTBDataset", root="/data", label="Synth")],
        )

        import eovot.experiment.runner as runner_mod

        original_build_dataset = runner_mod.build_dataset

        def patched_build_dataset(ds_cfg):
            return synth_ds

        monkeypatch.setattr(runner_mod, "build_dataset", patched_build_dataset)

        runner = ExperimentRunner(cfg)
        results = runner.run()

        assert len(results) == 1
        r = results[0]
        assert "summary" in r
        assert "sequences" in r
        assert r["summary"]["tracker"] == "MOSSE"
        assert r["experiment"]["seed"] == 42

    def test_run_creates_output_files(self, tmp_path, monkeypatch):
        synth_ds = SyntheticDataset(n_frames=6)

        cfg = ExperimentConfig(
            name="file-test",
            output_dir=str(tmp_path / "out"),
            seed=0,
            verbose=False,
            trackers=[TrackerConfig(name="KCF")],
            datasets=[DatasetConfig(loader="OTBDataset", root="/data", label="DS")],
        )

        import eovot.experiment.runner as runner_mod
        monkeypatch.setattr(runner_mod, "build_dataset", lambda _: synth_ds)

        ExperimentRunner(cfg).run()

        out_dir = tmp_path / "out"
        json_files = list(out_dir.glob("*.json"))
        csv_files = list(out_dir.glob("*.csv"))
        assert len(json_files) >= 1
        assert len(csv_files) >= 1
