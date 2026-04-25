"""Tests for eovot.experiment.runner and eovot.experiment.snapshot."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eovot.experiment.snapshot import ReproducibilitySnapshot


# ---------------------------------------------------------------------------
# ReproducibilitySnapshot tests
# ---------------------------------------------------------------------------

class TestReproducibilitySnapshot:
    def test_capture_populates_timestamp(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.timestamp
        assert "T" in snap.timestamp  # ISO-8601 format

    def test_capture_populates_python_version(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.python_version
        assert "." in snap.python_version

    def test_capture_populates_platform(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.platform_info

    def test_capture_cpu_count_positive(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.cpu_count >= 1

    def test_capture_seed_stored(self):
        snap = ReproducibilitySnapshot.capture(seed=42)
        assert snap.random_seed == 42

    def test_capture_seed_none_by_default(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.random_seed is None

    def test_capture_numpy_in_packages(self):
        snap = ReproducibilitySnapshot.capture()
        assert "numpy" in snap.package_versions

    def test_to_dict_is_json_serializable(self):
        snap = ReproducibilitySnapshot.capture(seed=7)
        d = snap.to_dict()
        serialized = json.dumps(d)
        assert serialized  # non-empty string

    def test_to_dict_required_keys(self):
        snap = ReproducibilitySnapshot.capture()
        d = snap.to_dict()
        for key in ("timestamp", "python_version", "platform", "cpu_count", "packages"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_seed_present(self):
        snap = ReproducibilitySnapshot.capture(seed=99)
        assert snap.to_dict()["seed"] == 99

    def test_str_contains_timestamp(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.timestamp[:10] in str(snap)

    def test_git_commit_none_or_str(self):
        snap = ReproducibilitySnapshot.capture()
        assert snap.git_commit is None or isinstance(snap.git_commit, str)

    def test_git_dirty_is_bool(self):
        snap = ReproducibilitySnapshot.capture()
        assert isinstance(snap.git_dirty, bool)

    def test_package_versions_all_strings(self):
        snap = ReproducibilitySnapshot.capture()
        for k, v in snap.package_versions.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# ExperimentRunner._build_leaderboard tests
# (No real dataset needed — we unit-test the Markdown generation logic)
# ---------------------------------------------------------------------------

class TestLeaderboard:
    @pytest.fixture
    def runner(self, tmp_path):
        from eovot.experiment.runner import ExperimentRunner
        return ExperimentRunner(output_dir=str(tmp_path))

    def _make_result(self, tracker, miou, fps=100.0, mem=10.0, n=5):
        return {
            "summary": {
                "tracker": tracker,
                "dataset": "FakeOTB",
                "mean_iou": miou,
                "mean_fps": fps,
                "peak_memory_mb": mem,
                "num_sequences": n,
            }
        }

    def test_sorted_descending_by_miou(self, runner):
        results = [
            self._make_result("B", 0.4),
            self._make_result("A", 0.7),
            self._make_result("C", 0.2),
        ]
        table = runner._build_leaderboard(results)
        # Extract data rows (skip header lines)
        rows = [l for l in table.splitlines() if l.startswith("| ") and "Rank" not in l and "---" not in l]
        assert rows[0].startswith("| 1 |")
        assert "A" in rows[0]
        assert "C" in rows[-1]

    def test_rank_column_present(self, runner):
        results = [self._make_result("MOSSE", 0.5)]
        table = runner._build_leaderboard(results)
        assert "| Rank |" in table
        assert "| 1 |" in table

    def test_empty_returns_message(self, runner):
        out = runner._build_leaderboard([])
        assert "No results" in out

    def test_single_tracker(self, runner):
        results = [self._make_result("KCF", 0.55)]
        table = runner._build_leaderboard(results)
        assert "KCF" in table
        assert "0.5500" in table

    def test_miou_formatted_to_4dp(self, runner):
        results = [self._make_result("T", 0.123456789)]
        table = runner._build_leaderboard(results)
        assert "0.1235" in table

    def test_all_trackers_present(self, runner):
        names = ["Alpha", "Beta", "Gamma"]
        results = [self._make_result(n, 0.5 - i * 0.1) for i, n in enumerate(names)]
        table = runner._build_leaderboard(results)
        for n in names:
            assert n in table

    def test_leaderboard_is_valid_markdown(self, runner):
        results = [self._make_result("X", 0.6), self._make_result("Y", 0.3)]
        table = runner._build_leaderboard(results)
        lines = table.splitlines()
        header_idx = next(i for i, l in enumerate(lines) if "Rank" in l)
        sep_line = lines[header_idx + 1]
        assert "---" in sep_line
        assert sep_line.startswith("|")


# ---------------------------------------------------------------------------
# ExperimentRunner._build_tracker tests
# ---------------------------------------------------------------------------

class TestBuildTracker:
    @pytest.fixture
    def runner(self, tmp_path):
        from eovot.experiment.runner import ExperimentRunner
        return ExperimentRunner(output_dir=str(tmp_path))

    def test_build_mosse(self, runner):
        from eovot.trackers.mosse import MOSSETracker
        tracker = runner._build_tracker({"name": "MOSSE", "params": {}})
        assert isinstance(tracker, MOSSETracker)

    def test_build_kcf(self, runner):
        from eovot.trackers.kcf import KCFTracker
        tracker = runner._build_tracker({"name": "KCF", "params": {}})
        assert isinstance(tracker, KCFTracker)

    def test_build_with_params(self, runner):
        from eovot.trackers.mosse import MOSSETracker
        tracker = runner._build_tracker({
            "name": "MOSSE",
            "params": {"learning_rate": 0.2, "sigma": 3.0},
        })
        assert isinstance(tracker, MOSSETracker)

    def test_unknown_tracker_raises(self, runner):
        with pytest.raises(ValueError, match="Unknown tracker"):
            runner._build_tracker({"name": "FakeTracker", "params": {}})

    def test_none_params_treated_as_empty(self, runner):
        from eovot.trackers.kcf import KCFTracker
        tracker = runner._build_tracker({"name": "KCF", "params": None})
        assert isinstance(tracker, KCFTracker)
