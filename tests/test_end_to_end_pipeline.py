"""End-to-end integration tests for the EOVOT benchmark pipeline.

Covers the full YAML-driven flow via ExperimentRunner.run_from_config()
using the built-in SyntheticDataset so no external data download is needed.
Also validates the --synthetic flag of scripts/run_benchmark.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow importing scripts/ when running tests from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Minimal synthetic experiment config shared across test cases
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, trackers=None, **dataset_overrides) -> dict:
    if trackers is None:
        trackers = [{"name": "MOSSE", "params": {}}]
    dataset = {
        "loader": "SyntheticDataset",
        "name": "Synthetic-Test",
        "num_sequences": 3,
        "num_frames": 20,
        "frame_size": [160, 120],
        "bbox_size": [20, 20],
        "motion": "linear",
        "seed": 0,
    }
    dataset.update(dataset_overrides)
    return {
        "experiment": {
            "name": "e2e-test",
            "seed": 1,
            "tdp_watts": None,
        },
        "dataset": dataset,
        "trackers": trackers,
        "benchmark": {"verbose": False},
        "reporting": {"formats": ["json"], "print_summary": False},
    }


# ---------------------------------------------------------------------------
# ExperimentRunner.run_from_config — single tracker
# ---------------------------------------------------------------------------

class TestExperimentRunnerSingleTracker:
    @pytest.fixture
    def runner(self, tmp_path):
        from eovot.experiment.runner import ExperimentRunner
        return ExperimentRunner(output_dir=str(tmp_path), verbose=False)

    @pytest.fixture
    def output(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        return runner.run_from_config(cfg)

    def test_returns_dict_with_required_keys(self, output):
        assert "metadata" in output
        assert "results" in output
        assert "leaderboard" in output

    def test_results_list_length(self, output):
        assert len(output["results"]) == 1

    def test_result_has_summary(self, output):
        r = output["results"][0]
        assert "summary" in r

    def test_summary_has_tracker_name(self, output):
        s = output["results"][0]["summary"]
        assert s["tracker"] == "MOSSE"

    def test_summary_mean_iou_in_range(self, output):
        s = output["results"][0]["summary"]
        assert 0.0 <= s["mean_iou"] <= 1.0

    def test_summary_fps_positive(self, output):
        s = output["results"][0]["summary"]
        assert s["mean_fps"] > 0.0

    def test_summary_num_sequences(self, output):
        s = output["results"][0]["summary"]
        assert s["num_sequences"] == 3

    def test_sequences_list_length(self, output):
        seqs = output["results"][0]["sequences"]
        assert len(seqs) == 3

    def test_leaderboard_contains_tracker(self, output):
        assert "MOSSE" in output["leaderboard"]

    def test_metadata_experiment_name(self, output):
        assert output["metadata"]["experiment"] == "e2e-test"

    def test_metadata_reproducibility_keys(self, output):
        snap = output["metadata"]["reproducibility"]
        assert "timestamp" in snap
        assert "python_version" in snap

    def test_output_dir_contains_json(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        runner.run_from_config(cfg)
        jsons = list((tmp_path / "e2e-test").glob("*.json"))
        assert len(jsons) >= 1

    def test_leaderboard_md_written(self, runner, tmp_path):
        cfg = _make_config(tmp_path)
        runner.run_from_config(cfg)
        lb_path = tmp_path / "e2e-test" / "leaderboard.md"
        assert lb_path.exists()
        assert "MOSSE" in lb_path.read_text()


# ---------------------------------------------------------------------------
# ExperimentRunner.run_from_config — multiple trackers (comparison)
# ---------------------------------------------------------------------------

class TestExperimentRunnerMultiTracker:
    @pytest.fixture
    def output(self, tmp_path):
        from eovot.experiment.runner import ExperimentRunner
        trackers = [
            {"name": "MOSSE", "params": {}},
            {"name": "KCF", "params": {}},
        ]
        cfg = _make_config(tmp_path, trackers=trackers)
        runner = ExperimentRunner(output_dir=str(tmp_path), verbose=False)
        return runner.run_from_config(cfg)

    def test_two_results_returned(self, output):
        assert len(output["results"]) == 2

    def test_both_trackers_in_leaderboard(self, output):
        lb = output["leaderboard"]
        assert "MOSSE" in lb
        assert "KCF" in lb

    def test_leaderboard_ranked_descending(self, output):
        lb = output["leaderboard"]
        lines = [l for l in lb.splitlines() if l.startswith("| ") and "Rank" not in l and "---" not in l]
        # The first data row should be rank 1 (highest AUC).
        assert lines[0].startswith("| 1 |")

    def test_per_tracker_json_files_exist(self, tmp_path):
        from eovot.experiment.runner import ExperimentRunner
        trackers = [{"name": "MOSSE", "params": {}}, {"name": "KCF", "params": {}}]
        cfg = _make_config(tmp_path, trackers=trackers)
        runner = ExperimentRunner(output_dir=str(tmp_path), verbose=False)
        runner.run_from_config(cfg)
        exp_dir = tmp_path / "e2e-test"
        jsons = {p.stem for p in exp_dir.glob("*.json") if p.stem != "metadata"}
        assert any("MOSSE" in s for s in jsons)
        assert any("KCF" in s for s in jsons)


# ---------------------------------------------------------------------------
# Motion patterns — each SyntheticDataset motion type works end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("motion", ["linear", "circular", "random"])
def test_motion_patterns(tmp_path, motion):
    from eovot.experiment.runner import ExperimentRunner
    cfg = _make_config(tmp_path, motion=motion)
    runner = ExperimentRunner(output_dir=str(tmp_path), verbose=False)
    output = runner.run_from_config(cfg)
    s = output["results"][0]["summary"]
    assert 0.0 <= s["mean_iou"] <= 1.0


# ---------------------------------------------------------------------------
# Resume mode — skip already-completed trackers
# ---------------------------------------------------------------------------

class TestResume:
    def test_resume_skips_existing_result(self, tmp_path):
        from eovot.experiment.runner import ExperimentRunner
        cfg = _make_config(tmp_path)
        runner = ExperimentRunner(output_dir=str(tmp_path), verbose=False)

        # First run: writes the result file.
        out1 = runner.run_from_config(cfg)

        # Second run with resume=True: the tracker is already done.
        runner2 = ExperimentRunner(output_dir=str(tmp_path), verbose=False, resume=True)
        out2 = runner2.run_from_config(cfg)

        assert len(out2["results"]) == 1
        # mIoU must match (loaded from disk, not re-run).
        assert (out1["results"][0]["summary"]["mean_iou"]
                == out2["results"][0]["summary"]["mean_iou"])


# ---------------------------------------------------------------------------
# scripts/run_benchmark.py --synthetic CLI flag
# ---------------------------------------------------------------------------

class TestSyntheticCLI:
    """Validate the --synthetic flag in the run_benchmark script."""

    def _run_cli(self, args: list, tmp_path: Path) -> dict:
        """Invoke run_benchmark.main() with the given args and return the summary JSON."""
        out_dir = str(tmp_path / "results")
        full_args = ["eovot", "--tracker", "MOSSE",
                     "--synthetic",
                     "--synthetic-sequences", "2",
                     "--synthetic-frames", "15",
                     "--output-dir", out_dir,
                     "--quiet"] + args
        with patch("sys.argv", full_args):
            from scripts.run_benchmark import main
            main()
        json_files = list(Path(out_dir).glob("*.json"))
        assert json_files, "Expected at least one JSON output from CLI run"
        with open(json_files[0]) as fh:
            return json.load(fh)

    def test_synthetic_flag_produces_output(self, tmp_path):
        summary = self._run_cli([], tmp_path)
        assert "tracker" in summary or "mean_iou" in summary

    def test_synthetic_flag_tracker_name(self, tmp_path):
        summary = self._run_cli([], tmp_path)
        assert summary.get("tracker") == "MOSSE"

    def test_synthetic_motion_linear(self, tmp_path):
        summary = self._run_cli(["--synthetic-motion", "linear"], tmp_path)
        assert 0.0 <= summary["mean_iou"] <= 1.0

    def test_synthetic_motion_circular(self, tmp_path):
        summary = self._run_cli(["--synthetic-motion", "circular"], tmp_path)
        assert 0.0 <= summary["mean_iou"] <= 1.0

    def test_synthetic_motion_random(self, tmp_path):
        summary = self._run_cli(["--synthetic-motion", "random"], tmp_path)
        assert 0.0 <= summary["mean_iou"] <= 1.0

    def test_synthetic_num_sequences_respected(self, tmp_path):
        summary = self._run_cli(["--synthetic-sequences", "2"], tmp_path)
        assert summary["num_sequences"] == 2

    def test_synthetic_size_flag_accepted(self, tmp_path):
        """--synthetic-size should not raise an error."""
        summary = self._run_cli(["--synthetic-size", "160", "120"], tmp_path)
        assert summary is not None
