"""Unit and integration tests for HyperparamSweep."""

from __future__ import annotations

import json
import pytest

from eovot.experiment.sweep import HyperparamSweep, SweepEntry, SweepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_dataset():
    from eovot.datasets.synthetic import SyntheticDataset
    return SyntheticDataset(num_sequences=2, num_frames=20, seed=0)


def _make_entry(lr=0.1, miou=0.5, fps=100.0):
    return SweepEntry(
        params={"learning_rate": lr},
        mean_iou=miou,
        success_auc=miou,
        precision_auc=0.6,
        mean_fps=fps,
        peak_memory_mb=50.0,
        wall_time_s=1.0,
        result_dict={},
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestHyperparamSweepInit:
    def test_defaults(self):
        s = HyperparamSweep("KCF", {"learning_rate": [0.1]})
        assert s.tracker_name == "KCF"
        assert s.primary_metric == "mean_iou"
        assert s.output_dir is None
        assert not s.resume

    def test_invalid_primary_metric_raises(self):
        with pytest.raises(ValueError, match="primary_metric must be one of"):
            HyperparamSweep("KCF", {}, primary_metric="bad_metric")

    def test_valid_primary_metrics_accepted(self):
        for m in ("mean_iou", "success_auc", "precision_auc", "mean_fps"):
            s = HyperparamSweep("KCF", {}, primary_metric=m)
            assert s.primary_metric == m

    def test_output_dir_created_on_init(self, tmp_path):
        out = tmp_path / "sweep_out"
        HyperparamSweep("KCF", {}, output_dir=str(out))
        assert out.exists()

    def test_no_output_dir_is_fine(self):
        s = HyperparamSweep("KCF", {})
        assert s.output_dir is None

    def test_resume_default_false(self):
        s = HyperparamSweep("KCF", {})
        assert s.resume is False

    def test_resume_can_be_set(self):
        s = HyperparamSweep("KCF", {}, resume=True)
        assert s.resume is True


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------

class TestCombinations:
    def test_empty_grid_returns_one_empty_combo(self):
        assert HyperparamSweep("KCF", {}).combinations() == [{}]

    def test_single_param_length(self):
        combos = HyperparamSweep("KCF", {"lr": [0.1, 0.2, 0.3]}).combinations()
        assert len(combos) == 3

    def test_single_param_values(self):
        combos = HyperparamSweep("KCF", {"lr": [0.1, 0.2]}).combinations()
        assert {"lr": 0.1} in combos
        assert {"lr": 0.2} in combos

    def test_two_params_cartesian(self):
        combos = HyperparamSweep("KCF", {"a": [1, 2], "b": [3, 4]}).combinations()
        assert len(combos) == 4

    def test_three_params_cartesian(self):
        combos = HyperparamSweep(
            "KCF", {"a": [1, 2], "b": [3], "c": [4, 5]}
        ).combinations()
        assert len(combos) == 4

    def test_single_value_grid(self):
        combos = HyperparamSweep("KCF", {"lr": [0.125]}).combinations()
        assert len(combos) == 1
        assert combos[0] == {"lr": 0.125}

    def test_two_single_values(self):
        combos = HyperparamSweep("KCF", {"a": [1], "b": [2]}).combinations()
        assert len(combos) == 1
        assert combos[0] == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_single_entry_on_front(self):
        e = _make_entry()
        assert e in HyperparamSweep._pareto_front([e])

    def test_dominated_entry_excluded(self):
        a = _make_entry(miou=0.8, fps=200)
        b = _make_entry(lr=0.2, miou=0.5, fps=100)
        front = HyperparamSweep._pareto_front([a, b])
        assert a in front
        assert b not in front

    def test_no_dominance_both_on_front(self):
        a = _make_entry(miou=0.8, fps=50)
        b = _make_entry(lr=0.2, miou=0.4, fps=200)
        front = HyperparamSweep._pareto_front([a, b])
        assert len(front) == 2

    def test_equal_entries_both_on_front(self):
        a = _make_entry(lr=0.1, miou=0.5, fps=100)
        b = _make_entry(lr=0.2, miou=0.5, fps=100)
        front = HyperparamSweep._pareto_front([a, b])
        assert len(front) == 2

    def test_fully_dominated_chain(self):
        best = _make_entry(lr=0.1, miou=0.9, fps=300)
        mid = _make_entry(lr=0.2, miou=0.5, fps=100)
        worst = _make_entry(lr=0.3, miou=0.2, fps=30)
        front = HyperparamSweep._pareto_front([best, mid, worst])
        assert best in front
        assert mid not in front
        assert worst not in front

    def test_three_point_all_on_front(self):
        high_iou = _make_entry(lr=0.1, miou=0.9, fps=30)
        balanced = _make_entry(lr=0.2, miou=0.6, fps=100)
        high_fps = _make_entry(lr=0.3, miou=0.3, fps=300)
        front = HyperparamSweep._pareto_front([high_iou, balanced, high_fps])
        assert len(front) == 3

    def test_empty_list_returns_empty(self):
        assert HyperparamSweep._pareto_front([]) == []


# ---------------------------------------------------------------------------
# SweepEntry dataclass
# ---------------------------------------------------------------------------

class TestSweepEntry:
    def test_params_str_alphabetical(self):
        e = SweepEntry(
            params={"sigma": 1.0, "learning_rate": 0.1},
            mean_iou=0.5, success_auc=0.5, precision_auc=0.6,
            mean_fps=100.0, peak_memory_mb=50.0, wall_time_s=1.0,
            result_dict={},
        )
        assert e.params_str.index("learning_rate") < e.params_str.index("sigma")

    def test_params_str_empty_params(self):
        e = _make_entry()
        e.params = {}
        assert e.params_str == ""

    def test_params_str_single_key(self):
        e = _make_entry(lr=0.125)
        assert "learning_rate=0.125" in e.params_str


# ---------------------------------------------------------------------------
# SweepResult dataclass
# ---------------------------------------------------------------------------

class TestSweepResult:
    def _make_result(self, n=2):
        entries = [
            _make_entry(lr=0.1 * i, miou=0.3 + 0.1 * i, fps=80.0 + 20 * i)
            for i in range(1, n + 1)
        ]
        return SweepResult("KCF", {}, "mean_iou", entries, entries[-1], [entries[-1]])

    def test_to_markdown_has_header_columns(self):
        sr = self._make_result()
        md = sr.to_markdown()
        assert "mIoU" in md
        assert "FPS" in md
        assert "Parameters" in md
        assert "Pareto" in md

    def test_to_markdown_has_pareto_mark(self):
        sr = self._make_result()
        assert "✓" in sr.to_markdown()

    def test_to_markdown_shows_best_line(self):
        sr = self._make_result()
        assert "Best" in sr.to_markdown()

    def test_to_markdown_shows_pareto_count(self):
        sr = self._make_result()
        assert "Pareto-optimal" in sr.to_markdown()

    def test_to_dict_required_keys(self):
        sr = self._make_result()
        d = sr.to_dict()
        for key in ("tracker_name", "param_grid", "primary_metric", "best_params", "entries"):
            assert key in d

    def test_to_dict_entry_keys(self):
        sr = self._make_result()
        entry_d = sr.to_dict()["entries"][0]
        for key in ("params", "mean_iou", "success_auc", "mean_fps", "peak_memory_mb"):
            assert key in entry_d

    def test_to_dict_best_params_matches_best_entry(self):
        sr = self._make_result()
        assert sr.to_dict()["best_params"] == sr.best.params

    def test_to_dict_tracker_name(self):
        sr = self._make_result()
        assert sr.to_dict()["tracker_name"] == "KCF"


# ---------------------------------------------------------------------------
# Integration: run() with SyntheticDataset
# ---------------------------------------------------------------------------

class TestSweepRunIntegration:
    def test_empty_grid_returns_one_entry(self):
        sweep = HyperparamSweep("MOSSE", {}, verbose=False)
        result = sweep.run(_small_dataset())
        assert len(result.entries) == 1

    def test_single_param_correct_count(self):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.05, 0.125, 0.25]}, verbose=False
        )
        result = sweep.run(_small_dataset())
        assert len(result.entries) == 3

    def test_two_params_cartesian_count(self):
        sweep = HyperparamSweep(
            "MOSSE",
            {"learning_rate": [0.05, 0.25], "sigma": [1.0, 2.0]},
            verbose=False,
        )
        result = sweep.run(_small_dataset())
        assert len(result.entries) == 4

    def test_entries_sorted_descending_by_mean_iou(self):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.05, 0.125, 0.25]}, verbose=False
        )
        result = sweep.run(_small_dataset())
        ious = [e.mean_iou for e in result.entries]
        assert ious == sorted(ious, reverse=True)

    def test_entries_sorted_by_fps_when_requested(self):
        sweep = HyperparamSweep(
            "MOSSE",
            {"learning_rate": [0.05, 0.25]},
            primary_metric="mean_fps",
            verbose=False,
        )
        result = sweep.run(_small_dataset())
        fps_vals = [e.mean_fps for e in result.entries]
        assert fps_vals == sorted(fps_vals, reverse=True)

    def test_best_is_first_entry(self):
        sweep = HyperparamSweep(
            "KCF", {"learning_rate": [0.05, 0.15]}, verbose=False
        )
        result = sweep.run(_small_dataset())
        assert result.best is result.entries[0]

    def test_pareto_front_non_empty(self):
        sweep = HyperparamSweep("MOSSE", {"learning_rate": [0.05, 0.25]}, verbose=False)
        result = sweep.run(_small_dataset())
        assert len(result.pareto_front) >= 1

    def test_pareto_entries_are_subset_of_all(self):
        sweep = HyperparamSweep("MOSSE", {"learning_rate": [0.05, 0.25]}, verbose=False)
        result = sweep.run(_small_dataset())
        all_ids = {id(e) for e in result.entries}
        for p in result.pareto_front:
            assert id(p) in all_ids

    def test_fps_positive(self):
        sweep = HyperparamSweep("MOSSE", {"learning_rate": [0.125]}, verbose=False)
        for e in sweep.run(_small_dataset()).entries:
            assert e.mean_fps > 0

    def test_iou_in_unit_interval(self):
        sweep = HyperparamSweep("MOSSE", {"learning_rate": [0.125]}, verbose=False)
        for e in sweep.run(_small_dataset()).entries:
            assert 0.0 <= e.mean_iou <= 1.0

    def test_result_tracker_name(self):
        sweep = HyperparamSweep("MOSSE", {}, verbose=False)
        result = sweep.run(_small_dataset())
        assert result.tracker_name == "MOSSE"

    def test_result_primary_metric(self):
        sweep = HyperparamSweep("MOSSE", {}, primary_metric="success_auc", verbose=False)
        result = sweep.run(_small_dataset())
        assert result.primary_metric == "success_auc"

    def test_output_dir_summary_created(self, tmp_path):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.1, 0.2]},
            output_dir=str(tmp_path / "s"), verbose=False,
        )
        sweep.run(_small_dataset())
        assert (tmp_path / "s" / "sweep_summary.json").exists()

    def test_output_dir_leaderboard_md_created(self, tmp_path):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.125]},
            output_dir=str(tmp_path / "s"), verbose=False,
        )
        sweep.run(_small_dataset())
        assert (tmp_path / "s" / "sweep_leaderboard.md").exists()

    def test_output_combo_files_count(self, tmp_path):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.1, 0.2]},
            output_dir=str(tmp_path / "s"), verbose=False,
        )
        sweep.run(_small_dataset())
        assert len(list((tmp_path / "s").glob("combo_*.json"))) == 2

    def test_resume_skips_cached_combos(self, tmp_path):
        out = tmp_path / "sweep"
        sweep1 = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.1, 0.2]},
            output_dir=str(out), verbose=False,
        )
        r1 = sweep1.run(_small_dataset())

        sweep2 = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.1, 0.2]},
            output_dir=str(out), verbose=False, resume=True,
        )
        r2 = sweep2.run(_small_dataset())
        assert len(r2.entries) == len(r1.entries)

    def test_summary_json_parseable(self, tmp_path):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.125]},
            output_dir=str(tmp_path / "s"), verbose=False,
        )
        sweep.run(_small_dataset())
        data = json.loads((tmp_path / "s" / "sweep_summary.json").read_text())
        assert "best_params" in data

    def test_leaderboard_md_has_table(self, tmp_path):
        sweep = HyperparamSweep(
            "MOSSE", {"learning_rate": [0.125]},
            output_dir=str(tmp_path / "s"), verbose=False,
        )
        sweep.run(_small_dataset())
        md = (tmp_path / "s" / "sweep_leaderboard.md").read_text()
        assert "|" in md

    def test_kcf_sweep_runs_successfully(self):
        sweep = HyperparamSweep(
            "KCF", {"learning_rate": [0.05, 0.10]}, verbose=False
        )
        result = sweep.run(_small_dataset())
        assert len(result.entries) == 2

    def test_wall_time_non_negative(self):
        sweep = HyperparamSweep("MOSSE", {"learning_rate": [0.125]}, verbose=False)
        result = sweep.run(_small_dataset())
        assert result.entries[0].wall_time_s >= 0.0
