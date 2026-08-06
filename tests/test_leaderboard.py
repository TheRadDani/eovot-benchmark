"""Tests for LeaderboardAggregator and Leaderboard."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from eovot.results.leaderboard import Leaderboard, LeaderboardAggregator, LeaderboardRow


# ---------------------------------------------------------------------------
# Minimal fake BenchmarkResult
# ---------------------------------------------------------------------------

class _FakeResult:
    """Minimal stand-in for BenchmarkResult — provides summary() only."""

    def __init__(self, tracker: str, dataset: str, **metrics) -> None:
        self._summary = {
            "tracker": tracker,
            "dataset": dataset,
            "num_sequences": metrics.pop("num_sequences", 5),
            **metrics,
        }

    def summary(self) -> Dict[str, Any]:
        return dict(self._summary)


def _fake_json(tracker: str, dataset: str, **metrics) -> Dict[str, Any]:
    """Build the dict structure saved by BenchmarkResult.save()."""
    return {
        "summary": {
            "tracker": tracker,
            "dataset": dataset,
            "num_sequences": metrics.pop("num_sequences", 5),
            **metrics,
        },
        "sequences": [],
    }


# ---------------------------------------------------------------------------
# LeaderboardAggregator.add / basic state
# ---------------------------------------------------------------------------

class TestAggregatorAdd:
    def test_add_increases_length(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.5, mean_fps=200.0, peak_memory_mb=50.0))
        assert len(agg) == 1

    def test_duplicate_key_overwrites(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.5, mean_fps=200.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.9, mean_fps=300.0, peak_memory_mb=60.0))
        assert len(agg) == 1
        board = agg.build("OTB100")
        assert board.rows[0].metrics["mean_iou"] == pytest.approx(0.9)

    def test_available_datasets(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "OTB100", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("B", "GOT10k", mean_iou=0.4, mean_fps=80.0, peak_memory_mb=60.0))
        assert set(agg.available_datasets()) == {"OTB100", "GOT10k"}

    def test_available_trackers(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("KCF", "OTB100", mean_iou=0.6, mean_fps=90.0, peak_memory_mb=60.0))
        assert set(agg.available_trackers()) == {"MOSSE", "KCF"}


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_load_single_json_file(self, tmp_path):
        p = tmp_path / "mosse.json"
        p.write_text(json.dumps(_fake_json("MOSSE", "OTB100", mean_iou=0.55,
                                           mean_fps=150.0, peak_memory_mb=40.0)))
        agg = LeaderboardAggregator()
        agg.load_json(p)
        assert len(agg) == 1

    def test_load_json_missing_file_raises(self, tmp_path):
        agg = LeaderboardAggregator()
        with pytest.raises(FileNotFoundError):
            agg.load_json(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# load_dir
# ---------------------------------------------------------------------------

class TestLoadDir:
    def test_load_dir_returns_count(self, tmp_path):
        for name in ["mosse", "kcf", "csrt"]:
            (tmp_path / f"{name}.json").write_text(
                json.dumps(_fake_json(name.upper(), "OTB100",
                                     mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
            )
        agg = LeaderboardAggregator()
        n = agg.load_dir(tmp_path)
        assert n == 3

    def test_load_dir_skips_malformed_json(self, tmp_path):
        (tmp_path / "good.json").write_text(
            json.dumps(_fake_json("MOSSE", "OTB100",
                                  mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
        )
        (tmp_path / "bad.json").write_text("{not valid json")
        agg = LeaderboardAggregator()
        n = agg.load_dir(tmp_path)
        assert n == 1

    def test_load_dir_nonexistent_raises(self, tmp_path):
        agg = LeaderboardAggregator()
        with pytest.raises(FileNotFoundError):
            agg.load_dir(tmp_path / "does_not_exist")

    def test_load_dir_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.json").write_text(
            json.dumps(_fake_json("A", "D1", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
        )
        (sub / "b.json").write_text(
            json.dumps(_fake_json("B", "D1", mean_iou=0.4, mean_fps=90.0, peak_memory_mb=60.0))
        )
        agg = LeaderboardAggregator()
        n = agg.load_dir(tmp_path, recursive=True)
        assert n == 2


# ---------------------------------------------------------------------------
# build — ranking
# ---------------------------------------------------------------------------

class TestBuildRanking:
    def _populated_agg(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.50, mean_fps=200.0, peak_memory_mb=40.0))
        agg.add(_FakeResult("KCF",   "OTB100", mean_iou=0.65, mean_fps=150.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("CSRT",  "OTB100", mean_iou=0.72, mean_fps=80.0,  peak_memory_mb=60.0))
        return agg

    def test_top_rank_has_highest_iou(self):
        board = self._populated_agg().build("OTB100", sort_by="mean_iou")
        assert board.rows[0].tracker_name == "CSRT"
        assert board.rows[0].rank == 1

    def test_bottom_rank_has_lowest_iou(self):
        board = self._populated_agg().build("OTB100", sort_by="mean_iou")
        assert board.rows[-1].tracker_name == "MOSSE"
        assert board.rows[-1].rank == 3

    def test_ascending_sort_for_memory(self):
        board = self._populated_agg().build(
            "OTB100", sort_by="peak_memory_mb", higher_is_better=False
        )
        assert board.rows[0].tracker_name == "MOSSE"  # lowest memory = rank 1

    def test_no_results_raises(self):
        agg = LeaderboardAggregator()
        with pytest.raises(ValueError, match="No results found"):
            agg.build("OTB100")

    def test_dataset_filter(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "OTB100", mean_iou=0.7, mean_fps=100.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("B", "GOT10k", mean_iou=0.6, mean_fps=90.0,  peak_memory_mb=60.0))
        board = agg.build("OTB100")
        assert len(board) == 1
        assert board.rows[0].tracker_name == "A"


# ---------------------------------------------------------------------------
# build — baseline / deltas
# ---------------------------------------------------------------------------

class TestBuildBaseline:
    def _agg(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.50, mean_fps=200.0, peak_memory_mb=40.0))
        agg.add(_FakeResult("KCF",   "OTB100", mean_iou=0.65, mean_fps=150.0, peak_memory_mb=50.0))
        return agg

    def test_baseline_row_has_empty_deltas(self):
        board = self._agg().build("OTB100", baseline="MOSSE")
        mosse = next(r for r in board.rows if r.tracker_name == "MOSSE")
        assert mosse.deltas == {}

    def test_non_baseline_row_has_deltas(self):
        board = self._agg().build("OTB100", baseline="MOSSE")
        kcf = next(r for r in board.rows if r.tracker_name == "KCF")
        assert "mean_iou" in kcf.deltas
        assert kcf.deltas["mean_iou"] == pytest.approx(0.65 - 0.50, abs=1e-6)

    def test_missing_baseline_raises(self):
        with pytest.raises(ValueError, match="Baseline tracker"):
            self._agg().build("OTB100", baseline="Nonexistent")

    def test_negative_delta_for_worse_tracker(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("Strong", "D1", mean_iou=0.9, mean_fps=100.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("Weak",   "D1", mean_iou=0.3, mean_fps=80.0,  peak_memory_mb=40.0))
        board = agg.build("D1", baseline="Strong")
        weak = next(r for r in board.rows if r.tracker_name == "Weak")
        assert weak.deltas["mean_iou"] < 0.0


# ---------------------------------------------------------------------------
# Leaderboard.to_markdown
# ---------------------------------------------------------------------------

class TestMarkdownOutput:
    def _board(self, baseline=None):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "OTB100", mean_iou=0.50, mean_fps=200.0, peak_memory_mb=40.0))
        agg.add(_FakeResult("KCF",   "OTB100", mean_iou=0.65, mean_fps=150.0, peak_memory_mb=50.0))
        return agg.build("OTB100", baseline=baseline)

    def test_markdown_contains_tracker_names(self):
        md = self._board().to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_markdown_contains_metric_names(self):
        md = self._board().to_markdown()
        assert "mean_iou" in md
        assert "mean_fps" in md

    def test_markdown_delta_columns_present_with_baseline(self):
        md = self._board(baseline="MOSSE").to_markdown(show_delta=True)
        assert "Δ mean_iou" in md

    def test_markdown_no_delta_without_baseline(self):
        md = self._board(baseline=None).to_markdown(show_delta=True)
        assert "Δ" not in md

    def test_markdown_header_row_present(self):
        md = self._board().to_markdown()
        assert "Rank" in md
        assert "Tracker" in md


# ---------------------------------------------------------------------------
# Leaderboard.save_csv
# ---------------------------------------------------------------------------

class TestSaveCsv:
    def test_csv_file_created(self, tmp_path):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "DS", mean_iou=0.6, mean_fps=100.0, peak_memory_mb=50.0))
        board = agg.build("DS")
        out = board.save_csv(tmp_path / "board")  # no extension provided
        assert out.suffix == ".csv"
        assert out.exists()

    def test_csv_contains_tracker_name(self, tmp_path):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("MOSSE", "DS", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
        board = agg.build("DS")
        out = board.save_csv(tmp_path / "board.csv")
        content = out.read_text()
        assert "MOSSE" in content

    def test_csv_has_header_row(self, tmp_path):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "DS", mean_iou=0.6, mean_fps=100.0, peak_memory_mb=50.0))
        board = agg.build("DS")
        out = board.save_csv(tmp_path / "board.csv")
        first_line = out.read_text().split("\n")[0]
        assert "rank" in first_line
        assert "tracker_name" in first_line


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------

class TestBuildAll:
    def test_build_all_returns_dict_keyed_by_dataset(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "DS1", mean_iou=0.7, mean_fps=100.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("B", "DS2", mean_iou=0.6, mean_fps=90.0,  peak_memory_mb=60.0))
        boards = agg.build_all()
        assert set(boards.keys()) == {"DS1", "DS2"}

    def test_each_board_is_leaderboard(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "DS1", mean_iou=0.7, mean_fps=100.0, peak_memory_mb=50.0))
        boards = agg.build_all()
        assert isinstance(boards["DS1"], Leaderboard)


# ---------------------------------------------------------------------------
# LeaderboardRow helpers
# ---------------------------------------------------------------------------

class TestLeaderboardRow:
    def test_metric_returns_value(self):
        row = LeaderboardRow(
            rank=1, tracker_name="T", dataset_name="D",
            metrics={"mean_iou": 0.75}, num_sequences=10
        )
        assert row.metric("mean_iou") == pytest.approx(0.75)

    def test_metric_returns_none_for_missing(self):
        row = LeaderboardRow(
            rank=1, tracker_name="T", dataset_name="D",
            metrics={}, num_sequences=10
        )
        assert row.metric("not_present") is None

    def test_delta_returns_none_when_empty(self):
        row = LeaderboardRow(rank=1, tracker_name="T", dataset_name="D", metrics={})
        assert row.delta("mean_iou") is None


# ---------------------------------------------------------------------------
# to_dict round-trip
# ---------------------------------------------------------------------------

class TestToDictRoundtrip:
    def test_to_dict_has_expected_keys(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("X", "DS", mean_iou=0.5, mean_fps=100.0, peak_memory_mb=50.0))
        d = agg.build("DS").to_dict()
        assert "rows" in d
        assert "dataset_name" in d
        assert "sort_metric" in d

    def test_to_dict_rows_match_count(self):
        agg = LeaderboardAggregator()
        agg.add(_FakeResult("A", "DS", mean_iou=0.6, mean_fps=100.0, peak_memory_mb=50.0))
        agg.add(_FakeResult("B", "DS", mean_iou=0.5, mean_fps=90.0,  peak_memory_mb=60.0))
        d = agg.build("DS").to_dict()
        assert len(d["rows"]) == 2
