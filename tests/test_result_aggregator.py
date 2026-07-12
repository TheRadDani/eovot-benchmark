"""Unit tests for eovot.reporting.aggregator."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from eovot.reporting.aggregator import (
    ResultAggregator,
    TrackerSummary,
    _composite_score,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_summary(
    tracker: str = "KCF",
    dataset: str = "OTB100",
    *,
    mean_iou: float = 0.55,
    success_auc: float = 0.60,
    precision_auc: float = 0.72,
    mean_center_distance_px: float = 18.0,
    mean_fps: float = 120.0,
    peak_memory_mb: float = 64.0,
    total_energy_j: float = 0.5,
    mean_energy_per_frame_mj: float = 0.05,
    num_sequences: int = 50,
) -> TrackerSummary:
    return TrackerSummary(
        tracker=tracker,
        dataset=dataset,
        num_sequences=num_sequences,
        mean_iou=mean_iou,
        success_auc=success_auc,
        precision_auc=precision_auc,
        mean_center_distance_px=mean_center_distance_px,
        mean_fps=mean_fps,
        peak_memory_mb=peak_memory_mb,
        total_energy_j=total_energy_j,
        mean_energy_per_frame_mj=mean_energy_per_frame_mj,
    )


def _write_json(path: Path, items: list[dict]) -> None:
    with path.open("w") as fh:
        json.dump(items, fh)


# --------------------------------------------------------------------------- #
# TrackerSummary
# --------------------------------------------------------------------------- #

class TestTrackerSummary:
    def test_from_dict_roundtrip(self):
        s = _make_summary()
        d = s.to_dict()
        s2 = TrackerSummary.from_dict(d)
        assert s2.tracker == s.tracker
        assert s2.mean_iou == pytest.approx(s.mean_iou)
        assert s2.success_auc == pytest.approx(s.success_auc)

    def test_extra_fields_preserved(self):
        d = _make_summary().to_dict()
        d["custom_field"] = "hello"
        s = TrackerSummary.from_dict(d)
        assert s.extra["custom_field"] == "hello"

    def test_to_dict_includes_extra(self):
        s = _make_summary()
        s.extra["note"] = "test"
        d = s.to_dict()
        assert d["note"] == "test"

    def test_to_dict_no_extra_key(self):
        s = _make_summary()
        d = s.to_dict()
        assert "extra" not in d

    def test_partial_from_dict(self):
        # Only required numeric fields present; missing fields use defaults if supplied
        d = {
            "tracker": "MOSSE",
            "dataset": "GOT10k",
            "num_sequences": 10,
            "mean_iou": 0.4,
            "success_auc": 0.45,
            "precision_auc": 0.50,
            "mean_center_distance_px": 25.0,
            "mean_fps": 200.0,
            "peak_memory_mb": 30.0,
            "total_energy_j": 0.2,
            "mean_energy_per_frame_mj": 0.01,
        }
        s = TrackerSummary.from_dict(d)
        assert s.tracker == "MOSSE"


# --------------------------------------------------------------------------- #
# _composite_score
# --------------------------------------------------------------------------- #

class TestCompositeScore:
    def test_range_zero_to_one(self):
        s = _make_summary(mean_iou=0.0, success_auc=0.0, mean_fps=0.0, peak_memory_mb=0.0)
        score = _composite_score(s)
        assert 0.0 <= score <= 1.0

    def test_perfect_score(self):
        s = _make_summary(
            mean_iou=1.0, success_auc=1.0, mean_fps=30.0, peak_memory_mb=0.0
        )
        score = _composite_score(s)
        assert score == pytest.approx(1.0)

    def test_fps_capped_at_scale(self):
        s_fast = _make_summary(mean_fps=300.0)
        s_norm = _make_summary(mean_fps=30.0)
        # Both should yield the same speed component since 300 >> fps_scale=30
        assert _composite_score(s_fast) == pytest.approx(_composite_score(s_norm), abs=1e-6)

    def test_memory_floored_at_zero(self):
        s = _make_summary(peak_memory_mb=9999.0)
        score = _composite_score(s)
        assert score >= 0.0

    def test_weights_respected(self):
        s = _make_summary(mean_iou=1.0, success_auc=1.0, mean_fps=30.0, peak_memory_mb=0.0)
        score = _composite_score(s, accuracy_weight=1.0, speed_weight=0.0, memory_weight=0.0)
        assert score == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# ResultAggregator — loading
# --------------------------------------------------------------------------- #

class TestAggregatorLoading:
    def test_load_valid_json(self, tmp_path):
        p = tmp_path / "results.json"
        _write_json(p, [_make_summary().to_dict()])
        agg = ResultAggregator().load(p)
        assert len(agg.entries()) == 1

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ResultAggregator().load(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            ResultAggregator().load(p)

    def test_load_non_array_json_raises(self, tmp_path):
        p = tmp_path / "obj.json"
        p.write_text(json.dumps({"tracker": "KCF"}))
        with pytest.raises(ValueError):
            ResultAggregator().load(p)

    def test_load_many(self, tmp_path):
        p1 = tmp_path / "r1.json"
        p2 = tmp_path / "r2.json"
        _write_json(p1, [_make_summary("KCF").to_dict()])
        _write_json(p2, [_make_summary("MOSSE").to_dict()])
        agg = ResultAggregator().load_many([p1, p2])
        assert len(agg.entries()) == 2

    def test_add(self):
        agg = ResultAggregator()
        agg.add(_make_summary())
        assert len(agg.entries()) == 1

    def test_chaining(self, tmp_path):
        p = tmp_path / "r.json"
        _write_json(p, [_make_summary().to_dict()])
        agg = ResultAggregator().load(p).add(_make_summary("CSRT"))
        assert len(agg.entries()) == 2


# --------------------------------------------------------------------------- #
# ResultAggregator — filtering
# --------------------------------------------------------------------------- #

class TestAggregatorFiltering:
    def _populated_agg(self) -> ResultAggregator:
        agg = ResultAggregator()
        agg.add(_make_summary("KCF", "OTB100"))
        agg.add(_make_summary("MOSSE", "OTB100"))
        agg.add(_make_summary("KCF", "GOT10k"))
        return agg

    def test_entries_all(self):
        assert len(self._populated_agg().entries()) == 3

    def test_entries_by_dataset(self):
        entries = self._populated_agg().entries(dataset="OTB100")
        assert len(entries) == 2
        assert all(e.dataset == "OTB100" for e in entries)

    def test_entries_by_tracker(self):
        entries = self._populated_agg().entries(tracker="KCF")
        assert len(entries) == 2
        assert all(e.tracker == "KCF" for e in entries)

    def test_entries_by_both(self):
        entries = self._populated_agg().entries(tracker="KCF", dataset="GOT10k")
        assert len(entries) == 1

    def test_entries_no_match(self):
        assert self._populated_agg().entries(dataset="LaSOT") == []


# --------------------------------------------------------------------------- #
# ResultAggregator — ranking
# --------------------------------------------------------------------------- #

class TestAggregatorRanking:
    def _agg_with_three(self) -> ResultAggregator:
        agg = ResultAggregator()
        agg.add(_make_summary("Best", mean_iou=0.80, success_auc=0.85, mean_fps=60.0, peak_memory_mb=32.0))
        agg.add(_make_summary("Mid", mean_iou=0.55, success_auc=0.60, mean_fps=30.0, peak_memory_mb=128.0))
        agg.add(_make_summary("Low", mean_iou=0.30, success_auc=0.35, mean_fps=10.0, peak_memory_mb=256.0))
        return agg

    def test_ranked_descending_composite(self):
        rows = self._agg_with_three().ranked_entries()
        scores = [r["composite_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_present(self):
        rows = self._agg_with_three().ranked_entries()
        assert [r["rank"] for r in rows] == [1, 2, 3]

    def test_sort_by_iou(self):
        rows = self._agg_with_three().ranked_entries(sort_by="mean_iou")
        ious = [r["mean_iou"] for r in rows]
        assert ious == sorted(ious, reverse=True)

    def test_sort_by_memory_ascending(self):
        rows = self._agg_with_three().ranked_entries(sort_by="peak_memory_mb")
        mems = [r["peak_memory_mb"] for r in rows]
        assert mems == sorted(mems)

    def test_sort_by_unknown_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            self._agg_with_three().ranked_entries(sort_by="nonexistent_field")


# --------------------------------------------------------------------------- #
# ResultAggregator — aggregation
# --------------------------------------------------------------------------- #

class TestAggregateByTracker:
    def test_averages_across_datasets(self):
        agg = ResultAggregator()
        agg.add(_make_summary("KCF", "OTB100", mean_iou=0.5))
        agg.add(_make_summary("KCF", "GOT10k", mean_iou=0.7))
        result = agg.aggregate_by_tracker()
        assert "KCF" in result
        assert result["KCF"]["mean_iou"] == pytest.approx(0.6)

    def test_num_datasets(self):
        agg = ResultAggregator()
        agg.add(_make_summary("KCF", "OTB100"))
        agg.add(_make_summary("KCF", "GOT10k"))
        assert agg.aggregate_by_tracker()["KCF"]["num_datasets"] == 2

    def test_separate_trackers(self):
        agg = ResultAggregator()
        agg.add(_make_summary("KCF"))
        agg.add(_make_summary("MOSSE"))
        result = agg.aggregate_by_tracker()
        assert set(result.keys()) == {"KCF", "MOSSE"}

    def test_composite_score_present(self):
        agg = ResultAggregator()
        agg.add(_make_summary())
        result = agg.aggregate_by_tracker()
        assert "composite_score" in list(result.values())[0]


# --------------------------------------------------------------------------- #
# ResultAggregator — leaderboard / cross_dataset_summary
# --------------------------------------------------------------------------- #

class TestLeaderboard:
    def _agg(self) -> ResultAggregator:
        agg = ResultAggregator()
        agg.add(_make_summary("KCF", "OTB100"))
        agg.add(_make_summary("MOSSE", "OTB100"))
        return agg

    def test_returns_string(self):
        assert isinstance(self._agg().leaderboard(), str)

    def test_contains_tracker_names(self):
        lb = self._agg().leaderboard()
        assert "KCF" in lb
        assert "MOSSE" in lb

    def test_top_n(self):
        lb = self._agg().leaderboard(top_n=1)
        assert "| 2 |" not in lb

    def test_dataset_filter(self):
        agg = ResultAggregator()
        agg.add(_make_summary("KCF", "OTB100"))
        agg.add(_make_summary("MOSSE", "GOT10k"))
        lb = agg.leaderboard(dataset="OTB100")
        assert "MOSSE" not in lb

    def test_cross_dataset_returns_string(self):
        assert isinstance(self._agg().cross_dataset_summary(), str)

    def test_cross_dataset_contains_tracker(self):
        s = self._agg().cross_dataset_summary()
        assert "KCF" in s


# --------------------------------------------------------------------------- #
# ResultAggregator — export
# --------------------------------------------------------------------------- #

class TestExport:
    def test_to_csv(self, tmp_path):
        agg = ResultAggregator()
        agg.add(_make_summary())
        out = tmp_path / "out.csv"
        agg.to_csv(out)
        assert out.exists()
        content = out.read_text()
        assert "tracker" in content
        assert "KCF" in content

    def test_to_csv_creates_parent_dir(self, tmp_path):
        agg = ResultAggregator()
        agg.add(_make_summary())
        out = tmp_path / "sub" / "out.csv"
        agg.to_csv(out)
        assert out.exists()

    def test_to_json(self, tmp_path):
        agg = ResultAggregator()
        agg.add(_make_summary())
        out = tmp_path / "out.json"
        agg.to_json(out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert isinstance(loaded, list)
        assert loaded[0]["tracker"] == "KCF"

    def test_to_json_creates_parent_dir(self, tmp_path):
        agg = ResultAggregator()
        agg.add(_make_summary())
        out = tmp_path / "nested" / "out.json"
        agg.to_json(out)
        assert out.exists()

    def test_empty_to_csv_no_error(self, tmp_path):
        agg = ResultAggregator()
        agg.to_csv(tmp_path / "empty.csv")  # should not raise
