"""Unit tests for the Pareto analysis and edge deployment scoring module.

Covers:
- Pareto dominance relation (_dominates)
- Pareto front identification (compute_pareto_front)
- Edge score computation (compute_edge_score)
- Tracker ranking (rank_trackers)
- ParetoAnalyzer: load_from_json, load_from_dict, analyze, output methods
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eovot.analysis.pareto import (
    ParetoAnalyzer,
    ParetoResult,
    TrackerProfile,
    _dominates,
    compute_edge_score,
    compute_pareto_front,
    rank_trackers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_profile(
    name: str,
    iou: float = 0.5,
    fps: float = 30.0,
    latency: float = 33.0,
    memory: float = 100.0,
    energy: float = 50.0,
) -> TrackerProfile:
    return TrackerProfile(
        name=name,
        mean_iou=iou,
        fps=fps,
        latency_ms=latency,
        memory_mb=memory,
        energy_mj=energy,
    )


# ---------------------------------------------------------------------------
# _dominates
# ---------------------------------------------------------------------------


class TestDominates:
    def test_clearly_dominated(self):
        worse = make_profile("worse", iou=0.3, fps=10, latency=100, memory=200)
        better = make_profile("better", iou=0.5, fps=30, latency=33, memory=100)
        assert _dominates(better, worse)

    def test_dominated_not_symmetric(self):
        worse = make_profile("worse", iou=0.3, fps=10, latency=100, memory=200)
        better = make_profile("better", iou=0.5, fps=30, latency=33, memory=100)
        assert not _dominates(worse, better)

    def test_accuracy_speed_tradeoff_not_dominated(self):
        # High accuracy, low speed vs. low accuracy, high speed
        accurate = make_profile("accurate", iou=0.9, fps=10, latency=100, memory=300)
        fast = make_profile("fast", iou=0.3, fps=200, latency=5, memory=20)
        assert not _dominates(accurate, fast)
        assert not _dominates(fast, accurate)

    def test_identical_profiles_not_dominated(self):
        p = make_profile("a", iou=0.5, fps=30, latency=33, memory=100)
        q = make_profile("b", iou=0.5, fps=30, latency=33, memory=100)
        # Must be *strictly* better in at least one dimension
        assert not _dominates(p, q)
        assert not _dominates(q, p)

    def test_one_dimension_better_rest_equal(self):
        p = make_profile("p", iou=0.5, fps=30, latency=33, memory=100)
        q = make_profile("q", iou=0.6, fps=30, latency=33, memory=100)  # q better IoU
        assert _dominates(q, p)
        assert not _dominates(p, q)

    def test_latency_is_minimise(self):
        p = make_profile("slow", iou=0.5, fps=30, latency=100, memory=100)
        q = make_profile("fast", iou=0.5, fps=30, latency=10, memory=100)
        assert _dominates(q, p)

    def test_memory_is_minimise(self):
        p = make_profile("bloat", iou=0.5, fps=30, latency=33, memory=500)
        q = make_profile("lean", iou=0.5, fps=30, latency=33, memory=50)
        assert _dominates(q, p)


# ---------------------------------------------------------------------------
# compute_pareto_front
# ---------------------------------------------------------------------------


class TestComputeParetoFront:
    def test_single_tracker_is_pareto(self):
        front = compute_pareto_front([make_profile("only")])
        assert "only" in front

    def test_empty_list(self):
        assert compute_pareto_front([]) == []

    def test_accuracy_speed_tradeoff_all_pareto(self):
        profiles = [
            make_profile("accurate", iou=0.9, fps=5, latency=200, memory=500),
            make_profile("balanced", iou=0.6, fps=30, latency=33, memory=150),
            make_profile("fast", iou=0.2, fps=300, latency=3, memory=20),
        ]
        front = compute_pareto_front(profiles)
        assert set(front) == {"accurate", "balanced", "fast"}

    def test_dominated_tracker_excluded(self):
        profiles = [
            make_profile("good", iou=0.8, fps=50, latency=20, memory=100),
            make_profile("bad", iou=0.3, fps=5, latency=200, memory=500),
        ]
        front = compute_pareto_front(profiles)
        assert "good" in front
        assert "bad" not in front

    def test_pareto_is_subset_of_all(self):
        profiles = [make_profile(f"t{i}", iou=i / 10) for i in range(1, 6)]
        front = compute_pareto_front(profiles)
        all_names = {p.name for p in profiles}
        assert set(front).issubset(all_names)

    def test_order_preserved(self):
        # The pareto front should respect original list order
        profiles = [
            make_profile("a", iou=0.8, fps=10, latency=100, memory=200),
            make_profile("b", iou=0.3, fps=100, latency=10, memory=20),
        ]
        front = compute_pareto_front(profiles)
        assert front.index("a") < front.index("b")


# ---------------------------------------------------------------------------
# compute_edge_score
# ---------------------------------------------------------------------------


class TestComputeEdgeScore:
    def test_perfect_profile_high_score(self):
        p = make_profile("perfect", iou=1.0, fps=30.0, latency=0.0, memory=0.0, energy=0.0)
        # energy=0 means not measured; weight redistributed to accuracy
        score = compute_edge_score(p, fps_target=30.0, memory_budget_mb=512.0)
        assert score > 0.9

    def test_zero_profile_zero_score(self):
        p = make_profile("zero", iou=0.0, fps=0.0, latency=9999, memory=1024, energy=200)
        score = compute_edge_score(
            p, fps_target=30.0, memory_budget_mb=512.0, energy_budget_mj=100.0
        )
        assert score == pytest.approx(0.0, abs=0.01)

    def test_faster_tracker_scores_higher_ceteris_paribus(self):
        slow = make_profile("slow", iou=0.5, fps=5.0, memory=100, energy=0)
        fast = make_profile("fast", iou=0.5, fps=60.0, memory=100, energy=0)
        assert compute_edge_score(fast) > compute_edge_score(slow)

    def test_lower_memory_scores_higher(self):
        bloat = make_profile("bloat", iou=0.5, fps=30, memory=400, energy=0)
        lean = make_profile("lean", iou=0.5, fps=30, memory=50, energy=0)
        assert compute_edge_score(lean) > compute_edge_score(bloat)

    def test_score_in_unit_interval(self):
        for iou in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for fps in [1.0, 15.0, 30.0, 100.0]:
                p = make_profile("t", iou=iou, fps=fps, energy=0)
                score = compute_edge_score(p)
                assert 0.0 <= score <= 1.0, f"score={score} for iou={iou}, fps={fps}"

    def test_no_energy_measurement_still_valid(self):
        p = make_profile("no_energy", iou=0.6, fps=30, memory=100, energy=0.0)
        score = compute_edge_score(p, w_energy=0.1)
        # Should compute without error; weight redistributed to accuracy
        assert 0.0 <= score <= 1.0

    def test_custom_weights_sum_to_different_total(self):
        p = make_profile("t", iou=0.5, fps=30, memory=100, energy=50)
        # Non-standard weights (but still valid)
        score = compute_edge_score(
            p, w_accuracy=0.5, w_speed=0.3, w_memory=0.1, w_energy=0.1
        )
        assert 0.0 <= score <= 1.0

    def test_fps_above_target_clamped_to_one(self):
        p = make_profile("super_fast", iou=1.0, fps=1000.0, memory=0.0, energy=0.0)
        score = compute_edge_score(p, fps_target=30.0)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# rank_trackers
# ---------------------------------------------------------------------------


class TestRankTrackers:
    def test_better_tracker_ranked_higher(self):
        profiles = [
            make_profile("bad", iou=0.2, fps=5, memory=400, energy=0),
            make_profile("good", iou=0.8, fps=50, memory=100, energy=0),
            make_profile("medium", iou=0.5, fps=25, memory=200, energy=0),
        ]
        rankings = rank_trackers(profiles)
        keys = list(rankings.keys())
        assert keys[0] == "good"
        assert keys[-1] == "bad"

    def test_rankings_all_in_unit_interval(self):
        profiles = [make_profile(f"t{i}", iou=i / 5, fps=30, energy=0) for i in range(1, 6)]
        rankings = rank_trackers(profiles)
        for score in rankings.values():
            assert 0.0 <= score <= 1.0

    def test_empty_list_returns_empty_dict(self):
        assert rank_trackers([]) == {}


# ---------------------------------------------------------------------------
# ParetoAnalyzer
# ---------------------------------------------------------------------------


class TestParetoAnalyzer:
    def setup_method(self):
        self.analyzer = ParetoAnalyzer(
            fps_target=30.0,
            memory_budget_mb=512.0,
            energy_budget_mj=100.0,
        )

    def test_analyze_empty(self):
        result = self.analyzer.analyze([])
        assert isinstance(result, ParetoResult)
        assert result.pareto_front == []
        assert result.rankings == {}
        assert result.dominated == []

    def test_analyze_basic(self):
        profiles = [
            make_profile("fast", iou=0.3, fps=200, latency=5, memory=20, energy=0),
            make_profile("accurate", iou=0.9, fps=10, latency=100, memory=300, energy=0),
        ]
        result = self.analyzer.analyze(profiles)
        # Both on Pareto front (accuracy/speed trade-off)
        assert set(result.pareto_front) == {"fast", "accurate"}
        assert len(result.dominated) == 0

    def test_analyze_dominated_tracker(self):
        profiles = [
            make_profile("good", iou=0.8, fps=60, latency=17, memory=80, energy=0),
            make_profile("bad", iou=0.3, fps=5, latency=200, memory=400, energy=0),
        ]
        result = self.analyzer.analyze(profiles)
        assert "good" in result.pareto_front
        assert "bad" in result.dominated

    def test_rankings_keys_match_profiles(self):
        profiles = [make_profile(f"t{i}", energy=0) for i in range(4)]
        result = self.analyzer.analyze(profiles)
        assert set(result.rankings.keys()) == {p.name for p in profiles}

    def test_load_from_json_single_tracker(self, tmp_path):
        data = {
            "tracker": "mosse",
            "summary": {
                "mean_iou": 0.52,
                "mean_fps": 450.0,
                "mean_latency_ms": 2.2,
                "peak_memory_mb": 15.0,
                "mean_energy_per_frame_mj": 0.0,
            },
        }
        jf = tmp_path / "mosse.json"
        jf.write_text(json.dumps(data))
        profiles = self.analyzer.load_from_json(str(jf))
        assert len(profiles) == 1
        assert profiles[0].name == "mosse"
        assert profiles[0].mean_iou == pytest.approx(0.52)
        assert profiles[0].fps == pytest.approx(450.0)

    def test_load_from_json_multi_tracker(self, tmp_path):
        data = {
            "mosse": {"summary": {"mean_iou": 0.52, "mean_fps": 450.0,
                                   "mean_latency_ms": 2.2, "peak_memory_mb": 15.0}},
            "kcf": {"summary": {"mean_iou": 0.60, "mean_fps": 280.0,
                                 "mean_latency_ms": 3.6, "peak_memory_mb": 30.0}},
        }
        jf = tmp_path / "comparison.json"
        jf.write_text(json.dumps(data))
        profiles = self.analyzer.load_from_json(str(jf))
        assert len(profiles) == 2
        names = {p.name for p in profiles}
        assert names == {"mosse", "kcf"}

    def test_load_from_dict(self):
        data = {
            "tracker": "test_tracker",
            "summary": {"mean_iou": 0.65, "mean_fps": 45.0,
                        "mean_latency_ms": 22.0, "peak_memory_mb": 120.0},
        }
        profiles = self.analyzer.load_from_dict(data)
        assert len(profiles) == 1
        assert profiles[0].name == "test_tracker"

    def test_print_leaderboard_runs_without_error(self, capsys):
        profiles = [make_profile("tracker_a", iou=0.7, fps=30, memory=150, energy=0)]
        result = self.analyzer.analyze(profiles)
        self.analyzer.print_leaderboard(result)
        captured = capsys.readouterr()
        assert "tracker_a" in captured.out
        assert "Leaderboard" in captured.out

    def test_print_leaderboard_marks_pareto_trackers(self, capsys):
        profiles = [
            make_profile("on_front", iou=0.8, fps=60, latency=17, memory=80, energy=0),
            make_profile("dominated", iou=0.3, fps=5, latency=200, memory=400, energy=0),
        ]
        result = self.analyzer.analyze(profiles)
        self.analyzer.print_leaderboard(result)
        captured = capsys.readouterr()
        assert "*" in captured.out  # Pareto marker present

    def test_to_markdown_contains_table(self):
        profiles = [make_profile("t1", iou=0.7, fps=30, energy=0)]
        result = self.analyzer.analyze(profiles)
        md = self.analyzer.to_markdown(result)
        assert "|" in md
        assert "t1" in md
        assert "Leaderboard" in md

    def test_to_json_is_valid_json(self):
        profiles = [
            make_profile("a", iou=0.7, fps=30, energy=0),
            make_profile("b", iou=0.4, fps=100, energy=0),
        ]
        result = self.analyzer.analyze(profiles)
        text = self.analyzer.to_json(result)
        parsed = json.loads(text)
        assert "pareto_front" in parsed
        assert "rankings" in parsed
        assert len(parsed["rankings"]) == 2

    def test_to_json_pareto_flag(self):
        profiles = [
            make_profile("good", iou=0.8, fps=60, latency=17, memory=80, energy=0),
            make_profile("bad", iou=0.3, fps=5, latency=200, memory=400, energy=0),
        ]
        result = self.analyzer.analyze(profiles)
        parsed = json.loads(self.analyzer.to_json(result))
        good_entry = next(r for r in parsed["rankings"] if r["name"] == "good")
        bad_entry = next(r for r in parsed["rankings"] if r["name"] == "bad")
        assert good_entry["pareto_optimal"] is True
        assert bad_entry["pareto_optimal"] is False

    def test_deduplication_on_repeated_name(self, tmp_path):
        """Duplicate tracker names across files should take the last-loaded profile."""
        data1 = {"tracker": "mosse", "summary": {"mean_iou": 0.4, "mean_fps": 100.0,
                                                   "mean_latency_ms": 10.0, "peak_memory_mb": 15.0}}
        data2 = {"tracker": "mosse", "summary": {"mean_iou": 0.55, "mean_fps": 450.0,
                                                   "mean_latency_ms": 2.2, "peak_memory_mb": 15.0}}
        f1 = tmp_path / "mosse_old.json"
        f2 = tmp_path / "mosse_new.json"
        f1.write_text(json.dumps(data1))
        f2.write_text(json.dumps(data2))
        all_profiles = (
            self.analyzer.load_from_json(str(f1))
            + self.analyzer.load_from_json(str(f2))
        )
        # Deduplicate (as analyze_results.py does)
        seen = {}
        for p in all_profiles:
            seen[p.name] = p
        profiles = list(seen.values())
        assert len(profiles) == 1
        assert profiles[0].mean_iou == pytest.approx(0.55)  # Last wins
