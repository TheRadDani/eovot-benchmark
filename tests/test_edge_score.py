"""Tests for eovot.metrics.edge_score — EdgeDeploymentScorer."""

from __future__ import annotations

import numpy as np
import pytest

from eovot.metrics.edge_score import (
    HARDWARE_PROFILES,
    EdgeDeploymentScorer,
    EdgeScore,
    ParetoPoint,
    TrackerMetrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_trackers() -> list:
    """Fast-but-inaccurate, balanced, slow-but-accurate."""
    return [
        TrackerMetrics("MOSSE", mean_iou=0.42, fps=450.0, peak_memory_mb=45.0, energy_per_frame_mj=0.2),
        TrackerMetrics("KCF",   mean_iou=0.51, fps=180.0, peak_memory_mb=52.0, energy_per_frame_mj=0.5),
        TrackerMetrics("CSRT",  mean_iou=0.65, fps=40.0,  peak_memory_mb=68.0, energy_per_frame_mj=1.2),
    ]


@pytest.fixture
def scorer_equal() -> EdgeDeploymentScorer:
    return EdgeDeploymentScorer(
        weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
        target_fps=25.0,
        max_memory_mb=512.0,
    )


# ---------------------------------------------------------------------------
# TrackerMetrics
# ---------------------------------------------------------------------------


class TestTrackerMetrics:
    def test_default_energy_is_zero(self):
        t = TrackerMetrics("T", mean_iou=0.5, fps=100.0, peak_memory_mb=50.0)
        assert t.energy_per_frame_mj == 0.0

    def test_from_benchmark_result(self):
        class _FakeResult:
            tracker_name = "MOSSE"
            mean_iou = 0.42
            mean_fps = 300.0
            peak_memory_mb = 48.0
            mean_energy_per_frame_mj = 0.8

        tm = TrackerMetrics.from_benchmark_result(_FakeResult())
        assert tm.name == "MOSSE"
        assert tm.mean_iou == pytest.approx(0.42)
        assert tm.fps == pytest.approx(300.0)
        assert tm.peak_memory_mb == pytest.approx(48.0)
        assert tm.energy_per_frame_mj == pytest.approx(0.8)

    def test_from_benchmark_result_none_energy(self):
        class _FakeResult:
            tracker_name = "T"
            mean_iou = 0.5
            mean_fps = 100.0
            peak_memory_mb = 60.0
            mean_energy_per_frame_mj = None

        tm = TrackerMetrics.from_benchmark_result(_FakeResult())
        assert tm.energy_per_frame_mj == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HARDWARE_PROFILES
# ---------------------------------------------------------------------------


class TestHardwareProfiles:
    def test_all_required_profiles_present(self):
        for key in ("raspberry_pi_4", "jetson_nano", "laptop_cpu", "desktop_gpu"):
            assert key in HARDWARE_PROFILES

    def test_each_profile_has_required_keys(self):
        required = {"description", "target_fps", "max_memory_mb", "tdp_watts", "weights"}
        for profile in HARDWARE_PROFILES.values():
            assert required <= set(profile)

    def test_each_profile_weights_sum_to_one(self):
        for name, profile in HARDWARE_PROFILES.items():
            total = sum(profile["weights"].values())
            assert abs(total - 1.0) < 1e-6, f"Profile {name} weights don't sum to 1"


# ---------------------------------------------------------------------------
# EdgeDeploymentScorer construction
# ---------------------------------------------------------------------------


class TestEdgeDeploymentScorerConstruction:
    def test_default_weights_equal(self):
        scorer = EdgeDeploymentScorer()
        for val in scorer.weights.values():
            assert val == pytest.approx(0.25)

    def test_custom_weights_stored(self):
        w = {"accuracy": 0.5, "fps": 0.3, "memory": 0.1, "energy": 0.1}
        scorer = EdgeDeploymentScorer(weights=w)
        assert scorer.weights == w

    def test_invalid_weights_missing_key(self):
        with pytest.raises(ValueError, match="Missing weight keys"):
            EdgeDeploymentScorer(weights={"accuracy": 0.5, "fps": 0.5})

    def test_invalid_weights_wrong_sum(self):
        with pytest.raises(ValueError, match="sum to 1"):
            EdgeDeploymentScorer(
                weights={"accuracy": 0.4, "fps": 0.4, "memory": 0.1, "energy": 0.5}
            )

    def test_from_hardware_profile_raspberry_pi(self):
        scorer = EdgeDeploymentScorer.from_hardware_profile("raspberry_pi_4")
        assert scorer.target_fps == pytest.approx(25.0)
        assert scorer.max_memory_mb == pytest.approx(512.0)

    def test_from_hardware_profile_jetson_nano(self):
        scorer = EdgeDeploymentScorer.from_hardware_profile("jetson_nano")
        assert scorer.target_fps == pytest.approx(30.0)

    def test_from_hardware_profile_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            EdgeDeploymentScorer.from_hardware_profile("doesnotexist")


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    def test_returns_list_of_edge_scores(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        assert len(scores) == 3
        assert all(isinstance(s, EdgeScore) for s in scores)

    def test_sorted_descending(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        vals = [s.score for s in scores]
        assert vals == sorted(vals, reverse=True)

    def test_score_in_unit_interval(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        for s in scores:
            assert 0.0 <= s.score <= 1.0

    def test_empty_input_returns_empty(self, scorer_equal):
        assert scorer_equal.score([]) == []

    def test_single_tracker_gets_score_one(self, scorer_equal):
        only = [TrackerMetrics("T", mean_iou=0.5, fps=100.0, peak_memory_mb=50.0)]
        scores = scorer_equal.score(only)
        # All normalised to 1.0 when only one tracker → EDS = 1.0
        assert scores[0].score == pytest.approx(1.0)

    def test_normalized_values_in_range(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        for s in scores:
            for v in s.normalized.values():
                assert 0.0 <= v <= 1.0

    def test_meets_fps_target_flag(self, three_trackers):
        scorer = EdgeDeploymentScorer(
            weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
            target_fps=100.0,   # MOSSE 450 ok, KCF 180 ok, CSRT 40 NOT ok
            max_memory_mb=9999.0,
        )
        scores = {s.tracker_name: s for s in scorer.score(three_trackers)}
        assert scores["MOSSE"].meets_fps_target is True
        assert scores["KCF"].meets_fps_target is True
        assert scores["CSRT"].meets_fps_target is False

    def test_meets_memory_target_flag(self, three_trackers):
        scorer = EdgeDeploymentScorer(
            weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
            target_fps=0.0,
            max_memory_mb=50.0,  # MOSSE 45 ok, KCF 52 NOT ok, CSRT 68 NOT ok
        )
        scores = {s.tracker_name: s for s in scorer.score(three_trackers)}
        assert scores["MOSSE"].meets_memory_target is True
        assert scores["KCF"].meets_memory_target is False
        assert scores["CSRT"].meets_memory_target is False

    def test_is_deployable_requires_both_constraints(self, three_trackers):
        scorer = EdgeDeploymentScorer(
            weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
            target_fps=100.0,
            max_memory_mb=50.0,
        )
        scores = {s.tracker_name: s for s in scorer.score(three_trackers)}
        # MOSSE: fps=450 >= 100 AND mem=45 <= 50 → deployable
        assert scores["MOSSE"].is_deployable is True
        # KCF: fps=180 >= 100 but mem=52 > 50 → NOT deployable
        assert scores["KCF"].is_deployable is False

    def test_score_higher_accuracy_weight_favours_csrt(self, three_trackers):
        accuracy_heavy = EdgeDeploymentScorer(
            weights={"accuracy": 0.90, "fps": 0.05, "memory": 0.03, "energy": 0.02},
        )
        scores = accuracy_heavy.score(three_trackers)
        # CSRT has highest IoU → should rank first
        assert scores[0].tracker_name == "CSRT"

    def test_score_higher_fps_weight_favours_mosse(self, three_trackers):
        fps_heavy = EdgeDeploymentScorer(
            weights={"accuracy": 0.05, "fps": 0.90, "memory": 0.03, "energy": 0.02},
        )
        scores = fps_heavy.score(three_trackers)
        # MOSSE has highest FPS → should rank first
        assert scores[0].tracker_name == "MOSSE"

    def test_to_dict_keys(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        d = scores[0].to_dict()
        for key in ("tracker", "eds", "mean_iou", "fps", "peak_memory_mb", "is_deployable"):
            assert key in d


# ---------------------------------------------------------------------------
# pareto_frontier()
# ---------------------------------------------------------------------------


class TestParetoFrontier:
    def test_returns_list_of_pareto_points(self, scorer_equal, three_trackers):
        points = scorer_equal.pareto_frontier(three_trackers)
        assert len(points) == 3
        assert all(isinstance(p, ParetoPoint) for p in points)

    def test_empty_input_returns_empty(self, scorer_equal):
        assert scorer_equal.pareto_frontier([]) == []

    def test_dominated_tracker_not_optimal(self, scorer_equal):
        # B dominates A on both axes: more accurate AND faster
        trackers = [
            TrackerMetrics("A", mean_iou=0.3, fps=50.0, peak_memory_mb=50.0),
            TrackerMetrics("B", mean_iou=0.6, fps=100.0, peak_memory_mb=50.0),
            TrackerMetrics("C", mean_iou=0.8, fps=20.0, peak_memory_mb=50.0),
        ]
        pts = {p.tracker_name: p for p in scorer_equal.pareto_frontier(trackers)}
        assert pts["A"].is_pareto_optimal is False
        assert pts["B"].is_pareto_optimal is True
        assert pts["C"].is_pareto_optimal is True

    def test_single_tracker_always_pareto_optimal(self, scorer_equal):
        trackers = [TrackerMetrics("X", mean_iou=0.5, fps=100.0, peak_memory_mb=60.0)]
        pts = scorer_equal.pareto_frontier(trackers)
        assert pts[0].is_pareto_optimal is True

    def test_identical_trackers_both_optimal(self, scorer_equal):
        trackers = [
            TrackerMetrics("A", mean_iou=0.5, fps=100.0, peak_memory_mb=50.0),
            TrackerMetrics("B", mean_iou=0.5, fps=100.0, peak_memory_mb=50.0),
        ]
        pts = scorer_equal.pareto_frontier(trackers)
        assert all(p.is_pareto_optimal for p in pts)

    def test_pareto_points_have_correct_attributes(self, scorer_equal, three_trackers):
        pts = {p.tracker_name: p for p in scorer_equal.pareto_frontier(three_trackers)}
        assert pts["MOSSE"].fps == pytest.approx(450.0)
        assert pts["CSRT"].mean_iou == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# format_leaderboard()
# ---------------------------------------------------------------------------


class TestFormatLeaderboard:
    def test_contains_header(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        md = scorer_equal.format_leaderboard(scores)
        assert "| Rank |" in md
        assert "EDS" in md

    def test_contains_all_tracker_names(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        md = scorer_equal.format_leaderboard(scores)
        for name in ("MOSSE", "KCF", "CSRT"):
            assert name in md

    def test_deployable_column(self, three_trackers):
        scorer = EdgeDeploymentScorer(
            weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
            target_fps=50.0,
            max_memory_mb=100.0,
        )
        scores = scorer.score(three_trackers)
        md = scorer.format_leaderboard(scores)
        assert "Yes" in md or "No" in md


# ---------------------------------------------------------------------------
# suitability_report()
# ---------------------------------------------------------------------------


class TestSuitabilityReport:
    def test_report_contains_profile_name(self, three_trackers):
        scorer = EdgeDeploymentScorer.from_hardware_profile("raspberry_pi_4")
        scores = scorer.score(three_trackers)
        report = scorer.suitability_report(scores, profile_name="raspberry_pi_4")
        assert "raspberry_pi_4" in report

    def test_report_contains_target_fps(self, scorer_equal, three_trackers):
        scores = scorer_equal.score(three_trackers)
        report = scorer_equal.suitability_report(scores)
        assert "25" in report  # target_fps = 25.0

    def test_report_lists_deployable_trackers(self, three_trackers):
        scorer = EdgeDeploymentScorer(
            weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
            target_fps=50.0,
            max_memory_mb=100.0,
        )
        scores = scorer.score(three_trackers)
        report = scorer.suitability_report(scores)
        # MOSSE (450 FPS) and KCF (180 FPS) should be in deployable section
        assert "MOSSE" in report

    def test_report_lists_not_meeting_constraints(self, three_trackers):
        scorer = EdgeDeploymentScorer(
            weights={"accuracy": 0.25, "fps": 0.25, "memory": 0.25, "energy": 0.25},
            target_fps=200.0,  # Only MOSSE (450 FPS) meets this
            max_memory_mb=100.0,
        )
        scores = scorer.score(three_trackers)
        report = scorer.suitability_report(scores)
        assert "Not Meeting Constraints" in report
        assert "CSRT" in report
