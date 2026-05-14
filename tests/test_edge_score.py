"""Unit tests for eovot.metrics.edge_score."""

from __future__ import annotations

import math

import pytest

from eovot.metrics.edge_score import EdgeDeploymentScore, EdgeScoreCalculator


class TestEdgeScoreCalculatorConstruction:
    def test_defaults(self):
        calc = EdgeScoreCalculator()
        assert calc.target_fps == 30.0
        assert calc.reference_energy_mj == 10.0

    def test_invalid_target_fps(self):
        with pytest.raises(ValueError, match="target_fps"):
            EdgeScoreCalculator(target_fps=0.0)

    def test_invalid_reference_energy(self):
        with pytest.raises(ValueError, match="reference_energy_mj"):
            EdgeScoreCalculator(reference_energy_mj=-1.0)

    def test_invalid_weights(self):
        with pytest.raises(ValueError):
            EdgeScoreCalculator(w_acc=0.0, w_fps=0.5, w_eff=0.5)


class TestComputeNoEnergy:
    def setup_method(self):
        self.calc = EdgeScoreCalculator(target_fps=30.0)

    def test_returns_edge_deployment_score(self):
        score = self.calc.compute("T", mean_iou=0.6, fps=60.0)
        assert isinstance(score, EdgeDeploymentScore)

    def test_efficiency_score_is_none_without_energy(self):
        score = self.calc.compute("T", mean_iou=0.6, fps=60.0)
        assert score.efficiency_score is None

    def test_tracker_name_stored(self):
        score = self.calc.compute("MOSSE", mean_iou=0.5, fps=300.0)
        assert score.tracker_name == "MOSSE"

    def test_perfect_tracker(self):
        score = self.calc.compute("P", mean_iou=1.0, fps=30.0)
        assert score.eds == pytest.approx(1.0, abs=1e-6)

    def test_zero_accuracy_gives_zero_eds(self):
        score = self.calc.compute("Z", mean_iou=0.0, fps=60.0)
        assert score.eds == pytest.approx(0.0, abs=1e-6)

    def test_throughput_capped_at_one(self):
        score = self.calc.compute("Fast", mean_iou=0.8, fps=300.0)
        assert score.throughput_score == pytest.approx(1.0)

    def test_throughput_below_target(self):
        score = self.calc.compute("Slow", mean_iou=0.8, fps=15.0)
        assert score.throughput_score == pytest.approx(0.5)

    def test_eds_in_unit_interval(self):
        for iou in [0.0, 0.3, 0.6, 0.9, 1.0]:
            for fps in [1.0, 15.0, 30.0, 120.0]:
                score = self.calc.compute(mean_iou=iou, fps=fps)
                assert 0.0 <= score.eds <= 1.0, f"EDS={score.eds} out of range for iou={iou}, fps={fps}"


class TestComputeWithEnergy:
    def setup_method(self):
        self.calc = EdgeScoreCalculator(
            target_fps=30.0,
            reference_energy_mj=10.0,
        )

    def test_efficiency_score_present(self):
        score = self.calc.compute("T", mean_iou=0.6, fps=30.0, energy_per_frame_mj=5.0)
        assert score.efficiency_score is not None

    def test_reference_energy_gives_half_efficiency(self):
        score = self.calc.compute("T", mean_iou=1.0, fps=30.0,
                                  energy_per_frame_mj=10.0)
        # 1 / (1 + 10/10) = 0.5
        assert score.efficiency_score == pytest.approx(0.5)

    def test_zero_energy_gives_maximum_efficiency(self):
        score = self.calc.compute("T", mean_iou=1.0, fps=30.0,
                                  energy_per_frame_mj=0.0)
        assert score.efficiency_score == pytest.approx(1.0)

    def test_high_energy_lowers_eds(self):
        low_e = self.calc.compute("A", mean_iou=0.6, fps=30.0,
                                   energy_per_frame_mj=1.0)
        high_e = self.calc.compute("B", mean_iou=0.6, fps=30.0,
                                    energy_per_frame_mj=100.0)
        assert low_e.eds > high_e.eds

    def test_eds_in_unit_interval_with_energy(self):
        for energy in [0.1, 1.0, 10.0, 100.0]:
            score = self.calc.compute(mean_iou=0.7, fps=25.0,
                                      energy_per_frame_mj=energy)
            assert 0.0 <= score.eds <= 1.0


class TestFromBenchmarkSummary:
    def setup_method(self):
        self.calc = EdgeScoreCalculator()

    def test_basic(self):
        summary = {"tracker": "KCF", "mean_iou": 0.61, "mean_fps": 120.0}
        score = self.calc.from_benchmark_summary(summary)
        assert score.tracker_name == "KCF"
        assert score.mean_iou == pytest.approx(0.61)

    def test_with_energy(self):
        summary = {
            "tracker": "CSRT",
            "mean_iou": 0.70,
            "mean_fps": 25.0,
            "mean_energy_per_frame_mj": 8.5,
        }
        score = self.calc.from_benchmark_summary(summary)
        assert score.efficiency_score is not None

    def test_missing_energy_key(self):
        summary = {"tracker": "MOSSE", "mean_iou": 0.55, "mean_fps": 312.0}
        score = self.calc.from_benchmark_summary(summary)
        assert score.efficiency_score is None


class TestRankAndLeaderboard:
    def setup_method(self):
        self.calc = EdgeScoreCalculator(target_fps=30.0)
        self.summaries = [
            {"tracker": "MOSSE", "mean_iou": 0.55, "mean_fps": 312.0},
            {"tracker": "KCF",   "mean_iou": 0.61, "mean_fps": 120.0},
            {"tracker": "CSRT",  "mean_iou": 0.70, "mean_fps": 25.0},
        ]

    def test_rank_returns_list(self):
        ranked = self.calc.rank(self.summaries)
        assert len(ranked) == 3

    def test_rank_descending(self):
        ranked = self.calc.rank(self.summaries)
        eds_vals = [r.eds for r in ranked]
        assert eds_vals == sorted(eds_vals, reverse=True)

    def test_leaderboard_markdown_contains_headers(self):
        scores = self.calc.rank(self.summaries)
        md = self.calc.leaderboard_markdown(scores)
        assert "Rank" in md
        assert "EDS" in md
        assert "MOSSE" in md or "KCF" in md or "CSRT" in md

    def test_leaderboard_markdown_row_count(self):
        scores = self.calc.rank(self.summaries)
        md = self.calc.leaderboard_markdown(scores)
        # Count data rows (lines starting with |) minus header rows
        data_rows = [l for l in md.splitlines()
                     if l.startswith("|") and "---" not in l and "Rank" not in l
                     and "Target" not in l]
        assert len(data_rows) == 3


class TestToDict:
    def test_to_dict_has_required_keys(self):
        calc = EdgeScoreCalculator()
        score = calc.compute("T", mean_iou=0.6, fps=30.0, energy_per_frame_mj=5.0)
        d = score.to_dict()
        for key in ("tracker_name", "eds", "accuracy_score", "throughput_score",
                    "efficiency_score", "mean_iou", "fps", "target_fps"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_none_energy(self):
        calc = EdgeScoreCalculator()
        score = calc.compute("T", mean_iou=0.6, fps=30.0)
        d = score.to_dict()
        assert d["efficiency_score"] is None
        assert d["energy_per_frame_mj"] is None


class TestEdgeDeploymentScoreStr:
    def test_str_contains_tracker_name(self):
        calc = EdgeScoreCalculator()
        score = calc.compute("MOSSE", mean_iou=0.55, fps=312.0)
        assert "MOSSE" in str(score)

    def test_str_contains_eds(self):
        calc = EdgeScoreCalculator()
        score = calc.compute("X", mean_iou=0.7, fps=30.0)
        assert "EDS=" in str(score)

    def test_str_na_when_no_energy(self):
        calc = EdgeScoreCalculator()
        score = calc.compute("X", mean_iou=0.7, fps=30.0)
        assert "N/A" in str(score)
