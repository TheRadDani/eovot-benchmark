"""Tests for eovot.analysis.pareto."""

import pytest

from eovot.analysis.pareto import ParetoAnalyzer, ParetoPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(tracker: str, mean_iou: float, mean_fps: float) -> dict:
    return {
        "summary": {
            "tracker": tracker,
            "mean_iou": mean_iou,
            "mean_fps": mean_fps,
        },
        "sequences": [],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_tracker_results():
    """CSRT: high accuracy / high FPS; MOSSE: low accuracy / very high FPS;
    KCF: medium accuracy / low FPS (dominated by CSRT on both axes)."""
    return {
        "CSRT": _make_result("CSRT", 0.80, 200.0),   # dominates KCF
        "MOSSE": _make_result("MOSSE", 0.50, 500.0),  # higher FPS than all
        "KCF": _make_result("KCF", 0.65, 150.0),      # dominated by CSRT
    }


# ---------------------------------------------------------------------------
# ParetoAnalyzer.analyze()
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_returns_all_trackers(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        names = {p.tracker for p in points}
        assert names == {"CSRT", "MOSSE", "KCF"}

    def test_extreme_accuracy_tracker_is_pareto(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        pareto_names = {p.tracker for p in points if p.is_pareto}
        assert "CSRT" in pareto_names

    def test_extreme_efficiency_tracker_is_pareto(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        pareto_names = {p.tracker for p in points if p.is_pareto}
        assert "MOSSE" in pareto_names

    def test_dominated_tracker_not_pareto(self, three_tracker_results):
        """KCF is dominated by CSRT on both accuracy (0.80>0.65) and FPS (200>150)."""
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        kcf = next(p for p in points if p.tracker == "KCF")
        assert not kcf.is_pareto

    def test_sorted_by_accuracy_descending(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        accs = [p.accuracy for p in points]
        assert accs == sorted(accs, reverse=True)

    def test_scores_in_unit_range(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        for p in points:
            assert 0.0 <= p.score <= 1.0

    def test_empty_results_returns_empty(self):
        analyzer = ParetoAnalyzer()
        assert analyzer.analyze({}) == []

    def test_single_tracker_is_pareto(self):
        analyzer = ParetoAnalyzer()
        result = {"Solo": _make_result("Solo", 0.7, 100.0)}
        points = analyzer.analyze(result)
        assert len(points) == 1
        assert points[0].is_pareto

    def test_two_non_dominated_trackers_both_pareto(self):
        """When one tracker wins on accuracy and the other on FPS, both are Pareto."""
        results = {
            "Fast": _make_result("Fast", 0.4, 400.0),
            "Accurate": _make_result("Accurate", 0.9, 40.0),
        }
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(results)
        assert all(p.is_pareto for p in points)

    def test_dominated_tracker_removed_from_frontier(self):
        """A tracker strictly worse on both axes should be dominated."""
        results = {
            "Best": _make_result("Best", 0.9, 200.0),
            "Worst": _make_result("Worst", 0.5, 100.0),
        }
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(results)
        worst = next(p for p in points if p.tracker == "Worst")
        assert not worst.is_pareto

    def test_custom_metrics(self):
        results = {
            "A": {
                "summary": {"tracker": "A", "success_auc": 0.8, "latency_ms": 5.0},
                "sequences": [],
            },
            "B": {
                "summary": {"tracker": "B", "success_auc": 0.5, "latency_ms": 2.0},
                "sequences": [],
            },
        }
        analyzer = ParetoAnalyzer()
        # Use latency_ms as efficiency (lower is better — not ideal, but tests the plumbing)
        points = analyzer.analyze(
            results, accuracy_metric="success_auc", efficiency_metric="latency_ms"
        )
        names = {p.tracker for p in points}
        assert names == {"A", "B"}


# ---------------------------------------------------------------------------
# ParetoAnalyzer.pareto_scores()
# ---------------------------------------------------------------------------


class TestParetoScores:
    def test_keys_match_tracker_names(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        scores = analyzer.pareto_scores(three_tracker_results)
        assert set(scores) == {"CSRT", "MOSSE", "KCF"}

    def test_scores_in_unit_range(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        scores = analyzer.pareto_scores(three_tracker_results)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# ParetoAnalyzer.efficiency_frontier()
# ---------------------------------------------------------------------------


class TestEfficiencyFrontier:
    def test_returns_only_pareto_points(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        frontier = analyzer.efficiency_frontier(three_tracker_results)
        # Only CSRT and MOSSE are Pareto → 2 points
        assert len(frontier) == 2

    def test_sorted_by_efficiency_ascending(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        frontier = analyzer.efficiency_frontier(three_tracker_results)
        effs = [t[0] for t in frontier]
        assert effs == sorted(effs)

    def test_tuple_structure(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        frontier = analyzer.efficiency_frontier(three_tracker_results)
        for item in frontier:
            eff, acc, name = item
            assert isinstance(eff, float)
            assert isinstance(acc, float)
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# ParetoAnalyzer.to_markdown()
# ---------------------------------------------------------------------------


class TestToMarkdown:
    def test_header_present(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        md = analyzer.to_markdown(points)
        assert "| Rank |" in md
        assert "| Tracker |" in md

    def test_pareto_checkmark_present(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        md = analyzer.to_markdown(points)
        assert "✓" in md

    def test_all_tracker_names_present(self, three_tracker_results):
        analyzer = ParetoAnalyzer()
        points = analyzer.analyze(three_tracker_results)
        md = analyzer.to_markdown(points)
        for name in ["CSRT", "MOSSE", "KCF"]:
            assert name in md
