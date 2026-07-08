"""Tests for CrossDatasetEvaluator and CrossDatasetReport."""

import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.cross_dataset import CrossDatasetEvaluator, CrossDatasetReport
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


def _datasets(n: int = 2) -> dict:
    """Create n small synthetic datasets with different seeds."""
    return {
        f"Syn-{i}": SyntheticDataset(num_sequences=2, num_frames=15, seed=i)
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# CrossDatasetEvaluator construction
# ---------------------------------------------------------------------------

class TestCrossDatasetEvaluatorConstruction:
    def test_empty_datasets_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CrossDatasetEvaluator(datasets={})

    def test_single_dataset_accepted(self):
        ev = CrossDatasetEvaluator(
            datasets={"syn": SyntheticDataset(num_sequences=1, num_frames=10)},
            verbose=False,
        )
        assert len(ev.datasets) == 1


# ---------------------------------------------------------------------------
# CrossDatasetReport properties
# ---------------------------------------------------------------------------

class TestCrossDatasetReport:
    def _run(self, n_datasets=2):
        ev = CrossDatasetEvaluator(datasets=_datasets(n_datasets), verbose=False)
        return ev.run(MOSSETracker())

    def test_tracker_name_correct(self):
        report = self._run()
        assert report.tracker_name == "MOSSE"

    def test_per_dataset_keys_present(self):
        report = self._run(2)
        assert set(report.per_dataset.keys()) == {"Syn-0", "Syn-1"}

    def test_per_dataset_has_mean_iou(self):
        report = self._run()
        for s in report.per_dataset.values():
            assert "mean_iou" in s
            assert 0.0 <= s["mean_iou"] <= 1.0

    def test_per_dataset_has_mean_fps(self):
        report = self._run()
        for s in report.per_dataset.values():
            assert "mean_fps" in s
            assert s["mean_fps"] > 0.0

    def test_aggregate_mean_iou_mean_present(self):
        report = self._run(2)
        agg = report.aggregate
        assert "mean_iou_mean" in agg
        assert "mean_iou_std" in agg

    def test_aggregate_fps_mean_present(self):
        report = self._run(2)
        agg = report.aggregate
        assert "mean_fps_mean" in agg

    def test_aggregate_num_datasets_correct(self):
        report = self._run(3)
        assert report.aggregate["num_datasets"] == 3

    def test_aggregate_std_zero_for_single_dataset(self):
        """With one dataset, std should be 0.0 (no variance)."""
        ev = CrossDatasetEvaluator(
            datasets={"only": SyntheticDataset(num_sequences=2, num_frames=15)},
            verbose=False,
        )
        report = ev.run(MOSSETracker())
        assert report.aggregate["mean_iou_std"] == 0.0

    def test_raw_results_populated(self):
        from eovot.benchmark.engine import BenchmarkResult
        report = self._run(2)
        for ds_name in ("Syn-0", "Syn-1"):
            assert ds_name in report.raw_results
            assert isinstance(report.raw_results[ds_name], BenchmarkResult)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestCrossDatasetReportSerialisation:
    def test_to_dict_structure(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(2), verbose=False)
        report = ev.run(MOSSETracker())
        d = report.to_dict()
        assert "tracker" in d
        assert "per_dataset" in d
        assert "aggregate" in d

    def test_to_dict_is_json_serialisable(self):
        import json
        ev = CrossDatasetEvaluator(datasets=_datasets(2), verbose=False)
        report = ev.run(MOSSETracker())
        # Should not raise
        json.dumps(report.to_dict())

    def test_to_markdown_contains_tracker_name(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(2), verbose=False)
        report = ev.run(MOSSETracker())
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "Cross-Dataset" in md

    def test_to_markdown_contains_dataset_names(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(2), verbose=False)
        report = ev.run(MOSSETracker())
        md = report.to_markdown()
        assert "Syn-0" in md
        assert "Syn-1" in md

    def test_to_markdown_aggregate_row_present(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(2), verbose=False)
        report = ev.run(MOSSETracker())
        md = report.to_markdown()
        assert "mean" in md.lower()

    def test_to_markdown_empty_report(self):
        report = CrossDatasetReport(tracker_name="X")
        md = report.to_markdown()
        assert "No results" in md


# ---------------------------------------------------------------------------
# compare() and to_comparison_markdown()
# ---------------------------------------------------------------------------

class TestCrossDatasetCompare:
    def test_compare_returns_one_report_per_tracker(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(1), verbose=False)
        trackers = [MOSSETracker(), KCFTracker()]
        reports = ev.compare(trackers)
        assert set(reports.keys()) == {"MOSSE", "KCF"}

    def test_compare_each_report_is_cross_dataset_report(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(1), verbose=False)
        reports = ev.compare([MOSSETracker()])
        assert isinstance(reports["MOSSE"], CrossDatasetReport)

    def test_comparison_markdown_contains_both_trackers(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(1), verbose=False)
        reports = ev.compare([MOSSETracker(), KCFTracker()])
        md = ev.to_comparison_markdown(reports)
        assert "MOSSE" in md
        assert "KCF" in md

    def test_comparison_markdown_has_rank_column(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(1), verbose=False)
        reports = ev.compare([MOSSETracker()])
        md = ev.to_comparison_markdown(reports)
        assert "Rank" in md

    def test_comparison_markdown_empty_reports(self):
        ev = CrossDatasetEvaluator(datasets=_datasets(1), verbose=False)
        md = ev.to_comparison_markdown({})
        assert "No reports" in md


# ---------------------------------------------------------------------------
# max_sequences parameter
# ---------------------------------------------------------------------------

class TestMaxSequences:
    def test_max_sequences_caps_evaluation(self):
        datasets = {
            "Big": SyntheticDataset(num_sequences=10, num_frames=10, seed=0)
        }
        ev = CrossDatasetEvaluator(datasets=datasets, verbose=False, max_sequences=3)
        report = ev.run(MOSSETracker())
        assert report.per_dataset["Big"]["num_sequences"] == 3
