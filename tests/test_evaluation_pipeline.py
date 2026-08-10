"""Unit and integration tests for EvaluationPipeline and EvaluationReport."""

from __future__ import annotations

import json
import pytest
import numpy as np

from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.mosse import MOSSETracker
from eovot.trackers.kcf import KCFTracker
from eovot.evaluation import EvaluationPipeline, EvaluationReport
from eovot.benchmark.engine import BenchmarkEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    return SyntheticDataset(num_sequences=3, num_frames=20, motion="linear", seed=0)


@pytest.fixture(scope="module")
def single_result(small_dataset):
    engine = BenchmarkEngine(verbose=False)
    return engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")


@pytest.fixture(scope="module")
def two_results(small_dataset):
    engine = BenchmarkEngine(verbose=False)
    r1 = engine.run(MOSSETracker(), small_dataset, dataset_name="Synthetic")
    r2 = engine.run(KCFTracker(), small_dataset, dataset_name="Synthetic")
    return [r1, r2]


# ---------------------------------------------------------------------------
# EvaluationReport construction
# ---------------------------------------------------------------------------

class TestEvaluationReportFromSingleResult:
    def test_returns_report(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert isinstance(report, EvaluationReport)

    def test_tracker_names(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert report.tracker_names == ["MOSSE"]

    def test_dataset_name_propagated(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert report.dataset_name == "Synthetic"

    def test_single_tracker_no_statistical(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert report.statistical is None

    def test_benchmark_results_stored(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert len(report.benchmark_results) == 1
        assert report.benchmark_results[0].tracker_name == "MOSSE"


# ---------------------------------------------------------------------------
# Robustness layer
# ---------------------------------------------------------------------------

class TestRobustnessLayer:
    def test_robustness_populated(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert "MOSSE" in report.robustness

    def test_robustness_has_aggregate(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        agg = report.robustness["MOSSE"]["aggregate"]
        assert "mean_eao" in agg
        assert "mean_survival_rate" in agg
        assert "total_failures" in agg

    def test_eao_in_range(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        eao = report.robustness["MOSSE"]["aggregate"]["mean_eao"]
        assert 0.0 <= eao <= 1.0


# ---------------------------------------------------------------------------
# Temporal consistency layer
# ---------------------------------------------------------------------------

class TestTemporalLayer:
    def test_temporal_populated(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert "MOSSE" in report.temporal

    def test_temporal_has_smoothness(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        agg = report.temporal["MOSSE"]["aggregate"]
        assert "mean_smoothness_score" in agg

    def test_smoothness_in_range(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        s = report.temporal["MOSSE"]["aggregate"]["mean_smoothness_score"]
        assert 0.0 < s <= 1.0


# ---------------------------------------------------------------------------
# Attribute layer
# ---------------------------------------------------------------------------

class TestAttributeLayer:
    def test_attributes_populated(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert "MOSSE" in report.attributes

    def test_attribute_table_has_tracker_name(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        table = report.attributes["MOSSE"]
        assert table.tracker_name == "MOSSE"

    def test_attribute_table_markdown_is_string(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        md = report.attributes["MOSSE"].to_markdown()
        assert isinstance(md, str) and len(md) > 0


# ---------------------------------------------------------------------------
# Efficiency layer
# ---------------------------------------------------------------------------

class TestEfficiencyLayer:
    def test_efficiency_populated(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert len(report.efficiency) == 1

    def test_ees_positive(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert report.efficiency[0].ees > 0.0

    def test_pareto_single_tracker(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        assert report.efficiency[0].on_pareto_front is True

    def test_efficiency_sorted_by_ees(self, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        ees_values = [e.ees for e in report.efficiency]
        assert ees_values == sorted(ees_values, reverse=True)


# ---------------------------------------------------------------------------
# Statistical layer
# ---------------------------------------------------------------------------

class TestStatisticalLayer:
    def test_two_trackers_has_statistical(self, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        assert report.statistical is not None

    def test_pairwise_trackers_listed(self, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        trackers = report.statistical.trackers
        assert "MOSSE" in trackers
        assert "KCF" in trackers

    def test_pairwise_markdown_non_empty(self, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        md = report.statistical.to_markdown()
        assert "MOSSE" in md and "KCF" in md


# ---------------------------------------------------------------------------
# EvaluationReport.to_markdown
# ---------------------------------------------------------------------------

class TestReportMarkdown:
    def test_markdown_sections_present(self, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        md = report.to_markdown()
        assert "Accuracy Overview" in md
        assert "Robustness" in md
        assert "Temporal Consistency" in md
        assert "Edge Efficiency" in md
        assert "Per-Attribute Performance" in md
        assert "Statistical Significance" in md

    def test_markdown_contains_tracker_names(self, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        md = report.to_markdown()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_markdown_is_string(self, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        md = report.to_markdown()
        assert isinstance(md, str) and len(md) > 100


# ---------------------------------------------------------------------------
# EvaluationReport.save
# ---------------------------------------------------------------------------

class TestReportSave:
    def test_save_creates_markdown(self, tmp_path, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        paths = report.save(str(tmp_path), name="test_report")
        assert paths["markdown"].exists()
        assert paths["markdown"].suffix == ".md"

    def test_save_creates_json(self, tmp_path, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        paths = report.save(str(tmp_path), name="test_report")
        assert paths["json"].exists()
        data = json.loads(paths["json"].read_text())
        assert "tracker_names" in data
        assert "robustness" in data

    def test_save_json_valid_structure(self, tmp_path, two_results):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results(two_results)
        paths = report.save(str(tmp_path), name="two_tracker")
        data = json.loads(paths["json"].read_text())
        assert set(data["tracker_names"]) == {"MOSSE", "KCF"}
        assert "efficiency" in data
        assert len(data["efficiency"]) == 2

    def test_save_creates_output_dir(self, tmp_path, single_result):
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run_from_results([single_result])
        nested = tmp_path / "nested" / "subdir"
        report.save(str(nested), name="report")
        assert nested.exists()


# ---------------------------------------------------------------------------
# EvaluationPipeline.run (end-to-end)
# ---------------------------------------------------------------------------

class TestPipelineRunEndToEnd:
    def test_run_single_tracker(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=15, motion="linear", seed=1)
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run([MOSSETracker()], ds, dataset_name="Synthetic")
        assert report.tracker_names == ["MOSSE"]
        assert len(report.benchmark_results) == 1

    def test_run_two_trackers(self):
        ds = SyntheticDataset(num_sequences=2, num_frames=15, motion="linear", seed=2)
        pipeline = EvaluationPipeline(verbose=False)
        report = pipeline.run(
            [MOSSETracker(), KCFTracker()], ds, dataset_name="Synthetic"
        )
        assert len(report.tracker_names) == 2
        assert report.statistical is not None

    def test_empty_results_raises(self):
        pipeline = EvaluationPipeline(verbose=False)
        with pytest.raises(ValueError, match="at least one"):
            pipeline.run_from_results([])
