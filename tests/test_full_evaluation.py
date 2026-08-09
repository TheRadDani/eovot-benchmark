"""Tests for eovot.experiment.full_evaluation.

Uses SyntheticDataset so no real video data is required.  All four metric
dimensions are verified — accuracy, robustness, temporal consistency,
and edge efficiency — along with report serialisation and the leaderboard.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eovot.datasets.synthetic import SyntheticDataset
from eovot.experiment.full_evaluation import (
    FullEvaluationPipeline,
    FullEvaluationReport,
    TrackerDimensions,
)
from eovot.trackers.registry import build_tracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dataset():
    return SyntheticDataset(
        num_sequences=3,
        num_frames=30,
        frame_size=(160, 120),
        bbox_size=(30, 30),
        motion="linear",
        seed=42,
    )


@pytest.fixture(scope="module")
def single_tracker_report(tiny_dataset):
    pipeline = FullEvaluationPipeline(
        memory_budget_mb=256.0,
        verbose=False,
    )
    return pipeline.run(
        trackers=build_tracker("MOSSE"),
        dataset=tiny_dataset,
        dataset_name="Synthetic-Linear",
    )


@pytest.fixture(scope="module")
def multi_tracker_report(tiny_dataset):
    pipeline = FullEvaluationPipeline(
        memory_budget_mb=256.0,
        verbose=False,
    )
    return pipeline.run(
        trackers=[build_tracker("MOSSE"), build_tracker("KCF")],
        dataset=tiny_dataset,
        dataset_name="Synthetic-Linear",
    )


# ---------------------------------------------------------------------------
# TrackerDimensions
# ---------------------------------------------------------------------------

class TestTrackerDimensions:
    def test_to_dict_keys(self):
        dim = TrackerDimensions(tracker_name="MOSSE", dataset_name="DS")
        d = dim.to_dict()
        assert set(d.keys()) == {
            "tracker", "dataset",
            "accuracy", "profiling",
            "robustness", "temporal_consistency", "edge_efficiency",
        }

    def test_to_dict_accuracy_keys(self):
        dim = TrackerDimensions(
            tracker_name="T", dataset_name="D",
            mean_iou=0.5, success_auc=0.6, precision_auc=0.7,
        )
        acc = dim.to_dict()["accuracy"]
        assert acc["mean_iou"] == pytest.approx(0.5, abs=1e-4)
        assert acc["success_auc"] == pytest.approx(0.6, abs=1e-4)
        assert acc["precision_auc"] == pytest.approx(0.7, abs=1e-4)

    def test_to_dict_none_auc(self):
        dim = TrackerDimensions(tracker_name="T", dataset_name="D")
        d = dim.to_dict()
        assert d["accuracy"]["success_auc"] is None
        assert d["accuracy"]["precision_auc"] is None

    def test_str_repr(self):
        dim = TrackerDimensions(tracker_name="KCF", dataset_name="OTB")
        s = str(dim)
        assert "KCF" in s
        assert "OTB" in s
        assert "mIoU" in s
        assert "EES" in s


# ---------------------------------------------------------------------------
# FullEvaluationPipeline — single tracker
# ---------------------------------------------------------------------------

class TestFullEvaluationPipelineSingle:
    def test_returns_report(self, single_tracker_report):
        assert isinstance(single_tracker_report, FullEvaluationReport)

    def test_one_evaluation(self, single_tracker_report):
        assert len(single_tracker_report.evaluations) == 1

    def test_tracker_name(self, single_tracker_report):
        assert single_tracker_report.evaluations[0].tracker_name == "MOSSE"

    def test_dataset_name(self, single_tracker_report):
        assert single_tracker_report.dataset_name == "Synthetic-Linear"

    def test_mean_iou_range(self, single_tracker_report):
        dim = single_tracker_report.evaluations[0]
        assert 0.0 <= dim.mean_iou <= 1.0

    def test_mean_fps_positive(self, single_tracker_report):
        dim = single_tracker_report.evaluations[0]
        assert dim.mean_fps > 0.0

    def test_ees_positive(self, single_tracker_report):
        dim = single_tracker_report.evaluations[0]
        # EES = mean_iou × log1p(fps) / (1 + mem/budget) ≥ 0
        assert dim.ees >= 0.0

    def test_robustness_populated(self, single_tracker_report):
        dim = single_tracker_report.evaluations[0]
        assert 0.0 <= dim.mean_eao <= 1.0
        assert 0.0 <= dim.mean_survival_rate <= 1.0
        assert dim.total_failures >= 0

    def test_temporal_populated(self, single_tracker_report):
        dim = single_tracker_report.evaluations[0]
        assert 0.0 < dim.mean_smoothness_score <= 1.0
        assert dim.mean_position_jitter >= 0.0
        assert dim.mean_scale_jitter >= 0.0
        assert 0.0 <= dim.mean_velocity_outlier_ratio <= 1.0

    def test_single_tracker_always_pareto(self, single_tracker_report):
        # With only one tracker there is nothing to dominate it.
        assert single_tracker_report.evaluations[0].on_pareto_front is True


# ---------------------------------------------------------------------------
# FullEvaluationPipeline — multi-tracker
# ---------------------------------------------------------------------------

class TestFullEvaluationPipelineMulti:
    def test_two_evaluations(self, multi_tracker_report):
        assert len(multi_tracker_report.evaluations) == 2

    def test_tracker_names(self, multi_tracker_report):
        names = {e.tracker_name for e in multi_tracker_report.evaluations}
        assert names == {"MOSSE", "KCF"}

    def test_at_least_one_pareto(self, multi_tracker_report):
        pareto = [e for e in multi_tracker_report.evaluations if e.on_pareto_front]
        assert len(pareto) >= 1

    def test_best_by_ees_is_pareto(self, multi_tracker_report):
        best = multi_tracker_report.best_by_ees()
        assert best is not None
        assert best.on_pareto_front

    def test_best_by_accuracy_not_none(self, multi_tracker_report):
        assert multi_tracker_report.best_by_accuracy() is not None

    def test_pareto_trackers_list(self, multi_tracker_report):
        pareto = multi_tracker_report.pareto_trackers()
        assert all(e.on_pareto_front for e in pareto)


# ---------------------------------------------------------------------------
# FullEvaluationReport — serialisation
# ---------------------------------------------------------------------------

class TestFullEvaluationReportSave:
    def test_save_creates_file(self, single_tracker_report, tmp_path):
        out = tmp_path / "report.json"
        path = single_tracker_report.save(out)
        assert path.exists()

    def test_save_auto_adds_suffix(self, single_tracker_report, tmp_path):
        out = tmp_path / "report"
        path = single_tracker_report.save(out)
        assert path.suffix == ".json"

    def test_save_valid_json(self, single_tracker_report, tmp_path):
        out = tmp_path / "report.json"
        single_tracker_report.save(out)
        data = json.loads(out.read_text())
        assert "dataset" in data
        assert "evaluations" in data
        assert isinstance(data["evaluations"], list)
        assert len(data["evaluations"]) == 1

    def test_save_evaluation_structure(self, multi_tracker_report, tmp_path):
        out = tmp_path / "multi.json"
        multi_tracker_report.save(out)
        data = json.loads(out.read_text())
        for ev in data["evaluations"]:
            assert "accuracy" in ev
            assert "robustness" in ev
            assert "temporal_consistency" in ev
            assert "edge_efficiency" in ev

    def test_save_creates_parent_dirs(self, single_tracker_report, tmp_path):
        out = tmp_path / "deep" / "nested" / "report.json"
        path = single_tracker_report.save(out)
        assert path.exists()

    def test_roundtrip_ees(self, multi_tracker_report, tmp_path):
        out = tmp_path / "rt.json"
        multi_tracker_report.save(out)
        data = json.loads(out.read_text())
        loaded_ees = [
            ev["edge_efficiency"]["ees"] for ev in data["evaluations"]
        ]
        orig_ees = [e.ees for e in multi_tracker_report.evaluations]
        for le, oe in zip(sorted(loaded_ees), sorted(orig_ees)):
            assert le == pytest.approx(oe, abs=1e-3)


# ---------------------------------------------------------------------------
# FullEvaluationReport — Markdown table
# ---------------------------------------------------------------------------

class TestMarkdownTable:
    def test_markdown_contains_tracker_name(self, multi_tracker_report):
        md = multi_tracker_report.to_markdown_table()
        assert "MOSSE" in md
        assert "KCF" in md

    def test_markdown_has_table_header(self, multi_tracker_report):
        md = multi_tracker_report.to_markdown_table()
        assert "| Rank |" in md
        assert "EES" in md

    def test_markdown_has_legend(self, multi_tracker_report):
        md = multi_tracker_report.to_markdown_table()
        assert "EES" in md
        assert "Pareto" in md


# ---------------------------------------------------------------------------
# Pipeline constructor validation
# ---------------------------------------------------------------------------

def test_invalid_memory_budget():
    with pytest.raises(ValueError, match="memory_budget_mb"):
        FullEvaluationPipeline(memory_budget_mb=0.0)

    with pytest.raises(ValueError, match="memory_budget_mb"):
        FullEvaluationPipeline(memory_budget_mb=-100.0)
