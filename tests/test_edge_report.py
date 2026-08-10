"""Unit and integration tests for EdgeDeploymentReporter."""

from __future__ import annotations

import json
import pytest
import numpy as np

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.edge_report import EdgeDeploymentReporter, DeploymentVerdict
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_result() -> BenchmarkResult:
    """Run MOSSE on a tiny synthetic dataset for fast test execution."""
    ds = SyntheticDataset(num_sequences=3, num_frames=20, seed=0)
    engine = BenchmarkEngine(verbose=False)
    return engine.run(MOSSETracker(), ds, dataset_name="SyntheticTest")


# ---------------------------------------------------------------------------
# Reporter construction
# ---------------------------------------------------------------------------

def test_default_construction():
    reporter = EdgeDeploymentReporter()
    assert reporter.min_fps == 10.0
    assert reporter.memory_budget_mb == 512.0
    assert reporter.sustained_seconds == 60.0


def test_invalid_min_fps():
    with pytest.raises(ValueError):
        EdgeDeploymentReporter(min_fps=0.0)

    with pytest.raises(ValueError):
        EdgeDeploymentReporter(min_fps=-5.0)


# ---------------------------------------------------------------------------
# generate() returns valid Markdown
# ---------------------------------------------------------------------------

def test_generate_returns_string(small_result):
    reporter = EdgeDeploymentReporter()
    md = reporter.generate(small_result)
    assert isinstance(md, str) and len(md) > 100


def test_generate_contains_expected_sections(small_result):
    reporter = EdgeDeploymentReporter()
    md = reporter.generate(small_result)
    assert "Edge Deployment Report" in md
    assert "Accuracy Metrics" in md
    assert "Host Machine Profiling" in md
    assert "Edge Device Projections" in md
    assert "Edge Efficiency Score" in md
    assert "Deployment Verdicts" in md


def test_generate_contains_tracker_name(small_result):
    reporter = EdgeDeploymentReporter()
    md = reporter.generate(small_result)
    assert small_result.tracker_name in md


def test_generate_contains_dataset_name(small_result):
    reporter = EdgeDeploymentReporter()
    md = reporter.generate(small_result)
    assert small_result.dataset_name in md


# ---------------------------------------------------------------------------
# Device subset
# ---------------------------------------------------------------------------

def test_device_subset(small_result):
    reporter = EdgeDeploymentReporter(device_names=["rpi4", "rpi5"])
    md = reporter.generate(small_result)
    assert "Raspberry Pi" in md


def test_single_device(small_result):
    reporter = EdgeDeploymentReporter(device_names=["jetson_nano"])
    md = reporter.generate(small_result)
    assert "Jetson" in md


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_structure(small_result):
    reporter = EdgeDeploymentReporter()
    d = reporter.to_dict(small_result)
    assert "tracker" in d
    assert "dataset" in d
    assert "accuracy" in d
    assert "host_profiling" in d
    assert "device_projections" in d
    assert "efficiency" in d
    assert "verdicts" in d


def test_to_dict_accuracy_fields(small_result):
    reporter = EdgeDeploymentReporter()
    d = reporter.to_dict(small_result)
    acc = d["accuracy"]
    assert 0.0 <= acc["mean_iou"] <= 1.0
    assert "success_auc" in acc


def test_to_dict_verdicts_list(small_result):
    reporter = EdgeDeploymentReporter()
    d = reporter.to_dict(small_result)
    assert isinstance(d["verdicts"], list)
    assert len(d["verdicts"]) > 0
    for v in d["verdicts"]:
        assert "device" in v
        assert "deployable" in v
        assert isinstance(v["deployable"], bool)


def test_to_dict_json_serialisable(small_result):
    reporter = EdgeDeploymentReporter()
    d = reporter.to_dict(small_result)
    serialised = json.dumps(d)
    recovered = json.loads(serialised)
    assert recovered["tracker"] == small_result.tracker_name


# ---------------------------------------------------------------------------
# Deployment verdicts
# ---------------------------------------------------------------------------

def test_high_min_fps_produces_no_deploy(small_result):
    """With a very high FPS requirement, no device should be deployable."""
    reporter = EdgeDeploymentReporter(min_fps=100_000.0)
    sim_results = reporter._run_simulation(small_result)
    verdicts = reporter._compute_verdicts(small_result, sim_results)
    assert all(not v.deployable for v in verdicts)


def test_zero_min_fps_produces_all_deploy_if_memory_ok(small_result):
    """With a trivially low FPS requirement, memory-fitting devices should deploy."""
    reporter = EdgeDeploymentReporter(min_fps=0.001, memory_budget_mb=512.0)
    sim_results = reporter._run_simulation(small_result)
    verdicts = reporter._compute_verdicts(small_result, sim_results)
    # At least some devices should be deployable when FPS bar is nearly zero
    assert any(v.deployable for v in verdicts)


def test_verdict_reason_non_empty(small_result):
    reporter = EdgeDeploymentReporter()
    sim_results = reporter._run_simulation(small_result)
    verdicts = reporter._compute_verdicts(small_result, sim_results)
    for v in verdicts:
        assert isinstance(v.reason, str) and len(v.reason) > 0


# ---------------------------------------------------------------------------
# save() writes a file
# ---------------------------------------------------------------------------

def test_save_writes_file(small_result, tmp_path):
    reporter = EdgeDeploymentReporter()
    out = reporter.save(small_result, path=str(tmp_path / "report.md"))
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Edge Deployment Report" in content


def test_save_creates_parent_dirs(small_result, tmp_path):
    deep_path = tmp_path / "a" / "b" / "c" / "report.md"
    reporter = EdgeDeploymentReporter()
    out = reporter.save(small_result, path=str(deep_path))
    assert out.exists()


# ---------------------------------------------------------------------------
# With energy profiling enabled
# ---------------------------------------------------------------------------

def test_with_energy_profiling():
    ds = SyntheticDataset(num_sequences=2, num_frames=15, seed=1)
    engine = BenchmarkEngine(verbose=False, tdp_watts=15.0)
    result = engine.run(MOSSETracker(), ds, dataset_name="EnergyTest")
    reporter = EdgeDeploymentReporter()
    md = reporter.generate(result)
    assert "Total energy" in md or "energy" in md.lower()
