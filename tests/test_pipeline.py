"""Smoke tests for FullAnalysisPipeline on SyntheticDataset.

No real-data downloads required — all sequences are generated in-memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eovot.analysis.pipeline import FullAnalysisPipeline
from eovot.datasets.synthetic import SyntheticDataset
from eovot.trackers.registry import build_tracker


@pytest.fixture
def tiny_dataset():
    """3 sequences of 30 frames each — fast but exercises all code paths."""
    return SyntheticDataset(num_sequences=3, num_frames=30, motion="linear", seed=0)


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "pipeline_test")


def test_pipeline_single_tracker(tiny_dataset, output_dir):
    """Pipeline runs on a single tracker and returns expected top-level keys."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],  # skip device sim
    )
    tracker = build_tracker("MOSSE")
    report = pipeline.run([tracker], tiny_dataset, dataset_name="Synthetic")

    assert "tracker_results" in report
    assert "efficiency" in report
    assert "leaderboard_md" in report
    assert "output_dir" in report

    assert len(report["tracker_results"]) == 1
    tr = report["tracker_results"][0]
    assert tr["tracker_name"] == "MOSSE"
    assert "benchmark" in tr
    assert "robustness" in tr
    assert "temporal" in tr
    assert "attributes" in tr


def test_pipeline_multi_tracker(tiny_dataset, output_dir):
    """Pipeline ranks multiple trackers by EES and marks Pareto front."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    trackers = [build_tracker("MOSSE"), build_tracker("KCF")]
    report = pipeline.run(trackers, tiny_dataset, dataset_name="Synthetic")

    assert len(report["tracker_results"]) == 2
    ranking = report["efficiency"]["ranking"]
    assert len(ranking) == 2
    # EES must be sorted descending
    assert ranking[0]["ees"] >= ranking[1]["ees"]


def test_pipeline_benchmark_metrics(tiny_dataset, output_dir):
    """Benchmark section contains all scalar metrics."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    report = pipeline.run(
        [build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic"
    )
    bm = report["tracker_results"][0]["benchmark"]
    assert "mean_iou" in bm
    assert "mean_fps" in bm
    assert "peak_memory_mb" in bm


def test_pipeline_robustness_keys(tiny_dataset, output_dir):
    """Robustness section exposes expected aggregate keys."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    report = pipeline.run(
        [build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic"
    )
    rob = report["tracker_results"][0]["robustness"]
    assert "mean_eao" in rob
    assert "total_failures" in rob
    assert "mean_survival_rate" in rob


def test_pipeline_temporal_keys(tiny_dataset, output_dir):
    """Temporal section exposes expected aggregate keys."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    report = pipeline.run(
        [build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic"
    )
    temp = report["tracker_results"][0]["temporal"]
    assert "mean_smoothness_score" in temp
    assert "mean_position_jitter" in temp


def test_pipeline_device_sim(tiny_dataset, output_dir):
    """Device simulation section is included when device_names is not empty."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=["rpi4", "jetson_nano"],
        sustained_seconds=30.0,
    )
    report = pipeline.run(
        [build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic"
    )
    tr = report["tracker_results"][0]
    assert "device_sim" in tr
    devices = {d["device"] for d in tr["device_sim"]}
    assert "rpi4" in devices
    assert "jetson_nano" in devices


def test_pipeline_json_saved(tiny_dataset, output_dir):
    """full_analysis.json is written and is valid JSON."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    pipeline.run([build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic")
    json_path = Path(output_dir) / "full_analysis.json"
    assert json_path.exists()
    with open(json_path) as fh:
        data = json.load(fh)
    assert "tracker_results" in data


def test_pipeline_leaderboard_md_saved(tiny_dataset, output_dir):
    """leaderboard.md is written and contains the tracker name."""
    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    pipeline.run([build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic")
    md_path = Path(output_dir) / "leaderboard.md"
    assert md_path.exists()
    content = md_path.read_text()
    assert "MOSSE" in content


def test_pipeline_csv_columns(tiny_dataset, output_dir):
    """CSV output uses updated column names (success_auc, precision_auc)."""
    import csv

    pipeline = FullAnalysisPipeline(
        output_dir=output_dir,
        verbose=False,
        device_names=[],
    )
    pipeline.run([build_tracker("MOSSE")], tiny_dataset, dataset_name="Synthetic")
    csv_path = Path(output_dir) / "MOSSE" / "MOSSE-Synthetic.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
    assert "success_auc" in fieldnames
    assert "precision_auc" in fieldnames
    # The old stale column name must not appear.
    assert "precision_score" not in fieldnames


def test_pipeline_max_sequences(tmp_path):
    """max_sequences limits evaluation without breaking the pipeline."""
    dataset = SyntheticDataset(num_sequences=5, num_frames=20, seed=99)
    pipeline = FullAnalysisPipeline(
        output_dir=str(tmp_path / "max_seq"),
        verbose=False,
        device_names=[],
        max_sequences=2,
    )
    report = pipeline.run(
        [build_tracker("MOSSE")], dataset, dataset_name="Synthetic"
    )
    tr = report["tracker_results"][0]
    assert tr["benchmark"]["num_sequences"] == 2
