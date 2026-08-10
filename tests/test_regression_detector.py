"""Unit tests for RegressionDetector and related utilities.

Tests cover:
- MetricStatus enum values
- RegressionDetector construction validation
- compare() with IMPROVED / PASS / WARN / FAIL / MISSING outcomes
- Directionality correctness (higher-is-better vs lower-is-better)
- Exit code derivation
- RegressionReport.summary() rendering
- RegressionReport.to_dict() schema
- load_scalars_from_file() with nested and flat JSON formats
- RegressionReport.to_json() serialisation round-trip
- compare_results() with live BenchmarkResult objects
- CLI script via main() call
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eovot.analysis.regression import (
    ALL_TRACKED_METRICS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MetricDelta,
    MetricStatus,
    RegressionDetector,
    RegressionReport,
    load_scalars_from_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _baseline() -> dict:
    return {
        "mean_iou": 0.600,
        "success_auc": 0.550,
        "precision_auc": 0.700,
        "mean_fps": 120.0,
        "peak_memory_mb": 200.0,
        "mean_latency_ms": 8.3,
        "mean_center_distance_px": 15.0,
    }


def _detector(warn=2.0, fail=5.0) -> RegressionDetector:
    return RegressionDetector(warn_pct=warn, fail_pct=fail)


# ---------------------------------------------------------------------------
# MetricStatus
# ---------------------------------------------------------------------------

class TestMetricStatus:
    def test_all_values_present(self):
        assert MetricStatus.IMPROVED.value == "IMPROVED"
        assert MetricStatus.PASS.value    == "PASS"
        assert MetricStatus.WARN.value    == "WARN"
        assert MetricStatus.FAIL.value    == "FAIL"
        assert MetricStatus.MISSING.value == "MISSING"


# ---------------------------------------------------------------------------
# Directionality tables
# ---------------------------------------------------------------------------

class TestDirectionalityTables:
    def test_higher_is_better_contents(self):
        assert "mean_iou" in HIGHER_IS_BETTER
        assert "success_auc" in HIGHER_IS_BETTER
        assert "mean_fps" in HIGHER_IS_BETTER

    def test_lower_is_better_contents(self):
        assert "peak_memory_mb" in LOWER_IS_BETTER
        assert "mean_latency_ms" in LOWER_IS_BETTER

    def test_no_overlap(self):
        assert HIGHER_IS_BETTER.isdisjoint(LOWER_IS_BETTER)

    def test_all_tracked_union(self):
        assert ALL_TRACKED_METRICS == HIGHER_IS_BETTER | LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# RegressionDetector construction
# ---------------------------------------------------------------------------

class TestRegressionDetectorConstruction:
    def test_defaults(self):
        d = RegressionDetector()
        assert d.warn_pct == 2.0
        assert d.fail_pct == 5.0

    def test_custom_thresholds(self):
        d = RegressionDetector(warn_pct=1.0, fail_pct=3.0)
        assert d.warn_pct == 1.0
        assert d.fail_pct == 3.0

    def test_invalid_warn_pct_raises(self):
        with pytest.raises(ValueError):
            RegressionDetector(warn_pct=0.0)
        with pytest.raises(ValueError):
            RegressionDetector(warn_pct=-1.0)

    def test_fail_pct_must_exceed_warn_pct(self):
        with pytest.raises(ValueError):
            RegressionDetector(warn_pct=5.0, fail_pct=5.0)
        with pytest.raises(ValueError):
            RegressionDetector(warn_pct=5.0, fail_pct=3.0)


# ---------------------------------------------------------------------------
# compare() — status classification
# ---------------------------------------------------------------------------

class TestCompareStatusClassification:
    def test_no_change_is_pass(self):
        b = {"mean_iou": 0.6}
        report = _detector().compare(b, dict(b))
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.PASS

    def test_small_improvement_is_improved(self):
        b = {"mean_iou": 0.600}
        c = {"mean_iou": 0.620}  # +3.3 % > warn_pct=2 %
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.IMPROVED

    def test_small_regression_within_noise_is_pass(self):
        b = {"mean_iou": 0.600}
        c = {"mean_iou": 0.598}  # −0.33 % within warn=2 %
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.PASS

    def test_moderate_regression_is_warn(self):
        b = {"mean_iou": 0.600}
        c = {"mean_iou": 0.575}  # −4.17 % between warn=2 % and fail=5 %
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.WARN

    def test_large_regression_is_fail(self):
        b = {"mean_iou": 0.600}
        c = {"mean_iou": 0.540}  # −10 % > fail=5 %
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.FAIL

    def test_missing_in_current_is_missing(self):
        b = {"mean_iou": 0.600}
        c: dict = {}
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.MISSING
        assert delta.current is None

    def test_missing_in_baseline_is_missing(self):
        b: dict = {}
        c = {"mean_iou": 0.600}
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_iou")
        assert delta.status == MetricStatus.MISSING
        assert delta.baseline is None


# ---------------------------------------------------------------------------
# Directionality correctness
# ---------------------------------------------------------------------------

class TestDirectionality:
    def test_lower_memory_is_improved(self):
        b = {"peak_memory_mb": 200.0}
        c = {"peak_memory_mb": 190.0}  # −5 % (good for lower-is-better)
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "peak_memory_mb")
        assert delta.status == MetricStatus.IMPROVED
        assert delta.higher_is_better is False

    def test_higher_memory_is_fail(self):
        b = {"peak_memory_mb": 200.0}
        c = {"peak_memory_mb": 220.0}  # +10 % (bad for lower-is-better)
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "peak_memory_mb")
        assert delta.status == MetricStatus.FAIL

    def test_lower_latency_is_improved(self):
        b = {"mean_latency_ms": 10.0}
        c = {"mean_latency_ms": 9.0}   # −10 % better
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_latency_ms")
        assert delta.status == MetricStatus.IMPROVED

    def test_higher_fps_is_improved(self):
        b = {"mean_fps": 100.0}
        c = {"mean_fps": 110.0}  # +10 % better
        report = _detector().compare(b, c)
        delta = next(d for d in report.deltas if d.name == "mean_fps")
        assert delta.status == MetricStatus.IMPROVED


# ---------------------------------------------------------------------------
# Multi-metric report
# ---------------------------------------------------------------------------

class TestMultiMetricReport:
    def test_full_baseline_comparison(self):
        b = _baseline()
        # Slightly better everywhere except a large FPS regression
        c = {
            "mean_iou":               0.615,   # +2.5 % improved
            "success_auc":            0.552,   # +0.36 % pass
            "precision_auc":          0.700,   # 0 % pass
            "mean_fps":               90.0,    # −25 % FAIL
            "peak_memory_mb":         195.0,   # −2.5 % improved
            "mean_latency_ms":        8.3,     # 0 % pass
            "mean_center_distance_px": 15.0,  # 0 % pass
        }
        report = _detector().compare(b, c)
        assert report.has_failures  # fps regressed by 25 %
        assert report.exit_code == 2

    def test_all_improved_gives_exit_0(self):
        b = _baseline()
        c = {k: v * 1.1 if k in HIGHER_IS_BETTER else v * 0.9 for k, v in b.items()}
        report = _detector().compare(b, c)
        assert not report.has_failures
        assert not report.has_warnings
        assert report.exit_code == 0

    def test_warn_only_gives_exit_1(self):
        b = {"mean_iou": 0.600}
        c = {"mean_iou": 0.582}  # −3 % between warn=2 % and fail=5 %
        report = _detector().compare(b, c)
        assert report.has_warnings
        assert not report.has_failures
        assert report.exit_code == 1


# ---------------------------------------------------------------------------
# RegressionReport properties
# ---------------------------------------------------------------------------

class TestRegressionReportProperties:
    def test_overall_status_pass(self):
        b = {"mean_iou": 0.6}
        report = _detector().compare(b, dict(b))
        assert report.overall_status == "PASS"

    def test_overall_status_warn(self):
        b = {"mean_iou": 0.6}
        c = {"mean_iou": 0.582}
        report = _detector().compare(b, c)
        assert report.overall_status == "WARN"

    def test_overall_status_fail(self):
        b = {"mean_iou": 0.6}
        c = {"mean_iou": 0.54}
        report = _detector().compare(b, c)
        assert report.overall_status == "FAIL"

    def test_summary_contains_tracker_name(self):
        report = _detector().compare({"mean_iou": 0.6}, {"mean_iou": 0.6},
                                     tracker_name="KCF", dataset_name="OTB100")
        s = report.summary(color=False)
        assert "KCF" in s
        assert "OTB100" in s

    def test_summary_no_color(self):
        report = _detector().compare({"mean_iou": 0.6}, {"mean_iou": 0.54})
        s = report.summary(color=False)
        assert "\033[" not in s

    def test_summary_with_color(self):
        report = _detector().compare({"mean_iou": 0.6}, {"mean_iou": 0.54})
        s = report.summary(color=True)
        assert "\033[" in s  # ANSI codes present

    def test_to_dict_schema(self):
        report = _detector().compare({"mean_iou": 0.6}, {"mean_iou": 0.54},
                                     tracker_name="T", dataset_name="D")
        d = report.to_dict()
        assert "tracker_name" in d
        assert "dataset_name" in d
        assert "overall" in d
        assert "exit_code" in d
        assert isinstance(d["metrics"], list)
        m = d["metrics"][0]
        assert "name" in m and "status" in m and "relative_pct" in m

    def test_to_dict_overall_values(self):
        report = _detector().compare({"mean_iou": 0.6}, {"mean_iou": 0.6})
        d = report.to_dict()
        assert d["overall"] == "pass"
        assert d["exit_code"] == 0


# ---------------------------------------------------------------------------
# load_scalars_from_file
# ---------------------------------------------------------------------------

class TestLoadScalarsFromFile:
    def _write_json(self, data: dict) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="/tmp"
        ) as f:
            json.dump(data, f)
            return Path(f.name)

    def test_nested_format(self):
        data = {
            "summary": {"mean_iou": 0.55, "mean_fps": 110.0, "tracker": "MOSSE"},
            "sequences": [],
        }
        path = self._write_json(data)
        scalars = load_scalars_from_file(path)
        assert scalars["mean_iou"] == pytest.approx(0.55)
        assert scalars["mean_fps"] == pytest.approx(110.0)
        # Non-tracked fields should not appear
        assert "tracker" not in scalars

    def test_flat_format(self):
        data = {"mean_iou": 0.60, "peak_memory_mb": 250.0}
        path = self._write_json(data)
        scalars = load_scalars_from_file(path)
        assert scalars["mean_iou"] == pytest.approx(0.60)
        assert scalars["peak_memory_mb"] == pytest.approx(250.0)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_scalars_from_file("/nonexistent/path/result.json")

    def test_unknown_metrics_excluded(self):
        data = {"mean_iou": 0.5, "foo_bar_metric": 42.0}
        path = self._write_json(data)
        scalars = load_scalars_from_file(path)
        assert "foo_bar_metric" not in scalars


# ---------------------------------------------------------------------------
# RegressionReport.to_json
# ---------------------------------------------------------------------------

class TestToJson:
    def test_roundtrip(self):
        b = _baseline()
        c = {**b, "mean_iou": 0.540}  # one failure
        report = _detector().compare(b, c, tracker_name="MOSSE", dataset_name="Syn")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report"
            saved = report.to_json(out)
            assert saved.suffix == ".json"
            with open(saved) as fh:
                loaded = json.load(fh)
        assert loaded["overall"] == "fail"
        assert loaded["tracker_name"] == "MOSSE"
        assert isinstance(loaded["metrics"], list)


# ---------------------------------------------------------------------------
# compare_results() with live BenchmarkResult
# ---------------------------------------------------------------------------

class TestCompareResults:
    def _make_result(self, tracker: str, iou: float):
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.trackers.mosse import MOSSETracker

        dataset = SyntheticDataset(num_sequences=1, num_frames=20, motion="linear")
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), dataset, dataset_name="Synthetic")
        result.tracker_name = tracker
        return result

    def test_compare_results_same_run_passes(self):
        result = self._make_result("MOSSE", 0.5)
        report = _detector().compare_results(result, result)
        assert report.exit_code == 0

    def test_compare_results_names_inferred(self):
        result = self._make_result("KCF", 0.5)
        report = _detector().compare_results(result, result)
        assert report.tracker_name == "KCF"
        assert report.dataset_name == "Synthetic"


# ---------------------------------------------------------------------------
# CLI main() function
# ---------------------------------------------------------------------------

class TestCLIMain:
    def _write_result(self, path: Path, scalars: dict) -> None:
        data = {"summary": scalars, "sequences": []}
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_cli_pass(self, tmp_path):
        from scripts.diff_results import main

        b_path = tmp_path / "baseline.json"
        c_path = tmp_path / "current.json"
        scalars = {"mean_iou": 0.600, "mean_fps": 120.0}
        self._write_result(b_path, scalars)
        self._write_result(c_path, scalars)

        code = main([str(b_path), str(c_path), "--no-color"])
        assert code == 0

    def test_cli_fail(self, tmp_path):
        from scripts.diff_results import main

        b_path = tmp_path / "baseline.json"
        c_path = tmp_path / "current.json"
        self._write_result(b_path, {"mean_iou": 0.600})
        self._write_result(c_path, {"mean_iou": 0.500})  # −16.7 % → FAIL

        code = main([str(b_path), str(c_path), "--no-color"])
        assert code == 2

    def test_cli_output_json(self, tmp_path):
        from scripts.diff_results import main

        b_path = tmp_path / "b.json"
        c_path = tmp_path / "c.json"
        out_path = tmp_path / "out.json"
        scalars = {"mean_iou": 0.600}
        self._write_result(b_path, scalars)
        self._write_result(c_path, scalars)

        main([str(b_path), str(c_path), "--no-color", "--output-json", str(out_path)])
        assert out_path.exists()
        with open(out_path) as fh:
            d = json.load(fh)
        assert "overall" in d

    def test_cli_missing_file_returns_2(self, tmp_path):
        from scripts.diff_results import main

        code = main([str(tmp_path / "nonexistent.json"), str(tmp_path / "other.json")])
        assert code == 2
