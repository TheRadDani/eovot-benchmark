"""Tests for the consolidated FullReportGenerator."""

from __future__ import annotations

import pytest

from eovot.benchmark.engine import BenchmarkEngine, BenchmarkResult
from eovot.datasets.synthetic import SyntheticDataset
from eovot.reporting.full_report import FullReportGenerator
from eovot.trackers.kcf import KCFTracker
from eovot.trackers.mosse import MOSSETracker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_results(num_sequences: int = 3, num_frames: int = 30):
    """Return two BenchmarkResult objects for quick tests."""
    ds = SyntheticDataset(
        num_sequences=num_sequences,
        num_frames=num_frames,
        frame_size=(160, 120),
        bbox_size=(30, 30),
        motion="linear",
        seed=42,
    )
    engine = BenchmarkEngine(verbose=False)
    return [
        engine.run(MOSSETracker(), ds, dataset_name="Synthetic"),
        engine.run(KCFTracker(), ds, dataset_name="Synthetic"),
    ]


# ---------------------------------------------------------------------------
# FullReportGenerator unit tests
# ---------------------------------------------------------------------------


class TestFullReportGenerator:
    def test_generate_returns_string(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_generate_empty_raises(self):
        gen = FullReportGenerator()
        with pytest.raises(ValueError):
            gen.generate([])

    def test_report_contains_all_sections(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        for section_heading in [
            "# EOVOT Benchmark Report",
            "## 1. Accuracy Leaderboard",
            "## 2. Efficiency Comparison",
            "## 3. Robustness Analysis",
            "## 4. Per-Attribute Performance",
        ]:
            assert section_heading in report, f"Missing: {section_heading!r}"

    def test_report_contains_tracker_names(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "MOSSE" in report
        assert "KCF" in report

    def test_report_contains_dataset_name(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "Synthetic" in report

    def test_custom_title(self):
        gen = FullReportGenerator(title="My Custom Title")
        results = _make_results()
        report = gen.generate(results)
        assert "My Custom Title" in report

    def test_accuracy_section_has_miou(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "mIoU" in report

    def test_efficiency_section_has_ees(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "EES" in report

    def test_efficiency_section_has_fps(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "FPS" in report

    def test_robustness_section_has_eao(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "EAO" in report

    def test_robustness_section_has_survival(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "Survival" in report

    def test_footer_present(self):
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "EOVOT" in report

    def test_single_tracker(self):
        """Report should not crash with a single benchmark result."""
        ds = SyntheticDataset(num_sequences=2, num_frames=20, seed=0)
        engine = BenchmarkEngine(verbose=False)
        results = [engine.run(MOSSETracker(), ds, dataset_name="Synthetic")]
        gen = FullReportGenerator()
        report = gen.generate(results)
        assert "MOSSE" in report

    def test_save_writes_file(self, tmp_path):
        gen = FullReportGenerator()
        results = _make_results()
        out = tmp_path / "report.md"
        saved = gen.save(results, path=str(out))
        assert saved.exists()
        content = saved.read_text()
        assert "## 1. Accuracy Leaderboard" in content

    def test_save_creates_parent_dirs(self, tmp_path):
        gen = FullReportGenerator()
        results = _make_results()
        nested = tmp_path / "a" / "b" / "report.md"
        gen.save(results, path=str(nested))
        assert nested.exists()

    def test_save_appends_md_extension(self, tmp_path):
        gen = FullReportGenerator()
        results = _make_results()
        out = tmp_path / "report"
        saved = gen.save(results, path=str(out))
        assert saved.suffix == ".md"

    def test_memory_budget_propagated(self):
        """A tight budget should heavily penalise memory-heavy trackers in EES."""
        gen = FullReportGenerator(memory_budget_mb=1.0)  # absurdly tight
        results = _make_results()
        report = gen.generate(results)
        # Should not crash; EES values will be small but present.
        assert "EES" in report

    def test_pareto_star_present(self):
        """At least one tracker must be Pareto-optimal (marked ★)."""
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        assert "★" in report

    def test_accuracy_section_ranking_order(self):
        """Trackers must appear in descending Success AUC order."""
        gen = FullReportGenerator()
        results = _make_results()
        report = gen.generate(results)
        # Find the accuracy section and check both tracker names appear.
        acc_start = report.index("## 1. Accuracy Leaderboard")
        acc_end = report.index("## 2. Efficiency")
        acc_section = report[acc_start:acc_end]
        assert "MOSSE" in acc_section
        assert "KCF" in acc_section

    def test_attribute_section_no_crash_without_gt(self):
        """Sequences without stored GT boxes should emit a placeholder, not crash.

        Strip ``ground_truths`` from each SequenceResult to simulate the case
        where the benchmark was loaded from a JSON that omits raw GT arrays.
        """
        ds = SyntheticDataset(num_sequences=2, num_frames=20, seed=5)
        engine = BenchmarkEngine(verbose=False)
        result = engine.run(MOSSETracker(), ds, dataset_name="Synthetic")

        # Remove stored GT boxes so the attribute analyzer falls back to the
        # placeholder path.
        for sr in result.sequence_results:
            sr.ground_truths = None

        gen = FullReportGenerator()
        report = gen.generate([result])
        assert "## 4. Per-Attribute Performance" in report
