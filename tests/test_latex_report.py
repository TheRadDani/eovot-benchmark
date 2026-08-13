"""Tests for eovot.reporting.latex_report.LaTeXReportGenerator."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest

from eovot.reporting.latex_report import (
    LaTeXReportGenerator,
    _compute_ees,
    DEFAULT_COLUMNS,
    _BUILTIN_COLUMNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    tracker="MOSSE",
    dataset="OTB-100",
    mean_iou=0.65,
    success_auc=0.60,
    precision_auc=0.75,
    mean_fps=120.0,
    peak_memory_mb=64.0,
    mean_latency_ms=8.3,
    mean_energy_per_frame_mj=0.012,
) -> dict:
    return {
        "summary": {
            "tracker": tracker,
            "dataset": dataset,
            "mean_iou": mean_iou,
            "success_auc": success_auc,
            "precision_auc": precision_auc,
            "mean_fps": mean_fps,
            "peak_memory_mb": peak_memory_mb,
            "mean_latency_ms": mean_latency_ms,
            "mean_energy_per_frame_mj": mean_energy_per_frame_mj,
        }
    }


SINGLE = [_make_result()]

MULTI = [
    _make_result("MOSSE", "OTB-100", mean_iou=0.65, mean_fps=120.0, peak_memory_mb=64.0),
    _make_result("KCF",   "OTB-100", mean_iou=0.72, mean_fps=80.0,  peak_memory_mb=128.0),
    _make_result("CSRT",  "OTB-100", mean_iou=0.80, mean_fps=25.0,  peak_memory_mb=256.0),
]

MULTI_DATASET = [
    _make_result("MOSSE", "OTB-100",  mean_iou=0.65),
    _make_result("MOSSE", "LaSOT",    mean_iou=0.40),
]


# ---------------------------------------------------------------------------
# _compute_ees
# ---------------------------------------------------------------------------

class TestComputeEES:
    def test_basic_formula(self):
        iou, fps, mem, budget = 0.5, 30.0, 256.0, 512.0
        expected = (iou * math.log1p(fps)) / (1.0 + mem / budget)
        assert _compute_ees(iou, fps, mem, budget) == pytest.approx(expected, rel=1e-9)

    def test_zero_fps_returns_zero(self):
        assert _compute_ees(0.8, 0.0, 64.0) == 0.0

    def test_negative_fps_returns_zero(self):
        assert _compute_ees(0.8, -5.0, 64.0) == 0.0

    def test_negative_iou_returns_zero(self):
        assert _compute_ees(-0.1, 30.0, 64.0) == 0.0

    def test_zero_iou_yields_zero(self):
        assert _compute_ees(0.0, 30.0, 64.0) == 0.0

    def test_custom_budget(self):
        result = _compute_ees(1.0, math.e - 1, 0.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_high_memory_penalises(self):
        low_mem  = _compute_ees(0.7, 60.0, 64.0,  512.0)
        high_mem = _compute_ees(0.7, 60.0, 512.0, 512.0)
        assert high_mem < low_mem


# ---------------------------------------------------------------------------
# Constructor / validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_columns(self):
        gen = LaTeXReportGenerator()
        assert gen.columns == list(DEFAULT_COLUMNS)

    def test_custom_columns_accepted(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou", "fps"])
        assert gen.columns == ["tracker", "miou", "fps"]

    def test_unknown_column_raises(self):
        with pytest.raises(ValueError, match="Unknown column"):
            LaTeXReportGenerator(columns=["tracker", "nonexistent"])

    def test_bold_best_default_true(self):
        gen = LaTeXReportGenerator()
        assert gen.bold_best is True

    def test_group_by_dataset_default_true(self):
        gen = LaTeXReportGenerator()
        assert gen.group_by_dataset is True

    def test_memory_budget_stored(self):
        gen = LaTeXReportGenerator(memory_budget_mb=256.0)
        assert gen.memory_budget_mb == 256.0


# ---------------------------------------------------------------------------
# generate_table — structure
# ---------------------------------------------------------------------------

class TestGenerateTableStructure:
    def test_empty_results_returns_comment(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table([])
        assert "% No results" in out

    def test_contains_table_environment(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table(SINGLE)
        assert r"\begin{table}" in out
        assert r"\end{table}" in out

    def test_contains_tabular_environment(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table(SINGLE)
        assert r"\begin{tabular}" in out
        assert r"\end{tabular}" in out

    def test_booktabs_rules(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table(SINGLE)
        assert r"\toprule" in out
        assert r"\midrule" in out
        assert r"\bottomrule" in out

    def test_caption_in_output(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table(SINGLE, caption="My Caption")
        assert "My Caption" in out

    def test_label_in_output(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table(SINGLE, label="tab:my-label")
        assert "tab:my-label" in out

    def test_custom_position(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_table(SINGLE, position="t")
        assert r"\begin{table}[t]" in out

    def test_header_cells_bolded(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.generate_table(SINGLE)
        assert r"\textbf{Tracker}" in out
        assert r"\textbf{mIoU}" in out

    def test_tracker_name_present(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.generate_table(SINGLE)
        assert "MOSSE" in out

    def test_alignment_string_correct(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou", "fps"])
        out = gen.generate_table(SINGLE)
        assert "{lrr}" in out

    def test_row_ends_with_backslash(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.generate_table(SINGLE)
        data_rows = [l for l in out.splitlines() if "MOSSE" in l]
        assert data_rows
        assert data_rows[0].strip().endswith(r"\\")

    def test_none_value_rendered_as_dash(self):
        result = _make_result()
        result["summary"]["mean_iou"] = None
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.generate_table([result])
        assert "--" in out


# ---------------------------------------------------------------------------
# generate_table — bold-best logic
# ---------------------------------------------------------------------------

class TestBoldBest:
    def test_best_miou_is_bolded(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"], bold_best=True)
        out = gen.generate_table(MULTI)
        assert r"\mathbf{" in out

    def test_bold_best_false_no_mathbf(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"], bold_best=False)
        out = gen.generate_table(MULTI)
        assert r"\mathbf{" not in out

    def test_best_miou_row_is_csrt(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"], bold_best=True)
        out = gen.generate_table(MULTI)
        lines = out.splitlines()
        csrt_line = next(l for l in lines if "CSRT" in l)
        assert r"\mathbf{" in csrt_line

    def test_best_memory_is_lowest(self):
        gen = LaTeXReportGenerator(columns=["tracker", "memory"], bold_best=True)
        out = gen.generate_table(MULTI)
        lines = out.splitlines()
        mosse_line = next(l for l in lines if "MOSSE" in l)
        assert r"\mathbf{" in mosse_line

    def test_text_columns_never_bolded(self):
        gen = LaTeXReportGenerator(columns=["tracker"], bold_best=True)
        out = gen.generate_table(MULTI)
        assert r"\mathbf{" not in out

    def test_single_result_best_is_bolded(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"], bold_best=True)
        out = gen.generate_table(SINGLE)
        assert r"\mathbf{" in out


# ---------------------------------------------------------------------------
# generate_table — dataset grouping
# ---------------------------------------------------------------------------

class TestDatasetGrouping:
    def test_midrule_between_datasets(self):
        gen = LaTeXReportGenerator(
            columns=["tracker", "dataset", "miou"],
            group_by_dataset=True,
        )
        out = gen.generate_table(MULTI_DATASET)
        midrule_count = out.count(r"\midrule")
        assert midrule_count >= 2

    def test_no_extra_midrule_same_dataset(self):
        gen = LaTeXReportGenerator(
            columns=["tracker", "miou"],
            group_by_dataset=True,
        )
        out = gen.generate_table(MULTI)
        midrule_count = out.count(r"\midrule")
        assert midrule_count == 1

    def test_group_by_dataset_false_no_extra_midrule(self):
        gen = LaTeXReportGenerator(
            columns=["tracker", "dataset", "miou"],
            group_by_dataset=False,
        )
        out = gen.generate_table(MULTI_DATASET)
        midrule_count = out.count(r"\midrule")
        assert midrule_count == 1


# ---------------------------------------------------------------------------
# EES column
# ---------------------------------------------------------------------------

class TestEESColumn:
    def test_ees_computed_and_present(self):
        gen = LaTeXReportGenerator(columns=["tracker", "ees"])
        out = gen.generate_table(SINGLE)
        assert "EES" in out
        assert "--" not in out

    def test_ees_none_when_fps_missing(self):
        result = _make_result()
        result["summary"]["mean_fps"] = None
        gen = LaTeXReportGenerator(columns=["tracker", "ees"])
        out = gen.generate_table([result])
        assert "--" in out

    def test_ees_uses_memory_budget(self):
        gen_small = LaTeXReportGenerator(columns=["tracker", "ees"], memory_budget_mb=64.0)
        gen_large = LaTeXReportGenerator(columns=["tracker", "ees"], memory_budget_mb=2048.0)
        out_small = gen_small.generate_table(SINGLE)
        out_large = gen_large.generate_table(SINGLE)
        assert out_small != out_large


# ---------------------------------------------------------------------------
# Numeric formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_miou_4_decimal_places(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.generate_table(SINGLE)
        assert "0.6500" in out

    def test_fps_1_decimal_place(self):
        gen = LaTeXReportGenerator(columns=["tracker", "fps"])
        out = gen.generate_table(SINGLE)
        assert "120.0" in out

    def test_memory_1_decimal_place(self):
        gen = LaTeXReportGenerator(columns=["tracker", "memory"])
        out = gen.generate_table(SINGLE)
        assert "64.0" in out

    def test_latency_2_decimal_places(self):
        gen = LaTeXReportGenerator(columns=["tracker", "latency"])
        out = gen.generate_table(SINGLE)
        assert "8.30" in out

    def test_energy_3_decimal_places(self):
        gen = LaTeXReportGenerator(columns=["tracker", "energy"])
        out = gen.generate_table(SINGLE)
        assert "0.012" in out


# ---------------------------------------------------------------------------
# generate_preamble_comment
# ---------------------------------------------------------------------------

class TestPreambleComment:
    def test_contains_booktabs(self):
        gen = LaTeXReportGenerator()
        comment = gen.generate_preamble_comment()
        assert "booktabs" in comment

    def test_contains_amsmath(self):
        gen = LaTeXReportGenerator()
        comment = gen.generate_preamble_comment()
        assert "amsmath" in comment

    def test_lines_are_comments(self):
        gen = LaTeXReportGenerator()
        for line in gen.generate_preamble_comment().splitlines():
            assert line.startswith("%")


# ---------------------------------------------------------------------------
# generate_full
# ---------------------------------------------------------------------------

class TestGenerateFull:
    def test_includes_preamble_and_table(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_full(SINGLE)
        assert "booktabs" in out
        assert r"\begin{table}" in out

    def test_preamble_before_table(self):
        gen = LaTeXReportGenerator()
        out = gen.generate_full(SINGLE)
        assert out.index("booktabs") < out.index(r"\begin{table}")


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class TestSave:
    def test_creates_file(self, tmp_path):
        gen = LaTeXReportGenerator()
        p = gen.save(SINGLE, str(tmp_path / "out.tex"))
        assert p.exists()

    def test_appends_tex_suffix(self, tmp_path):
        gen = LaTeXReportGenerator()
        p = gen.save(SINGLE, str(tmp_path / "out"))
        assert p.suffix == ".tex"

    def test_creates_parent_directories(self, tmp_path):
        gen = LaTeXReportGenerator()
        p = gen.save(SINGLE, str(tmp_path / "deep" / "dir" / "out.tex"))
        assert p.exists()

    def test_content_readable_back(self, tmp_path):
        gen = LaTeXReportGenerator()
        p = gen.save(SINGLE, str(tmp_path / "out.tex"))
        content = p.read_text(encoding="utf-8")
        assert r"\begin{table}" in content

    def test_include_preamble_false(self, tmp_path):
        gen = LaTeXReportGenerator()
        p = gen.save(SINGLE, str(tmp_path / "out.tex"), include_preamble=False)
        content = p.read_text(encoding="utf-8")
        assert "booktabs" not in content
        assert r"\begin{table}" in content

    def test_returns_path_object(self, tmp_path):
        gen = LaTeXReportGenerator()
        result = gen.save(SINGLE, str(tmp_path / "out.tex"))
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# to_markdown_preview
# ---------------------------------------------------------------------------

class TestMarkdownPreview:
    def test_empty_returns_no_results(self):
        gen = LaTeXReportGenerator()
        assert "No results" in gen.to_markdown_preview([])

    def test_contains_pipe_characters(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(SINGLE)
        assert "|" in out

    def test_header_row_present(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(SINGLE)
        lines = out.splitlines()
        assert "Tracker" in lines[0]
        assert "mIoU" in lines[0]

    def test_separator_row_present(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(SINGLE)
        lines = out.splitlines()
        assert "---" in lines[1]

    def test_tracker_name_in_body(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(SINGLE)
        assert "MOSSE" in out

    def test_right_aligned_separator_for_numeric(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(SINGLE)
        lines = out.splitlines()
        assert "---:" in lines[1]

    def test_left_aligned_separator_for_tracker(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(SINGLE)
        lines = out.splitlines()
        sep_cells = [c.strip() for c in lines[1].split("|") if c.strip()]
        assert sep_cells[0] == "---"

    def test_multiple_rows(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        out = gen.to_markdown_preview(MULTI)
        lines = [l for l in out.splitlines() if l.startswith("|") and "---" not in l and "mIoU" not in l]
        assert len(lines) == 3

    def test_none_value_is_dash(self):
        result = _make_result()
        result["summary"]["mean_fps"] = None
        gen = LaTeXReportGenerator(columns=["tracker", "fps"])
        out = gen.to_markdown_preview([result])
        assert "--" in out


# ---------------------------------------------------------------------------
# _extract_rows — fallback key resolution
# ---------------------------------------------------------------------------

class TestExtractRows:
    def test_tracker_name_key(self):
        result = {"summary": {"tracker_name": "CSRT", "dataset": "X"}}
        gen = LaTeXReportGenerator(columns=["tracker"])
        rows = gen._extract_rows([result])
        assert rows[0]["tracker"] == "CSRT"

    def test_dataset_name_key(self):
        result = {"summary": {"tracker": "X", "dataset_name": "LaSOT"}}
        gen = LaTeXReportGenerator(columns=["dataset"])
        rows = gen._extract_rows([result])
        assert rows[0]["dataset"] == "LaSOT"

    def test_missing_tracker_defaults_to_question_mark(self):
        result = {"summary": {"dataset": "X"}}
        gen = LaTeXReportGenerator(columns=["tracker"])
        rows = gen._extract_rows([result])
        assert rows[0]["tracker"] == "?"

    def test_ees_computed_when_requested(self):
        gen = LaTeXReportGenerator(columns=["tracker", "ees"])
        rows = gen._extract_rows(SINGLE)
        assert rows[0]["_ees"] is not None
        assert rows[0]["_ees"] > 0

    def test_ees_not_computed_when_not_requested(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        rows = gen._extract_rows(SINGLE)
        assert "_ees" not in rows[0]


# ---------------------------------------------------------------------------
# _find_best
# ---------------------------------------------------------------------------

class TestFindBest:
    def _col_specs(self, keys):
        return [_BUILTIN_COLUMNS[k] for k in keys]

    def test_higher_is_better_picks_max(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        rows = gen._extract_rows(MULTI)
        best = gen._find_best(rows, self._col_specs(["tracker", "miou"]))
        assert best["miou"] == 2  # CSRT has 0.80

    def test_lower_is_better_picks_min(self):
        gen = LaTeXReportGenerator(columns=["tracker", "memory"])
        rows = gen._extract_rows(MULTI)
        best = gen._find_best(rows, self._col_specs(["tracker", "memory"]))
        assert best["memory"] == 0  # MOSSE has 64 MB

    def test_text_columns_excluded(self):
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        rows = gen._extract_rows(MULTI)
        best = gen._find_best(rows, self._col_specs(["tracker", "miou"]))
        assert "tracker" not in best

    def test_all_none_column_excluded(self):
        results = [_make_result()]
        results[0]["summary"]["mean_iou"] = None
        gen = LaTeXReportGenerator(columns=["tracker", "miou"])
        rows = gen._extract_rows(results)
        best = gen._find_best(rows, self._col_specs(["tracker", "miou"]))
        assert "miou" not in best
