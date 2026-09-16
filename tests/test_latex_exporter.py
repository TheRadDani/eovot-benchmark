"""Tests for eovot.reporting.latex_exporter."""
from unittest.mock import MagicMock

import pytest

from eovot.reporting.latex_exporter import (
    DEFAULT_METRICS,
    LATEX_PREAMBLE_HINT,
    LatexTableExporter,
)


def _make_result(
    tracker,
    dataset,
    mean_iou=0.5,
    success_auc=0.45,
    precision_auc=0.6,
    mean_fps=30.0,
    peak_memory_mb=100.0,
    include_auc=True,
):
    """Create a mock BenchmarkResult-like object."""
    summary_dict = {
        "tracker": tracker,
        "dataset": dataset,
        "num_sequences": 10,
        "mean_iou": mean_iou,
        "mean_fps": mean_fps,
        "peak_memory_mb": peak_memory_mb,
    }
    if include_auc:
        summary_dict["success_auc"] = success_auc
        summary_dict["precision_auc"] = precision_auc
    mock = MagicMock()
    mock.summary.return_value = summary_dict
    return mock


class TestLatexTableExporter:
    def setup_method(self):
        self.exporter = LatexTableExporter(precision=3)

    def test_single_dataset_contains_booktabs(self):
        results = [_make_result("MOSSE", "OTB100"), _make_result("KCF", "OTB100")]
        tex = self.exporter.single_dataset_table(results)
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "\\bottomrule" in tex
        assert "\\begin{tabular}" in tex

    def test_single_dataset_tracker_names_present(self):
        results = [_make_result("MOSSE", "OTB100"), _make_result("KCF", "OTB100")]
        tex = self.exporter.single_dataset_table(results)
        assert "MOSSE" in tex
        assert "KCF" in tex

    def test_caption_and_label_injected(self):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(
            results, caption="My caption", label="my-label"
        )
        assert "\\caption{My caption}" in tex
        assert "\\label{tab:my-label}" in tex

    def test_label_prefix_not_doubled(self):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(results, label="tab:already")
        assert "\\label{tab:already}" in tex
        assert "tab:tab:" not in tex

    def test_best_value_bolded(self):
        r1 = _make_result("MOSSE", "OTB100", success_auc=0.40)
        r2 = _make_result("KCF",   "OTB100", success_auc=0.55)
        tex = self.exporter.single_dataset_table([r1, r2], metrics=["success_auc"])
        assert "\\textbf{0.550}" in tex

    def test_second_best_underlined(self):
        r1 = _make_result("MOSSE", "OTB100", success_auc=0.40)
        r2 = _make_result("KCF",   "OTB100", success_auc=0.55)
        r3 = _make_result("CSRT",  "OTB100", success_auc=0.70)
        tex = self.exporter.single_dataset_table([r1, r2, r3], metrics=["success_auc"])
        assert "\\textbf{0.700}" in tex
        assert "\\underline{0.550}" in tex

    def test_no_highlight_when_disabled(self):
        exporter = LatexTableExporter(highlight_best=False)
        r1 = _make_result("MOSSE", "OTB100", success_auc=0.40)
        r2 = _make_result("KCF",   "OTB100", success_auc=0.55)
        tex = exporter.single_dataset_table([r1, r2], metrics=["success_auc"])
        assert "\\textbf" not in tex
        assert "\\underline" not in tex

    def test_missing_metric_shows_dash(self):
        result = _make_result("MOSSE", "OTB100", include_auc=False)
        tex = self.exporter.single_dataset_table([result], metrics=["success_auc"])
        assert "--" in tex

    def test_fps_arrow_in_header(self):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(results, metrics=["mean_fps"])
        assert r"\uparrow" in tex

    def test_memory_arrow_in_header(self):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(results, metrics=["peak_memory_mb"])
        assert r"\downarrow" in tex

    def test_font_size_injected(self):
        results = [_make_result("KCF", "OTB100")]
        tex = self.exporter.single_dataset_table(results, font_size="\\small")
        assert "\\small" in tex

    def test_underscore_in_tracker_name_escaped(self):
        results = [_make_result("SiamRPN_v2", "GOT10k")]
        tex = self.exporter.single_dataset_table(results)
        assert r"SiamRPN\_v2" in tex

    def test_custom_table_environment(self):
        results = [_make_result("KCF", "OTB100")]
        tex = self.exporter.single_dataset_table(results, table_env="table*")
        assert "\\begin{table*}" in tex
        assert "\\end{table*}" in tex

    def test_multi_dataset_midrule_between_groups(self):
        r_otb = [_make_result("KCF", "OTB100")]
        r_got = [_make_result("KCF", "GOT10k")]
        tex = self.exporter.multi_dataset_table({"OTB100": r_otb, "GOT10k": r_got})
        assert tex.count("\\midrule") >= 2

    def test_multi_dataset_dataset_names_present(self):
        tex = self.exporter.multi_dataset_table({
            "OTB100": [_make_result("KCF", "OTB100")],
            "GOT10k": [_make_result("KCF", "GOT10k")],
        })
        assert "OTB100" in tex
        assert "GOT10k" in tex

    def test_multi_dataset_multirow_present(self):
        tex = self.exporter.multi_dataset_table({
            "OTB100": [
                _make_result("KCF",   "OTB100"),
                _make_result("MOSSE", "OTB100"),
            ],
        })
        assert "\\multirow" in tex

    def test_save_writes_file(self, tmp_path):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(results)
        out = tmp_path / "table.tex"
        path = self.exporter.save(tex, str(out))
        assert path.exists()
        content = path.read_text()
        assert "\\begin{table}" in content
        assert "booktabs" in content

    def test_save_adds_tex_extension(self, tmp_path):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(results)
        path = self.exporter.save(tex, str(tmp_path / "table"))
        assert path.suffix == ".tex"

    def test_save_no_preamble_hint(self, tmp_path):
        results = [_make_result("MOSSE", "OTB100")]
        tex = self.exporter.single_dataset_table(results)
        path = self.exporter.save(tex, str(tmp_path / "t.tex"), include_preamble_hint=False)
        assert "booktabs" not in path.read_text()

    def test_col_spec_correct_length(self):
        results = [_make_result("KCF", "OTB100")]
        n_metrics = 3
        custom = ["success_auc", "mean_fps", "peak_memory_mb"]
        tex = self.exporter.single_dataset_table(results, metrics=custom)
        expected_spec = "{l" + "c" * n_metrics + "}"
        assert expected_spec in tex

    def test_lower_fps_preferred_not_bolded(self):
        # Memory: lower is better → the tracker with less memory should be bold
        r1 = _make_result("MOSSE", "OTB100", peak_memory_mb=50.0)
        r2 = _make_result("KCF",   "OTB100", peak_memory_mb=200.0)
        tex = self.exporter.single_dataset_table([r1, r2], metrics=["peak_memory_mb"])
        assert "\\textbf{50.000}" in tex

    def test_single_tracker_only_best_highlighted(self):
        result = _make_result("MOSSE", "OTB100", success_auc=0.42)
        tex = self.exporter.single_dataset_table([result], metrics=["success_auc"])
        assert "\\textbf{0.420}" in tex
        assert "\\underline" not in tex
