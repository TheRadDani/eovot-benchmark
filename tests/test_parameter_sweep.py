"""Tests for eovot.analysis.parameter_sweep.

Covers:
- SweepEntry / SweepResult dataclass API
- _mark_pareto correctness on known configurations
- ParameterSweep.run() with a real SyntheticDataset (integration smoke-test)
- ParameterSweep.sensitivity_1d() convenience wrapper
- SweepResult.pareto_configs() filtering by budget_fps
- SweepResult.sensitivity() aggregation
- Markdown / dict serialisation round-trips
"""

from __future__ import annotations

import math
import pytest

from eovot.analysis.parameter_sweep import (
    ParameterSweep,
    SweepEntry,
    SweepResult,
    _mark_pareto,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(mean_iou: float, mean_fps: float, **kwargs) -> SweepEntry:
    defaults = dict(
        params={"p": 1},
        tracker_name="T",
        dataset_name="D",
        mean_iou=mean_iou,
        success_auc=mean_iou,
        precision_auc=0.5,
        mean_fps=mean_fps,
        mean_latency_ms=1000.0 / max(mean_fps, 1e-9),
        peak_memory_mb=50.0,
        num_sequences=5,
    )
    defaults.update(kwargs)
    return SweepEntry(**defaults)


# ---------------------------------------------------------------------------
# SweepEntry
# ---------------------------------------------------------------------------

class TestSweepEntry:
    def test_params_label_sorted(self):
        e = _entry(0.5, 30.0, params={"lr": 0.1, "sigma": 2.0})
        assert e.params_label() == "lr=0.1, sigma=2.0"

    def test_params_label_single(self):
        e = _entry(0.6, 25.0, params={"learning_rate": 0.25})
        assert e.params_label() == "learning_rate=0.25"

    def test_str_includes_pareto(self):
        e = _entry(0.7, 40.0)
        e.on_pareto_front = True
        assert "[Pareto]" in str(e)

    def test_str_no_pareto_when_dominated(self):
        e = _entry(0.3, 10.0)
        e.on_pareto_front = False
        assert "[Pareto]" not in str(e)

    def test_str_contains_key_metrics(self):
        e = _entry(0.55, 28.3, mean_latency_ms=35.3, peak_memory_mb=62.1)
        s = str(e)
        assert "mIoU=0.5500" in s
        assert "FPS=28.3" in s
        assert "lat=35.30ms" in s
        assert "mem=62.1MB" in s

    def test_default_on_pareto_front_false(self):
        e = _entry(0.4, 20.0)
        assert e.on_pareto_front is False

    def test_wall_time_defaults_to_zero(self):
        e = _entry(0.4, 20.0)
        assert e.wall_time_s == 0.0


# ---------------------------------------------------------------------------
# _mark_pareto
# ---------------------------------------------------------------------------

class TestMarkPareto:
    def test_single_entry_is_pareto(self):
        entries = [_entry(0.5, 30.0)]
        _mark_pareto(entries)
        assert entries[0].on_pareto_front is True

    def test_dominated_entry_not_pareto(self):
        # B strictly dominates A in both objectives
        a = _entry(0.4, 20.0, params={"p": 0})
        b = _entry(0.6, 30.0, params={"p": 1})
        _mark_pareto([a, b])
        assert a.on_pareto_front is False
        assert b.on_pareto_front is True

    def test_tradeoff_both_pareto(self):
        # A is faster; B is more accurate — neither dominates
        a = _entry(0.4, 60.0, params={"p": 0})
        b = _entry(0.7, 20.0, params={"p": 1})
        _mark_pareto([a, b])
        assert a.on_pareto_front is True
        assert b.on_pareto_front is True

    def test_all_equal_all_pareto(self):
        entries = [_entry(0.5, 30.0, params={"p": i}) for i in range(3)]
        _mark_pareto(entries)
        # When values are identical, no strict inequality exists → all are non-dominated
        assert all(e.on_pareto_front for e in entries)

    def test_chain_dominated(self):
        # A < B < C in both objectives — only C is on front
        a = _entry(0.3, 10.0, params={"p": 0})
        b = _entry(0.5, 20.0, params={"p": 1})
        c = _entry(0.7, 30.0, params={"p": 2})
        _mark_pareto([a, b, c])
        assert a.on_pareto_front is False
        assert b.on_pareto_front is False
        assert c.on_pareto_front is True

    def test_mixed_set(self):
        # Three-entry set: two on front, one dominated
        a = _entry(0.8, 15.0, params={"p": 0})  # high accuracy, low speed
        b = _entry(0.5, 50.0, params={"p": 1})  # low accuracy, high speed
        c = _entry(0.6, 20.0, params={"p": 2})  # dominated by a AND b for respective objectives
        _mark_pareto([a, b, c])
        assert a.on_pareto_front is True
        assert b.on_pareto_front is True
        # c: a has higher iou AND a or b can be checked; b has higher fps AND b has lower iou than a
        # c: iou=0.6, fps=20 -> a dominates? a.iou=0.8>=0.6 AND a.fps=15<20 -> no
        #    b dominates? b.iou=0.5<0.6 -> no; so c is NOT dominated
        assert c.on_pareto_front is True

    def test_pareto_reset_on_second_call(self):
        entries = [_entry(0.5, 30.0, params={"p": 0}), _entry(0.9, 50.0, params={"p": 1})]
        _mark_pareto(entries)
        assert entries[0].on_pareto_front is False
        # Calling again should be idempotent
        _mark_pareto(entries)
        assert entries[0].on_pareto_front is False
        assert entries[1].on_pareto_front is True

    def test_empty_list_no_error(self):
        _mark_pareto([])  # Should not raise


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------

class TestSweepResult:
    def _make_result(self) -> SweepResult:
        entries = [
            _entry(0.8, 15.0, params={"lr": 0.05}),
            _entry(0.5, 60.0, params={"lr": 0.5}),
            _entry(0.65, 30.0, params={"lr": 0.25}),
        ]
        _mark_pareto(entries)
        return SweepResult(
            tracker_name="MOSSE",
            dataset_name="Synthetic",
            param_names=["lr"],
            entries=entries,
        )

    def test_best_by_accuracy(self):
        r = self._make_result()
        best = r.best_by_accuracy()
        assert best is not None
        assert abs(best.mean_iou - 0.8) < 1e-9

    def test_best_by_fps(self):
        r = self._make_result()
        best = r.best_by_fps()
        assert best is not None
        assert abs(best.mean_fps - 60.0) < 1e-9

    def test_best_by_accuracy_empty(self):
        r = SweepResult("T", "D", [])
        assert r.best_by_accuracy() is None

    def test_best_by_fps_empty(self):
        r = SweepResult("T", "D", [])
        assert r.best_by_fps() is None

    def test_pareto_configs_no_filter(self):
        r = self._make_result()
        pareto = r.pareto_configs()
        assert all(e.on_pareto_front for e in pareto)

    def test_pareto_configs_budget_fps(self):
        r = self._make_result()
        # Only entries with fps >= 30 should be considered
        pareto = r.pareto_configs(budget_fps=30.0)
        assert all(e.mean_fps >= 30.0 for e in pareto)

    def test_pareto_configs_budget_fps_too_high(self):
        r = self._make_result()
        # No entry has fps >= 1000
        pareto = r.pareto_configs(budget_fps=1000.0)
        assert pareto == []

    def test_pareto_configs_sorted_by_iou(self):
        r = self._make_result()
        pareto = r.pareto_configs()
        ious = [e.mean_iou for e in pareto]
        assert ious == sorted(ious, reverse=True)

    def test_sensitivity_aggregation(self):
        # Two entries with same lr=0.1 — sensitivity should average them
        e1 = _entry(0.4, 20.0, params={"lr": 0.1, "sigma": 1.0})
        e2 = _entry(0.6, 40.0, params={"lr": 0.1, "sigma": 2.0})
        r = SweepResult("T", "D", ["lr", "sigma"], entries=[e1, e2])
        rows = r.sensitivity("lr")
        assert len(rows) == 1
        val, mean_iou, mean_fps = rows[0]
        assert val == 0.1
        assert abs(mean_iou - 0.5) < 1e-9
        assert abs(mean_fps - 30.0) < 1e-9

    def test_sensitivity_multiple_values_sorted(self):
        entries = [
            _entry(0.7, 30.0, params={"lr": 0.5}),
            _entry(0.4, 60.0, params={"lr": 0.1}),
            _entry(0.55, 45.0, params={"lr": 0.25}),
        ]
        r = SweepResult("T", "D", ["lr"], entries=entries)
        rows = r.sensitivity("lr")
        vals = [row[0] for row in rows]
        assert vals == sorted(vals)

    def test_sensitivity_unknown_param(self):
        r = self._make_result()
        rows = r.sensitivity("nonexistent_param")
        # All entries have None for this param — one row with val=None
        assert len(rows) == 1
        assert rows[0][0] is None


# ---------------------------------------------------------------------------
# Markdown / serialisation
# ---------------------------------------------------------------------------

class TestSweepResultMarkdown:
    def _result(self) -> SweepResult:
        entries = [
            _entry(0.7, 30.0, params={"lr": 0.25}),
            _entry(0.5, 55.0, params={"lr": 0.5}),
        ]
        _mark_pareto(entries)
        return SweepResult("MOSSE", "Synthetic", ["lr"], entries=entries)

    def test_to_markdown_has_header(self):
        md = self._result().to_markdown()
        assert "Parameter Sweep" in md
        assert "MOSSE" in md
        assert "Synthetic" in md

    def test_to_markdown_table_rows(self):
        md = self._result().to_markdown()
        assert "lr=0.25" in md
        assert "lr=0.5" in md

    def test_to_markdown_rank_column(self):
        md = self._result().to_markdown()
        assert "| 1 |" in md
        assert "| 2 |" in md

    def test_to_markdown_empty(self):
        r = SweepResult("T", "D", [])
        assert "no results" in r.to_markdown()

    def test_to_sensitivity_markdown_header(self):
        r = self._result()
        md = r.to_sensitivity_markdown("lr")
        assert "Sensitivity" in md
        assert "lr" in md

    def test_to_sensitivity_markdown_values(self):
        r = self._result()
        md = r.to_sensitivity_markdown("lr")
        assert "0.25" in md
        assert "0.5" in md

    def test_to_sensitivity_markdown_unknown_param(self):
        r = self._result()
        md = r.to_sensitivity_markdown("nonexistent")
        assert "no data" in md.lower() or "nonexistent" in md

    def test_to_dict_keys(self):
        d = self._result().to_dict()
        assert "tracker_name" in d
        assert "dataset_name" in d
        assert "param_names" in d
        assert "entries" in d

    def test_to_dict_entries_structure(self):
        d = self._result().to_dict()
        entry = d["entries"][0]
        for key in ("params", "mean_iou", "success_auc", "precision_auc",
                    "mean_fps", "mean_latency_ms", "peak_memory_mb",
                    "num_sequences", "wall_time_s", "on_pareto_front"):
            assert key in entry, f"Missing key: {key}"

    def test_to_dict_rounded_values(self):
        r = self._result()
        d = r.to_dict()
        # mean_iou should be rounded to 6 decimal places
        iou = d["entries"][0]["mean_iou"]
        assert isinstance(iou, float)
        assert len(str(iou).split(".")[-1]) <= 7  # at most 6 decimal digits


# ---------------------------------------------------------------------------
# ParameterSweep integration (uses SyntheticDataset + BenchmarkEngine)
# ---------------------------------------------------------------------------

class TestParameterSweepIntegration:
    """Integration tests that actually invoke the benchmark engine."""

    @pytest.fixture(scope="class")
    def dataset(self):
        from eovot.datasets.synthetic import SyntheticDataset
        return SyntheticDataset(num_sequences=2, num_frames=15, seed=99)

    def test_run_single_param_mosse(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.1, 0.3]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert isinstance(result, SweepResult)
        assert len(result.entries) == 2

    def test_run_returns_sorted_by_iou(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.05, 0.2, 0.4]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        ious = [e.mean_iou for e in result.entries]
        assert ious == sorted(ious, reverse=True)

    def test_run_pareto_flags_set(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.05, 0.25, 0.5]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        # At least one entry should be on the Pareto front
        assert any(e.on_pareto_front for e in result.entries)

    def test_run_entries_have_positive_fps(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.125]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert result.entries[0].mean_fps > 0.0

    def test_run_entries_have_positive_memory(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.125]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert result.entries[0].peak_memory_mb > 0.0

    def test_run_entries_have_num_sequences(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.25]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert result.entries[0].num_sequences == 2

    def test_run_cartesian_product_count(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.1, 0.3], "sigma": [1.0, 2.0]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert len(result.entries) == 4  # 2 × 2

    def test_run_wall_time_positive(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.25]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert result.entries[0].wall_time_s > 0.0

    def test_run_empty_param_grid_raises(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        with pytest.raises(ValueError, match="param_grid"):
            sweep.run(param_grid={}, dataset_name="Synthetic")

    def test_run_tracker_name_propagated(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.25]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert result.tracker_name == "MOSSE"
        assert result.entries[0].tracker_name == "MOSSE"

    def test_run_dataset_name_propagated(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.25]},
            dataset_name="MyDataset",
            max_sequences=2,
        )
        assert result.dataset_name == "MyDataset"
        assert result.entries[0].dataset_name == "MyDataset"

    def test_sensitivity_1d_wrapper(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.sensitivity_1d(
            "learning_rate",
            values=[0.1, 0.3],
            dataset_name="Synthetic",
            max_sequences=2,
        )
        assert isinstance(result, SweepResult)
        assert len(result.entries) == 2

    def test_sensitivity_1d_with_fixed_params(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.sensitivity_1d(
            "learning_rate",
            values=[0.1, 0.25],
            fixed_params={"sigma": 2.0},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        # Both entries should have sigma=2.0 in their params
        for e in result.entries:
            assert e.params.get("sigma") == 2.0

    def test_markdown_output_from_real_sweep(self, dataset):
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.1, 0.3]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        md = result.to_markdown()
        assert "MOSSE" in md
        assert "learning_rate" in md

    def test_dict_serialisable(self, dataset):
        import json
        sweep = ParameterSweep("MOSSE", dataset=dataset, verbose=False)
        result = sweep.run(
            param_grid={"learning_rate": [0.1]},
            dataset_name="Synthetic",
            max_sequences=2,
        )
        d = result.to_dict()
        json_str = json.dumps(d)  # must not raise
        assert len(json_str) > 0
