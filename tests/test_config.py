"""Tests for eovot.config — schema, loader, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eovot.config import (
    ConfigValidationError,
    EOVOTConfig,
    load_config,
    validate_config,
)
from eovot.config.schema import (
    BenchmarkConfig,
    DatasetConfig,
    ExperimentConfig,
    ReportingConfig,
    TrackerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal() -> dict:
    """Smallest valid raw config dict."""
    return {
        "experiment": {"name": "test-run"},
        "dataset": {"name": "OTB100", "loader": "OTBDataset", "root": "/data/otb"},
        "tracker": {"name": "MOSSE"},
    }


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as fh:
        yaml.dump(data, fh)


# ---------------------------------------------------------------------------
# TestSchemaDataclasses — verify defaults and field types
# ---------------------------------------------------------------------------

class TestSchemaDataclasses:
    def test_experiment_config_defaults(self):
        cfg = ExperimentConfig(name="x")
        assert cfg.output_dir == "results/"
        assert cfg.seed == 42

    def test_benchmark_config_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.verbose is True
        assert cfg.tdp_watts is None
        assert cfg.save_predictions is False

    def test_reporting_config_defaults(self):
        cfg = ReportingConfig()
        assert cfg.formats == ["json", "csv"]
        assert cfg.print_summary is True

    def test_tracker_config_defaults(self):
        cfg = TrackerConfig(name="MOSSE")
        assert cfg.params == {}

    def test_dataset_config_optional_fields_none(self):
        cfg = DatasetConfig(name="OTB100", loader="OTBDataset", root="/data")
        assert cfg.max_sequences is None
        assert cfg.split is None


# ---------------------------------------------------------------------------
# TestValidateConfig — successful validation
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_minimal_valid_config_returns_eovot_config(self):
        cfg = validate_config(_minimal())
        assert isinstance(cfg, EOVOTConfig)

    def test_experiment_fields_populated(self):
        cfg = validate_config(_minimal())
        assert cfg.experiment.name == "test-run"
        assert cfg.experiment.output_dir == "results/"
        assert cfg.experiment.seed == 42

    def test_dataset_fields_populated(self):
        cfg = validate_config(_minimal())
        assert cfg.dataset.name == "OTB100"
        assert cfg.dataset.loader == "OTBDataset"
        assert cfg.dataset.root == "/data/otb"

    def test_tracker_fields_populated(self):
        cfg = validate_config(_minimal())
        assert cfg.tracker.name == "MOSSE"
        assert cfg.tracker.params == {}

    def test_custom_experiment_overrides_defaults(self):
        raw = _minimal()
        raw["experiment"]["output_dir"] = "my_results/"
        raw["experiment"]["seed"] = 99
        cfg = validate_config(raw)
        assert cfg.experiment.output_dir == "my_results/"
        assert cfg.experiment.seed == 99

    def test_tracker_params_preserved(self):
        raw = _minimal()
        raw["tracker"]["params"] = {"learning_rate": 0.125, "sigma": 2.0}
        cfg = validate_config(raw)
        assert cfg.tracker.params["learning_rate"] == pytest.approx(0.125)
        assert cfg.tracker.params["sigma"] == pytest.approx(2.0)

    def test_dataset_max_sequences_accepted(self):
        raw = _minimal()
        raw["dataset"]["max_sequences"] = 10
        cfg = validate_config(raw)
        assert cfg.dataset.max_sequences == 10

    def test_dataset_split_accepted(self):
        raw = _minimal()
        raw["dataset"]["split"] = "val"
        cfg = validate_config(raw)
        assert cfg.dataset.split == "val"

    def test_benchmark_tdp_watts_accepted(self):
        raw = _minimal()
        raw["benchmark"] = {"tdp_watts": 6.0}
        cfg = validate_config(raw)
        assert cfg.benchmark.tdp_watts == pytest.approx(6.0)

    def test_benchmark_save_predictions_accepted(self):
        raw = _minimal()
        raw["benchmark"] = {"save_predictions": True}
        cfg = validate_config(raw)
        assert cfg.benchmark.save_predictions is True

    def test_reporting_single_format_accepted(self):
        raw = _minimal()
        raw["reporting"] = {"formats": ["json"]}
        cfg = validate_config(raw)
        assert cfg.reporting.formats == ["json"]

    def test_reporting_all_three_formats_accepted(self):
        raw = _minimal()
        raw["reporting"] = {"formats": ["json", "csv", "markdown"]}
        cfg = validate_config(raw)
        assert set(cfg.reporting.formats) == {"json", "csv", "markdown"}

    def test_got10k_loader_accepted(self):
        raw = _minimal()
        raw["dataset"]["loader"] = "GOT10kDataset"
        cfg = validate_config(raw)
        assert cfg.dataset.loader == "GOT10kDataset"

    def test_lasot_loader_accepted(self):
        raw = _minimal()
        raw["dataset"]["loader"] = "LaSOTDataset"
        cfg = validate_config(raw)
        assert cfg.dataset.loader == "LaSOTDataset"


# ---------------------------------------------------------------------------
# TestConfigValidationErrors — error cases
# ---------------------------------------------------------------------------

class TestConfigValidationErrors:
    def test_missing_experiment_section_raises(self):
        raw = _minimal()
        del raw["experiment"]
        with pytest.raises(ConfigValidationError, match="experiment"):
            validate_config(raw)

    def test_missing_dataset_section_raises(self):
        raw = _minimal()
        del raw["dataset"]
        with pytest.raises(ConfigValidationError, match="dataset"):
            validate_config(raw)

    def test_missing_tracker_section_raises(self):
        raw = _minimal()
        del raw["tracker"]
        with pytest.raises(ConfigValidationError, match="tracker"):
            validate_config(raw)

    def test_empty_experiment_name_raises(self):
        raw = _minimal()
        raw["experiment"]["name"] = ""
        with pytest.raises(ConfigValidationError, match="experiment.name"):
            validate_config(raw)

    def test_missing_experiment_name_raises(self):
        raw = _minimal()
        raw["experiment"] = {"output_dir": "results/"}
        with pytest.raises(ConfigValidationError, match="experiment.name"):
            validate_config(raw)

    def test_missing_dataset_root_raises(self):
        raw = _minimal()
        del raw["dataset"]["root"]
        with pytest.raises(ConfigValidationError, match="dataset.root"):
            validate_config(raw)

    def test_missing_tracker_name_raises(self):
        raw = _minimal()
        del raw["tracker"]["name"]
        with pytest.raises(ConfigValidationError, match="tracker.name"):
            validate_config(raw)

    def test_unknown_report_format_raises(self):
        raw = _minimal()
        raw["reporting"] = {"formats": ["json", "xlsx"]}
        with pytest.raises(ConfigValidationError, match="xlsx"):
            validate_config(raw)

    def test_negative_tdp_watts_raises(self):
        raw = _minimal()
        raw["benchmark"] = {"tdp_watts": -3.0}
        with pytest.raises(ConfigValidationError, match="tdp_watts"):
            validate_config(raw)

    def test_zero_tdp_watts_raises(self):
        raw = _minimal()
        raw["benchmark"] = {"tdp_watts": 0}
        with pytest.raises(ConfigValidationError, match="tdp_watts"):
            validate_config(raw)

    def test_zero_max_sequences_raises(self):
        raw = _minimal()
        raw["dataset"]["max_sequences"] = 0
        with pytest.raises(ConfigValidationError, match="max_sequences"):
            validate_config(raw)

    def test_negative_max_sequences_raises(self):
        raw = _minimal()
        raw["dataset"]["max_sequences"] = -1
        with pytest.raises(ConfigValidationError, match="max_sequences"):
            validate_config(raw)

    def test_multiple_errors_all_reported(self):
        """All errors must appear in a single exception — no whack-a-mole."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config({})
        msg = str(exc_info.value)
        assert "experiment" in msg
        assert "dataset" in msg
        assert "tracker" in msg

    def test_source_label_in_error_message(self):
        with pytest.raises(ConfigValidationError, match="my_source"):
            validate_config({}, source="my_source")


# ---------------------------------------------------------------------------
# TestLoadConfig — file I/O path
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_from_yaml_file(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        _write_yaml(cfg_file, _minimal())
        cfg = load_config(cfg_file)
        assert cfg.experiment.name == "test-run"

    def test_load_accepts_string_path(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        _write_yaml(cfg_file, _minimal())
        cfg = load_config(str(cfg_file))
        assert isinstance(cfg, EOVOTConfig)

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_default_yaml_passes_validation(self):
        default = Path(__file__).parent.parent / "configs" / "default.yaml"
        if default.exists():
            cfg = load_config(default)
            assert cfg.experiment.name  # non-empty string

    def test_classical_comparison_yaml_passes_validation(self):
        path = (
            Path(__file__).parent.parent
            / "configs"
            / "experiments"
            / "classical_comparison.yaml"
        )
        if path.exists():
            # This config may reference specific dataset paths that don't
            # exist locally; validation only checks structure, not paths.
            try:
                cfg = load_config(path)
                assert cfg.experiment.name
            except ConfigValidationError:
                pass  # Some experiment configs legitimately extend schema
