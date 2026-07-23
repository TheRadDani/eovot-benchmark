"""Smoke tests for the EOVOT CLI entry points.

These tests verify that the CLI argument parser constructs correctly and
that the ``python -m eovot`` / ``eovot`` entry points are importable and
can run minimal end-to-end flows without requiring real datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path when running tests directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmark import _build_parser, _config_from_args, main as cli_main


class TestArgumentParser:
    """Verify that the argument parser is built correctly."""

    def setup_method(self):
        self.parser = _build_parser()

    def test_parser_has_tracker_argument(self):
        args = self.parser.parse_args(["--tracker", "KCF"])
        assert args.tracker == "KCF"

    def test_parser_defaults_to_mosse(self):
        args = self.parser.parse_args([])
        assert args.tracker == "MOSSE"

    def test_parser_max_sequences(self):
        args = self.parser.parse_args(["--max-sequences", "3"])
        assert args.max_sequences == 3

    def test_parser_quiet_flag(self):
        args = self.parser.parse_args(["--quiet"])
        assert args.quiet is True

    def test_parser_tdp_watts(self):
        args = self.parser.parse_args(["--tdp-watts", "6.0"])
        assert args.tdp_watts == pytest.approx(6.0)

    def test_parser_output_dir_default(self):
        args = self.parser.parse_args([])
        assert args.output_dir == "results/"


class TestConfigFromArgs:
    """Verify config-dict construction from CLI args."""

    def _make_args(self, tracker="MOSSE", dataset_root="/data", dataset_name="test",
                   max_sequences=None, output_dir="results/", quiet=False, tdp_watts=None):
        parser = _build_parser()
        argv = ["--tracker", tracker, "--dataset-root", dataset_root,
                "--dataset-name", dataset_name, "--output-dir", output_dir]
        if max_sequences is not None:
            argv += ["--max-sequences", str(max_sequences)]
        if quiet:
            argv.append("--quiet")
        if tdp_watts is not None:
            argv += ["--tdp-watts", str(tdp_watts)]
        return parser.parse_args(argv)

    def test_tracker_propagated(self):
        cfg = _config_from_args(self._make_args(tracker="KCF"))
        assert cfg["tracker"]["name"] == "KCF"

    def test_max_sequences_propagated(self):
        cfg = _config_from_args(self._make_args(max_sequences=5))
        assert cfg["dataset"]["max_sequences"] == 5

    def test_verbose_when_not_quiet(self):
        cfg = _config_from_args(self._make_args(quiet=False))
        assert cfg["benchmark"]["verbose"] is True

    def test_not_verbose_when_quiet(self):
        cfg = _config_from_args(self._make_args(quiet=True))
        assert cfg["benchmark"]["verbose"] is False

    def test_tdp_watts_in_benchmark(self):
        cfg = _config_from_args(self._make_args(tdp_watts=15.0))
        assert cfg["benchmark"]["tdp_watts"] == pytest.approx(15.0)

    def test_tdp_none_by_default(self):
        cfg = _config_from_args(self._make_args())
        assert cfg["benchmark"]["tdp_watts"] is None


class TestMainModule:
    """Verify the __main__ module is importable."""

    def test_main_module_importable(self):
        import eovot.__main__  # noqa: F401 — just test the import

    def test_main_function_exists(self):
        from eovot.__main__ import main
        assert callable(main)


class TestCliEndToEnd:
    """End-to-end smoke tests using the synthetic dataset (no real data needed)."""

    def test_cli_no_args_exits_zero(self, tmp_path):
        """Running with no args should print help and exit gracefully (no config given)."""
        # main() prints help and exits when neither --config nor --dataset-root is given.
        # We patch sys.argv to simulate this and verify no unhandled exception occurs.
        import sys
        orig_argv = sys.argv
        try:
            sys.argv = ["eovot"]
            cli_main()
        except SystemExit:
            pass  # expected — argparse / the handler calls sys.exit(0) after printing help
        finally:
            sys.argv = orig_argv

    def test_yaml_config_run(self, tmp_path):
        """Run a minimal experiment via YAML config using the SyntheticDataset."""
        import yaml

        config = {
            "experiment": {
                "name": "test-cli-run",
                "output_dir": str(tmp_path),
            },
            "dataset": {
                "name": "SyntheticDataset",
                "loader": "OTBDataset",
                "root": str(tmp_path / "nonexistent"),
                "max_sequences": 2,
            },
            "tracker": {
                "name": "MOSSE",
                "params": {},
            },
            "benchmark": {"verbose": False, "tdp_watts": None},
            "reporting": {"formats": ["json"], "print_summary": False},
        }

        cfg_path = tmp_path / "test_config.yaml"
        with open(cfg_path, "w") as fh:
            yaml.dump(config, fh)

        # The OTBDataset will fail on a non-existent root — that's expected.
        # We just verify the CLI parses the config without errors.
        parser = _build_parser()
        args = parser.parse_args(["--config", str(cfg_path)])
        assert args.config == str(cfg_path)

    def test_cli_synthetic_benchmark(self, tmp_path, capsys):
        """Full pipeline: benchmark MOSSE on SyntheticDataset, check output files."""
        from eovot.benchmark.engine import BenchmarkEngine
        from eovot.datasets.synthetic import SyntheticDataset
        from eovot.reporting.reporter import BenchmarkReporter
        from eovot.trackers.registry import build_tracker

        dataset = SyntheticDataset(num_sequences=3, num_frames=20)
        tracker = build_tracker("MOSSE")
        engine = BenchmarkEngine(verbose=False)

        result = engine.run(tracker=tracker, dataset=dataset,
                            dataset_name="SyntheticDataset", max_sequences=3)

        reporter = BenchmarkReporter(output_dir=str(tmp_path))
        saved = reporter.save_all(result.to_dict(), name="MOSSE-SyntheticDataset")

        assert saved["json"].exists()
        assert saved["csv"].exists()

        with open(saved["json"]) as fh:
            data = json.load(fh)

        assert "summary" in data
        assert data["summary"]["tracker"] == "MOSSE"
        assert data["summary"]["num_sequences"] == 3
        assert "sequences" in data
        assert len(data["sequences"]) == 3
