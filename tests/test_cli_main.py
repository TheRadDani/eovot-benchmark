"""Tests for the ``python -m eovot`` CLI entry point.

Validates argument parsing, subcommand dispatch, and the --dry-run / info /
list-trackers subcommands that do not require external datasets.

All tests run in-process via ``eovot.__main__.main()`` so they remain fast
and do not hit the filesystem beyond the built-in SyntheticDataset.
"""

from __future__ import annotations

import sys
import importlib
from io import StringIO
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(*argv: str) -> tuple[int, str]:
    """Run ``python -m eovot <argv>`` in-process; return (exit_code, stdout)."""
    captured = StringIO()
    with mock.patch("sys.stdout", captured), mock.patch("sys.argv", ["python -m eovot", *argv]):
        from eovot.__main__ import main
        try:
            code = main()
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 0
    return code, captured.getvalue()


# ---------------------------------------------------------------------------
# Smoke tests: subcommand parsing
# ---------------------------------------------------------------------------

class TestHelp:
    def test_top_level_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch("sys.argv", ["python -m eovot", "--help"]):
                from eovot.__main__ import main
                main()
        assert exc_info.value.code == 0

    def test_run_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch("sys.argv", ["python -m eovot", "run", "--help"]):
                from eovot.__main__ import main
                main()
        assert exc_info.value.code == 0

    def test_compare_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch("sys.argv", ["python -m eovot", "compare", "--help"]):
                from eovot.__main__ import main
                main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# list-trackers
# ---------------------------------------------------------------------------

class TestListTrackers:
    def test_exits_zero(self):
        code, out = _run_cli("list-trackers")
        assert code == 0

    def test_output_contains_known_trackers(self):
        code, out = _run_cli("list-trackers")
        assert code == 0
        for name in ("MOSSE", "KCF", "CamShift"):
            assert name in out

    def test_output_contains_class_path(self):
        code, out = _run_cli("list-trackers")
        assert "eovot.trackers" in out


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_exits_zero(self):
        code, out = _run_cli("info")
        assert code == 0

    def test_shows_version(self):
        import eovot
        code, out = _run_cli("info")
        assert eovot.__version__ in out

    def test_shows_numpy(self):
        code, out = _run_cli("info")
        assert "NumPy" in out

    def test_shows_opencv(self):
        code, out = _run_cli("info")
        assert "OpenCV" in out

    def test_shows_tracker_list(self):
        code, out = _run_cli("info")
        assert "Trackers:" in out


# ---------------------------------------------------------------------------
# run --dry-run (no real data needed)
# ---------------------------------------------------------------------------

class TestRunDryRun:
    def test_dry_run_exits_zero(self, tmp_path):
        """--dry-run should print the plan and return 0 without running anything."""
        config_content = """
experiment:
  name: test-dry-run
  seed: 0
  tdp_watts: null

dataset:
  loader: SyntheticDataset
  root: /tmp
  name: Synthetic

trackers:
  - name: KCF
    params:
      learning_rate: 0.125
"""
        cfg = tmp_path / "test.yaml"
        cfg.write_text(config_content)

        code, out = _run_cli("run", str(cfg), "--dry-run")
        assert code == 0
        assert "test-dry-run" in out
        assert "KCF" in out

    def test_missing_config_returns_nonzero(self):
        code, _ = _run_cli("run", "/nonexistent/path/config.yaml")
        assert code != 0

    def test_dry_run_shows_tracker_list(self, tmp_path):
        config_content = """
experiment:
  name: multi-tracker-dry
dataset:
  loader: SyntheticDataset
  root: /tmp
  name: Synthetic
trackers:
  - name: MOSSE
    params: {}
  - name: KCF
    params: {}
"""
        cfg = tmp_path / "multi.yaml"
        cfg.write_text(config_content)

        code, out = _run_cli("run", str(cfg), "--dry-run")
        assert code == 0
        assert "MOSSE" in out
        assert "KCF" in out


# ---------------------------------------------------------------------------
# compare — unknown tracker detection
# ---------------------------------------------------------------------------

class TestCompareValidation:
    def test_unknown_tracker_returns_nonzero(self):
        code, _ = _run_cli(
            "compare",
            "--trackers", "NON_EXISTENT_TRACKER",
            "--dataset-root", "/tmp",
        )
        assert code != 0


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

class TestModuleEntryPoint:
    def test_module_is_importable(self):
        import eovot.__main__ as m
        assert hasattr(m, "main")
        assert callable(m.main)

    def test_all_subcommand_funcs_are_callable(self):
        import eovot.__main__ as m
        for fn in (m._cmd_run, m._cmd_compare, m._cmd_list_trackers, m._cmd_info):
            assert callable(fn)
