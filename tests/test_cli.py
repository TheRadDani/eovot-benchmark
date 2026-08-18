"""Tests for the EOVOT unified CLI (python -m eovot)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from eovot.cli import build_parser, main, COMMANDS


# ---------------------------------------------------------------------------
# Parser smoke-tests
# ---------------------------------------------------------------------------

def test_parser_builds():
    parser = build_parser()
    assert parser is not None


def test_list_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["list"])
    assert args.command == "list"


def test_run_defaults():
    parser = build_parser()
    args = parser.parse_args(["run"])
    assert args.tracker == "MOSSE"
    assert args.sequences == 5
    assert args.frames == 100
    assert args.motion == "linear"
    assert args.dataset_root is None


def test_run_custom_tracker():
    parser = build_parser()
    args = parser.parse_args(["run", "--tracker", "KCF", "--sequences", "2"])
    assert args.tracker == "KCF"
    assert args.sequences == 2


def test_compare_parses():
    parser = build_parser()
    args = parser.parse_args(["compare", "--trackers", "MOSSE", "KCF"])
    assert args.trackers == ["MOSSE", "KCF"]


def test_report_requires_input():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["report"])


def test_bank_requires_bank():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bank"])


# ---------------------------------------------------------------------------
# Functional tests (run real subcommands)
# ---------------------------------------------------------------------------

def test_cmd_list(capsys):
    main(["list"])
    out = capsys.readouterr().out
    assert "MOSSE" in out
    assert "KCF" in out


def test_cmd_run_synthetic(capsys):
    """Run MOSSE on a tiny synthetic dataset end-to-end."""
    main(["run", "--tracker", "MOSSE", "--sequences", "2", "--frames", "15", "--quiet"])
    out = capsys.readouterr().out
    assert "MOSSE" in out
    assert "mIoU" in out


def test_cmd_run_saves_json(tmp_path, capsys):
    main([
        "run",
        "--tracker", "MOSSE",
        "--sequences", "1",
        "--frames", "10",
        "--output", str(tmp_path),
        "--quiet",
    ])
    capsys.readouterr()
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1, f"Expected 1 JSON file, found: {json_files}"
    data = json.loads(json_files[0].read_text())
    assert "summary" in data
    assert data["summary"]["tracker"] == "MOSSE"


def test_cmd_run_stores_in_bank(tmp_path, capsys):
    bank_dir = tmp_path / "bank"
    main([
        "run",
        "--tracker", "KCF",
        "--sequences", "1",
        "--frames", "10",
        "--output", str(tmp_path / "results"),
        "--bank", str(bank_dir),
        "--quiet",
    ])
    capsys.readouterr()
    assert bank_dir.exists(), "Bank directory should have been created"


def test_cmd_compare_two_trackers(capsys):
    main(["compare", "--trackers", "MOSSE", "KCF", "--sequences", "1", "--frames", "10", "--quiet"])
    out = capsys.readouterr().out
    assert "MOSSE" in out
    assert "KCF" in out
    assert "EES" in out or "Edge" in out or "mIoU" in out


def test_cmd_report(tmp_path, capsys):
    """Save a result then load it with the report subcommand."""
    # Step 1: run and save
    main([
        "run", "--tracker", "MOSSE",
        "--sequences", "1", "--frames", "10",
        "--output", str(tmp_path),
        "--quiet",
    ])
    capsys.readouterr()

    # Step 2: locate saved JSON
    json_files = list(tmp_path.glob("*.json"))
    assert json_files, "No JSON saved by run subcommand"

    # Step 3: report
    main(["report", "--input", str(json_files[0])])
    out = capsys.readouterr().out
    assert "MOSSE" in out


def test_python_m_eovot_list(capsys):
    """Simulate `python -m eovot list` via direct import."""
    main(["list"])
    out = capsys.readouterr().out
    assert "CSRT" in out
    assert "CamShift" in out


def test_unknown_tracker_exits(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--tracker", "NonExistentTracker9999", "--sequences", "1", "--frames", "5"])


def test_cmd_compare_motion_circular(capsys):
    main([
        "compare",
        "--trackers", "MOSSE", "KCF",
        "--sequences", "1",
        "--frames", "10",
        "--motion", "circular",
        "--quiet",
    ])
    out = capsys.readouterr().out
    assert "MOSSE" in out or "KCF" in out


def test_commands_dict_complete():
    """All expected commands are registered."""
    for cmd in ("list", "run", "compare", "report", "bank"):
        assert cmd in COMMANDS, f"Missing command: {cmd}"
