"""YAML config loading and validation for EOVOT experiments.

Provides two public entry points:

* :func:`load_config` — load a YAML file from disk and return a validated
  :class:`~eovot.config.schema.EOVOTConfig`.
* :func:`validate_config` — validate a raw ``dict`` (e.g. built in a test or
  script) and return a validated :class:`~eovot.config.schema.EOVOTConfig`.

Both functions raise :class:`ConfigValidationError` with a bullet-point list
of *all* errors found, so users can fix every problem in one pass rather than
playing whack-a-mole with successive ``ValueError`` exceptions.

Typical usage::

    from eovot.config import load_config

    cfg = load_config("configs/default.yaml")
    print(cfg.tracker.name)          # "MOSSE"
    print(cfg.benchmark.tdp_watts)   # None
    print(cfg.dataset.max_sequences) # None
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from .schema import (
    BUILTIN_LOADERS,
    VALID_REPORT_FORMATS,
    BenchmarkConfig,
    DatasetConfig,
    EOVOTConfig,
    ExperimentConfig,
    ReportingConfig,
    TrackerConfig,
)


class ConfigValidationError(ValueError):
    """Raised when a config contains invalid or missing fields.

    The error message is a formatted bullet-point list of every problem found,
    so researchers can fix all issues in one pass.
    """


def load_config(path: Union[str, Path]) -> EOVOTConfig:
    """Load and validate an EOVOT YAML config file.

    Args:
        path: Path to a YAML config file (``str`` or :class:`pathlib.Path`).

    Returns:
        Fully validated :class:`~eovot.config.schema.EOVOTConfig`.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
        ConfigValidationError: If required fields are missing or values are
            invalid.  The message lists every problem found.

    Example::

        cfg = load_config("configs/default.yaml")
        engine = BenchmarkEngine(
            verbose=cfg.benchmark.verbose,
            tdp_watts=cfg.benchmark.tdp_watts,
        )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"Config must be a YAML mapping at the top level, "
            f"got {type(raw).__name__!r} in {path}"
        )

    return validate_config(raw, source=str(path))


def validate_config(
    raw: Dict[str, Any],
    source: str = "<dict>",
) -> EOVOTConfig:
    """Validate a raw config dict and return a typed :class:`~eovot.config.schema.EOVOTConfig`.

    Collects *all* validation errors before raising so callers can fix every
    problem in one pass.

    Args:
        raw:    Dict as loaded from YAML or constructed programmatically.
        source: Human-readable label embedded in error messages (default: ``"<dict>"``).

    Returns:
        Validated :class:`~eovot.config.schema.EOVOTConfig`.

    Raises:
        ConfigValidationError: If any required field is missing or any value
            fails a type or range check.
    """
    errors: List[str] = []

    # ------------------------------------------------------------------ #
    # experiment section
    # ------------------------------------------------------------------ #
    exp_raw = raw.get("experiment") or {}
    exp_name = exp_raw.get("name") if isinstance(exp_raw, dict) else None
    if not isinstance(exp_raw, dict):
        errors.append("'experiment' must be a YAML mapping")
        exp_raw = {}
    if not exp_name:
        errors.append("experiment.name is required and must be a non-empty string")

    seed_raw = exp_raw.get("seed", 42)
    if not isinstance(seed_raw, int):
        errors.append(f"experiment.seed must be an integer, got {seed_raw!r}")
        seed_raw = 42

    exp_cfg = ExperimentConfig(
        name=exp_name or "unnamed",
        output_dir=str(exp_raw.get("output_dir", "results/")),
        seed=int(seed_raw),
    )

    # ------------------------------------------------------------------ #
    # dataset section
    # ------------------------------------------------------------------ #
    ds_raw = raw.get("dataset") or {}
    if not isinstance(ds_raw, dict):
        errors.append("'dataset' must be a YAML mapping")
        ds_raw = {}

    ds_name = ds_raw.get("name")
    ds_loader = ds_raw.get("loader")
    ds_root = ds_raw.get("root")

    if not ds_name:
        errors.append("dataset.name is required")
    if not ds_loader:
        errors.append("dataset.loader is required")
    elif ds_loader not in BUILTIN_LOADERS:
        # Warn, not error — users may supply custom loaders outside EOVOT.
        # We include it as a soft hint rather than a hard failure.
        errors.append(
            f"dataset.loader '{ds_loader}' is not a built-in EOVOT loader "
            f"(known: {sorted(BUILTIN_LOADERS)}).  "
            "If this is a custom loader, ensure it subclasses BaseDataset."
        )
    if not ds_root:
        errors.append("dataset.root is required")

    max_seq = ds_raw.get("max_sequences")
    if max_seq is not None:
        if not isinstance(max_seq, int) or max_seq <= 0:
            errors.append(
                f"dataset.max_sequences must be a positive integer, got {max_seq!r}"
            )
            max_seq = None

    ds_cfg = DatasetConfig(
        name=str(ds_name) if ds_name else "unknown",
        loader=str(ds_loader) if ds_loader else "OTBDataset",
        root=str(ds_root) if ds_root else "",
        max_sequences=max_seq,
        split=ds_raw.get("split"),
    )

    # ------------------------------------------------------------------ #
    # tracker section
    # ------------------------------------------------------------------ #
    tr_raw = raw.get("tracker") or {}
    if not isinstance(tr_raw, dict):
        errors.append("'tracker' must be a YAML mapping")
        tr_raw = {}

    tr_name = tr_raw.get("name")
    if not tr_name:
        errors.append("tracker.name is required")

    tr_params = tr_raw.get("params") or {}
    if not isinstance(tr_params, dict):
        errors.append(f"tracker.params must be a YAML mapping, got {type(tr_params).__name__!r}")
        tr_params = {}

    tr_cfg = TrackerConfig(
        name=str(tr_name) if tr_name else "unknown",
        params=dict(tr_params),
    )

    # ------------------------------------------------------------------ #
    # benchmark section  (fully optional)
    # ------------------------------------------------------------------ #
    bm_raw = raw.get("benchmark") or {}
    if not isinstance(bm_raw, dict):
        errors.append("'benchmark' must be a YAML mapping")
        bm_raw = {}

    tdp = bm_raw.get("tdp_watts")
    if tdp is not None:
        if not isinstance(tdp, (int, float)) or tdp <= 0:
            errors.append(
                f"benchmark.tdp_watts must be a positive number (Watts), got {tdp!r}"
            )
            tdp = None

    bm_cfg = BenchmarkConfig(
        verbose=bool(bm_raw.get("verbose", True)),
        tdp_watts=float(tdp) if tdp is not None else None,
        save_predictions=bool(bm_raw.get("save_predictions", False)),
    )

    # ------------------------------------------------------------------ #
    # reporting section  (fully optional)
    # ------------------------------------------------------------------ #
    rp_raw = raw.get("reporting") or {}
    if not isinstance(rp_raw, dict):
        errors.append("'reporting' must be a YAML mapping")
        rp_raw = {}

    formats = rp_raw.get("formats", ["json", "csv"])
    if not isinstance(formats, list):
        errors.append(
            f"reporting.formats must be a YAML list, got {type(formats).__name__!r}"
        )
        formats = ["json", "csv"]
    else:
        unknown = [f for f in formats if f not in VALID_REPORT_FORMATS]
        if unknown:
            errors.append(
                f"reporting.formats contains unrecognised values {unknown}. "
                f"Valid formats: {sorted(VALID_REPORT_FORMATS)}"
            )

    rp_cfg = ReportingConfig(
        formats=list(formats),
        print_summary=bool(rp_raw.get("print_summary", True)),
    )

    # ------------------------------------------------------------------ #
    # Raise if any errors were collected
    # ------------------------------------------------------------------ #
    if errors:
        bullet = "\n  • ".join([""] + errors)
        raise ConfigValidationError(
            f"Config validation failed ({source}):{bullet}"
        )

    return EOVOTConfig(
        experiment=exp_cfg,
        dataset=ds_cfg,
        tracker=tr_cfg,
        benchmark=bm_cfg,
        reporting=rp_cfg,
    )
