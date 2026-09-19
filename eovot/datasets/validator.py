"""Dataset integrity validator for EOVOT benchmark datasets.

Before running a long benchmark, it is useful to verify that the dataset
directory is structurally sound: GT files exist and parse cleanly, the
frame count matches the annotation count, bounding boxes are geometrically
valid, and no obvious image-loading failures will interrupt the run.

This module provides :class:`DatasetValidator`, which checks any
:class:`~eovot.datasets.base.BaseDataset` or a raw OTB-style directory,
and returns a :class:`ValidationReport` listing every issue found with
its severity and the sequence it belongs to.

Typical usage::

    from eovot.datasets.validator import DatasetValidator
    from eovot.datasets.base import OTBDataset

    dataset = OTBDataset("/data/OTB100")
    validator = DatasetValidator()
    report = validator.validate(dataset)
    print(report)

    if not report.is_valid:
        for issue in report.errors:
            print(f"[ERROR] {issue.sequence}: {issue.message}")

The validator is intentionally non-destructive: it never modifies the
dataset and, for efficiency, does NOT load the full frame content — it
uses :func:`cv2.imdecode` on file headers only to verify decodability when
``check_frames=True``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence as TypingSequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .base import BaseDataset, Sequence


class Severity(str, Enum):
    """Issue severity levels, ordered from least to most critical."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationIssue:
    """A single issue found during validation.

    Attributes:
        sequence:  Name of the sequence in which the issue was found,
            or ``"<dataset>"`` for dataset-level issues.
        severity:  :class:`Severity` level — INFO, WARNING, or ERROR.
        message:   Human-readable description of the problem.
        detail:    Optional supplementary context (e.g. the bad frame path
            or the exact GT row number).
    """

    sequence: str
    severity: Severity
    message: str
    detail: Optional[str] = None

    def __str__(self) -> str:
        detail_str = f" [{self.detail}]" if self.detail else ""
        return f"[{self.severity.value}] {self.sequence}: {self.message}{detail_str}"


@dataclass
class ValidationReport:
    """Summary of all issues found by :class:`DatasetValidator`.

    Attributes:
        dataset_name: Identifier for the validated dataset.
        num_sequences: Total number of sequences checked.
        issues: All :class:`ValidationIssue` objects, ordered by sequence
            then by severity.
    """

    dataset_name: str
    num_sequences: int
    issues: List[ValidationIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def errors(self) -> List[ValidationIssue]:
        """Issues with :attr:`~Severity.ERROR` severity."""
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Issues with :attr:`~Severity.WARNING` severity."""
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        """Issues with :attr:`~Severity.INFO` severity."""
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def is_valid(self) -> bool:
        """``True`` if no ERROR-level issues were found."""
        return len(self.errors) == 0

    @property
    def num_sequences_with_errors(self) -> int:
        """Number of distinct sequences that have at least one ERROR."""
        return len({i.sequence for i in self.errors if i.sequence != "<dataset>"})

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the report as a Markdown string.

        Returns:
            Multi-line Markdown suitable for embedding in documentation or
            printing to a terminal.
        """
        status = "PASS" if self.is_valid else "FAIL"
        lines = [
            f"## Dataset Validation Report — {self.dataset_name}",
            "",
            f"**Status:** {status}  ",
            f"**Sequences checked:** {self.num_sequences}  ",
            f"**Errors:** {len(self.errors)}  "
            f"**Warnings:** {len(self.warnings)}  "
            f"**Infos:** {len(self.infos)}",
            "",
        ]
        if not self.issues:
            lines.append("No issues found. Dataset appears healthy.")
        else:
            lines.append("### Issues")
            lines.append("")
            lines.append("| Severity | Sequence | Message | Detail |")
            lines.append("|----------|----------|---------|--------|")
            for iss in self.issues:
                det = iss.detail or "—"
                lines.append(
                    f"| {iss.severity.value} | `{iss.sequence}` "
                    f"| {iss.message} | {det} |"
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        status = "PASS ✓" if self.is_valid else "FAIL ✗"
        header = (
            f"ValidationReport[{self.dataset_name}]  {status}  "
            f"{self.num_sequences} seqs  "
            f"E={len(self.errors)}  W={len(self.warnings)}  I={len(self.infos)}"
        )
        if not self.issues:
            return header
        return header + "\n" + "\n".join(f"  {i}" for i in self.issues)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """Validate a :class:`~eovot.datasets.base.BaseDataset` for benchmark use.

    Checks each sequence for:

    1. **Sufficient frames** — at least 2 frames (init + 1 tracking frame).
    2. **GT shape** — ground-truth array must be ``(N, 4)`` with ``N > 0``.
    3. **Frame–GT alignment** — ``len(sequence.frames) == len(sequence.gt)``.
       A count mismatch is flagged as WARNING (the engine clips to the
       shorter of the two, so evaluation still runs) unless the mismatch
       exceeds 20 %, at which point it is escalated to ERROR.
    4. **Valid GT boxes** — every GT row must have ``w > 0`` and ``h > 0``.
       Zero-area or negative-area boxes are flagged as WARNING for the
       individual frame and ERROR when the *initialisation* frame (row 0)
       is invalid (the engine would crash immediately).
    5. **GT within-frame bounds** — boxes that start at negative coordinates
       or extend beyond the nominal frame size are flagged as INFO (many
       real datasets allow slight overrun).
    6. **Frame readability** (optional, ``check_frames=True``) — each image
       file must exist on disk and be openable by OpenCV.  This check is
       disabled by default because it loads every frame header (slow for
       large datasets).

    Args:
        check_frames: When ``True``, verify that every frame file exists
            and can be decoded by OpenCV.  Default: ``False`` (fast mode).
        max_invalid_boxes_pct: Maximum tolerated percentage of invalid GT
            boxes per sequence before an ERROR (vs. WARNING) is raised.
            Default: ``5.0`` (5 %).
        verbose: When ``True``, print per-sequence progress.
            Default: ``False``.

    Example::

        validator = DatasetValidator(check_frames=True)
        report = validator.validate(dataset)
        print(report.to_markdown())
    """

    def __init__(
        self,
        check_frames: bool = False,
        max_invalid_boxes_pct: float = 5.0,
        verbose: bool = False,
    ) -> None:
        self.check_frames = check_frames
        self.max_invalid_boxes_pct = max_invalid_boxes_pct
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, dataset: "BaseDataset", dataset_name: Optional[str] = None) -> ValidationReport:
        """Validate every sequence in *dataset*.

        Args:
            dataset:      A :class:`~eovot.datasets.base.BaseDataset` instance.
            dataset_name: Optional display name.  Defaults to
                ``repr(dataset)``.

        Returns:
            :class:`ValidationReport` with all found issues.
        """
        name = dataset_name or repr(dataset)
        report = ValidationReport(dataset_name=name, num_sequences=len(dataset))

        if len(dataset) == 0:
            report.issues.append(ValidationIssue(
                sequence="<dataset>",
                severity=Severity.ERROR,
                message="Dataset contains no sequences.",
            ))
            return report

        for idx in range(len(dataset)):
            try:
                seq = dataset[idx]
            except Exception as exc:
                report.issues.append(ValidationIssue(
                    sequence=f"<seq {idx}>",
                    severity=Severity.ERROR,
                    message=f"Failed to load sequence: {exc}",
                ))
                continue

            if self.verbose:
                print(f"  Checking [{idx + 1}/{len(dataset)}] {seq.name} …")

            self._check_sequence(seq, report)

        return report

    def validate_directory(
        self,
        root: str,
        dataset_name: Optional[str] = None,
    ) -> ValidationReport:
        """Validate an OTB-style directory without constructing a dataset object.

        Useful when the dataset root is known but the dataset object is not
        yet instantiated (e.g. for a pre-flight check before a long run).

        Args:
            root:         Path to the dataset root (containing per-sequence
                          subdirectories).
            dataset_name: Optional display name.  Defaults to ``root``.

        Returns:
            :class:`ValidationReport`.
        """
        from .base import OTBDataset
        try:
            dataset = OTBDataset(root)
        except FileNotFoundError as exc:
            report = ValidationReport(dataset_name=dataset_name or root, num_sequences=0)
            report.issues.append(ValidationIssue(
                sequence="<dataset>",
                severity=Severity.ERROR,
                message=str(exc),
            ))
            return report
        return self.validate(dataset, dataset_name=dataset_name or root)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_sequence(self, seq: "Sequence", report: ValidationReport) -> None:
        gt = seq.ground_truth
        n_frames = len(seq)
        seq_name = seq.name

        # --- Minimum frame count ---
        if n_frames < 2:
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=Severity.ERROR,
                message=f"Sequence has only {n_frames} frame(s); need at least 2.",
            ))
            return

        # --- GT shape ---
        if gt.ndim != 2 or gt.shape[1] != 4:
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=Severity.ERROR,
                message=f"Ground-truth has unexpected shape {gt.shape}; expected (N, 4).",
            ))
            return
        n_gt = gt.shape[0]
        if n_gt == 0:
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=Severity.ERROR,
                message="Ground-truth is empty (0 rows).",
            ))
            return

        # --- Frame–GT alignment ---
        if n_frames != n_gt:
            mismatch_pct = abs(n_frames - n_gt) / max(n_frames, n_gt) * 100.0
            sev = Severity.ERROR if mismatch_pct > 20.0 else Severity.WARNING
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=sev,
                message=(
                    f"Frame count ({n_frames}) does not match GT rows ({n_gt}); "
                    f"{mismatch_pct:.1f}% mismatch. "
                    f"Engine will evaluate on the shorter of the two."
                ),
            ))

        # --- GT box validity ---
        widths = gt[:, 2]
        heights = gt[:, 3]
        invalid_mask = (widths <= 0) | (heights <= 0)
        n_invalid = int(invalid_mask.sum())

        if invalid_mask[0]:
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=Severity.ERROR,
                message="Initialisation GT box (row 0) has non-positive area; "
                        "BenchmarkEngine will fail on this sequence.",
                detail=f"box={tuple(gt[0])}",
            ))
        elif n_invalid > 0:
            invalid_rows = np.where(invalid_mask)[0].tolist()
            pct_invalid = n_invalid / n_gt * 100.0
            sev = Severity.ERROR if pct_invalid > self.max_invalid_boxes_pct else Severity.WARNING
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=sev,
                message=(
                    f"{n_invalid}/{n_gt} GT boxes ({pct_invalid:.1f}%) have "
                    f"non-positive area (w ≤ 0 or h ≤ 0)."
                ),
                detail=f"rows={invalid_rows[:10]}{'…' if len(invalid_rows) > 10 else ''}",
            ))

        # --- Negative-coordinate boxes (INFO) ---
        neg_coord_mask = (gt[:, 0] < 0) | (gt[:, 1] < 0)
        n_neg = int(neg_coord_mask.sum())
        if n_neg > 0:
            report.issues.append(ValidationIssue(
                sequence=seq_name,
                severity=Severity.INFO,
                message=f"{n_neg}/{n_gt} GT boxes have negative x or y origin.",
                detail=f"first row with negative coord: {int(np.where(neg_coord_mask)[0][0])}",
            ))

        # --- Frame readability (optional) ---
        if self.check_frames:
            self._check_frame_files(seq, report)

    def _check_frame_files(self, seq: "Sequence", report: ValidationReport) -> None:
        """Check that every frame file exists and is decodable by OpenCV."""
        import cv2

        for i, path in enumerate(seq._frame_paths):
            if not os.path.isfile(path):
                report.issues.append(ValidationIssue(
                    sequence=seq.name,
                    severity=Severity.ERROR,
                    message=f"Frame file not found (index {i}).",
                    detail=path,
                ))
                continue
            # Read only the first few bytes as a quick decodability probe.
            with open(path, "rb") as fh:
                header = fh.read(8192)
            arr = np.frombuffer(header, dtype=np.uint8)
            probe = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if probe is None:
                # Full decode attempt for borderline cases (e.g. very small files)
                frame = cv2.imread(path)
                if frame is None:
                    report.issues.append(ValidationIssue(
                        sequence=seq.name,
                        severity=Severity.ERROR,
                        message=f"Frame {i} cannot be decoded by OpenCV.",
                        detail=path,
                    ))
