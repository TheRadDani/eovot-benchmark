"""Validate an OTB-style dataset directory before running a benchmark.

Checks each sequence for:
  - Sufficient frame count (>= 2)
  - Valid GT file format (N x 4 float array)
  - Frame count matching GT row count
  - GT boxes with positive area
  - Boxes with non-negative coordinates
  - (Optional) Frame file readability by OpenCV

Usage::

    python scripts/validate_dataset.py --dataset-root /data/OTB100

    # Also check that each frame image file is readable
    python scripts/validate_dataset.py --dataset-root /data/OTB100 --check-frames

    # Write a Markdown report
    python scripts/validate_dataset.py --dataset-root /data/OTB100 --output report.md

    # Exit with non-zero code if errors are found (useful in CI)
    python scripts/validate_dataset.py --dataset-root /data/OTB100 --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an OTB-style dataset directory for EOVOT benchmarking.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root", "-d",
        required=True,
        help="Path to the dataset root (contains per-sequence subdirectories).",
    )
    parser.add_argument(
        "--check-frames", "-f",
        action="store_true",
        default=False,
        help=(
            "Also verify that every frame image file exists and can be\n"
            "decoded by OpenCV.  This is slow for large datasets."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write a Markdown report to this path (e.g. report.md).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with code 1 if any ERROR-level issues are found.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print per-sequence progress.",
    )
    args = parser.parse_args()

    try:
        from eovot.datasets.validator import DatasetValidator
    except ImportError as exc:
        print(f"Error: could not import eovot — is the package installed? ({exc})", file=sys.stderr)
        return 2

    validator = DatasetValidator(
        check_frames=args.check_frames,
        verbose=args.verbose,
    )

    print(f"Validating dataset at: {args.dataset_root}")
    report = validator.validate_directory(
        args.dataset_root,
        dataset_name=Path(args.dataset_root).name,
    )
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\nMarkdown report written to: {out_path}")

    if args.strict and not report.is_valid:
        print(f"\n{len(report.errors)} error(s) found — exiting with code 1.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
