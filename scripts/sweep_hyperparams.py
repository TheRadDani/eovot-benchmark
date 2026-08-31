#!/usr/bin/env python3
"""CLI: hyperparameter sensitivity sweep for EOVOT trackers.

Runs a 1D or 2D grid sweep over one or two hyperparameters of any registered
tracker, evaluating each configuration on a synthetic dataset (default) or any
real dataset.  Prints a Markdown sensitivity report and saves it to disk.

Examples::

    # 1D sweep of KCF learning_rate on synthetic data (fast)
    python scripts/sweep_hyperparams.py \\
        --tracker KCF \\
        --param learning_rate 0.02 0.05 0.075 0.125 0.2 0.3 \\
        --max-sequences 5

    # 2D sweep of KCF learning_rate x padding (use --param twice)
    python scripts/sweep_hyperparams.py \\
        --tracker KCF \\
        --param learning_rate 0.05 0.125 0.2 \\
        --param padding 1.0 1.5 2.0 \\
        --max-sequences 5

    # Sweep MOSSE on a real OTB dataset (10 sequences)
    python scripts/sweep_hyperparams.py \\
        --tracker MOSSE \\
        --param learning_rate 0.05 0.125 0.2 \\
        --dataset-root /data/OTB100 \\
        --dataset-name OTB100 \\
        --max-sequences 10 \\
        --output-dir results/sweeps/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eovot.analysis.hyperparam_sweep import HyperparamSweep, SweepAxis
from eovot.trackers.registry import available_trackers


def _parse_value(s: str):
    """Try to parse a string as int, then float, then keep as string."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _build_dataset(root: str | None, name: str):
    """Build a dataset from --dataset-root or fall back to SyntheticDataset."""
    if root is None:
        from eovot.datasets.synthetic import SyntheticDataset
        print("[info] No --dataset-root given — using SyntheticDataset (10 seq, 100 frames).")
        return SyntheticDataset(
            num_sequences=10,
            num_frames=100,
            frame_size=(320, 240),
            bbox_size=(40, 40),
            motion="linear",
            seed=42,
        ), "Synthetic"
    from eovot.datasets.base import OTBDataset
    return OTBDataset(root=root), name


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sweep_hyperparams",
        description="Hyperparameter sensitivity sweep for EOVOT trackers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracker",
        required=True,
        choices=available_trackers(),
        metavar="TRACKER",
        help=f"Tracker to sweep. Choices: {available_trackers()}",
    )
    parser.add_argument(
        "--param",
        dest="params",
        nargs="+",
        action="append",
        required=True,
        metavar=("PARAM_NAME", "VALUE"),
        help=(
            "Parameter name followed by values to sweep.  Use --param twice "
            "for a 2D sweep: --param lr 0.05 0.125 --param padding 1.0 1.5"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        metavar="PATH",
        help="Path to OTB dataset root. Omit to use built-in SyntheticDataset.",
    )
    parser.add_argument(
        "--dataset-name",
        default="OTB",
        metavar="NAME",
        help="Human-readable dataset label for reports.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        metavar="N",
        help="Cap on sequences evaluated per configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to save the Markdown report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-configuration progress.",
    )
    args = parser.parse_args()

    if len(args.params) > 2:
        parser.error("At most two --param flags are supported (1D or 2D sweep).")

    # Build SweepAxis objects
    axes = []
    for param_spec in args.params:
        if len(param_spec) < 2:
            parser.error(
                f"--param {param_spec[0]} requires at least one value to sweep."
            )
        param_name = param_spec[0]
        values = [_parse_value(v) for v in param_spec[1:]]
        axes.append(SweepAxis(param_name, values))

    dataset, dataset_name = _build_dataset(args.dataset_root, args.dataset_name)

    sweep = HyperparamSweep(
        tracker_name=args.tracker,
        dataset=dataset,
        dataset_name=dataset_name,
        max_sequences=args.max_sequences,
        verbose=args.verbose,
    )

    print(f"Sweeping {args.tracker} on {dataset_name} ...")
    if len(axes) == 1:
        result = sweep.run_1d(axes[0])
    else:
        result = sweep.run_2d(axes[0], axes[1])

    print()
    print(result.to_markdown())

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / f"sweep_{args.tracker}.md"
        md_path.write_text(result.to_markdown(), encoding="utf-8")
        print(f"\n[Markdown] saved → {md_path}")


if __name__ == "__main__":
    main()
