#!/usr/bin/env python3
"""CLI for EOVOT hyperparameter sensitivity analysis.

Runs a Cartesian-product sweep over a tracker's hyperparameters on a
SyntheticDataset (or a real OTB dataset) and prints ranked results with
Pareto-optimal configurations highlighted.

Usage examples
--------------
# Sweep MOSSE learning_rate on 10 synthetic sequences:
python scripts/run_parameter_sweep.py \\
    --tracker MOSSE \\
    --params "learning_rate=0.05,0.125,0.25,0.5" \\
    --num-sequences 10 --num-frames 100

# 2-D sweep + budget filter (need >= 30 FPS):
python scripts/run_parameter_sweep.py \\
    --tracker MOSSE \\
    --params "learning_rate=0.05,0.25,0.5" "sigma=1.0,2.0,4.0" \\
    --budget-fps 30 \\
    --output-dir results/sweeps/

# 1-D sensitivity sweep for KCF lambda:
python scripts/run_parameter_sweep.py \\
    --tracker KCF \\
    --params "lambda_=1e-4,1e-3,1e-2,0.1" \\
    --sensitivity-param lambda_ \\
    --num-sequences 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_param_spec(spec: str) -> tuple[str, list]:
    """Parse ``"name=v1,v2,v3"`` into ``(name, [v1, v2, v3])``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid param spec {spec!r}. Expected format: name=v1,v2,..."
        )
    name, raw = spec.split("=", 1)
    name = name.strip()
    values = []
    for token in raw.split(","):
        token = token.strip()
        try:
            values.append(int(token))
        except ValueError:
            try:
                values.append(float(token))
            except ValueError:
                values.append(token)
    return name, values


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EOVOT hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tracker", required=True, help="Tracker registry name (e.g. MOSSE, KCF)")
    p.add_argument(
        "--params",
        nargs="+",
        required=True,
        metavar="NAME=V1,V2,...",
        help="One or more parameter specs to sweep (Cartesian product)",
    )
    p.add_argument("--dataset-name", default="Synthetic", help="Label used in reports")
    p.add_argument("--num-sequences", type=int, default=10, help="SyntheticDataset sequences")
    p.add_argument("--num-frames", type=int, default=100, help="Frames per synthetic sequence")
    p.add_argument("--seed", type=int, default=42, help="SyntheticDataset RNG seed")
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap on sequences evaluated per config (default: all)",
    )
    p.add_argument(
        "--budget-fps",
        type=float,
        default=None,
        help="Filter Pareto configs to those with FPS >= this value",
    )
    p.add_argument(
        "--sensitivity-param",
        default=None,
        help="Print 1-D sensitivity table for this parameter after the main sweep",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON output (default: print only, no files written)",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-config progress lines")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Parse param specs
    param_grid: dict[str, list] = {}
    for spec in args.params:
        try:
            name, values = _parse_param_spec(spec)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
            return 1
        if not values:
            parser.error(f"No values for parameter {name!r}")
        param_grid[name] = values

    # Build dataset
    try:
        from eovot.datasets.synthetic import SyntheticDataset
    except ImportError as exc:
        print(f"[error] Failed to import SyntheticDataset: {exc}", file=sys.stderr)
        return 1

    dataset = SyntheticDataset(
        num_sequences=args.num_sequences,
        num_frames=args.num_frames,
        seed=args.seed,
    )

    # Run sweep
    try:
        from eovot.analysis.parameter_sweep import ParameterSweep
    except ImportError as exc:
        print(f"[error] Failed to import ParameterSweep: {exc}", file=sys.stderr)
        return 1

    sweep = ParameterSweep(
        tracker_name=args.tracker,
        dataset=dataset,
        verbose=not args.quiet,
    )

    try:
        result = sweep.run(
            param_grid=param_grid,
            dataset_name=args.dataset_name,
            max_sequences=args.max_sequences,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    # Print main sweep table
    print(result.to_markdown())

    # Pareto configs
    pareto = result.pareto_configs(budget_fps=args.budget_fps)
    if pareto:
        fps_tag = f" (FPS >= {args.budget_fps})" if args.budget_fps else ""
        print(f"\n### Pareto-Optimal Configurations{fps_tag}\n")
        for e in pareto:
            print(f"  {e}")
    else:
        budget_tag = f" with FPS >= {args.budget_fps}" if args.budget_fps else ""
        print(f"\n(No Pareto-optimal configurations found{budget_tag}.)")

    # 1-D sensitivity table
    if args.sensitivity_param:
        print()
        print(result.to_sensitivity_markdown(args.sensitivity_param))

    # Optional JSON output
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_tracker = args.tracker.replace("/", "_")
        json_path = out_dir / f"sweep_{safe_tracker}.json"
        with open(json_path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"\n[sweep] Results saved to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
