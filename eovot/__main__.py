"""Allow ``python -m eovot`` to invoke the EOVOT CLI.

Delegates to :func:`scripts.run_benchmark.main` so both ``python -m eovot``
and the ``eovot`` console-script entry point share a single implementation.

Usage::

    python -m eovot --config configs/experiments/synthetic_demo.yaml
    python -m eovot --tracker MOSSE --dataset-root /data/OTB100 --max-sequences 5
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point used by both ``python -m eovot`` and the ``eovot`` console script."""
    from scripts.run_benchmark import main as _run_benchmark_main

    sys.exit(_run_benchmark_main() or 0)


if __name__ == "__main__":
    main()
