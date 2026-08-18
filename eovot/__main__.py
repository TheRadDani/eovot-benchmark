"""Entry point for ``python -m eovot``.

Delegates to :func:`eovot.cli.main` so the package can be invoked directly::

    python -m eovot --help
    python -m eovot list
    python -m eovot run --tracker MOSSE
    python -m eovot compare --trackers MOSSE KCF CSRT
"""

from eovot.cli import main

if __name__ == "__main__":
    main()
