"""Write the figure-set contract files ``docs/figures/*.seti.json`` (ADR-0098).

The contract is defined in ``probstat.visualization.theme`` (entity styles,
recurring quantities, category orders); this script writes it in the datawiz
set format, one file per palette. Run from the repository root::

    PYTHONPATH=src python tools/generate_figure_set_contract.py

``tests/test_figure_set_contract.py`` fails when a committed file is stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Write both contract files and print their paths.

    Returns:
        Process exit status (0).

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    sys.path.insert(0, str(ROOT / "src"))
    from probstat.visualization.figure_set import CONTRACT_FILES, contract_text
    from probstat.visualization.theme import GRAPHITE, PORCELAIN

    target = ROOT / "docs" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for palette in (PORCELAIN, GRAPHITE):
        path = target / CONTRACT_FILES[palette.name]
        path.write_text(contract_text(palette), encoding="utf-8", newline="\n")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
