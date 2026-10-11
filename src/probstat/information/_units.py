"""Information units shared by the information-theory modules."""

from __future__ import annotations

import math
from typing import Literal

InformationUnit = Literal["nats", "bits"]
UNITS: tuple[str, ...] = ("nats", "bits")


def unit_factor(unit: str) -> float:
    """Return the factor converting nats to ``unit``.

    One bit is ``log 2`` nats, so a value in nats is divided by ``log 2``.

    Args:
        unit: ``"nats"`` (natural logarithm) or ``"bits"`` (base 2).

    Returns:
        ``1`` for nats and ``1/log 2`` for bits.

    Raises:
        ValueError: If the unit is unknown.

    Equation:
        [EQ-INFO-H-001].

    References:
        [REF-COVER-THOMAS-2006, Section 2.1].

    """
    if unit == "nats":
        return 1.0
    if unit == "bits":
        return 1.0 / math.log(2.0)
    raise ValueError(f"unit must be one of {UNITS}")
