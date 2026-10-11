"""Multiple-testing adjustments controlling FWER or FDR.

Bonferroni and Holm control the family-wise error rate (probability of at
least one false rejection) under arbitrary dependence. Benjamini-Hochberg
controls the false discovery rate (expected share of false rejections among
rejections) for independent or positively regression-dependent p-values;
Benjamini-Yekutieli controls it under arbitrary dependence at the cost of the
harmonic factor ``c(m) = sum_{i=1}^{m} 1/i``. Adjusted p-values are the
smallest level at which each hypothesis would be rejected by the procedure;
they are not probabilities that hypotheses are true.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.validation import require_probability

from ._hypothesis import check_level

AdjustmentMethod = Literal["bonferroni", "holm", "bh", "by"]


@dataclass(frozen=True, slots=True)
class MultipleTestingResult:
    """Adjusted p-values and the rejection set of a multiple-testing procedure.

    Args:
        method: ``bonferroni``, ``holm``, ``bh``, or ``by``.
        p_values: Raw p-values in input order.
        adjusted_p_values: Adjusted p-values in input order, capped at one.
        rejected: Whether each hypothesis is rejected at ``alpha``.
        alpha: Target FWER (Bonferroni, Holm) or FDR (BH, BY) level.
        error_rate: ``"FWER"`` or ``"FDR"``.

    References:
        [REF-HOLM-1979], [REF-BENJAMINI-HOCHBERG-1995],
        [REF-BENJAMINI-YEKUTIELI-2001].

    """

    method: str
    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    error_rate: str

    @property
    def rejection_count(self) -> int:
        """Return the number of rejected hypotheses.

        Returns:
            Count of ``True`` entries in ``rejected``.

        References:
            [REF-BENJAMINI-HOCHBERG-1995]. No equation is evaluated.

        """
        return sum(self.rejected)


def adjust_p_values(
    p_values: Iterable[object],
    *,
    method: AdjustmentMethod = "holm",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Adjust a family of p-values for multiple testing.

    With ``m`` p-values sorted ascending as ``p_(1) <= ... <= p_(m)``:

    - Bonferroni: ``min(1, m p_i)``.
    - Holm: ``max_{j <= i} min(1, (m - j + 1) p_(j))`` (step-down).
    - BH: ``min_{j >= i} min(1, m p_(j) / j)`` (step-up).
    - BY: BH multiplied by ``c(m) = sum_{k=1}^{m} 1/k``.

    A hypothesis is rejected when its adjusted p-value is at most ``alpha``,
    which reproduces each procedure's rejection set.

    Args:
        p_values: Non-empty iterable of p-values in ``[0, 1]``.
        method: ``"bonferroni"``, ``"holm"``, ``"bh"``, or ``"by"``.
        alpha: Target error-rate level in ``(0, 1)``.

    Returns:
        Adjusted p-values and rejection decisions in input order.

    Raises:
        TypeError: If a p-value is not real.
        ValueError: If the family is empty, a p-value is outside ``[0, 1]``,
            or the method is unknown.

    Equation:
        [EQ-INF-MT-001], [EQ-INF-MT-002].

    References:
        [REF-HOLM-1979], [REF-BENJAMINI-HOCHBERG-1995],
        [REF-BENJAMINI-YEKUTIELI-2001], [REF-LEHMANN-ROMANO-2005, Chapter 9].

    """
    raw = tuple(
        require_probability(value, name=f"p_values[{index}]")
        for index, value in enumerate(p_values)
    )
    if not raw:
        raise ValueError("p_values must not be empty")
    level = check_level(alpha, name="alpha")
    m = len(raw)
    order = sorted(range(m), key=lambda index: raw[index])
    adjusted = [0.0] * m
    if method == "bonferroni":
        adjusted = [min(1.0, m * p) for p in raw]
        error_rate = "FWER"
    elif method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (m - rank) * raw[index]))
            adjusted[index] = running
        error_rate = "FWER"
    elif method in ("bh", "by"):
        factor = math.fsum(1.0 / k for k in range(1, m + 1)) if method == "by" else 1.0
        running = 1.0
        for rank in range(m - 1, -1, -1):
            index = order[rank]
            running = min(running, min(1.0, factor * m * raw[index] / (rank + 1)))
            adjusted[index] = running
        error_rate = "FDR"
    else:
        raise ValueError("method must be 'bonferroni', 'holm', 'bh', or 'by'")
    return MultipleTestingResult(
        method,
        raw,
        tuple(adjusted),
        tuple(p <= level for p in adjusted),
        level,
        error_rate,
    )
