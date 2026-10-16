"""Learning (experience, progress) curves for recurring production cost.

Both classical theories use the power law ``Y = T_1 x^b`` with
``b = log2(s)``, where the *slope* ``s`` (for example 0.85 for an "85 %
curve") is the factor by which ``Y`` falls each time the quantity doubles:

- **Wright (cumulative-average) theory**: ``Y`` is the cumulative *average*
  cost of the first ``n`` units, so the cumulative total is ``T_1 n^(b+1)``
  and the cost of unit ``n`` is ``T_1 [n^(b+1) - (n-1)^(b+1)]``.
- **Crawford (unit) theory**: ``Y`` is the cost of unit ``n`` itself,
  ``T_n = T_1 n^b``, and totals are sums ``T_1 sum_i i^b``.

Lot costs are computed *exactly* by these sums. The classical lot-midpoint
shortcut replaces the unit-theory sum by the integral
``int_{F-1/2}^{L+1/2} x^b dx``; it is an approximation and is reported as
such together with its error relative to the exact sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations import require_finite_real

THEORIES = ("unit", "cumulative_average")


def _positive_integer(value: object, *, name: str) -> int:
    """Validate a positive integer unit number.

    Args:
        value: Candidate integer.
        name: Field name for error messages.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer (booleans are rejected).
        ValueError: If it is below one.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def learning_exponent(slope: float) -> float:
    """Return ``b = log2(slope)`` for a learning-curve slope.

    Args:
        slope: Slope in ``(0, 1]`` (``1`` means no learning). Values above one
            (cost growth with quantity) are accepted.

    Returns:
        Exponent ``b`` (negative for learning).

    Raises:
        ValueError: If the slope is not positive.

    Equation:
        [EQ-COST-LEARN-001].

    References:
        [REF-WRIGHT-1936], [REF-YELLE-1979].

    """
    s = require_finite_real(slope, name="slope")
    if s <= 0.0:
        raise ValueError("slope must be positive")
    return math.log2(s)


@dataclass(frozen=True, slots=True)
class LearningCurve:
    """A power-law learning curve under the unit or cumulative-average theory.

    Args:
        first_unit_cost: Finite positive ``T_1``.
        slope: Positive slope ``s``; ``b = log2(s)``.
        theory: ``"unit"`` (Crawford) or ``"cumulative_average"`` (Wright).

    References:
        [REF-WRIGHT-1936], [REF-YELLE-1979], [REF-NASA-CEH-2015].

    """

    first_unit_cost: float
    slope: float
    theory: str = "unit"

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If ``T_1`` or the slope is not positive, or the
                theory is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        t1 = require_finite_real(self.first_unit_cost, name="first_unit_cost")
        if t1 <= 0.0:
            raise ValueError("first_unit_cost must be positive")
        learning_exponent(self.slope)
        if self.theory not in THEORIES:
            raise ValueError("theory must be 'unit' or 'cumulative_average'")
        object.__setattr__(self, "first_unit_cost", t1)
        object.__setattr__(self, "slope", float(self.slope))

    @property
    def exponent(self) -> float:
        """Return ``b = log2(slope)``.

        Equation:
            [EQ-COST-LEARN-001].

        References:
            [REF-WRIGHT-1936].

        """
        return learning_exponent(self.slope)

    def unit_cost(self, n: int) -> float:
        """Return the cost of unit ``n``.

        Args:
            n: Unit number (at least one).

        Returns:
            ``T_1 n^b`` (unit theory) or ``T_1 [n^(b+1) - (n-1)^(b+1)]``
            (cumulative-average theory).

        Equation:
            [EQ-COST-LEARN-001], [EQ-COST-LEARN-002].

        References:
            [REF-WRIGHT-1936], [REF-YELLE-1979].

        """
        unit = _positive_integer(n, name="n")
        b = self.exponent
        if self.theory == "unit":
            return self.first_unit_cost * float(unit**b)
        return self.cumulative_cost(unit) - (
            self.cumulative_cost(unit - 1) if unit > 1 else 0.0
        )

    def cumulative_average_cost(self, n: int) -> float:
        """Return the average cost of the first ``n`` units.

        Args:
            n: Quantity (at least one).

        Returns:
            ``T_1 n^b`` (cumulative-average theory) or the exact unit-theory
            sum divided by ``n``.

        Equation:
            [EQ-COST-LEARN-001], [EQ-COST-LEARN-002].

        References:
            [REF-WRIGHT-1936], [REF-YELLE-1979].

        """
        quantity = _positive_integer(n, name="n")
        if self.theory == "cumulative_average":
            return self.first_unit_cost * float(quantity**self.exponent)
        return self.cumulative_cost(quantity) / quantity

    def cumulative_cost(self, n: int) -> float:
        """Return the exact total cost of units ``1, ..., n``.

        Args:
            n: Quantity (at least one).

        Returns:
            ``T_1 n^(b+1)`` (cumulative-average theory) or
            ``T_1 sum_{i=1}^n i^b`` (unit theory, summed with ``math.fsum``).

        Equation:
            [EQ-COST-LEARN-001], [EQ-COST-LEARN-002].

        References:
            [REF-WRIGHT-1936], [REF-YELLE-1979].

        """
        quantity = _positive_integer(n, name="n")
        b = self.exponent
        if self.theory == "cumulative_average":
            return self.first_unit_cost * float(quantity ** (b + 1.0))
        return self.first_unit_cost * math.fsum(i**b for i in range(1, quantity + 1))

    def lot_cost(self, first: int, last: int) -> float:
        """Return the exact total cost of units ``first, ..., last``.

        Args:
            first: First unit of the lot (at least one).
            last: Last unit of the lot (not below ``first``).

        Returns:
            Exact lot cost by the theory's formula.

        Raises:
            ValueError: If ``last < first``.

        Equation:
            [EQ-COST-LEARN-002].

        References:
            [REF-WRIGHT-1936], [REF-YELLE-1979].

        """
        f = _positive_integer(first, name="first")
        last_unit = _positive_integer(last, name="last")
        if last_unit < f:
            raise ValueError("last must not precede first")
        b = self.exponent
        if self.theory == "cumulative_average":
            before = (f - 1) ** (b + 1.0) if f > 1 else 0.0
            return self.first_unit_cost * (float(last_unit ** (b + 1.0)) - before)
        return self.first_unit_cost * math.fsum(i**b for i in range(f, last_unit + 1))

    def lot_midpoint_approximation(self, first: int, last: int) -> LotApproximation:
        """Return the unit-theory lot cost by the lot-midpoint approximation.

        The sum ``sum_{i=F}^{L} i^b`` is replaced by the midpoint-rule
        integral ``[(L + 1/2)^(b+1) - (F - 1/2)^(b+1)] / (b + 1)`` (for
        ``b = -1`` the logarithm). The lot midpoint is the unit whose cost
        equals the lot's average cost under this approximation. The exact sum
        is returned next to it so the approximation error is visible; it is
        largest for the first lot and steep curves.

        Args:
            first: First unit of the lot.
            last: Last unit of the lot.

        Returns:
            Approximate and exact lot costs, the lot midpoint, and the
            relative error.

        Raises:
            ValueError: If the theory is not ``"unit"`` or ``last < first``.

        Equation:
            [EQ-COST-LEARN-003].

        References:
            [REF-NASA-CEH-2015], [REF-YELLE-1979].

        """
        if self.theory != "unit":
            raise ValueError("the lot-midpoint approximation applies to unit theory")
        exact = self.lot_cost(first, last)
        b = self.exponent
        upper, lower = last + 0.5, first - 0.5
        if abs(b + 1.0) < 1e-12:
            integral = math.log(upper / lower)
        else:
            integral = (upper ** (b + 1.0) - lower ** (b + 1.0)) / (b + 1.0)
        approximate = self.first_unit_cost * integral
        size = last - first + 1
        average_factor = integral / size
        midpoint = math.sqrt(first * last) if b == 0.0 else average_factor ** (1.0 / b)
        return LotApproximation(
            first, last, approximate, exact, midpoint, (approximate - exact) / exact
        )


@dataclass(frozen=True, slots=True)
class LotApproximation:
    """Lot-midpoint approximation of a unit-theory lot cost.

    Args:
        first: First unit.
        last: Last unit.
        approximate_cost: Midpoint-integral approximation (an approximation).
        exact_cost: Exact sum of unit costs.
        lot_midpoint: Unit number ``x`` with ``T_1 x^b`` equal to the
            approximate average unit cost (undefined, reported as the
            geometric mean of the end units, when ``b = 0``).
        relative_error: ``(approximate - exact) / exact``.

    References:
        [REF-NASA-CEH-2015].

    """

    first: int
    last: int
    approximate_cost: float
    exact_cost: float
    lot_midpoint: float
    relative_error: float
