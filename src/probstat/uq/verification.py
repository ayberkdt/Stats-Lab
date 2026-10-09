"""Numerical verification primitives for refinement and error accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
)


@dataclass(frozen=True, slots=True)
class NumericalErrorBudget:
    """Non-negative absolute bounds for major numerical error sources.

    The total is an additive conservative bound, not a root-sum-square estimate;
    no independence assumption is silently introduced.

    Args:
        truncation: Bound attributed to finite-series or iteration truncation.
        roundoff: Bound attributed to finite-precision arithmetic.
        discretization: Bound attributed to spatial or temporal discretization.
        iterative: Bound attributed to incomplete nonlinear/linear solves.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

    """

    truncation: float = 0.0
    roundoff: float = 0.0
    discretization: float = 0.0
    iterative: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize every absolute error bound.

        Raises:
            TypeError: If a component is not real.
            ValueError: If a component is negative or non-finite.

        References:
            [EQ-VV-003], [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        for name in ("truncation", "roundoff", "discretization", "iterative"):
            value = require_non_negative_finite_real(getattr(self, name), name=name)
            object.__setattr__(self, name, value)

    @property
    def total_bound(self) -> float:
        """Return the conservative sum of componentwise absolute bounds.

        Equation:
            [EQ-VV-003].

        References:
            [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        return math.fsum(
            (self.truncation, self.roundoff, self.discretization, self.iterative)
        )


def observed_order_of_accuracy(
    coarse: float,
    medium: float,
    fine: float,
    *,
    refinement_ratio: float = 2.0,
) -> float:
    """Estimate convergence order from three uniformly refined solutions.

    Args:
        coarse: QoI evaluated on the coarsest resolution.
        medium: QoI evaluated after one uniform refinement.
        fine: QoI evaluated after a second uniform refinement.
        refinement_ratio: Resolution ratio ``r>1`` shared by both refinements.

    Returns:
        Observed asymptotic order ``p``.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is non-finite, ``r<=1``, or consecutive
            solution differences are zero.

    Equation:
        [EQ-VV-001].

    References:
        [REF-ROACHE-1998, Chapters 4-5].

    """
    coarse_value = require_finite_real(coarse, name="coarse")
    medium_value = require_finite_real(medium, name="medium")
    fine_value = require_finite_real(fine, name="fine")
    ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
    if ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one")
    coarse_difference = abs(coarse_value - medium_value)
    fine_difference = abs(medium_value - fine_value)
    if coarse_difference == 0.0 or fine_difference == 0.0:
        raise ValueError("consecutive refinement differences must be non-zero")
    return math.log(coarse_difference / fine_difference) / math.log(ratio)


def richardson_extrapolation(
    medium: float,
    fine: float,
    *,
    refinement_ratio: float,
    order: float,
) -> float:
    """Extrapolate a refined QoI toward zero discretization size.

    Args:
        medium: QoI at resolution ``h``.
        fine: QoI at resolution ``h/refinement_ratio``.
        refinement_ratio: Uniform refinement ratio ``r>1``.
        order: Positive observed or theoretical convergence order.

    Returns:
        Richardson-extrapolated zero-size estimate.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is non-finite, ``r<=1``, or ``order<=0``.
        OverflowError: If exponentiation or the extrapolated result is non-finite.

    Equation:
        [EQ-VV-002].

    References:
        [REF-ROACHE-1998, Chapters 4-5].

    """
    medium_value = require_finite_real(medium, name="medium")
    fine_value = require_finite_real(fine, name="fine")
    ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
    checked_order = require_finite_real(order, name="order")
    if ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one")
    if checked_order <= 0.0:
        raise ValueError("order must be positive")
    try:
        denominator = math.pow(ratio, checked_order) - 1.0
    except OverflowError as error:
        raise OverflowError("refinement factor overflowed finite arithmetic") from error
    if not math.isfinite(denominator):
        raise OverflowError("refinement factor overflowed finite arithmetic")
    result = fine_value + (fine_value - medium_value) / denominator
    if not math.isfinite(result):
        raise OverflowError("Richardson extrapolation is not finite")
    return result
