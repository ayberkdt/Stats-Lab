"""Result record and shared helpers for the error-controlled integrators.

``gauss_kronrod``, ``tanh_sinh_interval``, ``exp_sinh``, ``sinh_sinh`` and
``integrate`` all return an :class:`IntegrationResult`, so a caller can read
the estimate, its error estimate, the work done, and an honest convergence
flag in the same way whichever rule ran.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from probstat.foundations.validation import require_finite_real

# QUADPACK refuses relative tolerances below 50 machine epsilons when no
# absolute tolerance is given, because its error estimates never drop below
# 50 eps times the integral of |f|; the same floor applies here.
ROUNDOFF_FACTOR = 50.0 * 2.220446049250313e-16


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """An integral estimate with its error estimate and convergence evidence.

    Args:
        value: Integral estimate.
        error_estimate: Estimated absolute error (non-negative).
        evaluations: Number of integrand evaluations.
        converged: Whether ``error_estimate <= max(abs_tol, rel_tol |value|)``
            was reached before the subdivision or level limit.
        rule: Rule that produced the estimate: ``"gauss-kronrod-15"``,
            ``"tanh-sinh"``, ``"exp-sinh"`` or ``"sinh-sinh"``.
        intervals: Final subintervals. For Gauss-Kronrod on a finite range
            they are in ``x``; on an infinite range they are in the
            QUADPACK variable ``t`` of ``x = a + (1 - t)/t`` (folded for the
            whole line). Double-exponential rules report the single range.
        levels: Completed double-exponential halving levels (0 for
            Gauss-Kronrod).

    References:
        [REF-PIESSENS-QUADPACK-1983, Chapter 2], [REF-TAKAHASI-MORI-1974].

    """

    value: float
    error_estimate: float
    evaluations: int
    converged: bool
    rule: str
    intervals: tuple[tuple[float, float], ...]
    levels: int


def check_tolerances(abs_tol: float, rel_tol: float) -> tuple[float, float]:
    """Validate an absolute/relative tolerance pair.

    Args:
        abs_tol: Non-negative absolute tolerance.
        rel_tol: Non-negative relative tolerance; at least ``50 eps`` when
            ``abs_tol`` is zero, since smaller targets cannot be certified.

    Returns:
        The validated pair as floats.

    Raises:
        TypeError: If a tolerance is not real.
        ValueError: If a tolerance is negative or the pair is unattainable.

    References:
        [REF-PIESSENS-QUADPACK-1983, Chapter 4] (input checks of QAG).

    """
    checked_abs = require_finite_real(abs_tol, name="abs_tol")
    checked_rel = require_finite_real(rel_tol, name="rel_tol")
    if checked_abs < 0.0 or checked_rel < 0.0:
        raise ValueError("tolerances must be non-negative")
    if checked_abs == 0.0 and checked_rel < ROUNDOFF_FACTOR:
        raise ValueError("rel_tol must be at least 50 eps when abs_tol is zero")
    return checked_abs, checked_rel


def check_limit(value: float) -> float:
    """Validate an integration limit, which may be infinite but not NaN.

    Args:
        value: Candidate limit.

    Returns:
        The limit as a float.

    Raises:
        TypeError: If the limit is not a real number.
        ValueError: If the limit is NaN.

    References:
        Project numerical validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("integration limits must be real numbers")
    limit = float(value)
    if math.isnan(limit):
        raise ValueError("integration limits must not be NaN")
    return limit
