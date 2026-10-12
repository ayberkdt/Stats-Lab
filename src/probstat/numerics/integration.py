"""One entry point that picks an error-controlled rule from the limits."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

from .de_transforms import exp_sinh, sinh_sinh, tanh_sinh_interval
from .gauss_kronrod import gauss_kronrod
from .integration_result import IntegrationResult, check_limit

_METHODS = ("auto", "double-exponential", "gauss-kronrod")


def integrate(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    method: str = "auto",
    abs_tol: float = 0.0,
    rel_tol: float = 1.0e-10,
    limit: int = 200,
    max_level: int = 9,
) -> IntegrationResult:
    """Integrate ``f`` over ``[lower, upper]``, either limit possibly infinite.

    ``method="double-exponential"`` dispatches on the limits: tanh-sinh for a
    finite range, exp-sinh for a half-line, sinh-sinh for the whole line.
    ``method="gauss-kronrod"`` runs adaptive G7-K15 (QAGI substitution on
    infinite ranges). ``method="auto"`` runs the double-exponential rule
    first and, only if it does not converge (typically an interior kink or
    peak), also runs Gauss-Kronrod and returns whichever result has the
    smaller error estimate, with the evaluations of both runs counted.
    ``result.rule`` names the rule whose estimate is returned.

    Args:
        function: Integrand returning finite reals (see the rules for the
            tail-skipping policy of the double-exponential rules).
        lower: Lower limit (``-inf`` allowed).
        upper: Upper limit greater than ``lower`` (``inf`` allowed).
        method: ``"auto"``, ``"double-exponential"`` or ``"gauss-kronrod"``.
        abs_tol: Absolute error target.
        rel_tol: Relative error target.
        limit: Gauss-Kronrod panel limit.
        max_level: Double-exponential level limit.

    Returns:
        The chosen rule's result.

    Raises:
        TypeError: If an argument or integrand value has an invalid type.
        ValueError: If the method, limits, or tolerances are invalid, or the
            integrand is non-finite where it must be finite.

    Equation:
        [EQ-NUM-GK15-001], [EQ-NUM-QAG-001], [EQ-NUM-DE-001].

    References:
        [REF-PIESSENS-QUADPACK-1983], [REF-TAKAHASI-MORI-1974],
        [REF-MORI-SUGIHARA-2001].

    """
    if method not in _METHODS:
        raise ValueError(
            "method must be 'auto', 'double-exponential' or 'gauss-kronrod'"
        )
    a = check_limit(lower)
    b = check_limit(upper)
    if not a < b:
        raise ValueError("upper must be greater than lower")

    def run_gauss_kronrod() -> IntegrationResult:
        """Run adaptive Gauss-Kronrod with the caller's settings.

        References:
            [REF-PIESSENS-QUADPACK-1983].

        """
        return gauss_kronrod(
            function, a, b, abs_tol=abs_tol, rel_tol=rel_tol, limit=limit
        )

    if method == "gauss-kronrod":
        return run_gauss_kronrod()
    if math.isfinite(a) and math.isfinite(b):
        first = tanh_sinh_interval(
            function, a, b, abs_tol=abs_tol, rel_tol=rel_tol, max_level=max_level
        )
    elif math.isfinite(a) or math.isfinite(b):
        first = exp_sinh(
            function, a, b, abs_tol=abs_tol, rel_tol=rel_tol, max_level=max_level
        )
    else:
        first = sinh_sinh(
            function, abs_tol=abs_tol, rel_tol=rel_tol, max_level=max_level
        )
    if method == "double-exponential" or first.converged:
        return first
    second = run_gauss_kronrod()
    chosen = second if second.error_estimate < first.error_estimate else first
    return replace(chosen, evaluations=first.evaluations + second.evaluations)
