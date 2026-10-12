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
    peak), also runs Gauss-Kronrod and returns the result whose error
    estimate is the smaller multiple of its own target
    ``max(abs_tol, rel_tol |value|)``, with the evaluations of both runs
    counted. When neither rule converges and their values differ by more
    than the sum of their error estimates, the double-exponential result is
    returned, because Gauss-Kronrod can miss all of the mass of a narrow
    peak far from the origin. If the double-exponential rule raises
    ``ValueError`` (for example because ``upper - lower`` overflows), the
    Gauss-Kronrod result is returned alone. ``result.rule`` names the rule
    whose estimate is returned.

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

    def run_double_exponential() -> IntegrationResult:
        """Run the double-exponential rule that matches the limits.

        References:
            [REF-TAKAHASI-MORI-1974], [REF-MORI-SUGIHARA-2001].

        """
        if math.isfinite(a) and math.isfinite(b):
            return tanh_sinh_interval(
                function, a, b, abs_tol=abs_tol, rel_tol=rel_tol, max_level=max_level
            )
        if math.isfinite(a) or math.isfinite(b):
            return exp_sinh(
                function, a, b, abs_tol=abs_tol, rel_tol=rel_tol, max_level=max_level
            )
        return sinh_sinh(
            function, abs_tol=abs_tol, rel_tol=rel_tol, max_level=max_level
        )

    if method == "double-exponential":
        return run_double_exponential()
    try:
        first = run_double_exponential()
    except ValueError:
        # For example a width that overflows (tanh-sinh needs a finite
        # upper - lower) or a non-finite value at a node Gauss-Kronrod never
        # samples; adaptive Gauss-Kronrod gets its own chance.
        return run_gauss_kronrod()
    if first.converged:
        return first
    second = run_gauss_kronrod()

    def miss(result: IntegrationResult) -> float:
        """Return the error estimate in units of the result's error target.

        References:
            [EQ-NUM-QAG-001].

        """
        target = max(abs_tol, rel_tol * abs(result.value))
        if target > 0.0:
            return result.error_estimate / target
        return 0.0 if result.error_estimate == 0.0 else math.inf

    disagree = abs(first.value - second.value) > (
        first.error_estimate + second.error_estimate
    )
    # A Gauss-Kronrod run whose nodes all missed the mass returns (nearly)
    # zero with a tiny error estimate and calls itself converged; when the
    # double-exponential run saw far more mass, that is not evidence.
    blind = abs(second.value) + second.error_estimate < 1e-12 * abs(first.value)
    evaluations = first.evaluations + second.evaluations
    if disagree or blind:
        # The two rules contradict each other, so at least one error
        # estimate is wrong and neither result may be called converged. The
        # value of the rule with the smaller relative miss is returned (the
        # double-exponential one when Gauss-Kronrod is blind), with the
        # disagreement folded into its error estimate.
        chosen = first if blind or miss(first) <= miss(second) else second
        spread = abs(first.value - second.value)
        return replace(
            chosen,
            converged=False,
            error_estimate=max(chosen.error_estimate, spread),
            evaluations=evaluations,
        )
    chosen = second if miss(second) < miss(first) else first
    return replace(chosen, evaluations=evaluations)
