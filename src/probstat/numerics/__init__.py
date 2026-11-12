"""Dependency-free deterministic numerical algorithms and diagnostics.

Quadrature (fixed composite rules, adaptive Gauss-Kronrod, and
double-exponential rules), forward- and reverse-mode automatic
differentiation, special functions, reference values, and solver
benchmark profiles. Nothing here is random.

Main entry points:
    ``integrate``: one-dimensional integral on a finite or infinite range,
        choosing the rule from the limits.
    ``gauss_kronrod``: adaptive G7-K15 quadrature, the right call for an
        interior kink or peak.
    ``tanh_sinh``: double-exponential rule for ``g(u, 1 - u)`` on a
        sub-interval of ``[0, 1]``, for quantile-scale integrands.
    ``composite_simpson``: fixed-grid Simpson rule with an exact
        evaluation count and no error estimate.
    ``derivative``: exact forward-mode derivative of a scalar function.
    ``reverse_gradient``: reverse-mode gradient whose cost does not grow
        with the dimension.
    ``GradientEvaluator``: value-and-gradient oracle that chooses between
        the differentiation modes.

Assumptions:
    Integrands return finite real numbers wherever the rule places weight,
    and the reported error estimates assume a smooth (double-exponential:
    analytic) integrand between the points the caller knows about; they
    are estimates, not guaranteed bounds. Automatic differentiation sees
    only code written with arithmetic operators, comparisons, ``abs``, and
    ``admath`` functions.

Common pitfalls:
    - ``integrate`` neither raises nor warns when it misses its error
      target; read ``result.converged`` and ``result.error_estimate``.
    - The default target is relative (``abs_tol=0.0``,
      ``rel_tol=1e-10``), so an integral whose value is zero cannot be
      certified: ``integrate(math.sin, 0.0, 2 * math.pi)`` returns
      ``converged=False``; pass ``abs_tol`` for such integrals. The related
      ``tanh_sinh`` names its targets differently (``tolerance=1e-11``,
      ``absolute_tolerance=0.0``).
    - Limits must satisfy ``lower < upper``; reversed limits raise
      ``ValueError`` instead of flipping the sign.
    - ``composite_simpson`` requires an even ``intervals`` and, like
      ``composite_trapezoid``, reports no error estimate.
    - ``math.exp(x)``, ``float(x)``, and other float-only code raise
      ``AutodiffUnsupportedError`` (a ``TypeError``) on a dual number; use
      ``admath.exp`` and friends. At a kink such as ``abs`` at zero the
      derivative is defined as 0.

Guide: docs/prerequisites/numerical-integration.md

References:
    [REF-PIESSENS-QUADPACK-1983] (Gauss-Kronrod and its error estimate),
    [REF-TAKAHASI-MORI-1974] and [REF-MORI-SUGIHARA-2001]
    (double-exponential rules), [REF-DAVIS-RABINOWITZ-1984] (composite
    rules), [REF-GRIEWANK-WALTHER-2008] (automatic differentiation).

"""

from .autodiff import (
    AutodiffUnsupportedError,
    Dual,
    admath,
    derivative,
    gradient,
    hessian,
    jacobian,
    value_and_gradient,
    value_gradient_and_hessian,
)
from .de_transforms import exp_sinh, sinh_sinh, tanh_sinh_interval
from .double_exponential import DEQuadrature, tanh_sinh
from .gauss_kronrod import gauss_kronrod
from .gradient_oracle import GRADIENT_OPTIONS, GradientEvaluator
from .integration import integrate
from .integration_result import IntegrationResult
from .networks import ACTIVATIONS, MLP
from .profiles import (
    DataProfile,
    PerformanceProfile,
    data_profile,
    first_converged_evaluation,
    performance_profile,
    performance_ratios,
)
from .quadrature import QuadratureResult, composite_simpson, composite_trapezoid
from .reference import (
    ReferenceComparison,
    compare_to_reference,
    exact_regularized_beta,
    reference_erf,
    reference_erfc,
    reference_exp,
    reference_log,
    reference_log_gamma,
    reference_normal_cdf,
    reference_normal_quantile,
    reference_pi,
    reference_regularized_beta,
    reference_regularized_gamma_p,
    reference_regularized_gamma_q,
    ulp_error,
)
from .reverse_autodiff import (
    NonDifferentiableError,
    ReverseVariable,
    hessian_vector_product,
    reverse_gradient,
    reverse_hessian,
    reverse_jacobian,
    reverse_value_and_gradient,
    reverse_value_gradient_and_hessian,
    vector_jacobian_product,
)
from .special import digamma, log_minus_digamma, trigamma

__all__ = [
    "AutodiffUnsupportedError",
    "DEQuadrature",
    "Dual",
    "GRADIENT_OPTIONS",
    "GradientEvaluator",
    "IntegrationResult",
    "NonDifferentiableError",
    "QuadratureResult",
    "ReverseVariable",
    "admath",
    "ACTIVATIONS",
    "MLP",
    "composite_simpson",
    "composite_trapezoid",
    "derivative",
    "digamma",
    "exp_sinh",
    "gauss_kronrod",
    "gradient",
    "hessian",
    "hessian_vector_product",
    "integrate",
    "jacobian",
    "log_minus_digamma",
    "reverse_gradient",
    "reverse_hessian",
    "reverse_jacobian",
    "reverse_value_and_gradient",
    "reverse_value_gradient_and_hessian",
    "sinh_sinh",
    "tanh_sinh",
    "tanh_sinh_interval",
    "trigamma",
    "value_and_gradient",
    "value_gradient_and_hessian",
    "vector_jacobian_product",
    "DataProfile",
    "PerformanceProfile",
    "ReferenceComparison",
    "compare_to_reference",
    "data_profile",
    "exact_regularized_beta",
    "first_converged_evaluation",
    "performance_profile",
    "performance_ratios",
    "reference_erf",
    "reference_erfc",
    "reference_exp",
    "reference_log",
    "reference_log_gamma",
    "reference_normal_cdf",
    "reference_normal_quantile",
    "reference_pi",
    "reference_regularized_beta",
    "reference_regularized_gamma_p",
    "reference_regularized_gamma_q",
    "ulp_error",
]
