"""Dependency-free deterministic numerical algorithms and diagnostics."""

from .autodiff import (
    AutodiffUnsupportedError,
    Dual,
    GradientEvaluator,
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
from .integration import integrate
from .integration_result import IntegrationResult
from .quadrature import QuadratureResult, composite_simpson, composite_trapezoid
from .special import digamma, log_minus_digamma, trigamma

__all__ = [
    "AutodiffUnsupportedError",
    "DEQuadrature",
    "Dual",
    "GradientEvaluator",
    "IntegrationResult",
    "QuadratureResult",
    "admath",
    "composite_simpson",
    "composite_trapezoid",
    "derivative",
    "digamma",
    "exp_sinh",
    "gauss_kronrod",
    "gradient",
    "hessian",
    "integrate",
    "jacobian",
    "log_minus_digamma",
    "sinh_sinh",
    "tanh_sinh",
    "tanh_sinh_interval",
    "trigamma",
    "value_and_gradient",
    "value_gradient_and_hessian",
]
