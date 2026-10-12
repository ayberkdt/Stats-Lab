"""Dependency-free deterministic numerical algorithms and diagnostics."""

from .de_transforms import exp_sinh, sinh_sinh, tanh_sinh_interval
from .double_exponential import DEQuadrature, tanh_sinh
from .gauss_kronrod import gauss_kronrod
from .integration import integrate
from .integration_result import IntegrationResult
from .quadrature import QuadratureResult, composite_simpson, composite_trapezoid
from .special import digamma, log_minus_digamma, trigamma

__all__ = [
    "DEQuadrature",
    "IntegrationResult",
    "QuadratureResult",
    "composite_simpson",
    "composite_trapezoid",
    "digamma",
    "exp_sinh",
    "gauss_kronrod",
    "integrate",
    "log_minus_digamma",
    "sinh_sinh",
    "tanh_sinh",
    "tanh_sinh_interval",
    "trigamma",
]
