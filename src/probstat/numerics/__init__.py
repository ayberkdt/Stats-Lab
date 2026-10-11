"""Dependency-free deterministic numerical algorithms and diagnostics."""

from .double_exponential import DEQuadrature, tanh_sinh
from .quadrature import QuadratureResult, composite_simpson, composite_trapezoid
from .special import digamma, log_minus_digamma, trigamma

__all__ = [
    "DEQuadrature",
    "QuadratureResult",
    "composite_simpson",
    "composite_trapezoid",
    "digamma",
    "log_minus_digamma",
    "tanh_sinh",
    "trigamma",
]
