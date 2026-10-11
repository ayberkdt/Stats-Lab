"""Dependency-free deterministic numerical algorithms and diagnostics."""

from .quadrature import QuadratureResult, composite_simpson, composite_trapezoid
from .special import digamma, log_minus_digamma, trigamma

__all__ = [
    "QuadratureResult",
    "composite_simpson",
    "composite_trapezoid",
    "digamma",
    "log_minus_digamma",
    "trigamma",
]
