"""Dependency-free deterministic numerical algorithms and diagnostics."""

from .quadrature import QuadratureResult, composite_simpson, composite_trapezoid

__all__ = ["QuadratureResult", "composite_simpson", "composite_trapezoid"]
