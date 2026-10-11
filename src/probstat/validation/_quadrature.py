"""Compatibility import: tanh-sinh quadrature now lives in ``numerics``."""

from probstat.numerics.double_exponential import DEQuadrature, tanh_sinh

__all__ = ["DEQuadrature", "tanh_sinh"]
