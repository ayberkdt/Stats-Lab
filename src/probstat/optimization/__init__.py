"""Dependency-free optimization, root finding, and numerical differentiation."""

from .solvers import (
    BoxTransform,
    LeastSquaresResult,
    MultiStartResult,
    OptimizationResult,
    RootResult,
    brent_root,
    central_gradient,
    central_hessian,
    central_jacobian,
    forward_gradient,
    gauss_newton_covariance,
    least_squares,
    minimize_bfgs,
    minimize_nelder_mead,
    multi_start,
)
from .stochastic import AdamResult, adam_minimize

__all__ = [
    "AdamResult",
    "BoxTransform",
    "MultiStartResult",
    "LeastSquaresResult",
    "OptimizationResult",
    "RootResult",
    "adam_minimize",
    "brent_root",
    "central_gradient",
    "central_hessian",
    "central_jacobian",
    "forward_gradient",
    "gauss_newton_covariance",
    "least_squares",
    "minimize_bfgs",
    "minimize_nelder_mead",
    "multi_start",
]
