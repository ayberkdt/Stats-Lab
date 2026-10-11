"""Dependency-free optimization, root finding, and numerical differentiation."""

from .solvers import (
    LeastSquaresResult,
    OptimizationResult,
    RootResult,
    brent_root,
    central_gradient,
    central_hessian,
    central_jacobian,
    gauss_newton_covariance,
    least_squares,
    minimize_bfgs,
    minimize_nelder_mead,
)

__all__ = [
    "LeastSquaresResult",
    "OptimizationResult",
    "RootResult",
    "brent_root",
    "central_gradient",
    "central_hessian",
    "central_jacobian",
    "gauss_newton_covariance",
    "least_squares",
    "minimize_bfgs",
    "minimize_nelder_mead",
]
