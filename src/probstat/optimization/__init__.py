"""Dependency-free optimization, root finding, and numerical differentiation.

These solvers are deliberately small and transparent: each returns its
evaluation count and a convergence flag instead of silently returning the last
iterate. They serve reliability (design points), inverse problems (least
squares and MAP estimates), and surrogate hyperparameter fitting.

One module per solver family (ADR-0096):

- ``finite_differences``: ``central_gradient``, ``forward_gradient``,
  ``central_jacobian``, ``central_hessian``;
- ``quasi_newton``: ``minimize_bfgs`` and its strong-Wolfe line search;
- ``nelder_mead``: ``minimize_nelder_mead``;
- ``multistart``: ``multi_start``;
- ``nonlinear_least_squares``: ``least_squares`` (Levenberg-Marquardt) and
  ``gauss_newton_covariance``;
- ``root_finding``: ``brent_root``;
- ``stochastic``: ``adam_minimize``;
- ``bounds``: ``BoxTransform``;
- ``results``: ``OptimizationResult``, ``LeastSquaresResult``, ``RootResult``,
  ``MultiStartResult``.

The private ``_common`` holds the callable aliases, machine epsilon, and value
validation that several families share.
"""

from .bounds import BoxTransform
from .finite_differences import (
    central_gradient,
    central_hessian,
    central_jacobian,
    forward_gradient,
)
from .multistart import multi_start
from .nelder_mead import minimize_nelder_mead
from .nonlinear_least_squares import gauss_newton_covariance, least_squares
from .quasi_newton import minimize_bfgs
from .results import (
    LeastSquaresResult,
    MultiStartResult,
    OptimizationResult,
    RootResult,
)
from .root_finding import brent_root
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
