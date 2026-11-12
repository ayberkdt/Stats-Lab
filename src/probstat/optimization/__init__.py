"""Dependency-free optimization, root finding, and numerical differentiation.

Local minimizers (BFGS with a strong-Wolfe line search, Nelder-Mead,
Levenberg-Marquardt), seeded multi-start, Adam for stochastic objectives,
Brent's bracketing root finder, and central finite differences. The
deterministic solvers return their evaluation count and a convergence flag;
Adam runs a fixed number of iterations and returns its loss trace.

Main entry points:
    ``minimize_bfgs``: smooth unconstrained or box-constrained minimization.
    ``minimize_nelder_mead``: derivative-free minimization for noisy or
        non-smooth objectives.
    ``least_squares``: Levenberg-Marquardt for a vector of residuals.
    ``gauss_newton_covariance``: asymptotic parameter covariance
        ``s^2 (J'J)^-1`` of a least-squares fit.
    ``brent_root``: scalar root inside a sign-changing bracket.
    ``multi_start``: seeded random restarts of a local minimizer in a box.
    ``central_gradient``: central-difference gradient with its call count.

Assumptions:
    The minimizers find a local minimum near the start; nothing certifies a
    global one. BFGS and the finite differences assume a smooth objective,
    ``least_squares`` a residual vector of fixed length, and
    ``gauss_newton_covariance`` a model that is locally linear in the
    parameters with independent residuals of equal variance.

Common pitfalls:
    - Hitting an iteration limit returns a result with
      ``converged=False`` (message ``"iteration limit reached"``) instead
      of raising; check ``result.converged`` before using ``result.x``.
    - ``LeastSquaresResult.cost`` is ``0.5 ||r||^2``, half the residual sum
      of squares; ``gauss_newton_covariance`` estimates the noise variance
      as ``2 cost / (m - n)`` unless ``noise_variance`` is given, and
      raises when ``m <= n``.
    - Seeds default differently: ``multi_start(seed=None)`` draws fresh
      entropy, whereas ``adam_minimize`` defaults to ``seed=0``.
      ``multi_start`` also needs finite bounds to draw its starts.
    - ``central_gradient``, ``central_hessian``, and ``central_jacobian``
      return a ``(value, calls)`` pair, not the derivative alone.
    - ``minimize_nelder_mead`` reports ``gradient_norm`` as ``nan``, and
      ``minimize_bfgs`` tests the gradient norm against an absolute
      ``gradient_tolerance`` (``1e-8``), so rescaling the objective changes
      when it stops.

Guide: docs/prerequisites/optimization.md

References:
    [REF-NOCEDAL-WRIGHT-2006] (BFGS, line searches, least squares),
    [REF-MORE-1978] (Levenberg-Marquardt), [REF-BRENT-1973] (root
    finding), [REF-SEBER-WILD-2003] (Gauss-Newton covariance),
    [REF-KINGMA-BA-2015] (Adam).

"""

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
