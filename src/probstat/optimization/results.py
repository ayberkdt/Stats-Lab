"""Result records returned by the optimizers, the least-squares solver, and Brent.

Every record reports its evaluation count and a convergence flag instead of
silently returning the last iterate.
"""

from __future__ import annotations

from dataclasses import dataclass

from probstat.linalg import Matrix, Vector


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Outcome of an unconstrained minimization.

    Args:
        x: Final iterate.
        value: Objective value at ``x``.
        gradient_norm: Euclidean gradient norm at ``x`` (``nan`` if unused).
        iterations: Completed iterations.
        evaluations: Objective evaluations, including finite differences.
        converged: Whether a stopping tolerance was met.
        message: Human-readable termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2].

    """

    x: Vector
    value: float
    gradient_norm: float
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class LeastSquaresResult:
    """Outcome of a nonlinear least-squares fit.

    Args:
        x: Final parameter vector.
        residuals: Residual vector ``r(x)``.
        cost: Half the residual sum of squares ``0.5 ||r||^2``.
        jacobian: Residual Jacobian at ``x``.
        iterations: Completed Levenberg-Marquardt iterations.
        evaluations: Residual-function evaluations.
        converged: Whether a stopping tolerance was met.
        message: Termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 10], [REF-MORE-1978].

    """

    x: Vector
    residuals: Vector
    cost: float
    jacobian: Matrix
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True, slots=True)
class RootResult:
    """Outcome of a bracketing scalar root search.

    Args:
        root: Approximate root.
        value: Function value at ``root``.
        iterations: Completed iterations.
        evaluations: Function evaluations.
        converged: Whether the bracket width or residual tolerance was met.

    References:
        [REF-BRENT-1973, Chapter 4].

    """

    root: float
    value: float
    iterations: int
    evaluations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class MultiStartResult:
    """Outcome of a seeded multi-start minimization.

    Args:
        best: Best run: the lowest value among converged runs, or among all
            runs when none converged.
        runs: Every run in start order.
        starts: Start points in the original coordinates.
        seed: Seed of the stream that drew the random starts.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

    """

    best: OptimizationResult
    runs: tuple[OptimizationResult, ...]
    starts: tuple[Vector, ...]
    seed: int

    @property
    def converged_runs(self) -> int:
        """Return the number of runs that met a stopping tolerance.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return sum(run.converged for run in self.runs)

    @property
    def evaluations(self) -> int:
        """Return the total number of objective evaluations of all runs.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return sum(run.evaluations for run in self.runs)
