"""Stein variational gradient descent: deterministic particle inference.

Stein variational gradient descent (SVGD; Liu and Wang 2016) moves a set of
``n`` particles so that their empirical distribution approaches a target
density ``p`` known up to a constant. Each iteration applies
``x_i <- x_i + eps phi(x_i)`` with ([EQ-INF-SVGD-001])

    ``phi(x) = (1/n) sum_j [ k(x_j, x) grad log p(x_j) + grad_{x_j} k(x_j, x) ]``.

The first term pulls every particle towards high density, weighting the
scores of nearby particles; the second pushes particles apart and stops them
from collapsing onto the mode. ``phi`` is the direction in the unit ball of
the kernel's reproducing-kernel Hilbert space that decreases the
Kullback-Leibler divergence to ``p`` fastest.

Kernel ([EQ-INF-SVGD-002])
    The radial basis function kernel ``k(x, x') = exp(-|x - x'|^2 / h)`` with
    the bandwidth of the source, ``h = med^2 / log n`` where ``med`` is the
    median pairwise distance of the current particles (recomputed at every
    iteration), or a fixed ``h`` supplied by the caller. The median rule is
    a heuristic: it makes ``sum_j k(x_i, x_j)`` about ``n exp(-log n) = 1``
    so that the two terms of ``phi`` are balanced.

What the output is, and is not
    - With one particle ``k = 1`` and the repulsion vanishes: the iteration
      is gradient ascent on ``log p`` and ends at a mode.
    - The particles are a deterministic function of their starting points.
      They are **not independent draws** from ``p``: no Monte Carlo standard
      error applies to their averages, and tail quantiles from a few dozen
      particles are unreliable.
    - With few particles relative to the dimension SVGD **underestimates
      the variance** of the target (the particles concentrate near the
      mode); in one dimension the spread is close to the target's already
      for a few dozen particles, and the shortfall grows with dimension.
      Compare with an MCMC run before trusting the spread.
    - The step size ``eps`` is fixed here. The source uses an adaptive
      (AdaGrad) step, which is not implemented; too large a step makes the
      iteration oscillate or diverge, which the ``converged`` flag and the
      final update norm reveal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_vector
from probstat.statistics import median

from .hamiltonian import GradientFunction, _Target
from .mcmc import LogDensity


@dataclass(frozen=True, slots=True)
class SVGDResult:
    """Final particles of a Stein variational gradient descent run.

    Args:
        particles: Final particle positions.
        initial: Starting positions.
        bandwidth: Kernel bandwidth ``h`` of the last iteration.
        step_size: Step ``eps``.
        iterations: Iterations performed.
        update_norm: Largest Euclidean norm of ``phi(x_i)`` over the
            particles at the final positions.
        tolerance: Convergence tolerance on ``update_norm``.
        converged: Whether ``update_norm <= tolerance`` was reached within
            the iteration budget.
        gradient_evaluations: Score evaluations (one per particle and
            iteration, plus the final check).
        gradient_method: ``"analytic"``, ``"autodiff"``, ``"reverse"``, or
            ``"finite-difference"``.

    References:
        [EQ-INF-SVGD-001], [REF-LIU-WANG-2016].

    """

    particles: tuple[Vector, ...]
    initial: tuple[Vector, ...]
    bandwidth: float
    step_size: float
    iterations: int
    update_norm: float
    tolerance: float
    converged: bool
    gradient_evaluations: int
    gradient_method: str

    def mean(self) -> Vector:
        """Return the particle mean.

        References:
            [EQ-MC-MEAN-001], [REF-LIU-WANG-2016].

        """
        count = len(self.particles)
        return tuple(
            math.fsum(particle[i] for particle in self.particles) / count
            for i in range(len(self.particles[0]))
        )

    def covariance(self) -> Matrix:
        """Return the particle covariance with divisor ``n``.

        The divisor is ``n`` (not ``n - 1``) because the particles are a
        deterministic approximation of the target, not a random sample.
        SVGD with few particles underestimates the target covariance.

        References:
            [EQ-INF-SVGD-001], [REF-LIU-WANG-2016].

        """
        count = len(self.particles)
        dimension = len(self.particles[0])
        center = self.mean()
        return tuple(
            tuple(
                math.fsum(
                    (particle[i] - center[i]) * (particle[j] - center[j])
                    for particle in self.particles
                )
                / count
                for j in range(dimension)
            )
            for i in range(dimension)
        )


def median_bandwidth(particles: Sequence[Sequence[float]]) -> float:
    """Return the median-heuristic bandwidth ``h = med^2 / log n``.

    ``med`` is the median of the ``n (n - 1) / 2`` pairwise Euclidean
    distances. For a single particle, or when the median distance is zero,
    the bandwidth is not defined by the rule and ``1.0`` is returned (with
    one particle the kernel value is one whatever ``h`` is).

    Args:
        particles: ``n`` points of a common positive dimension.

    Returns:
        The bandwidth of ``k(x, x') = exp(-|x - x'|^2 / h)``.

    Raises:
        ValueError: For no particles or inconsistent dimensions.

    Equation:
        [EQ-INF-SVGD-002].

    References:
        [REF-LIU-WANG-2016, Section 5].

    """
    points = _checked_particles(particles)
    count = len(points)
    if count == 1:
        return 1.0
    distances = [
        math.dist(points[i], points[j]) for i in range(count) for j in range(i)
    ]
    middle = median(distances)
    if middle == 0.0:
        return 1.0
    return middle * middle / math.log(count)


def _checked_particles(particles: Sequence[Sequence[float]]) -> tuple[Vector, ...]:
    """Validate a non-empty set of points of one positive dimension.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    points = tuple(as_vector(point, name="particle") for point in particles)
    if not points:
        raise ValueError("at least one particle is required")
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError("particles must share a positive dimension")
    return points


def svgd_direction(
    particles: Sequence[Sequence[float]],
    scores: Sequence[Sequence[float]],
    *,
    bandwidth: float | None = None,
) -> tuple[Vector, ...]:
    """Return the SVGD update direction ``phi(x_i)`` for every particle.

    With ``k(x, x') = exp(-|x - x'|^2 / h)`` and scores
    ``s_j = grad log p(x_j)``,

        ``phi(x_i) = (1/n) sum_j k(x_j, x_i) [ s_j + (2 / h) (x_i - x_j) ]``.

    Args:
        particles: Current positions ``x_1, ..., x_n``.
        scores: ``grad log p`` at each particle.
        bandwidth: Fixed ``h > 0``; ``None`` uses :func:`median_bandwidth`.

    Returns:
        One direction vector per particle.

    Raises:
        ValueError: For mismatched shapes or a non-positive bandwidth.

    Equation:
        [EQ-INF-SVGD-001], [EQ-INF-SVGD-002].

    References:
        [REF-LIU-WANG-2016, Algorithm 1].

    """
    points = _checked_particles(particles)
    gradients = tuple(as_vector(score, name="score") for score in scores)
    dimension = len(points[0])
    if len(gradients) != len(points) or any(len(g) != dimension for g in gradients):
        raise ValueError("scores must have the shape of particles")
    if bandwidth is None:
        h = median_bandwidth(points)
    else:
        h = require_finite_real(bandwidth, name="bandwidth")
        if h <= 0.0:
            raise ValueError("bandwidth must be positive")
    return _direction(points, gradients, h)


def _direction(
    points: tuple[Vector, ...], gradients: tuple[Vector, ...], h: float
) -> tuple[Vector, ...]:
    """Evaluate ``phi`` at every particle for a given bandwidth.

    References:
        [EQ-INF-SVGD-001], [REF-LIU-WANG-2016, Algorithm 1].

    """
    count = len(points)
    dimension = len(points[0])
    totals = [[0.0] * dimension for _ in range(count)]
    for i in range(count):
        row = totals[i]
        for a in range(dimension):
            row[a] += gradients[i][a]
        for j in range(i):
            difference = [points[i][a] - points[j][a] for a in range(dimension)]
            kernel = math.exp(-math.fsum(d * d for d in difference) / h)
            other = totals[j]
            for a in range(dimension):
                repulsion = 2.0 / h * difference[a]
                row[a] += kernel * (gradients[j][a] + repulsion)
                other[a] += kernel * (gradients[i][a] - repulsion)
    return tuple(tuple(value / count for value in row) for row in totals)


def stein_variational_gradient_descent(
    log_density: LogDensity,
    initial: Sequence[Sequence[float]],
    *,
    step_size: float,
    iterations: int,
    gradient: GradientFunction | str = "auto",
    bandwidth: float | None = None,
    tolerance: float = 0.0,
    relative_step: float = 1e-6,
) -> SVGDResult:
    """Approximate a density by particles with Stein variational gradient descent.

    Iterates ``x_i <- x_i + eps phi(x_i)`` for all particles simultaneously,
    with ``phi`` from :func:`svgd_direction`, and stops early when the
    largest ``|phi(x_i)|`` is at most ``tolerance``. The algorithm is
    deterministic given the starting particles; draw them from a broad
    distribution with a seeded generator for a reproducible run.

    The particles are not independent draws, and with few particles in
    several dimensions their spread is smaller than the target's (see the
    module docstring).

    Args:
        log_density: Target ``log p(x)`` up to a constant.
        initial: Starting particles, one point per particle.
        step_size: Fixed step ``eps > 0``.
        iterations: Maximum number of iterations.
        gradient: ``grad log p``: a callable, or one of the options of
            :class:`~probstat.numerics.GradientEvaluator` (``"auto"`` uses
            forward-mode autodiff and falls back to finite differences).
        bandwidth: Fixed kernel bandwidth ``h``; ``None`` applies the median
            heuristic at every iteration.
        tolerance: Stop when ``max_i |phi(x_i)| <= tolerance``; the default
            ``0`` runs all iterations and reports ``converged=False`` unless
            the update vanishes exactly.
        relative_step: Finite-difference step relative to ``max(1, |x_i|)``.

    Returns:
        Final particles with the last bandwidth, the final update norm, the
        convergence flag, and the gradient cost.

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: For invalid particles, step, iteration count, bandwidth,
            or tolerance, or when the score is not finite at a particle.

    Equation:
        [EQ-INF-SVGD-001], [EQ-INF-SVGD-002].

    References:
        [REF-LIU-WANG-2016].

    """
    if not callable(log_density):
        raise TypeError("log_density must be callable")
    points = _checked_particles(initial)
    start = points
    eps = require_finite_real(step_size, name="step_size")
    if eps <= 0.0:
        raise ValueError("step_size must be positive")
    budget = require_non_negative_integer(iterations, name="iterations")
    limit = require_non_negative_finite_real(tolerance, name="tolerance")
    h_step = require_finite_real(relative_step, name="relative_step")
    if h_step <= 0.0:
        raise ValueError("relative_step must be positive")
    fixed: float | None = None
    if bandwidth is not None:
        fixed = require_finite_real(bandwidth, name="bandwidth")
        if fixed <= 0.0:
            raise ValueError("bandwidth must be positive")
    target = _Target(log_density, gradient, h_step)

    def scores(current: tuple[Vector, ...]) -> tuple[Vector, ...]:
        """Return ``grad log p`` at every particle.

        References:
            [EQ-INF-SVGD-001], [EQ-AD-DUAL-001], [EQ-OPT-FD-001].

        """
        values: list[Vector] = []
        for point in current:
            _, score = target.evaluate(point)
            if score is None:
                raise ValueError(
                    "log_density or its gradient is not finite at a particle; "
                    "reduce step_size or check the support"
                )
            values.append(tuple(score))
        return tuple(values)

    done = 0
    while True:
        h = median_bandwidth(points) if fixed is None else fixed
        direction = _direction(points, scores(points), h)
        norm = max(math.hypot(*vector) for vector in direction)
        if norm <= limit or done >= budget:
            break
        points = tuple(
            tuple(x + eps * d for x, d in zip(point, vector, strict=True))
            for point, vector in zip(points, direction, strict=True)
        )
        done += 1
    return SVGDResult(
        points,
        start,
        h,
        eps,
        done,
        norm,
        limit,
        norm <= limit,
        target.gradient_calls,
        target.evaluator.method,
    )
