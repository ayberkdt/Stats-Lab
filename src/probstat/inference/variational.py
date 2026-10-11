"""Automatic differentiation variational inference with Gaussian families.

ADVI (Kucukelbir et al., 2017) fits a Gaussian ``q(x) = N(mu, L L')`` to a
target density ``p(x)`` on **unconstrained** ``R^d`` by maximizing the
evidence lower bound

``ELBO(q) = E_q[log p(x)] + H[q]``,  ``H[q] = sum_i log L_ii + d/2 (1 + log 2 pi)``.

The expectation is differentiated by the reparameterization
``x = mu + L eta`` with ``eta ~ N(0, I)``, which gives unbiased Monte Carlo
gradients from ``grad log p``; the parameters are updated by Adam
(Kingma and Ba, 2015). Two families are offered:

* ``"mean-field"``: ``L = diag(exp(omega))`` (independent coordinates);
* ``"full-rank"``: ``L`` lower triangular with diagonal ``exp(lambda_i)`` and
  free strictly-lower entries (any covariance). Keeping the diagonal positive
  through a log parameterization is a project choice; it spans the same
  Gaussian family as an unconstrained Cholesky factor.

Constrained parameters must be transformed by the caller (for example
``log sigma`` for a scale, or the prior's standard-normal Nataf space used by
:meth:`probstat.inverse.CalibrationProblem.standard_log_posterior`); the
Jacobian of that transformation belongs in ``log p``. The Gaussian family
underestimates posterior variance when the target is skewed or has
correlations the family cannot express (mean-field), so treat ADVI as a fast
approximation and confirm important results with MCMC.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_vector
from probstat.optimization import central_gradient

from .mcmc import LogDensity, MarkovChain, MCMCResult

GradientFunction = Callable[[Vector], Sequence[float]]
_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class VariationalResult:
    """Fitted Gaussian variational approximation ``N(mean, L L')``.

    Args:
        family: ``"mean-field"`` or ``"full-rank"``.
        names: Coordinate names.
        mean: Variational mean ``mu``.
        cholesky_factor: Lower-triangular ``L`` (diagonal for mean-field).
        elbo_trace: ``(iteration, ELBO estimate)`` pairs recorded every
            ``evaluation_interval`` iterations.
        elbo: ELBO estimate of the returned approximation, from the returned
            independent draws when there are more of them than
            ``elbo_samples`` (otherwise the last trace value).
        converged: Whether the relative-change criterion was met.
        iterations: Adam iterations performed.
        gradient_evaluations: Calls of ``grad log p`` (one per Monte Carlo
            gradient sample).
        log_density_evaluations: Calls of ``log p`` (ELBO estimates,
            finite-difference gradients, and the returned draws).
        samples: Independent draws from the fitted ``q``.
        log_density: Target ``log p`` at each draw.
        seed: Seed of the stream that drove the optimization.

    References:
        [REF-KUCUKELBIR-ETAL-2017], [REF-BLEI-KUCUKELBIR-MCAULIFFE-2017].

    """

    family: str
    names: tuple[str, ...]
    mean: Vector
    cholesky_factor: Matrix
    elbo_trace: tuple[tuple[int, float], ...]
    elbo: float
    converged: bool
    iterations: int
    gradient_evaluations: int
    log_density_evaluations: int
    samples: tuple[Vector, ...]
    log_density: tuple[float, ...]
    seed: int

    @property
    def dimension(self) -> int:
        """Return the parameter dimension.

        References:
            [REF-KUCUKELBIR-ETAL-2017]. No equation is evaluated.

        """
        return len(self.mean)

    @property
    def covariance(self) -> Matrix:
        """Return the variational covariance ``L L'``.

        Equation:
            [EQ-VI-ADVI-001].

        References:
            [REF-KUCUKELBIR-ETAL-2017].

        """
        lower = self.cholesky_factor
        size = len(lower)
        return tuple(
            tuple(
                math.fsum(lower[i][k] * lower[j][k] for k in range(min(i, j) + 1))
                for j in range(size)
            )
            for i in range(size)
        )

    @property
    def standard_deviations(self) -> Vector:
        """Return the marginal standard deviations ``sqrt(diag(L L'))``.

        Equation:
            [EQ-VI-ADVI-001].

        References:
            [REF-KUCUKELBIR-ETAL-2017].

        """
        lower = self.cholesky_factor
        return tuple(
            math.sqrt(math.fsum(lower[i][k] ** 2 for k in range(i + 1)))
            for i in range(len(lower))
        )

    def log_pdf(self, x: Sequence[float]) -> float:
        """Return the normalized log-density of the fitted Gaussian ``q``.

        Args:
            x: Point of the parameter dimension.

        Returns:
            ``log q(x)``.

        Raises:
            ValueError: If ``x`` has the wrong length.

        Equation:
            [EQ-VI-ADVI-001], [EQ-DIST-MVN-001].

        References:
            [REF-KUCUKELBIR-ETAL-2017].

        """
        point = as_vector(x, name="x")
        if len(point) != self.dimension:
            raise ValueError("x must match the approximation dimension")
        lower = self.cholesky_factor
        white: list[float] = []
        for i in range(self.dimension):
            partial = math.fsum(lower[i][k] * white[k] for k in range(i))
            white.append((point[i] - self.mean[i] - partial) / lower[i][i])
        log_det = math.fsum(math.log(lower[i][i]) for i in range(self.dimension))
        return -0.5 * (
            math.fsum(w * w for w in white) + self.dimension * _LOG_TWO_PI
        ) - log_det

    def sample(
        self, size: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw independent vectors ``mu + L eta`` from ``q``.

        Args:
            size: Non-negative number of draws.
            seed: Integer seed or stream.

        Returns:
            Draws from the fitted Gaussian.

        Equation:
            [EQ-VI-REPARAM-001].

        References:
            [REF-KUCUKELBIR-ETAL-2017].

        """
        count = require_non_negative_integer(size, name="size")
        generator = resolve_stream(seed).generator()
        return tuple(
            _transform(
                self.mean,
                self.cholesky_factor,
                tuple(generator.gauss(0.0, 1.0) for _ in range(self.dimension)),
            )
            for _ in range(count)
        )

    def as_mcmc_result(self) -> MCMCResult:
        """Wrap the stored independent draws as a one-chain :class:`MCMCResult`.

        The draws are independent, so ``ESS`` is close to their number and
        summaries and quantiles reuse the MCMC tooling. ``R-hat`` from one
        chain of independent draws carries no convergence information about
        the optimization.

        Returns:
            One-chain result labelled ``"ADVI (<family>)"``.

        Raises:
            ValueError: If no draws were stored.

        References:
            [REF-KUCUKELBIR-ETAL-2017], [REF-ROBERT-CASELLA-2004, Chapter 3].
            No equation is evaluated.

        """
        if not self.samples:
            raise ValueError("no variational draws were stored")
        chain = MarkovChain(
            self.samples,
            self.log_density,
            1.0,
            math.nan,
            self.covariance,
            self.log_density_evaluations,
            self.seed,
        )
        return MCMCResult((chain,), self.names, f"ADVI ({self.family})", 0, 1)


def _transform(mean: Vector, lower: Matrix, eta: Vector) -> Vector:
    """Return ``mu + L eta`` for lower-triangular ``L``.

    References:
        [EQ-VI-REPARAM-001], [REF-KUCUKELBIR-ETAL-2017].

    """
    return tuple(
        mean[i] + math.fsum(lower[i][k] * eta[k] for k in range(i + 1))
        for i in range(len(mean))
    )


class _Parameters:
    """Unconstrained variational parameters and their map to ``(mu, L)``.

    Layout: ``mu`` (d), log-diagonal (d), then strictly-lower entries of ``L``
    row by row (full-rank only).

    References:
        [EQ-VI-ADVI-001], [REF-KUCUKELBIR-ETAL-2017].

    """

    __slots__ = ("dimension", "full_rank", "pairs")

    def __init__(self, dimension: int, full_rank: bool) -> None:
        """Record the layout of the strictly-lower entries.

        References:
            Project API policy. No equation is evaluated.

        """
        self.dimension = dimension
        self.full_rank = full_rank
        self.pairs: tuple[tuple[int, int], ...] = (
            tuple((i, j) for i in range(dimension) for j in range(i))
            if full_rank
            else ()
        )

    @property
    def size(self) -> int:
        """Return the number of unconstrained parameters.

        References:
            Project API policy. No equation is evaluated.

        """
        return 2 * self.dimension + len(self.pairs)

    def unpack(self, values: Sequence[float]) -> tuple[Vector, Matrix]:
        """Return ``(mu, L)`` from the parameter vector.

        References:
            [EQ-VI-ADVI-001]. No new equation is introduced.

        """
        d = self.dimension
        mean = tuple(values[:d])
        lower = [[0.0] * d for _ in range(d)]
        for i in range(d):
            lower[i][i] = math.exp(min(values[d + i], 700.0))
        for offset, (i, j) in enumerate(self.pairs):
            lower[i][j] = values[2 * d + offset]
        return mean, tuple(tuple(row) for row in lower)


def _entropy(lower: Matrix) -> float:
    """Return the Gaussian entropy ``sum log L_ii + d/2 (1 + log 2 pi)``.

    References:
        [EQ-VI-ELBO-001], [REF-COVER-THOMAS-2006, Chapter 8].

    """
    size = len(lower)
    return math.fsum(math.log(lower[i][i]) for i in range(size)) + 0.5 * size * (
        1.0 + _LOG_TWO_PI
    )


def advi(
    log_density: LogDensity,
    initial: Sequence[float],
    *,
    family: str = "mean-field",
    gradient: GradientFunction | None = None,
    gradient_samples: int = 1,
    learning_rate: float = 0.02,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    max_iterations: int = 10000,
    evaluation_interval: int = 100,
    elbo_samples: int = 100,
    tolerance: float = 1e-3,
    initial_scale: float = 1.0,
    draws: int = 1000,
    relative_step: float = 1e-6,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> VariationalResult:
    """Fit a Gaussian approximation by reparameterization-gradient ADVI.

    Each Adam iteration draws ``gradient_samples`` vectors ``eta_s ~ N(0, I)``,
    sets ``x_s = mu + L eta_s``, and averages the reparameterization
    gradients ``grad_mu = g_s``, ``grad_{L_ij} = g_{s,i} eta_{s,j}`` (``i > j``),
    ``grad_{lambda_i} = g_{s,i} eta_{s,i} L_ii + 1`` with
    ``g_s = grad log p(x_s)``; the ``+1`` is the entropy gradient. Every
    ``evaluation_interval`` iterations the ELBO is estimated with
    ``elbo_samples`` fixed draws (common random numbers) at the average of
    the iterates over the most recent half of the iterations (Polyak-Ruppert
    tail averaging), which removes most of the Adam step noise from the
    returned approximation. The run
    stops when the mean or the median of the recent relative ELBO changes
    ``|ELBO_k - ELBO_{k-1}| / max(1, |ELBO_k|)`` falls below ``tolerance``
    and the ELBO has not risen by more than ``tolerance`` (relative) over the
    whole buffer (a project heuristic; the buffer holds the last
    ``B = max(2, 0.1 max_iterations / evaluation_interval)`` changes). The
    second condition stops slow climbs along a flat ridge, where every
    single change is small, from being reported as converged; such a run
    ends at ``max_iterations`` with ``converged=False``.

    Args:
        log_density: Target ``log p(x)`` up to a constant on ``R^d``; must be
            finite wherever ``q`` puts appreciable mass.
        initial: Initial mean (``L`` starts at ``initial_scale I``).
        family: ``"mean-field"`` or ``"full-rank"``.
        gradient: ``grad log p(x)``; ``None`` uses central differences
            (``2 d`` density evaluations per gradient sample).
        gradient_samples: Monte Carlo draws per gradient estimate.
        learning_rate: Adam step size ``alpha``.
        beta1: Adam first-moment decay in ``[0, 1)``.
        beta2: Adam second-moment decay in ``[0, 1)``.
        adam_epsilon: Adam denominator offset (positive).
        max_iterations: Iteration cap.
        evaluation_interval: Iterations between ELBO estimates.
        elbo_samples: Draws per ELBO estimate.
        tolerance: Relative ELBO change regarded as converged.
        initial_scale: Initial standard deviation of every coordinate.
        draws: Independent draws from the fitted ``q`` to return.
        relative_step: Finite-difference step relative to ``max(1, |x_i|)``.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.

    Returns:
        Variational mean, Cholesky factor, ELBO trace, convergence flag,
        evaluation counts, and draws.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid settings, or if ``log p`` or its gradient is
            non-finite at a variational draw.

    Equation:
        [EQ-VI-ELBO-001], [EQ-VI-REPARAM-001], [EQ-VI-ADVI-001],
        [EQ-OPT-ADAM-001].

    References:
        [REF-KUCUKELBIR-ETAL-2017], [REF-BLEI-KUCUKELBIR-MCAULIFFE-2017],
        [REF-KINGMA-BA-2015], [REF-POLYAK-JUDITSKY-1992].

    """
    if not callable(log_density):
        raise TypeError("log_density must be callable")
    if gradient is not None and not callable(gradient):
        raise TypeError("gradient must be callable or None")
    if family not in ("mean-field", "full-rank"):
        raise ValueError("family must be 'mean-field' or 'full-rank'")
    start = as_vector(initial, name="initial")
    if not start:
        raise ValueError("initial must be non-empty")
    d = len(start)
    samples_per_step = require_non_negative_integer(
        gradient_samples, name="gradient_samples"
    )
    if samples_per_step == 0:
        raise ValueError("gradient_samples must be positive")
    rate = require_finite_real(learning_rate, name="learning_rate")
    if rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    b1 = require_finite_real(beta1, name="beta1")
    b2 = require_finite_real(beta2, name="beta2")
    if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    offset = require_finite_real(adam_epsilon, name="adam_epsilon")
    if offset <= 0.0:
        raise ValueError("adam_epsilon must be positive")
    cap = require_non_negative_integer(max_iterations, name="max_iterations")
    interval = require_non_negative_integer(
        evaluation_interval, name="evaluation_interval"
    )
    elbo_draws = require_non_negative_integer(elbo_samples, name="elbo_samples")
    if cap == 0 or interval == 0 or elbo_draws == 0:
        raise ValueError(
            "max_iterations, evaluation_interval, and elbo_samples must be positive"
        )
    tol = require_finite_real(tolerance, name="tolerance")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    scale = require_finite_real(initial_scale, name="initial_scale")
    if scale <= 0.0:
        raise ValueError("initial_scale must be positive")
    output_draws = require_non_negative_integer(draws, name="draws")
    h = require_finite_real(relative_step, name="relative_step")
    if h <= 0.0:
        raise ValueError("relative_step must be positive")
    coordinate_names = (
        tuple(f"x{i}" for i in range(d)) if names is None else tuple(names)
    )
    if len(coordinate_names) != d or len(set(coordinate_names)) != d:
        raise ValueError("names must be distinct and match the dimension")

    stream = resolve_stream(seed)
    noise = stream.child("gradient").generator()
    layout = _Parameters(d, family == "full-rank")
    params = list(start) + [math.log(scale)] * d + [0.0] * len(layout.pairs)
    first_moment = [0.0] * layout.size
    second_moment = [0.0] * layout.size
    density_calls = 0
    gradient_calls = 0

    def density(point: Vector) -> float:
        """Evaluate and count ``log p``.

        References:
            [EQ-VI-ELBO-001]. No new equation is introduced.

        """
        nonlocal density_calls
        density_calls += 1
        return float(log_density(point))

    def grad_log_p(point: Vector) -> Vector:
        """Return a finite ``grad log p`` or raise ``ValueError``.

        References:
            [EQ-VI-REPARAM-001], [EQ-OPT-FD-001].

        """
        nonlocal gradient_calls
        gradient_calls += 1
        if gradient is None:
            values, _ = central_gradient(density, point, relative_step=h)
        else:
            values = tuple(float(g) for g in gradient(point))
        if len(values) != d or not all(math.isfinite(g) for g in values):
            raise ValueError("gradient of log p is not finite at a variational draw")
        return values

    elbo_generator = stream.child("elbo").generator()
    elbo_noise = tuple(
        tuple(elbo_generator.gauss(0.0, 1.0) for _ in range(d))
        for _ in range(elbo_draws)
    )

    def estimate_elbo(values: Sequence[float]) -> float:
        """Estimate the ELBO with one fixed set of standard-normal draws.

        Reusing the same ``eta`` (common random numbers) makes the estimate a
        deterministic function of the parameters, so changes between
        evaluations reflect parameter changes rather than resampling noise.

        References:
            [EQ-VI-ELBO-001], [REF-KUCUKELBIR-ETAL-2017].

        """
        mean, lower = layout.unpack(values)
        total: list[float] = []
        for eta in elbo_noise:
            value = density(_transform(mean, lower, eta))
            if not math.isfinite(value):
                raise ValueError("log p is not finite at a variational draw")
            total.append(value)
        return math.fsum(total) / elbo_draws + _entropy(lower)

    trace: list[tuple[int, float]] = []
    changes: list[float] = []
    buffer_size = max(2, int(0.1 * cap / interval))
    window_sum = [0.0] * layout.size
    window_count = 0
    interval_sums: list[list[float]] = []
    interval_counts: list[int] = []
    averaged = list(params)
    previous: float | None = None
    converged = False
    iteration = 0
    while iteration < cap:
        iteration += 1
        mean, lower = layout.unpack(params)
        grad = [0.0] * layout.size
        for _ in range(samples_per_step):
            eta = tuple(noise.gauss(0.0, 1.0) for _ in range(d))
            g = grad_log_p(_transform(mean, lower, eta))
            for i in range(d):
                grad[i] += g[i]
                grad[d + i] += g[i] * eta[i] * lower[i][i]
            for k, (i, j) in enumerate(layout.pairs):
                grad[2 * d + k] += g[i] * eta[j]
        inverse = 1.0 / samples_per_step
        for i in range(layout.size):
            grad[i] *= inverse
        for i in range(d):
            grad[d + i] += 1.0
        correction1 = 1.0 - b1**iteration
        correction2 = 1.0 - b2**iteration
        for i in range(layout.size):
            first_moment[i] = b1 * first_moment[i] + (1.0 - b1) * grad[i]
            second_moment[i] = b2 * second_moment[i] + (1.0 - b2) * grad[i] ** 2
            step = (first_moment[i] / correction1) / (
                math.sqrt(second_moment[i] / correction2) + offset
            )
            params[i] += rate * step
            window_sum[i] += params[i]
        window_count += 1
        if iteration % interval == 0 or iteration == cap:
            interval_sums.append(window_sum)
            interval_counts.append(window_count)
            window_sum = [0.0] * layout.size
            window_count = 0
            half = (len(interval_sums) + 1) // 2
            tail_count = sum(interval_counts[-half:])
            averaged = [
                math.fsum(block[i] for block in interval_sums[-half:]) / tail_count
                for i in range(layout.size)
            ]
            current = estimate_elbo(averaged)
            trace.append((iteration, current))
            if previous is not None:
                changes.append(abs(current - previous) / max(1.0, abs(current)))
                recent = changes[-buffer_size:]
                settled = (
                    math.fsum(recent) / len(recent) < tol or median(recent) < tol
                )
                # A flat ridge can keep per-interval changes below tol while
                # the ELBO still climbs; also require no net rise over the
                # whole buffer before declaring convergence.
                rise = math.inf
                if len(trace) > buffer_size:
                    rise = current - trace[-1 - buffer_size][1]
                drift_free = rise / max(1.0, abs(current)) < tol
                if settled and drift_free:
                    converged = True
                    break
            previous = current
    mean, lower = layout.unpack(averaged)
    final_elbo = trace[-1][1]
    draw_generator = stream.child("draws").generator()
    output: list[Vector] = []
    output_density: list[float] = []
    for _ in range(output_draws):
        eta = tuple(draw_generator.gauss(0.0, 1.0) for _ in range(d))
        point = _transform(mean, lower, eta)
        output.append(point)
        value = density(point)
        if not math.isfinite(value):
            raise ValueError("log p is not finite at a variational draw")
        output_density.append(value)
    if output_draws > elbo_draws:
        final_elbo = math.fsum(output_density) / output_draws + _entropy(lower)
    return VariationalResult(
        family,
        coordinate_names,
        mean,
        lower,
        tuple(trace),
        final_elbo,
        converged,
        iteration,
        gradient_calls,
        density_calls,
        tuple(output),
        tuple(output_density),
        stream.seed,
    )

