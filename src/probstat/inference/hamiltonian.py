"""Hamiltonian Monte Carlo and the No-U-Turn Sampler for smooth log-densities.

Both samplers augment the state ``x`` with a Gaussian momentum
``p ~ N(0, M)`` and simulate Hamiltonian dynamics for
``H(x, p) = -log p(x) + p' M^-1 p / 2`` with the leapfrog integrator. Leapfrog
is volume-preserving and time-reversible for *any* position-dependent force,
so a Metropolis correction with the exact ``H`` leaves the target invariant
even when the gradient is a finite-difference approximation; a poor gradient
costs efficiency, never correctness.

Samplers
    :func:`hamiltonian_monte_carlo` runs a fixed number of leapfrog steps per
    iteration (Neal, 2011). :func:`no_u_turn_sampler` is the efficient
    recursive NUTS of Hoffman and Gelman (2014, Algorithms 3 and 6) with
    **slice sampling**: a slice variable ``u ~ Uniform(0, exp(-H_0))`` is
    drawn per iteration, the trajectory is doubled forwards or backwards until
    a U-turn, a divergence, or the maximum tree depth, and the next state is
    chosen by progressive (biased) sampling among the in-slice states. The
    multinomial variant used by current Stan is *not* implemented.

Warm-up adaptation (simplified Stan-style windows)
    Warm-up is split into an initial fast buffer (step size only), a sequence
    of doubling slow windows (step size and inverse metric), and a terminal
    fast buffer (step size only). With ``warmup >= 150`` the buffers are 75 and
    50 iterations and the first slow window has 25 iterations; shorter warm-ups
    use 15 % / 75 % / 10 % with one slow window; below 20 iterations only the
    step size is adapted. At the end of each slow window the inverse metric is
    set to the shrunk sample covariance (or its diagonal) of the window's
    draws, the step size is re-initialized by the heuristic of Hoffman and
    Gelman (Algorithm 4), and dual averaging restarts. After warm-up the
    step size is frozen at the dual-averaging iterate average, so retained
    draws come from one time-homogeneous kernel and the diagnostics in
    :mod:`probstat.inference.mcmc` apply unchanged.

Divergences
    A trajectory is flagged divergent when the energy error
    ``H(x', p') - H(x_0, p_0)`` exceeds ``max_energy_error`` (default 1000,
    the value of Hoffman and Gelman's ``Delta_max``) or when the log-density or
    its gradient becomes non-finite. Divergences are counted per chain; they
    indicate regions of high curvature that the chosen step size cannot
    resolve, so estimates may be biased even when ``R-hat`` looks fine.

Cost
    Each leapfrog step needs one gradient. With ``gradient=None`` the gradient
    is approximated by central differences ([EQ-OPT-FD-001]) at a cost of
    ``2 d`` extra log-density evaluations per step, so a NUTS iteration with
    tree depth ``j`` costs about ``2^j (2 d + 1)`` evaluations. Supply an
    analytic gradient whenever one is available.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    solve_upper,
    symmetrize,
    transpose,
)
from probstat.optimization import central_gradient

from .mcmc import (
    DEFAULT_PROBABILITIES,
    LogDensity,
    MarkovChain,
    MCMCResult,
    ParameterSummary,
)

GradientFunction = Callable[[Vector], Sequence[float]]
MetricKind = Literal["diag", "dense", "unit"]

_DEFAULT_MAX_ENERGY_ERROR = 1000.0
_METRIC_SHRINKAGE_COUNT = 5.0
_METRIC_SHRINKAGE_TARGET = 1e-3


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HamiltonianChainStatistics:
    """Sampler diagnostics of one Hamiltonian chain (post-warm-up unless noted).

    Args:
        step_size: Frozen leapfrog step size used for every retained draw.
        inverse_metric: Adapted inverse mass matrix ``M^-1``.
        metric: ``"diag"``, ``"dense"``, or ``"unit"``.
        divergent_iterations: Post-warm-up iteration indices (zero-based,
            before thinning) whose trajectory diverged.
        warmup_divergences: Divergent trajectories during warm-up.
        acceptance_statistics: Mean Metropolis acceptance probability of each
            post-warm-up iteration (the quantity targeted by dual averaging).
        tree_depths: NUTS tree depth of each post-warm-up iteration (empty for
            static HMC).
        leapfrog_steps: Leapfrog steps of each post-warm-up iteration.
        energies: Hamiltonian ``H`` at the start of each post-warm-up iteration
            (after momentum refresh).
        max_tree_depth: Tree-depth limit (``0`` for static HMC).
        gradient_evaluations: Gradient evaluations including warm-up.
        log_density_evaluations: Log-density evaluations including warm-up and
            finite-difference gradients.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-BETANCOURT-2017], [REF-NEAL-2011].

    """

    step_size: float
    inverse_metric: Matrix
    metric: str
    divergent_iterations: tuple[int, ...]
    warmup_divergences: int
    acceptance_statistics: tuple[float, ...]
    tree_depths: tuple[int, ...]
    leapfrog_steps: tuple[int, ...]
    energies: tuple[float, ...]
    max_tree_depth: int
    gradient_evaluations: int
    log_density_evaluations: int

    @property
    def divergences(self) -> int:
        """Return the number of divergent post-warm-up trajectories.

        References:
            [REF-BETANCOURT-2017]. No equation is evaluated.

        """
        return len(self.divergent_iterations)

    @property
    def mean_acceptance(self) -> float:
        """Return the mean post-warm-up acceptance statistic.

        References:
            [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

        """
        values = self.acceptance_statistics
        return math.fsum(values) / len(values) if values else math.nan

    @property
    def mean_tree_depth(self) -> float:
        """Return the mean NUTS tree depth (``nan`` for static HMC).

        References:
            [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

        """
        if not self.tree_depths:
            return math.nan
        return math.fsum(self.tree_depths) / len(self.tree_depths)

    @property
    def max_tree_depth_hits(self) -> int:
        """Return how many iterations stopped at the tree-depth limit.

        References:
            [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

        """
        if self.max_tree_depth == 0:
            return 0
        return sum(depth >= self.max_tree_depth for depth in self.tree_depths)

    @property
    def mean_leapfrog_steps(self) -> float:
        """Return the mean number of leapfrog steps per post-warm-up iteration.

        References:
            [REF-NEAL-2011]. No equation is evaluated.

        """
        steps = self.leapfrog_steps
        return math.fsum(steps) / len(steps) if steps else math.nan

    def tree_depth_counts(self) -> dict[int, int]:
        """Return a histogram ``{depth: count}`` of NUTS tree depths.

        References:
            [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

        """
        counts: dict[int, int] = {}
        for depth in self.tree_depths:
            counts[depth] = counts.get(depth, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def e_bfmi(self) -> float:
        """Return the energy Bayesian fraction of missing information.

        ``E-BFMI = sum_n (E_n - E_{n-1})^2 / sum_n (E_n - Ebar)^2`` over the
        post-warm-up energies. Values below about 0.3 suggest that momentum
        resampling explores the energy distribution poorly (heavy tails,
        funnels).

        Returns:
            E-BFMI, or ``nan`` with fewer than two energies or zero variance.

        Equation:
            [EQ-HMC-BFMI-001].

        References:
            [REF-BETANCOURT-2017].

        """
        energies = self.energies
        if len(energies) < 2:
            return math.nan
        center = math.fsum(energies) / len(energies)
        denominator = math.fsum((e - center) ** 2 for e in energies)
        if denominator == 0.0:
            return math.nan
        numerator = math.fsum(
            (energies[i] - energies[i - 1]) ** 2 for i in range(1, len(energies))
        )
        return numerator / denominator


@dataclass(frozen=True, slots=True)
class HamiltonianResult:
    """Draws and sampler diagnostics of a multi-chain HMC or NUTS run.

    The draws live in an ordinary :class:`~probstat.inference.MCMCResult`
    (``mcmc``), so ``R-hat``, ESS, MCSE, and summaries apply unchanged. Each
    :class:`MarkovChain` stores the mean acceptance statistic as
    ``acceptance_rate`` and the adapted inverse metric ``M^-1`` as
    ``proposal_covariance``; ``evaluations`` counts log-density evaluations.

    Args:
        mcmc: Multi-chain draws.
        statistics: One diagnostics record per chain.

    Raises:
        ValueError: If the number of statistics differs from the chains.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-NEAL-2011], [REF-VEHTARI-ETAL-2021].

    """

    mcmc: MCMCResult
    statistics: tuple[HamiltonianChainStatistics, ...]

    def __post_init__(self) -> None:
        """Validate that every chain has one statistics record.

        References:
            Project API policy. No equation is evaluated.

        """
        statistics = tuple(self.statistics)
        if len(statistics) != len(self.mcmc.chains):
            raise ValueError("one statistics record per chain is required")
        object.__setattr__(self, "statistics", statistics)

    @property
    def algorithm(self) -> str:
        """Return the sampler label.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.mcmc.algorithm

    @property
    def names(self) -> tuple[str, ...]:
        """Return the coordinate names.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.mcmc.names

    @property
    def divergences(self) -> int:
        """Return the total number of divergent post-warm-up trajectories.

        References:
            [REF-BETANCOURT-2017]. No equation is evaluated.

        """
        return sum(item.divergences for item in self.statistics)

    @property
    def divergences_per_chain(self) -> tuple[int, ...]:
        """Return divergent post-warm-up trajectories of each chain.

        References:
            [REF-BETANCOURT-2017]. No equation is evaluated.

        """
        return tuple(item.divergences for item in self.statistics)

    @property
    def gradient_evaluations(self) -> int:
        """Return gradient evaluations of all chains including warm-up.

        References:
            [REF-NEAL-2011]. No equation is evaluated.

        """
        return sum(item.gradient_evaluations for item in self.statistics)

    @property
    def evaluations(self) -> int:
        """Return log-density evaluations of all chains including warm-up.

        References:
            [REF-NEAL-2011]. No equation is evaluated.

        """
        return sum(item.log_density_evaluations for item in self.statistics)

    @property
    def step_sizes(self) -> tuple[float, ...]:
        """Return the frozen step size of each chain.

        References:
            [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

        """
        return tuple(item.step_size for item in self.statistics)

    def mean(self) -> Vector:
        """Return the pooled posterior mean vector.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7].

        """
        return self.mcmc.mean()

    def covariance(self) -> Matrix:
        """Return the pooled unbiased sample covariance matrix.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 7].

        """
        return self.mcmc.covariance()

    def summary(
        self, probabilities: Sequence[float] = DEFAULT_PROBABILITIES
    ) -> tuple[ParameterSummary, ...]:
        """Summarize every coordinate with moments, quantiles, and diagnostics.

        Args:
            probabilities: Quantile levels in ``[0, 1]``.

        Returns:
            One summary per coordinate.

        Equation:
            [EQ-MCMC-RHAT-002], [EQ-MCMC-ESS-001], [EQ-MCMC-MCSE-001].

        References:
            [REF-VEHTARI-ETAL-2021].

        """
        return self.mcmc.summary(probabilities)


# ---------------------------------------------------------------------------
# Hamiltonian system
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Point:
    """Phase-space point with cached log-density and gradient.

    References:
        [REF-NEAL-2011]. No equation is evaluated.

    """

    x: Vector
    p: Vector
    log_density: float
    gradient: Vector | None


class _Metric:
    """Inverse mass matrix ``M^-1`` in diagonal or dense form.

    References:
        [EQ-HMC-H-001], [REF-NEAL-2011], [REF-BETANCOURT-2017].

    """

    __slots__ = ("diagonal", "dense", "lower")

    def __init__(self, inverse_metric: Matrix, dense: bool) -> None:
        """Store ``M^-1`` and, for a dense metric, its Cholesky factor.

        References:
            [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

        """
        size = len(inverse_metric)
        self.diagonal: Vector = tuple(inverse_metric[i][i] for i in range(size))
        if any(value <= 0.0 or not math.isfinite(value) for value in self.diagonal):
            raise ValueError("inverse metric must have a positive diagonal")
        self.dense: Matrix | None = inverse_metric if dense else None
        self.lower: Matrix | None = cholesky(inverse_metric) if dense else None

    def matrix(self) -> Matrix:
        """Return ``M^-1`` as a full matrix.

        References:
            [REF-NEAL-2011]. No equation is evaluated.

        """
        if self.dense is not None:
            return self.dense
        size = len(self.diagonal)
        return tuple(
            tuple(self.diagonal[i] if i == j else 0.0 for j in range(size))
            for i in range(size)
        )

    def velocity(self, momentum: Vector) -> Vector:
        """Return ``dx/dt = M^-1 p``.

        References:
            [EQ-HMC-H-001], [REF-NEAL-2011].

        """
        if self.dense is None:
            return tuple(m * p for m, p in zip(self.diagonal, momentum, strict=True))
        return tuple(
            math.fsum(a * b for a, b in zip(row, momentum, strict=True))
            for row in self.dense
        )

    def kinetic(self, momentum: Vector) -> float:
        """Return the kinetic energy ``K(p) = p' M^-1 p / 2``.

        References:
            [EQ-HMC-H-001], [REF-NEAL-2011].

        """
        velocity = self.velocity(momentum)
        return 0.5 * math.fsum(a * b for a, b in zip(velocity, momentum, strict=True))

    def sample_momentum(self, generator: Random) -> Vector:
        """Draw ``p ~ N(0, M)``.

        For ``M^-1 = L L'`` the draw is ``p = L^-T z`` with ``z ~ N(0, I)``,
        whose covariance is ``L^-T L^-1 = M``.

        References:
            [EQ-HMC-H-001], [REF-NEAL-2011].

        """
        z = tuple(generator.gauss(0.0, 1.0) for _ in self.diagonal)
        if self.lower is None:
            return tuple(
                v / math.sqrt(m) for v, m in zip(z, self.diagonal, strict=True)
            )
        return solve_upper(transpose(self.lower), z)


class _Target:
    """Log-density and gradient with evaluation counting and failure capture.

    Non-finite values, ``ArithmeticError``, and ``ValueError`` raised by the
    user callables at a *trajectory* point are mapped to ``log p = -inf`` so
    that the sampler can flag a divergence instead of crashing.

    References:
        [REF-HOFFMAN-GELMAN-2014], [EQ-OPT-FD-001].

    """

    __slots__ = (
        "log_density_function",
        "gradient_function",
        "relative_step",
        "density_calls",
        "gradient_calls",
    )

    def __init__(
        self,
        log_density: LogDensity,
        gradient: GradientFunction | None,
        relative_step: float,
    ) -> None:
        """Store the callables and zero the counters.

        References:
            Project API policy. No equation is evaluated.

        """
        self.log_density_function = log_density
        self.gradient_function = gradient
        self.relative_step = relative_step
        self.density_calls = 0
        self.gradient_calls = 0

    def _density(self, x: Vector) -> float:
        """Evaluate and count one log-density value.

        References:
            [REF-NEAL-2011]. No equation is evaluated.

        """
        self.density_calls += 1
        return float(self.log_density_function(x))

    def evaluate(self, x: Vector) -> tuple[float, Vector | None]:
        """Return ``(log p(x), grad log p(x))`` or ``(-inf, None)`` on failure.

        References:
            [EQ-OPT-FD-001], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        try:
            value = self._density(x)
            if not math.isfinite(value):
                return -math.inf, None
            self.gradient_calls += 1
            if self.gradient_function is None:
                gradient, _ = central_gradient(
                    self._density, x, relative_step=self.relative_step
                )
            else:
                gradient = tuple(float(g) for g in self.gradient_function(x))
                if len(gradient) != len(x):
                    raise ValueError("gradient must match the state dimension")
        except (ArithmeticError, ValueError):
            return -math.inf, None
        if not all(math.isfinite(g) for g in gradient):
            return -math.inf, None
        return value, gradient

    def start(self, x: Vector) -> tuple[float, Vector]:
        """Evaluate a starting point, raising if it is not usable.

        References:
            Project validation policy. No equation is evaluated.

        """
        value = self._density(x)
        if math.isnan(value):
            raise ValueError("log_density returned NaN at the initial point")
        if not math.isfinite(value):
            raise ValueError("initial point must have a finite log-density")
        self.density_calls -= 1
        evaluated, gradient = self.evaluate(x)
        if gradient is None:
            raise ValueError("gradient is not finite at the initial point")
        return evaluated, gradient


def _leapfrog_step(
    target: _Target, metric: _Metric, point: _Point, step: float
) -> _Point:
    """Advance one leapfrog step ``(x, p) -> (x', p')``.

    References:
        [EQ-HMC-LEAPFROG-001], [REF-NEAL-2011].

    """
    gradient = point.gradient
    if gradient is None:
        raise ValueError("cannot integrate from a point without a gradient")
    half = 0.5 * step
    p_half = tuple(p + half * g for p, g in zip(point.p, gradient, strict=True))
    velocity = metric.velocity(p_half)
    x_new = tuple(x + step * v for x, v in zip(point.x, velocity, strict=True))
    value, gradient_new = target.evaluate(x_new)
    if gradient_new is None:
        return _Point(x_new, p_half, -math.inf, None)
    p_new = tuple(p + half * g for p, g in zip(p_half, gradient_new, strict=True))
    return _Point(x_new, p_new, value, gradient_new)


def _joint(metric: _Metric, point: _Point) -> float:
    """Return ``-H(x, p) = log p(x) - K(p)`` (``-inf`` outside the support).

    References:
        [EQ-HMC-H-001], [REF-NEAL-2011].

    """
    if point.gradient is None or not math.isfinite(point.log_density):
        return -math.inf
    value = point.log_density - metric.kinetic(point.p)
    return value if math.isfinite(value) else -math.inf


# ---------------------------------------------------------------------------
# Public leapfrog utilities
# ---------------------------------------------------------------------------


def _optional_metric(
    inverse_metric: Sequence[Sequence[float]] | None, dimension: int
) -> _Metric:
    """Build a dense metric from an optional ``M^-1`` (identity by default).

    References:
        [REF-NEAL-2011]. No equation is evaluated.

    """
    if inverse_metric is None:
        return _Metric(_identity(dimension), dense=False)
    matrix = symmetrize(inverse_metric, name="inverse_metric")
    if len(matrix) != dimension:
        raise ValueError("inverse_metric must match the state dimension")
    try:
        return _Metric(matrix, dense=True)
    except ValueError as error:
        raise ValueError("inverse_metric must be positive definite") from error


def hamiltonian_energy(
    log_density: LogDensity,
    position: Sequence[float],
    momentum: Sequence[float],
    *,
    inverse_metric: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return the Hamiltonian ``H(x, p) = -log p(x) + p' M^-1 p / 2``.

    Args:
        log_density: Target ``log p(x)`` up to a constant.
        position: State ``x``.
        momentum: Momentum ``p`` of the same length.
        inverse_metric: Symmetric positive-definite ``M^-1``; identity by
            default.

    Returns:
        Total energy (``+inf`` where ``log p = -inf``).

    Raises:
        ValueError: If lengths differ or the metric is invalid.

    Equation:
        [EQ-HMC-H-001].

    References:
        [REF-NEAL-2011], [REF-BETANCOURT-2017].

    """
    x = as_vector(position, name="position")
    p = as_vector(momentum, name="momentum")
    if len(x) != len(p) or not x:
        raise ValueError("position and momentum must share a positive length")
    metric = _optional_metric(inverse_metric, len(x))
    value = float(log_density(x))
    if value == -math.inf:
        return math.inf
    return -value + metric.kinetic(p)


def leapfrog(
    gradient: GradientFunction,
    position: Sequence[float],
    momentum: Sequence[float],
    *,
    step_size: float,
    steps: int,
    inverse_metric: Sequence[Sequence[float]] | None = None,
) -> tuple[Vector, Vector]:
    """Integrate Hamiltonian dynamics with the leapfrog (Stormer-Verlet) scheme.

    Each step is ``p <- p + (eps/2) grad log p(x)``, ``x <- x + eps M^-1 p``,
    ``p <- p + (eps/2) grad log p(x)``. The map is symplectic and reversible;
    its global energy error over a fixed integration time is ``O(eps^2)``.

    Args:
        gradient: ``grad log p(x)``.
        position: Initial ``x``.
        momentum: Initial ``p``.
        step_size: Step ``eps`` (negative integrates backwards in time).
        steps: Non-negative number of steps.
        inverse_metric: Symmetric positive-definite ``M^-1``; identity by
            default.

    Returns:
        Final ``(x, p)``.

    Raises:
        ValueError: For mismatched lengths, a non-finite step, an invalid
            metric, or a non-finite gradient.

    Equation:
        [EQ-HMC-LEAPFROG-001].

    References:
        [REF-NEAL-2011], [REF-BETANCOURT-2017].

    """
    x = as_vector(position, name="position")
    p = as_vector(momentum, name="momentum")
    if len(x) != len(p) or not x:
        raise ValueError("position and momentum must share a positive length")
    eps = require_finite_real(step_size, name="step_size")
    count = require_non_negative_integer(steps, name="steps")
    metric = _optional_metric(inverse_metric, len(x))

    def strict_density(point: Vector) -> float:
        """Return zero; the utility integrates without density checks.

        References:
            [EQ-HMC-LEAPFROG-001]. No equation is evaluated.

        """
        return 0.0

    target = _Target(strict_density, gradient, 1e-6)
    _, start_gradient = target.evaluate(x)
    if start_gradient is None:
        raise ValueError("gradient is not finite at the initial position")
    point = _Point(x, p, 0.0, start_gradient)
    for _ in range(count):
        point = _leapfrog_step(target, metric, point, eps)
        if point.gradient is None:
            raise ValueError("gradient became non-finite during integration")
    return point.x, point.p


# ---------------------------------------------------------------------------
# Step-size and metric adaptation
# ---------------------------------------------------------------------------


class _DualAveraging:
    """Nesterov dual averaging of ``log eps`` (Hoffman-Gelman Algorithm 5).

    References:
        [EQ-HMC-DA-001], [REF-NESTEROV-2009], [REF-HOFFMAN-GELMAN-2014].

    """

    __slots__ = (
        "mu",
        "target",
        "gamma",
        "t0",
        "kappa",
        "iteration",
        "h_bar",
        "log_eps",
        "log_eps_bar",
    )

    def __init__(
        self,
        step_size: float,
        target: float,
        *,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ) -> None:
        """Restart averaging with ``mu = log(10 eps_0)``.

        References:
            [EQ-HMC-DA-001], [REF-HOFFMAN-GELMAN-2014].

        """
        self.mu = math.log(10.0 * step_size)
        self.target = target
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self.iteration = 0
        self.h_bar = 0.0
        self.log_eps = math.log(step_size)
        self.log_eps_bar = 0.0

    def update(self, acceptance: float) -> float:
        """Update with one acceptance statistic and return the next ``eps``.

        References:
            [EQ-HMC-DA-001], [REF-HOFFMAN-GELMAN-2014].

        """
        self.iteration += 1
        m = self.iteration
        weight = 1.0 / (m + self.t0)
        self.h_bar = (1.0 - weight) * self.h_bar + weight * (self.target - acceptance)
        self.log_eps = self.mu - math.sqrt(m) / self.gamma * self.h_bar
        eta = m ** (-self.kappa)
        self.log_eps_bar = eta * self.log_eps + (1.0 - eta) * self.log_eps_bar
        self.log_eps = min(self.log_eps, 700.0)
        return math.exp(self.log_eps)

    def final(self) -> float:
        """Return ``exp(log eps_bar)``, the step size frozen after warm-up.

        References:
            [EQ-HMC-DA-001], [REF-HOFFMAN-GELMAN-2014].

        """
        if self.iteration == 0:
            return math.exp(self.log_eps)
        return math.exp(min(self.log_eps_bar, 700.0))


def adaptation_windows(warmup: int) -> tuple[int, int, tuple[int, ...]]:
    """Return the simplified Stan-style warm-up schedule.

    Args:
        warmup: Non-negative number of warm-up iterations.

    Returns:
        ``(initial_buffer, terminal_buffer, window_ends)``: iterations
        ``[0, initial_buffer)`` adapt only the step size; each slow window
        ends at an index in ``window_ends`` (exclusive), where the inverse
        metric is updated; the last ``terminal_buffer`` iterations adapt only
        the step size. ``window_ends`` is empty when ``warmup < 20``.

    Raises:
        ValueError: If ``warmup`` is negative.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-BETANCOURT-2017]. The window sizes are
        a documented project simplification; no equation is evaluated.

    """
    total = require_non_negative_integer(warmup, name="warmup")
    if total < 20:
        return total, 0, ()
    if total >= 150:
        initial, terminal, base = 75, 50, 25
    else:
        initial = int(0.15 * total)
        terminal = int(0.10 * total)
        base = total - initial - terminal
    stop = total - terminal
    ends: list[int] = []
    start, size = initial, base
    while start < stop:
        end = start + size
        if end + 2 * size > stop:
            end = stop
        ends.append(end)
        start = end
        size *= 2
    return initial, terminal, tuple(ends)


def _identity(dimension: int) -> Matrix:
    """Return the identity matrix.

    References:
        [REF-GOLUB-VANLOAN-2013]. No equation is evaluated.

    """
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(dimension))
        for i in range(dimension)
    )


def _window_metric(points: Sequence[Vector], dense: bool) -> Matrix | None:
    """Return the shrunk window covariance used as the new ``M^-1``.

    ``C* = n/(n+5) C + 1e-3 * 5/(n+5) I`` with the unbiased sample covariance
    ``C`` of the ``n`` window draws (diagonal only for a diagonal metric).

    References:
        [EQ-HMC-METRIC-001], [REF-BETANCOURT-2017].

    """
    count = len(points)
    if count < 3:
        return None
    dimension = len(points[0])
    center = [math.fsum(point[i] for point in points) / count for i in range(dimension)]
    weight = count / (count + _METRIC_SHRINKAGE_COUNT)
    shrink = _METRIC_SHRINKAGE_TARGET * _METRIC_SHRINKAGE_COUNT / (
        count + _METRIC_SHRINKAGE_COUNT
    )
    rows: list[tuple[float, ...]] = []
    for i in range(dimension):
        row: list[float] = []
        for j in range(dimension):
            if i != j and not dense:
                row.append(0.0)
                continue
            covariance = math.fsum(
                (point[i] - center[i]) * (point[j] - center[j]) for point in points
            ) / (count - 1)
            row.append(weight * covariance + (shrink if i == j else 0.0))
        rows.append(tuple(row))
    matrix = tuple(rows)
    try:
        cholesky(matrix)
    except ValueError:
        return None
    return matrix


def _find_reasonable_step_size(
    target: _Target,
    metric: _Metric,
    point: _Point,
    step_size: float,
    generator: Random,
) -> float:
    """Double or halve ``eps`` until one leapfrog step has acceptance near 1/2.

    References:
        [REF-HOFFMAN-GELMAN-2014] (Algorithm 4). No new equation is
        introduced.

    """
    eps = step_size
    momentum = metric.sample_momentum(generator)
    start = _Point(point.x, momentum, point.log_density, point.gradient)
    joint0 = _joint(metric, start)

    def log_ratio(trial: float) -> float:
        """Return ``-H(after one step) + H(start)``.

        References:
            [EQ-HMC-H-001]. No new equation is introduced.

        """
        moved = _leapfrog_step(target, metric, start, trial)
        value = _joint(metric, moved) - joint0
        return value if not math.isnan(value) else -math.inf

    ratio = log_ratio(eps)
    direction = 1.0 if ratio > math.log(0.5) else -1.0
    for _ in range(100):
        if not direction * ratio > -direction * math.log(2.0):
            break
        candidate = eps * 2.0**direction
        if not 1e-10 <= candidate <= 1e10:
            break
        eps = candidate
        ratio = log_ratio(eps)
    return eps


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Transition:
    """Outcome of one sampler iteration.

    References:
        [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

    """

    point: _Point
    acceptance: float
    energy: float
    steps: int
    depth: int
    divergent: bool


def _hmc_transition(
    target: _Target,
    metric: _Metric,
    current: _Point,
    step_size: float,
    steps: int,
    max_energy_error: float,
    generator: Random,
) -> _Transition:
    """Run one static-HMC iteration with a Metropolis correction.

    References:
        [EQ-HMC-H-001], [EQ-HMC-LEAPFROG-001], [EQ-HMC-ACCEPT-001],
        [REF-NEAL-2011].

    """
    momentum = metric.sample_momentum(generator)
    point = _Point(current.x, momentum, current.log_density, current.gradient)
    joint0 = _joint(metric, point)
    divergent = False
    taken = 0
    for _ in range(steps):
        point = _leapfrog_step(target, metric, point, step_size)
        taken += 1
        joint = _joint(metric, point)
        if joint == -math.inf or joint0 - joint > max_energy_error:
            divergent = True
            break
    if divergent:
        return _Transition(current, 0.0, -joint0, taken, 0, True)
    log_alpha = _joint(metric, point) - joint0
    acceptance = 1.0 if log_alpha >= 0.0 else math.exp(log_alpha)
    if math.log(1.0 - generator.random()) < log_alpha:
        return _Transition(point, acceptance, -joint0, taken, 0, False)
    return _Transition(current, acceptance, -joint0, taken, 0, False)


@dataclass(slots=True)
class _Tree:
    """Subtree summary of the recursive NUTS doubling.

    References:
        [REF-HOFFMAN-GELMAN-2014] (Algorithms 3 and 6). No equation is
        evaluated.

    """

    minus: _Point
    plus: _Point
    proposal: _Point
    count: int
    keep_going: bool
    alpha: float
    alpha_count: int
    divergent: bool
    steps: int


def _no_u_turn(metric: _Metric, minus: _Point, plus: _Point) -> bool:
    """Return whether the trajectory ends still move apart.

    The criterion is ``(x+ - x-) . M^-1 p- >= 0`` and
    ``(x+ - x-) . M^-1 p+ >= 0``.

    References:
        [EQ-NUTS-UTURN-001], [REF-HOFFMAN-GELMAN-2014].

    """
    span = tuple(a - b for a, b in zip(plus.x, minus.x, strict=True))
    back = metric.velocity(minus.p)
    front = metric.velocity(plus.p)
    return (
        math.fsum(s * v for s, v in zip(span, back, strict=True)) >= 0.0
        and math.fsum(s * v for s, v in zip(span, front, strict=True)) >= 0.0
    )


def _nuts_transition(
    target: _Target,
    metric: _Metric,
    current: _Point,
    step_size: float,
    max_depth: int,
    max_energy_error: float,
    generator: Random,
) -> _Transition:
    """Run one efficient NUTS iteration with slice sampling.

    References:
        [EQ-NUTS-SLICE-001], [EQ-NUTS-UTURN-001], [REF-HOFFMAN-GELMAN-2014]
        (Algorithms 3 and 6).

    """
    momentum = metric.sample_momentum(generator)
    start = _Point(current.x, momentum, current.log_density, current.gradient)
    joint0 = _joint(metric, start)
    log_slice = joint0 + math.log(1.0 - generator.random())

    def build(point: _Point, direction: int, depth: int) -> _Tree:
        """Build a balanced subtree of ``2^depth`` leapfrog steps.

        References:
            [REF-HOFFMAN-GELMAN-2014] (Algorithm 6, BuildTree).

        """
        if depth == 0:
            moved = _leapfrog_step(target, metric, point, direction * step_size)
            joint = _joint(metric, moved)
            count = 1 if log_slice <= joint else 0
            keep = joint > log_slice - max_energy_error
            if joint == -math.inf:
                alpha = 0.0
            else:
                difference = joint - joint0
                alpha = 1.0 if difference >= 0.0 else math.exp(difference)
            return _Tree(moved, moved, moved, count, keep, alpha, 1, not keep, 1)
        first = build(point, direction, depth - 1)
        if not first.keep_going:
            return first
        if direction < 0:
            second = build(first.minus, direction, depth - 1)
            minus, plus = second.minus, first.plus
        else:
            second = build(first.plus, direction, depth - 1)
            minus, plus = first.minus, second.plus
        total = first.count + second.count
        proposal = first.proposal
        if total > 0 and generator.random() < second.count / total:
            proposal = second.proposal
        keep = second.keep_going and _no_u_turn(metric, minus, plus)
        return _Tree(
            minus,
            plus,
            proposal,
            total,
            keep,
            first.alpha + second.alpha,
            first.alpha_count + second.alpha_count,
            first.divergent or second.divergent,
            first.steps + second.steps,
        )

    minus = plus = start
    chosen = current
    count = 1
    keep_going = True
    depth = 0
    alpha_sum = 0.0
    alpha_count = 0
    steps = 0
    divergent = False
    while keep_going and depth < max_depth:
        direction = -1 if generator.random() < 0.5 else 1
        if direction < 0:
            tree = build(minus, direction, depth)
            minus = tree.minus
        else:
            tree = build(plus, direction, depth)
            plus = tree.plus
        alpha_sum += tree.alpha
        alpha_count += tree.alpha_count
        steps += tree.steps
        divergent = divergent or tree.divergent
        if tree.keep_going and generator.random() < tree.count / count:
            chosen = tree.proposal
        count += tree.count
        keep_going = tree.keep_going and _no_u_turn(metric, minus, plus)
        depth += 1
    acceptance = alpha_sum / alpha_count if alpha_count else 0.0
    return _Transition(chosen, acceptance, -joint0, steps, depth, divergent)


# ---------------------------------------------------------------------------
# Chain driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Settings:
    """Validated sampler settings shared by all chains.

    References:
        Project API policy. No equation is evaluated.

    """

    algorithm: Literal["hmc", "nuts"]
    draws: int
    warmup: int
    thin: int
    metric: MetricKind
    inverse_metric: Matrix
    adapt_metric: bool
    step_size: float | None
    adapt_step_size: bool
    target_acceptance: float
    leapfrog_steps: int
    step_size_jitter: float
    max_tree_depth: int
    max_energy_error: float


def _run_hamiltonian_chain(
    target: _Target,
    start: Vector,
    settings: _Settings,
    stream: RandomStream,
) -> tuple[MarkovChain, HamiltonianChainStatistics]:
    """Run warm-up and sampling for one HMC or NUTS chain.

    References:
        [EQ-HMC-DA-001], [EQ-HMC-METRIC-001], [REF-HOFFMAN-GELMAN-2014],
        [REF-NEAL-2011].

    """
    generator = stream.generator()
    dense = settings.metric == "dense"
    metric = _Metric(settings.inverse_metric, dense=dense)
    log_value, gradient = target.start(start)
    current = _Point(start, tuple(0.0 for _ in start), log_value, gradient)
    if settings.step_size is not None:
        step = settings.step_size
    else:
        step = _find_reasonable_step_size(target, metric, current, 1.0, generator)
    averaging = _DualAveraging(step, settings.target_acceptance)
    initial, _, window_ends = adaptation_windows(settings.warmup)
    adapt_metric = settings.adapt_metric and settings.metric != "unit"
    window_start = initial
    window_points: list[Vector] = []
    remaining_ends = list(window_ends) if adapt_metric else []

    def transition(point: _Point, eps: float) -> _Transition:
        """Run one iteration of the configured sampler.

        References:
            [REF-HOFFMAN-GELMAN-2014], [REF-NEAL-2011].

        """
        if settings.algorithm == "nuts":
            return _nuts_transition(
                target,
                metric,
                point,
                eps,
                settings.max_tree_depth,
                settings.max_energy_error,
                generator,
            )
        if settings.step_size_jitter > 0.0:
            eps *= 1.0 + settings.step_size_jitter * (2.0 * generator.random() - 1.0)
        return _hmc_transition(
            target,
            metric,
            point,
            eps,
            settings.leapfrog_steps,
            settings.max_energy_error,
            generator,
        )

    warm_acceptance: list[float] = []
    warm_divergences = 0
    for iteration in range(settings.warmup):
        outcome = transition(current, step)
        current = outcome.point
        warm_acceptance.append(outcome.acceptance)
        warm_divergences += outcome.divergent
        if settings.adapt_step_size:
            step = averaging.update(outcome.acceptance)
        if remaining_ends and window_start <= iteration < remaining_ends[0]:
            window_points.append(current.x)
        if remaining_ends and iteration + 1 == remaining_ends[0]:
            remaining_ends.pop(0)
            updated = _window_metric(window_points, dense)
            if updated is not None:
                metric = _Metric(updated, dense=dense)
            window_points = []
            window_start = iteration + 1
            if settings.adapt_step_size:
                step = _find_reasonable_step_size(
                    target, metric, current, step, generator
                )
                averaging = _DualAveraging(step, settings.target_acceptance)
    if settings.adapt_step_size and settings.warmup > 0:
        step = averaging.final()

    samples: list[Vector] = []
    trace: list[float] = []
    acceptance: list[float] = []
    depths: list[int] = []
    steps_taken: list[int] = []
    energies: list[float] = []
    divergent_at: list[int] = []
    for iteration in range(settings.draws * settings.thin):
        outcome = transition(current, step)
        current = outcome.point
        acceptance.append(outcome.acceptance)
        steps_taken.append(outcome.steps)
        energies.append(outcome.energy)
        if settings.algorithm == "nuts":
            depths.append(outcome.depth)
        if outcome.divergent:
            divergent_at.append(iteration)
        if (iteration + 1) % settings.thin == 0:
            samples.append(current.x)
            trace.append(current.log_density)
    mean_acceptance = math.fsum(acceptance) / len(acceptance)
    warm_rate = (
        math.fsum(warm_acceptance) / len(warm_acceptance)
        if warm_acceptance
        else math.nan
    )
    chain = MarkovChain(
        tuple(samples),
        tuple(trace),
        mean_acceptance,
        warm_rate,
        metric.matrix(),
        target.density_calls,
        stream.seed,
    )
    statistics = HamiltonianChainStatistics(
        step,
        metric.matrix(),
        settings.metric,
        tuple(divergent_at),
        warm_divergences,
        tuple(acceptance),
        tuple(depths),
        tuple(steps_taken),
        tuple(energies),
        settings.max_tree_depth if settings.algorithm == "nuts" else 0,
        target.gradient_calls,
        target.density_calls,
    )
    return chain, statistics


def _initial_points(
    initial: Sequence[float] | Sequence[Sequence[float]], chains: int
) -> tuple[Vector, ...]:
    """Normalize one shared start or one start per chain.

    References:
        [REF-GELMAN-RUBIN-1992]. No equation is evaluated.

    """
    items: list[object] = list(initial)
    if not items:
        raise ValueError("initial must be non-empty")
    nested = [
        item
        for item in items
        if isinstance(item, Sequence) and not isinstance(item, str)
    ]
    if len(nested) == len(items):
        points = tuple(as_vector(item, name="initial point") for item in nested)
        if len(points) != chains:
            raise ValueError("provide one initial point per chain")
        dimension = len(points[0])
        if dimension == 0 or any(len(point) != dimension for point in points):
            raise ValueError("initial points must share a positive dimension")
        return points
    point = as_vector(items, name="initial")
    return tuple(point for _ in range(chains))


def _sample(
    algorithm: Literal["hmc", "nuts"],
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    gradient: GradientFunction | None,
    draws: int,
    warmup: int,
    chains: int,
    thin: int,
    metric: str,
    initial_inverse_metric: Sequence[Sequence[float]] | None,
    adapt_metric: bool,
    step_size: float | None,
    adapt_step_size: bool,
    target_acceptance: float,
    leapfrog_steps: int,
    step_size_jitter: float,
    max_tree_depth: int,
    max_energy_error: float,
    relative_step: float,
    seed: int | RandomStream | None,
    names: Sequence[str] | None,
) -> HamiltonianResult:
    """Validate arguments and run every chain of an HMC or NUTS sampler.

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-NEAL-2011]. No equation is evaluated.

    """
    if not callable(log_density):
        raise TypeError("log_density must be callable")
    if gradient is not None and not callable(gradient):
        raise TypeError("gradient must be callable or None")
    checked_draws = require_non_negative_integer(draws, name="draws")
    checked_warmup = require_non_negative_integer(warmup, name="warmup")
    checked_chains = require_non_negative_integer(chains, name="chains")
    checked_thin = require_non_negative_integer(thin, name="thin")
    if checked_draws == 0:
        raise ValueError("draws must be positive")
    if checked_chains == 0:
        raise ValueError("chains must be positive")
    if checked_thin == 0:
        raise ValueError("thin must be positive")
    if metric not in ("diag", "dense", "unit"):
        raise ValueError("metric must be 'diag', 'dense', or 'unit'")
    kind: MetricKind = "unit"
    if metric == "diag":
        kind = "diag"
    elif metric == "dense":
        kind = "dense"
    starts = _initial_points(initial, checked_chains)
    dimension = len(starts[0])
    if initial_inverse_metric is None or kind == "unit":
        inverse = _identity(dimension)
    else:
        inverse = symmetrize(initial_inverse_metric, name="initial_inverse_metric")
        if len(inverse) != dimension:
            raise ValueError("initial_inverse_metric must match the state dimension")
        try:
            cholesky(inverse)
        except ValueError as error:
            raise ValueError("initial_inverse_metric must be positive definite") from (
                error
            )
        if kind == "diag":
            inverse = tuple(
                tuple(inverse[i][i] if i == j else 0.0 for j in range(dimension))
                for i in range(dimension)
            )
    fixed_step: float | None = None
    if step_size is not None:
        fixed_step = require_finite_real(step_size, name="step_size")
        if fixed_step <= 0.0:
            raise ValueError("step_size must be positive")
    elif not adapt_step_size:
        raise ValueError("step_size is required when adapt_step_size is False")
    delta = require_finite_real(target_acceptance, name="target_acceptance")
    if not 0.0 < delta < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    steps = require_non_negative_integer(leapfrog_steps, name="leapfrog_steps")
    if steps == 0:
        raise ValueError("leapfrog_steps must be positive")
    jitter = require_finite_real(step_size_jitter, name="step_size_jitter")
    if not 0.0 <= jitter < 1.0:
        raise ValueError("step_size_jitter must lie in [0, 1)")
    depth = require_non_negative_integer(max_tree_depth, name="max_tree_depth")
    if depth == 0:
        raise ValueError("max_tree_depth must be positive")
    threshold = require_finite_real(max_energy_error, name="max_energy_error")
    if threshold <= 0.0:
        raise ValueError("max_energy_error must be positive")
    h = require_finite_real(relative_step, name="relative_step")
    if h <= 0.0:
        raise ValueError("relative_step must be positive")
    settings = _Settings(
        algorithm,
        checked_draws,
        checked_warmup,
        checked_thin,
        kind,
        inverse,
        adapt_metric,
        fixed_step,
        adapt_step_size,
        delta,
        steps,
        jitter,
        depth,
        threshold,
    )
    label = "nuts-chain" if algorithm == "nuts" else "hmc-chain"
    streams = resolve_stream(seed).spawn(checked_chains, label=label)
    chains_out: list[MarkovChain] = []
    statistics: list[HamiltonianChainStatistics] = []
    for start, stream in zip(starts, streams, strict=True):
        target = _Target(log_density, gradient, h)
        chain, record = _run_hamiltonian_chain(target, start, settings, stream)
        chains_out.append(chain)
        statistics.append(record)
    coordinate_names = (
        tuple(f"x{i}" for i in range(dimension)) if names is None else tuple(names)
    )
    title = "NUTS" if algorithm == "nuts" else "Hamiltonian Monte Carlo"
    result = MCMCResult(
        tuple(chains_out), coordinate_names, title, checked_warmup, checked_thin
    )
    return HamiltonianResult(result, tuple(statistics))


def hamiltonian_monte_carlo(
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    gradient: GradientFunction | None = None,
    draws: int,
    warmup: int = 500,
    chains: int = 4,
    thin: int = 1,
    leapfrog_steps: int = 16,
    step_size_jitter: float = 0.0,
    metric: str = "diag",
    initial_inverse_metric: Sequence[Sequence[float]] | None = None,
    adapt_metric: bool = True,
    step_size: float | None = None,
    adapt_step_size: bool = True,
    target_acceptance: float = 0.8,
    max_energy_error: float = _DEFAULT_MAX_ENERGY_ERROR,
    relative_step: float = 1e-6,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> HamiltonianResult:
    """Sample a smooth log-density by static Hamiltonian Monte Carlo.

    Each iteration draws ``p ~ N(0, M)``, takes ``leapfrog_steps`` leapfrog
    steps of size ``eps``, and accepts the end point with probability
    ``min(1, exp(H_0 - H_L))``. During warm-up ``eps`` is tuned by dual
    averaging toward ``target_acceptance`` and ``M^-1`` is estimated in
    doubling windows (see the module docstring); both are frozen afterwards.
    A trajectory whose energy error exceeds ``max_energy_error`` or that
    leaves the support is stopped, rejected, and counted as divergent.
    The integration time ``L eps`` is fixed; on near-Gaussian targets it can
    resonate with the oscillation period and stall the chain, so use
    ``step_size_jitter > 0`` or prefer :func:`no_u_turn_sampler`.

    Args:
        log_density: Target ``log p(x)`` up to a constant on ``R^d``.
        initial: One start shared by all chains, or one start per chain.
        gradient: ``grad log p(x)``; ``None`` uses central differences
            (``2 d`` extra density evaluations per leapfrog step).
        draws: Kept draws per chain.
        warmup: Adaptation and burn-in iterations per chain.
        chains: Number of independent chains (``RandomStream.spawn``).
        thin: Keep every ``thin``-th post-warm-up state.
        leapfrog_steps: Leapfrog steps per iteration ``L``.
        step_size_jitter: ``j`` in ``[0, 1)``; each iteration uses
            ``eps (1 + j (2U - 1))`` with ``U ~ Uniform(0, 1)``. A positive
            ``j`` breaks the periodicity that makes a fixed ``L eps`` close to
            a multiple of the target's oscillation period mix very slowly.
        metric: ``"diag"``, ``"dense"``, or ``"unit"`` inverse metric.
        initial_inverse_metric: Starting ``M^-1`` (identity by default).
        adapt_metric: Whether to estimate ``M^-1`` during warm-up.
        step_size: Initial (or, without adaptation, fixed) step size; by
            default found by the Hoffman-Gelman heuristic.
        adapt_step_size: Whether to tune ``eps`` by dual averaging.
        target_acceptance: Dual-averaging target ``delta`` in ``(0, 1)``.
        max_energy_error: Divergence threshold on ``H - H_0``.
        relative_step: Finite-difference step relative to ``max(1, |x_i|)``.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.

    Returns:
        Draws as an :class:`MCMCResult` plus per-chain divergence, step-size,
        acceptance, and energy diagnostics.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid run lengths or settings, or a start with
            non-finite log-density or gradient.

    Equation:
        [EQ-HMC-H-001], [EQ-HMC-LEAPFROG-001], [EQ-HMC-ACCEPT-001],
        [EQ-HMC-DA-001], [EQ-HMC-METRIC-001].

    References:
        [REF-NEAL-2011], [REF-HOFFMAN-GELMAN-2014], [REF-NESTEROV-2009],
        [REF-BETANCOURT-2017].

    """
    return _sample(
        "hmc",
        log_density,
        initial,
        gradient=gradient,
        draws=draws,
        warmup=warmup,
        chains=chains,
        thin=thin,
        metric=metric,
        initial_inverse_metric=initial_inverse_metric,
        adapt_metric=adapt_metric,
        step_size=step_size,
        adapt_step_size=adapt_step_size,
        target_acceptance=target_acceptance,
        leapfrog_steps=leapfrog_steps,
        step_size_jitter=step_size_jitter,
        max_tree_depth=1,
        max_energy_error=max_energy_error,
        relative_step=relative_step,
        seed=seed,
        names=names,
    )


def no_u_turn_sampler(
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    gradient: GradientFunction | None = None,
    draws: int,
    warmup: int = 500,
    chains: int = 4,
    thin: int = 1,
    max_tree_depth: int = 10,
    metric: str = "diag",
    initial_inverse_metric: Sequence[Sequence[float]] | None = None,
    adapt_metric: bool = True,
    step_size: float | None = None,
    adapt_step_size: bool = True,
    target_acceptance: float = 0.8,
    max_energy_error: float = _DEFAULT_MAX_ENERGY_ERROR,
    relative_step: float = 1e-6,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> HamiltonianResult:
    """Sample a smooth log-density by the No-U-Turn Sampler (slice variant).

    This is the efficient recursive NUTS of Hoffman and Gelman (2014,
    Algorithm 6): a slice level ``log u = -H_0 + log U`` is drawn, the
    trajectory is doubled in a random direction, each subtree selects a
    candidate by progressive sampling among states with ``-H >= log u``, the
    new subtree's candidate replaces the current one with probability
    ``min(1, n'/n)``, and doubling stops at a U-turn of the whole trajectory
    or of any subtree, at a divergence (``-H < log u - max_energy_error``), or
    at ``max_tree_depth``. The acceptance statistic used by dual averaging is
    the mean of ``min(1, exp(H_0 - H))`` over all leapfrog states.

    Args:
        log_density: Target ``log p(x)`` up to a constant on ``R^d``.
        initial: One start shared by all chains, or one start per chain.
        gradient: ``grad log p(x)``; ``None`` uses central differences.
        draws: Kept draws per chain.
        warmup: Adaptation and burn-in iterations per chain.
        chains: Number of independent chains.
        thin: Keep every ``thin``-th post-warm-up state.
        max_tree_depth: Maximum number of trajectory doublings
            (at most ``2^max_tree_depth - 1`` leapfrog steps per iteration).
        metric: ``"diag"``, ``"dense"``, or ``"unit"`` inverse metric.
        initial_inverse_metric: Starting ``M^-1`` (identity by default).
        adapt_metric: Whether to estimate ``M^-1`` during warm-up.
        step_size: Initial (or fixed) step size.
        adapt_step_size: Whether to tune ``eps`` by dual averaging.
        target_acceptance: Dual-averaging target ``delta`` in ``(0, 1)``.
        max_energy_error: Divergence threshold ``Delta_max``.
        relative_step: Finite-difference step relative to ``max(1, |x_i|)``.
        seed: Integer seed or stream.
        names: Optional coordinate names.

    Returns:
        Draws as an :class:`MCMCResult` plus per-chain tree-depth,
        divergence, step-size, acceptance, and energy diagnostics.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid settings or a non-finite start.

    Equation:
        [EQ-HMC-H-001], [EQ-HMC-LEAPFROG-001], [EQ-NUTS-SLICE-001],
        [EQ-NUTS-UTURN-001], [EQ-HMC-DA-001], [EQ-HMC-METRIC-001].

    References:
        [REF-HOFFMAN-GELMAN-2014], [REF-NESTEROV-2009], [REF-BETANCOURT-2017],
        [REF-NEAL-2011].

    """
    return _sample(
        "nuts",
        log_density,
        initial,
        gradient=gradient,
        draws=draws,
        warmup=warmup,
        chains=chains,
        thin=thin,
        metric=metric,
        initial_inverse_metric=initial_inverse_metric,
        adapt_metric=adapt_metric,
        step_size=step_size,
        adapt_step_size=adapt_step_size,
        target_acceptance=target_acceptance,
        leapfrog_steps=1,
        step_size_jitter=0.0,
        max_tree_depth=max_tree_depth,
        max_energy_error=max_energy_error,
        relative_step=relative_step,
        seed=seed,
        names=names,
    )
