"""Approximate Bayesian computation: rejection and population Monte Carlo.

ABC replaces an unavailable likelihood by simulation. A parameter ``theta``
drawn from the prior is kept when data simulated under it have a summary
``s(y_sim)`` within ``epsilon`` of the observed summary ``s(y_obs)``
(Pritchard, Seielstad, Perez-Lezaun and Feldman 1999). The kept draws follow
the ABC posterior
``pi_eps(theta) ∝ p(theta) P(rho(s(y_sim), s(y_obs)) <= eps | theta)``
([EQ-ABC-REJ-001]), which equals the posterior given the summary only in the
limit ``eps -> 0``, and the posterior given the data only when the summary
is sufficient. Two approximations are therefore always present: the
tolerance (bias that shrinks with ``eps`` at a simulation cost that grows
like ``eps^-q`` for a ``q``-dimensional summary) and the summary (information
lost by a non-sufficient summary, which no tolerance recovers). The distance
also matters: summaries on very different scales should be standardized
first, since the Euclidean default weighs them equally.

:func:`abc_rejection` keeps the draws within a fixed tolerance, or the
closest fraction ``quantile`` of a fixed number of prior simulations (the
tolerance is then the data-dependent order statistic).

:func:`abc_smc` is the ABC population Monte Carlo algorithm of Beaumont,
Cornuet, Marin and Robert (2009): generation ``t`` resamples the previous
weighted population, perturbs each draw with a Gaussian kernel of
covariance ``2 Sigma_{t-1}`` (twice the weighted covariance of the previous
population), keeps proposals within ``eps_t`` and weights them by
``w_i ∝ p(theta_i) / sum_j w_j^(t-1) K(theta_i | theta_j^(t-1))``
([EQ-ABC-PMC-001]). The tolerance schedule is adaptive: ``eps_t`` is the
``quantile`` of the accepted distances of generation ``t - 1``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_vector, cholesky, matvec, solve_lower
from probstat.statistics import quantile as sample_quantile

from .mcmc import LogDensity

PriorSampler = Callable[[random.Random], Sequence[float]]
Simulator = Callable[[Vector, random.Random], Sequence[float]]
Distance = Callable[[Vector, Vector], float]

STOPPING_REASONS = ("draws", "generations", "tolerance", "budget")


@dataclass(frozen=True, slots=True)
class ABCResult:
    """Accepted parameters, weights, and costs of an ABC run.

    Args:
        names: Parameter names.
        parameters: Accepted parameter vectors (final population).
        weights: Normalized importance weights of the accepted parameters.
        distances: Distance of each accepted simulation to the observation.
        tolerances: Tolerance of each generation (one entry for rejection).
        simulations: Total number of simulator calls, including those of a
            generation abandoned when the budget ran out.
        simulations_per_generation: Simulator calls of each generation.
        acceptance_rates: Accepted fraction of the simulations of each
            generation.
        method: ``"rejection"`` or ``"population Monte Carlo"``.
        stopped_by: Why the run ended: ``"draws"`` (rejection),
            ``"generations"``, ``"tolerance"`` (reached ``final_tolerance``),
            or ``"budget"`` (the next generation would exceed
            ``max_simulations``; the last complete generation is returned).

    References:
        [REF-PRITCHARD-ETAL-1999], [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

    """

    names: tuple[str, ...]
    parameters: tuple[Vector, ...]
    weights: tuple[float, ...]
    distances: tuple[float, ...]
    tolerances: tuple[float, ...]
    simulations: int
    simulations_per_generation: tuple[int, ...]
    acceptance_rates: tuple[float, ...]
    method: str
    stopped_by: str

    @property
    def tolerance(self) -> float:
        """Return the tolerance of the final population.

        References:
            [EQ-ABC-REJ-001]. No new equation is introduced.

        """
        return self.tolerances[-1]

    @property
    def effective_sample_size(self) -> float:
        """Return ``1 / sum_i w_i^2`` of the final weights.

        Equation:
            [EQ-FILT-PF-002].

        References:
            [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

        """
        return 1.0 / math.fsum(w * w for w in self.weights)

    def mean(self) -> Vector:
        """Return the weighted mean of the accepted parameters.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

        """
        return _weighted_mean(self.parameters, self.weights)

    def covariance(self) -> Matrix:
        """Return the weighted covariance ``sum w (theta - m)(theta - m)'``.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

        """
        return _weighted_covariance(self.parameters, self.weights)


def _weighted_mean(points: Sequence[Vector], weights: Sequence[float]) -> Vector:
    """Return ``sum_i w_i x_i`` for normalized weights.

    References:
        [EQ-MC-MEAN-001]. No new equation is introduced.

    """
    return tuple(
        math.fsum(w * point[k] for w, point in zip(weights, points, strict=True))
        for k in range(len(points[0]))
    )


def _weighted_covariance(points: Sequence[Vector], weights: Sequence[float]) -> Matrix:
    """Return ``sum_i w_i (x_i - m)(x_i - m)'`` for normalized weights.

    References:
        [EQ-MC-VAR-001]. No new equation is introduced.

    """
    center = _weighted_mean(points, weights)
    dimension = len(center)
    return tuple(
        tuple(
            math.fsum(
                w * (point[i] - center[i]) * (point[j] - center[j])
                for w, point in zip(weights, points, strict=True)
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def euclidean_distance(simulated: Vector, observed: Vector) -> float:
    """Return the Euclidean distance between two summary vectors.

    Args:
        simulated: Simulated summary.
        observed: Observed summary of the same length.

    Returns:
        ``sqrt(sum (s_k - o_k)^2)``.

    Raises:
        ValueError: If the lengths differ.

    References:
        [REF-PRITCHARD-ETAL-1999]. No registered equation is evaluated.

    """
    if len(simulated) != len(observed):
        raise ValueError("summaries must have the same length")
    return math.sqrt(
        math.fsum((a - b) * (a - b) for a, b in zip(simulated, observed, strict=True))
    )


class _Problem:
    """Validated simulator, summary, and distance shared by both samplers.

    References:
        [REF-PRITCHARD-ETAL-1999]. No equation is evaluated.

    """

    def __init__(
        self,
        simulator: Simulator,
        prior_sampler: PriorSampler,
        observed_summary: Sequence[float],
        distance: Distance | None,
    ) -> None:
        """Validate the callables and the observed summary.

        References:
            [REF-PRITCHARD-ETAL-1999]. No equation is evaluated.

        """
        if not callable(simulator):
            raise TypeError("simulator must be callable")
        if not callable(prior_sampler):
            raise TypeError("prior_sampler must be callable")
        if distance is not None and not callable(distance):
            raise TypeError("distance must be callable or None")
        self.simulator = simulator
        self.prior_sampler = prior_sampler
        self.observed = as_vector(observed_summary, name="observed_summary")
        if not self.observed:
            raise ValueError("observed_summary must be non-empty")
        self.distance = euclidean_distance if distance is None else distance
        self.simulations = 0
        self.dimension: int | None = None

    def prior_draw(self, generator: random.Random) -> Vector:
        """Draw one prior parameter and check its dimension.

        References:
            [REF-PRITCHARD-ETAL-1999]. No equation is evaluated.

        """
        theta = as_vector(self.prior_sampler(generator), name="prior draw")
        self.check_dimension(theta)
        return theta

    def check_dimension(self, theta: Vector) -> None:
        """Record or check the parameter dimension.

        References:
            [REF-PRITCHARD-ETAL-1999]. No equation is evaluated.

        """
        if self.dimension is None:
            if not theta:
                raise ValueError("parameters must be non-empty")
            self.dimension = len(theta)
        elif len(theta) != self.dimension:
            raise ValueError("parameter draws must share one dimension")

    def simulate(self, theta: Vector, generator: random.Random) -> float:
        """Simulate one summary and return its distance to the observation.

        References:
            [EQ-ABC-REJ-001]. No new equation is introduced.

        """
        self.simulations += 1
        summary = as_vector(self.simulator(theta, generator), name="simulated summary")
        if len(summary) != len(self.observed):
            raise ValueError("simulated and observed summaries differ in length")
        value = float(self.distance(summary, self.observed))
        if math.isnan(value) or value < 0.0:
            raise ValueError("distance must return a non-negative number")
        return value


def _labels(names: Sequence[str] | None, dimension: int) -> tuple[str, ...]:
    """Return parameter names, defaulting to ``theta0, theta1, ...``.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    labels = (
        tuple(f"theta{i}" for i in range(dimension)) if names is None else tuple(names)
    )
    if len(labels) != dimension or len(set(labels)) != dimension:
        raise ValueError("names must be distinct and match the parameter dimension")
    return labels


def _budget(max_simulations: int | None, default: int) -> int:
    """Validate a simulation budget.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if max_simulations is None:
        return default
    budget = require_non_negative_integer(max_simulations, name="max_simulations")
    if budget == 0:
        raise ValueError("max_simulations must be positive")
    return budget


def abc_rejection(
    simulator: Simulator,
    prior_sampler: PriorSampler,
    observed_summary: Sequence[float],
    distance: Distance | None = None,
    *,
    draws: int,
    tolerance: float | None = None,
    quantile: float | None = None,
    max_simulations: int | None = None,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> ABCResult:
    """Sample the ABC posterior by rejection.

    Give exactly one of ``tolerance`` and ``quantile``. With ``tolerance``
    the prior is simulated until ``draws`` parameters have distance at most
    ``tolerance``. With ``quantile = q`` exactly ``ceil(draws / q)`` prior
    simulations are run and the ``draws`` closest are kept; the reported
    tolerance is the largest kept distance.

    Args:
        simulator: ``(theta, generator) -> summary`` (a ``random.Random`` is
            passed); it returns the summary statistics of simulated data.
        prior_sampler: ``generator -> theta``.
        observed_summary: Summary of the observed data.
        distance: ``(simulated, observed) -> rho >= 0``; Euclidean if
            ``None``.
        draws: Number of accepted parameters (positive).
        tolerance: Fixed tolerance ``eps >= 0``.
        quantile: Accepted fraction in ``(0, 1]``.
        max_simulations: Simulation budget of the tolerance mode (default
            ``1000 draws``); exceeding it raises.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional parameter names (default ``theta0, theta1, ...``).

    Returns:
        Equally weighted accepted parameters, their distances, the
        tolerance, and the number of simulations.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid options, inconsistent dimensions, a negative
            distance, or (tolerance mode) a budget exhausted before ``draws``
            acceptances.

    Equation:
        [EQ-ABC-REJ-001].

    References:
        [REF-PRITCHARD-ETAL-1999], [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009].

    """
    problem = _Problem(simulator, prior_sampler, observed_summary, distance)
    count = require_non_negative_integer(draws, name="draws")
    if count == 0:
        raise ValueError("draws must be positive")
    if (tolerance is None) == (quantile is None):
        raise ValueError("give exactly one of tolerance and quantile")
    stream = resolve_stream(seed)
    prior_generator = stream.child("prior").generator()
    data_generator = stream.child("simulator").generator()
    accepted: list[tuple[float, Vector]] = []
    if tolerance is not None:
        epsilon = require_finite_real(tolerance, name="tolerance")
        if epsilon < 0.0:
            raise ValueError("tolerance must be non-negative")
        budget = _budget(max_simulations, 1000 * count)
        while len(accepted) < count:
            if problem.simulations >= budget:
                raise ValueError(
                    f"only {len(accepted)} of {count} draws accepted within "
                    f"max_simulations={budget}; raise the tolerance or the budget"
                )
            theta = problem.prior_draw(prior_generator)
            rho = problem.simulate(theta, data_generator)
            if rho <= epsilon:
                accepted.append((rho, theta))
        reported = epsilon
    else:
        fraction = require_finite_real(quantile, name="quantile")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")
        if max_simulations is not None:
            raise ValueError("max_simulations applies to the tolerance mode only")
        total = math.ceil(count / fraction - 1e-9)
        simulated: list[tuple[float, int, Vector]] = []
        for index in range(total):
            theta = problem.prior_draw(prior_generator)
            simulated.append((problem.simulate(theta, data_generator), index, theta))
        simulated.sort(key=lambda item: (item[0], item[1]))
        accepted = [(rho, theta) for rho, _, theta in simulated[:count]]
        reported = accepted[-1][0]
    dimension = problem.dimension if problem.dimension is not None else 0
    return ABCResult(
        _labels(names, dimension),
        tuple(theta for _, theta in accepted),
        tuple(1.0 / count for _ in range(count)),
        tuple(rho for rho, _ in accepted),
        (reported,),
        problem.simulations,
        (problem.simulations,),
        (count / problem.simulations,),
        "rejection",
        "draws",
    )


def _log_kernel_mixture(
    whitened: Vector,
    previous: Sequence[Vector],
    log_weights: Sequence[float],
) -> float:
    """Return ``log sum_j w_j exp(-|z - z_j|^2 / 2)`` in whitened coordinates.

    ``z = L^-1 theta`` and ``z_j = L^-1 theta_j`` with ``L L' = 2 Sigma``, so
    ``|z - z_j|^2`` is the Mahalanobis distance of the Gaussian kernel. Its
    normalizing constant is common to every point and cancels when the
    importance weights are normalized.

    References:
        [EQ-ABC-PMC-001]. No new equation is introduced.

    """
    terms = [
        log_weight
        - 0.5
        * math.fsum((a - b) * (a - b) for a, b in zip(whitened, center, strict=True))
        for log_weight, center in zip(log_weights, previous, strict=True)
    ]
    top = max(terms)
    return top + math.log(math.fsum(math.exp(term - top) for term in terms))


def abc_smc(
    simulator: Simulator,
    prior_sampler: PriorSampler,
    log_prior: LogDensity,
    observed_summary: Sequence[float],
    distance: Distance | None = None,
    *,
    draws: int,
    generations: int = 5,
    quantile: float = 0.5,
    initial_tolerance: float | None = None,
    final_tolerance: float | None = None,
    max_simulations: int | None = None,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> ABCResult:
    """Sample the ABC posterior by population Monte Carlo (ABC-PMC).

    Generation 1 is rejection sampling from the prior at
    ``initial_tolerance`` (``inf`` by default: ``draws`` prior simulations,
    all kept). Generation ``t >= 2`` uses ``eps_t = max(final_tolerance,
    Q_quantile(distances_{t-1}))``, draws ``theta*`` from the previous
    population with its weights, proposes ``theta ~ N(theta*, 2 Sigma_{t-1})``
    (proposals with zero prior density are discarded without simulation),
    keeps ``theta`` when its simulated distance is at most ``eps_t``, and
    weights it by ``p(theta) / sum_j w_j K(theta | theta_j)``.

    Args:
        simulator: ``(theta, generator) -> summary``.
        prior_sampler: ``generator -> theta``.
        log_prior: Prior log-density (up to a constant; ``-inf`` outside the
            support).
        observed_summary: Summary of the observed data.
        distance: ``(simulated, observed) -> rho >= 0``; Euclidean if
            ``None``.
        draws: Population size (at least two).
        generations: Maximum number of generations (positive).
        quantile: Tolerance-schedule quantile in ``(0, 1)``.
        initial_tolerance: Tolerance of generation 1 (``None`` = ``inf``).
        final_tolerance: Stop after the generation whose tolerance reaches
            this value (``None`` = run ``generations`` generations).
        max_simulations: Total simulation budget (default
            ``1000 draws generations``). A generation that would exceed it is
            abandoned and the last complete one returned
            (``stopped_by = "budget"``).
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional parameter names.

    Returns:
        Final weighted population with the tolerance schedule, simulations
        per generation, and acceptance rates.

    Raises:
        TypeError: If a callable argument is not callable.
        ValueError: For invalid options, inconsistent dimensions, a prior
            draw with zero prior density, a singular population covariance,
            or a first generation that does not fit in the budget.

    Equation:
        [EQ-ABC-REJ-001], [EQ-ABC-PMC-001].

    References:
        [REF-BEAUMONT-CORNUET-MARIN-ROBERT-2009], [REF-PRITCHARD-ETAL-1999].

    """
    problem = _Problem(simulator, prior_sampler, observed_summary, distance)
    if not callable(log_prior):
        raise TypeError("log_prior must be callable")
    count = require_non_negative_integer(draws, name="draws")
    if count < 2:
        raise ValueError("draws must be at least two")
    rounds = require_non_negative_integer(generations, name="generations")
    if rounds == 0:
        raise ValueError("generations must be positive")
    level = require_finite_real(quantile, name="quantile")
    if not 0.0 < level < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    first = math.inf
    if initial_tolerance is not None:
        first = require_finite_real(initial_tolerance, name="initial_tolerance")
        if first < 0.0:
            raise ValueError("initial_tolerance must be non-negative")
    floor = 0.0
    if final_tolerance is not None:
        floor = require_finite_real(final_tolerance, name="final_tolerance")
        if floor < 0.0:
            raise ValueError("final_tolerance must be non-negative")
    budget = _budget(max_simulations, 1000 * count * rounds)
    stream = resolve_stream(seed)

    def checked_prior(theta: Vector) -> float:
        """Evaluate and validate the prior log-density.

        References:
            [EQ-ABC-PMC-001]. No new equation is introduced.

        """
        value = float(log_prior(theta))
        if math.isnan(value) or value == math.inf:
            raise ValueError("log_prior returned NaN or +inf")
        return value

    generation_stream = stream.child("generation", 1)
    prior_generator = generation_stream.child("prior").generator()
    data_generator = generation_stream.child("simulator").generator()
    points: list[Vector] = []
    distances: list[float] = []
    while len(points) < count:
        if problem.simulations >= budget:
            raise ValueError("the first generation does not fit in max_simulations")
        theta = problem.prior_draw(prior_generator)
        if checked_prior(theta) == -math.inf:
            raise ValueError("prior_sampler returned a draw with zero prior density")
        rho = problem.simulate(theta, data_generator)
        if rho <= first:
            points.append(theta)
            distances.append(rho)
    weights = [1.0 / count] * count
    tolerances = [first]
    per_generation = [problem.simulations]
    rates = [count / problem.simulations]
    stopped_by = "generations"
    if first <= floor:
        stopped_by = "tolerance"
    while stopped_by == "generations" and len(tolerances) < rounds:
        epsilon = max(floor, sample_quantile(distances, level))
        covariance = _weighted_covariance(points, weights)
        doubled = tuple(tuple(2.0 * value for value in row) for row in covariance)
        try:
            lower = cholesky(doubled)
        except ValueError as error:
            raise ValueError(
                "the population covariance is singular; the perturbation "
                "kernel 2 Sigma is undefined"
            ) from error
        log_previous = [math.log(w) if w > 0.0 else -math.inf for w in weights]
        whitened_previous = [solve_lower(lower, point) for point in points]
        cumulative = list(accumulate(weights))
        generation_stream = stream.child("generation", len(tolerances) + 1)
        proposal_generator = generation_stream.child("proposal").generator()
        data_generator = generation_stream.child("simulator").generator()
        started = problem.simulations
        new_points: list[Vector] = []
        new_distances: list[float] = []
        log_new: list[float] = []
        exhausted = False
        while len(new_points) < count:
            if problem.simulations >= budget:
                exhausted = True
                break
            center = proposal_generator.choices(points, cum_weights=cumulative)[0]
            z = tuple(proposal_generator.gauss(0.0, 1.0) for _ in center)
            theta = tuple(a + b for a, b in zip(center, matvec(lower, z), strict=True))
            prior_value = checked_prior(theta)
            if prior_value == -math.inf:
                continue
            rho = problem.simulate(theta, data_generator)
            if rho <= epsilon:
                new_points.append(theta)
                new_distances.append(rho)
                log_new.append(
                    prior_value
                    - _log_kernel_mixture(
                        solve_lower(lower, theta), whitened_previous, log_previous
                    )
                )
        if exhausted:
            stopped_by = "budget"
            break
        top = max(log_new)
        raw = [math.exp(value - top) for value in log_new]
        total = math.fsum(raw)
        points, distances = new_points, new_distances
        weights = [value / total for value in raw]
        tolerances.append(epsilon)
        used = problem.simulations - started
        per_generation.append(used)
        rates.append(count / used)
        if epsilon <= floor and final_tolerance is not None:
            stopped_by = "tolerance"
    dimension = problem.dimension if problem.dimension is not None else 0
    return ABCResult(
        _labels(names, dimension),
        tuple(points),
        tuple(weights),
        tuple(distances),
        tuple(tolerances),
        problem.simulations,
        tuple(per_generation),
        tuple(rates),
        "population Monte Carlo",
        stopped_by,
    )
