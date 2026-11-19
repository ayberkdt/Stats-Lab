"""Sequential Monte Carlo with adaptive likelihood tempering and evidence.

The sampler moves a population of ``N`` particles from the prior ``p(x)``
to the posterior through the tempered laws
``pi_t(x) ∝ p(x) L(x)^beta_t``, ``0 = beta_0 < beta_1 < ... < beta_T = 1``
(Del Moral, Doucet and Jasra 2006; Chopin 2002 for the static-model
precursor). One stage ``t`` performs:

1. **Reweight.** Incremental weights ``w_i = L(x_i)^(beta_t - beta_{t-1})``
   ([EQ-SMC-TEMPER-001]). The next temperature is chosen adaptively: the
   largest ``beta_t <= 1`` whose effective sample size
   ``ESS = (sum w)^2 / sum w^2`` is at least ``ess_fraction`` times the
   number of particles with finite log-likelihood, found by bisection
   (``ESS`` is non-increasing in the increment), as in Jasra, Stephens,
   Doucet and Tsagaris (2011) ([EQ-SMC-TEMPER-002]).
2. **Resample** ``N`` ancestors by multinomial (default) or systematic
   resampling. Because the adaptive rule fixes the ESS after reweighting at
   the threshold, resampling every stage is the usual ESS-threshold rule.
3. **Mutate** each particle by ``mutation_steps`` Markov moves that leave
   ``pi_t`` invariant: random-walk Metropolis with covariance
   ``s^2 Sigma_t`` (``Sigma_t`` the weighted particle covariance after
   reweighting, ``s = 2.38 / sqrt(d)`` by default), or preconditioned
   Crank-Nicolson for an ``N(0, I)`` prior, with ``beta`` adapted between
   stages towards a target acceptance.

The kernels are chosen from the particle population of the current stage
before it moves, which is the standard adaptive SMC construction; it keeps
each mutation invariant for ``pi_t``.

Evidence
    ``Z = integral p(x) L(x) dx`` is estimated by the product of the mean
    incremental weights, ``log Z_hat = sum_t log mean_i w_{t,i}``
    ([EQ-SMC-EVID-001]). ``L`` must therefore carry *all* normalizing
    constants of the data density for ``Z_hat`` to be the marginal
    likelihood used for model comparison (Bayes factors are ratios of
    ``Z``). For fixed temperatures and kernels ``Z_hat`` is unbiased.

Error estimate
    With multinomial resampling the result reports the estimator of Lee and
    Whiteley (2018, equation (4) with ``phi = G_{T-1}``) of the relative
    variance of ``Z_hat`` from the genealogy of one run
    ([EQ-SMC-EVID-002]):
    ``v = 1 - (N/(N-1))^T sum_{i,j: E_i != E_j} w_i w_j / (sum_i w_i)^2``,
    where ``w`` are the final-stage incremental weights and ``E_i`` the index
    of the initial (prior) particle from which particle ``i`` descends. For
    fixed temperatures and kernels ``Z^2 v`` estimates ``Var(Z_hat)``
    without bias (their Theorem 1) and ``N v`` is consistent for the
    asymptotic relative variance; ``sqrt(v)`` is then the delta-method
    standard error of ``log Z_hat``. Limitations: (i) here the temperatures
    and proposal scales are chosen from the particles, which the theorem
    does not cover, so the value is a consistent-in-spirit approximation;
    (ii) ``v`` can be negative (reported as a NaN standard error) and it
    degenerates when all particles descend from few initial particles,
    which happens when the number of stages is large relative to ``N``;
    (iii) it measures Monte Carlo variability only, not the bias of a
    poorly mixing mutation kernel, which can make ``Z_hat`` wrong by many
    reported standard errors. Systematic resampling is outside the theorem,
    so no estimate is reported for it (NaN).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_vector, cholesky, matvec

from .mcmc import LogDensity

PriorSampler = Callable[[random.Random], Sequence[float]]

MUTATIONS = ("rwm", "pcn")
RESAMPLING_SCHEMES = ("multinomial", "systematic")
_BISECTION_STEPS = 200


@dataclass(frozen=True, slots=True)
class SMCResult:
    """Particles, temperature schedule, and evidence of a tempered SMC run.

    Args:
        names: Coordinate names.
        particles: Final particles (equally weighted after the last
            resampling and mutation at temperature one).
        weights: Normalized final weights (all ``1 / N``).
        log_likelihoods: ``log L`` at each final particle.
        temperatures: ``beta_0 = 0, ..., beta_T = 1``.
        effective_sample_sizes: ESS of the incremental weights of each stage.
        acceptance_rates: Mutation acceptance rate of each stage.
        proposal_scales: Random-walk factor ``s`` or pCN ``beta`` of each
            stage.
        log_evidence: ``log Z_hat`` ([EQ-SMC-EVID-001]).
        log_evidence_increments: ``log mean_i w_{t,i}`` per stage.
        log_evidence_relative_variance: Lee-Whiteley estimate ``v`` of
            ``Var(Z_hat) / Z^2`` (NaN for systematic resampling).
        log_evidence_standard_error: ``sqrt(v)``, the delta-method standard
            error of ``log Z_hat`` (NaN when ``v`` is unavailable or
            negative).
        mutation: ``"rwm"`` or ``"pcn"``.
        resampling: ``"multinomial"`` or ``"systematic"``.
        mutation_steps: Markov moves per particle and stage.
        evaluations: Log-likelihood evaluations.

    References:
        [REF-DELMORAL-DOUCET-JASRA-2006], [REF-LEE-WHITELEY-2018].

    """

    names: tuple[str, ...]
    particles: tuple[Vector, ...]
    weights: tuple[float, ...]
    log_likelihoods: tuple[float, ...]
    temperatures: tuple[float, ...]
    effective_sample_sizes: tuple[float, ...]
    acceptance_rates: tuple[float, ...]
    proposal_scales: tuple[float, ...]
    log_evidence: float
    log_evidence_increments: tuple[float, ...]
    log_evidence_relative_variance: float
    log_evidence_standard_error: float
    mutation: str
    resampling: str
    mutation_steps: int
    evaluations: int

    @property
    def stages(self) -> int:
        """Return the number of tempering stages ``T``.

        References:
            [REF-JASRA-STEPHENS-DOUCET-TSAGARIS-2011]. No equation is
            evaluated.

        """
        return len(self.temperatures) - 1

    @property
    def dimension(self) -> int:
        """Return the state dimension.

        References:
            [REF-DELMORAL-DOUCET-JASRA-2006]. No equation is evaluated.

        """
        return len(self.names)

    def mean(self) -> Vector:
        """Return the weighted particle mean.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-DELMORAL-DOUCET-JASRA-2006].

        """
        return _weighted_mean(self.particles, self.weights)

    def covariance(self) -> Matrix:
        """Return the weighted particle covariance ``sum w (x - m)(x - m)'``.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-DELMORAL-DOUCET-JASRA-2006].

        """
        return _weighted_covariance(self.particles, self.weights)


def _weighted_mean(points: Sequence[Vector], weights: Sequence[float]) -> Vector:
    """Return ``sum_i w_i x_i`` for normalized weights.

    References:
        [EQ-MC-MEAN-001]. No new equation is introduced.

    """
    dimension = len(points[0])
    return tuple(
        math.fsum(w * point[k] for w, point in zip(weights, points, strict=True))
        for k in range(dimension)
    )


def _weighted_covariance(points: Sequence[Vector], weights: Sequence[float]) -> Matrix:
    """Return ``sum_i w_i (x_i - m)(x_i - m)'`` for normalized weights.

    References:
        [EQ-MC-VAR-001]. No new equation is introduced.

    """
    center = _weighted_mean(points, weights)
    dimension = len(center)
    deviations = [
        tuple(point[k] - center[k] for k in range(dimension)) for point in points
    ]
    return tuple(
        tuple(
            math.fsum(
                w * deviation[i] * deviation[j]
                for w, deviation in zip(weights, deviations, strict=True)
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def _log_sum_exp(values: Sequence[float]) -> float:
    """Return ``log sum exp(values)``; ``-inf`` if every value is ``-inf``.

    References:
        [EQ-SMC-EVID-001]. No new equation is introduced.

    """
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(value - top) for value in values))


def _ess(log_weights: Sequence[float]) -> float:
    """Return ``(sum w)^2 / sum w^2`` from log-weights.

    References:
        [EQ-FILT-PF-002], [REF-DOUCET-JOHANSEN-2011].

    """
    first = _log_sum_exp(log_weights)
    if first == -math.inf:
        return 0.0
    second = _log_sum_exp([2.0 * value for value in log_weights])
    return math.exp(2.0 * first - second)


def _next_temperature(
    log_likelihoods: Sequence[float], current: float, target_ess: float
) -> float:
    """Return the next tempering exponent by ESS bisection.

    Finds the largest ``beta' in (current, 1]`` with
    ``ESS(exp((beta' - current) l_i)) >= target_ess`` (``beta' = 1`` when the
    full step already satisfies it). ``ESS`` is a non-increasing function of
    the increment, so bisection on ``[0, 1 - current]`` brackets the root.

    Args:
        log_likelihoods: Current ``l_i = log L(x_i)`` (``-inf`` allowed).
        current: Current exponent in ``[0, 1)``.
        target_ess: Required effective sample size; it must not exceed the
            number of finite ``l_i``.

    Returns:
        Next exponent, strictly larger than ``current``.

    Raises:
        ValueError: If ``current`` is outside ``[0, 1)``, the target is not
            attainable, or the bisection cannot make progress.

    Equation:
        [EQ-SMC-TEMPER-002].

    References:
        [REF-JASRA-STEPHENS-DOUCET-TSAGARIS-2011],
        [REF-DELMORAL-DOUCET-JASRA-2006].

    """
    beta = require_finite_real(current, name="current")
    if not 0.0 <= beta < 1.0:
        raise ValueError("current must lie in [0, 1)")
    values = tuple(float(value) for value in log_likelihoods)
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("log-likelihoods must be finite or -inf")
    finite = [value for value in values if value != -math.inf]
    target = require_finite_real(target_ess, name="target_ess")
    if not 0.0 < target <= len(finite):
        raise ValueError("target_ess must lie in (0, number of finite values]")

    def ess_at(increment: float) -> float:
        """Return the ESS for one exponent increment.

        References:
            [EQ-SMC-TEMPER-002]. No new equation is introduced.

        """
        return _ess([increment * value for value in finite])

    remaining = 1.0 - beta
    if ess_at(remaining) >= target:
        return 1.0
    low, high = 0.0, remaining
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if ess_at(middle) >= target:
            low = middle
        else:
            high = middle
    if low <= 0.0:
        raise ValueError(
            "tempering cannot advance: the log-likelihood spread is too large "
            "for the ESS target at double precision"
        )
    return min(1.0, beta + low)


def _zero_log_density(point: Vector) -> float:
    """Return zero, the flat placeholder prior of the pCN mutation.

    References:
        [EQ-MCMC-PCN-001]. No new equation is introduced.

    """
    return 0.0


def _multinomial(
    weights: Sequence[float], generator: random.Random
) -> tuple[int, ...]:
    """Draw ``N`` ancestor indices with probabilities ``weights``.

    References:
        [REF-DOUCET-JOHANSEN-2011]. No equation is evaluated.

    """
    count = len(weights)
    return tuple(generator.choices(range(count), weights=weights, k=count))


def _systematic(weights: Sequence[float], generator: random.Random) -> tuple[int, ...]:
    """Draw ``N`` ancestors by systematic resampling with one uniform offset.

    References:
        [EQ-FILT-PF-003], [REF-KITAGAWA-1996].

    """
    count = len(weights)
    offset = generator.random()
    indices: list[int] = []
    cumulative = weights[0]
    source = 0
    for j in range(count):
        position = (offset + j) / count
        while position >= cumulative and source < count - 1:
            source += 1
            cumulative += weights[source]
        indices.append(source)
    return tuple(indices)


def _initial_particles(
    prior: PriorSampler | Sequence[Sequence[float]],
    particles: int | None,
    stream: RandomStream,
) -> tuple[Vector, ...]:
    """Draw or validate the stage-zero particles.

    References:
        [REF-DELMORAL-DOUCET-JASRA-2006]. No equation is evaluated.

    """
    if callable(prior):
        count = 1000 if particles is None else particles
        generator = stream.child("prior").generator()
        points = tuple(
            as_vector(prior(generator), name="prior draw") for _ in range(count)
        )
    else:
        points = tuple(as_vector(item, name="prior draw") for item in prior)
        if particles is not None and particles != len(points):
            raise ValueError("particles must equal the number of prior draws")
    if len(points) < 2:
        raise ValueError("at least two particles are required")
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError("prior draws must share a positive dimension")
    return points


def sequential_monte_carlo(
    prior: PriorSampler | Sequence[Sequence[float]],
    log_prior: LogDensity | None,
    log_likelihood: LogDensity,
    *,
    particles: int | None = None,
    mutation: str = "rwm",
    mutation_steps: int = 5,
    ess_fraction: float = 0.5,
    resampling: str = "multinomial",
    proposal_scale: float | None = None,
    target_acceptance: float = 0.25,
    regularization: float = 1e-10,
    max_stages: int = 1000,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> SMCResult:
    """Sample a posterior and estimate its evidence by tempered SMC.

    See the module docstring for the algorithm, the evidence estimator, and
    the meaning and limits of its standard error.

    Args:
        prior: Prior sampler ``generator -> x`` (a ``random.Random`` is
            passed) or a sequence of prior draws, one per particle.
        log_prior: Prior log-density (up to a constant; ``-inf`` outside the
            support), required by ``mutation="rwm"``. Must be ``None`` for
            ``mutation="pcn"``, which assumes the prior is exactly
            ``N(0, I)`` (it cannot check this).
        log_likelihood: ``log L(x)`` including every normalizing constant of
            the data density when the evidence is wanted; ``-inf`` allowed.
        particles: Number of particles (default 1000 with a sampler, the
            number of draws otherwise).
        mutation: ``"rwm"`` (random-walk Metropolis with the particle
            covariance) or ``"pcn"`` (preconditioned Crank-Nicolson).
        mutation_steps: Markov moves per particle and stage (positive).
        ess_fraction: ESS target as a fraction of the particles with finite
            log-likelihood, in ``(0, 1)``.
        resampling: ``"multinomial"`` (default) or ``"systematic"``.
        proposal_scale: Random-walk factor ``s`` (default ``2.38 / sqrt(d)``)
            or the initial pCN ``beta`` in ``(0, 1]`` (default 0.5).
        target_acceptance: Target of the between-stage pCN adaptation
            ``logit(beta) += 4 (rate - target)``, in ``(0, 1)``.
        regularization: ``eps >= 0`` added to the diagonal of the particle
            covariance before the random-walk Cholesky factorization.
        max_stages: Maximum number of tempering stages.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names (default ``x0, x1, ...``).

    Returns:
        Final particles, temperatures, per-stage ESS and acceptance rates,
        ``log Z_hat`` and its error estimate.

    Raises:
        TypeError: If a density is not callable.
        ValueError: For invalid options, fewer than two particles with finite
            log-likelihood, a likelihood returning NaN or ``+inf``, a
            non-positive-definite particle covariance, or more than
            ``max_stages`` stages.

    Equation:
        [EQ-SMC-TEMPER-001], [EQ-SMC-TEMPER-002], [EQ-SMC-EVID-001],
        [EQ-SMC-EVID-002], [EQ-MCMC-PCN-001], [EQ-MCMC-MH-001].

    References:
        [REF-DELMORAL-DOUCET-JASRA-2006], [REF-CHOPIN-2002],
        [REF-JASRA-STEPHENS-DOUCET-TSAGARIS-2011], [REF-LEE-WHITELEY-2018],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    if not callable(log_likelihood):
        raise TypeError("log_likelihood must be callable")
    if mutation not in MUTATIONS:
        raise ValueError("mutation must be 'rwm' or 'pcn'")
    if mutation == "rwm" and not callable(log_prior):
        raise TypeError("mutation='rwm' needs a callable log_prior")
    if mutation == "pcn" and log_prior is not None:
        raise ValueError(
            "mutation='pcn' assumes an N(0, I) prior; pass log_prior=None"
        )
    if resampling not in RESAMPLING_SCHEMES:
        raise ValueError("resampling must be 'multinomial' or 'systematic'")
    steps = require_non_negative_integer(mutation_steps, name="mutation_steps")
    if steps == 0:
        raise ValueError("mutation_steps must be positive")
    fraction = require_finite_real(ess_fraction, name="ess_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("ess_fraction must lie in (0, 1)")
    target = require_finite_real(target_acceptance, name="target_acceptance")
    if not 0.0 < target < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    epsilon = require_finite_real(regularization, name="regularization")
    if epsilon < 0.0:
        raise ValueError("regularization must be non-negative")
    stage_limit = require_non_negative_integer(max_stages, name="max_stages")
    if stage_limit == 0:
        raise ValueError("max_stages must be positive")
    if particles is not None:
        particles = require_non_negative_integer(particles, name="particles")
    stream = resolve_stream(seed)
    points = list(_initial_particles(prior, particles, stream))
    count = len(points)
    dimension = len(points[0])
    labels = tuple(f"x{i}" for i in range(dimension)) if names is None else tuple(names)
    if len(labels) != dimension or len(set(labels)) != dimension:
        raise ValueError("names must be distinct and match the dimension")
    if mutation == "rwm":
        scale = 2.38 / math.sqrt(dimension) if proposal_scale is None else (
            require_finite_real(proposal_scale, name="proposal_scale")
        )
        if scale <= 0.0:
            raise ValueError("proposal_scale must be positive")
    else:
        scale = 0.5 if proposal_scale is None else require_finite_real(
            proposal_scale, name="proposal_scale"
        )
        if not 0.0 < scale <= 1.0:
            raise ValueError("pCN proposal_scale must lie in (0, 1]")

    evaluations = 0
    prior_function: LogDensity = (
        log_prior if log_prior is not None else _zero_log_density
    )

    def likelihood(point: Vector) -> float:
        """Evaluate and validate one log-likelihood value.

        References:
            [EQ-SMC-TEMPER-001]. No new equation is introduced.

        """
        nonlocal evaluations
        evaluations += 1
        value = float(log_likelihood(point))
        if math.isnan(value) or value == math.inf:
            raise ValueError("log_likelihood returned NaN or +inf")
        return value

    def prior_density(point: Vector) -> float:
        """Evaluate and validate one prior log-density value.

        References:
            [EQ-MCMC-MH-001]. No new equation is introduced.

        """
        value = float(prior_function(point))
        if math.isnan(value) or value == math.inf:
            raise ValueError("log_prior returned NaN or +inf")
        return value

    loglik = [likelihood(point) for point in points]
    logprior = (
        [prior_density(point) for point in points]
        if mutation == "rwm"
        else [0.0] * count
    )
    eves = list(range(count))
    temperatures = [0.0]
    ess_values: list[float] = []
    rates: list[float] = []
    scales: list[float] = []
    increments: list[float] = []
    relative_variance = math.nan
    log_odds = math.log(scale) - math.log1p(-scale) if scale < 1.0 else 30.0
    beta = 0.0
    while beta < 1.0:
        if len(temperatures) > stage_limit:
            raise ValueError(f"tempering needed more than {stage_limit} stages")
        finite = sum(1 for value in loglik if value != -math.inf)
        if finite < 2:
            raise ValueError("fewer than two particles have a finite likelihood")
        new_beta = _next_temperature(loglik, beta, fraction * finite)
        increment = new_beta - beta
        log_weights = [
            increment * value if value != -math.inf else -math.inf for value in loglik
        ]
        total = _log_sum_exp(log_weights)
        increments.append(total - math.log(count))
        ess_values.append(_ess(log_weights))
        normalized = [math.exp(value - total) for value in log_weights]
        stage = len(temperatures)
        stage_stream = stream.child("stage", stage)
        if new_beta >= 1.0 and resampling == "multinomial":
            groups: dict[int, float] = {}
            for eve, weight in zip(eves, normalized, strict=True):
                groups[eve] = groups.get(eve, 0.0) + weight
            same = math.fsum(value * value for value in groups.values())
            relative_variance = 1.0 - (count / (count - 1)) ** stage * (1.0 - same)
        lower: Matrix | None = None
        if mutation == "rwm":
            covariance = _weighted_covariance(points, normalized)
            regularized = tuple(
                tuple(
                    scale * scale * (covariance[i][j] + (epsilon if i == j else 0.0))
                    for j in range(dimension)
                )
                for i in range(dimension)
            )
            try:
                lower = cholesky(regularized)
            except ValueError as error:
                raise ValueError(
                    "the particle covariance is not positive definite; "
                    "increase regularization or the number of particles"
                ) from error
            scales.append(scale)
        else:
            scale = 1.0 / (1.0 + math.exp(-log_odds)) if log_odds < 700 else 1.0
            scales.append(scale)
        resample = _multinomial if resampling == "multinomial" else _systematic
        ancestors = resample(normalized, stage_stream.child("resample").generator())
        points = [points[a] for a in ancestors]
        loglik = [loglik[a] for a in ancestors]
        logprior = [logprior[a] for a in ancestors]
        eves = [eves[a] for a in ancestors]
        generator = stage_stream.child("mutation").generator()
        accepted = 0
        contraction = math.sqrt(max(0.0, 1.0 - scale * scale))
        for index in range(count):
            state, state_loglik, state_prior = (
                points[index],
                loglik[index],
                logprior[index],
            )
            for _ in range(steps):
                if lower is not None:
                    z = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
                    proposal = tuple(
                        a + b for a, b in zip(state, matvec(lower, z), strict=True)
                    )
                    proposal_prior = prior_density(proposal)
                    if proposal_prior == -math.inf:
                        continue
                    proposal_loglik = likelihood(proposal)
                    log_ratio = (
                        proposal_prior
                        + new_beta * proposal_loglik
                        - state_prior
                        - new_beta * state_loglik
                    )
                else:
                    proposal = tuple(
                        contraction * value + scale * generator.gauss(0.0, 1.0)
                        for value in state
                    )
                    proposal_prior = 0.0
                    proposal_loglik = likelihood(proposal)
                    log_ratio = new_beta * (proposal_loglik - state_loglik)
                if math.log(1.0 - generator.random()) < log_ratio:
                    state, state_loglik, state_prior = (
                        proposal,
                        proposal_loglik,
                        proposal_prior,
                    )
                    accepted += 1
            points[index], loglik[index], logprior[index] = (
                state,
                state_loglik,
                state_prior,
            )
        rate = accepted / (count * steps)
        rates.append(rate)
        if mutation == "pcn":
            log_odds += 4.0 * (rate - target)
        temperatures.append(new_beta)
        beta = new_beta
    log_evidence = math.fsum(increments)
    error = (
        math.sqrt(relative_variance)
        if not math.isnan(relative_variance) and relative_variance >= 0.0
        else math.nan
    )
    return SMCResult(
        labels,
        tuple(points),
        tuple(1.0 / count for _ in range(count)),
        tuple(loglik),
        tuple(temperatures),
        tuple(ess_values),
        tuple(rates),
        tuple(scales),
        log_evidence,
        tuple(increments),
        relative_variance,
        error,
        mutation,
        resampling,
        steps,
        evaluations,
    )
