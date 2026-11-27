"""Transitional MCMC, Bayesian model class selection, and hierarchical calibration.

Experimental (added in 0.28).

Transitional MCMC (TMCMC)
    Ching and Chen (2007) sample a posterior ``p(x | D) ∝ p(x) L(x)`` through
    the transitional densities ``pi_j(x) ∝ p(x) L(x)^q_j`` with
    ``0 = q_0 < q_1 < ... < q_m = 1`` ([EQ-INF-TMCMC-001]). One stage
    ``j -> j + 1`` does the following.

    1. **Exponent.** The next exponent is the largest ``q_{j+1} <= 1`` for
       which the coefficient of variation of the plausibility weights
       ``w_i = L(x_i)^(q_{j+1} - q_j)`` equals a target (1 by default), found
       by bisection because the coefficient of variation is non-decreasing in
       the increment ([EQ-INF-TMCMC-002]; Betz, Papaioannou and Straub 2016
       keep this rule and show it is the delicate part of the method).
    2. **Evidence.** ``S_j = mean_i w_i`` and ``log Z_hat = sum_j log S_j``
       ([EQ-INF-TMCMC-003]). The mean runs over *all* ``N`` samples, samples
       with zero likelihood contributing zero, so the estimator is the
       importance-sampling estimator of ``Z_{j+1} / Z_j`` for every stage.
    3. **Resampling and Markov moves.** ``N`` ancestors are drawn with
       probabilities proportional to ``w_i``. Each distinct ancestor (a
       *leader*) starts one Metropolis chain whose recorded length equals its
       multiplicity, after ``burn_in`` discarded steps, with proposal
       ``N(x, beta^2 Sigma_j)`` where ``Sigma_j`` is the weighted sample
       covariance ([EQ-INF-TMCMC-004]). The factor ``beta`` is adapted
       between stages towards a target acceptance rate (Betz et al. 2016
       advocate tuning ``beta`` instead of the fixed 0.2 of the original
       paper); the adaptation uses the previous stage only, so every stage
       kernel is fixed given the current samples, as in the adaptive-SMC
       construction.

    Compared with :func:`probstat.inference.smc.sequential_monte_carlo`, which
    controls the effective sample size, TMCMC controls the coefficient of
    variation, restarts chains only from distinct leaders with burn-in, and
    reports the evidence as the product of mean weights without the
    Lee-Whiteley variance estimate. Neither estimator of the evidence comes
    with a standard error here; judge it by replicating over seeds.

    Failures are returned, not raised, in ``converged`` / ``stop_reason``:
    ``"all_zero_likelihood"``, ``"single_positive_likelihood"``,
    ``"singular_covariance"`` (the weighted covariance is not positive
    definite; no jitter is added), ``"tempering_stalled"`` (no positive
    increment meets the target at double precision) and ``"max_stages"``.
    ``log_evidence`` is NaN unless the run converged, and the posterior
    summaries raise ``ValueError`` for a run that did not. Invalid arguments
    and a likelihood that returns NaN or ``+inf`` raise.

Bayesian model class selection
    :func:`model_class_selection` turns log evidences and prior model
    probabilities into posterior model probabilities and Bayes factors
    ([EQ-INF-TMCMC-005], Beck and Yuen 2004; Kass and Raftery 1995).
    :func:`decompose_log_evidence` splits ``log Z`` into the posterior mean of
    the log-likelihood (data fit) minus the Kullback-Leibler divergence of
    the posterior from the prior (information gain, Beck and Yuen's Ockham
    factor) ([EQ-INF-TMCMC-006]); the estimator is stated there.
    :func:`model_averaged_predictive` mixes posterior predictive draws over
    models by their posterior probabilities ([EQ-INF-TMCMC-007]).

Hierarchical calibration across repeated tests
    :func:`hierarchical_calibration` puts the model
    ``theta_i ~ N(mu, diag(sigma^2))`` on the per-test parameters of ``m``
    tests with hyperpriors on ``mu`` and ``sigma`` and samples all
    ``(mu, log sigma, theta_1, ..., theta_m)`` jointly with TMCMC
    ([EQ-INF-TMCMC-008]). The funnel between ``sigma`` and ``theta`` is
    handled by the global weighted covariance only approximately; the NUTS
    route of :mod:`probstat.inference.hierarchical` remains the better choice
    for the scalar-normal hierarchy when gradients are available.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_vector, cholesky, matvec

from .mcmc import LogDensity
from .smc import _log_sum_exp, _weighted_covariance, _weighted_mean

PriorSampler = Callable[[random.Random], Sequence[float]]
PredictiveSampler = Callable[[Vector, random.Random], Sequence[float]]

STOP_REASONS = (
    "converged",
    "all_zero_likelihood",
    "single_positive_likelihood",
    "singular_covariance",
    "tempering_stalled",
    "max_stages",
)
_BISECTION_STEPS = 200
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class TMCMCResult:
    """Samples, exponent schedule, evidence and status of a TMCMC run.

    All per-stage tuples have one entry per *completed* stage except
    ``exponents`` and ``mean_log_likelihoods`` which also hold the prior stage.

    Args:
        names: Coordinate names.
        particles: Samples of the last completed stage; they follow the
            posterior only when ``converged`` (otherwise they follow
            ``pi_j`` with ``j`` the last exponent).
        log_likelihoods: ``log L`` at each sample.
        log_priors: ``log p`` at each sample.
        exponents: ``q_0 = 0, q_1, ..., q_m``.
        log_evidence_increments: ``log mean_i w_i`` of each completed stage.
        effective_sample_sizes: ``(sum w)^2 / sum w^2`` of each stage.
        coefficients_of_variation: Achieved coefficient of variation of the
            weights (over the samples with positive likelihood) per stage.
        acceptance_rates: Metropolis acceptance rate per stage (burn-in
            steps included).
        proposal_scales: The factor ``beta`` used in each stage.
        leaders: Number of distinct chain leaders per stage.
        positive_counts: Samples with positive likelihood at stage start.
        mean_log_likelihoods: ``mean log L`` of the samples at each exponent
            (NaN where a sample has zero likelihood); a diagnostic of how
            fast the data fit grows along the path.
        log_evidence: ``log Z_hat`` ([EQ-INF-TMCMC-003]); NaN unless
            ``converged``.
        converged: Whether the exponent reached one.
        stop_reason: One of :data:`STOP_REASONS`.
        cov_target: Target coefficient of variation.
        burn_in: Discarded Metropolis steps per chain and stage.
        thinning: Metropolis steps between recorded samples.
        evaluations: Log-likelihood evaluations.

    References:
        [REF-CHING-CHEN-2007], [REF-BETZ-PAPAIOANNOU-STRAUB-2016].

    """

    names: tuple[str, ...]
    particles: tuple[Vector, ...]
    log_likelihoods: tuple[float, ...]
    log_priors: tuple[float, ...]
    exponents: tuple[float, ...]
    log_evidence_increments: tuple[float, ...]
    effective_sample_sizes: tuple[float, ...]
    coefficients_of_variation: tuple[float, ...]
    acceptance_rates: tuple[float, ...]
    proposal_scales: tuple[float, ...]
    leaders: tuple[int, ...]
    positive_counts: tuple[int, ...]
    mean_log_likelihoods: tuple[float, ...]
    log_evidence: float
    converged: bool
    stop_reason: str
    cov_target: float
    burn_in: int
    thinning: int
    evaluations: int

    @property
    def stages(self) -> int:
        """Return the number of completed stages.

        References:
            [REF-CHING-CHEN-2007]. No equation is evaluated.

        """
        return len(self.exponents) - 1

    @property
    def dimension(self) -> int:
        """Return the state dimension.

        References:
            [REF-CHING-CHEN-2007]. No equation is evaluated.

        """
        return len(self.names)

    def _require_converged(self) -> None:
        """Raise unless the run reached exponent one.

        Raises:
            ValueError: If ``converged`` is false.

        References:
            [EQ-INF-TMCMC-001]. No new equation is introduced.

        """
        if not self.converged:
            raise ValueError(
                f"TMCMC did not converge (stop_reason={self.stop_reason!r}); "
                "posterior summaries are unavailable"
            )

    @property
    def expected_log_likelihood(self) -> float:
        """Return the sample mean of ``log L`` at the posterior samples.

        Raises:
            ValueError: If the run did not converge.

        Equation:
            [EQ-INF-TMCMC-006].

        References:
            [REF-BECK-YUEN-2004].

        """
        self._require_converged()
        return math.fsum(self.log_likelihoods) / len(self.log_likelihoods)

    def mean(self) -> Vector:
        """Return the posterior sample mean.

        Raises:
            ValueError: If the run did not converge.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-CHING-CHEN-2007].

        """
        self._require_converged()
        count = len(self.particles)
        return _weighted_mean(self.particles, tuple(1.0 / count for _ in range(count)))

    def covariance(self) -> Matrix:
        """Return the posterior sample covariance (divisor ``N``).

        Raises:
            ValueError: If the run did not converge.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-CHING-CHEN-2007].

        """
        self._require_converged()
        count = len(self.particles)
        return _weighted_covariance(
            self.particles, tuple(1.0 / count for _ in range(count))
        )


def _coefficient_of_variation(values: Sequence[float], increment: float) -> float:
    """Return the coefficient of variation of ``exp(increment * values)``.

    Uses the population standard deviation (divisor ``n``) and a shift by the
    maximum for overflow safety; the coefficient of variation is shift
    invariant.

    Args:
        values: Finite log-likelihoods.
        increment: Non-negative exponent increment.

    Returns:
        ``sqrt(max(0, mean(w^2) / mean(w)^2 - 1))``.

    Equation:
        [EQ-INF-TMCMC-002].

    References:
        [REF-CHING-CHEN-2007].

    """
    top = max(values)
    weights = [math.exp(increment * (value - top)) for value in values]
    count = len(weights)
    first = math.fsum(weights) / count
    second = math.fsum(w * w for w in weights) / count
    return math.sqrt(max(0.0, second / (first * first) - 1.0))


def _next_exponent(
    log_likelihoods: Sequence[float], current: float, target: float
) -> float | None:
    """Return the next exponent by bisection on the coefficient of variation.

    Finds the largest ``q' in (current, 1]`` with
    ``CoV(exp((q' - current) l_i)) <= target`` (``1.0`` if the full step
    already satisfies it). The coefficient of variation is non-decreasing in
    the increment because the cumulant function ``K`` of ``l`` is convex with
    ``K(0) = 0``, so ``K(2 d) - 2 K(d)`` is non-decreasing for ``d >= 0``.

    Args:
        log_likelihoods: Finite ``l_i = log L(x_i)``.
        current: Current exponent in ``[0, 1)``.
        target: Target coefficient of variation, positive.

    Returns:
        Next exponent, or ``None`` if no positive increment meets the target at
        double precision.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-INF-TMCMC-002].

    References:
        [REF-CHING-CHEN-2007], [REF-BETZ-PAPAIOANNOU-STRAUB-2016].

    """
    beta = require_finite_real(current, name="current")
    if not 0.0 <= beta < 1.0:
        raise ValueError("current must lie in [0, 1)")
    aim = require_finite_real(target, name="target")
    if aim <= 0.0:
        raise ValueError("target must be positive")
    values = tuple(float(value) for value in log_likelihoods)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("log_likelihoods must be a non-empty finite sequence")
    remaining = 1.0 - beta
    if _coefficient_of_variation(values, remaining) <= aim:
        return 1.0
    low, high = 0.0, remaining
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _coefficient_of_variation(values, middle) <= aim:
            low = middle
        else:
            high = middle
    if low <= 0.0:
        return None
    return min(1.0, beta + low)


def _checked_value(value: float, label: str) -> float:
    """Return ``value`` as a float, rejecting NaN and ``+inf``.

    Raises:
        ValueError: If the value is NaN or ``+inf``.

    References:
        [EQ-INF-TMCMC-001]. No new equation is introduced.

    """
    number = float(value)
    if math.isnan(number) or number == math.inf:
        raise ValueError(f"{label} returned NaN or +inf")
    return number


def transitional_mcmc(
    prior_sampler: PriorSampler,
    log_prior: LogDensity,
    log_likelihood: LogDensity,
    *,
    particles: int = 1000,
    cov_target: float = 1.0,
    burn_in: int = 5,
    thinning: int = 1,
    initial_scale: float = 0.2,
    target_acceptance: float = 0.25,
    adaptation_gain: float = 4.0,
    max_stages: int = 200,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> TMCMCResult:
    """Sample a posterior and estimate its evidence by transitional MCMC.

    See the module docstring for the algorithm, the failure statuses and the
    evidence estimator. ``log_likelihood`` must contain every normalizing
    constant of the data density if ``log_evidence`` is to be the marginal
    likelihood used for model comparison. ``log_prior`` need only be
    proportional to the density of ``prior_sampler`` (the constant cancels in
    the Metropolis ratio); the evidence estimator uses prior samples, never
    the prior density value.

    Args:
        prior_sampler: ``generator -> x`` drawing one prior sample from a
            ``random.Random``; the draws must have positive prior density.
        log_prior: Prior log-density (``-inf`` outside the support).
        log_likelihood: ``log L(x)``, ``-inf`` allowed.
        particles: Number of samples ``N`` (at least 2).
        cov_target: Target coefficient of variation of the plausibility
            weights, positive (Ching and Chen use 1).
        burn_in: Discarded Metropolis steps per chain and stage (>= 0).
        thinning: Metropolis steps between recorded samples (>= 1).
        initial_scale: Initial proposal factor ``beta > 0``.
        target_acceptance: Acceptance target of the adaptation, in ``(0, 1)``.
        adaptation_gain: ``g >= 0`` in
            ``log beta += g (rate - target_acceptance)`` between stages.
        max_stages: Maximum number of stages (positive).
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional distinct coordinate names (default ``x0, x1, ...``).

    Returns:
        Samples, schedule, per-stage diagnostics, ``log_evidence`` and status.

    Raises:
        TypeError: If a density or the prior sampler is not callable.
        ValueError: For invalid options, an inconsistent prior sampler, or a
            likelihood or prior returning NaN or ``+inf``.

    Equation:
        [EQ-INF-TMCMC-001], [EQ-INF-TMCMC-002], [EQ-INF-TMCMC-003],
        [EQ-INF-TMCMC-004].

    References:
        [REF-CHING-CHEN-2007], [REF-BETZ-PAPAIOANNOU-STRAUB-2016].

    """
    for function, label in (
        (prior_sampler, "prior_sampler"),
        (log_prior, "log_prior"),
        (log_likelihood, "log_likelihood"),
    ):
        if not callable(function):
            raise TypeError(f"{label} must be callable")
    count = require_non_negative_integer(particles, name="particles")
    if count < 2:
        raise ValueError("particles must be at least 2")
    aim = require_finite_real(cov_target, name="cov_target")
    if aim <= 0.0:
        raise ValueError("cov_target must be positive")
    burn = require_non_negative_integer(burn_in, name="burn_in")
    spacing = require_non_negative_integer(thinning, name="thinning")
    if spacing == 0:
        raise ValueError("thinning must be positive")
    scale0 = require_finite_real(initial_scale, name="initial_scale")
    if scale0 <= 0.0:
        raise ValueError("initial_scale must be positive")
    accept_aim = require_finite_real(target_acceptance, name="target_acceptance")
    if not 0.0 < accept_aim < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    gain = require_finite_real(adaptation_gain, name="adaptation_gain")
    if gain < 0.0:
        raise ValueError("adaptation_gain must be non-negative")
    stage_limit = require_non_negative_integer(max_stages, name="max_stages")
    if stage_limit == 0:
        raise ValueError("max_stages must be positive")
    stream = resolve_stream(seed)

    prior_generator = stream.child("prior").generator()
    points = [
        as_vector(prior_sampler(prior_generator), name="prior draw")
        for _ in range(count)
    ]
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError("prior draws must share a positive dimension")
    labels = tuple(f"x{i}" for i in range(dimension)) if names is None else tuple(names)
    if len(labels) != dimension or len(set(labels)) != dimension:
        raise ValueError("names must be distinct and match the dimension")

    evaluations = 0

    def likelihood(point: Vector) -> float:
        """Evaluate and validate one log-likelihood value.

        References:
            [EQ-INF-TMCMC-001]. No new equation is introduced.

        """
        nonlocal evaluations
        evaluations += 1
        return _checked_value(log_likelihood(point), "log_likelihood")

    def prior(point: Vector) -> float:
        """Evaluate and validate one prior log-density value.

        References:
            [EQ-INF-TMCMC-004]. No new equation is introduced.

        """
        return _checked_value(log_prior(point), "log_prior")

    logprior = [prior(point) for point in points]
    if any(value == -math.inf for value in logprior):
        raise ValueError("prior_sampler returned a point with zero prior density")
    loglik = [likelihood(point) for point in points]

    exponents = [0.0]
    increments: list[float] = []
    ess_values: list[float] = []
    cov_values: list[float] = []
    rates: list[float] = []
    scales: list[float] = []
    leader_counts: list[int] = []
    positive_counts: list[int] = []
    mean_loglik = [_mean_or_nan(loglik)]
    log_scale = math.log(scale0)
    exponent = 0.0
    reason = "converged"

    while exponent < 1.0:
        if len(exponents) > stage_limit:
            reason = "max_stages"
            break
        positive = [i for i, value in enumerate(loglik) if value != -math.inf]
        if len(positive) == 0:
            reason = "all_zero_likelihood"
            break
        if len(positive) == 1:
            reason = "single_positive_likelihood"
            break
        new_exponent = _next_exponent([loglik[i] for i in positive], exponent, aim)
        if new_exponent is None:
            reason = "tempering_stalled"
            break
        delta = new_exponent - exponent
        log_weights = [
            delta * value if value != -math.inf else -math.inf for value in loglik
        ]
        total = _log_sum_exp(log_weights)
        weights = [math.exp(value - total) for value in log_weights]
        scale = math.exp(log_scale)
        covariance = _weighted_covariance(points, weights)
        try:
            lower = cholesky(
                tuple(
                    tuple(scale * scale * covariance[i][j] for j in range(dimension))
                    for i in range(dimension)
                )
            )
        except ValueError:
            reason = "singular_covariance"
            break

        stage = len(exponents)
        stage_stream = stream.child("stage", stage)
        ancestors = (
            stage_stream.child("resample")
            .generator()
            .choices(range(count), weights=weights, k=count)
        )
        multiplicity: dict[int, int] = {}
        for ancestor in ancestors:
            multiplicity[ancestor] = multiplicity.get(ancestor, 0) + 1
        generator = stage_stream.child("chains").generator()
        new_points: list[Vector] = []
        new_loglik: list[float] = []
        new_logprior: list[float] = []
        proposals = 0
        accepted = 0
        for leader in sorted(multiplicity):
            state, state_loglik, state_prior = (
                points[leader],
                loglik[leader],
                logprior[leader],
            )
            for step in range(burn + multiplicity[leader] * spacing):
                proposals += 1
                z = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
                proposal = tuple(
                    a + b for a, b in zip(state, matvec(lower, z), strict=True)
                )
                proposal_prior = prior(proposal)
                if proposal_prior != -math.inf:
                    proposal_loglik = likelihood(proposal)
                    log_ratio = (
                        proposal_prior
                        + new_exponent * proposal_loglik
                        - state_prior
                        - new_exponent * state_loglik
                    )
                    if math.log(1.0 - generator.random()) < log_ratio:
                        state, state_loglik, state_prior = (
                            proposal,
                            proposal_loglik,
                            proposal_prior,
                        )
                        accepted += 1
                recorded = step - burn
                if recorded >= 0 and (recorded + 1) % spacing == 0:
                    new_points.append(state)
                    new_loglik.append(state_loglik)
                    new_logprior.append(state_prior)

        rate = accepted / proposals
        increments.append(total - math.log(count))
        ess_values.append(1.0 / math.fsum(w * w for w in weights))
        cov_values.append(
            _coefficient_of_variation([loglik[i] for i in positive], delta)
        )
        rates.append(rate)
        scales.append(scale)
        leader_counts.append(len(multiplicity))
        positive_counts.append(len(positive))
        points, loglik, logprior = new_points, new_loglik, new_logprior
        exponents.append(new_exponent)
        mean_loglik.append(_mean_or_nan(loglik))
        log_scale += gain * (rate - accept_aim)
        exponent = new_exponent

    converged = reason == "converged"
    return TMCMCResult(
        labels,
        tuple(points),
        tuple(loglik),
        tuple(logprior),
        tuple(exponents),
        tuple(increments),
        tuple(ess_values),
        tuple(cov_values),
        tuple(rates),
        tuple(scales),
        tuple(leader_counts),
        tuple(positive_counts),
        tuple(mean_loglik),
        math.fsum(increments) if converged else math.nan,
        converged,
        reason,
        aim,
        burn,
        spacing,
        evaluations,
    )


def _mean_or_nan(values: Sequence[float]) -> float:
    """Return the mean of ``values``, NaN if any value is ``-inf``.

    References:
        [EQ-INF-TMCMC-001]. No new equation is introduced.

    """
    if any(value == -math.inf for value in values):
        return math.nan
    return math.fsum(values) / len(values)


@dataclass(frozen=True, slots=True)
class EvidenceDecomposition:
    """Split of the log evidence into data fit minus information gain.

    Args:
        log_evidence: ``log Z_hat`` of the TMCMC run.
        data_fit: Posterior mean of ``log L`` (higher is a better fit).
        information_gain: ``data_fit - log_evidence``, the estimated
            Kullback-Leibler divergence of the posterior from the prior; it
            is non-negative in exact arithmetic and can come out slightly
            negative from Monte Carlo error.
        data_fit_standard_error: ``sd(log L) / sqrt(N)`` over the posterior
            samples. It treats the samples as independent, which they are not
            (they come from Markov chains), so it is a lower bound on the true
            Monte Carlo error.

    References:
        [REF-BECK-YUEN-2004].

    """

    log_evidence: float
    data_fit: float
    information_gain: float
    data_fit_standard_error: float


def decompose_log_evidence(result: TMCMCResult) -> EvidenceDecomposition:
    """Split ``log Z`` into posterior-mean log-likelihood minus relative entropy.

    The identity is exact:
    ``log Z = E_post[log L] - KL(post || prior)``. The estimator uses the
    TMCMC ``log_evidence`` for the left side and the sample mean of ``log L``
    over the final samples for ``E_post[log L]``; the information gain is the
    difference, so it carries the Monte Carlo errors of both.

    Args:
        result: A converged TMCMC run.

    Returns:
        The decomposition.

    Raises:
        TypeError: If ``result`` is not a :class:`TMCMCResult`.
        ValueError: If the run did not converge.

    Equation:
        [EQ-INF-TMCMC-006].

    References:
        [REF-BECK-YUEN-2004].

    """
    if not isinstance(result, TMCMCResult):
        raise TypeError("result must be a TMCMCResult")
    fit = result.expected_log_likelihood
    count = len(result.log_likelihoods)
    variance = math.fsum((v - fit) ** 2 for v in result.log_likelihoods) / count
    return EvidenceDecomposition(
        result.log_evidence,
        fit,
        fit - result.log_evidence,
        math.sqrt(variance / count),
    )


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Posterior model probabilities and Bayes factors of a model class set.

    Args:
        names: Model names.
        log_evidences: ``log Z_k`` per model.
        prior_probabilities: Prior model probabilities.
        posterior_probabilities: Posterior model probabilities.

    References:
        [REF-BECK-YUEN-2004], [REF-KASS-RAFTERY-1995].

    """

    names: tuple[str, ...]
    log_evidences: tuple[float, ...]
    prior_probabilities: tuple[float, ...]
    posterior_probabilities: tuple[float, ...]

    def _index(self, name: str) -> int:
        """Return the position of ``name``.

        Raises:
            ValueError: If the name is unknown.

        References:
            [EQ-INF-TMCMC-005]. No new equation is introduced.

        """
        if name not in self.names:
            raise ValueError(f"unknown model {name!r}")
        return self.names.index(name)

    def log_bayes_factor(self, numerator: str, denominator: str) -> float:
        """Return ``log (Z_numerator / Z_denominator)``.

        Args:
            numerator: Model name.
            denominator: Model name.

        Returns:
            The log Bayes factor (``nan`` if both evidences are zero).

        Raises:
            ValueError: If a name is unknown.

        Equation:
            [EQ-INF-TMCMC-005].

        References:
            [REF-KASS-RAFTERY-1995].

        """
        a = self.log_evidences[self._index(numerator)]
        b = self.log_evidences[self._index(denominator)]
        if a == b == -math.inf:
            return math.nan
        return a - b

    def bayes_factor(self, numerator: str, denominator: str) -> float:
        """Return ``Z_numerator / Z_denominator`` (``inf`` on overflow).

        Args:
            numerator: Model name.
            denominator: Model name.

        Returns:
            The Bayes factor.

        Raises:
            ValueError: If a name is unknown.

        Equation:
            [EQ-INF-TMCMC-005].

        References:
            [REF-KASS-RAFTERY-1995].

        """
        value = self.log_bayes_factor(numerator, denominator)
        if math.isnan(value):
            return math.nan
        return math.exp(value) if value < 709.0 else math.inf

    @property
    def best(self) -> str:
        """Return the model with the largest posterior probability.

        References:
            [REF-BECK-YUEN-2004]. No equation is evaluated.

        """
        top = max(range(len(self.names)), key=lambda k: self.posterior_probabilities[k])
        return self.names[top]


def _prior_probabilities(
    names: tuple[str, ...], priors: Mapping[str, float] | Sequence[float] | None
) -> tuple[float, ...]:
    """Return validated prior model probabilities (uniform by default).

    Raises:
        ValueError: If the probabilities are invalid or do not sum to one.

    References:
        [EQ-INF-TMCMC-005]. No new equation is introduced.

    """
    count = len(names)
    if priors is None:
        return tuple(1.0 / count for _ in range(count))
    if isinstance(priors, Mapping):
        if set(priors) != set(names):
            raise ValueError("prior_probabilities must have the model names as keys")
        values = tuple(float(priors[name]) for name in names)
    else:
        values = tuple(float(value) for value in priors)
        if len(values) != count:
            raise ValueError("prior_probabilities must have one entry per model")
    if any(not math.isfinite(v) or v < 0.0 for v in values):
        raise ValueError("prior probabilities must be finite and non-negative")
    if abs(math.fsum(values) - 1.0) > 1e-9:
        raise ValueError("prior probabilities must sum to one")
    return values


def model_class_selection(
    evidences: Mapping[str, float | TMCMCResult],
    prior_probabilities: Mapping[str, float] | Sequence[float] | None = None,
) -> ModelSelection:
    """Return posterior model probabilities from log evidences.

    ``P(M_k | D) = P(M_k) Z_k / sum_l P(M_l) Z_l`` evaluated with a
    log-sum-exp so that evidences differing by hundreds of nats do not
    overflow.

    Args:
        evidences: Model name to ``log Z_k`` (``-inf`` allowed) or to a
            converged :class:`TMCMCResult`.
        prior_probabilities: Prior model probabilities summing to one (a
            mapping with the model names or a sequence in the order of
            ``evidences``); uniform by default.

    Returns:
        Probabilities and Bayes factors.

    Raises:
        TypeError: If ``evidences`` is not a mapping.
        ValueError: For fewer than one model, a non-converged run, NaN or
            ``+inf`` evidence, invalid prior probabilities, or if every model
            has zero prior-weighted evidence.

    Equation:
        [EQ-INF-TMCMC-005].

    References:
        [REF-BECK-YUEN-2004], [REF-KASS-RAFTERY-1995].

    """
    if not isinstance(evidences, Mapping):
        raise TypeError("evidences must be a mapping from model name to evidence")
    if len(evidences) < 1:
        raise ValueError("at least one model is required")
    names = tuple(evidences)
    logs: list[float] = []
    for name in names:
        item = evidences[name]
        if isinstance(item, TMCMCResult):
            if not item.converged:
                raise ValueError(
                    f"model {name!r}: TMCMC did not converge "
                    f"(stop_reason={item.stop_reason!r})"
                )
            logs.append(item.log_evidence)
        else:
            value = float(item)
            if math.isnan(value) or value == math.inf:
                raise ValueError(f"model {name!r}: log evidence is NaN or +inf")
            logs.append(value)
    priors = _prior_probabilities(names, prior_probabilities)
    joint = [
        math.log(p) + z if p > 0.0 else -math.inf
        for p, z in zip(priors, logs, strict=True)
    ]
    total = _log_sum_exp(joint)
    if total == -math.inf:
        raise ValueError("every model has zero prior-weighted evidence")
    posterior = tuple(math.exp(value - total) for value in joint)
    return ModelSelection(names, tuple(logs), priors, posterior)


@dataclass(frozen=True, slots=True)
class ModelAveragedPredictive:
    """Posterior predictive draws mixed over model classes.

    Args:
        draws: Predictive vectors.
        model_names: The model that produced each draw.
        selection: The posterior model probabilities used as mixing weights.

    References:
        [REF-HOETING-ETAL-1999].

    """

    draws: tuple[Vector, ...]
    model_names: tuple[str, ...]
    selection: ModelSelection

    def mean(self) -> Vector:
        """Return the sample mean of the mixture.

        Equation:
            [EQ-INF-TMCMC-007].

        References:
            [REF-HOETING-ETAL-1999].

        """
        count = len(self.draws)
        return _weighted_mean(self.draws, tuple(1.0 / count for _ in range(count)))

    def model_frequencies(self) -> dict[str, float]:
        """Return the fraction of draws from each model.

        Equation:
            [EQ-INF-TMCMC-007].

        References:
            [REF-HOETING-ETAL-1999].

        """
        count = len(self.draws)
        return {
            name: sum(1 for m in self.model_names if m == name) / count
            for name in self.selection.names
        }


def model_averaged_predictive(
    results: Mapping[str, TMCMCResult],
    predictors: Mapping[str, PredictiveSampler],
    *,
    prior_probabilities: Mapping[str, float] | Sequence[float] | None = None,
    draws: int = 1000,
    seed: int | RandomStream | None = None,
) -> ModelAveragedPredictive:
    """Draw from the model-averaged posterior predictive distribution.

    For each draw a model is chosen with its posterior probability, one of
    its posterior samples is chosen uniformly, and that model's
    ``predictors[name](theta, generator)`` produces the predictive vector
    (including observation noise if the predictive of new data is wanted).
    The draws are samples of
    ``sum_k P(M_k | D) p(y_new | D, M_k)`` ([EQ-INF-TMCMC-007]).

    Args:
        results: Converged TMCMC run per model.
        predictors: Same keys as ``results``; ``(theta, generator) -> vector``.
        prior_probabilities: Prior model probabilities (uniform by default).
        draws: Number of predictive draws (positive).
        seed: Integer seed or stream.

    Returns:
        The mixed draws with the model that produced each.

    Raises:
        TypeError: If a predictor is not callable or an argument has the
            wrong type.
        ValueError: If the keys differ, a run did not converge, or the
            predictive vectors have different lengths.

    Equation:
        [EQ-INF-TMCMC-007].

    References:
        [REF-HOETING-ETAL-1999], [REF-BECK-YUEN-2004].

    """
    if not isinstance(results, Mapping) or not isinstance(predictors, Mapping):
        raise TypeError("results and predictors must be mappings")
    if set(results) != set(predictors):
        raise ValueError("results and predictors must have the same model names")
    if any(not callable(f) for f in predictors.values()):
        raise TypeError("every predictor must be callable")
    count = require_non_negative_integer(draws, name="draws")
    if count == 0:
        raise ValueError("draws must be positive")
    selection = model_class_selection(results, prior_probabilities)
    generator = resolve_stream(seed).child("model-averaged-predictive").generator()
    chosen = generator.choices(
        range(len(selection.names)), weights=selection.posterior_probabilities, k=count
    )
    output: list[Vector] = []
    labels: list[str] = []
    for index in chosen:
        name = selection.names[index]
        pool = results[name].particles
        theta = pool[generator.randrange(len(pool))]
        output.append(as_vector(predictors[name](theta, generator), name="prediction"))
        labels.append(name)
    if any(len(item) != len(output[0]) for item in output):
        raise ValueError("predictive vectors must share one length")
    return ModelAveragedPredictive(tuple(output), tuple(labels), selection)


@dataclass(frozen=True, slots=True)
class HierarchicalCalibrationResult:
    """TMCMC posterior of a hierarchical calibration across repeated tests.

    The sampled state is ``(mu, [log sigma], theta_1, ..., theta_m)`` with
    ``mu`` and ``log sigma`` of length ``dimension`` (``log sigma`` absent
    when the between-test scale is fixed).

    Args:
        tmcmc: The underlying :class:`TMCMCResult` (check ``converged``).
        tests: Number of tests ``m``.
        dimension: Parameters per test.
        scale_fixed: Whether the between-test scale was fixed by the user.
        fixed_scale: The fixed scale (empty tuple when sampled).

    References:
        [REF-GELMAN-ETAL-2013].

    """

    tmcmc: TMCMCResult
    tests: int
    dimension: int
    scale_fixed: bool
    fixed_scale: Vector

    @property
    def log_evidence(self) -> float:
        """Return the TMCMC log evidence of the hierarchical model.

        References:
            [EQ-INF-TMCMC-003]. No new equation is introduced.

        """
        return self.tmcmc.log_evidence

    def _blocks(self) -> tuple[int, int, int]:
        """Return the offsets of ``(log sigma, theta_1)`` and the state length.

        References:
            [EQ-INF-TMCMC-008]. No new equation is introduced.

        """
        d = self.dimension
        sigma_start = d
        theta_start = d if self.scale_fixed else 2 * d
        return sigma_start, theta_start, theta_start + self.tests * d

    def hyper_mean_draws(self) -> tuple[Vector, ...]:
        """Return the posterior draws of the population mean ``mu``.

        Raises:
            ValueError: If the run did not converge.

        Equation:
            [EQ-INF-TMCMC-008].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        self.tmcmc._require_converged()
        d = self.dimension
        return tuple(p[:d] for p in self.tmcmc.particles)

    def hyper_scale_draws(self) -> tuple[Vector, ...]:
        """Return the posterior draws of the population scale ``sigma``.

        Raises:
            ValueError: If the run did not converge.

        Equation:
            [EQ-INF-TMCMC-008].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        self.tmcmc._require_converged()
        if self.scale_fixed:
            return tuple(self.fixed_scale for _ in self.tmcmc.particles)
        start, end, _ = self._blocks()
        return tuple(
            tuple(math.exp(v) for v in p[start:end]) for p in self.tmcmc.particles
        )

    def test_draws(self, index: int) -> tuple[Vector, ...]:
        """Return the posterior draws of the parameters of test ``index``.

        Args:
            index: Zero-based test index.

        Raises:
            ValueError: If the index is out of range or the run did not
                converge.

        Equation:
            [EQ-INF-TMCMC-008].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        self.tmcmc._require_converged()
        if not 0 <= index < self.tests:
            raise ValueError("test index out of range")
        d = self.dimension
        _, base, _ = self._blocks()
        lo = base + index * d
        return tuple(p[lo : lo + d] for p in self.tmcmc.particles)

    def new_test_draws(
        self, count: int, *, seed: int | RandomStream | None = None
    ) -> tuple[Vector, ...]:
        """Draw parameters of a new, unobserved test from the population.

        Each draw takes one posterior sample ``(mu, sigma)`` and then
        ``theta ~ N(mu, diag(sigma^2))``, the posterior predictive of the
        parameters of a future test.

        Args:
            count: Number of draws (positive).
            seed: Integer seed or stream.

        Raises:
            ValueError: If ``count`` is not positive or the run did not
                converge.

        Equation:
            [EQ-INF-TMCMC-008].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        number = require_non_negative_integer(count, name="count")
        if number == 0:
            raise ValueError("count must be positive")
        means = self.hyper_mean_draws()
        scales = self.hyper_scale_draws()
        generator = resolve_stream(seed).child("new-test").generator()
        output: list[Vector] = []
        for _ in range(number):
            k = generator.randrange(len(means))
            output.append(
                tuple(
                    generator.gauss(m, s)
                    for m, s in zip(means[k], scales[k], strict=True)
                )
            )
        return tuple(output)


def _per_dimension(value: float | Sequence[float], dimension: int, name: str) -> Vector:
    """Broadcast a scalar or validate a sequence to a vector of length ``d``.

    Raises:
        ValueError: If the length is wrong or a value is not finite.

    References:
        [EQ-INF-TMCMC-008]. No new equation is introduced.

    """
    if isinstance(value, int | float):
        return tuple(require_finite_real(value, name=name) for _ in range(dimension))
    vector = as_vector(value, name=name)
    if len(vector) != dimension:
        raise ValueError(f"{name} must have length {dimension}")
    return vector


def _log_normal_density(x: float, mean: float, std: float) -> float:
    """Return ``log N(x; mean, std^2)``.

    References:
        [EQ-INF-TMCMC-008]. No new equation is introduced.

    """
    z = (x - mean) / std
    return -0.5 * z * z - math.log(std) - _LOG_SQRT_2PI


def hierarchical_calibration(
    test_log_likelihoods: Sequence[LogDensity],
    *,
    dimension: int = 1,
    mu_mean: float | Sequence[float],
    mu_std: float | Sequence[float],
    fixed_scale: float | Sequence[float] | None = None,
    log_scale_mean: float | Sequence[float] | None = None,
    log_scale_std: float | Sequence[float] | None = None,
    particles: int = 1000,
    cov_target: float = 1.0,
    burn_in: int = 5,
    thinning: int = 1,
    initial_scale: float = 0.2,
    target_acceptance: float = 0.25,
    adaptation_gain: float = 4.0,
    max_stages: int = 200,
    seed: int | RandomStream | None = None,
) -> HierarchicalCalibrationResult:
    """Calibrate per-test parameters hierarchically with TMCMC.

    The model for tests ``i = 1..m`` with data ``D_i`` is
    ``theta_i | mu, sigma ~ N(mu, diag(sigma^2))``,
    ``mu_k ~ N(mu_mean_k, mu_std_k^2)`` and either a known ``sigma`` or
    ``log sigma_k ~ N(log_scale_mean_k, log_scale_std_k^2)`` (a log-normal
    hyperprior on the between-test standard deviation), and
    ``L_i(theta_i) = p(D_i | theta_i)`` independent across tests
    ([EQ-INF-TMCMC-008]). The prior is sampled ancestrally and its exact
    log-density is used in the Metropolis ratio. ``log_likelihood`` of each
    test should be fully normalized if the evidence of the hierarchical model
    is to be compared with another model.

    Args:
        test_log_likelihoods: One ``theta_i -> log p(D_i | theta_i)`` per test
            (each takes a vector of length ``dimension``).
        dimension: Parameters per test (positive).
        mu_mean: Prior mean of ``mu`` (scalar or per-parameter).
        mu_std: Prior standard deviation of ``mu`` (positive).
        fixed_scale: Known ``sigma`` (positive); exclusive with the
            ``log_scale_*`` pair.
        log_scale_mean: Prior mean of ``log sigma``.
        log_scale_std: Prior standard deviation of ``log sigma`` (positive).
        particles: TMCMC sample count.
        cov_target: See :func:`transitional_mcmc`.
        burn_in: See :func:`transitional_mcmc`.
        thinning: See :func:`transitional_mcmc`.
        initial_scale: See :func:`transitional_mcmc`.
        target_acceptance: See :func:`transitional_mcmc`.
        adaptation_gain: See :func:`transitional_mcmc`.
        max_stages: See :func:`transitional_mcmc`.
        seed: Integer seed or stream.

    Returns:
        Hierarchical posterior wrapper around the TMCMC run.

    Raises:
        TypeError: If a likelihood is not callable.
        ValueError: For invalid hyperparameters, no tests, or ambiguous scale
            specification.

    Equation:
        [EQ-INF-TMCMC-008].

    References:
        [REF-CHING-CHEN-2007], [REF-GELMAN-ETAL-2013].

    """
    tests = tuple(test_log_likelihoods)
    if not tests:
        raise ValueError("at least one test is required")
    if any(not callable(f) for f in tests):
        raise TypeError("every test log-likelihood must be callable")
    d = require_non_negative_integer(dimension, name="dimension")
    if d == 0:
        raise ValueError("dimension must be positive")
    m = len(tests)
    mean0 = _per_dimension(mu_mean, d, "mu_mean")
    std0 = _per_dimension(mu_std, d, "mu_std")
    if any(s <= 0.0 for s in std0):
        raise ValueError("mu_std must be positive")
    scale_fixed = fixed_scale is not None
    sampled_pair = (log_scale_mean, log_scale_std)
    if scale_fixed == any(item is not None for item in sampled_pair):
        raise ValueError(
            "give either fixed_scale or both log_scale_mean and log_scale_std"
        )
    fixed: Vector = ()
    ls_mean: Vector = ()
    ls_std: Vector = ()
    if fixed_scale is not None:
        fixed = _per_dimension(fixed_scale, d, "fixed_scale")
        if any(s <= 0.0 for s in fixed):
            raise ValueError("fixed_scale must be positive")
    else:
        if log_scale_mean is None or log_scale_std is None:
            raise ValueError("log_scale_mean and log_scale_std are both required")
        ls_mean = _per_dimension(log_scale_mean, d, "log_scale_mean")
        ls_std = _per_dimension(log_scale_std, d, "log_scale_std")
        if any(s <= 0.0 for s in ls_std):
            raise ValueError("log_scale_std must be positive")
    theta_start = d if scale_fixed else 2 * d

    def split(x: Vector) -> tuple[Vector, Vector, list[Vector]]:
        """Return ``(mu, sigma, [theta_i])`` from a state vector.

        References:
            [EQ-INF-TMCMC-008]. No new equation is introduced.

        """
        sigma = fixed if scale_fixed else tuple(math.exp(v) for v in x[d : 2 * d])
        thetas = [x[theta_start + i * d : theta_start + (i + 1) * d] for i in range(m)]
        return x[:d], sigma, thetas

    def sampler(generator: random.Random) -> Vector:
        """Draw ``(mu, log sigma, theta)`` ancestrally from the hierarchy.

        References:
            [EQ-INF-TMCMC-008]. No new equation is introduced.

        """
        mu = tuple(generator.gauss(a, b) for a, b in zip(mean0, std0, strict=True))
        parts = list(mu)
        if scale_fixed:
            sigma = fixed
        else:
            log_sigma = tuple(
                generator.gauss(a, b) for a, b in zip(ls_mean, ls_std, strict=True)
            )
            parts.extend(log_sigma)
            sigma = tuple(math.exp(v) for v in log_sigma)
        for _ in range(m):
            parts.extend(generator.gauss(a, s) for a, s in zip(mu, sigma, strict=True))
        return tuple(parts)

    def log_prior(x: Vector) -> float:
        """Return the exact log-density of the hierarchical prior.

        References:
            [EQ-INF-TMCMC-008]. No new equation is introduced.

        """
        mu, sigma, thetas = split(x)
        total = math.fsum(
            _log_normal_density(v, a, b)
            for v, a, b in zip(mu, mean0, std0, strict=True)
        )
        if not scale_fixed:
            total += math.fsum(
                _log_normal_density(v, a, b)
                for v, a, b in zip(x[d : 2 * d], ls_mean, ls_std, strict=True)
            )
        for theta in thetas:
            total += math.fsum(
                _log_normal_density(t, a, s)
                for t, a, s in zip(theta, mu, sigma, strict=True)
            )
        return total

    def log_likelihood(x: Vector) -> float:
        """Return ``sum_i log p(D_i | theta_i)``.

        References:
            [EQ-INF-TMCMC-008]. No new equation is introduced.

        """
        thetas = split(x)[2]
        return math.fsum(f(theta) for f, theta in zip(tests, thetas, strict=True))

    labels = [f"mu{k}" for k in range(d)]
    if not scale_fixed:
        labels.extend(f"log_sigma{k}" for k in range(d))
    for i in range(m):
        labels.extend(f"theta{i}_{k}" for k in range(d))
    result = transitional_mcmc(
        sampler,
        log_prior,
        log_likelihood,
        particles=particles,
        cov_target=cov_target,
        burn_in=burn_in,
        thinning=thinning,
        initial_scale=initial_scale,
        target_acceptance=target_acceptance,
        adaptation_gain=adaptation_gain,
        max_stages=max_stages,
        seed=seed,
        names=tuple(labels),
    )
    return HierarchicalCalibrationResult(result, m, d, scale_fixed, fixed)
