"""Bayesian updating with structural reliability methods (BUS).

Rejection sampling from a posterior accepts a prior draw ``theta`` when an
independent ``p ~ U(0, 1)`` satisfies ``p <= c L(theta)``, where ``L`` is the
likelihood and ``c`` a constant with ``c L <= 1`` everywhere. BUS reads the
acceptance as a failure event in the augmented space ``(theta, p)``: accepted
points are posterior draws and the acceptance probability is ``c`` times the
model evidence. Any rare-event method can then replace plain rejection,
which matters when the acceptance probability is small.

* :func:`bus_limit_state` builds the limit state of the acceptance event on
  independent standard-normal inputs, usable by every model-based
  reliability method of this package.
* :func:`bus_subset_simulation` runs subset simulation on it with the
  conditional-sampling kernels of :mod:`simulation`, keeps the final states,
  and returns posterior draws, the evidence, and a check of ``c L <= 1``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest

from .model_based import ModelLimitState
from .simulation import (
    _acs_level,
    _chain_lengths,
    _cs_level,
    _level_cov,
    _require_samples,
    _require_unit_interval,
)

BUS_SAMPLERS = ("acs", "cs")
"""Subset-simulation kernels available to :func:`bus_subset_simulation`."""

BUS_MARGIN = "bus_margin"
"""Quantity name of the BUS limit-state value."""

BUS_SCALED_LIKELIHOOD = "bus_log_scaled_likelihood"
"""Quantity name of ``log c + log L(theta)``."""

BUS_VIOLATION_TOLERANCE = 1e-12
"""Value of ``log(c L)`` above which ``c L <= 1`` counts as violated."""

_CLAMP = 1e300
_STANDARD = Normal(0.0, 1.0)


def _require_log_likelihood(
    log_likelihood: object,
) -> Callable[[Vector], float]:
    """Return the argument when it is callable.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if not callable(log_likelihood):
        raise TypeError("log_likelihood must be callable")
    return log_likelihood


def _prior_dimension(prior: object, dimension: object) -> int:
    """Return the parameter dimension from the prior or the explicit count.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if prior is None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError("dimension must be given when prior is None")
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        return dimension
    if not isinstance(prior, JointInput):
        raise TypeError("prior must be a JointInput or None")
    if dimension is not None and dimension != prior.dimension:
        raise ValueError("dimension does not match the prior")
    return prior.dimension


def bus_limit_state(
    log_likelihood: Callable[[Vector], float],
    prior: JointInput | None = None,
    *,
    log_constant: float,
    dimension: int | None = None,
    name: str = "BUS",
) -> ModelLimitState:
    """Return the BUS acceptance event as a limit state in standard-normal space.

    The model has ``d + 1`` independent standard-normal inputs
    ``bus_u1, ..., bus_ud, bus_p``. The first ``d`` are mapped to the
    parameters by the prior's isoprobabilistic map, ``theta = T^-1(u)``
    (``theta = u`` without a prior, i.e. a standard-normal prior), and the
    last one carries the uniform variable ``p = Phi(u_p)``. The limit state

        ``g(u, u_p) = log Phi(u_p) - log c - log L(theta(u))``

    is non-positive exactly when ``p <= c L(theta)``. The source writes the
    event as ``p - c L(theta) <= 0``; the logarithm is a monotone transform
    of both sides, so the failure set is the same, and it keeps the
    intermediate thresholds of subset simulation on a usable scale when the
    likelihood varies over many orders of magnitude.

    If ``c L(theta) <= 1`` for every ``theta``, then
    ``P(g <= 0) = c Z`` with the evidence ``Z = E_prior[L]``, and ``theta``
    conditional on ``g <= 0`` follows the posterior. If ``c L > 1``
    somewhere, the conditional law is the posterior of the truncated
    likelihood ``min(L, 1/c)`` instead.

    The model also reports ``log c + log L(theta)`` as the quantity
    ``bus_log_scaled_likelihood`` so that a caller can check ``c L <= 1`` at
    the evaluated points. A zero likelihood (``log L = -inf``) gives the
    limit-state value ``1e300`` (never accepted).

    Args:
        log_likelihood: Function of the parameter vector ``theta`` (in the
            order of ``prior.names``) returning ``log L(theta)``; ``-inf`` is
            allowed.
        prior: Joint prior law of the parameters, or ``None`` for
            independent standard-normal parameters.
        log_constant: ``log c``. Posterior exactness needs
            ``log c <= -sup log L``.
        dimension: Number of parameters; required when ``prior`` is ``None``.
        name: Label of the limit state.

    Returns:
        Limit state with failure set ``g <= 0`` on ``d + 1`` standard-normal
        inputs.

    Raises:
        TypeError: If ``log_likelihood`` is not callable or ``prior`` has the
            wrong type.
        ValueError: If ``log_constant`` is not finite or the dimension is
            missing or inconsistent. Evaluations raise ``ValueError`` when
            the log-likelihood is ``nan`` or ``+inf``.

    Equation:
        [EQ-REL-BUS-001].

    References:
        [REF-STRAUB-PAPAIOANNOU-2015].

    """
    function = _require_log_likelihood(log_likelihood)
    size = _prior_dimension(prior, dimension)
    log_c = require_finite_real(log_constant, name="log_constant")
    names = tuple(f"bus_u{index + 1}" for index in range(size))

    def response(parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Return the BUS limit-state value and ``log(c L)`` for one input.

        References:
            [EQ-REL-BUS-001], [REF-STRAUB-PAPAIOANNOU-2015].

        """
        u = tuple(parameters[item] for item in names)
        theta = u if prior is None else prior.from_standard_normal(u)
        value = float(function(theta))
        if math.isnan(value) or value == math.inf:
            raise ValueError("log_likelihood must return a finite value or -inf")
        scaled = max(log_c + value, -_CLAMP)
        margin = _STANDARD.log_cdf(parameters["bus_p"]) - scaled
        return {
            BUS_MARGIN: min(margin, _CLAMP),
            BUS_SCALED_LIKELIHOOD: scaled,
        }

    inputs = JointInput(
        tuple(RandomInput(item, Normal(0.0, 1.0)) for item in (*names, "bus_p"))
    )
    model = ProbabilisticModel(
        inputs,
        response,
        (
            QuantityOfInterest(BUS_MARGIN, "BUS limit-state value log p - log(c L)"),
            QuantityOfInterest(
                BUS_SCALED_LIKELIHOOD, "scaled log-likelihood log c + log L"
            ),
        ),
        name=name,
    )
    return ModelLimitState(model, BUS_MARGIN, 0.0, "below", name)


@dataclass(frozen=True, slots=True)
class BUSResult:
    """Posterior draws and evidence from BUS with subset simulation.

    Args:
        samples: Posterior draws ``theta`` (correlated Markov-chain states
            with repeated values; empty when no state was accepted). They
            are not an independent sample: in a one-parameter conjugate
            problem about half of 1000 draws were distinct and the
            posterior mean scattered between runs like that of about 250
            independent draws (about 140 when ``c`` was 20 times smaller
            than necessary). :meth:`mean` and :meth:`covariance` come
            without a standard error.
        standard_samples: The same draws in the standard-normal space of the
            prior.
        log_likelihoods: ``log L`` at the draws.
        evidence: ``Z = P(p <= c L) / c``.
        log_evidence: ``log Z``.
        acceptance_probability: Subset-simulation estimate of
            ``P(p <= c L)``.
        coefficient_of_variation: C.o.v. of the acceptance probability, and
            therefore of the evidence, assuming uncorrelated levels
            [EQ-REL-SUBSET-003]; it typically understates the scatter.
        coefficient_of_variation_upper: The value for fully correlated
            levels.
        thresholds: Intermediate thresholds of the limit state, ending at 0.
        levels: Number of subset levels (including level 0).
        samples_per_level: Samples per level ``N``.
        final_level: Whether an additional conditional level at threshold 0
            was run so that ``samples`` holds ``N`` states.
        evaluations: Likelihood evaluations.
        converged: Whether the acceptance event was reached within
            ``max_levels``.
        sampler: ``"acs"`` or ``"cs"``.
        log_constant: ``log c`` used.
        max_log_scaled_likelihood: Largest ``log c + log L`` over every
            evaluated point.
        constant_violations: Number of evaluated points with
            ``log c + log L > BUS_VIOLATION_TOLERANCE``.
        constant_violated: ``True`` when ``constant_violations > 0``. The
            draws then follow the posterior of the truncated likelihood
            ``min(L, 1/c)`` and the evidence is biased low; rerun with
            ``log_constant <= log_constant - max_log_scaled_likelihood``.
            ``False`` only says that no evaluated point violated the bound.

    References:
        [REF-STRAUB-PAPAIOANNOU-2015], [REF-AU-BECK-2001].

    """

    samples: tuple[Vector, ...]
    standard_samples: tuple[Vector, ...]
    log_likelihoods: Vector
    evidence: float
    log_evidence: float
    acceptance_probability: float
    coefficient_of_variation: float
    coefficient_of_variation_upper: float
    thresholds: Vector
    levels: int
    samples_per_level: int
    final_level: bool
    evaluations: int
    converged: bool
    sampler: str
    log_constant: float
    max_log_scaled_likelihood: float
    constant_violations: int
    constant_violated: bool

    def mean(self) -> Vector:
        """Return the sample mean of the posterior draws.

        Raises:
            ValueError: If there are no draws.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-STRAUB-PAPAIOANNOU-2015].

        """
        if not self.samples:
            raise ValueError("the result holds no posterior draws")
        count = len(self.samples)
        return tuple(
            math.fsum(draw[i] for draw in self.samples) / count
            for i in range(len(self.samples[0]))
        )

    def covariance(self) -> Matrix:
        """Return the sample covariance (divisor ``n - 1``) of the draws.

        Raises:
            ValueError: If there are fewer than two draws.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-STRAUB-PAPAIOANNOU-2015].

        """
        count = len(self.samples)
        if count < 2:
            raise ValueError("at least two posterior draws are needed")
        centre = self.mean()
        size = len(centre)
        return tuple(
            tuple(
                math.fsum(
                    (draw[i] - centre[i]) * (draw[j] - centre[j])
                    for draw in self.samples
                )
                / (count - 1)
                for j in range(size)
            )
            for i in range(size)
        )


def bus_subset_simulation(
    log_likelihood: Callable[[Vector], float],
    prior: JointInput | None = None,
    *,
    log_constant: float,
    dimension: int | None = None,
    samples_per_level: int = 1000,
    level_probability: float = 0.1,
    sampler: str = "acs",
    correlation: float = 0.8,
    initial_scale: float = 0.6,
    target_acceptance: float = 0.44,
    max_levels: int = 20,
    final_level: bool = True,
    seed: int | RandomStream | None = None,
) -> BUSResult:
    """Draw from a posterior and estimate the evidence by BUS with subset simulation.

    Subset simulation is run on the limit state of :func:`bus_limit_state`
    with the conditional-sampling kernels of :func:`subset_simulation`
    (``"acs"`` adaptive, ``"cs"`` fixed correlation), using the same random
    stream, thresholds, and seeds; for the same options and seed the
    acceptance probability equals the ``probability`` that
    ``subset_simulation`` returns on that limit state. Unlike that function,
    the states of the last level are kept: those with ``g <= 0`` are
    posterior draws. With ``final_level=True`` they seed one more
    conditional level at threshold 0 (a random half of them when more than
    ``N / 2`` were accepted, because a chain needs two states), which gives
    ``N`` posterior states at the cost of ``N`` minus the number of seeds
    further evaluations.

    The evidence is ``Z = P(p <= c L) / c`` [EQ-REL-BUS-002]. Every
    evaluated point is checked for ``log c + log L <= 0``
    [EQ-REL-BUS-003]; a violation sets ``constant_violated`` and warns.
    The check covers the evaluated points only, so a clean check does not
    prove ``c <= 1 / sup L``.

    Args:
        log_likelihood: Function of ``theta`` returning ``log L(theta)``.
        prior: Joint prior law, or ``None`` for independent standard-normal
            parameters.
        log_constant: ``log c``; must not exceed ``-sup log L`` for exact
            posterior draws. A smaller value stays exact and lowers the
            acceptance probability by the same factor.
        dimension: Number of parameters; required when ``prior`` is ``None``.
        samples_per_level: Samples ``N`` per level; ``N p0`` must be a
            positive integer.
        level_probability: Conditional level probability ``p0`` in
            ``(0, 0.5]``.
        sampler: ``"acs"`` or ``"cs"`` (``BUS_SAMPLERS``).
        correlation: Fixed correlation in ``[0, 1)`` (``"cs"``).
        initial_scale: Starting scaling factor (``"acs"``).
        target_acceptance: Target acceptance rate in ``(0, 1)`` (``"acs"``).
        max_levels: Maximum number of subset levels.
        final_level: Whether to run the additional level at threshold 0.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Posterior draws, evidence with its c.o.v., level history, evaluation
        count, convergence flag, and the check of the constant.

    Raises:
        TypeError: If ``log_likelihood`` or ``prior`` has the wrong type.
        ValueError: If an option is invalid, the log-likelihood returns
            ``nan`` or ``+inf``, or limit-state values tie at the first
            threshold.

    Warns:
        UserWarning: If ``c L > 1`` at an evaluated point or the acceptance
            event was not reached within ``max_levels``.

    Equation:
        [EQ-REL-BUS-001], [EQ-REL-BUS-002], [EQ-REL-BUS-003],
        [EQ-REL-SUBSET-001], [EQ-REL-SUBSET-003], [EQ-REL-SUBSET-004],
        [EQ-REL-SUBSET-005].

    References:
        [REF-STRAUB-PAPAIOANNOU-2015], [REF-AU-BECK-2001],
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

    """
    state = bus_limit_state(
        log_likelihood, prior, log_constant=log_constant, dimension=dimension
    )
    log_c = require_finite_real(log_constant, name="log_constant")
    if not isinstance(sampler, str) or sampler not in BUS_SAMPLERS:
        raise ValueError(f"sampler must be one of {BUS_SAMPLERS}")
    total = _require_samples(samples_per_level, "samples_per_level", minimum=10)
    p0 = require_finite_real(level_probability, name="level_probability")
    if not 0.0 < p0 <= 0.5:
        raise ValueError("level_probability must lie in (0, 0.5]")
    seeds_count = round(total * p0)
    if seeds_count < 1 or abs(seeds_count - total * p0) > 1e-9:
        raise ValueError("samples_per_level * level_probability must be an integer")
    rho = _require_unit_interval(correlation, "correlation", closed_low=True)
    scale = require_finite_real(initial_scale, name="initial_scale")
    if scale <= 0.0:
        raise ValueError("initial_scale must be positive")
    target = _require_unit_interval(
        target_acceptance, "target_acceptance", closed_low=False
    )
    levels_limit = _require_samples(max_levels, "max_levels", minimum=1)
    if not isinstance(final_level, bool):
        raise TypeError("final_level must be a bool")
    window = max(1, math.ceil(100.0 * seeds_count / total - 1e-9))
    model = state.model
    size = model.dimension - 1
    scaled_values: dict[Vector, float] = {}
    worst = [-math.inf]
    violations = [0]

    def evaluate(point: Vector) -> float:
        """Return the limit-state value and record ``log(c L)`` of the point.

        References:
            [EQ-REL-BUS-001], [EQ-REL-BUS-003].

        """
        outputs = model.evaluate(model.inputs.from_standard_normal(point))
        scaled = outputs[BUS_SCALED_LIKELIHOOD]
        scaled_values[point] = scaled
        if scaled > worst[0]:
            worst[0] = scaled
        if scaled > BUS_VIOLATION_TOLERANCE:
            violations[0] += 1
        return outputs[BUS_MARGIN]

    def level(
        seeds: list[tuple[Vector, float]], threshold: float, lengths: list[int]
    ) -> tuple[list[Vector], list[float], list[int]]:
        """Populate one conditional level and update the carried scale.

        References:
            [EQ-REL-SUBSET-004], [EQ-REL-SUBSET-005],
            [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

        """
        if sampler == "cs":
            sample = _cs_level(evaluate, generator, seeds, threshold, lengths, rho)
        else:
            sample = _acs_level(
                evaluate,
                generator,
                seeds,
                threshold,
                lengths,
                carry[0],
                "unit",
                window,
                target,
            )
            carry[0] = sample.carry
        return sample.points, sample.values, sample.lengths

    generator = resolve_stream(seed).child("subset-simulation").generator()
    carry = [scale]
    before = model.evaluations
    points: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(size + 1)) for _ in range(total)
    ]
    values = [evaluate(u) for u in points]
    lengths = _chain_lengths(total, seeds_count)
    thresholds: list[float] = []
    level_covs: list[float] = []
    product = 1.0
    index = 0
    while True:
        correlated = index > 0
        failures = sum(1 for value in values if value <= 0.0)
        if failures >= seeds_count or index + 1 >= levels_limit:
            fraction = failures / total
            thresholds.append(0.0)
            level_covs.append(
                _level_cov(
                    [value <= 0.0 for value in values],
                    seeds_count,
                    fraction,
                    correlated,
                    lengths,
                )
            )
            probability = product * fraction
            converged = failures >= seeds_count
            break
        order = sorted(range(total), key=values.__getitem__)
        last_inside = values[order[seeds_count - 1]]
        first_outside = values[order[seeds_count]]
        if last_inside == first_outside and not correlated:
            raise ValueError(
                "limit-state values tie at the first threshold; BUS with subset "
                "simulation requires a continuous likelihood under the prior"
            )
        threshold = 0.5 * (last_inside + first_outside)
        thresholds.append(threshold)
        level_covs.append(
            _level_cov(
                [value <= threshold for value in values],
                seeds_count,
                p0,
                correlated,
                lengths,
            )
        )
        product *= p0
        seeds = [(points[item], values[item]) for item in order[:seeds_count]]
        points, values, lengths = level(seeds, threshold, lengths)
        index += 1
    accepted = sorted(
        (
            (point, value)
            for point, value in zip(points, values, strict=True)
            if value <= 0.0
        ),
        key=lambda item: item[1],
    )
    extra = final_level and converged and len(accepted) < total
    if extra:
        # Chains need at least two states, so at most N / 2 seeds are used.
        if len(accepted) > total // 2:
            accepted = generator.sample(accepted, total // 2)
        final_points, _, _ = level(accepted, 0.0, _chain_lengths(total, len(accepted)))
        draws = final_points
    else:
        draws = [
            point for point, value in zip(points, values, strict=True) if value <= 0.0
        ]
    standard = tuple(point[:size] for point in draws)
    if prior is None:
        physical = standard
    else:
        physical = tuple(prior.from_standard_normal(u) for u in standard)
    log_likelihoods = tuple(scaled_values[point] - log_c for point in draws)
    squared = math.fsum(delta * delta for delta in level_covs)
    log_evidence = math.log(probability) - log_c if probability > 0.0 else -math.inf
    constant_violated = violations[0] > 0
    if constant_violated:
        warnings.warn(
            f"c L > 1 at {violations[0]} evaluated points (largest log(c L) = "
            f"{worst[0]:.3g}); the draws follow a truncated likelihood. Lower "
            "log_constant by at least that amount",
            stacklevel=2,
        )
    if not converged:
        warnings.warn(
            f"BUS did not reach the acceptance event within {levels_limit} "
            "levels; the evidence and the draws are unreliable",
            stacklevel=2,
        )
    return BUSResult(
        samples=physical,
        standard_samples=standard,
        log_likelihoods=log_likelihoods,
        evidence=math.exp(log_evidence),
        log_evidence=log_evidence,
        acceptance_probability=probability,
        coefficient_of_variation=math.sqrt(squared),
        coefficient_of_variation_upper=math.fsum(level_covs),
        thresholds=tuple(thresholds),
        levels=len(thresholds),
        samples_per_level=total,
        final_level=bool(extra),
        evaluations=model.evaluations - before,
        converged=converged,
        sampler=sampler,
        log_constant=log_c,
        max_log_scaled_likelihood=worst[0],
        constant_violations=violations[0],
        constant_violated=constant_violated,
    )
