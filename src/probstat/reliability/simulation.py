"""Sampling-based reliability on a :class:`ModelLimitState`.

Crude Monte Carlo, importance sampling around the FORM design point, and
subset simulation all sample the shared joint input law of the model in
independent standard-normal space ``u`` and report their model-evaluation
cost. :func:`compare_reliability_methods` runs these estimators together with
FORM and SORM on one limit state and returns a table-like comparison;
cross-entropy importance sampling, line sampling from
:mod:`probstat.reliability.adaptive`, and subset simulation with the adaptive
conditional-sampling and Hamiltonian kernels are available as opt-in methods.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector, dot
from probstat.monte_carlo import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    estimate_mean,
    estimate_probability,
)

from .adaptive import cross_entropy_importance_sampling, line_sampling
from .model_based import (
    FORMResult,
    ModelLimitState,
    form_hlrf,
    generalized_reliability_index,
    sorm,
)

METHODS = ("form", "sorm", "is", "subset", "mc")
OPTIONAL_METHODS = ("ce", "line", "acs", "hmc-ss")
AVAILABLE_METHODS = METHODS + OPTIONAL_METHODS


def _require_limit_state(limit_state: object) -> ModelLimitState:
    """Return the argument when it is a :class:`ModelLimitState`.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    return limit_state


def _require_samples(value: object, name: str, minimum: int = 2) -> int:
    """Validate an integer sample count with a lower limit.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _cov(probability: float, standard_error: float) -> float:
    """Return ``standard_error / probability`` or ``inf`` for zero probability.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6]. No named equation is needed.

    """
    return standard_error / probability if probability > 0.0 else math.inf


@dataclass(frozen=True, slots=True)
class ModelMonteCarloResult:
    """Crude Monte Carlo failure probability of a model limit state.

    Args:
        limit_state: Label of the analyzed limit state.
        failure: Bernoulli estimate with Wilson interval.
        coefficient_of_variation: ``standard_error / probability``;
            ``inf`` when no sample failed.
        evaluations: Model evaluations.

    With no failures (or no survivors) in the sample the plug-in standard
    error ``sqrt(p (1 - p) / N)`` is zero, which does not mean the estimate
    is exact: report the Wilson interval ``failure.confidence_interval``
    instead, whose upper end stays positive (about ``3.8 / N`` at 95 %).

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5], [REF-WILSON-1927].

    """

    limit_state: str
    failure: ProbabilityEstimate
    coefficient_of_variation: float
    evaluations: int

    @property
    def probability(self) -> float:
        """Return the empirical failure probability.

        Equation:
            [EQ-REL-PF-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        return self.failure.probability

    @property
    def standard_error(self) -> float:
        """Return the Bernoulli plug-in standard error.

        It is zero when no sample (or every sample) failed; the Wilson
        interval of ``failure`` is the honest error statement then.

        Equation:
            [EQ-MC-PROB-001].

        References:
            [REF-WILSON-1927].

        """
        return self.failure.standard_error


def model_monte_carlo(
    limit_state: ModelLimitState,
    samples: int,
    *,
    seed: int | RandomStream | None = None,
    confidence: float = 0.95,
) -> ModelMonteCarloResult:
    """Estimate ``P(g(X) <= 0)`` by crude Monte Carlo on the model's joint law.

    Args:
        limit_state: Model limit state.
        samples: Number of independent joint input draws, at least two.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.
        confidence: Wilson-interval confidence.

    Returns:
        Probability estimate, coefficient of variation, and evaluation count.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If ``samples < 2``.

    Equation:
        [EQ-REL-PF-001], [EQ-MC-PROB-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5], [REF-WILSON-1927].

    """
    state = _require_limit_state(limit_state)
    count = _require_samples(samples, "samples")
    stream = resolve_stream(seed)
    before = state.model.evaluations
    draws = state.model.inputs.sample(count, seed=stream.child("crude-mc"))
    events = [state.physical(x) <= 0.0 for x in draws]
    failure = estimate_probability(events, confidence=confidence)
    return ModelMonteCarloResult(
        limit_state=state.label,
        failure=failure,
        coefficient_of_variation=_cov(failure.probability, failure.standard_error),
        evaluations=state.model.evaluations - before,
    )


@dataclass(frozen=True, slots=True)
class DesignPointImportanceResult:
    """Importance-sampling estimate with a proposal shifted to the design point.

    Args:
        limit_state: Label of the analyzed limit state.
        estimate: Mean of the weighted indicator contributions.
        coefficient_of_variation: ``standard_error / probability``.
        failures: Proposal samples in the failure domain.
        effective_sample_size: Kish effective sample size of all weights.
        center: Proposal mean ``u*`` in standard-normal space.
        evaluations: Model evaluations for sampling.
        form_evaluations: Model evaluations spent on FORM (zero when a FORM
            result or center was supplied).

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6], [REF-OWEN-2013, Chapter 9].

    """

    limit_state: str
    estimate: MonteCarloEstimate
    coefficient_of_variation: float
    failures: int
    effective_sample_size: float
    center: Vector
    evaluations: int
    form_evaluations: int

    @property
    def probability(self) -> float:
        """Return the importance-sampling failure probability estimate.

        Equation:
            [EQ-REL-ISDP-001].

        References:
            [REF-OWEN-2013, Chapter 9].

        """
        return self.estimate.estimate

    @property
    def standard_error(self) -> float:
        """Return the standard error of the weighted mean.

        Equation:
            [EQ-MC-SE-001].

        References:
            [REF-OWEN-2013, Chapter 9].

        """
        return self.estimate.standard_error


def design_point_importance_sampling(
    limit_state: ModelLimitState,
    samples: int,
    *,
    form_result: FORMResult | None = None,
    center: Sequence[float] | None = None,
    seed: int | RandomStream | None = None,
) -> DesignPointImportanceResult:
    """Estimate ``P_f`` with a standard normal proposal shifted to ``u*``.

    The proposal is ``h(u) = phi_n(u - u*)`` and the likelihood ratio is
    ``w(u) = phi_n(u) / h(u) = exp(-u . u* + ||u*||^2 / 2)``. The estimator is
    ``mean(I(G(u_i) <= 0) w(u_i))`` over proposal draws.

    Args:
        limit_state: Model limit state.
        samples: Number of proposal draws, at least two.
        form_result: Optional FORM result supplying ``u*``.
        center: Optional explicit proposal center (overrides ``form_result``).
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Weighted estimate, coefficient of variation, weight diagnostics, and
        evaluation counts.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If ``samples < 2`` or the center has the wrong length.

    Equation:
        [EQ-REL-ISDP-001], [EQ-REL-IS-001], [EQ-MC-ESS-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6], [REF-OWEN-2013, Chapter 9].

    """
    state = _require_limit_state(limit_state)
    count = _require_samples(samples, "samples")
    stream = resolve_stream(seed)
    form_evaluations = 0
    if center is not None:
        shift = as_vector(center, name="center")
    else:
        if form_result is None:
            form_result = form_hlrf(state)
            form_evaluations = form_result.evaluations
        elif not isinstance(form_result, FORMResult):
            raise TypeError("form_result must be a FORMResult")
        shift = form_result.design_point_u
    if len(shift) != state.dimension:
        raise ValueError("proposal center length must equal the input dimension")
    generator = stream.child("design-point-is").generator()
    half_norm = 0.5 * dot(shift, shift)
    before = state.model.evaluations
    contributions: list[float] = []
    weights: list[float] = []
    failures = 0
    for _ in range(count):
        u = tuple(generator.gauss(0.0, 1.0) + c for c in shift)
        weight = math.exp(half_norm - dot(u, shift))
        weights.append(weight)
        if state.standard(u) <= 0.0:
            failures += 1
            contributions.append(weight)
        else:
            contributions.append(0.0)
    estimate = estimate_mean(contributions)
    squared = math.fsum(w * w for w in weights)
    return DesignPointImportanceResult(
        limit_state=state.label,
        estimate=estimate,
        coefficient_of_variation=_cov(estimate.estimate, estimate.standard_error),
        failures=failures,
        effective_sample_size=math.fsum(weights) ** 2 / squared,
        center=shift,
        evaluations=state.model.evaluations - before,
        form_evaluations=form_evaluations,
    )


@dataclass(frozen=True, slots=True)
class SubsetSimulationResult:
    """Subset-simulation estimate of a small failure probability.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Product of the conditional level probabilities.
        thresholds: Intermediate thresholds ``b_1 > b_2 > ... > b_m = 0``.
        conditional_probabilities: Estimated ``P(F_j | F_{j-1})`` per level.
        level_coefficients_of_variation: Per-level c.o.v. ``delta_j``
            including the Markov-chain correlation factor.
        coefficient_of_variation: ``sqrt(sum_j delta_j^2)``, which assumes the
            level estimators are uncorrelated; it is an approximation that
            typically understates the true c.o.v.
        coefficient_of_variation_upper: ``sum_j delta_j``, the value for fully
            correlated level estimators.
        samples_per_level: Samples per level ``N``.
        levels: Number of simulation levels (including level 0).
        acceptance_rates: Fraction of accepted candidate states per
            conditional level.
        evaluations: Model evaluations.
        converged: Whether the failure threshold ``0`` was reached within
            ``max_levels``.
        sampler: Markov-chain kernel that populated the conditional levels:
            ``"mma"``, ``"cs"``, ``"acs"``, or ``"hmc"``.
        level_scales: One kernel scale per conditional level: the proposal
            spread ``s`` for ``"mma"``, the correlation ``rho`` for ``"cs"``,
            the scaling factor ``lambda`` after the level's last adaptation
            for ``"acs"``, and the trajectory time ``t_f`` after the level's
            last adaptation for ``"hmc"``.

    References:
        [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-WANG-BROCCARDO-SONG-2019].

    """

    limit_state: str
    probability: float
    thresholds: Vector
    conditional_probabilities: Vector
    level_coefficients_of_variation: Vector
    coefficient_of_variation: float
    coefficient_of_variation_upper: float
    samples_per_level: int
    levels: int
    acceptance_rates: Vector
    evaluations: int
    converged: bool
    sampler: str = "mma"
    level_scales: Vector = ()

    @property
    def beta(self) -> float:
        """Return the generalized reliability index of the estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.probability)


SUBSET_SAMPLERS = ("mma", "cs", "acs", "hmc")
_SCALE_REFERENCES = ("unit", "seeds")


def _level_cov(
    indicators: Sequence[bool],
    chains: int,
    probability: float,
    correlated: bool,
    lengths: Sequence[int] | None = None,
) -> float:
    """Return Au and Beck's level c.o.v. with the chain correlation factor.

    ``delta^2 = (1 - P)/(N P) (1 + gamma)`` with
    ``gamma = 2 sum_{k=1}^{Ns-1} (1 - k/Ns) rho(k)`` estimated from the
    indicator chains stored contiguously, ``Ns`` states per chain. When
    ``lengths`` gives chains of unequal length ``L_c``, the weight
    ``1 - k/Ns`` becomes ``sum_c max(0, L_c - k) / N``, the share of lag-``k``
    pairs, which reduces to ``1 - k/Ns`` for equal lengths.

    References:
        [EQ-REL-SUBSET-003], [EQ-REL-SUBSET-006], [REF-AU-BECK-2001].

    """
    total = len(indicators)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    base = (1.0 - probability) / (total * probability)
    if not correlated:
        return math.sqrt(base)
    variance = probability * (1.0 - probability)
    gamma = 0.0
    if lengths is None or len(set(lengths)) == 1:
        length = total // chains
        for lag in range(1, length):
            products = 0
            for chain in range(chains):
                offset = chain * length
                for position in range(length - lag):
                    if (
                        indicators[offset + position]
                        and indicators[offset + position + lag]
                    ):
                        products += 1
            covariance = products / (total - lag * chains) - probability * probability
            gamma += 2.0 * (1.0 - lag / length) * covariance / variance
        return math.sqrt(base * max(0.0, 1.0 + gamma))
    offsets = [0]
    for size in lengths[:-1]:
        offsets.append(offsets[-1] + size)
    for lag in range(1, max(lengths)):
        products = 0
        pairs = 0
        for offset, size in zip(offsets, lengths, strict=True):
            if size <= lag:
                continue
            pairs += size - lag
            for position in range(size - lag):
                if (
                    indicators[offset + position]
                    and indicators[offset + position + lag]
                ):
                    products += 1
        covariance = products / pairs - probability * probability
        gamma += 2.0 * (pairs / total) * covariance / variance
    return math.sqrt(base * max(0.0, 1.0 + gamma))


class _LevelSample:
    """States, chain lengths, and acceptance counts of one conditional level.

    Args:
        points: Chain states, chains stored contiguously, seeds first.
        values: Limit-state values of ``points``.
        lengths: Number of states of each chain in storage order.
        accepted: Accepted candidate states.
        proposed: Proposed candidate states.
        scale: Kernel scale reported for the level (see
            :class:`SubsetSimulationResult`).
        carry: Adapted scale handed to the next level.

    References:
        [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].
        No equation is evaluated.

    """

    __slots__ = (
        "accepted",
        "carry",
        "lengths",
        "points",
        "proposed",
        "scale",
        "values",
    )

    def __init__(
        self,
        points: list[Vector],
        values: list[float],
        lengths: list[int],
        accepted: int,
        proposed: int,
        scale: float,
        carry: float,
    ) -> None:
        """Store the level sample.

        References:
            [REF-AU-BECK-2001]. No equation is evaluated.

        """
        self.points = points
        self.values = values
        self.lengths = lengths
        self.accepted = accepted
        self.proposed = proposed
        self.scale = scale
        self.carry = carry


def _modified_metropolis_chain(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    start: Vector,
    start_value: float,
    threshold: float,
    length: int,
    spread: float,
    points: list[Vector],
    values: list[float],
) -> int:
    """Append one component-wise modified Metropolis chain and count acceptances.

    The seed is stored as the first state. Each of the ``length - 1`` steps
    proposes ``xi_k = u_k + s z_k`` per coordinate, keeps it with probability
    ``min(1, phi(xi_k)/phi(u_k))``, and evaluates the limit state only when at
    least one coordinate moved; the candidate replaces the state when
    ``G(xi) <= b``. The random draws are consumed in the same order as the
    original inline kernel, so seeded results are unchanged.

    References:
        [EQ-REL-SUBSET-002], [REF-AU-BECK-2001].

    """
    gauss = generator.gauss
    uniform = generator.random
    exp = math.exp
    add_point = points.append
    add_value = values.append
    current, current_value = start, start_value
    add_point(current)
    add_value(current_value)
    accepted = 0
    for _ in range(length - 1):
        candidate = []
        for coordinate in current:
            trial = coordinate + spread * gauss(0.0, 1.0)
            ratio = exp(0.5 * (coordinate * coordinate - trial * trial))
            candidate.append(trial if uniform() < ratio else coordinate)
        proposal = tuple(candidate)
        if proposal != current:
            proposal_value = evaluate(proposal)
            if proposal_value <= threshold:
                current, current_value = proposal, proposal_value
                accepted += 1
        add_point(current)
        add_value(current_value)
    return accepted


def _conditional_chain(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    start: Vector,
    start_value: float,
    threshold: float,
    length: int,
    correlations: Sequence[float],
    spreads: Sequence[float],
    points: list[Vector],
    values: list[float],
) -> int:
    """Append one conditional-sampling chain and count acceptances.

    Each step proposes ``v_i = rho_i u_i + sigma_i z_i`` with
    ``rho_i^2 + sigma_i^2 = 1``, which leaves the standard normal law
    invariant, and accepts ``v`` exactly when ``G(v) <= b``. With
    ``rho_i = cos t`` and ``sigma_i = sin t`` this is the exact Hamiltonian
    flow ``u(t) = p sin t + u_0 cos t`` with rejection outside ``F_j``.

    References:
        [EQ-REL-SUBSET-004], [EQ-REL-HMCSS-001],
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-WANG-BROCCARDO-SONG-2019].

    """
    gauss = generator.gauss
    add_point = points.append
    add_value = values.append
    current, current_value = start, start_value
    add_point(current)
    add_value(current_value)
    accepted = 0
    for _ in range(length - 1):
        proposal = tuple(
            rho * coordinate + sigma * gauss(0.0, 1.0)
            for coordinate, rho, sigma in zip(
                current, correlations, spreads, strict=True
            )
        )
        proposal_value = evaluate(proposal)
        if proposal_value <= threshold:
            current, current_value = proposal, proposal_value
            accepted += 1
        add_point(current)
        add_value(current_value)
    return accepted


def _chain_lengths(total: int, chains: int) -> list[int]:
    """Return ``floor(N/N_s)`` states per chain with the remainder on the first chains.

    References:
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015]. No equation is evaluated.

    """
    base, remainder = divmod(total, chains)
    return [base + 1 if chain < remainder else base for chain in range(chains)]


def _seed_spreads(seeds: Sequence[tuple[Vector, float]], dimension: int) -> list[float]:
    """Return the per-coordinate sample standard deviation of the seeds.

    Coordinates with fewer than two seeds or zero spread fall back to ``1``.

    References:
        [EQ-REL-SUBSET-005], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

    """
    count = len(seeds)
    if count < 2:
        return [1.0] * dimension
    spreads: list[float] = []
    for index in range(dimension):
        column = [point[index] for point, _ in seeds]
        mean = math.fsum(column) / count
        variance = math.fsum((item - mean) ** 2 for item in column) / (count - 1)
        spreads.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return spreads


def _mma_level(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    seeds: Sequence[tuple[Vector, float]],
    threshold: float,
    chain_length: int,
    spread: float,
) -> _LevelSample:
    """Populate a level with component-wise modified Metropolis chains.

    Chains start from the seeds in ascending order of ``G``.

    References:
        [EQ-REL-SUBSET-002], [REF-AU-BECK-2001].

    """
    points: list[Vector] = []
    values: list[float] = []
    accepted = 0
    for start, start_value in seeds:
        accepted += _modified_metropolis_chain(
            evaluate,
            generator,
            start,
            start_value,
            threshold,
            chain_length,
            spread,
            points,
            values,
        )
    proposed = len(seeds) * (chain_length - 1)
    lengths = [chain_length] * len(seeds)
    return _LevelSample(points, values, lengths, accepted, proposed, spread, spread)


def _cs_level(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    seeds: Sequence[tuple[Vector, float]],
    threshold: float,
    lengths: list[int],
    correlation: float,
) -> _LevelSample:
    """Populate a level with fixed-correlation conditional-sampling chains.

    References:
        [EQ-REL-SUBSET-004], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

    """
    dimension = len(seeds[0][0])
    correlations = [correlation] * dimension
    spreads = [math.sqrt(1.0 - correlation * correlation)] * dimension
    points: list[Vector] = []
    values: list[float] = []
    accepted = 0
    for (start, start_value), length in zip(seeds, lengths, strict=True):
        accepted += _conditional_chain(
            evaluate,
            generator,
            start,
            start_value,
            threshold,
            length,
            correlations,
            spreads,
            points,
            values,
        )
    proposed = sum(lengths) - len(lengths)
    return _LevelSample(
        points, values, list(lengths), accepted, proposed, correlation, correlation
    )


def _acs_level(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    seeds: list[tuple[Vector, float]],
    threshold: float,
    lengths: list[int],
    scale: float,
    reference: str,
    adaptation_chains: int,
    target: float,
) -> _LevelSample:
    """Populate a level with adaptive conditional-sampling chains.

    The seeds are randomly permuted. With ``sigma_0`` equal to one or to the
    seed standard deviations, each chain uses ``sigma_i = min(lambda
    sigma_0i, 1)`` and ``rho_i = sqrt(1 - sigma_i^2)``. After every
    ``N_a`` chains the mean acceptance ``a_hat`` (each chain's rate capped at
    one) updates ``ln lambda <- ln lambda + (a_hat - a*)/sqrt(i + 1)``,
    ``i = 0, 1, ...`` counting the updates of the level.

    References:
        [EQ-REL-SUBSET-004], [EQ-REL-SUBSET-005],
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015].

    """
    generator.shuffle(seeds)
    dimension = len(seeds[0][0])
    base = (
        _seed_spreads(seeds, dimension) if reference == "seeds" else [1.0] * dimension
    )

    def kernel(factor: float) -> tuple[list[float], list[float]]:
        """Return the correlations and spreads for one scaling factor.

        References:
            [EQ-REL-SUBSET-005]. No further equation is evaluated.

        """
        spreads = [min(factor * item, 1.0) for item in base]
        return [math.sqrt(1.0 - item * item) for item in spreads], spreads

    correlations, spreads = kernel(scale)
    points: list[Vector] = []
    values: list[float] = []
    accepted = 0
    window: list[float] = []
    updates = 0
    for chain, ((start, start_value), length) in enumerate(
        zip(seeds, lengths, strict=True)
    ):
        gained = _conditional_chain(
            evaluate,
            generator,
            start,
            start_value,
            threshold,
            length,
            correlations,
            spreads,
            points,
            values,
        )
        accepted += gained
        window.append(min(1.0, gained / (length - 1)))
        if (chain + 1) % adaptation_chains == 0:
            rate = math.fsum(window) / len(window)
            window.clear()
            scale = math.exp(math.log(scale) + (rate - target) / math.sqrt(updates + 1))
            updates += 1
            correlations, spreads = kernel(scale)
    proposed = sum(lengths) - len(lengths)
    return _LevelSample(points, values, list(lengths), accepted, proposed, scale, scale)


def _hmc_level(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    seeds: list[tuple[Vector, float]],
    threshold: float,
    lengths: list[int],
    trajectory_time: float,
    adaptation_chains: int,
) -> _LevelSample:
    """Populate a level with rejection-sampling Hamiltonian Monte Carlo chains.

    The seeds are randomly permuted. Each step draws a fresh momentum
    ``p ~ N(0, I)``, follows the exact flow ``u(t_f) = p sin t_f +
    u_0 cos t_f`` of the Gaussian Hamiltonian, and accepts the end point
    exactly when ``G <= b``. After every ``N_a`` chains, a mean acceptance
    ``a < 0.3`` sets ``t_f <- arcsin(sin t_f exp((a - 0.3)/2))`` and
    ``a > 0.5`` sets ``t_f <- arcsin(min(1, sin t_f exp((a - 0.5)/2)))``.

    References:
        [EQ-REL-HMCSS-001], [EQ-REL-HMCSS-002], [REF-WANG-BROCCARDO-SONG-2019].

    """
    generator.shuffle(seeds)
    dimension = len(seeds[0][0])
    time = trajectory_time
    correlations = [math.cos(time)] * dimension
    spreads = [math.sin(time)] * dimension
    points: list[Vector] = []
    values: list[float] = []
    accepted = 0
    window: list[float] = []
    for chain, ((start, start_value), length) in enumerate(
        zip(seeds, lengths, strict=True)
    ):
        gained = _conditional_chain(
            evaluate,
            generator,
            start,
            start_value,
            threshold,
            length,
            correlations,
            spreads,
            points,
            values,
        )
        accepted += gained
        window.append(gained / (length - 1))
        if (chain + 1) % adaptation_chains == 0:
            rate = math.fsum(window) / len(window)
            window.clear()
            if rate < 0.3 or rate > 0.5:
                anchor = 0.3 if rate < 0.3 else 0.5
                time = math.asin(
                    min(1.0, math.sin(time) * math.exp((rate - anchor) / 2.0))
                )
                correlations = [math.cos(time)] * dimension
                spreads = [math.sin(time)] * dimension
    proposed = sum(lengths) - len(lengths)
    return _LevelSample(points, values, list(lengths), accepted, proposed, time, time)


def _require_unit_interval(value: object, name: str, *, closed_low: bool) -> float:
    """Validate a real option in ``[0, 1)`` or ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    low_ok = number >= 0.0 if closed_low else number > 0.0
    if not (low_ok and number < 1.0):
        interval = "[0, 1)" if closed_low else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return number


def subset_simulation(
    limit_state: ModelLimitState,
    *,
    samples_per_level: int = 1000,
    level_probability: float = 0.1,
    proposal_spread: float = 1.0,
    max_levels: int = 20,
    seed: int | RandomStream | None = None,
    sampler: str = "mma",
    correlation: float = 0.8,
    initial_scale: float = 0.6,
    target_acceptance: float = 0.44,
    adaptation_chains: int | None = None,
    scale_reference: str = "unit",
    trajectory_time: float = math.pi / 4.0,
) -> SubsetSimulationResult:
    """Estimate a small failure probability by subset simulation.

    Level 0 draws ``N`` independent standard-normal vectors. At each level the
    intermediate threshold ``b_j`` is the midpoint between the
    ``N p0``-th and ``(N p0 + 1)``-th smallest limit-state values, so exactly
    ``N p0`` seeds satisfy ``G <= b_j``. Each seed is the first state of a
    Markov chain whose stationary law is the standard normal law restricted
    to ``F_j = {G <= b_j}``. The run stops when at least ``N p0`` samples fail
    (``G <= 0``), and ``P_f = p0^(m-1) N_f / N``.

    ``sampler`` selects the chain kernel:

    - ``"mma"`` (default): component-wise modified Metropolis. Every
      coordinate proposes ``xi_k = u_k + s z`` and keeps it with probability
      ``min(1, phi(xi_k)/phi(u_k))``; the vector is accepted only if
      ``G(xi) <= b_j``. Chains have length ``1/p0`` and start from the seeds in
      ascending order of ``G``. Seeded results are identical to earlier
      releases.
    - ``"cs"``: conditional sampling ``v = rho u + sqrt(1 - rho^2) z`` with a
      fixed ``rho`` (``correlation``), accepted iff ``G(v) <= b_j``.
    - ``"acs"``: adaptive conditional sampling. Seeds are randomly permuted,
      chains have ``floor(N/N_s)`` states (the remainder on the first
      chains), ``sigma_i = min(lambda sigma_0i, 1)``,
      ``rho_i = sqrt(1 - sigma_i^2)``, and ``lambda`` is adapted every
      ``N_a`` chains toward the acceptance rate ``a*``, starting from
      ``initial_scale`` and carried from level to level.
    - ``"hmc"``: rejection-sampling Hamiltonian Monte Carlo with the exact
      Gaussian flow ``u(t_f) = p sin t_f + u_0 cos t_f`` and a fresh momentum
      per step; ``t_f`` starts at ``trajectory_time``, is adapted every
      ``N_a`` chains to keep the acceptance rate in ``[0.3, 0.5]``, and is
      carried from level to level.

    Args:
        limit_state: Model limit state.
        samples_per_level: Samples ``N`` per level; ``N p0`` must be a
            positive integer, and for ``sampler="mma"`` it must divide ``N``.
        level_probability: Conditional level probability ``p0`` in
            ``(0, 0.5]``.
        proposal_spread: Standard deviation ``s`` of the per-coordinate
            Gaussian random-walk proposal (``"mma"``).
        max_levels: Maximum number of levels before stopping unconverged.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.
        sampler: Chain kernel, one of ``SUBSET_SAMPLERS`` = ``("mma", "cs",
            "acs", "hmc")``.
        correlation: Fixed correlation ``rho`` in ``[0, 1)`` (``"cs"``).
        initial_scale: Starting scaling factor ``lambda > 0`` (``"acs"``).
        target_acceptance: Target acceptance ``a*`` in ``(0, 1)``
            (``"acs"``).
        adaptation_chains: Chains ``N_a`` between adaptations (``"acs"`` and
            ``"hmc"``); ``None`` uses ``ceil(100 N_s / N) = ceil(100 p0)``.
        scale_reference: ``"unit"`` for ``sigma_0i = 1`` or ``"seeds"`` for the
            per-coordinate sample standard deviation of the seeds
            (``"acs"``).
        trajectory_time: Starting trajectory time ``t_f`` in ``(0, pi/2]``
            (``"hmc"``).

    Returns:
        Probability, thresholds, conditional probabilities, c.o.v. estimates,
        acceptance rates, evaluation count, convergence flag, sampler name,
        and the kernel scale of each conditional level.

    Raises:
        TypeError: If ``limit_state`` has the wrong type or an option has the
            wrong type.
        ValueError: If options are inconsistent, or limit-state values tie at
            an intermediate threshold (a continuous response is required).

    Equation:
        [EQ-REL-SUBSET-001], [EQ-REL-SUBSET-002], [EQ-REL-SUBSET-003],
        [EQ-REL-SUBSET-004], [EQ-REL-SUBSET-005], [EQ-REL-SUBSET-006],
        [EQ-REL-HMCSS-001], [EQ-REL-HMCSS-002].

    References:
        [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-WANG-BROCCARDO-SONG-2019], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    state = _require_limit_state(limit_state)
    if not isinstance(sampler, str) or sampler not in SUBSET_SAMPLERS:
        raise ValueError(f"sampler must be one of {SUBSET_SAMPLERS}")
    total = _require_samples(samples_per_level, "samples_per_level", minimum=10)
    p0 = require_finite_real(level_probability, name="level_probability")
    if not 0.0 < p0 <= 0.5:
        raise ValueError("level_probability must lie in (0, 0.5]")
    seeds_count = round(total * p0)
    if seeds_count < 1 or abs(seeds_count - total * p0) > 1e-9:
        raise ValueError("samples_per_level * level_probability must be an integer")
    if sampler == "mma" and total % seeds_count != 0:
        raise ValueError("samples_per_level * level_probability must divide N")
    chain_length = total // seeds_count
    spread = require_finite_real(proposal_spread, name="proposal_spread")
    if spread <= 0.0:
        raise ValueError("proposal_spread must be positive")
    levels_limit = _require_samples(max_levels, "max_levels", minimum=1)
    rho = _require_unit_interval(correlation, "correlation", closed_low=True)
    scale = require_finite_real(initial_scale, name="initial_scale")
    if scale <= 0.0:
        raise ValueError("initial_scale must be positive")
    target = _require_unit_interval(
        target_acceptance, "target_acceptance", closed_low=False
    )
    if adaptation_chains is None:
        window = max(1, math.ceil(100.0 * seeds_count / total - 1e-9))
    else:
        window = _require_samples(adaptation_chains, "adaptation_chains", minimum=1)
    if scale_reference not in _SCALE_REFERENCES:
        raise ValueError(f"scale_reference must be one of {_SCALE_REFERENCES}")
    time = require_finite_real(trajectory_time, name="trajectory_time")
    if not 0.0 < time <= 0.5 * math.pi:
        raise ValueError("trajectory_time must lie in (0, pi/2]")
    lengths = (
        [chain_length] * seeds_count
        if sampler == "mma"
        else _chain_lengths(total, seeds_count)
    )
    generator = resolve_stream(seed).child("subset-simulation").generator()
    dimension = state.dimension
    before = state.model.evaluations
    points: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(total)
    ]
    values = [state.standard(u) for u in points]
    thresholds: list[float] = []
    conditionals: list[float] = []
    level_covs: list[float] = []
    acceptance: list[float] = []
    scales: list[float] = []
    carry = {"mma": spread, "cs": rho, "acs": scale, "hmc": time}[sampler]
    product = 1.0
    converged = False
    probability = 0.0
    level = 0
    while True:
        correlated = level > 0
        failures = sum(1 for value in values if value <= 0.0)
        if failures >= seeds_count or level + 1 >= levels_limit:
            fraction = failures / total
            thresholds.append(0.0)
            conditionals.append(fraction)
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
                "limit-state values tie at an intermediate threshold; subset "
                "simulation requires a continuous response distribution"
            )
        # Repeated chain states make ties possible at conditional levels; the
        # threshold is then the tied value and the seeds are the first N p0
        # samples in sorted order, as in Au and Beck's construction.
        threshold = 0.5 * (last_inside + first_outside)
        thresholds.append(threshold)
        conditionals.append(p0)
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
        seeds = [(points[index], values[index]) for index in order[:seeds_count]]
        if sampler == "mma":
            sample = _mma_level(
                state.standard, generator, seeds, threshold, chain_length, spread
            )
        elif sampler == "cs":
            sample = _cs_level(
                state.standard, generator, seeds, threshold, lengths, rho
            )
        elif sampler == "acs":
            sample = _acs_level(
                state.standard,
                generator,
                seeds,
                threshold,
                lengths,
                carry,
                scale_reference,
                window,
                target,
            )
        else:
            sample = _hmc_level(
                state.standard, generator, seeds, threshold, lengths, carry, window
            )
        acceptance.append(sample.accepted / sample.proposed if sample.proposed else 0.0)
        scales.append(sample.scale)
        carry = sample.carry
        points, values, lengths = sample.points, sample.values, sample.lengths
        level += 1
    squared = math.fsum(delta * delta for delta in level_covs)
    return SubsetSimulationResult(
        limit_state=state.label,
        probability=probability,
        thresholds=tuple(thresholds),
        conditional_probabilities=tuple(conditionals),
        level_coefficients_of_variation=tuple(level_covs),
        coefficient_of_variation=math.sqrt(squared),
        coefficient_of_variation_upper=math.fsum(level_covs),
        samples_per_level=total,
        levels=len(conditionals),
        acceptance_rates=tuple(acceptance),
        evaluations=state.model.evaluations - before,
        converged=converged,
        sampler=sampler,
        level_scales=tuple(scales),
    )


@dataclass(frozen=True, slots=True)
class MethodComparisonRow:
    """One estimator's result in a reliability-method comparison.

    Args:
        method: Method label.
        probability: Failure probability estimate.
        beta: Equivalent index ``-Phi^-1(probability)``.
        evaluations: Cumulative model evaluations needed for this number
            (for SORM and IS this includes the FORM search they reuse).
        coefficient_of_variation: Sampling c.o.v., ``None`` for FORM/SORM.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5 and 6].

    """

    method: str
    probability: float
    beta: float
    evaluations: int
    coefficient_of_variation: float | None


@dataclass(frozen=True, slots=True)
class MethodComparison:
    """Table-like comparison of reliability methods on one limit state.

    Args:
        limit_state: Label of the analyzed limit state.
        rows: One row per method in evaluation order.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5 and 6].

    """

    limit_state: str
    rows: tuple[MethodComparisonRow, ...]

    def row(self, method: str) -> MethodComparisonRow:
        """Return the row of one method.

        Args:
            method: Method label such as ``"FORM"`` or ``"Subset"``.

        Returns:
            Matching row.

        Raises:
            KeyError: If the method is not in the comparison.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 6]. No equation is evaluated.

        """
        for item in self.rows:
            if item.method == method:
                return item
        raise KeyError(f"method {method!r} is not in the comparison")

    def as_dicts(self) -> tuple[dict[str, object], ...]:
        """Return the rows as plain dictionaries for tabulation or export.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 6]. No equation is evaluated.

        """
        return tuple(
            {
                "method": item.method,
                "probability": item.probability,
                "beta": item.beta,
                "evaluations": item.evaluations,
                "coefficient_of_variation": item.coefficient_of_variation,
            }
            for item in self.rows
        )


def compare_reliability_methods(
    limit_state: ModelLimitState,
    *,
    methods: Sequence[str] = METHODS,
    monte_carlo_samples: int = 10_000,
    importance_samples: int = 2_000,
    subset_samples: int = 1_000,
    cross_entropy_samples: int = 1_000,
    line_samples: int = 100,
    seed: int | RandomStream | None = None,
) -> MethodComparison:
    """Run FORM, SORM, IS, subset simulation, and crude MC on one limit state.

    FORM is run once and reused by SORM, importance sampling, and line
    sampling; their rows report cumulative cost. SORM contributes two rows,
    ``"SORM-Breitung"`` and ``"SORM-HR"``. Each sampling method uses an
    independent child stream. Cross-entropy importance sampling (``"ce"``,
    row ``"CE-IS"``), line sampling (``"line"``, row ``"Line"``), and subset
    simulation with adaptive conditional sampling (``"acs"``, row
    ``"Subset-aCS"``) or rejection-sampling Hamiltonian Monte Carlo
    (``"hmc-ss"``, row ``"Subset-HMC"``) are opt-in: they are in
    ``AVAILABLE_METHODS`` but not in the default ``METHODS``, so default
    comparisons are unchanged.

    Args:
        limit_state: Model limit state.
        methods: Subset of ``AVAILABLE_METHODS`` = ``("form", "sorm", "is",
            "subset", "mc", "ce", "line", "acs", "hmc-ss")``; default
            ``METHODS`` (the first five). SORM rows are omitted when the
            FORM index is not positive, because the curvature corrections are
            undefined there.
        monte_carlo_samples: Crude Monte Carlo sample size.
        importance_samples: Design-point importance-sampling sample size.
        subset_samples: Subset-simulation samples per level (also used by
            ``"acs"`` and ``"hmc-ss"``).
        cross_entropy_samples: Cross-entropy samples per level (also the final
            importance-sampling size).
        line_samples: Number of lines for line sampling.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Comparison with method, probability, equivalent beta, evaluations, and
        c.o.v. per row.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If a method name is unknown or a method's own validation
            fails (for example a non-converged FORM for SORM).

    Equation:
        [EQ-REL-BETA-001], [EQ-REL-FORM-002], [EQ-REL-SORM-002],
        [EQ-REL-SORM-003], [EQ-REL-ISDP-001], [EQ-REL-SUBSET-001],
        [EQ-REL-PF-001], [EQ-REL-CE-002], [EQ-REL-LINE-001],
        [EQ-REL-SUBSET-005], [EQ-REL-HMCSS-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5 and 6], [REF-AU-BECK-2001],
        [REF-RUBINSTEIN-KROESE-2004],
        [REF-KOUTSOURELAKIS-PRADLWARTER-SCHUELLER-2004],
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-WANG-BROCCARDO-SONG-2019].

    """
    state = _require_limit_state(limit_state)
    requested = tuple(methods)
    unknown = sorted(set(requested) - set(AVAILABLE_METHODS))
    if unknown:
        raise ValueError(f"unknown reliability methods: {unknown}")
    stream = resolve_stream(seed)
    rows: list[MethodComparisonRow] = []
    form_result: FORMResult | None = None
    if {"form", "sorm", "is", "line"} & set(requested):
        form_result = form_hlrf(state)
    if "form" in requested and form_result is not None:
        rows.append(
            MethodComparisonRow(
                "FORM",
                form_result.probability,
                form_result.beta,
                form_result.evaluations,
                None,
            )
        )
    if "sorm" in requested and form_result is not None and form_result.beta > 0.0:
        # SORM corrections are undefined for beta <= 0 (median input fails);
        # the comparison then omits the SORM rows instead of failing outright.
        second = sorm(state, form_result=form_result)
        for label, probability in (
            ("SORM-Breitung", second.breitung_probability),
            ("SORM-HR", second.hohenbichler_rackwitz_probability),
        ):
            rows.append(
                MethodComparisonRow(
                    label,
                    probability,
                    generalized_reliability_index(probability),
                    second.evaluations,
                    None,
                )
            )
    if "is" in requested and form_result is not None:
        weighted = design_point_importance_sampling(
            state,
            importance_samples,
            form_result=form_result,
            seed=stream.child("is"),
        )
        rows.append(
            MethodComparisonRow(
                "IS",
                weighted.probability,
                generalized_reliability_index(weighted.probability),
                weighted.evaluations + form_result.evaluations,
                weighted.coefficient_of_variation,
            )
        )
    if "subset" in requested:
        subset = subset_simulation(
            state, samples_per_level=subset_samples, seed=stream.child("subset")
        )
        rows.append(
            MethodComparisonRow(
                "Subset",
                subset.probability,
                subset.beta,
                subset.evaluations,
                subset.coefficient_of_variation,
            )
        )
    if "mc" in requested:
        crude = model_monte_carlo(state, monte_carlo_samples, seed=stream.child("mc"))
        rows.append(
            MethodComparisonRow(
                "MC",
                crude.probability,
                generalized_reliability_index(crude.probability),
                crude.evaluations,
                crude.coefficient_of_variation,
            )
        )
    if "ce" in requested:
        adaptive = cross_entropy_importance_sampling(
            state,
            samples_per_level=cross_entropy_samples,
            seed=stream.child("ce"),
        )
        rows.append(
            MethodComparisonRow(
                "CE-IS",
                adaptive.probability,
                generalized_reliability_index(adaptive.probability),
                adaptive.evaluations,
                adaptive.coefficient_of_variation,
            )
        )
    if "line" in requested and form_result is not None:
        lined = line_sampling(
            state,
            line_samples,
            form_result=form_result,
            seed=stream.child("line"),
        )
        rows.append(
            MethodComparisonRow(
                "Line",
                lined.probability,
                generalized_reliability_index(lined.probability),
                lined.evaluations + form_result.evaluations,
                lined.coefficient_of_variation,
            )
        )
    for name, sampler, label in (
        ("acs", "acs", "Subset-aCS"),
        ("hmc-ss", "hmc", "Subset-HMC"),
    ):
        if name in requested:
            chained = subset_simulation(
                state,
                samples_per_level=subset_samples,
                sampler=sampler,
                seed=stream.child(f"subset-{sampler}"),
            )
            rows.append(
                MethodComparisonRow(
                    label,
                    chained.probability,
                    chained.beta,
                    chained.evaluations,
                    chained.coefficient_of_variation,
                )
            )
    return MethodComparison(state.label, tuple(rows))
