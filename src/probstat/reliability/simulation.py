"""Sampling-based reliability on a :class:`ModelLimitState`.

Crude Monte Carlo, importance sampling around the FORM design point, and
subset simulation all sample the shared joint input law of the model in
independent standard-normal space ``u`` and report their model-evaluation
cost. :func:`compare_reliability_methods` runs these estimators together with
FORM and SORM on one limit state and returns a table-like comparison.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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

from .model_based import (
    FORMResult,
    ModelLimitState,
    form_hlrf,
    generalized_reliability_index,
    sorm,
)

METHODS = ("form", "sorm", "is", "subset", "mc")


def _require_limit_state(limit_state: object) -> ModelLimitState:
    """Return the argument when it is a :class:`ModelLimitState`.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    return limit_state


def _require_samples(value: object, name: str, minimum: int = 2) -> int:
    """Validate an integer sample count with a lower limit.

    References:
        Project validation policy. No equation is evaluated.

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
        coefficient_of_variation: ``standard_error / probability``.
        evaluations: Model evaluations.

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

    References:
        [REF-AU-BECK-2001].

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

    @property
    def beta(self) -> float:
        """Return the generalized reliability index of the estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.probability)


def _level_cov(
    indicators: Sequence[bool], chains: int, probability: float, correlated: bool
) -> float:
    """Return Au and Beck's level c.o.v. with the chain correlation factor.

    ``delta^2 = (1 - P)/(N P) (1 + gamma)`` with
    ``gamma = 2 sum_{k=1}^{Ns-1} (1 - k/Ns) rho(k)`` estimated from the
    indicator chains stored contiguously, ``Ns`` states per chain.

    References:
        [EQ-REL-SUBSET-003], [REF-AU-BECK-2001].

    """
    total = len(indicators)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    base = (1.0 - probability) / (total * probability)
    if not correlated:
        return math.sqrt(base)
    length = total // chains
    variance = probability * (1.0 - probability)
    gamma = 0.0
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


def subset_simulation(
    limit_state: ModelLimitState,
    *,
    samples_per_level: int = 1000,
    level_probability: float = 0.1,
    proposal_spread: float = 1.0,
    max_levels: int = 20,
    seed: int | RandomStream | None = None,
) -> SubsetSimulationResult:
    """Estimate a small failure probability by subset simulation.

    Level 0 draws ``N`` independent standard-normal vectors. At each level the
    intermediate threshold ``b_j`` is the midpoint between the
    ``N p0``-th and ``(N p0 + 1)``-th smallest limit-state values, so exactly
    ``N p0`` seeds satisfy ``G <= b_j``. Each seed starts a chain of length
    ``1/p0`` generated by the component-wise modified Metropolis algorithm:
    every coordinate proposes ``xi_k = u_k + s z`` and accepts it with
    probability ``min(1, phi(xi_k)/phi(u_k))``; the candidate vector is then
    accepted only if ``G(xi) <= b_j``. The run stops when at least ``N p0``
    samples fail (``G <= 0``), and ``P_f = p0^(m-1) N_f / N``.

    Args:
        limit_state: Model limit state.
        samples_per_level: Samples ``N`` per level; ``N p0`` must be a
            positive integer that divides ``N``.
        level_probability: Conditional level probability ``p0`` in
            ``(0, 0.5]``.
        proposal_spread: Standard deviation ``s`` of the per-coordinate
            Gaussian random-walk proposal.
        max_levels: Maximum number of levels before stopping unconverged.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Probability, thresholds, conditional probabilities, c.o.v. estimates,
        acceptance rates, evaluation count, and convergence flag.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If options are inconsistent, or limit-state values tie at
            an intermediate threshold (a continuous response is required).

    Equation:
        [EQ-REL-SUBSET-001], [EQ-REL-SUBSET-002], [EQ-REL-SUBSET-003].

    References:
        [REF-AU-BECK-2001].

    """
    state = _require_limit_state(limit_state)
    total = _require_samples(samples_per_level, "samples_per_level", minimum=10)
    p0 = require_finite_real(level_probability, name="level_probability")
    if not 0.0 < p0 <= 0.5:
        raise ValueError("level_probability must lie in (0, 0.5]")
    seeds_count = round(total * p0)
    if seeds_count < 1 or abs(seeds_count - total * p0) > 1e-9:
        raise ValueError("samples_per_level * level_probability must be an integer")
    if total % seeds_count != 0:
        raise ValueError("samples_per_level * level_probability must divide N")
    chain_length = total // seeds_count
    spread = require_finite_real(proposal_spread, name="proposal_spread")
    if spread <= 0.0:
        raise ValueError("proposal_spread must be positive")
    levels_limit = _require_samples(max_levels, "max_levels", minimum=1)
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
    product = 1.0
    converged = False
    probability = 0.0
    level = 0
    while True:
        correlated = level > 0
        failures = sum(1 for value in values if value <= 0.0)
        if failures >= seeds_count:
            fraction = failures / total
            thresholds.append(0.0)
            conditionals.append(fraction)
            level_covs.append(
                _level_cov(
                    [value <= 0.0 for value in values],
                    seeds_count,
                    fraction,
                    correlated,
                )
            )
            probability = product * fraction
            converged = True
            break
        if level + 1 >= levels_limit:
            fraction = failures / total
            conditionals.append(fraction)
            thresholds.append(0.0)
            level_covs.append(
                _level_cov(
                    [value <= 0.0 for value in values],
                    seeds_count,
                    fraction,
                    correlated,
                )
            )
            probability = product * fraction
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
            )
        )
        product *= p0
        new_points: list[Vector] = []
        new_values: list[float] = []
        accepted = 0
        proposed = 0
        for index in order[:seeds_count]:
            current, current_value = points[index], values[index]
            new_points.append(current)
            new_values.append(current_value)
            for _ in range(chain_length - 1):
                candidate = []
                for coordinate in current:
                    trial = coordinate + spread * generator.gauss(0.0, 1.0)
                    ratio = math.exp(0.5 * (coordinate * coordinate - trial * trial))
                    candidate.append(
                        trial if generator.random() < ratio else coordinate
                    )
                proposed += 1
                proposal = tuple(candidate)
                if proposal != current:
                    proposal_value = state.standard(proposal)
                    if proposal_value <= threshold:
                        current, current_value = proposal, proposal_value
                        accepted += 1
                new_points.append(current)
                new_values.append(current_value)
        acceptance.append(accepted / proposed if proposed else 0.0)
        points, values = new_points, new_values
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
    seed: int | RandomStream | None = None,
) -> MethodComparison:
    """Run FORM, SORM, IS, subset simulation, and crude MC on one limit state.

    FORM is run once and reused by SORM and importance sampling; their rows
    report cumulative cost. SORM contributes two rows, ``"SORM-Breitung"`` and
    ``"SORM-HR"``. Each sampling method uses an independent child stream.

    Args:
        limit_state: Model limit state.
        methods: Subset of ``("form", "sorm", "is", "subset", "mc")``. SORM
            rows are omitted when the FORM index is not positive, because the
            curvature corrections are undefined there.
        monte_carlo_samples: Crude Monte Carlo sample size.
        importance_samples: Design-point importance-sampling sample size.
        subset_samples: Subset-simulation samples per level.
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
        [EQ-REL-PF-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapters 5 and 6], [REF-AU-BECK-2001].

    """
    state = _require_limit_state(limit_state)
    requested = tuple(methods)
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown reliability methods: {unknown}")
    stream = resolve_stream(seed)
    rows: list[MethodComparisonRow] = []
    form_result: FORMResult | None = None
    if {"form", "sorm", "is"} & set(requested):
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
    return MethodComparison(state.label, tuple(rows))
