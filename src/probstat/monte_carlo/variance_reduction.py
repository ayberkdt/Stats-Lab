"""Variance-reduced Monte Carlo estimators and cost-aware diagnostics."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.inference import ConfidenceInterval

from .estimation import MonteCarloEstimate, estimate_mean


class CostedEstimate(Protocol):
    """Structural contract for an estimator with error and work metadata.

    Attributes:
        standard_error: Non-negative estimated standard deviation of the estimate.
        evaluations: Number of model or integrand evaluations charged.

    References:
        [REF-OWEN-2013, Chapters 2 and 8].

    """

    standard_error: float
    evaluations: int


@dataclass(frozen=True, slots=True)
class ControlVariateResult:
    """A control-variate estimate and its fitted adjustment diagnostics.

    Args:
        estimate: Estimate formed from adjusted sample contributions.
        coefficient: Applied control coefficient ``beta``.
        raw_variance: Unbiased sample variance before adjustment.
        adjusted_variance: Unbiased sample variance after adjustment.

    References:
        [REF-OWEN-2013, Chapter 8], [REF-ROBERT-CASELLA-2004, Chapter 4].

    """

    estimate: MonteCarloEstimate
    coefficient: float
    raw_variance: float
    adjusted_variance: float

    def __post_init__(self) -> None:
        """Validate the nested estimate and variance diagnostics.

        Raises:
            TypeError: If the nested estimate or coefficient has an invalid type.
            ValueError: If a variance is negative or a value is non-finite.

        References:
            [EQ-MC-CONTROL-001], [REF-OWEN-2013, Chapter 8].

        """
        if not isinstance(self.estimate, MonteCarloEstimate):
            raise TypeError("estimate must be a MonteCarloEstimate")
        coefficient = require_finite_real(self.coefficient, name="coefficient")
        raw_variance = require_non_negative_finite_real(
            self.raw_variance, name="raw_variance"
        )
        adjusted_variance = require_non_negative_finite_real(
            self.adjusted_variance, name="adjusted_variance"
        )
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "raw_variance", raw_variance)
        object.__setattr__(self, "adjusted_variance", adjusted_variance)


@dataclass(frozen=True, slots=True)
class StratifiedEstimate:
    """An equal-width stratified estimate with design-based uncertainty.

    Args:
        estimate: Weighted mean of the stratum sample means.
        estimator_variance: Estimated variance of the stratified mean.
        standard_error: Square root of ``estimator_variance``.
        confidence_interval: Two-sided large-sample normal interval.
        stratum_means: Sample mean in each equal-width stratum.
        stratum_variances: Unbiased sample variance in each stratum.
        allocations: Number of evaluations assigned to each stratum.
        evaluations: Total number of integrand evaluations.

    References:
        [REF-OWEN-2013, Chapter 8].

    """

    estimate: float
    estimator_variance: float
    standard_error: float
    confidence_interval: ConfidenceInterval
    stratum_means: tuple[float, ...]
    stratum_variances: tuple[float, ...]
    allocations: tuple[int, ...]
    evaluations: int

    def __post_init__(self) -> None:
        """Validate the stratified-design result invariants.

        Raises:
            TypeError: If fields have invalid types.
            ValueError: If dimensions, allocations, uncertainty, or cost disagree.

        References:
            [EQ-MC-STRAT-001], [REF-OWEN-2013, Chapter 8].

        """
        estimate = require_finite_real(self.estimate, name="estimate")
        estimator_variance = require_non_negative_finite_real(
            self.estimator_variance, name="estimator_variance"
        )
        standard_error = require_non_negative_finite_real(
            self.standard_error, name="standard_error"
        )
        evaluations = require_non_negative_integer(
            self.evaluations, name="evaluations"
        )
        if not isinstance(self.confidence_interval, ConfidenceInterval):
            raise TypeError("confidence_interval must be a ConfidenceInterval")
        if not self.allocations:
            raise ValueError("allocations must not be empty")
        if not (
            len(self.stratum_means)
            == len(self.stratum_variances)
            == len(self.allocations)
        ):
            raise ValueError("stratum diagnostics and allocations must align")
        checked_allocations = tuple(
            require_non_negative_integer(value, name=f"allocations[{index}]")
            for index, value in enumerate(self.allocations)
        )
        if any(value < 2 for value in checked_allocations):
            raise ValueError("each stratum allocation must be at least two")
        if sum(checked_allocations) != evaluations:
            raise ValueError("evaluations must equal the allocation total")
        checked_means = tuple(
            require_finite_real(value, name=f"stratum_means[{index}]")
            for index, value in enumerate(self.stratum_means)
        )
        checked_variances = tuple(
            require_non_negative_finite_real(
                value, name=f"stratum_variances[{index}]"
            )
            for index, value in enumerate(self.stratum_variances)
        )
        if not math.isclose(
            standard_error**2,
            estimator_variance,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("standard_error must square to estimator_variance")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "estimator_variance", estimator_variance)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "stratum_means", checked_means)
        object.__setattr__(self, "stratum_variances", checked_variances)
        object.__setattr__(self, "allocations", checked_allocations)
        object.__setattr__(self, "evaluations", evaluations)


@dataclass(frozen=True, slots=True)
class ImportanceSamplingResult:
    """An importance-sampling estimate with weight-degeneracy diagnostics.

    Args:
        estimate: Mean estimate of weighted integrand contributions.
        effective_sample_size: Weight-only heuristic in ``[1, N]``.
        max_normalized_weight: Largest normalized importance weight.

    References:
        [REF-OWEN-2013, Chapter 9], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """

    estimate: MonteCarloEstimate
    effective_sample_size: float
    max_normalized_weight: float

    def __post_init__(self) -> None:
        """Validate the estimate and weight diagnostics.

        Raises:
            TypeError: If the nested estimate has an invalid type.
            ValueError: If weight diagnostics lie outside their domains.

        References:
            [EQ-MC-ESS-001], [REF-OWEN-2013, Chapter 9].

        """
        if not isinstance(self.estimate, MonteCarloEstimate):
            raise TypeError("estimate must be a MonteCarloEstimate")
        effective_sample_size = require_non_negative_finite_real(
            self.effective_sample_size, name="effective_sample_size"
        )
        max_weight = require_probability(
            self.max_normalized_weight, name="max_normalized_weight"
        )
        if not 1.0 <= effective_sample_size <= self.estimate.sample_size + 1e-12:
            raise ValueError("effective_sample_size must lie in [1, sample_size]")
        if max_weight <= 0.0:
            raise ValueError("max_normalized_weight must be positive")
        object.__setattr__(self, "effective_sample_size", effective_sample_size)
        object.__setattr__(self, "max_normalized_weight", max_weight)

    @property
    def standard_error(self) -> float:
        """Return the weighted-contribution Monte Carlo standard error.

        References:
            [EQ-MC-SE-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

        """
        return self.estimate.standard_error

    @property
    def evaluations(self) -> int:
        """Return the number of weighted integrand evaluations.

        References:
            [REF-OWEN-2013, Chapter 9]. No equation is evaluated.

        """
        return self.estimate.evaluations


@dataclass(frozen=True, slots=True)
class EfficiencyComparison:
    """Cost-normalized comparison between two Monte Carlo strategies.

    Args:
        baseline_label: Non-empty name of the reference strategy.
        candidate_label: Non-empty name of the compared strategy.
        baseline_variance_cost: Baseline ``SE^2 * evaluations`` score.
        candidate_variance_cost: Candidate ``SE^2 * evaluations`` score.
        efficiency_gain: Baseline score divided by candidate score; values above
            one favor the candidate, while infinity denotes zero candidate error.

    References:
        [REF-OWEN-2013, Chapter 2].

    """

    baseline_label: str
    candidate_label: str
    baseline_variance_cost: float
    candidate_variance_cost: float
    efficiency_gain: float

    def __post_init__(self) -> None:
        """Validate comparison labels and non-negative cost scores.

        Raises:
            TypeError: If a label or score has an invalid type.
            ValueError: If a label is empty or a score is invalid.

        References:
            [EQ-MC-EFF-001], [REF-OWEN-2013, Chapter 2].

        """
        for name, value in (
            ("baseline_label", self.baseline_label),
            ("candidate_label", self.candidate_label),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        baseline = require_non_negative_finite_real(
            self.baseline_variance_cost, name="baseline_variance_cost"
        )
        candidate = require_non_negative_finite_real(
            self.candidate_variance_cost, name="candidate_variance_cost"
        )
        if isinstance(self.efficiency_gain, bool) or not isinstance(
            self.efficiency_gain, int | float
        ):
            raise TypeError("efficiency_gain must be a real number")
        gain = float(self.efficiency_gain)
        if math.isnan(gain) or gain < 0.0:
            raise ValueError("efficiency_gain must be non-negative and not NaN")
        object.__setattr__(self, "baseline_variance_cost", baseline)
        object.__setattr__(self, "candidate_variance_cost", candidate)
        object.__setattr__(self, "efficiency_gain", gain)


def estimate_antithetic_uniform(
    function: Callable[[float], float],
    *,
    pairs: int,
    seed: int | None = None,
    confidence: float = 0.95,
) -> MonteCarloEstimate:
    """Estimate ``E[f(U)]`` using paired ``U`` and ``1-U`` evaluations.

    Args:
        function: Scalar integrand on the unit interval.
        pairs: Number of independent antithetic pairs, at least two.
        seed: Optional seed for a call-local pseudo-random generator.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Estimate whose independent contributions are pair averages and whose
        evaluation count is twice the number of pairs.

    Raises:
        TypeError: If inputs or evaluated values have invalid types.
        ValueError: If the pair count, confidence, or values are invalid.

    Equation:
        [EQ-MC-ANTI-001].

    References:
        [REF-OWEN-2013, Section 8.2].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    checked_pairs = require_non_negative_integer(pairs, name="pairs")
    if checked_pairs < 2:
        raise ValueError("pairs must be at least two")
    generator = random.Random(seed)
    contributions: list[float] = []
    for index in range(checked_pairs):
        uniform = generator.random()
        first = require_finite_real(
            function(uniform), name=f"function(pair {index}, first)"
        )
        second = require_finite_real(
            function(1.0 - uniform), name=f"function(pair {index}, second)"
        )
        contributions.append(0.5 * (first + second))
    result = estimate_mean(contributions, confidence=confidence)
    return MonteCarloEstimate(
        result.estimate,
        result.sample_variance,
        result.standard_error,
        result.confidence_interval,
        result.sample_size,
        2 * checked_pairs,
    )


def estimate_control_variate(
    values: Iterable[object],
    controls: Iterable[object],
    *,
    known_control_mean: float,
    coefficient: float | None = None,
    confidence: float = 0.95,
) -> ControlVariateResult:
    """Adjust sampled outputs using a correlated control with known mean.

    Args:
        values: At least two finite target contributions.
        controls: Matching finite control observations.
        known_control_mean: Known expectation of the control variable.
        coefficient: Optional fixed ``beta``; when omitted, use sample
            covariance divided by sample control variance.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Adjusted mean estimate, applied coefficient, and variance diagnostics.

    Raises:
        TypeError: If samples or scalar inputs have invalid types.
        ValueError: If sample lengths differ, fewer than two pairs are supplied,
            or automatic fitting sees zero control variance.

    Equation:
        [EQ-MC-CONTROL-001].

    References:
        [REF-OWEN-2013, Section 8.9], [REF-ROBERT-CASELLA-2004, Chapter 4].

    """
    target_sample = coerce_finite_sample(values, name="values")
    control_sample = coerce_finite_sample(controls, name="controls")
    if len(target_sample) != len(control_sample):
        raise ValueError("values and controls must have equal lengths")
    if len(target_sample) < 2:
        raise ValueError("control-variate estimation requires at least two pairs")
    control_mean = require_finite_real(
        known_control_mean, name="known_control_mean"
    )
    raw_variance = statistics.variance(target_sample)
    if coefficient is None:
        control_variance = statistics.variance(control_sample)
        if control_variance == 0.0:
            raise ValueError("automatic coefficient requires control variation")
        fitted_coefficient = (
            statistics.covariance(target_sample, control_sample) / control_variance
        )
    else:
        fitted_coefficient = require_finite_real(coefficient, name="coefficient")
    adjusted = tuple(
        value - fitted_coefficient * (control - control_mean)
        for value, control in zip(target_sample, control_sample, strict=True)
    )
    result = estimate_mean(adjusted, confidence=confidence)
    return ControlVariateResult(
        result,
        fitted_coefficient,
        raw_variance,
        result.sample_variance,
    )


def estimate_conditional_expectation(
    conditional_expectations: Iterable[object],
    *,
    confidence: float = 0.95,
) -> MonteCarloEstimate:
    """Estimate an expectation from exact or analytically computed conditionals.

    The caller supplies realizations of ``E[Y | Z]``. Replacing raw ``Y`` by
    these conditional expectations preserves the mean and cannot increase the
    ideal population variance under the law of total variance.

    Args:
        conditional_expectations: At least two finite conditional means.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Sample-mean estimate of the supplied conditional expectations.

    Raises:
        TypeError: If values or confidence have invalid types.
        ValueError: If the sample is too short, non-finite, or confidence invalid.

    Equation:
        [EQ-MC-COND-001].

    References:
        [REF-OWEN-2013, Section 8.1], [REF-ROBERT-CASELLA-2004, Chapter 4].

    """
    return estimate_mean(conditional_expectations, confidence=confidence)


def estimate_stratified_uniform(
    function: Callable[[float], float],
    allocations: Sequence[int],
    *,
    seed: int | None = None,
    confidence: float = 0.95,
) -> StratifiedEstimate:
    """Estimate ``E[f(U)]`` over equal-width unit-interval strata.

    Args:
        function: Scalar integrand evaluated on the unit interval.
        allocations: Per-stratum sample counts; each must be at least two.
        seed: Optional seed for a call-local pseudo-random generator.
        confidence: Nominal large-sample confidence in ``(0, 1)``.

    Returns:
        Weighted stratified estimate with design-based variance and cost.

    Raises:
        TypeError: If the function, allocations, confidence, or outputs are invalid.
        ValueError: If no strata exist, an allocation is too small, or values fail.

    Equation:
        [EQ-MC-STRAT-001].

    References:
        [REF-OWEN-2013, Section 8.4].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(allocations, Sequence):
        raise TypeError("allocations must be a sequence")
    checked_allocations = tuple(
        require_non_negative_integer(value, name=f"allocations[{index}]")
        for index, value in enumerate(allocations)
    )
    if not checked_allocations:
        raise ValueError("allocations must not be empty")
    if any(value < 2 for value in checked_allocations):
        raise ValueError("each stratum allocation must be at least two")
    checked_confidence = _confidence_level(confidence)
    strata = len(checked_allocations)
    weight = 1.0 / strata
    generator = random.Random(seed)
    means: list[float] = []
    variances: list[float] = []
    for stratum, allocation in enumerate(checked_allocations):
        lower = stratum / strata
        upper = (stratum + 1) / strata
        sample = tuple(
            require_finite_real(
                function(generator.uniform(lower, upper)),
                name=f"function(stratum {stratum}, sample {index})",
            )
            for index in range(allocation)
        )
        means.append(math.fsum(sample) / allocation)
        variances.append(statistics.variance(sample))
    estimate = weight * math.fsum(means)
    estimator_variance = math.fsum(
        weight**2 * variance / allocation
        for variance, allocation in zip(
            variances, checked_allocations, strict=True
        )
    )
    standard_error = math.sqrt(estimator_variance)
    if standard_error == 0.0:
        interval = ConfidenceInterval(
            estimate,
            estimate,
            checked_confidence,
            "degenerate stratified normal approximation",
        )
    else:
        critical = NormalDist().inv_cdf(0.5 + 0.5 * checked_confidence)
        half_width = critical * standard_error
        interval = ConfidenceInterval(
            estimate - half_width,
            estimate + half_width,
            checked_confidence,
            "stratified large-sample normal approximation",
        )
    return StratifiedEstimate(
        estimate,
        estimator_variance,
        standard_error,
        interval,
        tuple(means),
        tuple(variances),
        checked_allocations,
        sum(checked_allocations),
    )


def estimate_importance_sampling(
    integrand: Callable[[float], float],
    target_density: Callable[[float], float],
    proposal_density: Callable[[float], float],
    proposal_sampler: Callable[[random.Random], float],
    *,
    sample_size: int,
    seed: int | None = None,
    confidence: float = 0.95,
) -> ImportanceSamplingResult:
    """Estimate a target expectation using ordinary importance sampling.

    Args:
        integrand: Target quantity ``g(x)``.
        target_density: Non-negative target density ``p(x)``.
        proposal_density: Strictly positive proposal density ``q(x)`` at draws.
        proposal_sampler: Callable receiving a local random generator and
            returning one proposal draw.
        sample_size: Number of weighted contributions, at least two.
        seed: Optional seed for the call-local pseudo-random generator.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Ordinary weighted mean plus weight-only effective-sample diagnostics.

    Raises:
        TypeError: If inputs or callable results have invalid types.
        ValueError: If sizes, densities, weights, or confidence are invalid.

    Equation:
        [EQ-MC-IS-001], [EQ-MC-ESS-001].

    References:
        [REF-OWEN-2013, Chapter 9], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    for name, function in (
        ("integrand", integrand),
        ("target_density", target_density),
        ("proposal_density", proposal_density),
        ("proposal_sampler", proposal_sampler),
    ):
        if not callable(function):
            raise TypeError(f"{name} must be callable")
    checked_size = require_non_negative_integer(sample_size, name="sample_size")
    if checked_size < 2:
        raise ValueError("sample_size must be at least two")
    generator = random.Random(seed)
    contributions: list[float] = []
    weights: list[float] = []
    for index in range(checked_size):
        draw = require_finite_real(
            proposal_sampler(generator), name=f"proposal_sampler(draw {index})"
        )
        target = require_non_negative_finite_real(
            target_density(draw), name=f"target_density(draw {index})"
        )
        proposal = require_non_negative_finite_real(
            proposal_density(draw), name=f"proposal_density(draw {index})"
        )
        if proposal == 0.0:
            raise ValueError("proposal density must be positive at every draw")
        weight = target / proposal
        contribution = require_finite_real(
            integrand(draw) * weight, name=f"weighted contribution {index}"
        )
        weights.append(weight)
        contributions.append(contribution)
    weight_sum = math.fsum(weights)
    if weight_sum == 0.0:
        raise ValueError("at least one importance weight must be positive")
    squared_weight_sum = math.fsum(weight * weight for weight in weights)
    effective_sample_size = weight_sum**2 / squared_weight_sum
    max_normalized_weight = max(weights) / weight_sum
    return ImportanceSamplingResult(
        estimate_mean(contributions, confidence=confidence),
        effective_sample_size,
        max_normalized_weight,
    )


def allocate_stratified_budget(
    stratum_standard_deviations: Iterable[object],
    total_samples: int,
    *,
    minimum_per_stratum: int = 2,
) -> tuple[int, ...]:
    """Allocate an integer budget by a lower-bounded Neyman rule.

    Args:
        stratum_standard_deviations: Non-negative pilot standard deviations.
        total_samples: Total number of samples to allocate.
        minimum_per_stratum: Guaranteed samples in every stratum.

    Returns:
        Integer allocations summing exactly to ``total_samples``. Remaining
        samples after lower bounds are proportional to pilot standard deviations;
        all-zero pilot scales receive an even deterministic allocation.

    Raises:
        TypeError: If scales or counts have invalid types.
        ValueError: If scales are negative/non-finite or budget is insufficient.

    Equation:
        [EQ-MC-NEYMAN-001].

    References:
        [REF-OWEN-2013, Section 8.4].

    """
    scales = tuple(
        require_non_negative_finite_real(value, name=f"standard_deviations[{index}]")
        for index, value in enumerate(stratum_standard_deviations)
    )
    if not scales:
        raise ValueError("stratum_standard_deviations must not be empty")
    checked_total = require_non_negative_integer(total_samples, name="total_samples")
    checked_minimum = require_non_negative_integer(
        minimum_per_stratum, name="minimum_per_stratum"
    )
    if checked_minimum < 2:
        raise ValueError("minimum_per_stratum must be at least two")
    required = checked_minimum * len(scales)
    if checked_total < required:
        raise ValueError("total_samples cannot satisfy the per-stratum minimum")
    remaining = checked_total - required
    if remaining == 0:
        return tuple(checked_minimum for _ in scales)
    scale_sum = math.fsum(scales)
    if scale_sum == 0.0:
        exact_extras = tuple(remaining / len(scales) for _ in scales)
    else:
        exact_extras = tuple(remaining * scale / scale_sum for scale in scales)
    floors = [math.floor(value) for value in exact_extras]
    unassigned = remaining - sum(floors)
    order = sorted(
        range(len(scales)),
        key=lambda index: (exact_extras[index] - floors[index], -index),
        reverse=True,
    )
    for index in order[:unassigned]:
        floors[index] += 1
    return tuple(checked_minimum + extra for extra in floors)


def compare_efficiency(
    baseline: CostedEstimate,
    candidate: CostedEstimate,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> EfficiencyComparison:
    """Compare estimator uncertainty after normalizing for evaluation cost.

    Args:
        baseline: Reference estimate exposing standard error and evaluations.
        candidate: Compared estimate exposing standard error and evaluations.
        baseline_label: Human-readable reference strategy name.
        candidate_label: Human-readable candidate strategy name.

    Returns:
        Variance-cost scores and their ratio. A gain above one indicates that
        the candidate achieved lower estimated variance per unit work.

    Raises:
        TypeError: If estimator attributes have invalid types.
        ValueError: If errors/costs are invalid or both scores are zero.

    Equation:
        [EQ-MC-EFF-001].

    References:
        [REF-OWEN-2013, Chapter 2].

    """
    baseline_error = require_non_negative_finite_real(
        baseline.standard_error, name="baseline.standard_error"
    )
    candidate_error = require_non_negative_finite_real(
        candidate.standard_error, name="candidate.standard_error"
    )
    baseline_evaluations = require_non_negative_integer(
        baseline.evaluations, name="baseline.evaluations"
    )
    candidate_evaluations = require_non_negative_integer(
        candidate.evaluations, name="candidate.evaluations"
    )
    if baseline_evaluations == 0 or candidate_evaluations == 0:
        raise ValueError("estimator evaluations must be positive")
    baseline_score = baseline_error**2 * baseline_evaluations
    candidate_score = candidate_error**2 * candidate_evaluations
    if baseline_score == 0.0 and candidate_score == 0.0:
        raise ValueError("efficiency is undefined when both scores are zero")
    gain = math.inf if candidate_score == 0.0 else baseline_score / candidate_score
    return EfficiencyComparison(
        baseline_label,
        candidate_label,
        baseline_score,
        candidate_score,
        gain,
    )


def _confidence_level(value: float) -> float:
    """Validate a confidence level strictly between zero and one.

    Args:
        value: Candidate nominal confidence.

    Returns:
        Validated confidence as a float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If ``value`` is not strictly between zero and one.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9]. No equation is evaluated.

    """
    confidence = require_probability(value, name="confidence")
    if confidence in (0.0, 1.0):
        raise ValueError("confidence must be strictly between zero and one")
    return confidence
