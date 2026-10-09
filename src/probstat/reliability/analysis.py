"""Reliability estimators built around an explicit limit-state convention."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from probstat.distributions._special import regularized_gamma_q
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.monte_carlo import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    estimate_mean,
    estimate_probability,
)


@dataclass(frozen=True, slots=True)
class LimitState:
    """A named boundary where ``g(x) <= 0`` denotes system failure.

    Args:
        function: Deterministic scalar limit-state function.
        name: Non-empty engineering label.
        dimension: Optional positive input-vector dimension.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 2 and 3].

    """

    function: Callable[[tuple[float, ...]], float]
    name: str
    dimension: int | None = None

    def __post_init__(self) -> None:
        """Validate the callable, label, and optional dimension.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If the label is empty or dimension is not positive.

        References:
            [EQ-REL-LIMIT-001], [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        if not callable(self.function):
            raise TypeError("function must be callable")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.dimension is not None:
            dimension = require_non_negative_integer(
                self.dimension, name="dimension"
            )
            if dimension == 0:
                raise ValueError("dimension must be positive")
            object.__setattr__(self, "dimension", dimension)

    def evaluate(self, sample: Sequence[object]) -> float:
        """Evaluate the limit state at one validated finite vector.

        Args:
            sample: Non-empty finite coordinate sequence.

        Returns:
            Finite limit-state value; non-positive values denote failure.

        Raises:
            TypeError: If the sample or a coordinate has an invalid type.
            ValueError: If the sample is empty, has wrong dimension, or is non-finite.

        References:
            [EQ-REL-LIMIT-001], [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        if not isinstance(sample, Sequence):
            raise TypeError("sample must be a sequence")
        vector = tuple(
            require_finite_real(value, name=f"sample[{index}]")
            for index, value in enumerate(sample)
        )
        if not vector:
            raise ValueError("sample must not be empty")
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError("sample dimension does not match the limit state")
        return require_finite_real(self.function(vector), name="limit-state value")

    def failed(self, sample: Sequence[object]) -> bool:
        """Return whether one sample lies in the closed failure domain.

        Args:
            sample: Input vector accepted by :meth:`evaluate`.

        Returns:
            ``True`` exactly when ``g(sample) <= 0``.

        References:
            [EQ-REL-LIMIT-001], [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        return self.evaluate(sample) <= 0.0


@dataclass(frozen=True, slots=True)
class DirectReliabilityResult:
    """Direct Monte Carlo failure and reliability estimates.

    Args:
        failure: Bernoulli estimate for ``P(g(X)<=0)``.
        reliability: Complementary empirical satisfactory-performance rate.
        limit_state_name: Name of the evaluated failure boundary.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 3 and 5].

    """

    failure: ProbabilityEstimate
    reliability: float
    limit_state_name: str

    def __post_init__(self) -> None:
        """Validate the probability complement and descriptive label.

        Raises:
            TypeError: If fields have invalid types.
            ValueError: If reliability is inconsistent or the label is empty.

        References:
            [EQ-REL-PF-001], [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        if not isinstance(self.failure, ProbabilityEstimate):
            raise TypeError("failure must be a ProbabilityEstimate")
        reliability = require_probability(self.reliability, name="reliability")
        if not math.isclose(
            reliability, 1.0 - self.failure.probability, abs_tol=1e-15
        ):
            raise ValueError("reliability must complement failure probability")
        if not isinstance(self.limit_state_name, str):
            raise TypeError("limit_state_name must be a string")
        if not self.limit_state_name.strip():
            raise ValueError("limit_state_name must not be empty")
        object.__setattr__(self, "reliability", reliability)

    @property
    def standard_error(self) -> float:
        """Return the Bernoulli plug-in standard error.

        References:
            [EQ-MC-PROB-001], [REF-WILSON-1927].

        """
        return self.failure.standard_error

    @property
    def evaluations(self) -> int:
        """Return the number of limit-state evaluations.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5]. No equation is evaluated.

        """
        return self.failure.evaluations


@dataclass(frozen=True, slots=True)
class WeightedFailureResult:
    """An ordinary importance estimate for a rare failure event.

    Args:
        estimate: Mean of ``I(g(x)<=0) p(x)/q(x)`` contributions.
        observed_failures: Failure-region draws under the proposal.
        weight_effective_sample_size: Weight-concentration diagnostic.
        max_normalized_weight: Largest proposal-to-target normalized weight.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6], [REF-OWEN-2013, Chapter 9].

    """

    estimate: MonteCarloEstimate
    observed_failures: int
    weight_effective_sample_size: float
    max_normalized_weight: float

    def __post_init__(self) -> None:
        """Validate event counts and importance-weight diagnostics.

        Raises:
            TypeError: If fields have invalid types.
            ValueError: If counts or diagnostics are inconsistent.

        References:
            [EQ-REL-IS-001], [EQ-MC-ESS-001], [REF-OWEN-2013, Chapter 9].

        """
        if not isinstance(self.estimate, MonteCarloEstimate):
            raise TypeError("estimate must be a MonteCarloEstimate")
        failures = require_non_negative_integer(
            self.observed_failures, name="observed_failures"
        )
        if failures > self.estimate.sample_size:
            raise ValueError("observed_failures cannot exceed sample_size")
        ess = require_non_negative_finite_real(
            self.weight_effective_sample_size,
            name="weight_effective_sample_size",
        )
        if not 1.0 <= ess <= self.estimate.sample_size + 1e-12:
            raise ValueError("weight_effective_sample_size is outside [1, N]")
        max_weight = require_probability(
            self.max_normalized_weight, name="max_normalized_weight"
        )
        if max_weight == 0.0:
            raise ValueError("max_normalized_weight must be positive")
        object.__setattr__(self, "observed_failures", failures)
        object.__setattr__(self, "weight_effective_sample_size", ess)
        object.__setattr__(self, "max_normalized_weight", max_weight)


@dataclass(frozen=True, slots=True)
class LinearNormalFORMResult:
    """Closed-form FORM result for an affine standard-normal limit state.

    Args:
        reliability_index: Signed distance ``beta`` from origin to boundary.
        failure_probability: FORM value ``Phi(-beta)``.
        design_point: Nearest boundary point in standard-normal space.
        unit_normal: Unit coefficient direction pointing toward increasing ``g``.
        boundary_residual: Evaluated affine boundary residual at design point.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 4 and 5].

    """

    reliability_index: float
    failure_probability: float
    design_point: tuple[float, ...]
    unit_normal: tuple[float, ...]
    boundary_residual: float

    def __post_init__(self) -> None:
        """Validate the FORM geometry and probability domain.

        Raises:
            TypeError: If fields contain invalid values.
            ValueError: If vector dimensions or probability domains are invalid.

        References:
            [EQ-REL-FORM-001], [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        beta = require_finite_real(self.reliability_index, name="reliability_index")
        probability = require_probability(
            self.failure_probability, name="failure_probability"
        )
        residual = require_finite_real(
            self.boundary_residual, name="boundary_residual"
        )
        if not self.design_point or len(self.design_point) != len(self.unit_normal):
            raise ValueError("FORM vectors must be non-empty and equally sized")
        point = tuple(
            require_finite_real(value, name=f"design_point[{index}]")
            for index, value in enumerate(self.design_point)
        )
        normal = tuple(
            require_finite_real(value, name=f"unit_normal[{index}]")
            for index, value in enumerate(self.unit_normal)
        )
        if not math.isclose(math.sqrt(math.fsum(x * x for x in normal)), 1.0):
            raise ValueError("unit_normal must have Euclidean norm one")
        object.__setattr__(self, "reliability_index", beta)
        object.__setattr__(self, "failure_probability", probability)
        object.__setattr__(self, "design_point", point)
        object.__setattr__(self, "unit_normal", normal)
        object.__setattr__(self, "boundary_residual", residual)


@dataclass(frozen=True, slots=True)
class DirectionalSamplingResult:
    """Directional failure estimate in independent standard-normal space.

    Args:
        estimate: Mean conditional radial failure probability over directions.
        dimension: Standard-normal space dimension.
        directions: Number of independently sampled unit directions.
        finite_boundaries: Directions with a finite failure-entry radius.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6].

    """

    estimate: MonteCarloEstimate
    dimension: int
    directions: int
    finite_boundaries: int

    def __post_init__(self) -> None:
        """Validate directional counts and nested estimate consistency.

        Raises:
            TypeError: If fields have invalid types.
            ValueError: If dimensions or counts are inconsistent.

        References:
            [EQ-REL-DIR-001], [REF-MELCHERS-BECK-2018, Chapter 6].

        """
        if not isinstance(self.estimate, MonteCarloEstimate):
            raise TypeError("estimate must be a MonteCarloEstimate")
        dimension = require_non_negative_integer(self.dimension, name="dimension")
        directions = require_non_negative_integer(self.directions, name="directions")
        finite = require_non_negative_integer(
            self.finite_boundaries, name="finite_boundaries"
        )
        if dimension == 0:
            raise ValueError("dimension must be positive")
        if directions != self.estimate.sample_size:
            raise ValueError("directions must equal estimate sample_size")
        if finite > directions:
            raise ValueError("finite_boundaries cannot exceed directions")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "finite_boundaries", finite)

    @property
    def failure_probability(self) -> float:
        """Return the estimated directional failure probability.

        References:
            [EQ-REL-DIR-001], [REF-MELCHERS-BECK-2018, Chapter 6].

        """
        return self.estimate.estimate

    @property
    def standard_error(self) -> float:
        """Return directional sampling standard error.

        References:
            [EQ-MC-SE-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

        """
        return self.estimate.standard_error

    @property
    def evaluations(self) -> int:
        """Return charged radial-boundary evaluations.

        References:
            [REF-MELCHERS-BECK-2018, Chapter 6]. No equation is evaluated.

        """
        return self.estimate.evaluations


def estimate_failure_probability(
    limit_state: LimitState,
    samples: Iterable[Sequence[object]],
    *,
    confidence: float = 0.95,
) -> DirectReliabilityResult:
    """Estimate failure probability by direct Monte Carlo simulation.

    Args:
        limit_state: Boundary using the ``g(x)<=0`` failure convention.
        samples: Non-empty input-vector realizations.
        confidence: Nominal Wilson-interval confidence in ``(0,1)``.

    Returns:
        Failure probability, Wilson interval, complement reliability, and cost.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If no samples are supplied or an evaluation is invalid.

    Equation:
        [EQ-REL-PF-001], [EQ-MC-PROB-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5], [REF-WILSON-1927].

    """
    if not isinstance(limit_state, LimitState):
        raise TypeError("limit_state must be a LimitState")
    events = tuple(limit_state.failed(sample) for sample in samples)
    if not events:
        raise ValueError("samples must not be empty")
    failure = estimate_probability(events, confidence=confidence)
    return DirectReliabilityResult(
        failure,
        1.0 - failure.probability,
        limit_state.name,
    )


def estimate_weighted_failure_probability(
    limit_state: LimitState,
    samples: Iterable[Sequence[object]],
    likelihood_ratios: Iterable[object],
    *,
    confidence: float = 0.95,
) -> WeightedFailureResult:
    """Estimate a rare failure probability from proposal draws and ``p/q`` weights.

    Args:
        limit_state: Failure boundary evaluated at proposal draws.
        samples: At least two proposal-distributed input vectors.
        likelihood_ratios: Matching non-negative finite target/proposal weights.
        confidence: Nominal normal-approximation confidence in ``(0,1)``.

    Returns:
        Ordinary importance estimate and weight-concentration diagnostics.

    Raises:
        TypeError: If inputs have invalid types.
        ValueError: If lengths differ, fewer than two samples exist, all weights
            are zero, or an evaluation is invalid.

    Equation:
        [EQ-REL-IS-001], [EQ-MC-ESS-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6], [REF-OWEN-2013, Chapter 9].

    """
    if not isinstance(limit_state, LimitState):
        raise TypeError("limit_state must be a LimitState")
    materialized_samples = tuple(samples)
    weights = tuple(
        require_non_negative_finite_real(value, name=f"likelihood_ratios[{index}]")
        for index, value in enumerate(likelihood_ratios)
    )
    if len(materialized_samples) != len(weights):
        raise ValueError("samples and likelihood_ratios must have equal lengths")
    if len(weights) < 2:
        raise ValueError("weighted estimation requires at least two samples")
    failures = tuple(limit_state.failed(sample) for sample in materialized_samples)
    weight_sum = math.fsum(weights)
    if weight_sum == 0.0:
        raise ValueError("at least one likelihood ratio must be positive")
    contributions = tuple(
        weight if failed else 0.0
        for failed, weight in zip(failures, weights, strict=True)
    )
    estimate = estimate_mean(contributions, confidence=confidence)
    squared_sum = math.fsum(weight * weight for weight in weights)
    return WeightedFailureResult(
        estimate,
        sum(failures),
        weight_sum**2 / squared_sum,
        max(weights) / weight_sum,
    )


def linear_normal_form(
    intercept: float,
    coefficients: Iterable[object],
) -> LinearNormalFORMResult:
    """Solve FORM exactly for ``g(u)=intercept+a^T u`` and ``U~N(0,I)``.

    Args:
        intercept: Finite affine intercept.
        coefficients: Non-empty finite vector with positive Euclidean norm.

    Returns:
        Signed reliability index, ``Phi(-beta)``, and nearest boundary point.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If coefficients are empty or identically zero.

    Equation:
        [EQ-REL-FORM-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5].

    """
    checked_intercept = require_finite_real(intercept, name="intercept")
    vector = tuple(
        require_finite_real(value, name=f"coefficients[{index}]")
        for index, value in enumerate(coefficients)
    )
    if not vector:
        raise ValueError("coefficients must not be empty")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("coefficients must have positive Euclidean norm")
    unit_normal = tuple(value / norm for value in vector)
    beta = checked_intercept / norm
    design_point = tuple(-beta * value for value in unit_normal)
    residual = checked_intercept + math.fsum(
        coefficient * point
        for coefficient, point in zip(vector, design_point, strict=True)
    )
    return LinearNormalFORMResult(
        beta,
        NormalDist().cdf(-beta),
        design_point,
        unit_normal,
        residual,
    )


def estimate_directional_failure_probability(
    radial_boundary: Callable[[tuple[float, ...]], float | None],
    *,
    dimension: int,
    directions: int,
    seed: int | None = None,
    confidence: float = 0.95,
) -> DirectionalSamplingResult:
    """Estimate failure probability from radial boundaries along random directions.

    The failure set must be star-shaped from the standard-normal origin along
    sampled directions: ``None`` means no failure on that ray, while a finite
    non-negative radius means failure begins at and beyond that radius.

    Args:
        radial_boundary: Unit-direction callable returning entry radius or ``None``.
        dimension: Positive independent standard-normal dimension.
        directions: Number of random unit directions, at least two.
        seed: Optional seed for a call-local pseudo-random generator.
        confidence: Nominal normal-approximation confidence in ``(0,1)``.

    Returns:
        Mean conditional chi-radius tail probability and sampling diagnostics.

    Raises:
        TypeError: If arguments or returned radii have invalid types.
        ValueError: If counts/radii are invalid or confidence is outside domain.

    Equation:
        [EQ-REL-DIR-001].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 6].

    """
    if not callable(radial_boundary):
        raise TypeError("radial_boundary must be callable")
    checked_dimension = require_non_negative_integer(dimension, name="dimension")
    checked_directions = require_non_negative_integer(directions, name="directions")
    if checked_dimension == 0:
        raise ValueError("dimension must be positive")
    if checked_directions < 2:
        raise ValueError("directions must be at least two")
    generator = random.Random(seed)
    contributions: list[float] = []
    finite_boundaries = 0
    for index in range(checked_directions):
        raw = tuple(generator.gauss(0.0, 1.0) for _ in range(checked_dimension))
        norm = math.sqrt(math.fsum(value * value for value in raw))
        direction = tuple(value / norm for value in raw)
        radius = radial_boundary(direction)
        if radius is None:
            contributions.append(0.0)
            continue
        checked_radius = require_non_negative_finite_real(
            radius, name=f"radial_boundary(direction {index})"
        )
        finite_boundaries += 1
        contributions.append(
            regularized_gamma_q(
                0.5 * checked_dimension,
                0.5 * checked_radius**2,
            )
        )
    estimate = estimate_mean(contributions, confidence=confidence)
    return DirectionalSamplingResult(
        estimate,
        checked_dimension,
        checked_directions,
        finite_boundaries,
    )
