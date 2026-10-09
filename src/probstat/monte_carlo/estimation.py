"""Basic Monte Carlo estimators with uncertainty and cost metadata."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.inference import (
    ConfidenceInterval,
    normal_mean_confidence_interval,
    wilson_score_interval,
)


@dataclass(frozen=True, slots=True)
class MonteCarloEstimate:
    """A sample-mean estimate with sampling uncertainty and work count.

    Args:
        estimate: Finite sample mean or scaled integral estimate.
        sample_variance: Non-negative unbiased variance of sampled contributions.
        standard_error: Non-negative estimated standard deviation of the mean.
        confidence_interval: Two-sided normal-approximation interval.
        sample_size: Number of independent sampled contributions.
        evaluations: Number of integrand or model evaluations charged.

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4].

    """

    estimate: float
    sample_variance: float
    standard_error: float
    confidence_interval: ConfidenceInterval
    sample_size: int
    evaluations: int

    def __post_init__(self) -> None:
        """Validate estimate, uncertainty, sample-size, and cost invariants.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If a numerical field has an invalid domain.

        References:
            [EQ-MC-MEAN-001], [EQ-MC-SE-001], [REF-ROBERT-CASELLA-2004].

        """
        estimate = require_finite_real(self.estimate, name="estimate")
        variance = require_non_negative_finite_real(
            self.sample_variance, name="sample_variance"
        )
        standard_error = require_non_negative_finite_real(
            self.standard_error, name="standard_error"
        )
        sample_size = require_non_negative_integer(
            self.sample_size, name="sample_size"
        )
        evaluations = require_non_negative_integer(
            self.evaluations, name="evaluations"
        )
        if sample_size < 2:
            raise ValueError("sample_size must be at least two")
        if evaluations < sample_size:
            raise ValueError("evaluations must not be below sample_size")
        if not isinstance(self.confidence_interval, ConfidenceInterval):
            raise TypeError("confidence_interval must be a ConfidenceInterval")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "sample_variance", variance)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "evaluations", evaluations)


@dataclass(frozen=True, slots=True)
class ProbabilityEstimate:
    """A Bernoulli-event probability estimate with Wilson uncertainty.

    Args:
        probability: Empirical success proportion.
        standard_error: Plug-in Bernoulli standard error.
        confidence_interval: Wilson score interval.
        successes: Number of observed events.
        trials: Positive total sample count.
        evaluations: Number of event/model evaluations charged.

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3], [REF-WILSON-1927].

    """

    probability: float
    standard_error: float
    confidence_interval: ConfidenceInterval
    successes: int
    trials: int
    evaluations: int

    def __post_init__(self) -> None:
        """Validate the probability estimate and count invariants.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If probabilities, errors, or counts are invalid.

        References:
            [EQ-MC-PROB-001], [REF-WILSON-1927].

        """
        probability = require_probability(self.probability, name="probability")
        standard_error = require_non_negative_finite_real(
            self.standard_error, name="standard_error"
        )
        successes = require_non_negative_integer(self.successes, name="successes")
        trials = require_non_negative_integer(self.trials, name="trials")
        evaluations = require_non_negative_integer(
            self.evaluations, name="evaluations"
        )
        if trials == 0:
            raise ValueError("trials must be positive")
        if successes > trials:
            raise ValueError("successes must not exceed trials")
        if evaluations < trials:
            raise ValueError("evaluations must not be below trials")
        if not isinstance(self.confidence_interval, ConfidenceInterval):
            raise TypeError("confidence_interval must be a ConfidenceInterval")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "evaluations", evaluations)


def estimate_mean(
    values: Iterable[object], *, confidence: float = 0.95
) -> MonteCarloEstimate:
    """Estimate an expectation by a sample mean.

    Args:
        values: At least two finite sampled contributions.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Mean, unbiased sample variance, standard error, interval, and cost.

    Raises:
        TypeError: If observations or confidence have invalid types.
        ValueError: If the sample, variation, or confidence is invalid.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-VAR-001], [EQ-MC-SE-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4].

    """
    sample = coerce_finite_sample(values)
    if len(sample) < 2:
        raise ValueError("Monte Carlo estimation requires at least two values")
    variance = statistics.variance(sample)
    if variance == 0.0:
        center = math.fsum(sample) / len(sample)
        checked_confidence = require_probability(confidence, name="confidence")
        if checked_confidence in (0.0, 1.0):
            raise ValueError("confidence must be strictly between zero and one")
        interval = ConfidenceInterval(
            center,
            center,
            checked_confidence,
            "degenerate Monte Carlo mean",
        )
    else:
        interval = normal_mean_confidence_interval(
            sample,
            confidence=confidence,
        )
    estimate = math.fsum(sample) / len(sample)
    standard_error = math.sqrt(variance / len(sample))
    return MonteCarloEstimate(
        estimate,
        variance,
        standard_error,
        interval,
        len(sample),
        len(sample),
    )


def estimate_probability(
    events: Iterable[bool], *, confidence: float = 0.95
) -> ProbabilityEstimate:
    """Estimate an event probability from Bernoulli indicators.

    Args:
        events: Non-empty iterable containing exact Boolean indicators.
        confidence: Nominal Wilson-interval confidence in ``(0, 1)``.

    Returns:
        Empirical probability, plug-in standard error, Wilson interval, and cost.

    Raises:
        TypeError: If an item is not exactly Boolean.
        ValueError: If the iterable is empty or confidence is invalid.

    Equation:
        [EQ-MC-PROB-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3], [REF-WILSON-1927].

    """
    materialized: list[bool] = []
    for index, event in enumerate(events):
        if not isinstance(event, bool):
            raise TypeError(f"events[{index}] must be Boolean")
        materialized.append(event)
    if not materialized:
        raise ValueError("events must contain at least one indicator")
    trials = len(materialized)
    successes = sum(materialized)
    probability = successes / trials
    standard_error = math.sqrt(probability * (1.0 - probability) / trials)
    return ProbabilityEstimate(
        probability,
        standard_error,
        wilson_score_interval(successes, trials, confidence=confidence),
        successes,
        trials,
        trials,
    )


def estimate_uniform_integral(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    sample_size: int,
    seed: int | None = None,
    confidence: float = 0.95,
) -> MonteCarloEstimate:
    """Estimate a finite-interval integral using uniform Monte Carlo.

    Args:
        function: Scalar integrand returning finite real values.
        lower: Finite lower integration bound.
        upper: Finite upper bound greater than ``lower``.
        sample_size: Number of independent uniform evaluations, at least two.
        seed: Optional seed for a call-local pseudo-random generator.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Integral estimate with contribution variance, standard error, and cost.

    Raises:
        TypeError: If arguments or integrand results have invalid types.
        ValueError: If bounds, sample size, values, or confidence are invalid.

    Equation:
        [EQ-MC-INT-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    checked_lower = require_finite_real(lower, name="lower")
    checked_upper = require_finite_real(upper, name="upper")
    checked_size = require_non_negative_integer(sample_size, name="sample_size")
    if checked_upper <= checked_lower:
        raise ValueError("upper must be greater than lower")
    if checked_size < 2:
        raise ValueError("sample_size must be at least two")
    width = checked_upper - checked_lower
    generator = random.Random(seed)
    contributions = tuple(
        width
        * require_finite_real(
            function(generator.uniform(checked_lower, checked_upper)),
            name=f"function(sample {index})",
        )
        for index in range(checked_size)
    )
    return estimate_mean(contributions, confidence=confidence)


def convergence_trace(
    values: Iterable[object],
    checkpoints: Iterable[int],
    *,
    confidence: float = 0.95,
) -> tuple[MonteCarloEstimate, ...]:
    """Compute nested-prefix Monte Carlo estimates at explicit checkpoints.

    Args:
        values: Finite sampled contributions materialized once.
        checkpoints: Strictly increasing sample sizes, each at least two and no
            greater than the available number of values.
        confidence: Nominal normal-approximation confidence in ``(0, 1)``.

    Returns:
        Estimate sequence whose work counts equal the requested checkpoints.

    Raises:
        TypeError: If values or checkpoints have invalid types.
        ValueError: If checkpoints are empty, unordered, or outside bounds.

    Equation:
        [EQ-MC-ROOTN-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4].

    """
    sample = coerce_finite_sample(values)
    checked: list[int] = []
    for index, checkpoint in enumerate(checkpoints):
        value = require_non_negative_integer(
            checkpoint, name=f"checkpoints[{index}]"
        )
        if value < 2:
            raise ValueError("each checkpoint must be at least two")
        if checked and value <= checked[-1]:
            raise ValueError("checkpoints must be strictly increasing")
        if value > len(sample):
            raise ValueError("checkpoint exceeds available sample size")
        checked.append(value)
    if not checked:
        raise ValueError("checkpoints must not be empty")
    return tuple(
        estimate_mean(sample[:checkpoint], confidence=confidence)
        for checkpoint in checked
    )
