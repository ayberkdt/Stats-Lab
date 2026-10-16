"""Forward uncertainty propagation over explicit parameter samples."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from probstat.foundations.validation import require_finite_real
from probstat.monte_carlo import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    estimate_mean,
    estimate_probability,
)
from probstat.numerics import (
    QuadratureResult,
    composite_simpson,
    composite_trapezoid,
)
from probstat.uq import (
    FailedEvaluation,
    FailureLog,
    FailurePolicy,
    ForwardModel,
    FrozenMapping,
    QuantityOfInterest,
    evaluate_model,
)
from probstat.uq.evaluation import require_failure_policy


@dataclass(frozen=True, slots=True)
class OutputSummary:
    """Monte Carlo summary for one propagated quantity of interest.

    Args:
        mean_estimate: Mean, sample variance, standard error, interval, and cost.
        threshold: Optional finite exceedance threshold.
        exceedance: Optional estimate of ``P(Y > threshold)``.

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4],
        [REF-ROBERT-CASELLA-2004, Chapter 3].

    """

    mean_estimate: MonteCarloEstimate
    threshold: float | None = None
    exceedance: ProbabilityEstimate | None = None

    def __post_init__(self) -> None:
        """Validate summary objects and threshold/exceedance pairing.

        Raises:
            TypeError: If estimates or threshold have invalid types.
            ValueError: If threshold and exceedance are not supplied together.

        References:
            [EQ-PROP-THRESH-001], [REF-SMITH-UQ-2024, Chapter 4].

        """
        if not isinstance(self.mean_estimate, MonteCarloEstimate):
            raise TypeError("mean_estimate must be a MonteCarloEstimate")
        if (self.threshold is None) != (self.exceedance is None):
            raise ValueError("threshold and exceedance must be supplied together")
        if self.threshold is not None:
            threshold = require_finite_real(self.threshold, name="threshold")
            if not isinstance(self.exceedance, ProbabilityEstimate):
                raise TypeError("exceedance must be a ProbabilityEstimate")
            object.__setattr__(self, "threshold", threshold)

    @property
    def mean(self) -> float:
        """Return the propagated sample mean.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.mean_estimate.estimate

    @property
    def variance(self) -> float:
        """Return the unbiased propagated sample variance.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.mean_estimate.sample_variance

    @property
    def standard_deviation(self) -> float:
        """Return square root of the propagated sample variance.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return math.sqrt(self.variance)


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Immutable raw outputs, summaries, and forward-model evaluation cost.

    Args:
        outputs: Read-only QoI-name mapping to ordered propagated observations
            of the successful runs.
        summaries: Read-only QoI-name mapping to statistical summaries.
        model_evaluations: Number of complete forward-model calls.
        failures: Failed runs dropped under the ``"skip"``/``"record"``
            policies (indices refer to the input samples).

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4].

    """

    outputs: Mapping[str, tuple[float, ...]]
    summaries: Mapping[str, OutputSummary]
    model_evaluations: int
    failures: tuple[FailedEvaluation, ...] = field(default=(), repr=False)


def propagate_samples(
    model: ForwardModel,
    parameter_samples: Iterable[Mapping[str, float]],
    quantities: Iterable[QuantityOfInterest],
    *,
    thresholds: Mapping[str, float] | None = None,
    confidence: float = 0.95,
    on_failure: FailurePolicy = "raise",
) -> PropagationResult:
    """Propagate explicit input samples through a named forward model.

    Under ``on_failure="skip"`` or ``"record"`` failed runs are dropped and
    listed in the result; the summaries then describe the successful runs
    only and are biased unless failures are unrelated to the output.

    Args:
        model: Deterministic named-input/named-output forward map.
        parameter_samples: At least two finite named parameter realizations.
        quantities: Non-empty distinctly named quantities of interest.
        thresholds: Optional QoI-name mapping for strict exceedance ``Y>S``.
        confidence: Nominal confidence for mean and Wilson intervals.
        on_failure: ``"raise"`` (default), ``"skip"``, or ``"record"``.

    Returns:
        Immutable raw outputs, summaries, threshold probabilities, and cost.

    Raises:
        TypeError: If models, samples, quantities, thresholds, or values are invalid.
        ValueError: If collections are empty, samples are insufficient, names
            disagree, numerical domains are invalid, or fewer than two runs
            succeed.

    Equation:
        [EQ-UQ-001], [EQ-MC-MEAN-001], [EQ-MC-VAR-001],
        [EQ-PROP-THRESH-001].

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4],
        [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    requested = tuple(quantities)
    if not requested:
        raise ValueError("quantities must not be empty")
    samples = tuple(parameter_samples)
    if len(samples) < 2:
        raise ValueError("propagation requires at least two parameter samples")
    names = tuple(quantity.name for quantity in requested)
    if len(set(names)) != len(names):
        raise ValueError("quantity names must be distinct")
    checked_thresholds: dict[str, float] = {}
    if thresholds is not None:
        if not isinstance(thresholds, Mapping):
            raise TypeError("thresholds must be a mapping")
        unknown = set(thresholds) - set(names)
        if unknown:
            raise ValueError(
                f"threshold names are not requested QoIs: {sorted(unknown)}"
            )
        checked_thresholds = {
            name: require_finite_real(value, name=f"thresholds[{name!r}]")
            for name, value in thresholds.items()
        }

    policy = require_failure_policy(on_failure)
    log = FailureLog(max_records=max(1, len(samples)))
    failures: list[FailedEvaluation] = []
    collected: dict[str, list[float]] = {name: [] for name in names}
    for index, parameters in enumerate(samples):
        before = log.count
        outputs = evaluate_model(
            model, parameters, requested, on_failure=policy, failures=log
        )
        if outputs is None:
            if log.count > before:
                last = log.records[-1]
                failures.append(
                    FailedEvaluation(index, last.inputs, last.error_type, last.message)
                )
            continue
        for name in names:
            collected[name].append(outputs[name])
    completed = len(samples) - log.count
    if completed < 2:
        raise ValueError(
            f"only {completed} of {len(samples)} model runs succeeded; "
            "propagation needs at least two"
        )

    immutable_outputs = {
        name: tuple(values) for name, values in collected.items()
    }
    summaries: dict[str, OutputSummary] = {}
    for name, values in immutable_outputs.items():
        mean_estimate = estimate_mean(values, confidence=confidence)
        if name in checked_thresholds:
            threshold = checked_thresholds[name]
            exceedance = estimate_probability(
                (value > threshold for value in values),
                confidence=confidence,
            )
            summaries[name] = OutputSummary(
                mean_estimate,
                threshold,
                exceedance,
            )
        else:
            summaries[name] = OutputSummary(mean_estimate)
    return PropagationResult(
        FrozenMapping(immutable_outputs),
        FrozenMapping(summaries),
        completed,
        tuple(failures),
    )


def expectation_by_quadrature(
    transform: Callable[[float], float],
    density: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    intervals: int,
    method: str = "simpson",
) -> QuadratureResult:
    """Approximate a scalar expectation integral over finite bounds.

    The caller owns any probability mass outside ``[lower, upper]``. For an
    unbounded law this is a truncated expectation unless tail corrections are
    supplied separately.

    Args:
        transform: Scalar function ``g(x)`` whose expectation is requested.
        density: Non-negative finite probability density ``f_X(x)`` at nodes.
        lower: Finite lower integration bound.
        upper: Finite upper integration bound greater than ``lower``.
        intervals: Positive subinterval count; must be even for Simpson.
        method: Either ``"simpson"`` or ``"trapezoid"``.

    Returns:
        Quadrature result for the finite-bound expectation integral.

    Raises:
        TypeError: If callables or evaluated values have invalid types.
        ValueError: If bounds, intervals, density values, or method are invalid.

    Equation:
        [EQ-PROP-EXP-001], [EQ-NUM-SIMP-001], [EQ-NUM-TRAP-001].

    References:
        [REF-SMITH-UQ-2024, Chapter 3],
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2].

    """
    if not callable(transform):
        raise TypeError("transform must be callable")
    if not callable(density):
        raise TypeError("density must be callable")

    def weighted(point: float) -> float:
        """Return one validated expectation-integrand value.

        References:
            [EQ-PROP-EXP-001], [REF-SMITH-UQ-2024, Chapter 3].

        """
        density_value = require_finite_real(density(point), name="density(x)")
        if density_value < 0.0:
            raise ValueError("density(x) must be non-negative")
        transform_value = require_finite_real(transform(point), name="transform(x)")
        return transform_value * density_value

    if method == "simpson":
        return composite_simpson(
            weighted,
            lower,
            upper,
            intervals=intervals,
        )
    if method == "trapezoid":
        return composite_trapezoid(
            weighted,
            lower,
            upper,
            intervals=intervals,
        )
    raise ValueError("method must be 'simpson' or 'trapezoid'")
