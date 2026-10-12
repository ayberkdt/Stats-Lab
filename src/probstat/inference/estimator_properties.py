"""Finite-sample properties of point estimators: bias, risk, efficiency.

An *estimand* is the population quantity ``theta`` (or ``tau(theta)``); an
*estimator* is a function ``T(X_1, ..., X_n)`` of the sample. Under squared
error loss the risk is ``MSE(T) = E[(T - theta)^2] = Bias(T)^2 + Var(T)``.

This module provides

* :func:`evaluate_estimator`, a Monte Carlo harness reporting bias,
  variance, MSE, relative efficiency against a reference estimator, and
  interval coverage, each with its Monte Carlo standard error;
* Cramer-Rao lower bounds built on the closed-form per-observation Fisher
  information of :mod:`probstat.information.geometry`, and exact MLE
  efficiency for common families;
* classical worked examples with exact risks: the normal-variance divisor
  family (``n - 1``, ``n``, and the minimum-MSE ``n + 1``), the
  ``Uniform(0, theta)`` maximum and its UMVUE, and Rao-Blackwellization of
  the Poisson zero-probability estimator;
* a numerical factorization check for sufficiency and an exact consistency
  profile.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations import RandomStream, require_finite_real, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.information import (
    bernoulli_fisher_information,
    exponential_fisher_information,
    normal_fisher_information,
    poisson_fisher_information,
)
from probstat.linalg import Matrix, inverse_spd

Sample = Sequence[float]
Estimator = Callable[[Sample], float]
SampleGenerator = Callable[[int, random.Random], Sample]
IntervalProcedure = Callable[[Sample], tuple[float, float]]
MLEFamily = Literal[
    "bernoulli", "poisson", "normal_mean", "normal_variance", "exponential_rate"
]


def _positive(value: object, *, name: str) -> float:
    """Validate a strictly positive finite real.

    Args:
        value: Candidate value.
        name: Parameter name.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _count(value: object, *, name: str, minimum: int = 1) -> int:
    """Validate an integer that is at least ``minimum``.

    Args:
        value: Candidate value.
        name: Parameter name.
        minimum: Smallest admissible value.

    Returns:
        The integer.

    Raises:
        ValueError: If the value is below ``minimum``.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_non_negative_integer(value, name=name)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean by compensated summation.

    References:
        [REF-CASELLA-BERGER-2002, Section 5.2].

    """
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    """Return the unbiased sample variance.

    References:
        [REF-CASELLA-BERGER-2002, Section 5.2].

    """
    center = _mean(values)
    return math.fsum((v - center) ** 2 for v in values) / (len(values) - 1)


@dataclass(frozen=True, slots=True)
class EstimatorEvaluation:
    """Monte Carlo summary of an estimator's sampling distribution.

    Every quantity is accompanied by its Monte Carlo standard error, the
    standard deviation of the reported value over repeated evaluations
    with independent seeds.

    Args:
        n: Sample size of each simulated data set.
        replications: Number of simulated data sets ``R``.
        true_value: Estimand ``theta``.
        mean_estimate: Average of the ``R`` estimates.
        bias: ``mean_estimate - theta``.
        bias_se: ``s / sqrt(R)``.
        variance: Unbiased sample variance ``s^2`` of the estimates.
        variance_se: ``sqrt((m4 - s^4 (R - 3)/(R - 1)) / R)``.
        mse: Average of ``(estimate - theta)^2``.
        mse_se: Standard deviation of squared errors over ``sqrt(R)``.
        relative_efficiency: ``MSE(reference) / MSE(estimator)`` on the same
            data sets, or ``None``; values above one favour the estimator.
        relative_efficiency_se: Delta-method standard error of the ratio.
        coverage: Fraction of intervals containing ``theta``, or ``None``.
        coverage_se: ``sqrt(c (1 - c) / R)``.
        mean_interval_length: Average interval length, or ``None``.

    References:
        [EQ-EST-001], [EQ-EST-002], [EQ-EST-003].

    """

    n: int
    replications: int
    true_value: float
    mean_estimate: float
    bias: float
    bias_se: float
    variance: float
    variance_se: float
    mse: float
    mse_se: float
    relative_efficiency: float | None
    relative_efficiency_se: float | None
    coverage: float | None
    coverage_se: float | None
    mean_interval_length: float | None


def evaluate_estimator(
    estimator: Estimator,
    sampler: SampleGenerator,
    true_value: float,
    n: int,
    replications: int,
    seed: int | RandomStream | None = None,
    *,
    interval: IntervalProcedure | None = None,
    reference: Estimator | None = None,
) -> EstimatorEvaluation:
    """Simulate an estimator's bias, variance, MSE, efficiency, and coverage.

    Data set ``r`` is drawn by ``sampler(n, rng)`` with
    ``rng = seed.child("replication", r).generator()``, so the estimator,
    the reference estimator, and the interval procedure all see the same
    data sets (a paired design), and results do not depend on the order of
    evaluation.

    Args:
        estimator: Point estimator ``T(sample) -> float``.
        sampler: Data generator ``sampler(n, rng) -> sample``.
        true_value: Finite estimand ``theta``.
        n: Positive sample size.
        replications: Number of data sets ``R >= 4``.
        seed: Integer seed, stream, or ``None`` (not reproducible).
        interval: Optional procedure returning ``(lower, upper)``.
        reference: Optional competing estimator for relative efficiency.

    Returns:
        Evaluation summary with Monte Carlo standard errors.

    Raises:
        ValueError: If an estimate or interval is not finite, or the
            reference has zero mean squared error on the simulated data.

    Equation:
        [EQ-EST-001], [EQ-EST-002], [EQ-EST-003].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3.1], [REF-ROBERT-CASELLA-2004,
        Chapter 3], [REF-LEHMANN-CASELLA-1998, Chapter 1].

    """
    theta = require_finite_real(true_value, name="true_value")
    size = _count(n, name="n")
    count = _count(replications, name="replications", minimum=4)
    stream = resolve_stream(seed)
    estimates: list[float] = []
    reference_errors: list[float] = []
    covered = 0
    lengths: list[float] = []
    for replication in range(count):
        rng = stream.child("replication", replication).generator()
        sample = sampler(size, rng)
        value = float(estimator(sample))
        if not math.isfinite(value):
            raise ValueError("the estimator returned a non-finite value")
        estimates.append(value)
        if reference is not None:
            other = float(reference(sample))
            if not math.isfinite(other):
                raise ValueError("the reference estimator returned a non-finite value")
            reference_errors.append((other - theta) ** 2)
        if interval is not None:
            low, high = (float(bound) for bound in interval(sample))
            if not (math.isfinite(low) and math.isfinite(high)) or high < low:
                raise ValueError("interval bounds must be finite and ordered")
            covered += int(low <= theta <= high)
            lengths.append(high - low)
    center = _mean(estimates)
    variance = _sample_variance(estimates)
    m4 = math.fsum((v - center) ** 4 for v in estimates) / count
    variance_se = math.sqrt(
        max(m4 - variance * variance * (count - 3) / (count - 1), 0.0) / count
    )
    errors = [(v - theta) ** 2 for v in estimates]
    mse = _mean(errors)
    mse_se = math.sqrt(_sample_variance(errors) / count)
    efficiency: float | None = None
    efficiency_se: float | None = None
    if reference is not None:
        reference_mse = _mean(reference_errors)
        if mse == 0.0:
            raise ValueError("the estimator has zero simulated mean squared error")
        ratio = reference_mse / mse
        var_a = _sample_variance(reference_errors)
        var_b = _sample_variance(errors)
        cov_ab = math.fsum(
            (a - reference_mse) * (b - mse)
            for a, b in zip(reference_errors, errors, strict=True)
        ) / (count - 1)
        spread = var_a - 2.0 * ratio * cov_ab + ratio * ratio * var_b
        efficiency = ratio
        efficiency_se = math.sqrt(max(spread, 0.0) / count) / mse
    coverage: float | None = None
    coverage_se: float | None = None
    mean_length: float | None = None
    if interval is not None:
        coverage = covered / count
        coverage_se = math.sqrt(coverage * (1.0 - coverage) / count)
        mean_length = _mean(lengths)
    return EstimatorEvaluation(
        size,
        count,
        theta,
        center,
        center - theta,
        math.sqrt(variance / count),
        variance,
        variance_se,
        mse,
        mse_se,
        efficiency,
        efficiency_se,
        coverage,
        coverage_se,
        mean_length,
    )


def cramer_rao_bound(
    information: Matrix, n: int, gradient: Sequence[float] | None = None
) -> float:
    """Return the Cramer-Rao bound ``g' I(theta)^-1 g / n``.

    ``information`` is the per-observation Fisher information. For an
    unbiased estimator of ``tau(theta)``, ``g = grad tau``. For an estimator
    with expectation ``psi(theta)`` (bias ``psi - tau``), ``g = grad psi``
    gives the biased-estimator bound on its variance. ``gradient`` defaults
    to ``(1,)`` for a scalar parameter.

    Args:
        information: Symmetric positive-definite ``k x k`` information.
        n: Positive number of i.i.d. observations.
        gradient: Length-``k`` gradient ``g``.

    Returns:
        Lower bound on the variance.

    Raises:
        ValueError: If dimensions disagree or the gradient is missing for a
            vector parameter.

    Equation:
        [EQ-EST-004].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3.2], [REF-LEHMANN-CASELLA-1998,
        Sections 2.5-2.6].

    """
    size = _count(n, name="n")
    dimension = len(information)
    if gradient is None:
        if dimension != 1:
            raise ValueError("a gradient is required for a vector parameter")
        gradient = (1.0,)
    g = tuple(require_finite_real(v, name="gradient") for v in gradient)
    if len(g) != dimension:
        raise ValueError("gradient length must match the information matrix")
    inverse = inverse_spd(information)
    quadratic = math.fsum(
        g[i] * inverse[i][j] * g[j] for i in range(dimension) for j in range(dimension)
    )
    return quadratic / size


@dataclass(frozen=True, slots=True)
class MLEEfficiency:
    """Exact finite-sample risk of an MLE next to the Cramer-Rao bounds.

    Args:
        family: Model and estimand label.
        n: Sample size.
        true_value: Estimand value.
        bias: Exact bias of the MLE.
        variance: Exact variance of the MLE.
        crlb: Bound for unbiased estimators of the estimand.
        biased_crlb: Bound for estimators sharing the MLE's bias function.

    References:
        [EQ-EST-004], [EQ-EST-010].

    """

    family: str
    n: int
    true_value: float
    bias: float
    variance: float
    crlb: float
    biased_crlb: float

    @property
    def mse(self) -> float:
        """Return ``bias^2 + variance``.

        References:
            [EQ-EST-001].

        """
        return self.bias * self.bias + self.variance

    @property
    def efficiency(self) -> float:
        """Return ``crlb / mse``; it may exceed one for a biased estimator.

        References:
            [EQ-EST-004].

        """
        return self.crlb / self.mse


def mle_efficiency(family: MLEFamily, n: int, value: float) -> MLEEfficiency:
    """Return the exact MLE risk and Cramer-Rao bounds for a common family.

    * ``bernoulli`` (``value = p``): MLE ``Xbar``, unbiased, variance
      ``p (1 - p) / n``.
    * ``poisson`` (``value = lambda``): MLE ``Xbar``, unbiased, variance
      ``lambda / n``.
    * ``normal_mean`` (``value = sigma``, known): MLE ``Xbar``, unbiased,
      variance ``sigma^2 / n``; ``true_value`` is reported as ``nan``.
    * ``normal_variance`` (``value = sigma``): MLE ``SS / n``, bias
      ``-sigma^2 / n``, variance ``2 (n - 1) sigma^4 / n^2``.
    * ``exponential_rate`` (``value = lambda``): MLE ``1 / Xbar``, bias
      ``lambda / (n - 1)``, variance ``n^2 lambda^2 / ((n - 1)^2 (n - 2))``.

    The information comes from :mod:`probstat.information.geometry`
    (``normal_fisher_information`` is in ``(mu, sigma)``, so the variance
    bound uses the gradient ``(0, 2 sigma)`` of ``sigma^2``).

    Args:
        family: Family and estimand.
        n: Sample size (``n >= 3`` for ``exponential_rate``, ``n >= 2`` for
            ``normal_variance``).
        value: Parameter value as tabulated above.

    Returns:
        Exact efficiency record.

    Raises:
        ValueError: If the family is unknown or ``n`` is too small.

    Equation:
        [EQ-EST-004], [EQ-EST-010].

    References:
        [REF-CASELLA-BERGER-2002, Sections 7.2.2 and 7.3.2],
        [REF-LEHMANN-CASELLA-1998, Sections 2.5-2.6].

    """
    size = _count(n, name="n")
    if family == "bernoulli":
        p = require_finite_real(value, name="value")
        info = bernoulli_fisher_information(p)
        bound = cramer_rao_bound(info, size)
        return MLEEfficiency(family, size, p, 0.0, p * (1.0 - p) / size, bound, bound)
    if family == "poisson":
        lam = _positive(value, name="value")
        bound = cramer_rao_bound(poisson_fisher_information(lam), size)
        return MLEEfficiency(family, size, lam, 0.0, lam / size, bound, bound)
    if family == "normal_mean":
        sigma = _positive(value, name="value")
        info = normal_fisher_information(sigma)
        bound = cramer_rao_bound(info, size, (1.0, 0.0))
        return MLEEfficiency(family, size, math.nan, 0.0, sigma**2 / size, bound, bound)
    if family == "normal_variance":
        size = _count(n, name="n", minimum=2)
        sigma = _positive(value, name="value")
        info = normal_fisher_information(sigma)
        bound = cramer_rao_bound(info, size, (0.0, 2.0 * sigma))
        shrink = (size - 1) / size
        return MLEEfficiency(
            family,
            size,
            sigma * sigma,
            -(sigma**2) / size,
            2.0 * (size - 1) * sigma**4 / size**2,
            bound,
            cramer_rao_bound(info, size, (0.0, 2.0 * sigma * shrink)),
        )
    if family == "exponential_rate":
        size = _count(n, name="n", minimum=3)
        lam = _positive(value, name="value")
        info = exponential_fisher_information(lam)
        bound = cramer_rao_bound(info, size)
        return MLEEfficiency(
            family,
            size,
            lam,
            lam / (size - 1),
            (size * lam) ** 2 / ((size - 1) ** 2 * (size - 2)),
            bound,
            cramer_rao_bound(info, size, (size / (size - 1),)),
        )
    raise ValueError(f"unknown family {family!r}")


@dataclass(frozen=True, slots=True)
class EstimatorRisk:
    """Exact bias and variance of one estimator.

    Args:
        label: Short estimator description.
        bias: Exact bias.
        variance: Exact variance.

    References:
        [EQ-EST-001].

    """

    label: str
    bias: float
    variance: float

    @property
    def mse(self) -> float:
        """Return ``bias^2 + variance``.

        References:
            [EQ-EST-001].

        """
        return self.bias * self.bias + self.variance


def normal_variance_estimator_risk(
    standard_deviation: float, n: int, divisor: float
) -> EstimatorRisk:
    """Return the exact risk of ``SS / d`` for i.i.d. normal data.

    With ``SS = sum (X_i - Xbar)^2 ~ sigma^2 chi^2_{n-1}``:
    bias ``((n - 1)/d - 1) sigma^2``, variance ``2 (n - 1) sigma^4 / d^2``,
    and ``MSE = sigma^4 (2 (n - 1) + (n - 1 - d)^2) / d^2``, minimized at
    ``d = n + 1`` with value ``2 sigma^4 / (n + 1)``.

    Args:
        standard_deviation: Positive ``sigma``.
        n: Sample size ``n >= 2``.
        divisor: Positive divisor ``d``.

    Returns:
        Exact risk.

    Equation:
        [EQ-EST-005].

    References:
        [REF-CASELLA-BERGER-2002, Sections 5.3 and 7.3.1].

    """
    sigma2 = _positive(standard_deviation, name="standard_deviation") ** 2
    size = _count(n, name="n", minimum=2)
    d = _positive(divisor, name="divisor")
    return EstimatorRisk(
        f"SS/{d:g}",
        ((size - 1) / d - 1.0) * sigma2,
        2.0 * (size - 1) * sigma2 * sigma2 / (d * d),
    )


def minimum_mse_variance_divisor(n: int) -> int:
    """Return ``n + 1``, the divisor minimizing the normal-data MSE of ``SS/d``.

    Setting the derivative of ``(2 (n - 1) + (n - 1 - d)^2) / d^2`` to zero
    gives ``d = n + 1`` exactly.

    Args:
        n: Sample size ``n >= 2``.

    Returns:
        Minimum-MSE divisor.

    Equation:
        [EQ-EST-005].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3.1].

    """
    return _count(n, name="n", minimum=2) + 1


def uniform_maximum_risks(theta: float, n: int) -> tuple[EstimatorRisk, ...]:
    """Return exact risks of estimators of ``theta`` for ``Uniform(0, theta)``.

    With ``M = max X_i``, ``E M = n theta/(n+1)``, and
    ``Var M = n theta^2/((n+1)^2 (n+2))``:

    * MLE ``M``: bias ``-theta/(n+1)``, MSE ``2 theta^2/((n+1)(n+2))``;
    * UMVUE ``(n+1) M / n``: unbiased, variance ``theta^2/(n (n+2))``;
    * minimum-MSE multiple ``(n+2) M/(n+1)``: MSE ``theta^2/(n+1)^2``;
    * method of moments ``2 Xbar``: unbiased, variance ``theta^2/(3 n)``.

    Args:
        theta: Positive upper endpoint.
        n: Positive sample size.

    Returns:
        Risks in the order MLE, UMVUE, minimum-MSE multiple, moments.

    Equation:
        [EQ-EST-006].

    References:
        [REF-CASELLA-BERGER-2002, Sections 5.4 and 7.3.3],
        [REF-LEHMANN-CASELLA-1998, Chapter 2].

    """
    t = _positive(theta, name="theta")
    size = _count(n, name="n")
    t2 = t * t
    var_max = size * t2 / ((size + 1) ** 2 * (size + 2))
    c_umvue = (size + 1) / size
    c_min = (size + 2) / (size + 1)
    return (
        EstimatorRisk("max", -t / (size + 1), var_max),
        EstimatorRisk("(n+1)/n max", 0.0, c_umvue * c_umvue * var_max),
        EstimatorRisk(
            "(n+2)/(n+1) max", -t / (size + 1) ** 2, c_min * c_min * var_max
        ),
        EstimatorRisk("2 mean", 0.0, t2 / (3.0 * size)),
    )


def poisson_zero_indicator(sample: Sample) -> float:
    """Return the crude unbiased estimator ``1{X_1 = 0}`` of ``exp(-lambda)``.

    Args:
        sample: Non-empty Poisson counts.

    Returns:
        One if the first count is zero, else zero.

    Equation:
        [EQ-EST-007].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3.3].

    """
    if not sample:
        raise ValueError("sample must not be empty")
    return 1.0 if sample[0] == 0 else 0.0


def poisson_zero_rao_blackwell(sample: Sample) -> float:
    """Return ``E[1{X_1 = 0} | S] = ((n - 1)/n)^S`` with ``S = sum X_i``.

    Given ``S``, ``X_1 ~ Binomial(S, 1/n)``, so the conditional probability
    of ``X_1 = 0`` is ``(1 - 1/n)^S``.

    Args:
        sample: Non-empty Poisson counts.

    Returns:
        Rao-Blackwellized estimate (the UMVUE of ``exp(-lambda)``).

    Equation:
        [EQ-EST-007].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3.3], [REF-LEHMANN-CASELLA-1998,
        Chapter 2].

    """
    if not sample:
        raise ValueError("sample must not be empty")
    size = len(sample)
    return math.pow((size - 1) / size, math.fsum(sample))


@dataclass(frozen=True, slots=True)
class RaoBlackwellComparison:
    """Exact risks for estimating ``exp(-lambda)`` from Poisson data.

    Args:
        crude: Indicator ``1{X_1 = 0}``.
        rao_blackwellized: ``((n - 1)/n)^S``.
        mle: Plug-in ``exp(-Xbar)``.
        crlb: Cramer-Rao bound ``lambda exp(-2 lambda) / n``.

    References:
        [EQ-EST-007].

    """

    crude: EstimatorRisk
    rao_blackwellized: EstimatorRisk
    mle: EstimatorRisk
    crlb: float


def poisson_zero_probability_risks(lam: float, n: int) -> RaoBlackwellComparison:
    """Return exact risks of three estimators of ``P(X = 0) = exp(-lambda)``.

    With ``S ~ Poisson(n lambda)`` and ``E a^S = exp(n lambda (a - 1))``:

    * crude: unbiased, ``Var = e^{-lambda} (1 - e^{-lambda})``;
    * Rao-Blackwellized ``((n-1)/n)^S``: unbiased,
      ``Var = e^{-2 lambda} (e^{lambda/n} - 1)``;
    * plug-in ``e^{-Xbar} = (e^{-1/n})^S``: mean
      ``exp(n lambda (e^{-1/n} - 1))``, second moment
      ``exp(n lambda (e^{-2/n} - 1))``.

    Args:
        lam: Positive Poisson mean.
        n: Positive sample size.

    Returns:
        Comparison record.

    Equation:
        [EQ-EST-007], [EQ-EST-004].

    References:
        [REF-CASELLA-BERGER-2002, Sections 7.3.2-7.3.3],
        [REF-LEHMANN-CASELLA-1998, Chapter 2].

    """
    rate = _positive(lam, name="lam")
    size = _count(n, name="n")
    target = math.exp(-rate)
    crude = EstimatorRisk("1{X1=0}", 0.0, target * (1.0 - target))
    rb = EstimatorRisk(
        "((n-1)/n)^S", 0.0, math.exp(-2.0 * rate) * math.expm1(rate / size)
    )
    first = math.exp(size * rate * math.expm1(-1.0 / size))
    second = math.exp(size * rate * math.expm1(-2.0 / size))
    mle = EstimatorRisk("exp(-mean)", first - target, second - first * first)
    info = poisson_fisher_information(rate)
    bound = cramer_rao_bound(info, size, (-target,))
    return RaoBlackwellComparison(crude, rb, mle, bound)


@dataclass(frozen=True, slots=True)
class SufficiencyCheck:
    """Numerical factorization check for a candidate sufficient statistic.

    Args:
        statistic_a: ``T`` of the first sample.
        statistic_b: ``T`` of the second sample.
        same_statistic: Whether the statistics agree (relative ``1e-12``).
        log_ratio_spread: ``max - min`` over the parameter grid of
            ``log L(theta; a) - log L(theta; b)``.
        consistent: ``True`` when ``T`` agrees and the spread is within
            tolerance, ``False`` when ``T`` agrees but the ratio depends on
            ``theta`` (sufficiency refuted), ``None`` when ``T`` differs (no
            conclusion).

    References:
        [EQ-EST-008].

    """

    statistic_a: tuple[float, ...]
    statistic_b: tuple[float, ...]
    same_statistic: bool
    log_ratio_spread: float
    consistent: bool | None


def check_sufficiency(
    log_likelihood: Callable[[float, Sample], float],
    statistic: Callable[[Sample], float | Sequence[float]],
    sample_a: Sample,
    sample_b: Sample,
    parameters: Iterable[float],
    *,
    tolerance: float = 1e-9,
) -> SufficiencyCheck:
    """Check the factorization criterion on two samples with equal ``T``.

    If ``T`` is sufficient, ``L(theta; x) = g(T(x), theta) h(x)``, so for
    ``T(a) = T(b)`` the log-likelihood ratio ``log L(theta; a) -
    log L(theta; b) = log h(a) - log h(b)`` does not depend on ``theta``.
    A ``theta``-dependent ratio therefore *refutes* sufficiency; a constant
    ratio on a finite grid is necessary evidence, not a proof. For an
    exponential family ``f(x; theta) = h(x) exp(eta(theta)' T(x) -
    A(theta))``, ``sum T(x_i)`` is sufficient by construction.

    Args:
        log_likelihood: ``log_likelihood(theta, sample)``.
        statistic: Candidate ``T(sample)`` (scalar or tuple).
        sample_a: First sample.
        sample_b: Second sample.
        parameters: At least two parameter values.
        tolerance: Absolute tolerance on the spread.

    Returns:
        Check result.

    Raises:
        ValueError: If fewer than two parameter values are supplied.

    Equation:
        [EQ-EST-008].

    References:
        [REF-CASELLA-BERGER-2002, Section 6.2.1], [REF-LEHMANN-CASELLA-1998,
        Section 1.6].

    """

    def as_tuple(value: float | Sequence[float]) -> tuple[float, ...]:
        """Normalize a scalar or sequence statistic to a tuple.

        References:
            Project validation policy. No equation is evaluated.

        """
        if isinstance(value, int | float):
            return (float(value),)
        return tuple(float(v) for v in value)

    grid = tuple(require_finite_real(v, name="parameters") for v in parameters)
    if len(grid) < 2:
        raise ValueError("at least two parameter values are required")
    tol = require_finite_real(tolerance, name="tolerance")
    t_a = as_tuple(statistic(sample_a))
    t_b = as_tuple(statistic(sample_b))
    same = len(t_a) == len(t_b) and all(
        math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
        for x, y in zip(t_a, t_b, strict=True)
    )
    ratios = [
        float(log_likelihood(theta, sample_a)) - float(log_likelihood(theta, sample_b))
        for theta in grid
    ]
    spread = max(ratios) - min(ratios)
    consistent: bool | None = (spread <= tol) if same else None
    return SufficiencyCheck(t_a, t_b, same, spread, consistent)


@dataclass(frozen=True, slots=True)
class ConsistencyRow:
    """Exact risk of an estimator at one sample size.

    Args:
        n: Sample size.
        bias: Exact bias.
        variance: Exact variance.
        mse: ``bias^2 + variance``.
        tail_bound: ``min(1, MSE / eps^2) >= P(|T_n - theta| >= eps)``.

    References:
        [EQ-EST-009].

    """

    n: int
    bias: float
    variance: float
    mse: float
    tail_bound: float


def consistency_profile(
    risk: Callable[[int], EstimatorRisk], sizes: Iterable[int], epsilon: float
) -> tuple[ConsistencyRow, ...]:
    """Tabulate exact bias, variance, and MSE against ``n``.

    If bias and variance both tend to zero, ``MSE -> 0`` and Markov's
    inequality on ``(T_n - theta)^2`` gives ``P(|T_n - theta| >= eps) <=
    MSE / eps^2 -> 0``: the estimator is consistent.

    Args:
        risk: Exact risk ``risk(n)``, for example a ``functools.partial`` of
            :func:`normal_variance_estimator_risk`.
        sizes: Positive sample sizes.
        epsilon: Positive tolerance.

    Returns:
        One row per sample size.

    Equation:
        [EQ-EST-009].

    References:
        [REF-CASELLA-BERGER-2002, Section 10.1.1], [REF-WASSERMAN-2004,
        Chapter 6].

    """
    eps = _positive(epsilon, name="epsilon")
    rows: list[ConsistencyRow] = []
    for raw in sizes:
        size = _count(raw, name="n")
        r = risk(size)
        rows.append(
            ConsistencyRow(
                size, r.bias, r.variance, r.mse, min(1.0, r.mse / (eps * eps))
            )
        )
    return tuple(rows)
