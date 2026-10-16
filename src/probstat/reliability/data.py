"""Component reliability parameters from test and field data.

Following NUREG/CR-6823 (Atwood et al. 2003) practice, each estimator returns
both a *Bayesian* answer (a posterior law usable directly as an epistemic
:class:`~probstat.modeling.RandomInput`) and the *classical* confidence
bounds:

- **Failure on demand** (``x`` failures in ``n`` demands, binomial model):
  conjugate ``Beta(a + x, b + n - x)`` posterior from a ``Beta(a, b)`` prior
  (Jeffreys ``Beta(1/2, 1/2)``, uniform ``Beta(1, 1)``, or a user prior) and
  the Clopper-Pearson interval.
- **Failure rate** (``r`` failures in exposure ``T``, Poisson model):
  ``Gamma(alpha + r, beta + T)`` (shape, rate) from a ``Gamma(alpha, beta)``
  prior, with Jeffreys giving ``Gamma(r + 1/2, T)``, and chi-square bounds
  ``lambda_L = chi2_{alpha/2}(2r) / (2T)`` and
  ``lambda_U = chi2_{1-alpha/2}(2r + 2) / (2T)`` for a time-terminated test
  (``2r`` degrees of freedom in the upper bound for a failure-terminated
  test).
- **Zero-failure demonstration**: the success-run sample size
  ``n = ln(1 - C) / ln R`` and the exponential test time
  ``T = chi2_C(2) / (2 lambda) = -ln(1 - C) / lambda``, with the
  ``r``-allowed-failure generalisations.
- **Reliability growth**: the Crow-AMSAA (power-law NHPP) maximum-likelihood
  fit with the unbiased shape estimator, exact conditional confidence
  bounds on the shape, and the instantaneous MTBF.

Confidence levels default to 0.90 (two-sided 5 % / 95 % bounds), the
convention of NUREG/CR-6823.

Stability: this module is *experimental* (ADR-0063) inside a *provisional*
package; its names and signatures may change in any release.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.distributions import Beta, Binomial, ChiSquare, Gamma
from probstat.foundations import require_finite_real, require_probability
from probstat.foundations.validation import require_non_negative_integer
from probstat.inference.confidence import ConfidenceInterval, clopper_pearson_interval
from probstat.modeling import RandomInput

TERMINATIONS = ("time", "failure")
SIDES = ("two", "upper", "lower")


def _confidence(value: object) -> float:
    """Validate a confidence level strictly inside ``(0, 1)``.

    Args:
        value: Candidate level.

    Returns:
        The level.

    Raises:
        ValueError: If it is 0 or 1.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9]. No equation is evaluated.

    """
    level = require_probability(value, name="confidence")
    if level in (0.0, 1.0):
        raise ValueError("confidence must lie strictly between zero and one")
    return level


def _positive(value: object, *, name: str) -> float:
    """Validate a finite positive real.

    Args:
        value: Candidate value.
        name: Field name for error messages.

    Returns:
        The value.

    Raises:
        ValueError: If it is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _chi2(probability: float, degrees: float) -> float:
    """Return the chi-square quantile, ``0`` for zero degrees of freedom.

    Args:
        probability: Level in ``(0, 1)``.
        degrees: Non-negative degrees of freedom.

    Returns:
        ``chi2_p(nu)``.

    Equation:
        [EQ-DIST-CHI2-001].

    References:
        [REF-JKB-1994, Chapter 18].

    """
    if degrees == 0.0:
        return 0.0
    return float(ChiSquare(degrees).quantile(probability))


@dataclass(frozen=True, slots=True)
class DemandFailureEstimate:
    """Failure-on-demand probability from ``x`` failures in ``n`` demands.

    Args:
        failures: Observed failures ``x``.
        demands: Demands ``n``.
        prior_label: ``"jeffreys"``, ``"uniform"`` or ``"user"``.
        prior: Beta prior.
        posterior: Beta posterior ``Beta(a + x, b + n - x)``.
        maximum_likelihood: ``x / n``.
        credible_interval: Equal-tailed posterior interval.
        confidence_interval: Clopper-Pearson interval.
        confidence: Level of both intervals.

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """

    failures: int
    demands: int
    prior_label: str
    prior: Beta
    posterior: Beta
    maximum_likelihood: float
    credible_interval: tuple[float, float]
    confidence_interval: ConfidenceInterval
    confidence: float

    def random_input(self, name: str, *, description: str = "") -> RandomInput:
        """Return the posterior as an epistemic model input.

        Args:
            name: Input name.
            description: Optional meaning.

        Returns:
            Random input with the Beta posterior.

        References:
            [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

        """
        return RandomInput(name, self.posterior, description)


def demand_failure_probability(
    failures: int,
    demands: int,
    *,
    prior: str | Beta = "jeffreys",
    confidence: float = 0.90,
) -> DemandFailureEstimate:
    """Return the Beta posterior and Clopper-Pearson bounds of ``p``.

    Args:
        failures: Failures ``x`` (``0 <= x <= n``).
        demands: Positive number of demands ``n``.
        prior: ``"jeffreys"`` (``Beta(1/2, 1/2)``), ``"uniform"``
            (``Beta(1, 1)``) or a :class:`~probstat.distributions.Beta`.
        confidence: Two-sided level of both intervals.

    Returns:
        Bayesian and classical estimates.

    Raises:
        ValueError: If counts are inconsistent or the prior is unknown.

    Equation:
        [EQ-REL-DATA-001], [EQ-INF-CI-CP-001].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6],
        [REF-CLOPPER-PEARSON-1934].

    """
    x = require_non_negative_integer(failures, name="failures")
    n = require_non_negative_integer(demands, name="demands")
    if n == 0:
        raise ValueError("demands must be positive")
    if x > n:
        raise ValueError("failures must not exceed demands")
    level = _confidence(confidence)
    if isinstance(prior, Beta):
        law, label = prior, "user"
    elif prior == "jeffreys":
        law, label = Beta(0.5, 0.5), "jeffreys"
    elif prior == "uniform":
        law, label = Beta(1.0, 1.0), "uniform"
    else:
        raise ValueError("prior must be 'jeffreys', 'uniform', or a Beta law")
    posterior = Beta(law.alpha + x, law.beta + n - x)
    tail = 0.5 * (1.0 - level)
    return DemandFailureEstimate(
        x,
        n,
        label,
        law,
        posterior,
        x / n,
        (float(posterior.quantile(tail)), float(posterior.quantile(1.0 - tail))),
        clopper_pearson_interval(x, n, confidence=level),
        level,
    )


@dataclass(frozen=True, slots=True)
class FailureRateBounds:
    """Classical chi-square confidence bounds on a constant failure rate.

    Args:
        lower: Lower bound (``0`` when one-sided upper or ``r = 0``).
        upper: Upper bound (``inf`` when one-sided lower).
        confidence: Confidence level.
        termination: ``"time"`` or ``"failure"``.
        sided: ``"two"``, ``"upper"`` or ``"lower"``.

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6],
        [REF-RAUSAND-BARROS-HOYLAND-2021].

    """

    lower: float
    upper: float
    confidence: float
    termination: str
    sided: str


def failure_rate_bounds(
    failures: int,
    exposure: float,
    *,
    confidence: float = 0.90,
    termination: str = "time",
    sided: str = "two",
) -> FailureRateBounds:
    """Return chi-square confidence bounds on ``lambda`` from ``r`` failures in ``T``.

    Two-sided at level ``C = 1 - alpha``:

    - lower: ``chi2_{alpha/2}(2r) / (2T)`` (``0`` when ``r = 0``);
    - upper (time-terminated): ``chi2_{1-alpha/2}(2r + 2) / (2T)``;
    - upper (failure-terminated): ``chi2_{1-alpha/2}(2r) / (2T)``.

    One-sided bounds use ``alpha`` in place of ``alpha / 2``.

    Args:
        failures: Failures ``r`` (at least 1 if failure-terminated).
        exposure: Positive total exposure ``T`` (unit-hours, cycles, ...).
        confidence: Confidence level.
        termination: ``"time"`` (test stopped at ``T``) or ``"failure"``
            (stopped at the ``r``-th failure).
        sided: ``"two"``, ``"upper"`` (``[0, U]``) or ``"lower"``
            (``[L, inf)``).

    Returns:
        Confidence bounds.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-REL-DATA-003].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6],
        [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    r = require_non_negative_integer(failures, name="failures")
    t = _positive(exposure, name="exposure")
    level = _confidence(confidence)
    if termination not in TERMINATIONS:
        raise ValueError("termination must be 'time' or 'failure'")
    if sided not in SIDES:
        raise ValueError("sided must be 'two', 'upper', or 'lower'")
    if termination == "failure" and r == 0:
        raise ValueError("a failure-terminated test has at least one failure")
    alpha = 1.0 - level
    tail = 0.5 * alpha if sided == "two" else alpha
    upper_degrees = 2.0 * r + 2.0 if termination == "time" else 2.0 * r
    lower = 0.0 if sided == "upper" else _chi2(tail, 2.0 * r) / (2.0 * t)
    upper = (
        math.inf if sided == "lower" else _chi2(1.0 - tail, upper_degrees) / (2.0 * t)
    )
    return FailureRateBounds(lower, upper, level, termination, sided)


@dataclass(frozen=True, slots=True)
class FailureRateEstimate:
    """Constant failure rate from ``r`` failures in exposure ``T``.

    Args:
        failures: Failures ``r``.
        exposure: Exposure ``T``.
        prior_label: ``"jeffreys"`` or ``"user"``.
        posterior: Gamma posterior (shape, rate).
        maximum_likelihood: ``r / T``.
        credible_interval: Equal-tailed posterior interval.
        bounds: Classical chi-square bounds.
        confidence: Level of both intervals.

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """

    failures: int
    exposure: float
    prior_label: str
    posterior: Gamma
    maximum_likelihood: float
    credible_interval: tuple[float, float]
    bounds: FailureRateBounds
    confidence: float

    def random_input(self, name: str, *, description: str = "") -> RandomInput:
        """Return the posterior as an epistemic model input.

        Args:
            name: Input name.
            description: Optional meaning.

        Returns:
            Random input with the Gamma posterior.

        References:
            [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

        """
        return RandomInput(name, self.posterior, description)


def failure_rate_estimate(
    failures: int,
    exposure: float,
    *,
    prior: str | Gamma = "jeffreys",
    confidence: float = 0.90,
    termination: str = "time",
) -> FailureRateEstimate:
    """Return the Gamma posterior and chi-square bounds of a failure rate.

    Args:
        failures: Failures ``r``.
        exposure: Positive exposure ``T``.
        prior: ``"jeffreys"`` (improper ``Gamma(1/2, 0)``, posterior
            ``Gamma(r + 1/2, T)``) or a :class:`~probstat.distributions.Gamma`
            prior ``Gamma(alpha, beta)`` (shape, rate) giving
            ``Gamma(alpha + r, beta + T)``.
        confidence: Two-sided level of both intervals.
        termination: Test termination for the classical bounds.

    Returns:
        Bayesian and classical estimates.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-REL-DATA-002], [EQ-REL-DATA-003].

    References:
        [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """
    r = require_non_negative_integer(failures, name="failures")
    t = _positive(exposure, name="exposure")
    level = _confidence(confidence)
    if isinstance(prior, Gamma):
        posterior, label = Gamma(prior.shape + r, prior.rate + t), "user"
    elif prior == "jeffreys":
        posterior, label = Gamma(r + 0.5, t), "jeffreys"
    else:
        raise ValueError("prior must be 'jeffreys' or a Gamma law")
    tail = 0.5 * (1.0 - level)
    return FailureRateEstimate(
        r,
        t,
        label,
        posterior,
        r / t,
        (float(posterior.quantile(tail)), float(posterior.quantile(1.0 - tail))),
        failure_rate_bounds(r, t, confidence=level, termination=termination),
        level,
    )


def success_run_sample_size(
    reliability: float, confidence: float, *, allowed_failures: int = 0
) -> int:
    """Return the number of trials to demonstrate ``R`` at confidence ``C``.

    The smallest ``n`` with ``P(X <= r | n, 1 - R) <= 1 - C`` for
    ``X ~ Binomial(n, 1 - R)``. For ``r = 0`` this is the success-run
    theorem ``n = ceil(ln(1 - C) / ln R)``.

    Args:
        reliability: Reliability ``R`` to demonstrate, in ``(0, 1)``.
        confidence: Confidence ``C`` in ``(0, 1)``.
        allowed_failures: Failures ``r`` permitted in the test.

    Returns:
        Minimum number of trials.

    Equation:
        [EQ-REL-DATA-004].

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021], [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """
    target = _confidence(reliability)
    level = _confidence(confidence)
    r = require_non_negative_integer(allowed_failures, name="allowed_failures")
    alpha = 1.0 - level

    def passes(n: int) -> bool:
        """Return whether ``n`` trials with ``r`` failures demonstrate ``R``.

        References:
            [EQ-REL-DATA-004]. No new equation.

        """
        if r == 0:
            return n * math.log(target) <= math.log(alpha)
        return float(Binomial(n, 1.0 - target).cdf(r)) <= alpha

    if r == 0:
        n = max(1, math.ceil(math.log(alpha) / math.log(target)))
        while n > 1 and passes(n - 1):
            n -= 1
        while not passes(n):
            n += 1
        return n
    low, high = r, max(r + 1, 2 * r + 2)
    while not passes(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if passes(middle):
            high = middle
        else:
            low = middle
    return high


def demonstration_test_time(
    failure_rate: float, confidence: float, *, allowed_failures: int = 0
) -> float:
    """Return the total test time to demonstrate ``lambda <= lambda_0`` at ``C``.

    A time-terminated test with at most ``r`` failures demonstrates the rate
    when ``T = chi2_C(2r + 2) / (2 lambda_0)``; for ``r = 0`` this is
    ``-ln(1 - C) / lambda_0``.

    Args:
        failure_rate: Rate ``lambda_0`` to demonstrate (``1 / MTBF``).
        confidence: Confidence ``C``.
        allowed_failures: Failures ``r`` permitted.

    Returns:
        Total unit test time.

    Equation:
        [EQ-REL-DATA-004].

    References:
        [REF-RAUSAND-BARROS-HOYLAND-2021], [REF-ATWOOD-NUREG-6823-2003, Chapter 6].

    """
    rate = _positive(failure_rate, name="failure_rate")
    level = _confidence(confidence)
    r = require_non_negative_integer(allowed_failures, name="allowed_failures")
    if r == 0:
        return -math.log1p(-level) / rate
    return _chi2(level, 2.0 * r + 2.0) / (2.0 * rate)


@dataclass(frozen=True, slots=True)
class CrowAMSAAFit:
    """Power-law NHPP (Crow-AMSAA) reliability-growth fit.

    ``E[N(t)] = lambda t^beta``; ``beta < 1`` indicates growth (decreasing
    failure intensity).

    Args:
        failures: Number of failures ``n``.
        total_time: End of observation ``T`` (``t_n`` if failure-terminated).
        termination: ``"time"`` or ``"failure"``.
        beta: Maximum-likelihood shape.
        beta_unbiased: ``(n - 1)/n beta`` (time) or ``(n - 2)/n beta``
            (failure-terminated).
        lam: Scale ``n / T^beta`` (from the MLE shape).
        beta_interval: Exact conditional two-sided interval on ``beta``.
        confidence: Its level.

    References:
        [REF-CROW-1974], [REF-MIL-HDBK-189C-2011].

    """

    failures: int
    total_time: float
    termination: str
    beta: float
    beta_unbiased: float
    lam: float
    beta_interval: tuple[float, float]
    confidence: float

    def expected_failures(self, time: float) -> float:
        """Return ``lambda t^beta``.

        Args:
            time: Non-negative time.

        Equation:
            [EQ-REL-DATA-005].

        References:
            [REF-CROW-1974].

        """
        t = require_finite_real(time, name="time")
        if t < 0.0:
            raise ValueError("time must be non-negative")
        return self.lam * float(t**self.beta)

    def intensity(self, time: float) -> float:
        """Return the failure intensity ``lambda beta t^(beta - 1)``.

        Args:
            time: Positive time.

        Equation:
            [EQ-REL-DATA-005].

        References:
            [REF-CROW-1974].

        """
        t = _positive(time, name="time")
        return self.lam * self.beta * float(t ** (self.beta - 1.0))

    @property
    def instantaneous_mtbf(self) -> float:
        """Return ``1 / w(T)`` at the end of the test (current MTBF).

        Equation:
            [EQ-REL-DATA-005].

        References:
            [REF-CROW-1974], [REF-MIL-HDBK-189C-2011].

        """
        return 1.0 / self.intensity(self.total_time)

    @property
    def cumulative_mtbf(self) -> float:
        """Return ``T / n``.

        References:
            [REF-MIL-HDBK-189C-2011].

        """
        return self.total_time / self.failures


def crow_amsaa_fit(
    failure_times: Iterable[float],
    *,
    end_time: float | None = None,
    confidence: float = 0.90,
) -> CrowAMSAAFit:
    """Fit the Crow-AMSAA model to the cumulative failure times of one system.

    Time-terminated (``end_time = T >= t_n``):
    ``beta = n / sum_{i=1}^n ln(T / t_i)``. Failure-terminated
    (``end_time=None``, ``T = t_n``): ``beta = n / sum_{i=1}^{n-1} ln(t_n / t_i)``.
    In both cases ``lambda = n / T^beta``. Conditionally on ``n``,
    ``2 n beta_true / beta`` is chi-square with ``2n`` (time) or ``2n - 2``
    (failure-terminated) degrees of freedom, which gives the unbiased
    estimator and the exact interval.

    Args:
        failure_times: Strictly increasing positive cumulative test times.
        end_time: End of a time-terminated test, or ``None``.
        confidence: Level of the interval on ``beta``.

    Returns:
        Fit with unbiased shape and instantaneous MTBF.

    Raises:
        ValueError: If times are not strictly increasing and positive, or
            there are too few failures (2 time-terminated, 3
            failure-terminated).

    Equation:
        [EQ-REL-DATA-005], [EQ-REL-DATA-006].

    References:
        [REF-CROW-1974], [REF-MIL-HDBK-189C-2011],
        [REF-RAUSAND-BARROS-HOYLAND-2021].

    """
    times = tuple(_positive(t, name="failure time") for t in failure_times)
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("failure times must be strictly increasing")
    level = _confidence(confidence)
    n = len(times)
    if end_time is None:
        if n < 3:
            raise ValueError("a failure-terminated fit needs at least 3 failures")
        total, termination = times[-1], "failure"
        log_sum = math.fsum(math.log(total / t) for t in times[:-1])
        degrees, unbiased_factor = 2.0 * n - 2.0, (n - 2.0) / n
    else:
        if n < 2:
            raise ValueError("a time-terminated fit needs at least 2 failures")
        total, termination = _positive(end_time, name="end_time"), "time"
        if total < times[-1]:
            raise ValueError("end_time must not precede the last failure")
        log_sum = math.fsum(math.log(total / t) for t in times)
        degrees, unbiased_factor = 2.0 * n, (n - 1.0) / n
    if log_sum <= 0.0:
        raise ValueError("failure times give a degenerate shape estimate")
    beta = n / log_sum
    tail = 0.5 * (1.0 - level)
    interval = (
        beta * _chi2(tail, degrees) / (2.0 * n),
        beta * _chi2(1.0 - tail, degrees) / (2.0 * n),
    )
    return CrowAMSAAFit(
        n,
        total,
        termination,
        beta,
        unbiased_factor * beta,
        n / total**beta,
        interval,
        level,
    )
