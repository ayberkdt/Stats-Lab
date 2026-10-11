"""Nonparametric bootstrap, jackknife, bootstrap intervals, and permutation tests.

All resampling is driven by :class:`probstat.foundations.RandomStream`, so a
given integer seed reproduces every replicate exactly. Bootstrap intervals are
approximate repeated-sampling procedures; their nominal confidence is not a
probability statement about the fixed parameter. Permutation p-values are
probabilities, under exchangeability of the pooled observations, of a
statistic at least as extreme as observed; they are not probabilities that the
null hypothesis is true.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal, TypeVar

from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.statistics import quantile

from ._hypothesis import Alternative, check_alternative, check_level
from .confidence import ConfidenceInterval

T = TypeVar("T")
IntervalMethod = Literal["percentile", "basic", "normal", "studentized", "bca"]
_INTERVAL_METHODS: tuple[str, ...] = (
    "percentile",
    "basic",
    "normal",
    "studentized",
    "bca",
)


def _finite_statistic(value: object) -> float:
    """Validate one statistic evaluation.

    Args:
        value: Return value of a user statistic.

    Returns:
        Finite float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is not finite.

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 6]. No equation is evaluated.

    """
    return require_finite_real(value, name="statistic(data)")


def _materialize(data: Iterable[T]) -> tuple[T, ...]:
    """Materialize resampling data and require at least two observations.

    Args:
        data: Observations (any objects the statistic understands).

    Returns:
        Tuple of observations.

    Raises:
        ValueError: If fewer than two observations are supplied.

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 6]. No equation is evaluated.

    """
    items = tuple(data)
    if len(items) < 2:
        raise ValueError("data must contain at least two observations")
    return items


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Nonparametric bootstrap replicates and their summaries.

    Args:
        estimate: Statistic evaluated on the original data.
        replicates: Statistic evaluated on each resample, in draw order.
        bias: ``mean(replicates) - estimate``.
        standard_error: Sample standard deviation of the replicates.
        seed: Seed of the stream that generated the resamples.

    References:
        [REF-EFRON-1979], [REF-EFRON-TIBSHIRANI-1993, Chapters 6 and 10].

    """

    estimate: float
    replicates: tuple[float, ...]
    bias: float
    standard_error: float
    seed: int

    @property
    def bias_corrected_estimate(self) -> float:
        """Return ``estimate - bias = 2 estimate - mean(replicates)``.

        Bias correction can increase variance; report it next to, not
        instead of, the plain estimate.

        Returns:
            Bias-corrected estimate.

        Equation:
            [EQ-INF-BOOT-001].

        References:
            [REF-EFRON-TIBSHIRANI-1993, Chapter 10].

        """
        return self.estimate - self.bias


def bootstrap(
    statistic: Callable[[Sequence[T]], float],
    data: Iterable[T],
    *,
    replicates: int = 2000,
    seed: int | RandomStream | None = None,
) -> BootstrapResult:
    """Draw nonparametric bootstrap replicates of a statistic.

    Each resample draws ``n`` observations with replacement from the data.
    Observations may be any objects (for example ``(x, y)`` pairs), so the
    statistic is responsible for interpreting them.

    Args:
        statistic: Function mapping a sequence of observations to a finite
            real value.
        data: At least two observations.
        replicates: Number of bootstrap resamples, at least two.
        seed: Integer seed or stream for reproducibility; ``None`` is not
            reproducible.

    Returns:
        Original estimate, replicates, bootstrap bias, and standard error.

    Raises:
        TypeError: If an argument has an invalid type or the statistic does
            not return a real value.
        ValueError: If sizes are invalid or the statistic is not finite.

    Equation:
        [EQ-INF-BOOT-001].

    References:
        [REF-EFRON-1979], [REF-EFRON-TIBSHIRANI-1993, Chapters 6 and 10].

    """
    items = _materialize(data)
    count = require_non_negative_integer(replicates, name="replicates")
    if count < 2:
        raise ValueError("replicates must be at least two")
    stream = resolve_stream(seed)
    estimate = _finite_statistic(statistic(items))
    values = tuple(
        _finite_statistic(statistic(sample))
        for sample in _resamples(items, count, stream)
    )
    return BootstrapResult(
        estimate,
        values,
        statistics.fmean(values) - estimate,
        statistics.stdev(values),
        stream.seed,
    )


def _resamples(
    items: tuple[T, ...], count: int, stream: RandomStream
) -> Iterable[tuple[T, ...]]:
    """Yield ``count`` resamples with replacement from one generator.

    Args:
        items: Original observations.
        count: Number of resamples.
        stream: Stream whose ``"bootstrap"`` child drives the draws.

    Yields:
        Resampled tuples of the original length.

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 6], [REF-STEELE-LEA-FLOOD-2014].

    """
    generator = stream.child("bootstrap").generator()
    n = len(items)
    for _ in range(count):
        yield tuple(generator.choices(items, k=n))


@dataclass(frozen=True, slots=True)
class JackknifeResult:
    """Leave-one-out jackknife estimates.

    Args:
        estimate: Statistic on the full data.
        leave_one_out: Statistic with observation ``i`` removed, for each ``i``.
        bias: ``(n - 1)(mean(leave_one_out) - estimate)``.
        standard_error: ``sqrt((n - 1)/n sum (theta_(i) - theta_(.))^2)``.
        bias_corrected_estimate: ``estimate - bias``.

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 11].

    """

    estimate: float
    leave_one_out: tuple[float, ...]
    bias: float
    standard_error: float
    bias_corrected_estimate: float


def jackknife(
    statistic: Callable[[Sequence[T]], float], data: Iterable[T]
) -> JackknifeResult:
    """Compute jackknife bias and standard-error estimates of a statistic.

    The jackknife is deterministic. Its standard error is unreliable for
    non-smooth statistics such as the median.

    Args:
        statistic: Function mapping a sequence of observations to a real value.
        data: At least two observations.

    Returns:
        Leave-one-out values, bias, standard error, and corrected estimate.

    Raises:
        TypeError: If the statistic does not return a real value.
        ValueError: If fewer than two observations or a non-finite statistic.

    Equation:
        [EQ-INF-JACK-001].

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapter 11].

    """
    items = _materialize(data)
    n = len(items)
    estimate = _finite_statistic(statistic(items))
    values = tuple(
        _finite_statistic(statistic(items[:i] + items[i + 1 :])) for i in range(n)
    )
    center = statistics.fmean(values)
    bias = (n - 1) * (center - estimate)
    spread = math.fsum((v - center) ** 2 for v in values)
    return JackknifeResult(
        estimate,
        values,
        bias,
        math.sqrt((n - 1) / n * spread),
        estimate - bias,
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A bootstrap confidence interval with its construction diagnostics.

    Args:
        interval: Two-sided interval and nominal confidence.
        method: ``percentile``, ``basic``, ``normal``, ``studentized``, or
            ``bca``.
        bootstrap: Underlying replicates and summaries.
        bias_correction: BCa ``z_0`` (``None`` for other methods).
        acceleration: BCa jackknife acceleration ``a`` (``None`` otherwise).

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapters 12-14],
        [REF-DAVISON-HINKLEY-1997, Chapter 5].

    """

    interval: ConfidenceInterval
    method: str
    bootstrap: BootstrapResult
    bias_correction: float | None = None
    acceleration: float | None = None

    @property
    def lower(self) -> float:
        """Return the lower endpoint.

        Returns:
            Lower interval endpoint.

        References:
            [REF-DAVISON-HINKLEY-1997, Chapter 5]. No equation is evaluated.

        """
        return self.interval.lower

    @property
    def upper(self) -> float:
        """Return the upper endpoint.

        Returns:
            Upper interval endpoint.

        References:
            [REF-DAVISON-HINKLEY-1997, Chapter 5]. No equation is evaluated.

        """
        return self.interval.upper


def bootstrap_confidence_interval(
    statistic: Callable[[Sequence[T]], float],
    data: Iterable[T],
    *,
    method: IntervalMethod = "bca",
    confidence: float = 0.95,
    replicates: int = 2000,
    seed: int | RandomStream | None = None,
    standard_error: Callable[[Sequence[T]], float] | None = None,
) -> BootstrapInterval:
    """Construct a nonparametric bootstrap confidence interval.

    With ``alpha = 1 - confidence`` and replicate quantiles ``q_p`` (type-7
    interpolation):

    - ``percentile``: ``[q_{alpha/2}, q_{1-alpha/2}]``.
    - ``basic``: ``[2 theta - q_{1-alpha/2}, 2 theta - q_{alpha/2}]``.
    - ``normal``: ``theta - b +- z_{1-alpha/2} se`` with bootstrap bias ``b``.
    - ``studentized``: ``[theta - t*_{1-alpha/2} se, theta - t*_{alpha/2} se]``
      using ``t* = (theta* - theta) / se*``; ``se`` is ``standard_error``
      when given, otherwise the jackknife standard error of each resample.
    - ``bca``: percentile levels adjusted by bias correction ``z_0`` (from
      the share of replicates below ``theta``, counting ties as one half)
      and jackknife acceleration ``a``.

    Args:
        statistic: Function mapping observations to a finite real value.
        data: At least two observations.
        method: Interval construction.
        confidence: Nominal two-sided confidence level in ``(0, 1)``.
        replicates: Bootstrap resamples (at least 100 recommended; 2000 or
            more for BCa and studentized intervals).
        seed: Integer seed or stream; the same seed reproduces replicates
            across methods.
        standard_error: Optional standard-error function for the
            studentized interval.

    Returns:
        Interval, replicates, and BCa diagnostics when applicable.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the method is unknown, BCa replicates all lie on one
            side of the estimate, the jackknife is degenerate, or a
            resample has zero standard error.

    Equation:
        [EQ-INF-BOOT-002], [EQ-INF-BOOT-003], [EQ-INF-BOOT-004],
        [EQ-STAT-QUANT-001].

    References:
        [REF-EFRON-TIBSHIRANI-1993, Chapters 12-14], [REF-EFRON-1987],
        [REF-DAVISON-HINKLEY-1997, Chapter 5].

    """
    if method not in _INTERVAL_METHODS:
        raise ValueError(
            "method must be 'percentile', 'basic', 'normal', 'studentized', or 'bca'"
        )
    level = check_level(confidence, name="confidence")
    items = _materialize(data)
    alpha = 1.0 - level
    stream = resolve_stream(seed)
    normal = NormalDist()
    if method == "studentized":
        return _studentized_interval(
            statistic, items, level, replicates, stream, standard_error
        )
    result = bootstrap(statistic, items, replicates=replicates, seed=stream)
    ordered = sorted(result.replicates)
    theta = result.estimate
    if method == "percentile":
        lower = quantile(ordered, 0.5 * alpha)
        upper = quantile(ordered, 1.0 - 0.5 * alpha)
        return BootstrapInterval(
            ConfidenceInterval(lower, upper, level, "bootstrap percentile"),
            method,
            result,
        )
    if method == "basic":
        lower = 2.0 * theta - quantile(ordered, 1.0 - 0.5 * alpha)
        upper = 2.0 * theta - quantile(ordered, 0.5 * alpha)
        return BootstrapInterval(
            ConfidenceInterval(lower, upper, level, "bootstrap basic"),
            method,
            result,
        )
    if method == "normal":
        z = normal.inv_cdf(1.0 - 0.5 * alpha)
        center = theta - result.bias
        return BootstrapInterval(
            ConfidenceInterval(
                center - z * result.standard_error,
                center + z * result.standard_error,
                level,
                "bootstrap normal (bias-corrected)",
            ),
            method,
            result,
        )
    below = sum(1 for v in ordered if v < theta)
    equal = sum(1 for v in ordered if v == theta)
    share = (below + 0.5 * equal) / len(ordered)
    if share <= 0.0 or share >= 1.0:
        raise ValueError(
            "BCa bias correction is infinite: all replicates lie on one side "
            "of the estimate"
        )
    z0 = normal.inv_cdf(share)
    acceleration = _jackknife_acceleration(statistic, items)
    levels = []
    for tail in (0.5 * alpha, 1.0 - 0.5 * alpha):
        z_tail = normal.inv_cdf(tail)
        denominator = 1.0 - acceleration * (z0 + z_tail)
        if denominator <= 0.0:
            raise ValueError("BCa acceleration makes the adjusted level undefined")
        levels.append(normal.cdf(z0 + (z0 + z_tail) / denominator))
    lower = quantile(ordered, levels[0])
    upper = quantile(ordered, levels[1])
    return BootstrapInterval(
        ConfidenceInterval(
            min(lower, upper), max(lower, upper), level, "bootstrap BCa"
        ),
        method,
        result,
        z0,
        acceleration,
    )


def _jackknife_acceleration(
    statistic: Callable[[Sequence[T]], float], items: tuple[T, ...]
) -> float:
    """Return the BCa acceleration from jackknife influence values.

    Args:
        statistic: Statistic function.
        items: Original observations.

    Returns:
        ``a = sum (m - theta_(i))^3 / (6 (sum (m - theta_(i))^2)^{3/2})``.

    Raises:
        ValueError: If all leave-one-out values are equal.

    References:
        [EQ-INF-BOOT-004], [REF-EFRON-1987],
        [REF-EFRON-TIBSHIRANI-1993, Chapter 14].

    """
    values = jackknife(statistic, items).leave_one_out
    center = statistics.fmean(values)
    deviations = [center - v for v in values]
    squares = math.fsum(d * d for d in deviations)
    if squares == 0.0:
        raise ValueError("jackknife values are constant; acceleration undefined")
    return math.fsum(d * d * d for d in deviations) / (6.0 * math.sqrt(squares) ** 3)


def _studentized_interval(
    statistic: Callable[[Sequence[T]], float],
    items: tuple[T, ...],
    level: float,
    replicates: int,
    stream: RandomStream,
    standard_error: Callable[[Sequence[T]], float] | None,
) -> BootstrapInterval:
    """Construct the bootstrap-t (studentized) interval.

    Args:
        statistic: Statistic function.
        items: Original observations.
        level: Confidence level.
        replicates: Number of resamples.
        stream: Random stream.
        standard_error: Optional standard-error function; jackknife otherwise.

    Returns:
        Studentized interval.

    Raises:
        ValueError: If a standard error is zero or non-finite.

    References:
        [EQ-INF-BOOT-003], [REF-EFRON-TIBSHIRANI-1993, Chapter 12],
        [REF-DAVISON-HINKLEY-1997, Chapter 5].

    """
    count = require_non_negative_integer(replicates, name="replicates")
    if count < 2:
        raise ValueError("replicates must be at least two")

    def scale(sample: Sequence[T]) -> float:
        """Return a positive standard error for one (re)sample.

        References:
            [EQ-INF-BOOT-003], [EQ-INF-JACK-001].

        """
        if standard_error is None:
            value = jackknife(statistic, sample).standard_error
        else:
            value = require_finite_real(standard_error(sample), name="standard_error")
        if value <= 0.0:
            raise ValueError("a standard error is zero; studentization undefined")
        return value

    theta = _finite_statistic(statistic(items))
    se = scale(items)
    replicate_values: list[float] = []
    pivots: list[float] = []
    for sample in _resamples(items, count, stream):
        value = _finite_statistic(statistic(sample))
        replicate_values.append(value)
        pivots.append((value - theta) / scale(sample))
    alpha = 1.0 - level
    ordered = sorted(pivots)
    lower = theta - quantile(ordered, 1.0 - 0.5 * alpha) * se
    upper = theta - quantile(ordered, 0.5 * alpha) * se
    result = BootstrapResult(
        theta,
        tuple(replicate_values),
        statistics.fmean(replicate_values) - theta,
        statistics.stdev(replicate_values),
        stream.seed,
    )
    return BootstrapInterval(
        ConfidenceInterval(lower, upper, level, "bootstrap studentized (bootstrap-t)"),
        "studentized",
        result,
    )


@dataclass(frozen=True, slots=True)
class PermutationTestResult:
    """Two-sample permutation test for a difference in means.

    Args:
        statistic: Observed ``mean(first) - mean(second)``.
        p_value: Exact share (or Monte Carlo ``(b + 1)/(m + 1)`` estimate) of
            relabelings with a statistic at least as extreme as observed,
            under exchangeability; not the probability that the null is true.
        alternative: Direction of the alternative hypothesis.
        method: ``exact enumeration`` or ``Monte Carlo`` label.
        exact: Whether all relabelings were enumerated.
        permutations: Relabelings evaluated (all, or random draws).
        seed: Seed used for Monte Carlo draws, ``None`` when exact.

    References:
        [REF-GOOD-2005], [REF-PHIPSON-SMYTH-2010].

    """

    statistic: float
    p_value: float
    alternative: str
    method: str
    exact: bool
    permutations: int
    seed: int | None

    def __post_init__(self) -> None:
        """Validate the p-value.

        Raises:
            ValueError: If the p-value lies outside ``[0, 1]``.

        References:
            [REF-GOOD-2005]. No equation is evaluated.

        """
        object.__setattr__(
            self, "p_value", require_probability(self.p_value, name="p_value")
        )


def permutation_test(
    first: Iterable[object],
    second: Iterable[object],
    *,
    alternative: Alternative = "two-sided",
    permutations: int = 9999,
    max_exact: int = 100_000,
    seed: int | RandomStream | None = None,
) -> PermutationTestResult:
    """Test equality of distributions by permuting group labels.

    The statistic is ``mean(first) - mean(second)``. When the number of
    distinct relabelings ``C(m + n, m)`` is at most ``max_exact``, every
    relabeling is enumerated and the p-value is exact. Otherwise
    ``permutations`` random relabelings are drawn and the p-value is
    ``(b + 1)/(permutations + 1)``, where ``b`` counts draws at least as
    extreme; this estimate is never zero and keeps the test valid. The
    two-sided version compares absolute differences. Comparisons use a
    relative tolerance of ``1e-12`` so floating-point ties count as extreme.

    Args:
        first: First sample.
        second: Second sample.
        alternative: ``"greater"`` means the first mean tends to be larger.
        permutations: Random relabelings for the Monte Carlo branch.
        max_exact: Largest relabeling count enumerated exactly.
        seed: Integer seed or stream for the Monte Carlo branch.

    Returns:
        Observed difference, p-value, and branch metadata.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If a sample is empty or counts are invalid.

    Equation:
        [EQ-INF-PERM-001].

    References:
        [REF-GOOD-2005], [REF-PHIPSON-SMYTH-2010].

    """
    a = coerce_finite_sample(first, name="first")
    b = coerce_finite_sample(second, name="second")
    side = check_alternative(alternative)
    draws = require_non_negative_integer(permutations, name="permutations")
    limit = require_non_negative_integer(max_exact, name="max_exact")
    if draws < 1:
        raise ValueError("permutations must be positive")
    pooled = a + b
    m, n = len(a), len(b)
    total = math.fsum(pooled)

    def difference(first_sum: float) -> float:
        """Return the mean difference for a first-group sum.

        References:
            [EQ-INF-PERM-001].

        """
        return first_sum / m - (total - first_sum) / n

    observed = difference(math.fsum(a))
    tolerance = 1e-12 * max(1.0, abs(observed))

    def extreme(value: float) -> bool:
        """Return whether a relabeled statistic is at least as extreme.

        References:
            [EQ-INF-PERM-001].

        """
        if side == "greater":
            return value >= observed - tolerance
        if side == "less":
            return value <= observed + tolerance
        return abs(value) >= abs(observed) - tolerance

    relabelings = math.comb(m + n, m)
    if relabelings <= limit:
        hits = sum(
            1
            for chosen in itertools.combinations(pooled, m)
            if extreme(difference(math.fsum(chosen)))
        )
        return PermutationTestResult(
            observed,
            hits / relabelings,
            side,
            "permutation test (exact enumeration)",
            True,
            relabelings,
            None,
        )
    stream = resolve_stream(seed)
    generator = stream.child("permutation").generator()
    shuffled = list(pooled)
    hits = 0
    for _ in range(draws):
        generator.shuffle(shuffled)
        if extreme(difference(math.fsum(shuffled[:m]))):
            hits += 1
    return PermutationTestResult(
        observed,
        (hits + 1) / (draws + 1),
        side,
        "permutation test (Monte Carlo, (b+1)/(m+1))",
        False,
        draws,
        stream.seed,
    )
