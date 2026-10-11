"""Kaplan-Meier, Nelson-Aalen, median and restricted mean survival, and log-rank.

Every estimator here is nonparametric: it uses only the risk-set counts
``(n_j, d_j)`` at the distinct event times ``t_1 < t_2 < ...``. Right
censoring enters through the shrinking risk sets; a censored subject
contributes to ``n_j`` for every event time not after its censoring time.

Confidence intervals are pointwise, not simultaneous bands, and rely on
large-sample approximations (Greenwood's variance and the delta method).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, solve_spd

from ..sampling_distributions import chi_square_sf
from .data import SurvivalData


def _z_critical(confidence: float) -> float:
    """Return the two-sided standard normal critical value.

    Args:
        confidence: Confidence level strictly between zero and one.

    Returns:
        ``z_{(1 + confidence) / 2}``.

    Raises:
        ValueError: If ``confidence`` is not strictly inside ``(0, 1)``.

    References:
        [REF-CASELLA-BERGER-2002, Chapter 9]. No new equation is evaluated.

    """
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly inside (0, 1)")
    return NormalDist().inv_cdf(0.5 + 0.5 * level)


def _log_log_interval(
    survival: float, greenwood_sum: float, z: float
) -> tuple[float, float]:
    """Return the log-log transformed pointwise interval for ``S(t)``.

    Args:
        survival: Kaplan-Meier value ``S``.
        greenwood_sum: ``sum d_j / (n_j (n_j - d_j))`` up to ``t``.
        z: Two-sided normal critical value.

    Returns:
        ``(S^{exp(z w)}, S^{exp(-z w)})`` with
        ``w = sqrt(greenwood_sum) / |log S|``; ``(1, 1)`` when ``S = 1`` and
        ``(0, 0)`` when ``S = 0``.

    References:
        [EQ-SURV-KM-003], [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

    """
    if survival >= 1.0:
        return 1.0, 1.0
    if survival <= 0.0 or not math.isfinite(greenwood_sum):
        return 0.0, 0.0
    width = math.sqrt(greenwood_sum) / abs(math.log(survival))
    return survival ** math.exp(z * width), survival ** math.exp(-z * width)


@dataclass(frozen=True, slots=True)
class SurvivalQuantile:
    """A Kaplan-Meier survival quantile with a confidence interval.

    Args:
        probability: Cumulative event probability ``p`` (``0.5`` for the
            median); the quantile is the smallest ``t`` with
            ``S(t) <= 1 - p``.
        estimate: Quantile estimate, ``inf`` when the curve never falls to
            ``1 - p``.
        lower: Lower confidence limit (smallest ``t`` whose lower band is at
            most ``1 - p``), ``inf`` when not reached.
        upper: Upper confidence limit (smallest ``t`` whose upper band is at
            most ``1 - p``), ``inf`` when not reached (open-ended interval).
        confidence: Nominal pointwise level of the bands that were inverted.

    References:
        [EQ-SURV-MED-001], [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

    """

    probability: float
    estimate: float
    lower: float
    upper: float
    confidence: float

    @property
    def reached(self) -> bool:
        """Return whether the curve fell to ``1 - p`` within follow-up.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4]. No equation is evaluated.

        """
        return math.isfinite(self.estimate)


@dataclass(frozen=True, slots=True)
class RestrictedMeanSurvival:
    """Restricted mean survival time ``E[min(T, tau)]`` with its interval.

    Args:
        tau: Truncation time.
        estimate: ``int_0^tau S(u) du`` of the Kaplan-Meier curve.
        variance: Greenwood-type variance estimate.
        standard_error: Square root of ``variance``.
        lower: Normal-approximation lower limit.
        upper: Normal-approximation upper limit.
        confidence: Nominal confidence level.

    References:
        [EQ-SURV-RMST-001], [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

    """

    tau: float
    estimate: float
    variance: float
    standard_error: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True, slots=True)
class KaplanMeierEstimate:
    """Kaplan-Meier product-limit curve with Greenwood variance and intervals.

    Args:
        times: Distinct event times ``t_j``.
        at_risk: Risk-set sizes ``n_j``.
        events: Event counts ``d_j``.
        survival: ``S(t_j) = prod_{i <= j} (1 - d_i / n_i)``.
        greenwood_sums: ``sum_{i <= j} d_i / (n_i (n_i - d_i))`` (``inf``
            once ``S`` reaches zero).
        variance: Greenwood variance ``S(t_j)^2 * greenwood_sums[j]`` (zero
            where ``S = 0``).
        lower: Log-log pointwise lower confidence limits.
        upper: Log-log pointwise upper confidence limits.
        confidence: Nominal pointwise confidence level.
        last_time: Largest observed (event or censoring) time.
        size: Number of subjects.

    References:
        [EQ-SURV-KM-001], [EQ-SURV-KM-002], [EQ-SURV-KM-003],
        [REF-KAPLAN-MEIER-1958], [REF-GREENWOOD-1926].

    """

    times: Vector
    at_risk: tuple[int, ...]
    events: tuple[int, ...]
    survival: Vector
    greenwood_sums: Vector
    variance: Vector
    lower: Vector
    upper: Vector
    confidence: float
    last_time: float
    size: int

    @property
    def standard_errors(self) -> Vector:
        """Return Greenwood standard errors ``sqrt(Var S(t_j))``.

        References:
            [EQ-SURV-KM-002], [REF-GREENWOOD-1926].

        """
        return tuple(math.sqrt(value) for value in self.variance)

    def _index(self, time: float) -> int:
        """Return the number of event times not after ``time``.

        References:
            [EQ-SURV-KM-001], [REF-KAPLAN-MEIER-1958]. No new equation.

        """
        return bisect.bisect_right(self.times, time)

    def survival_at(self, time: float) -> float:
        """Evaluate the right-continuous step function ``S(t)``.

        Args:
            time: Finite evaluation time.

        Returns:
            ``1`` before the first event time, otherwise ``S(t_j)`` for the
            last event time ``t_j <= t``. Beyond ``last_time`` the last value
            is carried forward although the curve is not identified there.

        Equation:
            [EQ-SURV-KM-001].

        References:
            [REF-KAPLAN-MEIER-1958].

        """
        point = require_finite_real(time, name="time")
        index = self._index(point)
        return 1.0 if index == 0 else self.survival[index - 1]

    def interval_at(self, time: float) -> tuple[float, float]:
        """Return the log-log pointwise confidence interval at ``time``.

        Args:
            time: Finite evaluation time.

        Returns:
            ``(lower, upper)``; ``(1, 1)`` before the first event.

        Equation:
            [EQ-SURV-KM-003].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

        """
        point = require_finite_real(time, name="time")
        index = self._index(point)
        if index == 0:
            return 1.0, 1.0
        return self.lower[index - 1], self.upper[index - 1]

    def quantile(self, probability: float = 0.5) -> SurvivalQuantile:
        """Return the survival-time quantile and its band-inversion interval.

        Args:
            probability: Event probability ``p`` in ``(0, 1)``; ``0.5`` gives
                the median survival time.

        Returns:
            Estimate and limits; each is ``inf`` when not reached.

        Raises:
            ValueError: If ``probability`` is not strictly inside ``(0, 1)``.

        Equation:
            [EQ-SURV-MED-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 < p < 1.0:
            raise ValueError("probability must lie strictly inside (0, 1)")
        target = 1.0 - p
        tolerance = 1e-12

        def first(values: Vector) -> float:
            """Return the first event time whose value is at most the target.

            References:
                [EQ-SURV-MED-001]. No new equation is evaluated.

            """
            for time, value in zip(self.times, values, strict=True):
                if value <= target + tolerance:
                    return time
            return math.inf

        return SurvivalQuantile(
            p,
            first(self.survival),
            first(self.lower),
            first(self.upper),
            self.confidence,
        )

    def median(self) -> SurvivalQuantile:
        """Return the median survival time with its confidence interval.

        Returns:
            :meth:`quantile` at ``p = 0.5``.

        Equation:
            [EQ-SURV-MED-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

        """
        return self.quantile(0.5)

    def restricted_mean(self, tau: float) -> RestrictedMeanSurvival:
        """Return the restricted mean survival time up to ``tau``.

        ``mu(tau) = int_0^tau S(u) du`` with variance
        ``sum_{t_j <= tau} A_j^2 d_j / (n_j (n_j - d_j))`` and
        ``A_j = int_{t_j}^tau S(u) du``.

        Args:
            tau: Positive truncation time; it may exceed the last observed
                time only when the curve has already reached zero.

        Returns:
            Estimate, variance, and normal-approximation interval.

        Raises:
            ValueError: If ``tau`` is not positive or lies beyond the follow-up
                while ``S`` is still positive.

        Equation:
            [EQ-SURV-RMST-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

        """
        limit = require_finite_real(tau, name="tau")
        if not limit > 0.0:
            raise ValueError("tau must be positive")
        final = self.survival[-1] if self.survival else 1.0
        if limit > self.last_time and final > 0.0:
            raise ValueError(
                "tau exceeds the follow-up while the survival curve is positive; "
                "the restricted mean is not identified"
            )
        knots = [0.0] + [t for t in self.times if t < limit] + [limit]
        levels = [1.0] + list(self.survival[: len(knots) - 2])
        pieces = [
            level * (right - left)
            for level, left, right in zip(levels, knots[:-1], knots[1:], strict=True)
        ]
        estimate = math.fsum(pieces)
        variance_terms: list[float] = []
        for j in range(len(knots) - 2):
            tail_area = math.fsum(pieces[j + 1 :])
            n, d = self.at_risk[j], self.events[j]
            if n == d or tail_area == 0.0:
                continue
            variance_terms.append(tail_area * tail_area * d / (n * (n - d)))
        variance = math.fsum(variance_terms)
        error = math.sqrt(variance)
        z = _z_critical(self.confidence)
        return RestrictedMeanSurvival(
            limit,
            estimate,
            variance,
            error,
            estimate - z * error,
            estimate + z * error,
            self.confidence,
        )


def kaplan_meier(
    data: SurvivalData, *, confidence: float = 0.95
) -> KaplanMeierEstimate:
    """Estimate the survival function by the Kaplan-Meier product limit.

    ``S(t) = prod_{t_j <= t} (1 - d_j / n_j)`` over distinct event times, with
    Greenwood's variance ``S(t)^2 sum_{t_j <= t} d_j / (n_j (n_j - d_j))`` and
    log-log pointwise intervals ``S^{exp(+/- z w)}``,
    ``w = sqrt(sum d_j / (n_j (n_j - d_j))) / |log S|``.

    Args:
        data: Right-censored sample.
        confidence: Pointwise confidence level in ``(0, 1)``.

    Returns:
        Kaplan-Meier curve at every distinct event time (empty when there is
        no event).

    Raises:
        ValueError: If ``confidence`` is not strictly inside ``(0, 1)``.

    Equation:
        [EQ-SURV-KM-001], [EQ-SURV-KM-002], [EQ-SURV-KM-003].

    References:
        [REF-KAPLAN-MEIER-1958], [REF-GREENWOOD-1926],
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

    """
    z = _z_critical(confidence)
    rows = data.event_table()
    survival: list[float] = []
    sums: list[float] = []
    variance: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    current = 1.0
    running = 0.0
    for row in rows:
        n, d = row.at_risk, row.events
        current *= 1.0 - d / n
        running = math.inf if n == d else running + d / (n * (n - d))
        if current <= 0.0:
            current = 0.0
            running = math.inf
        survival.append(current)
        sums.append(running)
        variance.append(0.0 if current == 0.0 else current * current * running)
        low, high = _log_log_interval(current, running, z)
        lower.append(low)
        upper.append(high)
    return KaplanMeierEstimate(
        tuple(row.time for row in rows),
        tuple(row.at_risk for row in rows),
        tuple(row.events for row in rows),
        tuple(survival),
        tuple(sums),
        tuple(variance),
        tuple(lower),
        tuple(upper),
        float(confidence),
        max(data.times),
        data.size,
    )


@dataclass(frozen=True, slots=True)
class NelsonAalenEstimate:
    """Nelson-Aalen cumulative hazard with its variance and intervals.

    Args:
        times: Distinct event times ``t_j``.
        at_risk: Risk-set sizes ``n_j``.
        events: Event counts ``d_j``.
        cumulative_hazard: ``H(t_j) = sum_{i <= j} d_i / n_i``.
        variance: ``sum_{i <= j} d_i / n_i^2``.
        lower: Log-transformed lower limits ``H exp(-z se / H)``.
        upper: Log-transformed upper limits ``H exp(z se / H)``.
        confidence: Nominal pointwise confidence level.

    References:
        [EQ-SURV-NA-001], [REF-NELSON-1972], [REF-AALEN-1978].

    """

    times: Vector
    at_risk: tuple[int, ...]
    events: tuple[int, ...]
    cumulative_hazard: Vector
    variance: Vector
    lower: Vector
    upper: Vector
    confidence: float

    def cumulative_hazard_at(self, time: float) -> float:
        """Evaluate the right-continuous step function ``H(t)``.

        Args:
            time: Finite evaluation time.

        Returns:
            ``0`` before the first event, else ``H(t_j)`` with ``t_j <= t``.

        Equation:
            [EQ-SURV-NA-001].

        References:
            [REF-NELSON-1972], [REF-AALEN-1978].

        """
        point = require_finite_real(time, name="time")
        index = bisect.bisect_right(self.times, point)
        return 0.0 if index == 0 else self.cumulative_hazard[index - 1]

    def survival_at(self, time: float) -> float:
        """Return the Fleming-Harrington survival estimate ``exp(-H(t))``.

        Args:
            time: Finite evaluation time.

        Returns:
            ``exp(-H(t))``, never below the Kaplan-Meier value.

        Equation:
            [EQ-SURV-NA-001].

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

        """
        return math.exp(-self.cumulative_hazard_at(time))


def nelson_aalen(
    data: SurvivalData, *, confidence: float = 0.95
) -> NelsonAalenEstimate:
    """Estimate the cumulative hazard by the Nelson-Aalen estimator.

    ``H(t) = sum_{t_j <= t} d_j / n_j`` with variance ``sum d_j / n_j^2`` and
    log-transformed pointwise intervals ``H exp(+/- z sqrt(Var) / H)``.

    Args:
        data: Right-censored sample.
        confidence: Pointwise confidence level in ``(0, 1)``.

    Returns:
        Step-function estimate at every distinct event time.

    Raises:
        ValueError: If ``confidence`` is not strictly inside ``(0, 1)``.

    Equation:
        [EQ-SURV-NA-001].

    References:
        [REF-NELSON-1972], [REF-AALEN-1978],
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

    """
    z = _z_critical(confidence)
    rows = data.event_table()
    hazard: list[float] = []
    variance: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    total = 0.0
    spread = 0.0
    for row in rows:
        total += row.events / row.at_risk
        spread += row.events / (row.at_risk * row.at_risk)
        hazard.append(total)
        variance.append(spread)
        factor = math.exp(z * math.sqrt(spread) / total)
        lower.append(total / factor)
        upper.append(total * factor)
    return NelsonAalenEstimate(
        tuple(row.time for row in rows),
        tuple(row.at_risk for row in rows),
        tuple(row.events for row in rows),
        tuple(hazard),
        tuple(variance),
        tuple(lower),
        tuple(upper),
        float(confidence),
    )


@dataclass(frozen=True, slots=True)
class LogRankResult:
    """The ``k``-sample log-rank (Mantel-Haenszel) test.

    Args:
        statistic: ``U' V^{-1} U`` on the first ``k - 1`` groups.
        degrees_of_freedom: ``k - 1``.
        p_value: Upper chi-square tail probability.
        observed: Observed events ``O_g`` per group.
        expected: Expected events ``E_g = sum_j d_j n_gj / n_j`` per group.
        covariance: Full ``k x k`` hypergeometric covariance of ``O - E``.
        event_times: Number of distinct pooled event times used.
        method: Description of the test.

    References:
        [EQ-SURV-LR-001], [REF-MANTEL-1966],
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 7].

    """

    statistic: float
    degrees_of_freedom: int
    p_value: float
    observed: tuple[int, ...]
    expected: Vector
    covariance: Matrix
    event_times: int
    method: str = "log-rank test (chi-square approximation)"


def log_rank_test(*samples: SurvivalData) -> LogRankResult:
    """Test equality of ``k >= 2`` survival curves by the log-rank statistic.

    At each pooled distinct event time ``t_j`` with ``n_j`` at risk and ``d_j``
    events, group ``g`` has ``n_gj`` at risk and ``d_gj`` events.
    ``U_g = sum_j (d_gj - d_j n_gj / n_j)`` and
    ``V_gh = sum_j d_j (n_j - d_j) / (n_j - 1) (n_gj / n_j)
    (1[g = h] - n_hj / n_j)``. Under equal hazards
    ``U' V^{-1} U`` (first ``k - 1`` groups) is approximately chi-square with
    ``k - 1`` degrees of freedom.

    Args:
        *samples: Two or more independent right-censored samples.

    Returns:
        Statistic, p-value, observed and expected events, and covariance.

    Raises:
        TypeError: If a sample is not :class:`SurvivalData`.
        ValueError: If fewer than two groups are given, there is no event, or
            the covariance is singular (for example an empty risk set in a
            group at every event time).

    Equation:
        [EQ-SURV-LR-001].

    References:
        [REF-MANTEL-1966], [REF-KLEIN-MOESCHBERGER-2003, Chapter 7].

    """
    if len(samples) < 2:
        raise ValueError("the log-rank test needs at least two groups")
    for index, sample in enumerate(samples):
        if not isinstance(sample, SurvivalData):
            raise TypeError(f"samples[{index}] must be SurvivalData")
    groups = len(samples)
    sorted_times = [sorted(sample.times) for sample in samples]
    event_counts: list[dict[float, int]] = []
    for sample in samples:
        counts: dict[float, int] = {}
        for time, flag in zip(sample.times, sample.events, strict=True):
            if flag:
                counts[time] = counts.get(time, 0) + 1
        event_counts.append(counts)
    pooled = sorted({time for counts in event_counts for time in counts})
    if not pooled:
        raise ValueError("the log-rank test needs at least one event")
    observed = [0.0] * groups
    expected = [0.0] * groups
    covariance = [[0.0] * groups for _ in range(groups)]
    for time in pooled:
        at_risk = [
            len(times) - bisect.bisect_left(times, time) for times in sorted_times
        ]
        deaths = [counts.get(time, 0) for counts in event_counts]
        n = sum(at_risk)
        d = sum(deaths)
        for g in range(groups):
            observed[g] += deaths[g]
            expected[g] += d * at_risk[g] / n
        if n <= 1:
            continue
        factor = d * (n - d) / (n - 1)
        for g in range(groups):
            share_g = at_risk[g] / n
            for h in range(groups):
                indicator = 1.0 if g == h else 0.0
                covariance[g][h] += factor * share_g * (indicator - at_risk[h] / n)
    difference = tuple(o - e for o, e in zip(observed, expected, strict=True))
    reduced = tuple(tuple(row[: groups - 1]) for row in covariance[: groups - 1])
    try:
        solution = solve_spd(reduced, difference[: groups - 1])
    except ValueError as error:
        raise ValueError(
            "log-rank covariance is singular; a group may never be at risk"
        ) from error
    statistic = math.fsum(
        u * s for u, s in zip(difference[: groups - 1], solution, strict=True)
    )
    statistic = max(statistic, 0.0)
    df = groups - 1
    return LogRankResult(
        statistic,
        df,
        chi_square_sf(statistic, df),
        tuple(int(value) for value in observed),
        tuple(expected),
        tuple(tuple(row) for row in covariance),
        len(pooled),
    )
