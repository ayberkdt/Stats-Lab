"""Aalen-Johansen cumulative incidence functions for competing risks.

With competing risks every subject fails from at most one of several causes
(or is right-censored). The quantity of interest is the cumulative incidence
function (sub-distribution) ``F_k(t) = P(T <= t, cause = k)``, the
probability of having failed from cause ``k`` by ``t`` *in the presence of*
the other causes. The Aalen-Johansen estimator is

    ``F_k(t) = sum_{t_j <= t} S(t_j-) d_kj / n_j``

over the distinct event times of any cause, where ``n_j`` is the risk set,
``d_kj`` the number of cause-``k`` events at ``t_j`` and ``S`` the
Kaplan-Meier estimator of all-cause survival [EQ-SURV-CIF-001]. It satisfies
``sum_k F_k(t) = 1 - S(t)`` exactly.

The naive alternative ``1 - KM_k(t)``, the complement of a Kaplan-Meier
curve that treats the other causes as censoring, estimates the failure
probability in a hypothetical world without the other causes (and only
under independent latent failure times). It is never smaller than the
Aalen-Johansen estimate and overstates the incidence whenever a competing
event occurs before a cause-``k`` event.

Censoring is assumed independent of the event process. Ties follow the
package convention: a subject censored at ``t`` is at risk for an event at
``t``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Vector

from .nonparametric import _z_critical


@dataclass(frozen=True, slots=True)
class CumulativeIncidenceEstimate:
    """Aalen-Johansen cumulative incidence of one cause with its variance.

    All vectors have one entry per distinct time at which an event of *any*
    cause was observed, so that curves of different causes share their grid.

    Args:
        cause: Positive integer label ``k`` of the cause.
        times: Distinct all-cause event times ``t_j``.
        at_risk: Risk-set sizes ``n_j``.
        cause_events: Cause-``k`` event counts ``d_kj``.
        all_events: All-cause event counts ``d_j``.
        overall_survival: All-cause Kaplan-Meier values ``S(t_j)``.
        incidence: ``F_k(t_j)``.
        variance: Delta-method variance estimate of ``F_k(t_j)``.
        lower: Normal-approximation lower limits, clipped at zero.
        upper: Normal-approximation upper limits, clipped at one.
        confidence: Nominal pointwise confidence level.
        last_time: Largest observed (event or censoring) time.
        size: Number of subjects.

    References:
        [EQ-SURV-CIF-001], [EQ-SURV-CIF-002], [REF-AALEN-JOHANSEN-1978].

    """

    cause: int
    times: Vector
    at_risk: tuple[int, ...]
    cause_events: tuple[int, ...]
    all_events: tuple[int, ...]
    overall_survival: Vector
    incidence: Vector
    variance: Vector
    lower: Vector
    upper: Vector
    confidence: float
    last_time: float
    size: int

    @property
    def standard_errors(self) -> Vector:
        """Return the standard errors ``sqrt(Var F_k(t_j))``.

        References:
            [EQ-SURV-CIF-002]. No new equation is evaluated.

        """
        return tuple(math.sqrt(value) for value in self.variance)

    def incidence_at(self, time: float) -> float:
        """Evaluate the right-continuous step function ``F_k(t)``.

        Args:
            time: Finite evaluation time.

        Returns:
            ``0`` before the first event time, otherwise ``F_k(t_j)`` for the
            last event time ``t_j <= t``. Beyond ``last_time`` the last value
            is carried forward although the curve is not identified there.

        Raises:
            TypeError: If ``time`` is not real.
            ValueError: If ``time`` is not finite.

        Equation:
            [EQ-SURV-CIF-001].

        References:
            [REF-AALEN-JOHANSEN-1978].

        """
        point = require_finite_real(time, name="time")
        index = bisect.bisect_right(self.times, point)
        return 0.0 if index == 0 else self.incidence[index - 1]

    def interval_at(self, time: float) -> tuple[float, float]:
        """Return the pointwise confidence interval of ``F_k`` at ``time``.

        Args:
            time: Finite evaluation time.

        Returns:
            ``(lower, upper)``; ``(0, 0)`` before the first event time.

        Raises:
            TypeError: If ``time`` is not real.
            ValueError: If ``time`` is not finite.

        Equation:
            [EQ-SURV-CIF-002].

        References:
            [REF-AALEN-JOHANSEN-1978].

        """
        point = require_finite_real(time, name="time")
        index = bisect.bisect_right(self.times, point)
        if index == 0:
            return 0.0, 0.0
        return self.lower[index - 1], self.upper[index - 1]


def _cause_table(
    times: Iterable[object], causes: Iterable[object]
) -> tuple[tuple[tuple[float, int, dict[int, int]], ...], tuple[int, ...], float, int]:
    """Validate competing-risks data and count events per distinct event time.

    Args:
        times: Finite non-negative observed times.
        causes: Non-negative integers, ``0`` for a censored time.

    Returns:
        ``(rows, labels, last_time, size)``: rows ``(t_j, n_j, {k: d_kj})``
        at the distinct times with at least one event, the sorted observed
        cause labels, the largest observed time, and the sample size.

    Raises:
        TypeError: If a time is not real or a cause is not an integer.
        ValueError: If the sample is empty, lengths differ, or a time or
            cause is outside its domain.

    References:
        [EQ-SURV-CIF-001]. No equation is evaluated.

    """
    observed = tuple(
        require_finite_real(value, name=f"times[{index}]")
        for index, value in enumerate(times)
    )
    labels = tuple(
        require_non_negative_integer(value, name=f"causes[{index}]")
        for index, value in enumerate(causes)
    )
    if not observed:
        raise ValueError("competing-risks data need at least one observation")
    if len(observed) != len(labels):
        raise ValueError("times and causes must have equal length")
    if any(value < 0.0 for value in observed):
        raise ValueError("observed times must be non-negative")
    order = sorted(zip(observed, labels, strict=True))
    rows: list[tuple[float, int, dict[int, int]]] = []
    remaining = len(order)
    index = 0
    while index < len(order):
        time = order[index][0]
        counts: dict[int, int] = {}
        leaving = 0
        while index < len(order) and order[index][0] == time:
            label = order[index][1]
            if label > 0:
                counts[label] = counts.get(label, 0) + 1
            leaving += 1
            index += 1
        if counts:
            rows.append((time, remaining, counts))
        remaining -= leaving
    present = tuple(sorted({label for label in labels if label > 0}))
    return tuple(rows), present, max(observed), len(observed)


def _incidence(
    rows: tuple[tuple[float, int, dict[int, int]], ...],
    cause: int,
    z: float,
    confidence: float,
    last_time: float,
    size: int,
) -> CumulativeIncidenceEstimate:
    """Compute the Aalen-Johansen curve and its variance for one cause.

    With ``g_j = d_j / (n_j (n_j - d_j))``,
    ``a_j = S(t_j-)^2 d_kj (n_j - d_kj) / n_j^3`` and
    ``b_j = S(t_j-) d_kj / n_j^2`` the variance at ``t`` is
    ``sum_{t_j <= t} [(F(t) - F(t_j))^2 g_j + a_j - 2 (F(t) - F(t_j)) b_j]``,
    accumulated with running sums of ``g``, ``F g``, ``F^2 g``, ``a``, ``b``
    and ``F b``.

    Args:
        rows: Output rows of :func:`_cause_table`.
        cause: Cause label.
        z: Two-sided normal critical value.
        confidence: Nominal level stored in the result.
        last_time: Largest observed time.
        size: Sample size.

    Returns:
        The estimate at every distinct all-cause event time.

    References:
        [EQ-SURV-CIF-001], [EQ-SURV-CIF-002], [REF-AALEN-JOHANSEN-1978].

    """
    survival = 1.0
    current = 0.0
    sum_g = sum_fg = sum_ffg = sum_a = sum_b = sum_fb = 0.0
    survivals: list[float] = []
    incidence: list[float] = []
    variance: list[float] = []
    for _, n, counts in rows:
        d = sum(counts.values())
        dk = counts.get(cause, 0)
        current += survival * dk / n
        a = survival * survival * dk * (n - dk) / n**3
        b = survival * dk / (n * n)
        sum_a += a
        # The term of the current time has F(t) - F(t_j) = 0, so only the
        # earlier times enter the g and b sums.
        value = (
            current * current * sum_g
            - 2.0 * current * sum_fg
            + sum_ffg
            + sum_a
            - 2.0 * (current * sum_b - sum_fb)
        )
        variance.append(max(0.0, value))
        if n > d:
            g = d / (n * (n - d))
            sum_g += g
            sum_fg += current * g
            sum_ffg += current * current * g
        sum_b += b
        sum_fb += current * b
        survival *= 1.0 - d / n
        survivals.append(max(0.0, survival))
        incidence.append(current)
    errors = [math.sqrt(value) for value in variance]
    return CumulativeIncidenceEstimate(
        cause,
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2].get(cause, 0) for row in rows),
        tuple(sum(row[2].values()) for row in rows),
        tuple(survivals),
        tuple(incidence),
        tuple(variance),
        tuple(max(0.0, f - z * e) for f, e in zip(incidence, errors, strict=True)),
        tuple(min(1.0, f + z * e) for f, e in zip(incidence, errors, strict=True)),
        confidence,
        last_time,
        size,
    )


def cumulative_incidence(
    times: Iterable[object],
    causes: Iterable[object],
    *,
    cause: int,
    confidence: float = 0.95,
) -> CumulativeIncidenceEstimate:
    """Estimate the cumulative incidence of one cause by Aalen-Johansen.

    ``F_k(t) = sum_{t_j <= t} S(t_j-) d_kj / n_j`` with the all-cause
    Kaplan-Meier ``S``. The variance comes from the delta method applied to
    the conditionally independent multinomial counts
    ``(d_kj, d_j - d_kj, n_j - d_j)`` of the risk sets (the same argument
    that gives Greenwood's formula):

    ``Var F_k(t) = sum_{t_j <= t} [ (F_k(t) - F_k(t_j))^2 d_j / (n_j (n_j -
    d_j)) + S(t_j-)^2 d_kj (n_j - d_kj) / n_j^3
    - 2 (F_k(t) - F_k(t_j)) S(t_j-) d_kj / n_j^2 ]``.

    Intervals are ``F +/- z se`` clipped to ``[0, 1]``; they are pointwise
    and rely on a large-sample normal approximation that is poor where
    ``F_k`` is close to zero or the risk set is small.

    Args:
        times: Finite non-negative observed times.
        causes: One non-negative integer per time: ``0`` for a right-censored
            time and ``k >= 1`` for an event of cause ``k``.
        cause: The cause ``k >= 1`` to estimate. A cause that never occurs
            has zero incidence.
        confidence: Pointwise confidence level in ``(0, 1)``.

    Returns:
        The curve at every distinct all-cause event time (empty when no
        event of any cause was observed).

    Raises:
        TypeError: If a time is not real or a cause is not an integer.
        ValueError: If the sample is empty, lengths differ, a time is
            negative, ``cause`` is smaller than one, or ``confidence`` is
            not strictly inside ``(0, 1)``.

    Equation:
        [EQ-SURV-CIF-001], [EQ-SURV-CIF-002].

    References:
        [REF-AALEN-JOHANSEN-1978], [REF-GREENWOOD-1926].

    """
    z = _z_critical(confidence)
    label = require_non_negative_integer(cause, name="cause")
    if label < 1:
        raise ValueError("cause must be a positive integer (0 marks censoring)")
    rows, _, last_time, size = _cause_table(times, causes)
    return _incidence(rows, label, z, float(confidence), last_time, size)


def cumulative_incidence_curves(
    times: Iterable[object], causes: Iterable[object], *, confidence: float = 0.95
) -> Mapping[int, CumulativeIncidenceEstimate]:
    """Estimate the cumulative incidence of every observed cause.

    The curves share one time grid and add up to the complement of the
    all-cause Kaplan-Meier curve, ``sum_k F_k(t_j) = 1 - S(t_j)``.

    Args:
        times: Finite non-negative observed times.
        causes: ``0`` for censored, ``k >= 1`` for an event of cause ``k``.
        confidence: Pointwise confidence level in ``(0, 1)``.

    Returns:
        Read-only mapping from each observed cause label, in increasing
        order, to its estimate (empty when no event was observed).

    Raises:
        TypeError: If a time is not real or a cause is not an integer.
        ValueError: If the sample is empty, lengths differ, a time is
            negative, or ``confidence`` is not strictly inside ``(0, 1)``.

    Equation:
        [EQ-SURV-CIF-001], [EQ-SURV-CIF-002].

    References:
        [REF-AALEN-JOHANSEN-1978].

    """
    z = _z_critical(confidence)
    rows, labels, last_time, size = _cause_table(times, causes)
    return MappingProxyType(
        {
            label: _incidence(rows, label, z, float(confidence), last_time, size)
            for label in labels
        }
    )
