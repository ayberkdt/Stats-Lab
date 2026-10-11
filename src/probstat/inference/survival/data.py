"""Right-censored survival data and the risk-set table shared by every estimator.

An observation is a pair ``(t_i, delta_i)``: ``t_i`` is the observed time and
``delta_i = 1`` when the event was seen at ``t_i`` while ``delta_i = 0`` when
the subject was right-censored (still event-free when follow-up ended). The
standard convention for ties is used throughout the survival package: a
subject censored at ``t`` is still at risk for an event at ``t``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real


def _event_flag(value: object, index: int) -> bool:
    """Validate one event indicator.

    Args:
        value: ``True``/``False`` or the integers ``1``/``0``.
        index: Position used in error messages.

    Returns:
        ``True`` for an observed event, ``False`` for a censored time.

    Raises:
        TypeError: If the indicator is not a boolean or an integer.
        ValueError: If an integer indicator is not ``0`` or ``1``.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 3]. No equation is evaluated.

    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value not in (0, 1):
            raise ValueError(f"events[{index}] must be 0 or 1")
        return value == 1
    raise TypeError(f"events[{index}] must be a bool or the integer 0/1")


@dataclass(frozen=True, slots=True)
class RiskSetRow:
    """Counts at one distinct observed time.

    Args:
        time: Distinct observed time ``t_j``.
        at_risk: ``n_j``, the number of subjects with observed time ``>= t_j``.
        events: ``d_j``, the number of events at ``t_j``.
        censored: ``c_j``, the number censored at ``t_j``.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 4].

    """

    time: float
    at_risk: int
    events: int
    censored: int


@dataclass(frozen=True, slots=True)
class SurvivalData:
    """A right-censored sample ``(t_i, delta_i)``.

    Args:
        times: Finite non-negative observed times.
        events: Event indicators (``True``/``1`` event, ``False``/``0``
            right-censored), one per time.

    Raises:
        TypeError: If a time or indicator has the wrong type.
        ValueError: If the sample is empty, lengths differ, or a time is
            negative or non-finite.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 3],
        [REF-KAPLAN-MEIER-1958].

    """

    times: tuple[float, ...]
    events: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate and normalize times and indicators.

        Raises:
            TypeError: If a time or indicator has the wrong type.
            ValueError: If the sample is empty, lengths differ, or a time is
                negative or non-finite.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 3]. No equation is evaluated.

        """
        times = tuple(
            require_finite_real(value, name=f"times[{index}]")
            for index, value in enumerate(self.times)
        )
        events = tuple(
            _event_flag(value, index) for index, value in enumerate(self.events)
        )
        if not times:
            raise ValueError("survival data need at least one observation")
        if len(times) != len(events):
            raise ValueError("times and events must have equal length")
        if any(value < 0.0 for value in times):
            raise ValueError("observed times must be non-negative")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "events", events)

    @property
    def size(self) -> int:
        """Return the number of subjects ``n``.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 3]. No equation is evaluated.

        """
        return len(self.times)

    @property
    def event_count(self) -> int:
        """Return the number of observed events ``sum_i delta_i``.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 3]. No equation is evaluated.

        """
        return sum(1 for flag in self.events if flag)

    @property
    def censored_count(self) -> int:
        """Return the number of right-censored subjects.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 3]. No equation is evaluated.

        """
        return self.size - self.event_count

    def risk_table(self) -> tuple[RiskSetRow, ...]:
        """Return the risk-set counts at every distinct observed time.

        Subjects censored at ``t`` are counted in the risk set at ``t``.

        Returns:
            Rows in increasing time order.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4], [REF-KAPLAN-MEIER-1958].

        """
        order = sorted(zip(self.times, self.events, strict=True))
        rows: list[RiskSetRow] = []
        remaining = len(order)
        index = 0
        while index < len(order):
            time = order[index][0]
            events = 0
            censored = 0
            while index < len(order) and order[index][0] == time:
                if order[index][1]:
                    events += 1
                else:
                    censored += 1
                index += 1
            rows.append(RiskSetRow(time, remaining, events, censored))
            remaining -= events + censored
        return tuple(rows)

    def event_table(self) -> tuple[RiskSetRow, ...]:
        """Return only the rows with at least one event (``d_j > 0``).

        Returns:
            Rows at distinct event times in increasing order.

        References:
            [REF-KLEIN-MOESCHBERGER-2003, Chapter 4]. No equation is evaluated.

        """
        return tuple(row for row in self.risk_table() if row.events > 0)


def survival_data(times: Iterable[object], events: Iterable[object]) -> SurvivalData:
    """Build validated right-censored survival data.

    Args:
        times: Finite non-negative observed times.
        events: Event indicators (``1``/``True`` event, ``0``/``False``
            censored).

    Returns:
        Validated :class:`SurvivalData`.

    Raises:
        TypeError: If a value has the wrong type.
        ValueError: If lengths differ, the sample is empty, or a time is
            negative.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 3]. No equation is evaluated.

    """
    return SurvivalData(tuple(times), tuple(events))  # type: ignore[arg-type]


def require_positive_times(data: SurvivalData) -> None:
    """Reject zero times for likelihoods defined on ``log t``.

    Args:
        data: Survival data.

    Raises:
        ValueError: If any observed time is zero.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 12]. No equation is evaluated.

    """
    if any(not time > 0.0 for time in data.times):
        raise ValueError("parametric survival models need strictly positive times")


def covariate_rows(
    covariates: Sequence[Sequence[float]] | None, size: int, *, name: str
) -> tuple[tuple[float, ...], ...]:
    """Validate a subject-by-covariate matrix.

    Args:
        covariates: One row per subject, or ``None`` for no covariates.
        size: Required number of rows.
        name: Argument name used in errors.

    Returns:
        Rows of finite floats with a common length (empty rows when
        ``covariates`` is ``None``).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the row count or row lengths disagree, or an entry is
            not finite.

    References:
        [REF-KLEIN-MOESCHBERGER-2003, Chapter 8]. No equation is evaluated.

    """
    if covariates is None:
        return tuple(() for _ in range(size))
    rows = tuple(
        tuple(
            require_finite_real(value, name=f"{name}[{i}][{j}]")
            for j, value in enumerate(row)
        )
        for i, row in enumerate(covariates)
    )
    if len(rows) != size:
        raise ValueError(f"{name} must have one row per subject")
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} rows must have equal length")
    return rows
