"""Private bookkeeping helpers shared by the study stage modules.

The running-stage context, the constant-memory accumulators of sequential
propagation, and the small formatting and validation helpers used by
several stage modules.

References:
    [REF-SMITH-UQ-2024, Chapter 1], [REF-WELFORD-1962].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from probstat.design import Condition
from probstat.reliability import FORMResult
from probstat.risk import ReturnLevel
from probstat.statistics import ExceedanceCounter, P2Quantile, RunningMoments


def _orient_return_level(level: ReturnLevel, sign: float) -> ReturnLevel:
    """Map a return level of ``sign * Y`` back to QoI units.

    References:
        [REF-COLES-2001, Section 4.3]. No equation is evaluated.

    """
    if sign > 0.0:
        return level
    return ReturnLevel(
        level.return_period,
        -level.level,
        level.standard_error,
        -level.upper,
        -level.lower,
        level.confidence,
        level.method,
    )


def _format_settings(settings: Mapping[str, object]) -> str:
    """Render stage settings as ``name=value`` pairs.

    References:
        Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    parts = []
    for key, value in settings.items():
        if isinstance(value, float):
            text = f"{value:g}"
        elif isinstance(value, tuple):
            text = "(" + ", ".join(str(item) for item in value) + ")"
        else:
            text = str(value)
        parts.append(f"{key}={text}")
    return "; ".join(parts)


def _converged_text(result: FORMResult) -> str:
    """Return ``"yes"`` or ``"no (<termination reason>)"`` for a FORM search.

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5]. No equation is evaluated.

    """
    return "yes" if result.converged else f"no ({result.message})"


def _format_condition(condition: Condition) -> str:
    """Render a condition mapping as ``name=value`` pairs.

    References:
        Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    if not condition:
        return "(model defaults)"
    return ", ".join(f"{name}={value:g}" for name, value in condition.items())


def _sized(value: object) -> Sequence[Any]:
    """Return ``value`` if it is a list or tuple, else an empty tuple.

    Cost estimates must not consume a generator argument before the stage
    method receives it; an unsized argument simply gives no up-front cost.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    return value if isinstance(value, tuple | list) else ()


def _setting_value(value: object) -> object:
    """Return a compact, JSON-safe description of one stage argument.

    Scalars and short tuples of scalars are kept; other objects are named by
    type (and length for collections), so settings never embed large data.

    References:
        Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list) and len(value) <= 20:
        if all(
            item is None or isinstance(item, bool | int | float | str)
            for item in value
        ):
            return tuple(value)
    if isinstance(value, tuple | list):
        kinds = sorted({type(item).__name__ for item in value})
        return f"{len(value)} x {'/'.join(kinds) or 'item'}"
    if isinstance(value, Mapping) and len(value) <= 20:
        if all(isinstance(item, bool | int | float | str) for item in value.values()):
            return {str(key): item for key, item in value.items()}
    return f"<{type(value).__name__}>"


@dataclass
class _StageContext:
    """Bookkeeping of the stage that is currently running.

    References:
        [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

    """

    stage: str
    seed: int
    settings: dict[str, object]
    failed_before: int
    hits_before: int
    warnings: list[str] = field(default_factory=list)
    recorded: bool = False


class _Accumulators:
    """Constant-memory accumulators of one QoI for sequential propagation.

    References:
        [REF-WELFORD-1962], [REF-JAIN-CHLAMTAC-1985]. No equation is evaluated.

    """

    def __init__(
        self, levels: tuple[float, ...], event: Callable[[float], bool] | None
    ) -> None:
        """Create empty accumulators.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        self.moments = RunningMoments()
        self.markers = tuple(P2Quantile(p) for p in levels)
        self.event = event
        self.counter = None if event is None else ExceedanceCounter()

    @property
    def count(self) -> int:
        """Return the number of pushed outputs.

        References:
            [REF-WELFORD-1962]. No equation is evaluated.

        """
        return self.moments.count

    def push(self, value: float) -> None:
        """Push one model output into every accumulator.

        References:
            [REF-WELFORD-1962], [REF-JAIN-CHLAMTAC-1985]. Updates
            [EQ-STAT-STREAM-005] markers.

        """
        self.moments.push(value)
        for marker in self.markers:
            marker.push(value)
        if self.counter is not None and self.event is not None:
            self.counter.push(self.event(value))

    def relative_error(self) -> float:
        """Return the relative standard error of the controlled estimate.

        The mean uses ``s / (sqrt(n) |mean|)``; the probability uses
        ``sqrt((1 - p) / (n p))`` with ``inf`` before the first event and the
        half-count-adjusted proportion when every draw is an event.

        Equation:
            [EQ-STAT-STREAM-007], [EQ-MC-PROB-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4].

        """
        if self.moments.count < 2:
            return math.inf
        counter = self.counter
        if counter is None:
            return self.moments.relative_standard_error()
        if counter.events == 0:
            return math.inf
        p = counter.proportion
        if counter.events == counter.trials:
            # All draws are events: (1 - p) / (n p) would be exactly zero.
            p = (counter.events + 0.5) / (counter.trials + 1.0)
        return math.sqrt((1.0 - p) / (counter.trials * p))

    def quantiles(self) -> tuple[tuple[float, float], ...]:
        """Return ``(level, P² estimate)`` pairs.

        References:
            [REF-JAIN-CHLAMTAC-1985]. Reads [EQ-STAT-STREAM-005] estimates.

        """
        return tuple((marker.probability, marker.estimate) for marker in self.markers)
