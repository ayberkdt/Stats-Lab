"""Component time models: mission reliability and point availability.

Every model implements :class:`ComponentModel`:

* ``reliability(t)``: probability of no failure in ``[0, t]`` (mission
  reliability; for a repairable component, survival to the first failure);
* ``unreliability(t) = 1 - reliability(t)``, computed without cancellation;
* ``availability(t)``: probability that the component is up at time ``t``;
* ``unavailability(t) = 1 - availability(t)``, computed without cancellation.

For non-repairable models availability equals reliability. Times are
non-negative; ``t = math.inf`` is accepted and gives the limiting values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from probstat.foundations.validation import (
    require_non_negative_finite_real,
    require_probability,
)


def _require_time(t: object) -> float:
    """Validate a time ``t >= 0`` (``inf`` allowed).

    Raises:
        TypeError: If ``t`` is not real.
        ValueError: If ``t`` is negative or NaN.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(t, bool) or not isinstance(t, int | float):
        raise TypeError("time must be a real number")
    value = float(t)
    if math.isnan(value) or value < 0.0:
        raise ValueError("time must be non-negative")
    return value


def _require_positive(value: object, *, name: str) -> float:
    """Validate a finite positive real.

    Raises:
        ValueError: If the value is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    checked = require_non_negative_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _survival(rate: float, t: float) -> tuple[float, float]:
    """Return ``(exp(-rate t), 1 - exp(-rate t))`` without cancellation.

    References:
        [EQ-SYS-COMP-001], [REF-PYTHON-MATH].

    """
    if rate == 0.0:
        return 1.0, 0.0
    x = rate * t
    return math.exp(-x), -math.expm1(-x)


@runtime_checkable
class ComponentModel(Protocol):
    """Time-dependent reliability and availability of one component.

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-BARLOW-PROSCHAN-1975].

    """

    def reliability(self, t: float) -> float:
        """Return ``P(no failure in [0, t])``.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        ...

    def unreliability(self, t: float) -> float:
        """Return ``1 - reliability(t)``.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        ...

    def availability(self, t: float) -> float:
        """Return ``P(up at time t)``.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        ...

    def unavailability(self, t: float) -> float:
        """Return ``1 - availability(t)``.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        ...


@dataclass(frozen=True, slots=True)
class FixedProbability:
    """A time-independent component that works with probability ``p``.

    Args:
        p: Probability that the component works (its reliability), not its
            failure probability.

    References:
        [REF-BARLOW-PROSCHAN-1975].

    """

    p: float

    def __post_init__(self) -> None:
        """Validate ``p``.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        object.__setattr__(self, "p", require_probability(self.p, name="p"))

    def reliability(self, t: float) -> float:
        """Return ``p`` at every time.

        Args:
            t: Time (validated, unused).

        Returns:
            ``p``.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        _require_time(t)
        return self.p

    def unreliability(self, t: float) -> float:
        """Return ``1 - p``.

        Args:
            t: Time (validated, unused).

        Returns:
            ``1 - p``.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        _require_time(t)
        return 1.0 - self.p

    def availability(self, t: float) -> float:
        """Return ``p``.

        Args:
            t: Time (validated, unused).

        Returns:
            ``p``.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        return self.reliability(t)

    def unavailability(self, t: float) -> float:
        """Return ``1 - p``.

        Args:
            t: Time (validated, unused).

        Returns:
            ``1 - p``.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        return self.unreliability(t)


@dataclass(frozen=True, slots=True)
class DemandComponent:
    """A component that fails on demand with a fixed probability.

    Args:
        probability_per_demand: Failure probability per demand ``q``; the
            component works on the (single) demand with probability ``1 - q``
            whatever the time.

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """

    probability_per_demand: float

    def __post_init__(self) -> None:
        """Validate the demand failure probability.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        object.__setattr__(
            self,
            "probability_per_demand",
            require_probability(
                self.probability_per_demand, name="probability_per_demand"
            ),
        )

    def reliability(self, t: float) -> float:
        """Return ``1 - q``.

        Args:
            t: Time (validated, unused).

        Returns:
            Success probability on demand.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        _require_time(t)
        return 1.0 - self.probability_per_demand

    def unreliability(self, t: float) -> float:
        """Return ``q``.

        Args:
            t: Time (validated, unused).

        Returns:
            Failure probability on demand.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        _require_time(t)
        return self.probability_per_demand

    def availability(self, t: float) -> float:
        """Return ``1 - q``.

        Args:
            t: Time (validated, unused).

        Returns:
            Success probability on demand.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.reliability(t)

    def unavailability(self, t: float) -> float:
        """Return ``q``.

        Args:
            t: Time (validated, unused).

        Returns:
            Failure probability on demand.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.unreliability(t)


@dataclass(frozen=True, slots=True)
class ExponentialComponent:
    """A non-repairable component with constant failure rate ``lambda``.

    Args:
        failure_rate: ``lambda >= 0`` per unit time.

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-BARLOW-PROSCHAN-1975].

    """

    failure_rate: float

    def __post_init__(self) -> None:
        """Validate the rate.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(
            self,
            "failure_rate",
            require_non_negative_finite_real(self.failure_rate, name="failure_rate"),
        )

    def reliability(self, t: float) -> float:
        """Return ``exp(-lambda t)``.

        Args:
            t: Mission time.

        Returns:
            Mission reliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _survival(self.failure_rate, _require_time(t))[0]

    def unreliability(self, t: float) -> float:
        """Return ``-expm1(-lambda t)``.

        Args:
            t: Mission time.

        Returns:
            Mission unreliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _survival(self.failure_rate, _require_time(t))[1]

    def availability(self, t: float) -> float:
        """Return the reliability (no repair).

        Args:
            t: Time.

        Returns:
            Point availability.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.reliability(t)

    def unavailability(self, t: float) -> float:
        """Return the unreliability (no repair).

        Args:
            t: Time.

        Returns:
            Point unavailability.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.unreliability(t)


@dataclass(frozen=True, slots=True)
class WeibullComponent:
    """A non-repairable component with a two-parameter Weibull life.

    Args:
        shape: Shape ``beta > 0`` (``beta > 1``: wear-out).
        scale: Characteristic life ``eta > 0``.

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """

    shape: float
    scale: float

    def __post_init__(self) -> None:
        """Validate the parameters.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "shape", _require_positive(self.shape, name="shape"))
        object.__setattr__(self, "scale", _require_positive(self.scale, name="scale"))

    def _cumulative_hazard(self, t: float) -> float:
        """Return ``(t / eta)^beta``.

        References:
            [EQ-SYS-COMP-001]. No further equation is evaluated.

        """
        return float((_require_time(t) / self.scale) ** self.shape)

    def reliability(self, t: float) -> float:
        """Return ``exp(-(t / eta)^beta)``.

        Args:
            t: Mission time.

        Returns:
            Mission reliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return math.exp(-self._cumulative_hazard(t))

    def unreliability(self, t: float) -> float:
        """Return ``-expm1(-(t / eta)^beta)``.

        Args:
            t: Mission time.

        Returns:
            Mission unreliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return -math.expm1(-self._cumulative_hazard(t))

    def availability(self, t: float) -> float:
        """Return the reliability (no repair).

        Args:
            t: Time.

        Returns:
            Point availability.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.reliability(t)

    def unavailability(self, t: float) -> float:
        """Return the unreliability (no repair).

        Args:
            t: Time.

        Returns:
            Point unavailability.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.unreliability(t)


@dataclass(frozen=True, slots=True)
class LifeDistributionComponent:
    """A non-repairable component whose life follows any continuous law.

    Args:
        distribution: A law with ``sf(t)`` (and preferably ``cdf(t)``) on
            ``[0, inf)``, e.g. ``probstat.distributions.LogNormal``. The law
            is assumed proper, so ``reliability(inf) = 0``.

    Raises:
        TypeError: If the law has no ``sf`` method.

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """

    distribution: object

    def __post_init__(self) -> None:
        """Check the structural interface.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not callable(getattr(self.distribution, "sf", None)):
            raise TypeError("distribution must provide an sf(t) method")

    def reliability(self, t: float) -> float:
        """Return ``sf(t)``.

        Args:
            t: Mission time.

        Returns:
            Mission reliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        value = _require_time(t)
        if math.isinf(value):
            return 0.0
        return require_probability(
            self.distribution.sf(value),  # type: ignore[attr-defined]
            name="sf(t)",
        )

    def unreliability(self, t: float) -> float:
        """Return ``cdf(t)`` (or ``1 - sf(t)`` if the law has no ``cdf``).

        Args:
            t: Mission time.

        Returns:
            Mission unreliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        value = _require_time(t)
        if math.isinf(value):
            return 1.0
        cdf = getattr(self.distribution, "cdf", None)
        if callable(cdf):
            return require_probability(cdf(value), name="cdf(t)")
        return 1.0 - self.reliability(value)

    def availability(self, t: float) -> float:
        """Return the reliability (no repair).

        Args:
            t: Time.

        Returns:
            Point availability.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.reliability(t)

    def unavailability(self, t: float) -> float:
        """Return the unreliability (no repair).

        Args:
            t: Time.

        Returns:
            Point unavailability.

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.unreliability(t)


@dataclass(frozen=True, slots=True)
class RepairableComponent:
    """A component with exponential failure and repair times (up at ``t=0``).

    The two-state Markov model gives the point availability
    ``A(t) = mu/(lambda + mu) + lambda/(lambda + mu) exp(-(lambda + mu) t)``.
    Mission reliability is survival to the first failure, ``exp(-lambda t)``.

    Args:
        failure_rate: ``lambda >= 0``.
        repair_rate: ``mu > 0`` (mean time to repair ``1/mu``).

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-BARLOW-PROSCHAN-1975].

    """

    failure_rate: float
    repair_rate: float

    def __post_init__(self) -> None:
        """Validate the rates.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(
            self,
            "failure_rate",
            require_non_negative_finite_real(self.failure_rate, name="failure_rate"),
        )
        object.__setattr__(
            self, "repair_rate", _require_positive(self.repair_rate, name="repair_rate")
        )

    @property
    def steady_state_availability(self) -> float:
        """Return ``mu / (lambda + mu)``.

        Equation:
            [EQ-SYS-COMP-002].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.repair_rate / (self.failure_rate + self.repair_rate)

    def reliability(self, t: float) -> float:
        """Return ``exp(-lambda t)`` (no failure before ``t``).

        Args:
            t: Mission time.

        Returns:
            Mission reliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _survival(self.failure_rate, _require_time(t))[0]

    def unreliability(self, t: float) -> float:
        """Return ``1 - exp(-lambda t)``.

        Args:
            t: Mission time.

        Returns:
            Mission unreliability.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _survival(self.failure_rate, _require_time(t))[1]

    def unavailability(self, t: float) -> float:
        """Return ``lambda/(lambda + mu) (1 - exp(-(lambda + mu) t))``.

        Args:
            t: Time.

        Returns:
            Point unavailability.

        Equation:
            [EQ-SYS-COMP-002].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        total = self.failure_rate + self.repair_rate
        transient = _survival(total, _require_time(t))[1]
        return self.failure_rate / total * transient

    def availability(self, t: float) -> float:
        """Return ``mu/(lambda + mu) + lambda/(lambda + mu) exp(-(lambda + mu) t)``.

        Args:
            t: Time.

        Returns:
            Point availability.

        Equation:
            [EQ-SYS-COMP-002].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        total = self.failure_rate + self.repair_rate
        decay = _survival(total, _require_time(t))[0]
        return (self.repair_rate + self.failure_rate * decay) / total


def _mean_unavailability_factor(x: float) -> float:
    """Return ``1 - (1 - exp(-x)) / x`` without cancellation.

    Uses the alternating series ``sum_{k>=1} (-1)^(k+1) x^k / (k+1)!`` for
    ``x < 0.5`` and the closed form otherwise.

    References:
        [EQ-SYS-COMP-003], [REF-PYTHON-MATH].

    """
    if x == 0.0:
        return 0.0
    if x < 0.5:
        total = 0.0
        term = 1.0
        for k in range(1, 30):
            term *= x / (k + 1)
            total += term if k % 2 else -term
        return total
    return 1.0 + math.expm1(-x) / x


@dataclass(frozen=True, slots=True)
class PeriodicallyTestedComponent:
    """A standby component with hidden failures revealed by periodic tests.

    Failures occur at constant rate ``lambda`` and stay hidden until the
    next proof test. Tests are perfect and instantaneous and a failed unit is
    restored at once to as-good-as-new; tests start at ``first_test`` and
    repeat every ``test_interval`` (``first_test`` lets the tests of
    redundant units be staggered). Mission reliability ``exp(-lambda t)`` is
    the probability of no failure in ``[0, t]``.

    The point unavailability is ``1 - exp(-lambda a(t))`` with ``a(t)`` the
    time since the last test (or since ``0`` before the first test). The
    exact average over a test interval is
    ``1 - (1 - exp(-lambda tau)) / (lambda tau)``; the common approximation
    ``lambda tau / 2`` overstates it by about ``(lambda tau)^2 / 6``.

    Args:
        failure_rate: ``lambda >= 0``.
        test_interval: ``tau > 0``.
        first_test: Time of the first test (default ``tau``).

    References:
        [REF-RAUSAND-HOYLAND-2004].

    """

    failure_rate: float
    test_interval: float
    first_test: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameters.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(
            self,
            "failure_rate",
            require_non_negative_finite_real(self.failure_rate, name="failure_rate"),
        )
        object.__setattr__(
            self,
            "test_interval",
            _require_positive(self.test_interval, name="test_interval"),
        )
        if self.first_test is not None:
            object.__setattr__(
                self,
                "first_test",
                _require_positive(self.first_test, name="first_test"),
            )

    def _age(self, t: float) -> float:
        """Return the time since the last test (or since 0).

        References:
            [EQ-SYS-COMP-003]. No further equation is evaluated.

        """
        first = self.test_interval if self.first_test is None else self.first_test
        if t < first:
            return t
        return math.fmod(t - first, self.test_interval)

    def reliability(self, t: float) -> float:
        """Return ``exp(-lambda t)``.

        Args:
            t: Mission time.

        Returns:
            Probability of no failure in ``[0, t]``.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _survival(self.failure_rate, _require_time(t))[0]

    def unreliability(self, t: float) -> float:
        """Return ``1 - exp(-lambda t)``.

        Args:
            t: Mission time.

        Returns:
            Probability of at least one failure in ``[0, t]``.

        Equation:
            [EQ-SYS-COMP-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _survival(self.failure_rate, _require_time(t))[1]

    def unavailability(self, t: float) -> float:
        """Return ``1 - exp(-lambda a(t))``.

        Args:
            t: Finite time.

        Returns:
            Point unavailability (probability of a hidden failure).

        Raises:
            ValueError: If ``t`` is infinite (the sawtooth has no limit).

        Equation:
            [EQ-SYS-COMP-003].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        value = _require_time(t)
        if math.isinf(value):
            raise ValueError("the availability of a tested component has no limit")
        return _survival(self.failure_rate, self._age(value))[1]

    def availability(self, t: float) -> float:
        """Return ``exp(-lambda a(t))``.

        Args:
            t: Finite time.

        Returns:
            Point availability.

        Equation:
            [EQ-SYS-COMP-003].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        value = _require_time(t)
        if math.isinf(value):
            raise ValueError("the availability of a tested component has no limit")
        return _survival(self.failure_rate, self._age(value))[0]

    def average_unavailability(self) -> float:
        """Return the exact mean unavailability over one test interval.

        Returns:
            ``1 - (1 - exp(-lambda tau)) / (lambda tau)``.

        Equation:
            [EQ-SYS-COMP-003].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return _mean_unavailability_factor(self.failure_rate * self.test_interval)

    def average_unavailability_approximation(self) -> float:
        """Return the textbook approximation ``lambda tau / 2``.

        It exceeds the exact value; the relative error is about
        ``lambda tau / 3`` for small ``lambda tau``.

        Returns:
            ``lambda tau / 2``.

        Equation:
            [EQ-SYS-COMP-003].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return 0.5 * self.failure_rate * self.test_interval
