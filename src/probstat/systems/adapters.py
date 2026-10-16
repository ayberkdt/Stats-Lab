"""Hierarchical composition: subsystems used as components of a larger system.

A mission system is usually analysed bottom-up: a repairable pump train is a
continuous-time Markov model, an avionics string is itself a reliability
structure, and both are blocks of the mission-level structure. The adapters
in this module expose such subsystems through the
:class:`~probstat.systems.ComponentModel` protocol so that
:class:`~probstat.systems.SystemModel` evaluates the mission exactly.

Exactness condition: the composed result is exact when the subsystem's
internal components appear nowhere else in the larger structure, so the
subsystem's state is independent of the other blocks (modular decomposition,
Birnbaum, Esary and Saunders 1961; Barlow and Proschan 1975, Chapter 2). A
component shared between two subsystems must instead be modelled once, in a
single flat structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .markov import MarkovReliabilityModel
from .system import SystemModel


def _require_time(t: object) -> float:
    """Return a validated non-negative finite time.

    Raises:
        TypeError: If ``t`` is not a real number.
        ValueError: If ``t`` is negative or not finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(t, bool) or not isinstance(t, int | float):
        raise TypeError("t must be a real number")
    value = float(t)
    if not value >= 0.0 or value == float("inf"):
        raise ValueError("t must be finite and non-negative")
    return value


def _clip(value: float) -> float:
    """Clamp a probability computed with rounding error into ``[0, 1]``.

    References:
        Project validation policy. No equation is evaluated.

    """
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class MarkovComponent:
    """A continuous-time Markov subsystem used as one block of a system.

    ``reliability(t)`` is the probability that the subsystem has not entered
    a down state in ``[0, t]`` (down states made absorbing) and
    ``availability(t)`` the probability that it is up at ``t``; both come
    from the uniformization solution of :class:`MarkovReliabilityModel`. The
    truncation error bound of the last evaluation is kept in
    :attr:`last_error_bound` so it is never lost silently.

    Args:
        model: Markov reliability model of the subsystem.
        tolerance: Absolute truncation tolerance passed to the solver.

    Equation:
        [EQ-SYS-MKV-001], [EQ-SYS-ADAPT-001].

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-BARLOW-PROSCHAN-1975].

    """

    model: MarkovReliabilityModel
    tolerance: float = 1e-12
    _errors: dict[str, float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the wrapped model and the tolerance.

        Raises:
            TypeError: If ``model`` is not a :class:`MarkovReliabilityModel`.
            ValueError: If the tolerance is not in ``(0, 1e-3]``.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.model, MarkovReliabilityModel):
            raise TypeError("model must be a MarkovReliabilityModel")
        if isinstance(self.tolerance, bool) or not isinstance(
            self.tolerance, int | float
        ):
            raise TypeError("tolerance must be a real number")
        if not 0.0 < float(self.tolerance) <= 1e-3:
            raise ValueError("tolerance must lie in (0, 1e-3]")

    @property
    def last_error_bound(self) -> float:
        """Return the largest truncation bound seen so far (absolute).

        References:
            [REF-RAUSAND-HOYLAND-2004]. No equation is evaluated.

        """
        return max(self._errors.values(), default=0.0)

    def _record(self, key: str, value: object) -> float:
        """Store the error bound of a bounded value and return its value.

        References:
            Project validation policy. No equation is evaluated.

        """
        bound = float(getattr(value, "error_bound", 0.0))
        self._errors[key] = max(self._errors.get(key, 0.0), bound)
        return _clip(float(value))  # type: ignore[arg-type]

    def reliability(self, t: float) -> float:
        """Return ``P(no subsystem failure in [0, t])``.

        Args:
            t: Non-negative time.

        Returns:
            Subsystem reliability.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        time = _require_time(t)
        return self._record(
            "reliability", self.model.reliability(time, tolerance=self.tolerance)
        )

    def unreliability(self, t: float) -> float:
        """Return ``1 - reliability(t)`` computed without cancellation.

        Args:
            t: Non-negative time.

        Returns:
            Subsystem unreliability.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        time = _require_time(t)
        return self._record(
            "unreliability", self.model.unreliability(time, tolerance=self.tolerance)
        )

    def availability(self, t: float) -> float:
        """Return ``P(subsystem up at t)``.

        Args:
            t: Non-negative time.

        Returns:
            Point availability.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        time = _require_time(t)
        return self._record(
            "availability", self.model.availability(time, tolerance=self.tolerance)
        )

    def unavailability(self, t: float) -> float:
        """Return ``1 - availability(t)`` computed without cancellation.

        Args:
            t: Non-negative time.

        Returns:
            Point unavailability.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        time = _require_time(t)
        return self._record(
            "unavailability",
            self.model.unavailability(time, tolerance=self.tolerance),
        )


@dataclass(frozen=True)
class SubsystemComponent:
    """A :class:`SystemModel` used as one block of a larger system.

    The subsystem's exact reliability and availability (its own BDD,
    component models, and CCF groups) become the block's measures, so a
    mission can be described as a hierarchy of subsystems.

    Args:
        system: The subsystem model.

    Equation:
        [EQ-SYS-ADAPT-001].

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUSAND-HOYLAND-2004].

    """

    system: SystemModel

    def __post_init__(self) -> None:
        """Validate the wrapped subsystem.

        Raises:
            TypeError: If ``system`` is not a :class:`SystemModel`.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.system, SystemModel):
            raise TypeError("system must be a SystemModel")

    def reliability(self, t: float) -> float:
        """Return the subsystem's exact mission reliability.

        Args:
            t: Non-negative time.

        Returns:
            ``R_sub(t)``.

        Equation:
            [EQ-SYS-ADAPT-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.system.reliability(_require_time(t))

    def unreliability(self, t: float) -> float:
        """Return the subsystem's exact mission unreliability.

        Args:
            t: Non-negative time.

        Returns:
            ``1 - R_sub(t)``.

        Equation:
            [EQ-SYS-ADAPT-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.system.unreliability(_require_time(t))

    def availability(self, t: float) -> float:
        """Return the subsystem's exact point availability.

        Args:
            t: Non-negative time.

        Returns:
            ``A_sub(t)``.

        Equation:
            [EQ-SYS-ADAPT-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.system.availability(_require_time(t))

    def unavailability(self, t: float) -> float:
        """Return the subsystem's exact point unavailability.

        Args:
            t: Non-negative time.

        Returns:
            ``1 - A_sub(t)``.

        Equation:
            [EQ-SYS-ADAPT-001].

        References:
            [REF-RAUSAND-HOYLAND-2004].

        """
        return self.system.unavailability(_require_time(t))
