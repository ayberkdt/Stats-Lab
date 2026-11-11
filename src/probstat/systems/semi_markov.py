"""Semi-Markov reliability and availability models.

A :class:`SemiMarkovReliabilityModel` is a semi-Markov process whose states
are split into *up* and *down* states, with an initial law on the up states.
Sojourn times may follow any law (Weibull wear-out, lognormal or
deterministic-like repair), which removes the exponential restriction of
:class:`~probstat.systems.MarkovReliabilityModel` (Limnios and Oprişan 2001,
Chapter 5) [EQ-SYS-SMP-001]:

- point availability ``A(t) = P(Z_t in U)`` and unavailability, each solved
  directly from the Markov renewal equation (no ``1 - A`` cancellation);
- reliability ``R(t) = P(no entry into D in [0, t])`` from the process with
  the down states made absorbing;
- the mean time to failure, exact from the mean sojourn times and the
  fundamental matrix of the jump chain;
- the steady-state availability ``sum_(j in U) nu_j tau_j / sum_k nu_k
  tau_k``, which for an alternating renewal unit is ``E[U] / (E[U] + E[D])``.

Time-dependent measures carry the Richardson error estimate of the renewal
solver (:class:`~probstat.processes.SemiMarkovEstimate`); with exponential
sojourns they reproduce the CTMC results.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable

from probstat.processes.semi_markov import (
    RenewalCurve,
    SemiMarkovEstimate,
    SemiMarkovProcess,
)

State = Hashable
_DEFAULT_STEPS = 256


class SemiMarkovReliabilityModel:
    """A semi-Markov process with up/down states.

    Args:
        process: The system's semi-Markov process.
        up_states: Labels of the working states; every other state is down.
            Both sets must be non-empty.
        initial: Initial law (label, mapping or vector) on the up states.
        steps: Default number of coarse grid steps of the renewal solver.

    Raises:
        TypeError: If ``process`` is not a :class:`SemiMarkovProcess`.
        ValueError: If the partition or the initial law is invalid.

    Equation:
        [EQ-SYS-SMP-001].

    References:
        [REF-LIMNIOS-OPRISAN-2001, Chapter 5], [REF-RAUSAND-HOYLAND-2004].

    """

    def __init__(
        self,
        process: SemiMarkovProcess,
        up_states: Iterable[State],
        initial: object,
        *,
        steps: int = _DEFAULT_STEPS,
    ) -> None:
        """Validate the state partition and the initial law.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        if not isinstance(process, SemiMarkovProcess):
            raise TypeError("process must be a SemiMarkovProcess")
        chosen = {process.index(state) for state in up_states}
        if not chosen:
            raise ValueError("up_states must be non-empty")
        up = tuple(s for i, s in enumerate(process.states) if i in chosen)
        down = tuple(s for i, s in enumerate(process.states) if i not in chosen)
        if not down:
            raise ValueError("at least one state must be a down state")
        start = process.distribution(initial)
        if any(start[process.index(state)] > 0.0 for state in down):
            raise ValueError("the initial law must put all its mass on up states")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("steps must be an integer of at least 2")
        self._process = process
        self._up = up
        self._down = down
        self._initial = start
        self._absorbed = process.make_absorbing(down)
        self._steps = steps

    @property
    def process(self) -> SemiMarkovProcess:
        """Return the semi-Markov process.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5]. No equation is evaluated.

        """
        return self._process

    @property
    def up_states(self) -> tuple[State, ...]:
        """Return the up states.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5]. No equation is evaluated.

        """
        return self._up

    @property
    def down_states(self) -> tuple[State, ...]:
        """Return the down states.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5]. No equation is evaluated.

        """
        return self._down

    @property
    def initial_distribution(self) -> tuple[float, ...]:
        """Return the initial law in state order.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return self._initial

    @property
    def reliability_process(self) -> SemiMarkovProcess:
        """Return the process with every down state absorbing.

        Equation:
            [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._absorbed

    def _steps_or_default(self, steps: int | None) -> int:
        """Return the requested or default step count.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return self._steps if steps is None else steps

    def availability(self, t: float, *, steps: int | None = None) -> SemiMarkovEstimate:
        """Return ``A(t) = P(Z_t in U)``.

        Args:
            t: Non-negative time.
            steps: Coarse grid steps (default: the model's).

        Returns:
            Availability with its error estimate.

        Equation:
            [EQ-SYS-SMP-001], [EQ-PROC-SMP-002].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._process.probability(
            self._initial, self._up, t, steps=self._steps_or_default(steps)
        )

    def unavailability(
        self, t: float, *, steps: int | None = None
    ) -> SemiMarkovEstimate:
        """Return ``1 - A(t) = P(Z_t in D)`` solved directly.

        Args:
            t: Non-negative time.
            steps: Coarse grid steps.

        Returns:
            Unavailability with its error estimate.

        Equation:
            [EQ-SYS-SMP-001], [EQ-PROC-SMP-002].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._process.probability(
            self._initial, self._down, t, steps=self._steps_or_default(steps)
        )

    def availability_curve(
        self, horizon: float, *, steps: int | None = None
    ) -> RenewalCurve:
        """Return ``A(t)`` on the uniform grid of ``[0, horizon]``.

        Args:
            horizon: Positive final time.
            steps: Coarse grid steps.

        Returns:
            Availability curve.

        Equation:
            [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._process.probability_curve(
            self._initial, self._up, horizon, steps=self._steps_or_default(steps)
        )

    def reliability(self, t: float, *, steps: int | None = None) -> SemiMarkovEstimate:
        """Return ``R(t) = P(no failure in [0, t])``.

        Args:
            t: Non-negative time.
            steps: Coarse grid steps.

        Returns:
            Reliability with its error estimate.

        Equation:
            [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._absorbed.probability(
            self._initial, self._up, t, steps=self._steps_or_default(steps)
        )

    def unreliability(
        self, t: float, *, steps: int | None = None
    ) -> SemiMarkovEstimate:
        """Return ``1 - R(t)``, the first-passage CDF into the down states.

        Args:
            t: Non-negative time.
            steps: Coarse grid steps.

        Returns:
            Unreliability with its error estimate.

        Equation:
            [EQ-SYS-SMP-001], [EQ-PROC-SMP-002].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._absorbed.probability(
            self._initial, self._down, t, steps=self._steps_or_default(steps)
        )

    def reliability_curve(
        self, horizon: float, *, steps: int | None = None
    ) -> RenewalCurve:
        """Return ``R(t)`` on the uniform grid of ``[0, horizon]``.

        Args:
            horizon: Positive final time.
            steps: Coarse grid steps.

        Returns:
            Reliability curve.

        Equation:
            [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._absorbed.probability_curve(
            self._initial, self._up, horizon, steps=self._steps_or_default(steps)
        )

    def mean_time_to_failure(self) -> float:
        """Return the exact MTTF ``alpha N tau`` with the down states absorbing.

        Returns ``inf`` when a set of up states that is never left is
        reachable.

        Returns:
            Mean time to failure.

        Equation:
            [EQ-PROC-SMP-004], [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        return self._process.mean_first_passage_time(self._down, self._initial)

    def steady_state_availability(self) -> float:
        """Return ``lim A(t) = sum_(j in U) pi_j`` from the limiting law.

        Returns:
            Limiting availability.

        Raises:
            ValueError: If the jump chain has several closed classes.

        Equation:
            [EQ-PROC-SMP-005], [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        law = self._process.steady_state()
        return math.fsum(law[self._process.index(state)] for state in self._up)

    def steady_state_unavailability(self) -> float:
        """Return ``lim (1 - A(t)) = sum_(j in D) pi_j`` (direct sum).

        Returns:
            Limiting unavailability.

        Equation:
            [EQ-PROC-SMP-005], [EQ-SYS-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        law = self._process.steady_state()
        return math.fsum(law[self._process.index(state)] for state in self._down)


def alternating_renewal_unit(
    up_time: object, down_time: object, *, steps: int = _DEFAULT_STEPS
) -> SemiMarkovReliabilityModel:
    """Return a repairable unit with general up and down times.

    The two-state process ``"up" -> "down" -> "up"`` alternates independent
    up times ``U ~ up_time`` (for example Weibull) and repair times
    ``D ~ down_time`` (for example lognormal). ``R(t) = P(U > t)`` and
    ``A_inf = E[U] / (E[U] + E[D])`` [EQ-SYS-SMP-001].

    Args:
        up_time: Law of the time to failure (``cdf``, ``mean``; ``quantile``
            for simulation).
        down_time: Law of the repair time.
        steps: Default coarse grid steps of the renewal solver.

    Returns:
        Model starting in ``"up"``.

    Raises:
        TypeError: If a law lacks ``cdf`` or ``mean``.
        ValueError: If a law has mass at zero or a non-finite mean.

    Equation:
        [EQ-SYS-SMP-001].

    References:
        [REF-LIMNIOS-OPRISAN-2001, Chapter 5], [REF-BARLOW-PROSCHAN-1975].

    """
    process = SemiMarkovProcess(
        ("up", "down"),
        {("up", "down"): (1.0, up_time), ("down", "up"): (1.0, down_time)},
    )
    return SemiMarkovReliabilityModel(process, ("up",), "up", steps=steps)
