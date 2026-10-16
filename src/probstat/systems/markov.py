"""Analytic Markov reliability and availability models.

A :class:`MarkovReliabilityModel` is a continuous-time Markov chain whose
states are split into *up* (system working) and *down* (system failed)
states, together with an initial law on the up states. Every measure is
computed analytically from the chain (no Monte Carlo):

- point availability ``A(t) = Pr(X_t in U)`` and its limit;
- reliability ``R(t) = Pr(no failure in [0, t])`` from the chain in which the
  down states are made absorbing, and the mean time to (first) failure;
- steady-state failure frequency ``w = sum_(i in U, j in D) pi_i q_ij``,
  mean up time ``MUT = A / w``, mean down time ``MDT = (1 - A) / w`` and
  ``MTBF = MUT + MDT = 1 / w``;
- interval quantities over a mission ``[0, t]``: average availability,
  expected downtime, and expected number of failures (the expected number of
  up-to-down transitions).

Time-dependent measures come from uniformization and are returned as
:class:`~probstat.processes.BoundedValue` with a rigorous error bound.
Steady-state and mean-time measures come from subtraction-free (GTH-type)
direct solves.

Builders cover the textbook models: a single repairable unit, repairable
``k``-out-of-``n`` systems (birth-death), a parallel pair with shared or
independent repair, cold/warm/hot standby with imperfect switching, and a
load-sharing pair.

Modelling assumptions: every sojourn time is exponential (constant failure
and repair rates), so the future depends only on the current state;
components interact only through the states that are modelled (shared
repair crews, standby switching, load sharing). Deterministic or
non-exponential repair times, ageing, and periodic inspection need a
semi-Markov or non-homogeneous model instead.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.processes.ctmc import BoundedValue, ContinuousTimeMarkovChain

State = Hashable


def _rate(value: object, *, name: str) -> float:
    """Validate a non-negative finite rate.

    Args:
        value: Candidate rate.
        name: Parameter name for errors.

    Returns:
        The rate as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If it is negative or non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    rate = require_finite_real(value, name=name)
    if rate < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return rate


def _positive_rate(value: object, *, name: str) -> float:
    """Validate a strictly positive finite rate.

    Args:
        value: Candidate rate.
        name: Parameter name for errors.

    Returns:
        The rate as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If it is not positive or non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    rate = _rate(value, name=name)
    if rate == 0.0:
        raise ValueError(f"{name} must be positive")
    return rate


def _crews(value: object, units: int) -> int:
    """Validate a number of repair crews (``None`` means one per unit).

    Args:
        value: Positive integer or ``None``.
        units: Number of units in the system.

    Returns:
        Number of crews, at most ``units``.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If it is not positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    if value is None:
        return units
    crews = require_non_negative_integer(value, name="repair_crews")
    if crews == 0:
        raise ValueError("repair_crews must be positive")
    return min(crews, units)


@dataclass(frozen=True, eq=False)
class MarkovReliabilityModel:
    """A CTMC with up/down states: availability, reliability and frequencies.

    Args:
        chain: The system's continuous-time Markov chain.
        up_states: Labels of the states in which the system works; all other
            states are down states. Both sets must be non-empty.
        initial: Initial law (label, mapping, or probability vector, as in
            :meth:`ContinuousTimeMarkovChain.distribution`); it must put all
            its mass on up states. Stored as a probability vector.

    Raises:
        ValueError: If the state sets or the initial law are invalid.

    Equation:
        [EQ-SYS-MKV-001].

    References:
        [REF-TRIVEDI-2002], [REF-BILLINTON-ALLAN-1992],
        [REF-RAUSAND-HOYLAND-2004].

    """

    chain: ContinuousTimeMarkovChain
    up_states: tuple[State, ...]
    initial: object
    down_states: tuple[State, ...] = field(init=False)
    _absorbed: ContinuousTimeMarkovChain = field(init=False, repr=False)
    _failure_pairs: tuple[tuple[State, State], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the state partition and the initial law.

        Raises:
            TypeError: If ``chain`` is not a CTMC.
            ValueError: If the partition or initial law is invalid.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not isinstance(self.chain, ContinuousTimeMarkovChain):
            raise TypeError("chain must be a ContinuousTimeMarkovChain")
        chosen = set()
        for state in self.up_states:
            chosen.add(self.chain.index(state))
        if not chosen:
            raise ValueError("up_states must be non-empty")
        up = tuple(label for i, label in enumerate(self.chain.states) if i in chosen)
        down = tuple(
            label for i, label in enumerate(self.chain.states) if i not in chosen
        )
        if not down:
            raise ValueError("at least one state must be a down state")
        start = self.chain.distribution(self.initial)
        if any(start[self.chain.index(state)] > 0.0 for state in down):
            raise ValueError("the initial law must put all its mass on up states")
        pairs = tuple(
            (source, target)
            for source in up
            for target in down
            if self.chain.rate(source, target) > 0.0
        )
        object.__setattr__(self, "up_states", up)
        object.__setattr__(self, "down_states", down)
        object.__setattr__(self, "initial", start)
        object.__setattr__(self, "_absorbed", self.chain.make_absorbing(down))
        object.__setattr__(self, "_failure_pairs", pairs)

    @property
    def initial_distribution(self) -> tuple[float, ...]:
        """Return the initial law as a probability vector in state order.

        References:
            Project validation policy. No equation is evaluated.

        """
        return cast(tuple[float, ...], self.initial)

    @property
    def failure_transitions(self) -> tuple[tuple[State, State], ...]:
        """Return the ``(up, down)`` pairs with a positive rate.

        References:
            [REF-TRIVEDI-2002]. No equation is evaluated.

        """
        return self._failure_pairs

    @property
    def reliability_chain(self) -> ContinuousTimeMarkovChain:
        """Return the chain with every down state made absorbing.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-TRIVEDI-2002].

        """
        return self._absorbed

    # ---------------------------------------------------------------- time
    def availability(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return the point availability ``A(t) = sum_(i in U) pi_i(t)``.

        Args:
            t: Non-negative time.
            tolerance: Uniformization truncation tolerance.

        Returns:
            ``A(t)`` with an absolute error bound.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-TRIVEDI-2002], [REF-REIBMAN-TRIVEDI-1988].

        """
        return self.chain.transient(self.initial, t, tolerance).total(self.up_states)

    def unavailability(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return ``1 - A(t)``, summed over the down states (no cancellation).

        Args:
            t: Non-negative time.
            tolerance: Uniformization truncation tolerance.

        Returns:
            Point unavailability with an absolute error bound.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-TRIVEDI-2002].

        """
        return self.chain.transient(self.initial, t, tolerance).total(self.down_states)

    def availability_curve(
        self, times: Iterable[float], tolerance: float = 1e-12
    ) -> tuple[BoundedValue, ...]:
        """Return ``A(t)`` at several times from one uniformization pass.

        Args:
            times: Non-negative times.
            tolerance: Uniformization truncation tolerance.

        Returns:
            One bounded value per time.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        solutions = self.chain.transient_many(self.initial, times, tolerance)
        return tuple(solution.total(self.up_states) for solution in solutions)

    def reliability(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return ``R(t) = Pr(no entry into a down state during [0, t])``.

        Computed as the up-state mass of the chain whose down states are
        absorbing.

        Args:
            t: Non-negative time.
            tolerance: Uniformization truncation tolerance.

        Returns:
            Reliability with an absolute error bound.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-TRIVEDI-2002], [REF-RAUSAND-HOYLAND-2004].

        """
        solution = self._absorbed.transient(self.initial, t, tolerance)
        return solution.total(self.up_states)

    def unreliability(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return ``1 - R(t)`` summed over the absorbing down states.

        Args:
            t: Non-negative time.
            tolerance: Uniformization truncation tolerance.

        Returns:
            Probability of at least one failure in ``[0, t]``.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-TRIVEDI-2002].

        """
        solution = self._absorbed.transient(self.initial, t, tolerance)
        return solution.total(self.down_states)

    def reliability_curve(
        self, times: Iterable[float], tolerance: float = 1e-12
    ) -> tuple[BoundedValue, ...]:
        """Return ``R(t)`` at several times from one uniformization pass.

        Args:
            times: Non-negative times.
            tolerance: Uniformization truncation tolerance.

        Returns:
            One bounded value per time.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        solutions = self._absorbed.transient_many(self.initial, times, tolerance)
        return tuple(solution.total(self.up_states) for solution in solutions)

    def failure_frequency(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return the unconditional failure intensity ``w(t)`` at time ``t``.

        ``w(t) = sum_(i in U) pi_i(t) sum_(j in D) q_ij``: the rate of
        up-to-down transitions [EQ-SYS-MKV-003].

        Args:
            t: Non-negative time.
            tolerance: Uniformization truncation tolerance.

        Returns:
            Failure intensity with an absolute error bound.

        Equation:
            [EQ-SYS-MKV-003].

        References:
            [REF-BILLINTON-ALLAN-1992], [REF-RAUSAND-HOYLAND-2004].

        """
        solution = self.chain.transient(self.initial, t, tolerance)
        flux = self._down_flux()
        weights = tuple(flux.get(state, 0.0) for state in self.chain.states)
        return solution.weighted(weights)

    def interval_availability(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return the average availability ``(1/t) int_0^t A(s) ds``.

        Args:
            t: Positive horizon.
            tolerance: Truncation tolerance relative to ``t``.

        Returns:
            Interval availability with an absolute error bound.

        Raises:
            ValueError: If ``t`` is not positive.

        Equation:
            [EQ-SYS-MKV-004].

        References:
            [REF-TRIVEDI-2002], [REF-REIBMAN-TRIVEDI-1988].

        """
        horizon = require_finite_real(t, name="t")
        if horizon <= 0.0:
            raise ValueError("t must be positive")
        occupancy = self.chain.expected_time_in_states(self.initial, horizon, tolerance)
        total = occupancy.total(self.up_states)
        return BoundedValue(total.value / horizon, total.error_bound / horizon)

    def expected_downtime(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return ``E[time spent in down states during [0, t]]``.

        Args:
            t: Non-negative horizon.
            tolerance: Truncation tolerance relative to ``t``.

        Returns:
            Expected downtime (time units) with an absolute error bound.

        Equation:
            [EQ-SYS-MKV-004].

        References:
            [REF-TRIVEDI-2002], [REF-REIBMAN-TRIVEDI-1988].

        """
        occupancy = self.chain.expected_time_in_states(self.initial, t, tolerance)
        return occupancy.total(self.down_states)

    def expected_failures(self, t: float, tolerance: float = 1e-12) -> BoundedValue:
        """Return the expected number of system failures in ``[0, t]``.

        ``int_0^t w(s) ds = sum_(i in U, j in D) q_ij int_0^t pi_i(s) ds``,
        the accumulated up-to-down transition flux [EQ-SYS-MKV-004].

        Args:
            t: Non-negative horizon.
            tolerance: Truncation tolerance relative to ``t``.

        Returns:
            Expected failure count with an absolute error bound.

        Equation:
            [EQ-SYS-MKV-004].

        References:
            [REF-BILLINTON-ALLAN-1992], [REF-REIBMAN-TRIVEDI-1988].

        """
        if not self._failure_pairs:
            return BoundedValue(0.0, 0.0)
        return self.chain.expected_transition_count(
            self.initial, t, self._failure_pairs, tolerance
        )

    # ---------------------------------------------------------- mean times
    def mean_time_to_failure(self) -> float:
        """Return the mean time to first failure (MTTF) from the initial law.

        ``MTTF = alpha (-Q_UU)^-1 1`` with the down states absorbing
        [EQ-SYS-MKV-002], solved without subtraction. Returns ``inf`` when
        the initial law reaches, with positive probability, a set of up states
        that the system never leaves.

        Returns:
            Mean time to failure.

        Equation:
            [EQ-SYS-MKV-002].

        References:
            [REF-TRIVEDI-2002], [REF-RAUSAND-HOYLAND-2004],
            [REF-GRASSMANN-TAKSAR-HEYMAN-1985].

        """
        absorbed = self._absorbed
        down = set(self.down_states)
        trapped = [
            members
            for members in absorbed.closed_classes()
            if not (len(members) == 1 and members[0] in down)
        ]
        trapped_states = [state for members in trapped for state in members]
        start = self.initial_distribution
        if trapped_states and any(
            probability > 0.0 and absorbed.can_reach(state, trapped_states)
            for state, probability in zip(absorbed.states, start, strict=True)
        ):
            return math.inf
        if trapped_states:
            absorbed = absorbed.make_absorbing(trapped_states)
        return absorbed.absorption(start).initial_mean_time

    def _steady_law(self) -> tuple[float, ...]:
        """Return the limiting state law from the initial law.

        References:
            [REF-NORRIS-1997, Section 3.6]. No equation is evaluated.

        """
        return self.chain.limiting_distribution(self.initial)

    def _down_flux(self) -> dict[State, float]:
        """Return each up state's total rate into the down states.

        References:
            [REF-BILLINTON-ALLAN-1992]. No equation is evaluated.

        """
        flux = dict.fromkeys(self.up_states, 0.0)
        for source, target in self._failure_pairs:
            flux[source] += self.chain.rate(source, target)
        return flux

    def steady_state_availability(self) -> float:
        """Return ``A = lim A(t) = sum_(i in U) pi_i``.

        Uses the GTH stationary law (the limit from the initial law when the
        chain has several closed classes, e.g. ``0`` for a non-repairable
        system).

        Returns:
            Limiting availability.

        Equation:
            [EQ-SYS-MKV-001], [EQ-PROC-CTMC-005].

        References:
            [REF-TRIVEDI-2002], [REF-GRASSMANN-TAKSAR-HEYMAN-1985].

        """
        law = self._steady_law()
        return math.fsum(law[self.chain.index(state)] for state in self.up_states)

    def steady_state_unavailability(self) -> float:
        """Return ``1 - A`` summed over down states (no cancellation).

        Returns:
            Limiting unavailability.

        Equation:
            [EQ-SYS-MKV-001].

        References:
            [REF-TRIVEDI-2002].

        """
        law = self._steady_law()
        return math.fsum(law[self.chain.index(state)] for state in self.down_states)

    def steady_state_failure_frequency(self) -> float:
        """Return ``w = sum_(i in U) pi_i sum_(j in D) q_ij``.

        Returns:
            Long-run number of failures per unit time.

        Equation:
            [EQ-SYS-MKV-003].

        References:
            [REF-BILLINTON-ALLAN-1992], [REF-RAUSAND-HOYLAND-2004].

        """
        law = self._steady_law()
        flux = self._down_flux()
        return math.fsum(
            law[self.chain.index(state)] * flux[state] for state in self.up_states
        )

    def _frequency_or_raise(self) -> float:
        """Return the steady-state failure frequency, rejecting zero.

        Raises:
            ValueError: If no failures occur in steady state.

        References:
            [REF-BILLINTON-ALLAN-1992]. No equation is evaluated.

        """
        frequency = self.steady_state_failure_frequency()
        if frequency <= 0.0:
            raise ValueError(
                "no failures occur in steady state (non-repairable or "
                "never-failing system); MUT/MDT/MTBF are undefined"
            )
        return frequency

    def mean_up_time(self) -> float:
        """Return the steady-state mean up time ``MUT = A / w``.

        Raises:
            ValueError: If ``w = 0``.

        Equation:
            [EQ-SYS-MKV-003].

        References:
            [REF-BILLINTON-ALLAN-1992], [REF-RAUSAND-HOYLAND-2004].

        """
        return self.steady_state_availability() / self._frequency_or_raise()

    def mean_down_time(self) -> float:
        """Return the steady-state mean down time ``MDT = (1 - A) / w``.

        Raises:
            ValueError: If ``w = 0``.

        Equation:
            [EQ-SYS-MKV-003].

        References:
            [REF-BILLINTON-ALLAN-1992], [REF-RAUSAND-HOYLAND-2004].

        """
        return self.steady_state_unavailability() / self._frequency_or_raise()

    def mean_time_to_repair(self) -> float:
        """Return the system MTTR, i.e. the steady-state mean down time.

        For a system (as opposed to a single unit) the mean duration of a
        down period is ``MDT``; this alias exists for that usage.

        Raises:
            ValueError: If ``w = 0``.

        Equation:
            [EQ-SYS-MKV-003].

        References:
            [REF-BILLINTON-ALLAN-1992].

        """
        return self.mean_down_time()

    def mean_time_between_failures(self) -> float:
        """Return ``MTBF = MUT + MDT = 1 / w``.

        Raises:
            ValueError: If ``w = 0``.

        Equation:
            [EQ-SYS-MKV-003].

        References:
            [REF-BILLINTON-ALLAN-1992], [REF-RAUSAND-HOYLAND-2004].

        """
        return 1.0 / self._frequency_or_raise()


def from_states(
    states: Iterable[State],
    rates: Mapping[tuple[State, State], float] | Sequence[Sequence[float]],
    up_states: Iterable[State],
    initial: object,
) -> MarkovReliabilityModel:
    """Build a reliability model from states, rates and the up set.

    Args:
        states: Distinct state labels.
        rates: ``{(from, to): rate}`` mapping or generator matrix.
        up_states: Labels of the working states.
        initial: Initial law on the up states.

    Returns:
        The model.

    Raises:
        ValueError: If any part is invalid.

    Equation:
        [EQ-SYS-MKV-001].

    References:
        [REF-TRIVEDI-2002].

    """
    chain = ContinuousTimeMarkovChain(tuple(states), rates)
    return MarkovReliabilityModel(chain, tuple(up_states), initial)


def single_repairable_unit(
    failure_rate: float, repair_rate: float
) -> MarkovReliabilityModel:
    """Return the two-state model ``"up" <-> "down"``.

    ``A(t) = mu / (lambda + mu) + lambda / (lambda + mu) exp(-(lambda + mu) t)``
    and ``R(t) = exp(-lambda t)`` [EQ-SYS-MKV-005].

    Args:
        failure_rate: ``lambda > 0``.
        repair_rate: ``mu >= 0`` (zero: non-repairable).

    Returns:
        Model starting in ``"up"``.

    Raises:
        ValueError: If a rate is invalid.

    Equation:
        [EQ-SYS-MKV-005].

    References:
        [REF-TRIVEDI-2002], [REF-BILLINTON-ALLAN-1992].

    """
    failure = _positive_rate(failure_rate, name="failure_rate")
    repair = _rate(repair_rate, name="repair_rate")
    chain = ContinuousTimeMarkovChain(
        ("up", "down"), {("up", "down"): failure, ("down", "up"): repair}
    )
    return MarkovReliabilityModel(chain, ("up",), "up")


def repairable_k_out_of_n(
    n: int,
    k: int,
    failure_rate: float,
    repair_rate: float,
    repair_crews: int | None = 1,
    *,
    failures_when_down: bool = True,
) -> MarkovReliabilityModel:
    """Return the birth-death model of a ``k``-out-of-``n``:G system.

    State ``j`` (an ``int``) is the number of failed units; the system works
    while ``n - j >= k``. Each working unit fails at ``lambda``; failed units
    are repaired at ``mu`` each by ``c`` crews, so the repair rate in state
    ``j`` is ``min(j, c) mu`` [EQ-SYS-MKV-006]. The steady state has the
    product form ``pi_j ∝ prod_(i < j) (n - i) lambda / (min(i + 1, c) mu)``;
    with ``c >= n`` (independent repair) it is binomial with unit
    availability ``mu / (lambda + mu)``.

    Args:
        n: Number of identical units, ``n >= 1``.
        k: Units required, ``1 <= k <= n``.
        failure_rate: Unit failure rate ``lambda > 0``.
        repair_rate: Unit repair rate ``mu >= 0`` (zero: non-repairable).
        repair_crews: Number of repair crews ``c``; ``None`` means one per
            unit (independent repair).
        failures_when_down: Whether surviving units keep failing while the
            system is down (``False``: the system is stopped).

    Returns:
        Model starting with all units working (state ``0``).

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-SYS-MKV-006].

    References:
        [REF-TRIVEDI-2002], [REF-BILLINTON-ALLAN-1992].

    """
    units = require_non_negative_integer(n, name="n")
    required = require_non_negative_integer(k, name="k")
    if units < 1 or not 1 <= required <= units:
        raise ValueError("need n >= 1 and 1 <= k <= n")
    failure = _positive_rate(failure_rate, name="failure_rate")
    repair = _rate(repair_rate, name="repair_rate")
    crews = _crews(repair_crews, units)
    if not isinstance(failures_when_down, bool):
        raise TypeError("failures_when_down must be a bool")
    rates: dict[tuple[State, State], float] = {}
    for j in range(units + 1):
        up = units - j >= required
        if j < units and (up or failures_when_down):
            rates[(j, j + 1)] = (units - j) * failure
        if j > 0 and repair > 0.0:
            rates[(j, j - 1)] = min(j, crews) * repair
    chain = ContinuousTimeMarkovChain(tuple(range(units + 1)), rates)
    up_states = tuple(j for j in range(units + 1) if units - j >= required)
    return MarkovReliabilityModel(chain, up_states, 0)


def parallel_redundant_pair(
    failure_rate_1: float,
    failure_rate_2: float,
    repair_rate_1: float,
    repair_rate_2: float,
    *,
    shared_repair: bool = True,
) -> MarkovReliabilityModel:
    """Return a two-unit active-parallel system (1-out-of-2) with repair.

    States: ``"both up"``, ``"unit 1 down"``, ``"unit 2 down"`` (up) and the
    down state(s). With ``shared_repair=True`` one crew repairs in order of
    failure, so the down states are ``"both down, repairing 1"`` and
    ``"both down, repairing 2"``; otherwise each unit has its own crew and
    the single down state is ``"both down"`` (then the units are independent
    and ``1 - A = prod_i lambda_i / (lambda_i + mu_i)``). For identical units
    ``MTTF = (3 lambda + mu) / (2 lambda^2)`` [EQ-SYS-MKV-008].

    Args:
        failure_rate_1: Unit 1 failure rate (positive).
        failure_rate_2: Unit 2 failure rate (positive).
        repair_rate_1: Unit 1 repair rate (non-negative).
        repair_rate_2: Unit 2 repair rate (non-negative).
        shared_repair: One first-come-first-served crew if true.

    Returns:
        Model starting in ``"both up"``.

    Raises:
        ValueError: If a rate is invalid.

    Equation:
        [EQ-SYS-MKV-008].

    References:
        [REF-TRIVEDI-2002], [REF-BILLINTON-ALLAN-1992].

    """
    l1 = _positive_rate(failure_rate_1, name="failure_rate_1")
    l2 = _positive_rate(failure_rate_2, name="failure_rate_2")
    m1 = _rate(repair_rate_1, name="repair_rate_1")
    m2 = _rate(repair_rate_2, name="repair_rate_2")
    if not isinstance(shared_repair, bool):
        raise TypeError("shared_repair must be a bool")
    up = ("both up", "unit 1 down", "unit 2 down")
    rates: dict[tuple[State, State], float] = {
        ("both up", "unit 1 down"): l1,
        ("both up", "unit 2 down"): l2,
        ("unit 1 down", "both up"): m1,
        ("unit 2 down", "both up"): m2,
    }
    if shared_repair:
        first, second = "both down, repairing 1", "both down, repairing 2"
        states: tuple[State, ...] = (*up, first, second)
        rates[("unit 1 down", first)] = l2
        rates[("unit 2 down", second)] = l1
        rates[(first, "unit 2 down")] = m1
        rates[(second, "unit 1 down")] = m2
    else:
        states = (*up, "both down")
        rates[("unit 1 down", "both down")] = l2
        rates[("unit 2 down", "both down")] = l1
        rates[("both down", "unit 2 down")] = m1
        rates[("both down", "unit 1 down")] = m2
    chain = ContinuousTimeMarkovChain(states, rates)
    return MarkovReliabilityModel(chain, up, "both up")


def standby_system(
    spares: int,
    failure_rate: float,
    standby_failure_rate: float = 0.0,
    switch_success_probability: float = 1.0,
    repair_rate: float = 0.0,
    *,
    active: int = 1,
    repair_crews: int | None = 1,
) -> MarkovReliabilityModel:
    """Return a standby-redundant system with imperfect switching.

    ``active`` identical units must operate (each failing at ``lambda``);
    ``spares`` identical units wait in standby, failing at
    ``lambda_s`` each (``0``: cold, ``lambda``: hot, in between: warm).
    When an active unit fails and a spare is available, the switch-over
    succeeds with probability ``p``; if it fails the system goes down.
    Standby failures are revealed at once. States are ``"j failed"`` for
    ``j = 0..spares+1`` (up while ``j <= spares``) and, when ``p < 1``,
    ``"switch failure (j failed)"`` for ``j = 1..spares`` (down).
    With repair (rate ``mu`` per unit, ``c`` crews) a down state returns to
    ``"j-1 failed"`` when a repair completes; no unit fails while the system
    is down.

    Closed forms for one active unit, no repair [EQ-SYS-MKV-007]: cold
    standby with ``p = 1`` and ``n = spares + 1`` units gives
    ``R(t) = exp(-lambda t) sum_(k < n) (lambda t)^k / k!``; with one spare,
    ``R(t) = exp(-lambda t) (1 + p (lambda / lambda_s)(1 - exp(-lambda_s t)))``
    (warm), which tends to ``exp(-lambda t)(1 + p lambda t)`` as
    ``lambda_s -> 0``.

    Args:
        spares: Number of standby units (non-negative integer).
        failure_rate: Active failure rate ``lambda > 0``.
        standby_failure_rate: Standby failure rate ``lambda_s >= 0``.
        switch_success_probability: ``p`` in ``[0, 1]``.
        repair_rate: Unit repair rate ``mu >= 0`` (zero: non-repairable).
        active: Number of units that must operate (positive integer).
        repair_crews: Repair crews; ``None`` means one per unit.

    Returns:
        Model starting in ``"0 failed"``.

    Raises:
        ValueError: If an argument is invalid.

    Equation:
        [EQ-SYS-MKV-007].

    References:
        [REF-TRIVEDI-2002], [REF-RAUSAND-HOYLAND-2004].

    """
    standby = require_non_negative_integer(spares, name="spares")
    operating = require_non_negative_integer(active, name="active")
    if operating < 1:
        raise ValueError("active must be positive")
    failure = _positive_rate(failure_rate, name="failure_rate")
    dormant = _rate(standby_failure_rate, name="standby_failure_rate")
    success = require_probability(
        switch_success_probability, name="switch_success_probability"
    )
    repair = _rate(repair_rate, name="repair_rate")
    crews = _crews(repair_crews, operating + standby)

    def name(j: int) -> str:
        """Return the label of the state with ``j`` failed units.

        References:
            Project validation policy. No equation is evaluated.

        """
        return f"{j} failed"

    def switch(j: int) -> str:
        """Return the label of the switching-failure state.

        References:
            Project validation policy. No equation is evaluated.

        """
        return f"switch failure ({j} failed)"

    rates: dict[tuple[State, State], float] = {}

    def add(source: str, target: str, value: float) -> None:
        """Accumulate a positive rate.

        References:
            Project validation policy. No equation is evaluated.

        """
        if value > 0.0:
            rates[(source, target)] = rates.get((source, target), 0.0) + value

    states: list[State] = [name(j) for j in range(standby + 2)]
    with_switch = success < 1.0 and standby > 0
    if with_switch:
        states.extend(switch(j) for j in range(1, standby + 1))
    for j in range(standby + 1):
        waiting = standby - j
        demand = operating * failure
        if waiting >= 1:
            add(name(j), name(j + 1), demand * success)
            if with_switch:
                add(name(j), switch(j + 1), demand * (1.0 - success))
        else:
            add(name(j), name(j + 1), demand)
        add(name(j), name(j + 1), waiting * dormant)
        if j > 0:
            add(name(j), name(j - 1), min(j, crews) * repair)
    add(name(standby + 1), name(standby), min(standby + 1, crews) * repair)
    if with_switch:
        for j in range(1, standby + 1):
            add(switch(j), name(j - 1), min(j, crews) * repair)
    chain = ContinuousTimeMarkovChain(tuple(states), rates)
    up_states = tuple(name(j) for j in range(standby + 1))
    return MarkovReliabilityModel(chain, up_states, name(0))


def load_sharing_pair(
    failure_rate: float,
    loaded_failure_rate: float,
    repair_rate: float = 0.0,
    *,
    repair_crews: int | None = 1,
) -> MarkovReliabilityModel:
    """Return a two-unit load-sharing system.

    While both units work each fails at ``lambda``; the survivor carries the
    full load and fails at ``lambda_L``. States ``"both up"``, ``"one up"``
    (up) and ``"both down"``. Without repair
    ``R(t) = exp(-2 lambda t) + 2 lambda / (lambda_L - 2 lambda)
    (exp(-2 lambda t) - exp(-lambda_L t))`` for ``lambda_L != 2 lambda``
    [EQ-SYS-MKV-007].

    Args:
        failure_rate: Per-unit rate ``lambda > 0`` under shared load.
        loaded_failure_rate: Survivor rate ``lambda_L > 0``.
        repair_rate: Unit repair rate ``mu >= 0``.
        repair_crews: Repair crews; ``None`` means one per unit.

    Returns:
        Model starting in ``"both up"``.

    Raises:
        ValueError: If a rate is invalid.

    Equation:
        [EQ-SYS-MKV-007].

    References:
        [REF-RAUSAND-HOYLAND-2004], [REF-TRIVEDI-2002].

    """
    shared = _positive_rate(failure_rate, name="failure_rate")
    loaded = _positive_rate(loaded_failure_rate, name="loaded_failure_rate")
    repair = _rate(repair_rate, name="repair_rate")
    crews = _crews(repair_crews, 2)
    rates: dict[tuple[State, State], float] = {
        ("both up", "one up"): 2.0 * shared,
        ("one up", "both down"): loaded,
    }
    if repair > 0.0:
        rates[("one up", "both up")] = repair
        rates[("both down", "one up")] = min(2, crews) * repair
    chain = ContinuousTimeMarkovChain(("both up", "one up", "both down"), rates)
    return MarkovReliabilityModel(chain, ("both up", "one up"), "both up")
