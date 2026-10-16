"""Finite continuous-time Markov chains (CTMCs) solved analytically.

A homogeneous CTMC on a finite set of labelled states is described by its
generator ``Q``: ``q_ij >= 0`` is the transition rate from state ``i`` to
state ``j != i`` and ``q_ii = -sum_(j != i) q_ij``, so every row sums to zero.
The state law ``pi(t)`` solves the Kolmogorov forward equation
``pi'(t) = pi(t) Q``, i.e. ``pi(t) = pi(0) exp(Q t)`` [EQ-PROC-CTMC-001].

Numerical methods, all deterministic and dependency-free:

- **Transient law** by uniformization (Jensen 1953; Grassmann 1977; Reibman
  and Trivedi 1988) [EQ-PROC-CTMC-002]. With ``q >= max_i |q_ii|`` and the
  stochastic matrix ``P = I + Q / q``,
  ``pi(t) = sum_k Poisson(k; q t) pi(0) P^k``. Every quantity in the series
  is non-negative, so no cancellation occurs. The Poisson series is truncated
  to a window ``[L, R]`` chosen from rigorous Cramer-Chernoff tail bounds in
  log space (no underflow for large ``q t``); the window weights are computed
  by the Fox-Glynn mode-centred recurrence and normalized
  [EQ-PROC-CTMC-003]. Each result carries an error bound that adds the
  truncation bound and a first-order floating-point rounding bound.
- **Accumulated reward** ``int_0^t pi(s) r ds`` by the integrated
  uniformization series ``(1/q) sum_k Pr(N_(qt) > k) pi(0) P^k`` with its own
  truncation bound [EQ-PROC-CTMC-004].
- **Steady state** by the Grassmann-Taksar-Heyman (GTH) state-reduction
  algorithm, which uses no subtraction and is accurate even for very stiff
  rates [EQ-PROC-CTMC-005].
- **Absorption** (absorption probabilities, mean time to absorption, and the
  fundamental matrix ``(-Q_TT)^-1``) by a subtraction-free elimination of the
  same GTH type [EQ-PROC-CTMC-006].

Seeded path simulation (:meth:`ContinuousTimeMarkovChain.simulate`) exists
only as an independent cross-check of these analytic results.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, as_matrix
from probstat.processes.markov import MarkovChain

_ROW_SUM_TOLERANCE = 1e-10
_DISTRIBUTION_TOLERANCE = 1e-10
_UNIT_ROUNDOFF = 2.0**-53
_MIN_TOLERANCE = 1e-250
_MAX_TOLERANCE = 0.5
_DEFAULT_MAX_TERMS = 5_000_000
_BOUND_SAFETY = 1.01
_SQUARING_BLOCK = 512.0

State = Hashable


@dataclass(frozen=True, slots=True)
class BoundedValue:
    """A computed value together with a rigorous absolute error bound.

    Values computed from a truncated series (uniformization) are returned in
    this form so that the approximation is never presented as exact.
    ``float(result)`` returns the value.

    Args:
        value: Computed value.
        error_bound: Upper bound on ``|value - exact|``: the series
            truncation bound plus a first-order floating-point rounding
            bound (see :meth:`ContinuousTimeMarkovChain.transient`).

    References:
        [REF-REIBMAN-TRIVEDI-1988], [REF-FOX-GLYNN-1988].

    """

    value: float
    error_bound: float

    def __float__(self) -> float:
        """Return the computed value.

        References:
            Project validation policy. No equation is evaluated.

        """
        return self.value


def _subset_bound(
    value: float, truncation: float, rounding: float, fallback: float
) -> float:
    """Return the error bound of a sum of non-negative computed components.

    The truncation bound is an absolute L1 bound; the rounding bound is
    componentwise relative, so for a subset sum ``S`` it contributes at most
    ``rounding |S| / (1 - rounding)``.

    Args:
        value: Computed subset sum.
        truncation: Absolute L1 truncation bound.
        rounding: Componentwise relative rounding bound.
        fallback: Bound used if ``rounding`` is not small.

    Returns:
        Absolute error bound.

    References:
        [REF-HIGHAM-2002, Chapter 3]. No named equation is needed.

    """
    if rounding >= 0.5:
        return fallback
    return truncation + rounding * abs(value) / (1.0 - rounding)


def _weighted_bound(
    components: Sequence[float],
    values: Sequence[float],
    truncation: float,
    rounding: float,
    fallback: float,
) -> BoundedValue:
    """Return ``sum_i c_i r_i`` with a truncation-plus-rounding bound.

    Args:
        components: Computed non-negative components ``c``.
        values: Weights ``r`` (same length).
        truncation: Absolute L1 truncation bound of ``c``.
        rounding: Componentwise relative rounding bound of ``c``.
        fallback: L1 bound used if ``rounding`` is not small.

    Returns:
        Weighted sum and the bound ``truncation max|r| + rounding
        sum|c_i r_i| / (1 - rounding)``.

    Raises:
        ValueError: If the lengths differ.

    References:
        [REF-HIGHAM-2002, Chapter 3]. No named equation is needed.

    """
    if len(values) != len(components):
        raise ValueError("values length must equal the state count")
    value = math.fsum(a * b for a, b in zip(components, values, strict=True))
    magnitude = math.fsum(abs(a * b) for a, b in zip(components, values, strict=True))
    scale = max((abs(b) for b in values), default=0.0)
    if rounding >= 0.5:
        return BoundedValue(value, fallback * scale)
    return BoundedValue(
        value, truncation * scale + rounding * magnitude / (1.0 - rounding)
    )


@dataclass(frozen=True, slots=True)
class TransientSolution:
    """Transient state law ``pi(t)`` of a CTMC computed by uniformization.

    Args:
        states: State labels in chain order.
        time: Time ``t``.
        probabilities: ``pi(t)`` in chain order.
        error_bound: Bound on the L1 error ``sum_i |pi_hat_i - pi_i|``:
            the truncation bound plus the rounding bound times the total
            mass.
        truncation_error_bound: Rigorous absolute L1 bound from the Poisson
            mass outside the retained window (``2 tau``), or its propagation
            through the squarings.
        rounding_error_bound: First-order bound on the componentwise
            *relative* rounding error (every quantity in the computation is
            non-negative, so small probabilities keep their relative
            accuracy).
        uniformization_rate: The rate ``q``.
        method: ``"series"``, ``"squaring (s=...)"`` or ``"exact"``
            (``t = 0`` or no transitions).
        terms: Last Poisson term ``R`` of the series, or the number of
            squarings.

    References:
        [REF-JENSEN-1953], [REF-REIBMAN-TRIVEDI-1988].

    """

    states: tuple[State, ...]
    time: float
    probabilities: Vector
    error_bound: float
    truncation_error_bound: float
    rounding_error_bound: float
    uniformization_rate: float
    method: str
    terms: int

    def probability(self, state: State) -> float:
        """Return ``pi_state(t)``.

        Args:
            state: State label.

        Returns:
            Transient probability of the state.

        Raises:
            ValueError: If the label is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        if state not in self.states:
            raise ValueError(f"unknown state {state!r}")
        return self.probabilities[self.states.index(state)]

    def total(self, states: Iterable[State]) -> BoundedValue:
        """Return ``sum_(i in states) pi_i(t)`` with its error bound.

        The bound is ``truncation + rounding * value / (1 - rounding)``: the
        truncation part is absolute, the rounding part relative.

        Args:
            states: State labels (duplicates ignored).

        Returns:
            Probability of the set with an absolute error bound.

        Raises:
            ValueError: If a label is unknown.

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        chosen = set(states)
        for state in chosen:
            if state not in self.states:
                raise ValueError(f"unknown state {state!r}")
        value = math.fsum(
            p
            for label, p in zip(self.states, self.probabilities, strict=True)
            if label in chosen
        )
        bound = _subset_bound(
            value,
            self.truncation_error_bound,
            self.rounding_error_bound,
            self.error_bound,
        )
        return BoundedValue(value, bound)

    def weighted(self, values: Sequence[float]) -> BoundedValue:
        """Return ``sum_i pi_i(t) r_i`` for per-state values ``r`` with a bound.

        The bound is ``truncation * max |r_i| + rounding * sum |pi_i r_i| /
        (1 - rounding)``.

        Args:
            values: One finite value per state, in state order.

        Returns:
            Expected value of ``r(X_t)`` with an absolute error bound.

        Raises:
            ValueError: If the length is wrong.

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        return _weighted_bound(
            self.probabilities,
            values,
            self.truncation_error_bound,
            self.rounding_error_bound,
            self.error_bound,
        )


@dataclass(frozen=True, slots=True)
class OccupancySolution:
    """Expected time spent in each state over ``[0, t]``.

    Args:
        states: State labels in chain order.
        time: Horizon ``t``.
        occupancy: ``int_0^t pi_i(s) ds`` per state; the entries sum to ``t``.
        error_bound: Bound on the L1 error of ``occupancy`` (time units).
        truncation_error_bound: Rigorous absolute L1 truncation part (time
            units).
        rounding_error_bound: First-order componentwise relative rounding
            bound (dimensionless).
        uniformization_rate: The rate ``q``.
        method: ``"series"``, ``"squaring (s=...)"`` or ``"exact"``.
        terms: Number of series terms, or the number of squarings.

    References:
        [REF-REIBMAN-TRIVEDI-1988].

    """

    states: tuple[State, ...]
    time: float
    occupancy: Vector
    error_bound: float
    truncation_error_bound: float
    rounding_error_bound: float
    uniformization_rate: float
    method: str
    terms: int

    def time_in(self, state: State) -> float:
        """Return the expected time spent in ``state`` over ``[0, t]``.

        Args:
            state: State label.

        Returns:
            Expected occupancy time.

        Raises:
            ValueError: If the label is unknown.

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        if state not in self.states:
            raise ValueError(f"unknown state {state!r}")
        return self.occupancy[self.states.index(state)]

    def total(self, states: Iterable[State]) -> BoundedValue:
        """Return the expected time spent in a set of states.

        Args:
            states: State labels (duplicates ignored).

        Returns:
            Expected time in the set with an absolute error bound.

        Raises:
            ValueError: If a label is unknown.

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        chosen = set(states)
        for state in chosen:
            if state not in self.states:
                raise ValueError(f"unknown state {state!r}")
        return self.weighted(
            tuple(1.0 if label in chosen else 0.0 for label in self.states)
        )

    def weighted(self, values: Sequence[float]) -> BoundedValue:
        """Return ``sum_i L_i r_i`` for per-state values ``r`` with a bound.

        The bound is ``truncation * max |r_i| + rounding * sum |L_i r_i| /
        (1 - rounding)``.

        Args:
            values: One finite value per state, in state order.

        Returns:
            Weighted occupancy with an absolute error bound.

        Raises:
            ValueError: If the length is wrong.

        Equation:
            [EQ-PROC-CTMC-004].

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        return _weighted_bound(
            self.occupancy,
            values,
            self.truncation_error_bound,
            self.rounding_error_bound,
            self.error_bound,
        )


@dataclass(frozen=True, slots=True)
class CTMCAbsorption:
    """Absorption analysis of a CTMC whose closed classes are absorbing states.

    Args:
        transient_states: Labels of the transient states (rows below).
        absorbing_states: Labels of the absorbing states (columns of
            ``absorption_probabilities``).
        fundamental: ``N = (-Q_TT)^-1``; ``N_ij`` is the expected total time
            spent in transient ``j`` starting from transient ``i``.
        absorption_probabilities: ``B = N Q_TA``; ``B_ik`` is the probability
            of absorption in absorbing state ``k`` from transient ``i``.
        mean_times: ``m = N 1``; mean time to absorption per transient state.
        initial_absorption: Absorption probabilities from the initial law.
        initial_mean_time: Mean time to absorption from the initial law.

    References:
        [REF-NORRIS-1997, Section 3.3], [REF-KEMENY-SNELL-1976, Chapter III].

    """

    transient_states: tuple[State, ...]
    absorbing_states: tuple[State, ...]
    fundamental: Matrix
    absorption_probabilities: Matrix
    mean_times: Vector
    initial_absorption: Vector
    initial_mean_time: float

    def probability(self, start: State, target: State) -> float:
        """Return the probability of absorption in ``target`` from ``start``.

        Args:
            start: Transient or absorbing state label.
            target: Absorbing state label.

        Returns:
            Absorption probability.

        Raises:
            ValueError: If ``target`` is not absorbing or ``start`` unknown.

        Equation:
            [EQ-PROC-CTMC-006].

        References:
            [REF-NORRIS-1997, Section 3.3].

        """
        if target not in self.absorbing_states:
            raise ValueError("target must be an absorbing state")
        if start in self.absorbing_states:
            return 1.0 if start == target else 0.0
        if start not in self.transient_states:
            raise ValueError("start must be a transient or absorbing state")
        row = self.transient_states.index(start)
        return self.absorption_probabilities[row][self.absorbing_states.index(target)]

    def mean_time(self, start: State) -> float:
        """Return the mean time to absorption from ``start``.

        Args:
            start: Transient or absorbing state label.

        Returns:
            Mean absorption time; zero for an absorbing state.

        Raises:
            ValueError: If ``start`` is unknown.

        Equation:
            [EQ-PROC-CTMC-006].

        References:
            [REF-NORRIS-1997, Section 3.3].

        """
        if start in self.absorbing_states:
            return 0.0
        if start not in self.transient_states:
            raise ValueError("start must be a transient or absorbing state")
        return self.mean_times[self.transient_states.index(start)]


@dataclass(frozen=True, slots=True)
class CTMCPath:
    """One simulated CTMC path on ``[0, horizon]``.

    Args:
        states: Visited state labels; ``states[i]`` holds on
            ``[times[i], times[i+1])``.
        times: Jump times starting with ``0.0``.
        horizon: Final time of the path.

    References:
        [REF-NORRIS-1997, Section 2.6].

    """

    states: tuple[State, ...]
    times: tuple[float, ...]
    horizon: float

    def state_at(self, time: float) -> State:
        """Return the state occupied at ``time``.

        Args:
            time: Time in ``[0, horizon]``.

        Returns:
            State label.

        Raises:
            ValueError: If ``time`` is outside ``[0, horizon]``.

        References:
            [REF-NORRIS-1997, Section 2.6]. No equation is evaluated.

        """
        moment = require_finite_real(time, name="time")
        if not 0.0 <= moment <= self.horizon:
            raise ValueError("time must lie in [0, horizon]")
        return self.states[bisect.bisect_right(self.times, moment) - 1]

    def time_in(self, state: State) -> float:
        """Return the total time the path spends in ``state``.

        Args:
            state: State label.

        Returns:
            Occupancy time on ``[0, horizon]``.

        References:
            [REF-NORRIS-1997, Section 2.6]. No equation is evaluated.

        """
        ends = self.times[1:] + (self.horizon,)
        return math.fsum(
            end - start
            for label, start, end in zip(self.states, self.times, ends, strict=True)
            if label == state
        )


def _log_poisson_tail_bound(count: int, mean: float) -> float:
    """Return the log of the Cramer-Chernoff bound on a Poisson tail.

    For ``N ~ Poisson(mean)``: ``Pr(N >= k) <= exp(-mean) (e mean / k)^k``
    when ``k >= mean`` and ``Pr(N <= k) <= exp(-mean) (e mean / k)^k`` when
    ``k <= mean`` (with the value ``exp(-mean)`` at ``k = 0``). The log is
    evaluated as ``(k - mean) - k log1p((k - mean) / mean)`` so that no
    large terms cancel when ``mean`` is large.

    Args:
        count: Non-negative integer ``k``.
        mean: Positive Poisson mean.

    Returns:
        Log of the tail bound (at most zero).

    Equation:
        [EQ-PROC-CTMC-003].

    References:
        [REF-BOUCHERON-LUGOSI-MASSART-2013, Section 2.2].

    """
    if count == 0:
        return -mean
    gap = count - mean
    return min(0.0, gap - count * math.log1p(gap / mean))


def _poisson_window(
    mean: float, lower_tail: float, upper_tail: float
) -> tuple[int, int, float, float]:
    """Return a truncation window for Poisson(mean) with rigorous tail bounds.

    Args:
        mean: Positive Poisson mean ``lambda = q t``.
        lower_tail: Target bound on ``Pr(N < L)``.
        upper_tail: Target bound on ``Pr(N >= R)``.

    Returns:
        ``(L, R, tau_lower, tau_upper)`` with ``Pr(N < L) <= tau_lower <=
        lower_tail`` and ``Pr(N > R) <= Pr(N >= R) <= tau_upper <=
        upper_tail``.

    Equation:
        [EQ-PROC-CTMC-003].

    References:
        [REF-FOX-GLYNN-1988], [REF-BOUCHERON-LUGOSI-MASSART-2013, Section 2.2].

    """
    target_upper = math.log(upper_tail)
    start = math.floor(mean) + 1
    if _log_poisson_tail_bound(start, mean) <= target_upper:
        high = start
    else:
        low = start
        step = max(1, math.ceil(math.sqrt(mean)))
        high = start + step
        while _log_poisson_tail_bound(high, mean) > target_upper:
            low = high
            step *= 2
            high = start + step
        while high - low > 1:
            middle = (low + high) // 2
            if _log_poisson_tail_bound(middle, mean) <= target_upper:
                high = middle
            else:
                low = middle
    right = high
    tau_upper = math.exp(_log_poisson_tail_bound(right, mean))
    target_lower = math.log(lower_tail)
    if _log_poisson_tail_bound(0, mean) > target_lower:
        return 0, right, 0.0, tau_upper
    # Largest m < mean with bound(m) <= target; then L = m + 1.
    low = 0
    high = math.ceil(mean) - 1
    while high - low > 0:
        middle = (low + high + 1) // 2
        if _log_poisson_tail_bound(middle, mean) <= target_lower:
            low = middle
        else:
            high = middle - 1
    left = low + 1
    tau_lower = math.exp(_log_poisson_tail_bound(low, mean))
    return left, right, tau_lower, tau_upper


def _poisson_weights(mean: float, left: int, right: int) -> tuple[list[float], int]:
    """Return normalized Poisson weights on ``[left, right]`` (Fox-Glynn).

    The unnormalized weight of the mode is set to one and the others follow
    from ``w_(k+1) = w_k mean / (k + 1)`` and ``w_(k-1) = w_k k / mean``;
    the weights are then divided by their sum, so no ``exp(-mean)`` factor
    can underflow.

    Args:
        mean: Positive Poisson mean.
        left: First index.
        right: Last index, ``right >= left``.

    Returns:
        Normalized weights and the largest distance from the mode (used in
        the rounding bound).

    Equation:
        [EQ-PROC-CTMC-003].

    References:
        [REF-FOX-GLYNN-1988].

    """
    mode = min(max(math.floor(mean), left), right)
    weights = [0.0] * (right - left + 1)
    weights[mode - left] = 1.0
    for k in range(mode + 1, right + 1):
        weights[k - left] = weights[k - 1 - left] * (mean / k)
    for k in range(mode - 1, left - 1, -1):
        weights[k - left] = weights[k + 1 - left] * ((k + 1) / mean)
    total = math.fsum(weights)
    return [w / total for w in weights], max(right - mode, mode - left)


def _solve_exit_system(
    rates: list[list[float]], exits: list[float], rhs: list[list[float]]
) -> list[list[float]]:
    """Solve ``(-Q_TT) X = B`` by subtraction-free (GTH-type) elimination.

    ``Q_TT`` has off-diagonal rates ``rates[i][j] >= 0`` and diagonal
    ``-(sum_j rates[i][j] + exits[i])``. Eliminating state ``k`` redirects
    the flow into ``k`` proportionally to ``k``'s exits; the new diagonal is
    recomputed as the sum of the new off-diagonal rates plus the new exit
    rate, which equals the Schur complement exactly in real arithmetic but
    needs no subtraction. With a non-negative right-hand side every operation
    combines non-negative numbers, so the solution has small componentwise
    relative error even for very stiff rates.

    Args:
        rates: Square off-diagonal rate matrix among the states (diagonal
            ignored).
        exits: Total exit rate of each state out of the set.
        rhs: Non-negative right-hand side, one row per state.

    Returns:
        Solution rows ``X``.

    Raises:
        ValueError: If an elimination pivot is zero, i.e. leaving the set is
            not certain from some state.

    Equation:
        [EQ-PROC-CTMC-006].

    References:
        [REF-GRASSMANN-TAKSAR-HEYMAN-1985], [REF-STEWART-1994, Chapter 2].

    """
    size = len(rates)
    r = [list(row) for row in rates]
    d = list(exits)
    b = [list(row) for row in rhs]
    width = len(b[0]) if b else 0
    pivots = [0.0] * size
    for k in range(size - 1, -1, -1):
        row_k = r[k]
        pivot = math.fsum([*(row_k[j] for j in range(k)), d[k]])
        if not pivot > 0.0:
            raise ValueError(
                "leaving the transient set is not certain: a closed class "
                "of non-absorbing states is reachable"
            )
        pivots[k] = pivot
        b_k = b[k]
        for i in range(k):
            rate = r[i][k]
            if rate > 0.0:
                factor = rate / pivot
                row_i = r[i]
                for j in range(k):
                    if j != i and row_k[j] != 0.0:
                        row_i[j] += factor * row_k[j]
                d[i] += factor * d[k]
                b_i = b[i]
                for c in range(width):
                    b_i[c] += factor * b_k[c]
    solution: list[list[float]] = []
    for k in range(size):
        row_k = r[k]
        solution.append(
            [
                math.fsum([*(row_k[j] * solution[j][c] for j in range(k)), b[k][c]])
                / pivots[k]
                for c in range(width)
            ]
        )
    return solution


def _gth_stationary(rates: list[list[float]]) -> list[float]:
    """Return the stationary law of an irreducible generator by GTH.

    Args:
        rates: Off-diagonal rates of an irreducible CTMC (diagonal ignored).

    Returns:
        Stationary probability vector.

    Raises:
        ValueError: If the rates are not irreducible.

    Equation:
        [EQ-PROC-CTMC-005].

    References:
        [REF-GRASSMANN-TAKSAR-HEYMAN-1985], [REF-STEWART-1994, Section 2.7].

    """
    size = len(rates)
    a = [list(row) for row in rates]
    for k in range(size - 1, 0, -1):
        row_k = a[k]
        pivot = math.fsum(row_k[j] for j in range(k))
        if not pivot > 0.0:
            raise ValueError("generator restricted to the class is not irreducible")
        for i in range(k):
            a[i][k] /= pivot
        for i in range(k):
            factor = a[i][k]
            if factor != 0.0:
                row_i = a[i]
                for j in range(k):
                    if j != i and row_k[j] != 0.0:
                        row_i[j] += factor * row_k[j]
    weights = [1.0]
    for j in range(1, size):
        weights.append(math.fsum(weights[i] * a[i][j] for i in range(j)))
    total = math.fsum(weights)
    return [w / total for w in weights]


def _rescaled(row: list[float]) -> list[float]:
    """Return a non-negative row rescaled to sum to one.

    Args:
        row: Non-negative entries with a positive sum.

    Returns:
        Rescaled row (unchanged if its sum is zero).

    References:
        [REF-STEWART-1994, Chapter 1]. No named equation is needed.

    """
    total = math.fsum(row)
    if total <= 0.0 or total == 1.0:
        return row
    return [value / total for value in row]


def _require_tolerance(value: object) -> float:
    """Validate a series tolerance.

    Args:
        value: Candidate tolerance.

    Returns:
        The tolerance as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is outside ``[1e-250, 0.5]``.

    References:
        Project validation policy. No equation is evaluated.

    """
    tolerance = require_finite_real(value, name="tolerance")
    if not _MIN_TOLERANCE <= tolerance <= _MAX_TOLERANCE:
        raise ValueError("tolerance must lie in [1e-250, 0.5]")
    return tolerance


def _require_time(value: object, *, name: str = "t") -> float:
    """Validate a non-negative finite time.

    Args:
        value: Candidate time.
        name: Parameter name for errors.

    Returns:
        The time as a float.

    Raises:
        TypeError: If it is not real.
        ValueError: If it is negative or non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    moment = require_finite_real(value, name=name)
    if moment < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return moment


@dataclass(frozen=True, eq=False)
class ContinuousTimeMarkovChain:
    """Homogeneous continuous-time Markov chain on finitely many labelled states.

    Args:
        states: Distinct hashable state labels (a string is not accepted as a
            sequence of labels).
        rates: Either a mapping ``{(from_label, to_label): rate}`` of
            non-negative off-diagonal rates (missing pairs are zero; the
            diagonal is filled in as minus the row sum), or a square
            generator matrix in state order with non-negative off-diagonal
            entries whose rows sum to zero within ``1e-10`` times the row's
            total rate. The stored generator always has the diagonal
            recomputed as ``-sum`` of the off-diagonal entries.

    Raises:
        TypeError: If labels are unhashable or rates are not real.
        ValueError: If states are empty or repeated, a rate is negative or
            non-finite, a key is unknown or on the diagonal, or a matrix row
            does not sum to zero.

    Equation:
        [EQ-PROC-CTMC-001].

    References:
        [REF-NORRIS-1997, Sections 2.1-2.8], [REF-STEWART-1994, Chapter 1].

    """

    states: tuple[State, ...]
    rates: Mapping[tuple[State, State], float] | Sequence[Sequence[float]]
    _generator: Matrix = field(init=False, repr=False)
    _positions: Mapping[State, int] = field(init=False, repr=False)
    _successors: tuple[tuple[tuple[int, float], ...], ...] = field(
        init=False, repr=False
    )
    _reach: tuple[tuple[bool, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the states and rates and cache the generator.

        Raises:
            TypeError: If labels are unhashable or rates are not real.
            ValueError: If the specification is invalid.

        Equation:
            [EQ-PROC-CTMC-001].

        References:
            [REF-NORRIS-1997, Section 2.1].

        """
        if isinstance(self.states, str):
            raise TypeError("states must be a sequence of labels, not a string")
        labels = tuple(self.states)
        if not labels:
            raise ValueError("states must be non-empty")
        try:
            positions = {label: i for i, label in enumerate(labels)}
        except TypeError as error:
            raise TypeError("state labels must be hashable") from error
        if len(positions) != len(labels):
            raise ValueError("state labels must be distinct")
        size = len(labels)
        off = [[0.0] * size for _ in range(size)]
        if isinstance(self.rates, Mapping):
            for key, value in self.rates.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError("rate keys must be (from_state, to_state) pairs")
                source, target = key
                if source not in positions or target not in positions:
                    raise ValueError(f"rate key {key!r} names an unknown state")
                if source == target:
                    raise ValueError(f"rate key {key!r} is on the diagonal")
                rate = require_finite_real(value, name=f"rate{key!r}")
                if rate < 0.0:
                    raise ValueError(f"rate {key!r} must be non-negative")
                off[positions[source]][positions[target]] = rate
        else:
            matrix = as_matrix(self.rates, name="generator", square=True)
            if len(matrix) != size:
                raise ValueError("generator size must equal the number of states")
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    if i != j:
                        if value < 0.0:
                            raise ValueError(
                                f"generator[{i}][{j}] must be non-negative"
                            )
                        off[i][j] = value
                outflow = math.fsum(off[i])
                residual = abs(outflow + row[i])
                if residual > _ROW_SUM_TOLERANCE * max(outflow, abs(row[i])):
                    raise ValueError(f"generator row {i} does not sum to zero")
        generator = tuple(
            tuple(-math.fsum(off[i]) if i == j else off[i][j] for j in range(size))
            for i in range(size)
        )
        successors = tuple(
            tuple((j, off[i][j]) for j in range(size) if j != i and off[i][j] > 0.0)
            for i in range(size)
        )
        reach = [[i == j or off[i][j] > 0.0 for j in range(size)] for i in range(size)]
        for k in range(size):
            row_k = reach[k]
            for i in range(size):
                if reach[i][k]:
                    row_i = reach[i]
                    for j in range(size):
                        if row_k[j]:
                            row_i[j] = True
        normalized = {
            (labels[i], labels[j]): off[i][j]
            for i in range(size)
            for j in range(size)
            if j != i and off[i][j] > 0.0
        }
        object.__setattr__(self, "states", labels)
        object.__setattr__(self, "rates", MappingProxyType(normalized))
        object.__setattr__(self, "_generator", generator)
        object.__setattr__(self, "_positions", MappingProxyType(positions))
        object.__setattr__(self, "_successors", successors)
        object.__setattr__(self, "_reach", tuple(tuple(row) for row in reach))

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of states.

        References:
            [REF-NORRIS-1997, Section 2.1].

        """
        return len(self.states)

    @property
    def generator(self) -> Matrix:
        """Return the generator matrix ``Q`` in state order.

        Equation:
            [EQ-PROC-CTMC-001].

        References:
            [REF-NORRIS-1997, Section 2.1].

        """
        return self._generator

    def index(self, state: State) -> int:
        """Return the position of a state label.

        Args:
            state: State label.

        Returns:
            Index in :attr:`states`.

        Raises:
            ValueError: If the label is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        try:
            return self._positions[state]
        except (KeyError, TypeError) as error:
            raise ValueError(f"unknown state {state!r}") from error

    def rate(self, source: State, target: State) -> float:
        """Return the transition rate ``q_ij`` (``q_ii`` on the diagonal).

        Args:
            source: From-state label.
            target: To-state label.

        Returns:
            Generator entry.

        Raises:
            ValueError: If a label is unknown.

        References:
            [REF-NORRIS-1997, Section 2.1].

        """
        return self._generator[self.index(source)][self.index(target)]

    def exit_rate(self, state: State) -> float:
        """Return the total exit rate ``-q_ii`` of a state.

        Args:
            state: State label.

        Returns:
            Exit rate; zero for an absorbing state.

        Raises:
            ValueError: If the label is unknown.

        References:
            [REF-NORRIS-1997, Section 2.6].

        """
        i = self.index(state)
        return -self._generator[i][i]

    def absorbing_states(self) -> tuple[State, ...]:
        """Return the labels of states with zero exit rate.

        References:
            [REF-NORRIS-1997, Section 3.2].

        """
        return tuple(
            label
            for label, successors in zip(self.states, self._successors, strict=True)
            if not successors
        )

    def communicating_classes(self) -> tuple[tuple[State, ...], ...]:
        """Return the communicating classes ordered by first member.

        Returns:
            Tuples of labels partitioning the state space.

        References:
            [REF-NORRIS-1997, Section 3.2].

        """
        return tuple(
            tuple(self.states[i] for i in members) for members in self._classes()
        )

    def closed_classes(self) -> tuple[tuple[State, ...], ...]:
        """Return the closed communicating classes (recurrent classes).

        References:
            [REF-NORRIS-1997, Section 3.2].

        """
        return tuple(
            tuple(self.states[i] for i in members) for members in self._closed()
        )

    def is_irreducible(self) -> bool:
        """Return whether all states communicate.

        References:
            [REF-NORRIS-1997, Section 3.2].

        """
        return len(self._classes()) == 1

    def _classes(self) -> tuple[tuple[int, ...], ...]:
        """Return communicating classes as index tuples.

        References:
            [REF-NORRIS-1997, Section 3.2]. No equation is evaluated.

        """
        assigned: set[int] = set()
        classes: list[tuple[int, ...]] = []
        for i in range(self.size):
            if i in assigned:
                continue
            members = tuple(
                j for j in range(self.size) if self._reach[i][j] and self._reach[j][i]
            )
            assigned.update(members)
            classes.append(members)
        return tuple(classes)

    def _closed(self) -> tuple[tuple[int, ...], ...]:
        """Return closed classes as index tuples.

        References:
            [REF-NORRIS-1997, Section 3.2]. No equation is evaluated.

        """
        closed: list[tuple[int, ...]] = []
        for members in self._classes():
            inside = set(members)
            if all(j in inside for i in members for j, _ in self._successors[i]):
                closed.append(members)
        return tuple(closed)

    def can_reach(self, source: State, targets: Iterable[State]) -> bool:
        """Return whether some target is reachable from ``source``.

        Args:
            source: Start label.
            targets: Target labels.

        Returns:
            Whether a path of positive-rate transitions (possibly empty)
            leads from ``source`` into the targets.

        Raises:
            ValueError: If a label is unknown.

        References:
            [REF-NORRIS-1997, Section 3.2]. No equation is evaluated.

        """
        i = self.index(source)
        return any(self._reach[i][self.index(target)] for target in targets)

    def make_absorbing(self, states: Iterable[State]) -> ContinuousTimeMarkovChain:
        """Return a copy in which the given states have no outgoing rates.

        This is how reliability (survival up to the first entry into a set)
        is computed from an availability model.

        Args:
            states: Labels to make absorbing.

        Returns:
            New chain on the same labels.

        Raises:
            ValueError: If a label is unknown.

        References:
            [REF-NORRIS-1997, Section 3.2].

        """
        chosen = {self.index(state) for state in states}
        rates = {
            (self.states[i], self.states[j]): value
            for i, successors in enumerate(self._successors)
            if i not in chosen
            for j, value in successors
        }
        return ContinuousTimeMarkovChain(self.states, rates)

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    def distribution(self, initial: object) -> Vector:
        """Normalize an initial law to a probability vector in state order.

        Args:
            initial: A state label (a point mass), a mapping
                ``{label: probability}``, or a sequence of probabilities in
                state order. A value that is itself a state label is always
                read as a label.

        Returns:
            Probability vector.

        Raises:
            ValueError: If the law is unknown, negative, or does not sum to
                one within ``1e-10``.

        References:
            Project validation policy. No equation is evaluated.

        """
        try:
            if initial in self._positions:
                vector = [0.0] * self.size
                vector[self._positions[initial]] = 1.0
                return tuple(vector)
        except TypeError:
            pass
        if isinstance(initial, Mapping):
            vector = [0.0] * self.size
            for label, value in initial.items():
                probability = require_finite_real(value, name=f"initial[{label!r}]")
                vector[self.index(label)] += probability
            values: Vector = tuple(vector)
        elif isinstance(initial, Iterable) and not isinstance(initial, str):
            values = tuple(
                require_finite_real(value, name=f"initial[{i}]")
                for i, value in enumerate(initial)
            )
            if len(values) != self.size:
                raise ValueError("initial vector length must equal the state count")
        else:
            raise ValueError(f"unknown initial state {initial!r}")
        if any(value < 0.0 for value in values):
            raise ValueError("initial probabilities must be non-negative")
        if abs(math.fsum(values) - 1.0) > _DISTRIBUTION_TOLERANCE:
            raise ValueError("initial probabilities must sum to one")
        return values

    def state_values(self, values: object, *, name: str = "rewards") -> Vector:
        """Normalize per-state values (rewards) to a vector in state order.

        Args:
            values: Mapping ``{label: value}`` (missing labels are zero) or a
                sequence in state order.
            name: Parameter name for errors.

        Returns:
            Finite real vector.

        Raises:
            ValueError: If a label is unknown or the length is wrong.
            TypeError: If a value is not real.

        References:
            Project validation policy. No equation is evaluated.

        """
        if isinstance(values, Mapping):
            vector = [0.0] * self.size
            for label, value in values.items():
                vector[self.index(label)] = require_finite_real(
                    value, name=f"{name}[{label!r}]"
                )
            return tuple(vector)
        if isinstance(values, Iterable) and not isinstance(values, str):
            result = tuple(
                require_finite_real(value, name=f"{name}[{i}]")
                for i, value in enumerate(values)
            )
            if len(result) != self.size:
                raise ValueError(f"{name} length must equal the state count")
            return result
        raise TypeError(f"{name} must be a mapping or a sequence")

    # ------------------------------------------------------------------
    # Uniformization engine
    # ------------------------------------------------------------------

    def _uniformization_rate(self) -> float:
        """Return ``q = max_i |q_ii|``.

        References:
            [REF-JENSEN-1953]. No equation is evaluated.

        """
        return max(-self._generator[i][i] for i in range(self.size))

    def _uniformized_rows(
        self, rate: float
    ) -> tuple[tuple[tuple[int, float], ...], ...]:
        """Return the sparse rows of ``P = I + Q / q``.

        Args:
            rate: Uniformization rate ``q > 0``.

        Returns:
            Per-row ``(column, probability)`` pairs including the diagonal.

        Equation:
            [EQ-PROC-CTMC-002].

        References:
            [REF-JENSEN-1953], [REF-STEWART-1994, Section 8.4].

        """
        rows = []
        for i, successors in enumerate(self._successors):
            stay = (rate - (-self._generator[i][i])) / rate
            entries = [(j, value / rate) for j, value in successors]
            if stay > 0.0:
                entries.append((i, stay))
            rows.append(tuple(entries))
        return tuple(rows)

    def _series(
        self,
        vector: Sequence[float],
        rate: float,
        jobs: Sequence[tuple[int, Sequence[float]]],
        max_terms: int,
    ) -> tuple[list[list[float]], int]:
        """Accumulate ``sum_k c_k v P^k`` for several coefficient windows.

        Iterates ``v_(k+1) = v_k P`` once and adds ``c_k v_k`` to each job
        whose window contains ``k``. If an iterate reproduces its predecessor
        exactly in floating point, every later iterate is identical, so the
        remaining coefficients are summed in closed form; this changes no
        result. Each iterate is rescaled to the mass of ``v_0`` (exact in
        real arithmetic because ``P`` is stochastic), which removes the
        systematic drift caused by rounded row sums of ``P``.

        Args:
            vector: Starting non-negative vector ``v_0``.
            rate: Uniformization rate.
            jobs: ``(first_index, coefficients)`` pairs.
            max_terms: Maximum number of matrix-vector products.

        Returns:
            One accumulated vector per job, and the number of products used.

        Raises:
            ValueError: If more than ``max_terms`` products are required.

        Equation:
            [EQ-PROC-CTMC-002].

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        rows = self._uniformized_rows(rate)
        size = self.size
        last = max(start + len(coefficients) - 1 for start, coefficients in jobs)
        if last > max_terms:
            raise ValueError(
                f"uniformization needs {last} terms (q t is too large); raise "
                "max_terms or rescale the model"
            )
        sums = [[0.0] * size for _ in jobs]
        current = list(vector)
        mass = math.fsum(current)
        k = 0
        while True:
            for job, (start, coefficients) in enumerate(jobs):
                offset = k - start
                if 0 <= offset < len(coefficients):
                    weight = coefficients[offset]
                    if weight != 0.0:
                        accumulator = sums[job]
                        for j in range(size):
                            accumulator[j] += weight * current[j]
            if k >= last:
                return sums, k
            following = [0.0] * size
            for i, value in enumerate(current):
                if value != 0.0:
                    for j, probability in rows[i]:
                        following[j] += value * probability
            total = math.fsum(following)
            if total > 0.0 and total != mass:
                # P is stochastic, so v_k P keeps the mass of v_0 exactly in
                # real arithmetic; rescaling removes the systematic drift of
                # rounded row sums without changing the exact iteration.
                factor = mass / total
                following = [value * factor for value in following]
            k += 1
            if following == current:
                for job, (start, coefficients) in enumerate(jobs):
                    offset = max(k - start, 0)
                    if offset < len(coefficients):
                        weight = math.fsum(coefficients[offset:])
                        accumulator = sums[job]
                        for j in range(size):
                            accumulator[j] += weight * current[j]
                return sums, k
            current = following

    def _transient_window(
        self, rate: float, time: float, tolerance: float
    ) -> tuple[tuple[int, list[float]], float, int, int, int]:
        """Return the Poisson-window job for ``pi(t)``.

        Args:
            rate: Uniformization rate.
            time: Positive time.
            tolerance: Target L1 truncation bound.

        Returns:
            ``(job, truncation_bound, span, left, right)``.

        Equation:
            [EQ-PROC-CTMC-003].

        References:
            [REF-FOX-GLYNN-1988].

        """
        mean = rate * time
        left, right, tau_lower, tau_upper = _poisson_window(
            mean, tolerance / 4.0, tolerance / 4.0
        )
        weights, span = _poisson_weights(mean, left, right)
        truncation = 2.0 * (tau_lower + tau_upper) * _BOUND_SAFETY
        return (left, weights), truncation, span, left, right

    def _occupancy_window(
        self, rate: float, horizon: float, tolerance: float
    ) -> tuple[tuple[int, list[float]], float, int, int, int]:
        """Return the integrated-series job for ``int_0^t pi(s) ds``.

        Coefficients are ``Pr(N > k) / q`` for ``k < R`` with
        ``N ~ Poisson(q t)``; the window is widened until the truncation
        bound is at most ``tolerance * t``.

        Args:
            rate: Uniformization rate.
            horizon: Positive horizon.
            tolerance: Target truncation error relative to ``horizon``.

        Returns:
            ``(job, truncation_bound, span, left, right)``; the truncation
            bound is an L1 bound in time units.

        Equation:
            [EQ-PROC-CTMC-004].

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        mean = rate * horizon
        tail = tolerance / 8.0
        while True:
            left, right, tau_lower, tau_upper = _poisson_window(mean, tail, tail)
            tau = tau_lower + tau_upper
            truncation = (
                (
                    mean * tau_upper
                    + left * tau_lower
                    + (right - left) * (tau / (1.0 - tau) + tau_upper)
                )
                / rate
                * _BOUND_SAFETY
            )
            if truncation <= tolerance * horizon or tail <= _MIN_TOLERANCE:
                break
            tail = max(_MIN_TOLERANCE, tail * tolerance * horizon / truncation / 2.0)
        weights, span = _poisson_weights(mean, left, right)
        coefficients = [1.0 / rate] * right
        running = 0.0
        for k in range(right - 1, left - 1, -1):
            running += weights[k + 1 - left]
            coefficients[k] = running / rate
        return (0, coefficients), truncation, span, left, right

    def _plan(self, rate: float, horizon: float, method: str) -> int:
        """Return the number of squarings ``s`` (0 means the direct series).

        ``"auto"`` compares the cost of the direct series (about ``q t``
        sparse products) with scaling and squaring (``n`` series of about
        ``q t / 2^s`` products plus ``s`` dense ``n x n`` products).

        Args:
            rate: Uniformization rate.
            horizon: Positive time.
            method: ``"auto"``, ``"series"`` or ``"squaring"``.

        Returns:
            Number of squarings.

        Raises:
            ValueError: If the method is unknown.

        References:
            [REF-MOLER-VAN-LOAN-2003], [REF-REIBMAN-TRIVEDI-1988].

        """
        if method not in ("auto", "series", "squaring"):
            raise ValueError("method must be 'auto', 'series' or 'squaring'")
        mean = rate * horizon
        squarings = max(0, math.ceil(math.log2(mean / _SQUARING_BLOCK)))
        if method == "series":
            return 0
        if method == "squaring":
            return max(1, squarings)
        size = self.size
        work = sum(len(row) for row in self._successors) + 2 * size
        series_cost = (mean + 10.0 * math.sqrt(mean) + 10.0) * work
        block = mean / 2.0**squarings
        squaring_cost = size * (block + 10.0 * math.sqrt(block) + 10.0) * work + (
            squarings * size**3
        )
        return squarings if squarings > 0 and squaring_cost < series_cost else 0

    def _squared(
        self,
        rate: float,
        horizon: float,
        tolerance: float,
        squarings: int,
        max_terms: int,
        *,
        occupancy: bool,
    ) -> tuple[list[list[float]], float, float, list[list[float]], float, float]:
        """Return ``exp(Q t)`` (and ``int_0^t exp(Q s) ds``) by squaring.

        With ``tau = t / 2^s`` the rows of ``E = exp(Q tau)`` and
        ``M = int_0^tau exp(Q s) ds`` come from uniformization of each unit
        vector; then ``M <- M + E M`` and ``E <- E E`` are applied ``s``
        times [EQ-PROC-CTMC-008]. All factors are non-negative, so rounding
        stays componentwise relative. If the rows of ``E`` have L1 errors
        ``d``, the squared rows have at most ``2 d + d^2``; ``M`` errors
        follow ``d_M <- 2 d_M + d tau_k``.

        Args:
            rate: Uniformization rate.
            horizon: Positive time ``t``.
            tolerance: Target L1 truncation error (relative to ``t`` for
                ``M``).
            squarings: Number of squarings ``s >= 1``.
            max_terms: Product limit per row series.
            occupancy: Whether to compute ``M`` as well.

        Returns:
            ``(E, d_E, rho_E, M, d_M, rho_M)``: matrices, row-L1 truncation
            bounds and componentwise relative rounding bounds (``M`` empty
            when not requested).

        Equation:
            [EQ-PROC-CTMC-008].

        References:
            [REF-MOLER-VAN-LOAN-2003], [REF-REIBMAN-TRIVEDI-1988].

        """
        size = self.size
        step = math.ldexp(horizon, -squarings)
        row_tolerance = max(_MIN_TOLERANCE, math.ldexp(tolerance, -(squarings + 1)))
        transient_job, d_e, span_e, left_e, right_e = self._transient_window(
            rate, step, row_tolerance
        )
        jobs: list[tuple[int, Sequence[float]]] = [transient_job]
        d_m = 0.0
        span_m = right_m = 0
        if occupancy:
            occupancy_job, d_m, span_m, _, right_m = self._occupancy_window(
                rate, step, tolerance / 2.0
            )
            jobs.append(occupancy_job)
        rows_e: list[list[float]] = []
        rows_m: list[list[float]] = []
        products = 0
        for i in range(size):
            unit = [0.0] * size
            unit[i] = 1.0
            sums, used = self._series(unit, rate, jobs, max_terms)
            products = max(products, used)
            rows_e.append(_rescaled(sums[0]))
            if occupancy:
                rows_m.append(sums[1])
        rho_e = _UNIT_ROUNDOFF * (
            products * (size + 3) + 2 * span_e + (right_e - left_e) + 4
        )
        rho_m = _UNIT_ROUNDOFF * (products * (size + 3) + 2 * span_m + 2 * right_m + 6)
        current = step
        for _ in range(squarings):
            if occupancy:
                rows_m = [
                    [
                        rows_m[i][j]
                        + sum(rows_e[i][k] * rows_m[k][j] for k in range(size))
                        for j in range(size)
                    ]
                    for i in range(size)
                ]
                d_m = 2.0 * d_m + d_e * current + d_e * d_m
                rho_m = rho_e + rho_m + (size + 1) * _UNIT_ROUNDOFF
            rows_e = [
                [
                    sum(rows_e[i][k] * rows_e[k][j] for k in range(size))
                    for j in range(size)
                ]
                for i in range(size)
            ]
            # The truncated E is exactly row-stochastic in real arithmetic
            # (normalized Poisson weights), and so is every power of it, so
            # rescaling rows to sum one removes rounding drift only.
            rows_e = [_rescaled(row) for row in rows_e]
            d_e = 2.0 * d_e + d_e * d_e
            rho_e = 2.0 * rho_e + rho_e * rho_e + size * _UNIT_ROUNDOFF
            current *= 2.0
        return rows_e, d_e, rho_e, rows_m, d_m, rho_m

    # ------------------------------------------------------------------
    # Transient analysis
    # ------------------------------------------------------------------

    def transient(
        self,
        initial: object,
        t: float,
        tolerance: float = 1e-12,
        *,
        method: str = "auto",
        max_terms: int = _DEFAULT_MAX_TERMS,
    ) -> TransientSolution:
        """Return ``pi(t) = pi(0) exp(Q t)`` by uniformization.

        **Series** (default for moderate ``q t``): the Poisson(``q t``)
        series is truncated to ``[L, R]`` with ``Pr(N < L) <= tolerance / 4``
        and ``Pr(N >= R) <= tolerance / 4`` certified by Cramer-Chernoff
        bounds evaluated in log space. The retained weights are normalized,
        so the L1 truncation error is at most ``2 Pr(N outside [L, R]) <=
        tolerance``. Because ``P``, ``pi(0)`` and the weights are
        non-negative, floating-point errors are componentwise relative; their
        first-order bound ``u [m (n + 3) + 2 span + (R - L) + 4]``
        (``u = 2^-53``, ``m`` products, ``n`` states, ``span`` the largest
        distance from the Poisson mode) is reported separately. The cost is
        about ``q t + O(sqrt(q t))`` sparse matrix-vector products.

        **Squaring** (chosen by ``"auto"`` when cheaper, typically for
        ``q t`` well above ``1e4`` and few states): ``exp(Q t / 2^s)`` is
        built row by row with the series and squared ``s`` times; the
        truncation bound doubles with each squaring and is budgeted for.

        Args:
            initial: Label, mapping, or probability vector (see
                :meth:`distribution`).
            t: Non-negative time.
            tolerance: Target L1 truncation error in ``[1e-250, 0.5]``.
            method: ``"auto"``, ``"series"`` or ``"squaring"``.
            max_terms: Upper limit on matrix-vector products per series.

        Returns:
            Transient solution with error bounds.

        Raises:
            ValueError: If an input is invalid or more than ``max_terms``
                products are needed.

        Equation:
            [EQ-PROC-CTMC-002], [EQ-PROC-CTMC-003], [EQ-PROC-CTMC-008].

        References:
            [REF-JENSEN-1953], [REF-GRASSMANN-1977], [REF-REIBMAN-TRIVEDI-1988],
            [REF-FOX-GLYNN-1988].

        """
        return self.transient_many(
            initial, (t,), tolerance, method=method, max_terms=max_terms
        )[0]

    def transient_many(
        self,
        initial: object,
        times: Iterable[float],
        tolerance: float = 1e-12,
        *,
        method: str = "auto",
        max_terms: int = _DEFAULT_MAX_TERMS,
    ) -> tuple[TransientSolution, ...]:
        """Return ``pi(t)`` at several times.

        Times solved by the direct series share one pass: the iterates
        ``pi(0) P^k`` are computed once and each time has its own Poisson
        window. Times for which squaring is cheaper are solved separately.
        See :meth:`transient`.

        Args:
            initial: Label, mapping, or probability vector.
            times: Non-negative times (any order).
            tolerance: Target L1 truncation error per time.
            method: ``"auto"``, ``"series"`` or ``"squaring"``.
            max_terms: Upper limit on matrix-vector products per series.

        Returns:
            One solution per time, in the given order.

        Raises:
            ValueError: If an input is invalid.

        Equation:
            [EQ-PROC-CTMC-002], [EQ-PROC-CTMC-003], [EQ-PROC-CTMC-008].

        References:
            [REF-REIBMAN-TRIVEDI-1988], [REF-FOX-GLYNN-1988].

        """
        start = self.distribution(initial)
        checked = tuple(_require_time(time) for time in times)
        accuracy = _require_tolerance(tolerance)
        rate = self._uniformization_rate()
        size = self.size
        results: list[TransientSolution | None] = [None] * len(checked)
        jobs: list[tuple[int, Sequence[float]]] = []
        meta: list[tuple[int, float, int, int, int]] = []
        for position, time in enumerate(checked):
            if time == 0.0 or rate == 0.0:
                results[position] = TransientSolution(
                    self.states, time, start, 0.0, 0.0, 0.0, rate, "exact", 0
                )
                continue
            squarings = self._plan(rate, time, method)
            if squarings:
                rows, d_e, rho_e, _, _, _ = self._squared(
                    rate, time, accuracy, squarings, max_terms, occupancy=False
                )
                vector = tuple(
                    math.fsum(start[i] * rows[i][j] for i in range(size))
                    for j in range(size)
                )
                rounding = (rho_e + (size + 1) * _UNIT_ROUNDOFF) * _BOUND_SAFETY
                truncation = d_e * _BOUND_SAFETY
                results[position] = TransientSolution(
                    self.states,
                    time,
                    vector,
                    truncation + rounding * max(1.0, math.fsum(vector)),
                    truncation,
                    rounding,
                    rate,
                    f"squaring (s={squarings})",
                    squarings,
                )
                continue
            job, truncation, span, left, right = self._transient_window(
                rate, time, accuracy
            )
            jobs.append(job)
            meta.append((position, truncation, span, left, right))
        if jobs:
            sums, products = self._series(start, rate, jobs, max_terms)
            for (position, truncation, span, left, right), vector_list in zip(
                meta, sums, strict=True
            ):
                rounding = (
                    _UNIT_ROUNDOFF
                    * (products * (size + 3) + 2 * span + (right - left) + 4)
                    * _BOUND_SAFETY
                )
                probabilities = tuple(vector_list)
                results[position] = TransientSolution(
                    self.states,
                    checked[position],
                    probabilities,
                    truncation + rounding * max(1.0, math.fsum(probabilities)),
                    truncation,
                    rounding,
                    rate,
                    "series",
                    right,
                )
        return tuple(result for result in results if result is not None)

    def expected_time_in_states(
        self,
        initial: object,
        t: float,
        tolerance: float = 1e-12,
        *,
        method: str = "auto",
        max_terms: int = _DEFAULT_MAX_TERMS,
    ) -> OccupancySolution:
        """Return ``L(t) = int_0^t pi(s) ds``, the expected time in each state.

        Uses ``L(t) = (1/q) sum_(k >= 0) Pr(N > k) pi(0) P^k`` with
        ``N ~ Poisson(q t)`` [EQ-PROC-CTMC-004]. Terms beyond ``R`` contribute
        at most ``t Pr(N >= R)`` in L1 (because
        ``E[(N - R)^+] <= q t Pr(N >= R)``); the tail probabilities
        ``Pr(N > k)`` are formed from the normalized window weights and set to
        one below ``L``, which adds at most ``(L tau_L + (R - L)(tau /
        (1 - tau) + tau_U)) / q``. The window is widened until this
        truncation bound is at most ``tolerance * t``. A first-order relative
        rounding bound is reported as in :meth:`transient`. For large ``q t``
        the doubling ``M(2 tau) = M(tau) + exp(Q tau) M(tau)`` is used
        instead (``method="squaring"``) [EQ-PROC-CTMC-008].

        Args:
            initial: Label, mapping, or probability vector.
            t: Non-negative horizon.
            tolerance: Target truncation error relative to ``t``.
            method: ``"auto"``, ``"series"`` or ``"squaring"``.
            max_terms: Upper limit on matrix-vector products per series.

        Returns:
            Occupancy times with an L1 error bound in time units.

        Raises:
            ValueError: If an input is invalid.

        Equation:
            [EQ-PROC-CTMC-004], [EQ-PROC-CTMC-008].

        References:
            [REF-REIBMAN-TRIVEDI-1988], [REF-GRASSMANN-1977].

        """
        start = self.distribution(initial)
        horizon = _require_time(t)
        accuracy = _require_tolerance(tolerance)
        rate = self._uniformization_rate()
        size = self.size
        if horizon == 0.0 or rate == 0.0:
            occupancy = tuple(horizon * value for value in start)
            return OccupancySolution(
                self.states, horizon, occupancy, 0.0, 0.0, 0.0, rate, "exact", 0
            )
        squarings = self._plan(rate, horizon, method)
        if squarings:
            _, _, _, rows, d_m, rho_m = self._squared(
                rate, horizon, accuracy, squarings, max_terms, occupancy=True
            )
            occupancy = tuple(
                math.fsum(start[i] * rows[i][j] for i in range(size))
                for j in range(size)
            )
            truncation = d_m * _BOUND_SAFETY
            rounding = (rho_m + (size + 1) * _UNIT_ROUNDOFF) * _BOUND_SAFETY
            label = f"squaring (s={squarings})"
            terms = squarings
        else:
            job, truncation, span, _, right = self._occupancy_window(
                rate, horizon, accuracy
            )
            sums, products = self._series(start, rate, [job], max_terms)
            occupancy = tuple(sums[0])
            rounding = (
                _UNIT_ROUNDOFF
                * (products * (size + 3) + 2 * span + 2 * right + 6)
                * _BOUND_SAFETY
            )
            label = "series"
            terms = right
        return OccupancySolution(
            self.states,
            horizon,
            occupancy,
            truncation + rounding * max(horizon, math.fsum(occupancy)),
            truncation,
            rounding,
            rate,
            label,
            terms,
        )

    def expected_accumulated_reward(
        self,
        initial: object,
        t: float,
        rewards: object,
        tolerance: float = 1e-12,
        *,
        method: str = "auto",
        max_terms: int = _DEFAULT_MAX_TERMS,
    ) -> BoundedValue:
        """Return ``E[int_0^t r(X_s) ds] = L(t) . r`` for rate rewards ``r``.

        This is how mission cost (cost rate per state) or downtime (reward one
        on the down states) is computed.

        Args:
            initial: Label, mapping, or probability vector.
            t: Non-negative horizon.
            rewards: Reward rate per state (mapping or sequence).
            tolerance: Target truncation error of ``L(t)`` relative to ``t``.
            method: ``"auto"``, ``"series"`` or ``"squaring"``.
            max_terms: Upper limit on matrix-vector products per series.

        Returns:
            Expected accumulated reward; the bound is the L1 truncation bound
            of ``L(t)`` times ``max |r_i|`` plus the relative rounding bound
            times ``sum |L_i r_i|``.

        Raises:
            ValueError: If an input is invalid.

        Equation:
            [EQ-PROC-CTMC-004].

        References:
            [REF-REIBMAN-TRIVEDI-1988].

        """
        reward = self.state_values(rewards)
        occupancy = self.expected_time_in_states(
            initial, t, tolerance, method=method, max_terms=max_terms
        )
        return occupancy.weighted(reward)

    def expected_transition_count(
        self,
        initial: object,
        t: float,
        transitions: Iterable[tuple[State, State]],
        tolerance: float = 1e-12,
        *,
        method: str = "auto",
        max_terms: int = _DEFAULT_MAX_TERMS,
    ) -> BoundedValue:
        """Return the expected number of given transitions in ``[0, t]``.

        ``E[#{i -> j in [0, t]}] = q_ij int_0^t pi_i(s) ds``, summed over the
        given pairs (an impulse reward of one per transition)
        [EQ-PROC-CTMC-004].

        Args:
            initial: Label, mapping, or probability vector.
            t: Non-negative horizon.
            transitions: ``(from, to)`` label pairs (duplicates ignored).
            tolerance: Target truncation error of ``L(t)`` relative to ``t``.
            method: ``"auto"``, ``"series"`` or ``"squaring"``.
            max_terms: Upper limit on matrix-vector products per series.

        Returns:
            Expected count with error bound.

        Raises:
            ValueError: If a label is unknown or a pair is on the diagonal.

        Equation:
            [EQ-PROC-CTMC-004].

        References:
            [REF-REIBMAN-TRIVEDI-1988], [REF-NORRIS-1997, Section 2.6].

        """
        flux = [0.0] * self.size
        for source, target in set(transitions):
            i, j = self.index(source), self.index(target)
            if i == j:
                raise ValueError("a transition must change state")
            flux[i] += self._generator[i][j]
        occupancy = self.expected_time_in_states(
            initial, t, tolerance, method=method, max_terms=max_terms
        )
        return occupancy.weighted(flux)

    # ------------------------------------------------------------------
    # Long-run and absorption analysis
    # ------------------------------------------------------------------

    def _off_diagonal(self, members: Sequence[int]) -> list[list[float]]:
        """Return the off-diagonal rates restricted to ``members``.

        Args:
            members: State indices.

        Returns:
            Square rate matrix with zero diagonal.

        References:
            [REF-NORRIS-1997, Section 2.1]. No equation is evaluated.

        """
        return [
            [0.0 if a == b else self._generator[i][j] for b, j in enumerate(members)]
            for a, i in enumerate(members)
        ]

    def steady_state(self) -> Vector:
        """Return the unique stationary law ``pi Q = 0``, ``sum(pi) = 1``.

        Solved by GTH state reduction on the single closed class
        [EQ-PROC-CTMC-005]; transient states receive zero mass. GTH uses no
        subtraction, so tiny probabilities (e.g. ``1e-12`` unavailability)
        keep small relative error.

        Returns:
            Stationary probability vector in state order.

        Raises:
            ValueError: If there is more than one closed class (use
                :meth:`limiting_distribution` with an initial law).

        Equation:
            [EQ-PROC-CTMC-005].

        References:
            [REF-GRASSMANN-TAKSAR-HEYMAN-1985], [REF-NORRIS-1997, Section 3.5].

        """
        closed = self._closed()
        if len(closed) != 1:
            raise ValueError(
                f"stationary law is not unique: {len(closed)} closed classes; "
                "use limiting_distribution(initial)"
            )
        return self._class_law(closed[0])

    def _class_law(self, members: Sequence[int]) -> Vector:
        """Return the stationary law of one closed class embedded in all states.

        Args:
            members: Indices of a closed class.

        Returns:
            Probability vector supported on the class.

        References:
            [REF-GRASSMANN-TAKSAR-HEYMAN-1985].

        """
        local = _gth_stationary(self._off_diagonal(members))
        vector = [0.0] * self.size
        for i, value in zip(members, local, strict=True):
            vector[i] = value
        return tuple(vector)

    def _hitting(self, start: Vector) -> list[float]:
        """Return the probability of eventually entering each closed class.

        Args:
            start: Initial probability vector.

        Returns:
            One probability per closed class (order of :meth:`_closed`).

        Equation:
            [EQ-PROC-CTMC-006].

        References:
            [REF-NORRIS-1997, Section 3.3].

        """
        closed = self._closed()
        owner = {i: c for c, members in enumerate(closed) for i in members}
        transient = [i for i in range(self.size) if i not in owner]
        result = [0.0] * len(closed)
        for i, c in owner.items():
            result[c] += start[i]
        if transient and any(start[i] > 0.0 for i in transient):
            exits = [0.0] * len(transient)
            rhs = [[0.0] * len(closed) for _ in transient]
            for a, i in enumerate(transient):
                for j, value in self._successors[i]:
                    if j in owner:
                        exits[a] += value
                        rhs[a][owner[j]] += value
            solution = _solve_exit_system(self._off_diagonal(transient), exits, rhs)
            for a, i in enumerate(transient):
                if start[i] > 0.0:
                    for c in range(len(closed)):
                        result[c] += start[i] * solution[a][c]
        return result

    def limiting_distribution(self, initial: object) -> Vector:
        """Return ``lim_(t -> inf) pi(t)`` for the given initial law.

        ``sum_C Pr(enter C) pi_C`` over the closed classes ``C``, where
        ``pi_C`` is the GTH stationary law of class ``C`` and the entry
        probabilities come from the subtraction-free absorption solve.

        Args:
            initial: Label, mapping, or probability vector.

        Returns:
            Limiting probability vector.

        Raises:
            ValueError: If the initial law is invalid.

        Equation:
            [EQ-PROC-CTMC-005], [EQ-PROC-CTMC-006].

        References:
            [REF-NORRIS-1997, Sections 3.3-3.6].

        """
        start = self.distribution(initial)
        weights = self._hitting(start)
        vector = [0.0] * self.size
        for weight, members in zip(weights, self._closed(), strict=True):
            if weight > 0.0:
                law = self._class_law(members)
                for i in members:
                    vector[i] += weight * law[i]
        return tuple(vector)

    def absorption(self, initial: object) -> CTMCAbsorption:
        """Return absorption probabilities and mean times to absorption.

        With transient block ``Q_TT`` and transient-to-absorbing block
        ``Q_TA``: ``N = (-Q_TT)^-1``, ``B = N Q_TA``, ``m = N 1``
        [EQ-PROC-CTMC-006]. The system is solved by subtraction-free
        elimination, so mean times such as an MTTF of ``1e15`` with rates
        ``1e-6`` and ``1e3`` keep small relative error.

        Args:
            initial: Label, mapping, or probability vector.

        Returns:
            Absorption analysis including results for ``initial``.

        Raises:
            ValueError: If there is no absorbing or no transient state, or a
                closed class is not a single absorbing state.

        Equation:
            [EQ-PROC-CTMC-006].

        References:
            [REF-NORRIS-1997, Section 3.3], [REF-KEMENY-SNELL-1976, Chapter III],
            [REF-GRASSMANN-TAKSAR-HEYMAN-1985].

        """
        start = self.distribution(initial)
        closed = self._closed()
        if any(len(members) != 1 for members in closed):
            raise ValueError(
                "every closed class must be a single absorbing state; make "
                "recurrent classes absorbing first (make_absorbing)"
            )
        absorbing = [members[0] for members in closed]
        if not absorbing:
            raise ValueError("chain has no absorbing state")
        column = {i: c for c, i in enumerate(absorbing)}
        transient = [i for i in range(self.size) if i not in column]
        if not transient:
            raise ValueError("chain has no transient state")
        width = len(absorbing)
        size = len(transient)
        exits = [0.0] * size
        rhs = [[0.0] * (width + size) for _ in range(size)]
        for a, i in enumerate(transient):
            rhs[a][width + a] = 1.0
            for j, value in self._successors[i]:
                if j in column:
                    exits[a] += value
                    rhs[a][column[j]] += value
        solution = _solve_exit_system(self._off_diagonal(transient), exits, rhs)
        probabilities = tuple(tuple(row[:width]) for row in solution)
        fundamental = tuple(tuple(row[width:]) for row in solution)
        mean_times = tuple(math.fsum(row) for row in fundamental)
        initial_absorption = [start[i] for i in absorbing]
        for a, i in enumerate(transient):
            if start[i] > 0.0:
                for c in range(width):
                    initial_absorption[c] += start[i] * probabilities[a][c]
        initial_mean = math.fsum(
            start[i] * mean_times[a] for a, i in enumerate(transient)
        )
        return CTMCAbsorption(
            tuple(self.states[i] for i in transient),
            tuple(self.states[i] for i in absorbing),
            fundamental,
            probabilities,
            mean_times,
            tuple(initial_absorption),
            initial_mean,
        )

    def embedded_chain(self) -> MarkovChain:
        """Return the embedded jump chain as a discrete-time :class:`MarkovChain`.

        ``P_ij = q_ij / (-q_ii)`` for ``i != j``; an absorbing state gets
        ``P_ii = 1`` [EQ-PROC-CTMC-007]. Labels are converted with ``str``.

        Returns:
            Jump chain.

        Raises:
            ValueError: If two labels have the same string form.

        Equation:
            [EQ-PROC-CTMC-007].

        References:
            [REF-NORRIS-1997, Section 2.6].

        """
        size = self.size
        rows = []
        for i in range(size):
            exit_rate = -self._generator[i][i]
            if exit_rate == 0.0:
                rows.append(tuple(1.0 if j == i else 0.0 for j in range(size)))
            else:
                rows.append(
                    tuple(
                        0.0 if j == i else self._generator[i][j] / exit_rate
                        for j in range(size)
                    )
                )
        return MarkovChain(tuple(rows), tuple(str(label) for label in self.states))

    def simulate(
        self,
        initial: object,
        horizon: float,
        *,
        seed: int | RandomStream | None = None,
    ) -> CTMCPath:
        """Simulate one path on ``[0, horizon]`` (cross-check helper only).

        Holding times are exponential with the exit rate and jumps follow the
        embedded chain [EQ-PROC-CTMC-007]. Use the analytic methods for
        answers; this exists to cross-check them by Monte Carlo.

        Args:
            initial: Label, mapping, or probability vector.
            horizon: Non-negative final time.
            seed: Integer seed or stream; ``None`` is not reproducible.

        Returns:
            Simulated path.

        Raises:
            ValueError: If an input is invalid.

        Equation:
            [EQ-PROC-CTMC-007].

        References:
            [REF-NORRIS-1997, Section 2.6], [REF-DEVROYE-1986, Chapter III].

        """
        start = self.distribution(initial)
        end = _require_time(horizon, name="horizon")
        generator = resolve_stream(seed).child("ctmc").generator()
        state = _draw(start, generator.random())
        times = [0.0]
        visited = [self.states[state]]
        now = 0.0
        while True:
            exit_rate = -self._generator[state][state]
            if exit_rate == 0.0:
                break
            now += generator.expovariate(exit_rate)
            if now >= end:
                break
            successors = self._successors[state]
            weights = [value / exit_rate for _, value in successors]
            state = successors[_draw(weights, generator.random())][0]
            times.append(now)
            visited.append(self.states[state])
        return CTMCPath(tuple(visited), tuple(times), end)


def _draw(probabilities: Sequence[float], uniform: float) -> int:
    """Return an index drawn by inversion of a probability vector.

    Args:
        probabilities: Non-negative weights summing to (about) one.
        uniform: Draw from ``[0, 1)``.

    Returns:
        Index whose cumulative probability first exceeds ``uniform``; zero
        weights are never returned.

    References:
        [REF-DEVROYE-1986, Chapter III]. No named equation is needed.

    """
    running = 0.0
    last = 0
    for index, value in enumerate(probabilities):
        if value > 0.0:
            running += value
            last = index
            if uniform < running:
                return index
    return last
