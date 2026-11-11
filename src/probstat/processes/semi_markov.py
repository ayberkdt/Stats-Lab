"""Finite semi-Markov processes solved by the Markov renewal equation.

A homogeneous semi-Markov process on finitely many labelled states is given
by its *semi-Markov kernel* ``Q_ij(t) = p_ij F_ij(t)`` [EQ-PROC-SMP-001]:
on entering state ``i`` the next state ``j`` is chosen with the embedded
jump-chain probability ``p_ij`` and the sojourn in ``i`` before the jump has
the law ``F_ij``, an arbitrary distribution on ``(0, inf)`` (any object with
``cdf`` and ``mean``; ``sf`` is used when present and ``quantile`` is needed
for simulation). A state without outgoing transitions is absorbing. With
exponential sojourns ``F_ij = Exp(q_i)`` the process is the continuous-time
Markov chain with rates ``q_ij = q_i p_ij`` (:meth:`SemiMarkovProcess.from_ctmc`).

Numerical methods (Limnios and Oprişan 2001, Chapters 2, 3 and 5; Çinlar
1969):

- **Interval transition probabilities** ``phi_ij(t) = P(Z_t = j | Z_0 = i)``
  and, more generally, expected rewards ``u_i(t) = E_i[r(Z_t)]`` solve the
  Markov renewal equation ``u(t) = (1 - H(t)) r + int_0^t dQ(s) u(t - s)``
  [EQ-PROC-SMP-002]. It is discretised on a uniform grid with the
  trapezoidal rule in the integrand and exact kernel increments, which gives
  an implicit second-order scheme, solved twice (``n`` and ``2n`` steps)
  and combined by Richardson extrapolation (Corradi, Janssen and Manca 2004;
  Linz 1985, Chapters 7-8) [EQ-PROC-SMP-003]. The reported error estimate is
  the Richardson estimate ``|u_2n - u_n| / 3`` of the error of the finer
  solution; it is an estimate, not a rigorous bound.
- **Absorption**: absorption probabilities, the fundamental matrix
  ``N = (I - P_TT)^-1`` of expected visits, and mean times to absorption
  ``N tau`` from the mean sojourn times ``tau_i = sum_j p_ij E[T_ij]``
  [EQ-PROC-SMP-004], solved exactly (to rounding) by the subtraction-free
  elimination of :class:`~probstat.processes.ContinuousTimeMarkovChain`.
- **Limiting law** ``pi_j = nu_j tau_j / sum_k nu_k tau_k`` from the
  stationary law ``nu`` of the embedded chain [EQ-PROC-SMP-005].
- **Seeded simulation** as an independent cross-check [EQ-PROC-SMP-006].
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import mul
from types import MappingProxyType

from probstat.distributions import Exponential
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector
from probstat.processes.ctmc import ContinuousTimeMarkovChain, CTMCPath

State = Hashable

_ROW_SUM_TOLERANCE = 1e-10
_DISTRIBUTION_TOLERANCE = 1e-10
_DEFAULT_STEPS = 256
_MAX_STEPS = 20_000


@dataclass(frozen=True, slots=True)
class SemiMarkovEstimate:
    """A value from the discretised Markov renewal equation with its error estimate.

    ``float(result)`` returns the value.

    Args:
        value: Richardson-extrapolated value.
        error_estimate: Estimated absolute error ``|u_2n - u_n| / 3`` of the
            finer trapezoidal solution (an estimate, not a rigorous bound).
        steps: Number ``n`` of steps of the coarser grid.

    References:
        [EQ-PROC-SMP-003], [REF-CORRADI-JANSSEN-MANCA-2004].

    """

    value: float
    error_estimate: float
    steps: int

    def __float__(self) -> float:
        """Return the value.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return self.value


@dataclass(frozen=True, slots=True)
class SemiMarkovTransient:
    """State law ``pi(t) = alpha phi(t)`` of a semi-Markov process.

    Args:
        states: State labels in process order.
        time: Time ``t``.
        probabilities: ``pi(t)`` in state order (clipped to ``[0, 1]``).
        error_estimate: Estimated L1 error ``sum_j |pi_2n,j - pi_n,j| / 3``.
        steps: Steps of the coarser grid.

    References:
        [EQ-PROC-SMP-002], [EQ-PROC-SMP-003], [REF-LIMNIOS-OPRISAN-2001].

    """

    states: tuple[State, ...]
    time: float
    probabilities: Vector
    error_estimate: float
    steps: int

    def probability(self, state: State) -> float:
        """Return ``pi_state(t)``.

        Args:
            state: State label.

        Returns:
            Probability of occupying ``state`` at ``time``.

        Raises:
            ValueError: If the label is unknown.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3]. No equation is evaluated.

        """
        if state not in self.states:
            raise ValueError(f"unknown state {state!r}")
        return self.probabilities[self.states.index(state)]

    def total(self, states: Iterable[State]) -> float:
        """Return ``sum_(j in states) pi_j(t)``.

        Args:
            states: State labels (duplicates count once).

        Returns:
            Probability of the state set.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3]. No equation is evaluated.

        """
        chosen = set(states)
        for state in chosen:
            if state not in self.states:
                raise ValueError(f"unknown state {state!r}")
        return math.fsum(
            p
            for label, p in zip(self.states, self.probabilities, strict=True)
            if label in chosen
        )


@dataclass(frozen=True, slots=True)
class SemiMarkovTransitionMatrix:
    """Interval transition probabilities ``phi_ij(t)`` [EQ-PROC-SMP-002].

    Args:
        states: State labels (row and column order).
        time: Time ``t``.
        matrix: ``phi(t)``; row ``i`` is the law at ``t`` from ``Z_0 = i``.
        error_estimate: Largest entrywise Richardson error estimate.
        steps: Steps of the coarser grid.

    References:
        [REF-LIMNIOS-OPRISAN-2001, Chapter 3], [REF-CINLAR-1969].

    """

    states: tuple[State, ...]
    time: float
    matrix: Matrix
    error_estimate: float
    steps: int


@dataclass(frozen=True, slots=True)
class RenewalCurve:
    """Expected reward ``E_alpha[r(Z_t)]`` on a uniform time grid.

    Args:
        times: Grid ``t_m = m h``, ``m = 0..steps``.
        values: Richardson-extrapolated values at the grid times.
        error_estimates: Richardson error estimates at the grid times.
        steps: Steps of the coarser grid.

    References:
        [EQ-PROC-SMP-002], [EQ-PROC-SMP-003], [REF-CORRADI-JANSSEN-MANCA-2004].

    """

    times: tuple[float, ...]
    values: tuple[float, ...]
    error_estimates: tuple[float, ...]
    steps: int

    @property
    def max_error_estimate(self) -> float:
        """Return the largest error estimate on the grid.

        References:
            [EQ-PROC-SMP-003]. No further equation is evaluated.

        """
        return max(self.error_estimates, default=0.0)


@dataclass(frozen=True, slots=True)
class SemiMarkovAbsorption:
    """Exact absorption analysis of a semi-Markov process [EQ-PROC-SMP-004].

    Args:
        transient_states: Labels of the transient states.
        absorbing_states: Labels of the absorbing states.
        fundamental: ``N = (I - P_TT)^-1``; ``N_ij`` is the expected number of
            visits to transient ``j`` from transient ``i``.
        absorption_probabilities: ``B = N P_TA``.
        mean_times: ``m = N tau``; mean time to absorption per transient state.
        initial_absorption: Absorption probabilities from the initial law.
        initial_mean_time: Mean time to absorption from the initial law.

    References:
        [REF-LIMNIOS-OPRISAN-2001, Chapter 5], [REF-KEMENY-SNELL-1976, Chapter III].

    """

    transient_states: tuple[State, ...]
    absorbing_states: tuple[State, ...]
    fundamental: Matrix
    absorption_probabilities: Matrix
    mean_times: Vector
    initial_absorption: Vector
    initial_mean_time: float

    def mean_time(self, start: State) -> float:
        """Return the mean time to absorption from ``start``.

        Args:
            start: Transient or absorbing state label.

        Returns:
            Mean absorption time; zero for an absorbing state.

        Raises:
            ValueError: If ``start`` is unknown.

        Equation:
            [EQ-PROC-SMP-004].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        if start in self.absorbing_states:
            return 0.0
        if start not in self.transient_states:
            raise ValueError("start must be a transient or absorbing state")
        return self.mean_times[self.transient_states.index(start)]


@dataclass(frozen=True, slots=True)
class SemiMarkovMonteCarlo:
    """Monte Carlo estimate of ``P(Z_t in S)`` at several times.

    Args:
        times: Evaluation times.
        hits: Number of paths in the state set at each time.
        samples: Number of simulated paths.
        estimates: ``hits / samples``.
        standard_errors: ``sqrt(p (1 - p) / samples)`` per time.

    References:
        [EQ-PROC-SMP-006], [REF-DEVROYE-1986, Chapter II].

    """

    times: tuple[float, ...]
    hits: tuple[int, ...]
    samples: int
    estimates: tuple[float, ...]
    standard_errors: tuple[float, ...]


def _require_time(value: object, *, name: str = "t") -> float:
    """Validate a finite non-negative time.

    Raises:
        TypeError: If the value is not real.
        ValueError: If it is negative or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _require_steps(value: object) -> int:
    """Validate the number of coarse grid steps.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If it lies outside ``[2, 20000]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("steps must be an integer")
    if not 2 <= value <= _MAX_STEPS:
        raise ValueError(f"steps must lie in [2, {_MAX_STEPS}]")
    return value


def _law_mean(law: object, *, name: str) -> float:
    """Validate a sojourn law and return its finite positive mean.

    Raises:
        TypeError: If the law has no ``cdf`` method or no ``mean``.
        ValueError: If the mean is not finite and positive or the law has an
            atom at zero.

    References:
        [EQ-PROC-SMP-001], [REF-LIMNIOS-OPRISAN-2001, Chapter 2].

    """
    if not callable(getattr(law, "cdf", None)):
        raise TypeError(f"sojourn law {name} must provide cdf(x)")
    try:
        mean = float(getattr(law, "mean"))  # noqa: B009
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"sojourn law {name} must provide a finite mean") from error
    if not (math.isfinite(mean) and mean > 0.0):
        raise ValueError(f"sojourn law {name} must have a finite positive mean")
    at_zero = float(law.cdf(0.0))  # type: ignore[attr-defined]
    if at_zero != 0.0:
        raise ValueError(f"sojourn law {name} must have no mass at or below zero")
    return mean


def _tail_pair(law: object, x: float) -> tuple[float, float]:
    """Return ``(F(x), 1 - F(x))`` without cancellation in either part.

    ``sf`` is used for the upper tail when the law provides it.

    References:
        [EQ-PROC-SMP-001], [REF-PROBSTAT-VALIDATION].

    """
    lower = min(1.0, max(0.0, float(law.cdf(x))))  # type: ignore[attr-defined]
    survival = getattr(law, "sf", None)
    upper = (
        min(1.0, max(0.0, float(survival(x)))) if callable(survival) else 1.0 - lower
    )
    return lower, upper


class SemiMarkovProcess:
    """Homogeneous semi-Markov process on finitely many labelled states.

    Args:
        states: Distinct hashable state labels (not a string).
        transitions: Mapping ``{(from, to): (p, law)}`` of embedded jump
            probabilities ``p in (0, 1]`` and sojourn laws ``F_from,to`` with
            ``cdf``, a finite positive ``mean`` and ``cdf(0) = 0``. The
            probabilities out of each state with transitions must sum to one
            within ``1e-10``; a state without transitions is absorbing.
            Self-transitions (renewals of a state) are allowed.

    Raises:
        TypeError: If a label is unhashable or a law lacks ``cdf``/``mean``.
        ValueError: If the specification is invalid.

    Equation:
        [EQ-PROC-SMP-001].

    References:
        [REF-LIMNIOS-OPRISAN-2001, Chapters 2-3], [REF-CINLAR-1969].

    """

    def __init__(
        self,
        states: Sequence[State],
        transitions: Mapping[tuple[State, State], tuple[float, object]],
    ) -> None:
        """Validate the kernel and cache the embedded chain.

        References:
            [EQ-PROC-SMP-001], [REF-LIMNIOS-OPRISAN-2001, Chapter 2].

        """
        if isinstance(states, str):
            raise TypeError("states must be a sequence of labels, not a string")
        labels = tuple(states)
        if not labels:
            raise ValueError("states must be non-empty")
        try:
            positions = {label: i for i, label in enumerate(labels)}
        except TypeError as error:
            raise TypeError("state labels must be hashable") from error
        if len(positions) != len(labels):
            raise ValueError("state labels must be distinct")
        if not isinstance(transitions, Mapping):
            raise TypeError("transitions must be a mapping {(from, to): (p, law)}")
        arcs: list[list[tuple[int, float, object, float]]] = [[] for _ in labels]
        for key, entry in transitions.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("transition keys must be (from_state, to_state) pairs")
            source, target = key
            if source not in positions or target not in positions:
                raise ValueError(f"transition {key!r} names an unknown state")
            if not isinstance(entry, tuple | list) or len(entry) != 2:
                raise ValueError(f"transition {key!r} must map to (probability, law)")
            probability = require_finite_real(entry[0], name=f"p{key!r}")
            if not 0.0 < probability <= 1.0:
                raise ValueError(f"probability of {key!r} must lie in (0, 1]")
            mean = _law_mean(entry[1], name=repr(key))
            arcs[positions[source]].append(
                (positions[target], probability, entry[1], mean)
            )
        for i, row in enumerate(arcs):
            if row and abs(math.fsum(p for _, p, _, _ in row) - 1.0) > (
                _ROW_SUM_TOLERANCE
            ):
                raise ValueError(
                    f"jump probabilities out of {labels[i]!r} must sum to one"
                )
        self._states = labels
        self._positions: Mapping[State, int] = MappingProxyType(positions)
        self._arcs = tuple(tuple(row) for row in arcs)
        self._transitions = MappingProxyType(
            {
                (labels[i], labels[j]): (p, law)
                for i, row in enumerate(arcs)
                for j, p, law, _ in row
            }
        )
        self._mean_sojourn = tuple(
            math.fsum(p * mean for _, p, _, mean in row) for row in arcs
        )
        self._proxy = ContinuousTimeMarkovChain(
            labels,
            {
                (labels[i], labels[j]): p
                for i, row in enumerate(arcs)
                for j, p, _, _ in row
                if j != i
            },
        )

    # ------------------------------------------------------------------ build
    @classmethod
    def from_ctmc(cls, chain: ContinuousTimeMarkovChain) -> SemiMarkovProcess:
        """Return the semi-Markov form of a CTMC (exponential sojourns).

        ``p_ij = q_ij / q_i`` and ``F_ij = Exp(q_i)`` with ``q_i = -q_ii``
        [EQ-PROC-SMP-001]; states with ``q_i = 0`` are absorbing.

        Args:
            chain: Continuous-time Markov chain.

        Returns:
            Equivalent semi-Markov process.

        Raises:
            TypeError: If ``chain`` is not a CTMC.

        Equation:
            [EQ-PROC-SMP-001], [EQ-PROC-CTMC-007].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3], [REF-NORRIS-1997, Section 2.6].

        """
        if not isinstance(chain, ContinuousTimeMarkovChain):
            raise TypeError("chain must be a ContinuousTimeMarkovChain")
        transitions: dict[tuple[State, State], tuple[float, object]] = {}
        for source in chain.states:
            exit_rate = chain.exit_rate(source)
            if exit_rate <= 0.0:
                continue
            law = Exponential(exit_rate)
            for target in chain.states:
                rate = chain.rate(source, target) if target != source else 0.0
                if rate > 0.0:
                    transitions[(source, target)] = (rate / exit_rate, law)
        return cls(chain.states, transitions)

    # -------------------------------------------------------------- structure
    @property
    def states(self) -> tuple[State, ...]:
        """Return the state labels.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3]. No equation is evaluated.

        """
        return self._states

    @property
    def size(self) -> int:
        """Return the number of states.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3]. No equation is evaluated.

        """
        return len(self._states)

    @property
    def transitions(self) -> Mapping[tuple[State, State], tuple[float, object]]:
        """Return the read-only kernel ``{(i, j): (p_ij, F_ij)}``.

        References:
            [EQ-PROC-SMP-001], [REF-LIMNIOS-OPRISAN-2001, Chapter 2].

        """
        return self._transitions

    @property
    def embedded_matrix(self) -> Matrix:
        """Return the embedded jump matrix ``P`` (absorbing rows ``P_ii = 1``).

        Equation:
            [EQ-PROC-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 2].

        """
        rows = []
        for i, arcs in enumerate(self._arcs):
            row = [0.0] * self.size
            if not arcs:
                row[i] = 1.0
            for j, p, _, _ in arcs:
                row[j] += p
            rows.append(tuple(row))
        return tuple(rows)

    @property
    def mean_sojourn_times(self) -> Vector:
        """Return ``tau_i = sum_j p_ij E[T_ij]`` (zero for absorbing states).

        Equation:
            [EQ-PROC-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3].

        """
        return self._mean_sojourn

    @property
    def absorbing_states(self) -> tuple[State, ...]:
        """Return the states without outgoing transitions.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3]. No equation is evaluated.

        """
        return tuple(
            label
            for label, arcs in zip(self._states, self._arcs, strict=True)
            if not arcs
        )

    def index(self, state: State) -> int:
        """Return the position of a state label.

        Raises:
            ValueError: If the label is unknown.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        try:
            return self._positions[state]
        except (KeyError, TypeError) as error:
            raise ValueError(f"unknown state {state!r}") from error

    def embedded_chain(self) -> ContinuousTimeMarkovChain:
        """Return a CTMC with rates ``p_ij`` (``i != j``) carrying the jump structure.

        Its generator restricted to the transient states is ``-(I - P_TT)``,
        so its fundamental matrix is the expected-visit matrix of the jump
        chain and its stationary law is the jump chain's ``nu``.

        Equation:
            [EQ-PROC-SMP-004], [EQ-PROC-SMP-005].

        References:
            [REF-KEMENY-SNELL-1976, Chapter III], [REF-LIMNIOS-OPRISAN-2001, Chapter 2].

        """
        return self._proxy

    def distribution(self, initial: object) -> Vector:
        """Normalize an initial law to a probability vector in state order.

        Args:
            initial: A state label, a mapping ``{label: probability}`` or a
                probability sequence in state order.

        Returns:
            Probability vector.

        Raises:
            ValueError: If the law is unknown, negative or not normalized.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        return self._proxy.distribution(initial)

    def kernel(self, t: float) -> Matrix:
        """Return the semi-Markov kernel ``Q(t)`` with ``Q_ij(t) = p_ij F_ij(t)``.

        Args:
            t: Non-negative time.

        Returns:
            Kernel matrix in state order.

        Equation:
            [EQ-PROC-SMP-001].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 2].

        """
        time = _require_time(t)
        rows = []
        for arcs in self._arcs:
            row = [0.0] * self.size
            for j, p, law, _ in arcs:
                row[j] += p * _tail_pair(law, time)[0]
            rows.append(tuple(row))
        return tuple(rows)

    def make_absorbing(self, states: Iterable[State]) -> SemiMarkovProcess:
        """Return a copy in which ``states`` have no outgoing transitions.

        Args:
            states: Labels to make absorbing.

        Returns:
            Modified process.

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5]. No equation is evaluated.

        """
        chosen = {self.index(state) for state in states}
        return SemiMarkovProcess(
            self._states,
            {
                key: value
                for key, value in self._transitions.items()
                if self._positions[key[0]] not in chosen
            },
        )

    # ------------------------------------------------- Markov renewal equation
    def _grid_solution(
        self, rewards: Sequence[Sequence[float]], horizon: float, steps: int
    ) -> list[list[list[float]]]:
        """Solve the discretised Markov renewal equation on ``steps`` steps.

        With ``h = horizon / steps``, ``A_l = (Q(t_l) - Q(t_(l-1))) / 2`` and
        ``B_l = A_l + A_(l+1)``, the trapezoidal rule gives
        ``(I - A_1) u_n = (1 - H(t_n)) r + A_n u_0 + sum_(l=1)^(n-1) B_l u_(n-l)``.

        Args:
            rewards: ``S x C`` reward matrix (one column per reward vector).
            horizon: Positive final time.
            steps: Number of steps.

        Returns:
            ``u_m`` (``S x C``) for ``m = 0..steps``.

        Equation:
            [EQ-PROC-SMP-002], [EQ-PROC-SMP-003].

        References:
            [REF-CORRADI-JANSSEN-MANCA-2004], [REF-LINZ-1985, Chapter 7].

        """
        size = self.size
        width = len(rewards[0])
        h = horizon / steps
        grid = [m * h for m in range(steps + 1)]
        grid[-1] = horizon
        survival = [[1.0] * (steps + 1) for _ in range(size)]
        terms: list[list[tuple[int, list[float], list[float]]]] = [
            [] for _ in range(size)
        ]
        first = [[0.0] * size for _ in range(size)]
        for i, arcs in enumerate(self._arcs):
            if not arcs:
                continue
            totals = [0.0] * (steps + 1)
            for j, p, law, _ in arcs:
                pairs = [_tail_pair(law, x) for x in grid]
                halves = [0.0] * (steps + 2)
                for m in range(1, steps + 1):
                    low_a, up_a = pairs[m - 1]
                    low_b, up_b = pairs[m]
                    increment = low_b - low_a if low_b <= 0.5 else up_a - up_b
                    halves[m] = 0.5 * p * max(0.0, increment)
                for m in range(steps + 1):
                    totals[m] += p * pairs[m][1]
                blended = [halves[m] + halves[m + 1] for m in range(steps + 1)]
                terms[i].append((j, halves, blended))
                first[i][j] += halves[1]
            survival[i] = [min(1.0, value) for value in totals]
        system = [
            [(1.0 if a == b else 0.0) - first[a][b] for b in range(size)]
            for a in range(size)
        ]
        inverse = _invert(system)
        history = [[[float(rewards[k][c])] for c in range(width)] for k in range(size)]
        solution = [[[float(rewards[k][c]) for c in range(width)] for k in range(size)]]
        for n in range(1, steps + 1):
            rhs = []
            for i in range(size):
                row = []
                for c in range(width):
                    parts = [survival[i][n] * rewards[i][c]]
                    for k, halves, blended in terms[i]:
                        past = history[k][c]
                        parts.append(halves[n] * past[0])
                        if n > 1:
                            parts.append(
                                math.fsum(map(mul, blended[1:n], past[n - 1 : 0 : -1]))
                            )
                    row.append(math.fsum(parts))
                rhs.append(row)
            current = [
                [
                    math.fsum(inverse[i][k] * rhs[k][c] for k in range(size))
                    for c in range(width)
                ]
                for i in range(size)
            ]
            for k in range(size):
                for c in range(width):
                    history[k][c].append(current[k][c])
            solution.append(current)
        return solution

    def _extrapolated(
        self, rewards: Sequence[Sequence[float]], horizon: float, steps: int
    ) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        """Return Richardson-extrapolated values and error estimates on the grid.

        Returns:
            ``(values, errors)``, each ``u_m`` for ``m = 0..steps``.

        Equation:
            [EQ-PROC-SMP-003].

        References:
            [REF-LINZ-1985, Chapter 7], [REF-CORRADI-JANSSEN-MANCA-2004].

        """
        coarse = self._grid_solution(rewards, horizon, steps)
        fine = self._grid_solution(rewards, horizon, 2 * steps)
        values = []
        errors = []
        for m, block in enumerate(coarse):
            detail = fine[2 * m]
            values.append(
                [
                    [(4.0 * f - c) / 3.0 for f, c in zip(frow, crow, strict=True)]
                    for frow, crow in zip(detail, block, strict=True)
                ]
            )
            errors.append(
                [
                    [abs(f - c) / 3.0 for f, c in zip(frow, crow, strict=True)]
                    for frow, crow in zip(detail, block, strict=True)
                ]
            )
        return values, errors

    def transition_matrix(
        self, t: float, *, steps: int = _DEFAULT_STEPS
    ) -> SemiMarkovTransitionMatrix:
        """Return the interval transition probabilities ``phi(t)``.

        Args:
            t: Non-negative time.
            steps: Steps of the coarser grid (the finer grid has twice as many).

        Returns:
            Transition matrix with its Richardson error estimate.

        Raises:
            ValueError: If an argument is invalid.

        Equation:
            [EQ-PROC-SMP-002], [EQ-PROC-SMP-003].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3], [REF-CORRADI-JANSSEN-MANCA-2004].

        """
        time = _require_time(t)
        count = _require_steps(steps)
        size = self.size
        identity = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        if time == 0.0:
            return SemiMarkovTransitionMatrix(
                self._states, 0.0, tuple(tuple(r) for r in identity), 0.0, count
            )
        values, errors = self._extrapolated(identity, time, count)
        matrix = tuple(tuple(min(1.0, max(0.0, v)) for v in row) for row in values[-1])
        error = max(e for row in errors[-1] for e in row)
        return SemiMarkovTransitionMatrix(self._states, time, matrix, error, count)

    def transient(
        self, initial: object, t: float, *, steps: int = _DEFAULT_STEPS
    ) -> SemiMarkovTransient:
        """Return the state law ``pi(t) = alpha phi(t)``.

        Args:
            initial: Initial law (see :meth:`distribution`).
            t: Non-negative time.
            steps: Steps of the coarser grid.

        Returns:
            State law with an L1 error estimate.

        Equation:
            [EQ-PROC-SMP-002], [EQ-PROC-SMP-003].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3].

        """
        start = self.distribution(initial)
        time = _require_time(t)
        count = _require_steps(steps)
        size = self.size
        if time == 0.0:
            return SemiMarkovTransient(self._states, 0.0, start, 0.0, count)
        identity = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        values, errors = self._extrapolated(identity, time, count)
        law = tuple(
            min(
                1.0,
                max(0.0, math.fsum(start[i] * values[-1][i][j] for i in range(size))),
            )
            for j in range(size)
        )
        error = math.fsum(
            math.fsum(start[i] * errors[-1][i][j] for i in range(size))
            for j in range(size)
        )
        return SemiMarkovTransient(self._states, time, law, error, count)

    def expected_reward_curve(
        self,
        initial: object,
        rewards: Mapping[State, float] | Sequence[float],
        horizon: float,
        *,
        steps: int = _DEFAULT_STEPS,
    ) -> RenewalCurve:
        """Return ``E_alpha[r(Z_t)]`` on the uniform grid of ``[0, horizon]``.

        Args:
            initial: Initial law.
            rewards: Per-state reward (mapping, missing labels zero, or a
                sequence in state order).
            horizon: Positive final time.
            steps: Steps of the coarser grid.

        Returns:
            Curve with Richardson error estimates.

        Raises:
            ValueError: If an argument is invalid.

        Equation:
            [EQ-PROC-SMP-002], [EQ-PROC-SMP-003].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3], [REF-CORRADI-JANSSEN-MANCA-2004].

        """
        start = self.distribution(initial)
        values_r = self._proxy.state_values(rewards)
        end = _require_time(horizon, name="horizon")
        if end == 0.0:
            raise ValueError("horizon must be positive")
        count = _require_steps(steps)
        column = [[value] for value in values_r]
        values, errors = self._extrapolated(column, end, count)
        h = end / count
        times = tuple(min(end, m * h) for m in range(count + 1))
        curve = tuple(
            math.fsum(start[i] * block[i][0] for i in range(self.size))
            for block in values
        )
        bounds = tuple(
            math.fsum(start[i] * block[i][0] for i in range(self.size))
            for block in errors
        )
        return RenewalCurve(times, curve, bounds, count)

    def probability_curve(
        self,
        initial: object,
        states: Iterable[State],
        horizon: float,
        *,
        steps: int = _DEFAULT_STEPS,
    ) -> RenewalCurve:
        """Return ``P_alpha(Z_t in S)`` on the uniform grid of ``[0, horizon]``.

        Args:
            initial: Initial law.
            states: State set ``S``.
            horizon: Positive final time.
            steps: Steps of the coarser grid.

        Returns:
            Curve clipped to ``[0, 1]`` with error estimates.

        Equation:
            [EQ-PROC-SMP-002].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        chosen = {self.index(state) for state in states}
        rewards = [1.0 if i in chosen else 0.0 for i in range(self.size)]
        curve = self.expected_reward_curve(initial, rewards, horizon, steps=steps)
        return RenewalCurve(
            curve.times,
            tuple(min(1.0, max(0.0, v)) for v in curve.values),
            curve.error_estimates,
            curve.steps,
        )

    def probability(
        self,
        initial: object,
        states: Iterable[State],
        t: float,
        *,
        steps: int = _DEFAULT_STEPS,
    ) -> SemiMarkovEstimate:
        """Return ``P_alpha(Z_t in S)`` at one time.

        Args:
            initial: Initial law.
            states: State set ``S``.
            t: Non-negative time.
            steps: Steps of the coarser grid.

        Returns:
            Value with its error estimate.

        Equation:
            [EQ-PROC-SMP-002], [EQ-PROC-SMP-003].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        chosen = {self.index(state) for state in states}
        time = _require_time(t)
        count = _require_steps(steps)
        if time == 0.0:
            start = self.distribution(initial)
            return SemiMarkovEstimate(math.fsum(start[i] for i in chosen), 0.0, count)
        labels = [self._states[i] for i in sorted(chosen)]
        curve = self.probability_curve(initial, labels, time, steps=count)
        return SemiMarkovEstimate(curve.values[-1], curve.error_estimates[-1], count)

    def first_passage_curve(
        self,
        targets: Iterable[State],
        horizon: float,
        initial: object,
        *,
        steps: int = _DEFAULT_STEPS,
    ) -> RenewalCurve:
        """Return the first-passage distribution ``G(t) = P_alpha(T_S <= t)``.

        ``T_S`` is the first time the process is in ``S``; the targets are
        made absorbing and the occupation probability of ``S`` is solved by
        the Markov renewal equation [EQ-PROC-SMP-002].

        Args:
            targets: Target set ``S`` (non-empty).
            horizon: Positive final time.
            initial: Initial law.
            steps: Steps of the coarser grid.

        Returns:
            First-passage CDF on the grid.

        Raises:
            ValueError: If ``targets`` is empty or an argument is invalid.

        Equation:
            [EQ-PROC-SMP-002].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5], [REF-CINLAR-1969].

        """
        chosen = tuple(dict.fromkeys(targets))
        if not chosen:
            raise ValueError("targets must be non-empty")
        absorbed = self.make_absorbing(chosen)
        return absorbed.probability_curve(initial, chosen, horizon, steps=steps)

    # ------------------------------------------------------------ exact parts
    def absorption(self, initial: object) -> SemiMarkovAbsorption:
        """Return exact absorption probabilities and mean times to absorption.

        ``N = (I - P_TT)^-1``, ``B = N P_TA`` and ``m = N tau`` with the mean
        sojourn times ``tau`` [EQ-PROC-SMP-004]. Only the means of the
        sojourn laws enter.

        Args:
            initial: Initial law.

        Returns:
            Absorption analysis.

        Raises:
            ValueError: If there is no absorbing or no transient state, or a
                closed class of the jump chain is not a single absorbing
                state.

        Equation:
            [EQ-PROC-SMP-004].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5], [REF-KEMENY-SNELL-1976, Chapter III].

        """
        start = self.distribution(initial)
        visits = self._proxy.absorption(start)
        positions = [self.index(state) for state in visits.transient_states]
        tau = [self._mean_sojourn[i] for i in positions]
        mean_times = tuple(
            math.fsum(n * s for n, s in zip(row, tau, strict=True))
            for row in visits.fundamental
        )
        initial_mean = math.fsum(
            start[i] * m for i, m in zip(positions, mean_times, strict=True)
        )
        return SemiMarkovAbsorption(
            visits.transient_states,
            visits.absorbing_states,
            visits.fundamental,
            visits.absorption_probabilities,
            mean_times,
            visits.initial_absorption,
            initial_mean,
        )

    def mean_first_passage_time(
        self, targets: Iterable[State], initial: object
    ) -> float:
        """Return ``E_alpha[T_S]`` exactly, or ``inf`` if ``S`` may be missed.

        Args:
            targets: Non-empty target set ``S``.
            initial: Initial law.

        Returns:
            Mean first-passage time.

        Equation:
            [EQ-PROC-SMP-004].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 5].

        """
        chosen = tuple(dict.fromkeys(targets))
        if not chosen:
            raise ValueError("targets must be non-empty")
        start = self.distribution(initial)
        absorbed = self.make_absorbing(chosen)
        proxy = absorbed.embedded_chain()
        target_set = set(chosen)
        trapped = [
            state
            for members in proxy.closed_classes()
            if not (len(members) == 1 and members[0] in target_set)
            for state in members
        ]
        if trapped:
            if any(
                probability > 0.0 and proxy.can_reach(state, trapped)
                for state, probability in zip(proxy.states, start, strict=True)
            ):
                return math.inf
            absorbed = absorbed.make_absorbing(trapped)
        if all(
            start[i] == 0.0
            for i, state in enumerate(self._states)
            if state not in target_set
        ):
            return 0.0
        return absorbed.absorption(start).initial_mean_time

    def steady_state(self) -> Vector:
        """Return the limiting law ``pi_j = nu_j tau_j / sum_k nu_k tau_k``.

        ``nu`` is the stationary law of the embedded jump chain on its unique
        closed class (computed by GTH reduction) [EQ-PROC-SMP-005]; sojourn
        laws must be non-lattice for the limit to exist, which holds for
        laws with a density.

        Returns:
            Limiting probability vector in state order.

        Raises:
            ValueError: If the jump chain has more than one closed class.

        Equation:
            [EQ-PROC-SMP-005].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3], [REF-CINLAR-1969].

        """
        nu = self._proxy.steady_state()
        closed = self._proxy.closed_classes()[0]
        if len(closed) == 1 and not self._arcs[self.index(closed[0])]:
            return nu
        weights = [n * tau for n, tau in zip(nu, self._mean_sojourn, strict=True)]
        total = math.fsum(weights)
        return tuple(w / total for w in weights)

    # ------------------------------------------------------------- simulation
    def _jump(
        self, state: int, uniform_target: float, uniform_time: float
    ) -> tuple[int, float]:
        """Return the next state and the sojourn drawn by inversion.

        Raises:
            TypeError: If a sojourn law has no ``quantile``.

        References:
            [EQ-PROC-SMP-006], [REF-DEVROYE-1986, Chapter II].

        """
        arcs = self._arcs[state]
        running = 0.0
        chosen = arcs[-1]
        for arc in arcs:
            running += arc[1]
            if uniform_target < running:
                chosen = arc
                break
        quantile = getattr(chosen[2], "quantile", None)
        if not callable(quantile):
            raise TypeError("simulation needs sojourn laws with quantile(p)")
        return chosen[0], float(quantile(uniform_time))

    def simulate(
        self,
        initial: object,
        horizon: float,
        *,
        seed: int | RandomStream | None = None,
    ) -> CTMCPath:
        """Simulate one path on ``[0, horizon]`` (cross-check helper only).

        The next state is drawn from the jump chain and the sojourn from
        ``F_ij`` by inversion [EQ-PROC-SMP-006]. The returned record is the
        piecewise-constant path type shared with the CTMC module.

        Args:
            initial: Initial law.
            horizon: Non-negative final time.
            seed: Integer seed or stream; ``None`` is not reproducible.

        Returns:
            Simulated path.

        Equation:
            [EQ-PROC-SMP-006].

        References:
            [REF-LIMNIOS-OPRISAN-2001, Chapter 3], [REF-DEVROYE-1986, Chapter II].

        """
        start = self.distribution(initial)
        end = _require_time(horizon, name="horizon")
        generator = resolve_stream(seed).child("semi-markov").generator()
        state = _draw(start, generator.random())
        times = [0.0]
        visited = [self._states[state]]
        now = 0.0
        while self._arcs[state]:
            target, sojourn = self._jump(
                state, generator.random(), _open_uniform(generator.random())
            )
            now += sojourn
            if now >= end:
                break
            state = target
            times.append(now)
            visited.append(self._states[state])
        return CTMCPath(tuple(visited), tuple(times), end)

    def monte_carlo_probabilities(
        self,
        initial: object,
        states: Iterable[State],
        times: Sequence[float],
        *,
        samples: int = 10_000,
        seed: int | RandomStream | None = None,
    ) -> SemiMarkovMonteCarlo:
        """Estimate ``P_alpha(Z_t in S)`` at several times by simulation.

        Args:
            initial: Initial law.
            states: State set ``S``.
            times: Non-empty evaluation times.
            samples: Number of paths (at least 2).
            seed: Integer seed or stream.

        Returns:
            Estimates with binomial standard errors.

        Raises:
            ValueError: If an argument is invalid.

        Equation:
            [EQ-PROC-SMP-006].

        References:
            [REF-DEVROYE-1986, Chapter II], [REF-LIMNIOS-OPRISAN-2001, Chapter 3].

        """
        start = self.distribution(initial)
        chosen = {self.index(state) for state in states}
        moments = tuple(_require_time(t, name="times") for t in times)
        if not moments:
            raise ValueError("times must be non-empty")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError("samples must be an integer of at least 2")
        order = sorted(range(len(moments)), key=moments.__getitem__)
        horizon = max(moments)
        generator = resolve_stream(seed).child("semi-markov-mc").generator()
        hits = [0] * len(moments)
        for _ in range(samples):
            state = _draw(start, generator.random())
            jumps = [0.0]
            path = [state]
            now = 0.0
            while self._arcs[state] and now <= horizon:
                state, sojourn = self._jump(
                    state, generator.random(), _open_uniform(generator.random())
                )
                now += sojourn
                jumps.append(now)
                path.append(state)
            for index in order:
                position = bisect.bisect_right(jumps, moments[index]) - 1
                if path[position] in chosen:
                    hits[index] += 1
        estimates = tuple(h / samples for h in hits)
        errors = tuple(math.sqrt(p * (1.0 - p) / samples) for p in estimates)
        return SemiMarkovMonteCarlo(moments, tuple(hits), samples, estimates, errors)


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    """Invert a small dense matrix by Gauss-Jordan elimination with pivoting.

    Raises:
        ValueError: If the matrix is singular.

    References:
        [REF-LINZ-1985, Chapter 7]. No named equation is needed.

    """
    size = len(matrix)
    a = [
        list(row) + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(a[r][col]))
        if a[pivot][col] == 0.0:
            raise ValueError("the first-step system of the renewal scheme is singular")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(size):
            if row != col and a[row][col] != 0.0:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col], strict=True)]
    return [row[size:] for row in a]


def _open_uniform(value: float) -> float:
    """Map a draw from ``[0, 1)`` into ``(0, 1)`` for inversion.

    References:
        [REF-DEVROYE-1986, Chapter II]. No named equation is needed.

    """
    return value if value > 0.0 else 2.0**-53


def _draw(probabilities: Sequence[float], uniform: float) -> int:
    """Return an index drawn by inversion of a probability vector.

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
