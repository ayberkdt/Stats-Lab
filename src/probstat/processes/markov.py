"""Finite discrete-time Markov chains.

A homogeneous chain on states ``0..n-1`` is described by a row-stochastic
transition matrix ``P`` with ``P_ij = Pr(X_(t+1) = j | X_t = i)``. This module
validates ``P``, propagates distributions, classifies states (communicating
classes, closed classes, periods), solves for the stationary distribution,
analyzes absorption through the fundamental matrix, computes mean first
passage times, and simulates seeded paths.

Every linear system is solved directly by Gaussian elimination with partial
pivoting from :mod:`probstat.linalg`; no power iteration hides slow mixing.
Classification uses exact zero/non-zero structure of ``P``: an entry is an
edge exactly when it is strictly positive.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Matrix, Vector, as_matrix, as_vector, lu_solve

_ROW_SUM_TOLERANCE = 1e-10


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two square matrices with compensated summation.

    Args:
        left: Left factor.
        right: Right factor of matching size.

    Returns:
        Product matrix.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No named equation is needed.

    """
    size = len(left)
    columns = tuple(tuple(right[k][j] for k in range(size)) for j in range(size))
    return tuple(
        tuple(
            math.fsum(a * b for a, b in zip(row, column, strict=True))
            for column in columns
        )
        for row in left
    )


@dataclass(frozen=True, slots=True)
class AbsorptionAnalysis:
    """Fundamental-matrix summary of an absorbing Markov chain.

    Args:
        transient_states: Indices of transient states (rows of ``N``).
        absorbing_states: Indices of absorbing states (columns of ``B``).
        fundamental: ``N = (I - Q)^-1``; ``N_ij`` is the expected number of
            visits to transient ``j`` starting from transient ``i``.
        absorption_probabilities: ``B = N R``; ``B_ik`` is the probability of
            absorption in absorbing state ``k`` from transient state ``i``.
        expected_steps: ``t = N 1``; expected number of steps before
            absorption from each transient state.

    References:
        [REF-KEMENY-SNELL-1976, Chapter III], [REF-NORRIS-1997, Section 1.3].

    """

    transient_states: tuple[int, ...]
    absorbing_states: tuple[int, ...]
    fundamental: Matrix
    absorption_probabilities: Matrix
    expected_steps: Vector

    def probability(self, start: int, target: int) -> float:
        """Return the probability of absorption in ``target`` from ``start``.

        Args:
            start: Any state index; an absorbing start is absorbed at once.
            target: Absorbing state index.

        Returns:
            Absorption probability.

        Raises:
            ValueError: If ``target`` is not absorbing or ``start`` is unknown.

        Equation:
            [EQ-PROC-MC-005].

        References:
            [REF-KEMENY-SNELL-1976, Chapter III].

        """
        if target not in self.absorbing_states:
            raise ValueError("target must be an absorbing state")
        if start in self.absorbing_states:
            return 1.0 if start == target else 0.0
        if start not in self.transient_states:
            raise ValueError("start must be a transient or absorbing state")
        row = self.transient_states.index(start)
        return self.absorption_probabilities[row][self.absorbing_states.index(target)]

    def steps(self, start: int) -> float:
        """Return the expected number of steps to absorption from ``start``.

        Args:
            start: Transient or absorbing state index.

        Returns:
            Expected absorption time; zero for an absorbing state.

        Raises:
            ValueError: If ``start`` is neither transient nor absorbing.

        Equation:
            [EQ-PROC-MC-005].

        References:
            [REF-KEMENY-SNELL-1976, Chapter III].

        """
        if start in self.absorbing_states:
            return 0.0
        if start not in self.transient_states:
            raise ValueError("start must be a transient or absorbing state")
        return self.expected_steps[self.transient_states.index(start)]


@dataclass(frozen=True)
class MarkovChain:
    """Homogeneous discrete-time Markov chain on a finite state space.

    Args:
        transition: Row-stochastic matrix ``P``: non-negative entries, each
            row summing to one within ``1e-10``.
        labels: Optional distinct state labels; defaults to ``"0".."n-1"``.

    Raises:
        ValueError: If ``P`` is not square, has a negative entry, or a row
            does not sum to one; or if labels are invalid.

    References:
        [REF-NORRIS-1997, Section 1.1].

    """

    transition: Matrix
    labels: tuple[str, ...] = ()
    _reach: tuple[tuple[bool, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the stochastic matrix and cache reachability.

        Raises:
            ValueError: If the matrix or labels are invalid.

        Equation:
            [EQ-PROC-MC-001].

        References:
            [REF-NORRIS-1997, Sections 1.1-1.2].

        """
        matrix = as_matrix(self.transition, name="transition", square=True)
        for i, row in enumerate(matrix):
            if any(value < 0.0 for value in row):
                raise ValueError(f"transition row {i} has a negative entry")
            if abs(math.fsum(row) - 1.0) > _ROW_SUM_TOLERANCE:
                raise ValueError(f"transition row {i} does not sum to one")
        size = len(matrix)
        labels = tuple(self.labels) or tuple(str(i) for i in range(size))
        if len(labels) != size or len(set(labels)) != size:
            raise ValueError("labels must be distinct and match the state count")
        reach = [
            [i == j or matrix[i][j] > 0.0 for j in range(size)] for i in range(size)
        ]
        for k in range(size):
            for i in range(size):
                if reach[i][k]:
                    row_k = reach[k]
                    row_i = reach[i]
                    for j in range(size):
                        if row_k[j]:
                            row_i[j] = True
        object.__setattr__(self, "transition", matrix)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "_reach", tuple(tuple(row) for row in reach))

    @property
    def size(self) -> int:
        """Return the number of states.

        References:
            [REF-NORRIS-1997, Section 1.1].

        """
        return len(self.transition)

    def index(self, state: int | str) -> int:
        """Return the index of a state given by index or label.

        Args:
            state: Integer index or label.

        Returns:
            State index.

        Raises:
            ValueError: If the state is unknown.

        References:
            Project validation policy. No equation is evaluated.

        """
        if isinstance(state, str):
            if state not in self.labels:
                raise ValueError(f"unknown state label {state!r}")
            return self.labels.index(state)
        if isinstance(state, bool) or not isinstance(state, int):
            raise ValueError("state must be an index or a label")
        if not 0 <= state < self.size:
            raise ValueError("state index out of range")
        return state

    def _distribution(self, values: Iterable[object]) -> Vector:
        """Validate a probability vector over the states.

        Args:
            values: Candidate probabilities.

        Returns:
            Validated probability vector.

        Raises:
            ValueError: If the length, signs, or total are invalid.

        References:
            [REF-NORRIS-1997, Section 1.1]. No equation is evaluated.

        """
        vector = as_vector(values, name="distribution")
        if len(vector) != self.size:
            raise ValueError("distribution length must equal the state count")
        if any(value < 0.0 for value in vector):
            raise ValueError("distribution entries must be non-negative")
        if abs(math.fsum(vector) - 1.0) > _ROW_SUM_TOLERANCE:
            raise ValueError("distribution must sum to one")
        return vector

    def n_step_matrix(self, steps: int) -> Matrix:
        """Return ``P^n`` by repeated squaring.

        Args:
            steps: Non-negative number of steps ``n``.

        Returns:
            ``n``-step transition matrix; ``P^0 = I``.

        Raises:
            TypeError: If ``steps`` is not an integer.
            ValueError: If ``steps`` is negative.

        Equation:
            [EQ-PROC-MC-002].

        References:
            [REF-NORRIS-1997, Section 1.1].

        """
        remaining = require_non_negative_integer(steps, name="steps")
        size = self.size
        result: Matrix = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(size)) for i in range(size)
        )
        power = self.transition
        while remaining:
            if remaining & 1:
                result = _matmul(result, power)
            remaining >>= 1
            if remaining:
                power = _matmul(power, power)
        return result

    def distribution_after(self, initial: Sequence[float], steps: int) -> Vector:
        """Return the law of ``X_n`` given the law of ``X_0``.

        Args:
            initial: Probability vector of ``X_0``.
            steps: Non-negative number of steps.

        Returns:
            Row vector ``pi_0 P^n``.

        Raises:
            ValueError: If ``initial`` is not a probability vector.

        Equation:
            [EQ-PROC-MC-002].

        References:
            [REF-NORRIS-1997, Section 1.1].

        """
        current = self._distribution(initial)
        count = require_non_negative_integer(steps, name="steps")
        matrix = self.transition
        size = self.size
        for _ in range(count):
            current = tuple(
                math.fsum(current[i] * matrix[i][j] for i in range(size))
                for j in range(size)
            )
        return current

    def communicating_classes(self) -> tuple[tuple[int, ...], ...]:
        """Return the communicating classes ordered by smallest member.

        Returns:
            Tuple of sorted state-index tuples partitioning the state space.

        Equation:
            [EQ-PROC-MC-004].

        References:
            [REF-NORRIS-1997, Section 1.2].

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

    def closed_classes(self) -> tuple[tuple[int, ...], ...]:
        """Return the closed communicating classes (the recurrent classes).

        Returns:
            Classes that no transition leaves; in a finite chain these are
            exactly the recurrent classes.

        Equation:
            [EQ-PROC-MC-004].

        References:
            [REF-NORRIS-1997, Sections 1.2 and 1.5].

        """
        closed: list[tuple[int, ...]] = []
        for members in self.communicating_classes():
            inside = set(members)
            if all(
                self.transition[i][j] == 0.0
                for i in members
                for j in range(self.size)
                if j not in inside
            ):
                closed.append(members)
        return tuple(closed)

    def transient_states(self) -> tuple[int, ...]:
        """Return the states that belong to no closed class.

        Returns:
            Sorted transient state indices.

        References:
            [REF-NORRIS-1997, Section 1.5].

        """
        recurrent = {i for members in self.closed_classes() for i in members}
        return tuple(i for i in range(self.size) if i not in recurrent)

    def absorbing_states(self) -> tuple[int, ...]:
        """Return the states with ``P_ii = 1``.

        Returns:
            Sorted absorbing state indices.

        References:
            [REF-NORRIS-1997, Section 1.3].

        """
        return tuple(i for i in range(self.size) if self.transition[i][i] == 1.0)

    def is_irreducible(self) -> bool:
        """Return whether every state communicates with every other state.

        References:
            [REF-NORRIS-1997, Section 1.2].

        """
        return len(self.communicating_classes()) == 1

    def period(self, state: int | str) -> int:
        """Return the period of a state.

        The period is ``gcd{n >= 1 : (P^n)_ii > 0}``. It is computed within the
        communicating class by breadth-first levels ``d``: the gcd of
        ``d(u) + 1 - d(v)`` over all class edges ``u -> v``.

        Args:
            state: State index or label.

        Returns:
            Period, or ``0`` when the chain can never return to ``state``.

        Raises:
            ValueError: If the state is unknown.

        Equation:
            [EQ-PROC-MC-004].

        References:
            [REF-NORRIS-1997, Section 1.8].

        """
        start = self.index(state)
        members = next(c for c in self.communicating_classes() if start in c)
        inside = set(members)
        level = {start: 0}
        frontier = [start]
        while frontier:
            following: list[int] = []
            for u in frontier:
                for v in members:
                    if self.transition[u][v] > 0.0 and v not in level:
                        level[v] = level[u] + 1
                        following.append(v)
            frontier = following
        divisor = 0
        for u in members:
            for v in range(self.size):
                if v in inside and self.transition[u][v] > 0.0:
                    divisor = math.gcd(divisor, abs(level[u] + 1 - level[v]))
        return divisor

    def is_aperiodic(self) -> bool:
        """Return whether the chain is irreducible with period one.

        References:
            [REF-NORRIS-1997, Section 1.8].

        """
        return self.is_irreducible() and self.period(0) == 1

    def stationary_distribution(self) -> Vector:
        """Return the unique stationary distribution ``pi = pi P``.

        The balance system ``(P' - I) pi = 0`` with one equation replaced by
        ``sum(pi) = 1`` is solved directly. A unique solution exists exactly
        when the chain has a single closed class; transient states then
        receive zero mass.

        Returns:
            Stationary probability vector.

        Raises:
            ValueError: If the chain has more than one closed class, so the
                stationary distribution is not unique.

        Equation:
            [EQ-PROC-MC-003].

        References:
            [REF-NORRIS-1997, Section 1.7].

        """
        closed = self.closed_classes()
        if len(closed) != 1:
            raise ValueError(
                f"stationary distribution is not unique: {len(closed)} closed "
                "classes; analyze each closed class separately"
            )
        size = self.size
        rows = [
            [self.transition[j][i] - (1.0 if i == j else 0.0) for j in range(size)]
            for i in range(size)
        ]
        rows[-1] = [1.0] * size
        right = tuple(0.0 for _ in range(size - 1)) + (1.0,)
        solution = lu_solve(tuple(tuple(row) for row in rows), right)
        clipped = tuple(max(0.0, value) for value in solution)
        total = math.fsum(clipped)
        return tuple(value / total for value in clipped)

    def absorption(self) -> AbsorptionAnalysis:
        """Analyze absorption through the fundamental matrix.

        Writes ``P`` in canonical form with transient block ``Q`` and
        transient-to-absorbing block ``R`` and returns ``N = (I - Q)^-1``,
        ``B = N R``, and ``t = N 1``.

        Returns:
            Absorption analysis.

        Raises:
            ValueError: If there is no absorbing or no transient state, or a
                closed class is not a single absorbing state.

        Equation:
            [EQ-PROC-MC-005].

        References:
            [REF-KEMENY-SNELL-1976, Chapter III], [REF-NORRIS-1997, Section 1.3].

        """
        absorbing = self.absorbing_states()
        if not absorbing:
            raise ValueError("chain has no absorbing state")
        if any(len(members) != 1 for members in self.closed_classes()):
            raise ValueError(
                "every closed class must be a single absorbing state; "
                "lump recurrent classes before absorption analysis"
            )
        transient = self.transient_states()
        if not transient:
            raise ValueError("chain has no transient state")
        size = len(transient)
        system = tuple(
            tuple(
                (1.0 if a == b else 0.0) - self.transition[i][j]
                for b, j in enumerate(transient)
            )
            for a, i in enumerate(transient)
        )
        columns = tuple(
            lu_solve(system, tuple(1.0 if r == c else 0.0 for r in range(size)))
            for c in range(size)
        )
        fundamental = tuple(
            tuple(columns[c][r] for c in range(size)) for r in range(size)
        )
        probabilities = tuple(
            tuple(
                math.fsum(
                    fundamental[r][c] * self.transition[transient[c]][k]
                    for c in range(size)
                )
                for k in absorbing
            )
            for r in range(size)
        )
        steps = tuple(math.fsum(row) for row in fundamental)
        return AbsorptionAnalysis(
            transient, absorbing, fundamental, probabilities, steps
        )

    def mean_first_passage_times(self) -> Matrix:
        """Return the matrix of mean first passage times.

        For ``i != j``, ``m_ij = E[min{n >= 1 : X_n = j} | X_0 = i]`` solves
        ``m_ij = 1 + sum_(k != j) P_ik m_kj``; the diagonal holds the mean
        return times ``m_jj = 1 / pi_j``.

        Returns:
            Matrix ``M`` with ``M[i][j] = m_ij``.

        Raises:
            ValueError: If the chain is not irreducible.

        Equation:
            [EQ-PROC-MC-006].

        References:
            [REF-NORRIS-1997, Sections 1.3 and 1.7].

        """
        if not self.is_irreducible():
            raise ValueError("mean first passage times require an irreducible chain")
        size = self.size
        stationary = self.stationary_distribution()
        result = [[0.0] * size for _ in range(size)]
        for j in range(size):
            others = tuple(k for k in range(size) if k != j)
            if others:
                system = tuple(
                    tuple(
                        (1.0 if a == b else 0.0) - self.transition[i][k]
                        for b, k in enumerate(others)
                    )
                    for a, i in enumerate(others)
                )
                times = lu_solve(system, tuple(1.0 for _ in others))
                for a, i in enumerate(others):
                    result[i][j] = times[a]
            result[j][j] = 1.0 / stationary[j]
        return tuple(tuple(row) for row in result)

    def simulate(
        self,
        steps: int,
        *,
        initial: int | str | Sequence[float] = 0,
        seed: int | RandomStream | None = None,
    ) -> tuple[int, ...]:
        """Simulate one path ``X_0, ..., X_n`` by inverse-CDF row sampling.

        Args:
            steps: Non-negative number of transitions ``n``.
            initial: Starting state (index or label) or a probability vector
                from which ``X_0`` is drawn.
            seed: Integer seed or stream; ``None`` is not reproducible.

        Returns:
            Tuple of ``n + 1`` visited state indices.

        Raises:
            ValueError: If ``initial`` or ``steps`` is invalid.

        Equation:
            [EQ-PROC-MC-001].

        References:
            [REF-NORRIS-1997, Section 1.1], [REF-DEVROYE-1986, Chapter III].

        """
        count = require_non_negative_integer(steps, name="steps")
        generator = resolve_stream(seed).child("markov_chain").generator()
        cumulative = tuple(_cumulative(row) for row in self.transition)
        if isinstance(initial, int | str):
            state = self.index(initial)
        else:
            state = _draw(_cumulative(self._distribution(initial)), generator.random())
        path = [state]
        for _ in range(count):
            state = _draw(cumulative[state], generator.random())
            path.append(state)
        return tuple(path)


def _cumulative(probabilities: Sequence[float]) -> Vector:
    """Return running sums of a probability vector with the last forced to one.

    Args:
        probabilities: Non-negative weights summing to one.

    Returns:
        Non-decreasing cumulative sums ending at exactly one.

    References:
        [REF-DEVROYE-1986, Chapter III]. No named equation is needed.

    """
    running: list[float] = []
    total = 0.0
    for value in probabilities:
        total += value
        running.append(total)
    running[-1] = 1.0
    return tuple(running)


def _draw(cumulative: Vector, uniform: float) -> int:
    """Return the first index whose cumulative probability exceeds ``uniform``.

    Args:
        cumulative: Output of :func:`_cumulative`.
        uniform: Draw from ``[0, 1)``.

    Returns:
        Sampled index; zero-probability states are never returned.

    References:
        [REF-DEVROYE-1986, Chapter III]. No named equation is needed.

    """
    return bisect.bisect_right(cumulative, uniform)
