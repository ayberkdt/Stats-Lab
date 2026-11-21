"""Finite Markov decision processes solved by dynamic programming.

A finite Markov decision process has states ``s``, actions ``a``, transition
probabilities ``P_a(s, s')``, expected one-step rewards ``r_a(s)``, and a
discount factor ``gamma`` in ``[0, 1)``. Every action is available in every
state. For a stationary policy ``pi`` the value function is the unique
solution of the linear system ``(I - gamma P_pi) V = r_pi``; the optimal
value function is the unique fixed point of the Bellman operator
``(T V)(s) = max_a [r_a(s) + gamma sum_s' P_a(s, s') V(s')]``, which is a
``gamma``-contraction in the sup norm.

Two error bounds follow from the contraction property alone and are reported
with every solution. Write ``rho = |T V - V|_inf`` for the Bellman residual
of any ``V`` and ``pi`` for a policy that is greedy for ``V`` (so
``T_pi V = T V``). Then

- ``|V - V*| <= |V - T V| + |T V - T V*| <= rho + gamma |V - V*|``, hence
  ``|V - V*| <= rho / (1 - gamma)``;
- ``|V_pi - V| <= |T_pi V_pi - T_pi V| + |T V - V| <= gamma |V_pi - V| + rho``,
  hence ``|V_pi - V| <= rho / (1 - gamma)``;
- by the triangle inequality ``|V_pi - V*| <= 2 rho / (1 - gamma)``.

Value iteration ``V_{k+1} = T V_k`` has ``rho(V_{k+1}) <= gamma delta`` with
``delta = |V_{k+1} - V_k|_inf``, so stopping when
``delta < eps (1 - gamma) / (2 gamma)`` returns a greedy policy whose value is
within ``eps`` of optimal in every state, and values within ``eps / 2``.
The bound allows the returned greedy policy to be suboptimal: for the
two-state problem of the tests (``gamma = 0.9``, ``eps = 1``) value
iteration stops at a policy whose value is 0.0244 below the optimum in one
state.

Sources: Norris treats Markov decision processes as an application of
Markov chains [REF-NORRIS-1997, Section 5.4]; the discounted theory with
value iteration, this stopping rule, and policy iteration is in
[REF-PUTERMAN-1994, Chapter 6]. The bounds above are re-derived here and
checked against enumeration of all policies.

Tabular Q-learning is not provided: the bibliography of this library holds
no verified source for it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, lu_solve

_ROW_SUM_TOLERANCE = 1e-9
_IMPROVEMENT_TOLERANCE = 1e-12

Policy = Sequence[int] | Sequence[Sequence[float]]


def _labels(
    values: Iterable[object] | None, count: int, *, name: str
) -> tuple[str, ...]:
    """Return ``count`` distinct text labels, defaulting to ``"0", "1", ...``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if values is None:
        return tuple(str(index) for index in range(count))
    labels = tuple(values)
    if len(labels) != count:
        raise ValueError(f"{name} must have {count} labels, got {len(labels)}")
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{name} must be non-blank strings")
    if len(set(labels)) != count:
        raise ValueError(f"{name} must be distinct")
    return labels  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FiniteMDP:
    """Validated finite discounted Markov decision process.

    Args:
        transitions: ``P[a][s][s']``, the probability of moving from state
            ``s`` to ``s'`` under action ``a``; every row sums to one.
        rewards: Either ``R[a][s]`` (expected one-step reward) or
            ``R[a][s][s']`` (reward of the transition), which is reduced to
            ``r_a(s) = sum_s' P_a(s, s') R[a][s][s']``.
        discount: Discount factor ``gamma`` with ``0 <= gamma < 1``.
        states: Optional distinct state labels.
        actions: Optional distinct action labels.

    Raises:
        TypeError: If an entry is not a real number.
        ValueError: If shapes disagree, a probability is negative, a row
            does not sum to one within ``1e-9``, or ``gamma`` is outside
            ``[0, 1)``.

    Attributes:
        expected_rewards: ``r[a][s]`` after reduction.

    References:
        [EQ-RISK-MDP-001], [REF-NORRIS-1997].

    """

    transitions: tuple[Matrix, ...]
    rewards: tuple[Vector, ...] | tuple[Matrix, ...]
    discount: float
    states: tuple[str, ...] | None = None
    actions: tuple[str, ...] | None = None
    expected_rewards: tuple[Vector, ...] = ()

    def __post_init__(self) -> None:
        """Validate the tensors and store the expected rewards.

        References:
            [EQ-RISK-MDP-001], [REF-NORRIS-1997].

        """
        gamma = require_finite_real(self.discount, name="discount")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"discount must satisfy 0 <= gamma < 1, got {gamma}")
        tensor = tuple(tuple(tuple(row) for row in layer) for layer in self.transitions)
        if not tensor or not tensor[0]:
            raise ValueError("transitions must hold at least one action and state")
        n_states = len(tensor[0])
        checked: list[Matrix] = []
        for a, layer in enumerate(tensor):
            if len(layer) != n_states:
                raise ValueError(
                    f"transitions[{a}] must have {n_states} rows, got {len(layer)}"
                )
            rows: list[Vector] = []
            for s, row in enumerate(layer):
                if len(row) != n_states:
                    raise ValueError(
                        f"transitions[{a}][{s}] must have {n_states} entries"
                    )
                values = tuple(
                    require_finite_real(p, name=f"transitions[{a}][{s}][{t}]")
                    for t, p in enumerate(row)
                )
                if min(values) < 0.0:
                    raise ValueError(
                        f"transitions[{a}][{s}] holds a negative probability"
                    )
                total = math.fsum(values)
                if abs(total - 1.0) > _ROW_SUM_TOLERANCE:
                    raise ValueError(
                        f"transitions[{a}][{s}] must sum to one, got {total}"
                    )
                rows.append(values)
            checked.append(tuple(rows))
        raw = tuple(self.rewards)
        if len(raw) != len(checked):
            raise ValueError(
                f"rewards must have one block per action ({len(checked)}), "
                f"got {len(raw)}"
            )
        stored: list[tuple[object, ...]] = []
        expected: list[Vector] = []
        for a, block in enumerate(raw):
            entries = tuple(block)
            if len(entries) != n_states:
                raise ValueError(
                    f"rewards[{a}] must have {n_states} entries, got {len(entries)}"
                )
            if all(isinstance(entry, (tuple, list)) for entry in entries):
                table: list[Vector] = []
                for s, entry in enumerate(entries):
                    cells: tuple[object, ...] = tuple(entry)  # type: ignore[arg-type]
                    row = tuple(
                        require_finite_real(v, name=f"rewards[{a}][{s}][{t}]")
                        for t, v in enumerate(cells)
                    )
                    if len(row) != n_states:
                        raise ValueError(
                            f"rewards[{a}][{s}] must have {n_states} entries"
                        )
                    table.append(row)
                stored.append(tuple(table))
                expected.append(
                    tuple(
                        math.fsum(
                            p * v for p, v in zip(checked[a][s], table[s], strict=True)
                        )
                        for s in range(n_states)
                    )
                )
            elif any(isinstance(entry, (tuple, list)) for entry in entries):
                raise ValueError(
                    f"rewards[{a}] mixes scalar and per-transition entries"
                )
            else:
                vector = tuple(
                    require_finite_real(v, name=f"rewards[{a}][{s}]")
                    for s, v in enumerate(entries)
                )
                stored.append(vector)
                expected.append(vector)
        object.__setattr__(self, "transitions", tuple(checked))
        object.__setattr__(self, "rewards", tuple(stored))
        object.__setattr__(self, "discount", gamma)
        object.__setattr__(
            self, "states", _labels(self.states, n_states, name="states")
        )
        object.__setattr__(
            self, "actions", _labels(self.actions, len(checked), name="actions")
        )
        object.__setattr__(self, "expected_rewards", tuple(expected))

    @property
    def n_states(self) -> int:
        """Return the number of states.

        References:
            [REF-NORRIS-1997]. No equation is evaluated.

        """
        return len(self.transitions[0])

    @property
    def n_actions(self) -> int:
        """Return the number of actions.

        References:
            [REF-NORRIS-1997]. No equation is evaluated.

        """
        return len(self.transitions)


@dataclass(frozen=True, slots=True)
class MDPSolution:
    """Values and greedy policy returned by a dynamic-programming solver.

    Args:
        method: ``"value_iteration"`` or ``"policy_iteration"``.
        values: Value estimate ``V(s)`` per state.
        policy: Greedy action index per state for ``values``.
        q_values: ``Q[s][a] = r_a(s) + gamma sum_s' P_a(s, s') V(s')``.
        iterations: Bellman sweeps (value iteration) or policy improvement
            steps (policy iteration).
        converged: Whether the stopping rule was met within the iteration
            limit. An unconverged solution is still returned with its
            residual.
        bellman_residual: ``|T V - V|_inf`` of ``values``.
        last_change: ``|V_{k+1} - V_k|_inf`` of the final value-iteration
            sweep; ``0.0`` for policy iteration.
        discount: The discount factor of the problem.
        backups: Number of state-action backups ``r + gamma P V`` evaluated.
        linear_solves: Number of policy-evaluation linear systems solved.

    References:
        [EQ-RISK-MDP-002], [EQ-RISK-MDP-006], [REF-NORRIS-1997].

    """

    method: str
    values: Vector
    policy: tuple[int, ...]
    q_values: Matrix
    iterations: int
    converged: bool
    bellman_residual: float
    last_change: float
    discount: float
    backups: int
    linear_solves: int

    @property
    def value_error_bound(self) -> float:
        """Return the bound ``rho / (1 - gamma)`` on ``|V - V*|_inf``.

        References:
            [EQ-RISK-MDP-006].

        """
        return self.bellman_residual / (1.0 - self.discount)

    @property
    def policy_loss_bound(self) -> float:
        """Return the bound ``2 rho / (1 - gamma)`` on ``|V_pi - V*|_inf``.

        ``pi`` is the returned greedy policy. The bound ignores rounding in
        the residual itself.

        References:
            [EQ-RISK-MDP-006].

        """
        return 2.0 * self.bellman_residual / (1.0 - self.discount)


@dataclass(frozen=True, slots=True)
class FiniteHorizonSolution:
    """Stage-dependent values and decisions from backward induction.

    Args:
        values: ``values[t][s]`` is the optimal expected discounted reward
            from stage ``t`` in state ``s`` with ``horizon - t`` decisions
            left; ``values[horizon]`` is the terminal value.
        policy: ``policy[t][s]`` is an optimal action at stage ``t``
            (first index on ties), for ``t = 0, ..., horizon - 1``.
        horizon: Number of decision stages.
        discount: Discount factor used.
        backups: Number of state-action backups evaluated.

    References:
        [EQ-RISK-MDP-005], [REF-NORRIS-1997].

    """

    values: tuple[Vector, ...]
    policy: tuple[tuple[int, ...], ...]
    horizon: int
    discount: float
    backups: int


def _require_mdp(mdp: object) -> FiniteMDP:
    """Return ``mdp`` if it is a :class:`FiniteMDP`.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if not isinstance(mdp, FiniteMDP):
        raise TypeError("mdp must be a FiniteMDP")
    return mdp


def _values(mdp: FiniteMDP, values: Iterable[object] | None, *, name: str) -> Vector:
    """Validate a value vector of length ``n_states`` (zeros when ``None``).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if values is None:
        return (0.0,) * mdp.n_states
    vector = tuple(
        require_finite_real(v, name=f"{name}[{s}]") for s, v in enumerate(values)
    )
    if len(vector) != mdp.n_states:
        raise ValueError(
            f"{name} must have one entry per state ({mdp.n_states}), got {len(vector)}"
        )
    return vector


def _backup(mdp: FiniteMDP, values: Vector, gamma: float) -> Matrix:
    """Return ``Q[s][a] = r_a(s) + gamma sum_s' P_a(s, s') V(s')``.

    References:
        [EQ-RISK-MDP-002], [REF-NORRIS-1997].

    """
    return tuple(
        tuple(
            mdp.expected_rewards[a][s]
            + gamma
            * math.fsum(
                p * v for p, v in zip(mdp.transitions[a][s], values, strict=True)
            )
            for a in range(mdp.n_actions)
        )
        for s in range(mdp.n_states)
    )


def _argmax(row: Sequence[float]) -> int:
    """Return the first index of the largest entry.

    References:
        [EQ-RISK-MDP-002]. Ties go to the lowest index.

    """
    best = 0
    for index in range(1, len(row)):
        if row[index] > row[best]:
            best = index
    return best


def _policy_matrix(mdp: FiniteMDP, policy: Policy) -> Matrix:
    """Return a policy as action probabilities ``pi[s][a]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    entries = tuple(policy)
    if len(entries) != mdp.n_states:
        raise ValueError(
            f"policy must have one entry per state ({mdp.n_states}), got {len(entries)}"
        )
    rows: list[Vector] = []
    for s, entry in enumerate(entries):
        if isinstance(entry, (tuple, list)):
            row = tuple(
                require_finite_real(p, name=f"policy[{s}][{a}]")
                for a, p in enumerate(entry)
            )
            if len(row) != mdp.n_actions:
                raise ValueError(f"policy[{s}] must have {mdp.n_actions} probabilities")
            if min(row) < 0.0 or abs(math.fsum(row) - 1.0) > _ROW_SUM_TOLERANCE:
                raise ValueError(
                    f"policy[{s}] must be non-negative probabilities summing to one"
                )
            rows.append(row)
        else:
            action = require_non_negative_integer(entry, name=f"policy[{s}]")
            if action >= mdp.n_actions:
                raise ValueError(
                    f"policy[{s}] = {action} is not an action index below "
                    f"{mdp.n_actions}"
                )
            rows.append(
                tuple(1.0 if a == action else 0.0 for a in range(mdp.n_actions))
            )
    return tuple(rows)


def policy_evaluation(mdp: FiniteMDP, policy: Policy) -> Vector:
    """Return the value function of a stationary policy by a linear solve.

    With ``P_pi(s, s') = sum_a pi(a|s) P_a(s, s')`` and
    ``r_pi(s) = sum_a pi(a|s) r_a(s)`` the value function satisfies
    ``V = r_pi + gamma P_pi V``. Because ``gamma < 1`` and ``P_pi`` is
    stochastic, ``I - gamma P_pi`` is strictly diagonally dominant by rows
    and therefore non-singular.

    Args:
        mdp: The decision process.
        policy: One action index per state (deterministic), or one row of
            action probabilities per state (stochastic).

    Returns:
        ``V_pi(s)`` per state.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or the policy has
            entries of the wrong type.
        ValueError: If the policy has the wrong shape, an unknown action, or
            probabilities that do not sum to one.

    Equation:
        [EQ-RISK-MDP-001].

    References:
        [REF-NORRIS-1997], [REF-GOLUB-VANLOAN-2013] (the linear solve).

    """
    problem = _require_mdp(mdp)
    pi = _policy_matrix(problem, policy)
    n = problem.n_states
    gamma = problem.discount
    system: list[Vector] = []
    right: list[float] = []
    for s in range(n):
        row = [
            -gamma
            * math.fsum(
                pi[s][a] * problem.transitions[a][s][t]
                for a in range(problem.n_actions)
            )
            for t in range(n)
        ]
        row[s] += 1.0
        system.append(tuple(row))
        right.append(
            math.fsum(
                pi[s][a] * problem.expected_rewards[a][s]
                for a in range(problem.n_actions)
            )
        )
    return lu_solve(tuple(system), tuple(right))


def action_values(mdp: FiniteMDP, values: Iterable[object]) -> Matrix:
    """Return ``Q[s][a] = r_a(s) + gamma sum_s' P_a(s, s') V(s')``.

    Args:
        mdp: The decision process.
        values: A value function ``V``, one entry per state.

    Returns:
        Matrix with one row per state and one column per action.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or a value is not
            real.
        ValueError: If ``values`` has the wrong length.

    Equation:
        [EQ-RISK-MDP-002].

    References:
        [REF-NORRIS-1997].

    """
    problem = _require_mdp(mdp)
    return _backup(problem, _values(problem, values, name="values"), problem.discount)


def greedy_policy(mdp: FiniteMDP, values: Iterable[object]) -> tuple[int, ...]:
    """Return ``argmax_a Q[s][a]`` per state for a value function.

    Args:
        mdp: The decision process.
        values: A value function ``V``, one entry per state.

    Returns:
        One action index per state; ties go to the lowest index.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or a value is not
            real.
        ValueError: If ``values`` has the wrong length.

    Equation:
        [EQ-RISK-MDP-002].

    References:
        [REF-NORRIS-1997].

    """
    return tuple(_argmax(row) for row in action_values(mdp, values))


def bellman_residual(mdp: FiniteMDP, values: Iterable[object]) -> float:
    """Return ``|T V - V|_inf`` for the Bellman optimality operator ``T``.

    The residual is zero exactly for the optimal value function, and
    ``|V - V*|_inf <= residual / (1 - gamma)`` for any ``V``.

    Args:
        mdp: The decision process.
        values: A value function ``V``, one entry per state.

    Returns:
        The non-negative sup-norm residual.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or a value is not
            real.
        ValueError: If ``values`` has the wrong length.

    Equation:
        [EQ-RISK-MDP-002], [EQ-RISK-MDP-006].

    References:
        [REF-NORRIS-1997].

    """
    problem = _require_mdp(mdp)
    vector = _values(problem, values, name="values")
    table = _backup(problem, vector, problem.discount)
    return max(abs(max(row) - v) for row, v in zip(table, vector, strict=True))


def _solution(
    mdp: FiniteMDP,
    method: str,
    values: Vector,
    *,
    iterations: int,
    converged: bool,
    last_change: float,
    backups: int,
    linear_solves: int,
) -> MDPSolution:
    """Assemble a solution with Q-values, greedy policy, and residual.

    References:
        [EQ-RISK-MDP-002], [EQ-RISK-MDP-006].

    """
    table = _backup(mdp, values, mdp.discount)
    return MDPSolution(
        method=method,
        values=values,
        policy=tuple(_argmax(row) for row in table),
        q_values=table,
        iterations=iterations,
        converged=converged,
        bellman_residual=max(
            abs(max(row) - v) for row, v in zip(table, values, strict=True)
        ),
        last_change=last_change,
        discount=mdp.discount,
        backups=backups + mdp.n_states * mdp.n_actions,
        linear_solves=linear_solves,
    )


def value_iteration(
    mdp: FiniteMDP,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100_000,
    initial_values: Iterable[object] | None = None,
) -> MDPSolution:
    """Solve a discounted MDP by iterating the Bellman optimality operator.

    Iterates ``V_{k+1} = T V_k`` and stops when
    ``|V_{k+1} - V_k|_inf < tolerance (1 - gamma) / (2 gamma)``. At that
    point the greedy policy for ``V_{k+1}`` has a value within ``tolerance``
    of the optimum in every state and ``V_{k+1}`` is within
    ``tolerance / 2`` of ``V*`` (see the module docstring for the
    derivation). With ``gamma = 0`` one sweep is exact.

    The number of sweeps grows like ``log(1 / tolerance) / log(1 / gamma)``,
    so problems with ``gamma`` close to one are better served by
    :func:`policy_iteration`.

    Args:
        mdp: The decision process.
        tolerance: Positive bound ``eps`` on the loss of the returned greedy
            policy.
        max_iterations: Positive limit on the number of sweeps.
        initial_values: Starting ``V_0``; zeros when omitted.

    Returns:
        Solution with ``converged=False`` if the limit was reached first.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or an argument has
            the wrong type.
        ValueError: If ``tolerance`` or ``max_iterations`` is not positive or
            ``initial_values`` has the wrong length.

    Equation:
        [EQ-RISK-MDP-002], [EQ-RISK-MDP-003], [EQ-RISK-MDP-006].

    References:
        [REF-PUTERMAN-1994, Chapter 6], [REF-NORRIS-1997, Section 5.4].

    """
    problem = _require_mdp(mdp)
    eps = require_finite_real(tolerance, name="tolerance")
    if eps <= 0.0:
        raise ValueError(f"tolerance must be positive, got {eps}")
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    if limit == 0:
        raise ValueError("max_iterations must be positive")
    gamma = problem.discount
    threshold = math.inf if gamma == 0.0 else eps * (1.0 - gamma) / (2.0 * gamma)
    values = _values(problem, initial_values, name="initial_values")
    change = math.inf
    converged = False
    sweeps = 0
    while sweeps < limit:
        updated = tuple(max(row) for row in _backup(problem, values, gamma))
        change = max(abs(u - v) for u, v in zip(updated, values, strict=True))
        values = updated
        sweeps += 1
        if change < threshold:
            converged = True
            break
    return _solution(
        problem,
        "value_iteration",
        values,
        iterations=sweeps,
        converged=converged,
        last_change=change,
        backups=sweeps * problem.n_states * problem.n_actions,
        linear_solves=0,
    )


def policy_iteration(
    mdp: FiniteMDP,
    *,
    initial_policy: Sequence[int] | None = None,
    max_iterations: int = 1000,
) -> MDPSolution:
    """Solve a discounted MDP by alternating evaluation and improvement.

    Each step solves ``(I - gamma P_pi) V = r_pi`` exactly and then switches,
    in every state, to the action with the largest ``Q[s][a]``. An action is
    replaced only if it improves ``Q`` by more than ``1e-12`` (relative to
    ``max(1, |Q|)``), so ties and rounding noise never change the policy. In
    exact arithmetic every switch strictly improves the value function, no
    deterministic policy is visited twice, and the method stops after at most
    ``n_actions ** n_states`` steps at an optimal policy.

    Args:
        mdp: The decision process.
        initial_policy: Starting action index per state; action ``0``
            everywhere when omitted.
        max_iterations: Positive limit on the number of improvement steps.

    Returns:
        Solution whose ``values`` are the exact values of the returned
        policy (up to rounding in the linear solve); ``converged=False`` if
        the limit was reached first.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or an argument has
            the wrong type.
        ValueError: If ``max_iterations`` is not positive or the initial
            policy is invalid.

    Equation:
        [EQ-RISK-MDP-001], [EQ-RISK-MDP-002], [EQ-RISK-MDP-004].

    References:
        [REF-PUTERMAN-1994, Chapter 6], [REF-NORRIS-1997, Section 5.4].

    """
    problem = _require_mdp(mdp)
    limit = require_non_negative_integer(max_iterations, name="max_iterations")
    if limit == 0:
        raise ValueError("max_iterations must be positive")
    if initial_policy is None:
        policy = [0] * problem.n_states
    else:
        start = tuple(initial_policy)
        if any(isinstance(entry, (tuple, list)) for entry in start):
            raise ValueError("initial_policy must be deterministic action indices")
        policy = [_argmax(row) for row in _policy_matrix(problem, start)]
    converged = False
    steps = 0
    backups = 0
    values: Vector = ()
    while steps < limit:
        values = policy_evaluation(problem, policy)
        table = _backup(problem, values, problem.discount)
        backups += problem.n_states * problem.n_actions
        steps += 1
        changed = False
        for s, row in enumerate(table):
            best = _argmax(row)
            margin = _IMPROVEMENT_TOLERANCE * max(1.0, abs(row[policy[s]]))
            if row[best] > row[policy[s]] + margin:
                policy[s] = best
                changed = True
        if not changed:
            converged = True
            break
    if not converged:
        values = policy_evaluation(problem, policy)
        steps_solved = steps + 1
    else:
        steps_solved = steps
    return _solution(
        problem,
        "policy_iteration",
        values,
        iterations=steps,
        converged=converged,
        last_change=0.0,
        backups=backups,
        linear_solves=steps_solved,
    )


def backward_induction(
    mdp: FiniteMDP,
    horizon: int,
    *,
    terminal_values: Iterable[object] | None = None,
    discount: float | None = None,
) -> FiniteHorizonSolution:
    """Solve the finite-horizon problem by backward induction.

    Starting from ``V_N = terminal_values`` the recursion
    ``V_t(s) = max_a [r_a(s) + gamma sum_s' P_a(s, s') V_{t+1}(s')]`` gives
    the optimal expected discounted reward of the remaining ``N - t``
    decisions; the maximizing actions form an optimal, generally
    non-stationary, policy. Transitions and rewards are the stationary ones
    of ``mdp``.

    Args:
        mdp: The decision process.
        horizon: Positive number of decision stages ``N``.
        terminal_values: Value of ending in each state; zeros when omitted.
        discount: Discount in ``[0, 1]`` to use instead of ``mdp.discount``;
            ``1.0`` gives the undiscounted total reward.

    Returns:
        Values for stages ``0, ..., N`` and decisions for stages
        ``0, ..., N - 1``.

    Raises:
        TypeError: If ``mdp`` is not a :class:`FiniteMDP` or an argument has
            the wrong type.
        ValueError: If ``horizon`` is not positive, ``discount`` is outside
            ``[0, 1]``, or ``terminal_values`` has the wrong length.

    Equation:
        [EQ-RISK-MDP-005].

    References:
        [REF-NORRIS-1997].

    """
    problem = _require_mdp(mdp)
    stages = require_non_negative_integer(horizon, name="horizon")
    if stages == 0:
        raise ValueError("horizon must be positive")
    if discount is None:
        gamma = problem.discount
    else:
        gamma = require_finite_real(discount, name="discount")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"discount must lie in [0, 1], got {gamma}")
    current = _values(problem, terminal_values, name="terminal_values")
    values = [current]
    decisions: list[tuple[int, ...]] = []
    for _ in range(stages):
        table = _backup(problem, current, gamma)
        decisions.append(tuple(_argmax(row) for row in table))
        current = tuple(max(row) for row in table)
        values.append(current)
    values.reverse()
    decisions.reverse()
    return FiniteHorizonSolution(
        values=tuple(values),
        policy=tuple(decisions),
        horizon=stages,
        discount=gamma,
        backups=stages * problem.n_states * problem.n_actions,
    )
