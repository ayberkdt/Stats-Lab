"""Performance and data profiles for benchmarking solvers on problem sets.

A *performance profile* (Dolan and Moré) compares solvers by the ratio of
each solver's cost on a problem to the best cost any solver achieved on it;
``rho_s(tau)`` is the fraction of problems solver ``s`` solves within a factor
``tau`` of the best. A *data profile* (Moré and Wild) answers a budget
question instead: ``d_s(kappa)`` is the fraction of problems solved with at
most ``kappa (n_p + 1)`` function evaluations, ``n_p`` being the problem
dimension (one simplex gradient costs ``n_p + 1`` evaluations).

Costs are any positive performance measure (evaluations, time, error at a
fixed budget). A failure is passed as ``None``, ``inf``, or ``NaN`` and has
ratio ``r = inf``: it never counts as solved, which is the convention
``r_M > max r`` of Dolan and Moré with ``r_M`` sent to infinity.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Cost = float | int | None


def _failure(value: Cost) -> bool:
    """Return whether a cost entry marks a failure.

    Args:
        value: Cost entry.

    Returns:
        ``True`` for ``None``, NaN, and positive infinity.

    References:
        [EQ-NUM-PROFILE-001], [REF-DOLAN-MORE-2002, Section 2].

    """
    return value is None or math.isnan(value) or value == math.inf


def _cost_table(
    costs: Mapping[str, Sequence[Cost]], *, name: str
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...], int]:
    """Validate a solver-by-problem cost table.

    Args:
        costs: Per solver, one cost per problem (same problem order).
        name: Argument name for error messages.

    Returns:
        Solver names, costs with failures as ``inf``, and the problem count.

    Raises:
        TypeError: If a cost is not a real number or ``None``.
        ValueError: If the table is empty, rows differ in length, or a cost is
            not positive.

    References:
        [EQ-NUM-PROFILE-001], [REF-DOLAN-MORE-2002, Section 2].

    """
    if not costs:
        raise ValueError(f"{name} must name at least one solver")
    solvers = tuple(costs)
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for solver in solvers:
        row = costs[solver]
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise TypeError(f"{name}[{solver!r}] must be a sequence of costs")
        values: list[float] = []
        for value in row:
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float)
            ):
                raise TypeError(f"{name}[{solver!r}] holds a non-real cost")
            if value is None or _failure(value):
                values.append(math.inf)
                continue
            if not value > 0.0:
                raise ValueError(
                    f"{name}[{solver!r}] holds a non-positive cost; costs must be "
                    "positive (use None, inf, or NaN for a failure)"
                )
            values.append(float(value))
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError(f"every solver in {name} needs one cost per problem")
        rows.append(tuple(values))
    if not width:
        raise ValueError(f"{name} must contain at least one problem")
    return solvers, tuple(rows), width


def performance_ratios(
    costs: Mapping[str, Sequence[Cost]],
) -> dict[str, tuple[float, ...]]:
    """Return the performance ratios ``r_(p,s) = t_(p,s) / min_s t_(p,s)``.

    Args:
        costs: Per solver, one positive cost ``t_(p,s)`` per problem in a
            common order; ``None``, ``inf``, or ``NaN`` marks a failure.

    Returns:
        Per solver, the ratio on each problem (``1`` for the best solver,
        ``inf`` for a failure or a problem no solver solved).

    Raises:
        TypeError: If a cost is not a real number or ``None``.
        ValueError: If the table is empty or ragged or a cost is not positive.

    Equation:
        [EQ-NUM-PROFILE-001].

    References:
        [REF-DOLAN-MORE-2002, Section 2].

    """
    solvers, rows, count = _cost_table(costs, name="costs")
    best = [min(row[p] for row in rows) for p in range(count)]
    return {
        solver: tuple(
            row[p] / best[p] if math.isfinite(row[p]) else math.inf
            for p in range(count)
        )
        for solver, row in zip(solvers, rows, strict=True)
    }


def _require_fraction(fraction: float) -> float:
    """Validate a target fraction in ``(0, 1]``.

    Args:
        fraction: Candidate fraction of problems.

    Returns:
        The fraction as a float.

    Raises:
        TypeError: If ``fraction`` is not real.
        ValueError: If it lies outside ``(0, 1]``.

    References:
        [EQ-NUM-PROFILE-002]. No equation is evaluated.

    """
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise TypeError("fraction must be a real number")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    return float(fraction)


def _required_count(fraction: float, problems: int) -> int:
    """Return the number of problems a fraction of ``problems`` requires.

    ``ceil(fraction * problems)`` with a tolerance of ``1e-9`` problems, so
    that ``0.3 * 10`` asks for three problems despite binary rounding.

    Args:
        fraction: Fraction in ``(0, 1]``.
        problems: Problem count.

    Returns:
        Required count, at least one.

    References:
        [EQ-NUM-PROFILE-002]. No equation is evaluated.

    """
    return max(1, math.ceil(fraction * problems - 1e-9))


def _step_value(sorted_values: Sequence[float], threshold: float, total: int) -> float:
    """Return the fraction of sorted values at or below ``threshold``.

    Args:
        sorted_values: Ascending values (``inf`` for failures).
        threshold: Abscissa of the right-continuous step function.
        total: Number of problems.

    Returns:
        ``|{v <= threshold}| / total``.

    References:
        [EQ-NUM-PROFILE-002], [REF-DOLAN-MORE-2002, Section 2].

    """
    return bisect.bisect_right(sorted_values, threshold) / total


def _breakpoints(
    sorted_values: Sequence[float], total: int
) -> tuple[tuple[float, float], ...]:
    """Return the jumps ``(abscissa, value after the jump)`` of a step profile.

    Args:
        sorted_values: Ascending values (``inf`` for failures).
        total: Number of problems.

    Returns:
        One pair per distinct finite value.

    References:
        [EQ-NUM-PROFILE-002], [REF-DOLAN-MORE-2002, Section 2].

    """
    points: list[tuple[float, float]] = []
    for index, value in enumerate(sorted_values):
        if not math.isfinite(value):
            break
        if index + 1 < len(sorted_values) and sorted_values[index + 1] == value:
            continue
        points.append((value, (index + 1) / total))
    return tuple(points)


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    """Dolan-Moré performance profiles of several solvers on one problem set.

    Args:
        solvers: Solver names in input order.
        ratios: Per solver (same order), the performance ratio on each problem.
        problems: Number of problems ``n_p``.

    References:
        [EQ-NUM-PROFILE-001], [EQ-NUM-PROFILE-002],
        [REF-DOLAN-MORE-2002, Section 2].

    """

    solvers: tuple[str, ...]
    ratios: tuple[tuple[float, ...], ...]
    problems: int

    def _sorted(self, solver: str) -> list[float]:
        """Return the ascending ratios of one solver.

        Args:
            solver: Solver name.

        Returns:
            Sorted ratios.

        Raises:
            KeyError: If the solver is unknown.

        References:
            [EQ-NUM-PROFILE-002].

        """
        if solver not in self.solvers:
            raise KeyError(f"unknown solver {solver!r}")
        return sorted(self.ratios[self.solvers.index(solver)])

    @property
    def max_ratio(self) -> float:
        """Return the largest finite ratio over all solvers (at least 1).

        References:
            [EQ-NUM-PROFILE-001].

        """
        finite = [r for row in self.ratios for r in row if math.isfinite(r)]
        return max(finite, default=1.0)

    def value(self, solver: str, tau: float) -> float:
        """Return ``rho_s(tau)``, the fraction of problems with ``r <= tau``.

        Args:
            solver: Solver name.
            tau: Ratio threshold (any real; ``rho_s`` is zero below 1).

        Returns:
            Profile value in ``[0, 1]``.

        Raises:
            KeyError: If the solver is unknown.

        Equation:
            [EQ-NUM-PROFILE-002].

        References:
            [REF-DOLAN-MORE-2002, Section 2].

        """
        return _step_value(self._sorted(solver), tau, self.problems)

    def wins(self, solver: str) -> float:
        """Return ``rho_s(1)``: the fraction of problems the solver is best on.

        Ties count as a win for every tied solver.

        Args:
            solver: Solver name.

        Returns:
            Fraction of wins.

        Equation:
            [EQ-NUM-PROFILE-002].

        References:
            [REF-DOLAN-MORE-2002, Section 2].

        """
        return self.value(solver, 1.0)

    def solved_fraction(self, solver: str) -> float:
        """Return ``rho_s*``, the fraction of problems the solver solves at all.

        Args:
            solver: Solver name.

        Returns:
            ``lim rho_s(tau)`` as ``tau`` grows.

        Equation:
            [EQ-NUM-PROFILE-002].

        References:
            [REF-DOLAN-MORE-2002, Section 2].

        """
        solved = sum(1 for r in self._sorted(solver) if math.isfinite(r))
        return solved / self.problems

    def tau_at(self, solver: str, fraction: float) -> float:
        """Return the smallest ``tau`` with ``rho_s(tau) >= fraction``.

        Args:
            solver: Solver name.
            fraction: Target fraction of problems in ``(0, 1]``.

        Returns:
            The ratio at which the solver first reaches the fraction, or
            ``inf`` if it never does.

        Raises:
            KeyError: If the solver is unknown.
            TypeError: If ``fraction`` is not real.
            ValueError: If ``fraction`` lies outside ``(0, 1]``.

        Equation:
            [EQ-NUM-PROFILE-002].

        References:
            [REF-DOLAN-MORE-2002, Section 2].

        """
        target = _required_count(_require_fraction(fraction), self.problems)
        return self._sorted(solver)[target - 1]

    def area(self, solver: str, tau_max: float | None = None) -> float:
        """Return the normalized area under ``rho_s`` on a ``log2 tau`` axis.

        ``A_s = (1/L) int_0^L rho_s(2^t) dt`` with ``L = log2 tau_max``; for
        the step function this is
        ``(1/(n_p L)) sum_(r_p <= tau_max) (L - log2 r_p)``. A solver that is
        best on every problem scores 1; one that solves nothing within
        ``tau_max`` scores 0. When ``tau_max = 1`` the area is ``rho_s(1)``.

        Args:
            solver: Solver name.
            tau_max: Upper end of the ratio axis, at least 1; ``None`` uses
                :attr:`max_ratio`, so that solvers of one profile share it.

        Returns:
            Normalized area in ``[0, 1]``.

        Raises:
            KeyError: If the solver is unknown.
            ValueError: If ``tau_max`` is below 1 or not finite.

        Equation:
            [EQ-NUM-PROFILE-003].

        References:
            [REF-DOLAN-MORE-2002, Section 2].

        """
        upper = self.max_ratio if tau_max is None else float(tau_max)
        if not math.isfinite(upper) or upper < 1.0:
            raise ValueError("tau_max must be finite and at least 1")
        ratios = self._sorted(solver)
        if upper == 1.0:
            return _step_value(ratios, 1.0, self.problems)
        span = math.log2(upper)
        total = math.fsum(span - math.log2(r) for r in ratios if r <= upper)
        return total / (self.problems * span)

    def breakpoints(self, solver: str) -> tuple[tuple[float, float], ...]:
        """Return the jumps ``(tau, rho_s(tau))`` of the step profile.

        Args:
            solver: Solver name.

        Returns:
            One pair per distinct finite ratio, ascending.

        Raises:
            KeyError: If the solver is unknown.

        Equation:
            [EQ-NUM-PROFILE-002].

        References:
            [REF-DOLAN-MORE-2002, Section 2].

        """
        return _breakpoints(self._sorted(solver), self.problems)


def performance_profile(costs: Mapping[str, Sequence[Cost]]) -> PerformanceProfile:
    """Build Dolan-Moré performance profiles from a cost table.

    Args:
        costs: Per solver, one positive cost per problem in a common order;
            ``None``, ``inf``, or ``NaN`` marks a failure.

    Returns:
        The profiles of all solvers.

    Raises:
        TypeError: If a cost is not a real number or ``None``.
        ValueError: If the table is empty or ragged or a cost is not positive.

    Equation:
        [EQ-NUM-PROFILE-001], [EQ-NUM-PROFILE-002].

    References:
        [REF-DOLAN-MORE-2002, Section 2].

    """
    ratios = performance_ratios(costs)
    solvers = tuple(ratios)
    return PerformanceProfile(
        solvers=solvers,
        ratios=tuple(ratios[s] for s in solvers),
        problems=len(ratios[solvers[0]]),
    )


@dataclass(frozen=True, slots=True)
class DataProfile:
    """Moré-Wild data profiles of several solvers on one problem set.

    Args:
        solvers: Solver names in input order.
        budgets: Per solver (same order), the cost of solving each problem in
            simplex gradients ``t_(p,s) / (n_p + 1)`` (``inf`` if unsolved).
        problems: Number of problems.

    References:
        [EQ-NUM-PROFILE-004], [REF-MORE-WILD-2009].

    """

    solvers: tuple[str, ...]
    budgets: tuple[tuple[float, ...], ...]
    problems: int

    def _sorted(self, solver: str) -> list[float]:
        """Return the ascending budgets of one solver.

        Args:
            solver: Solver name.

        Returns:
            Sorted budgets.

        Raises:
            KeyError: If the solver is unknown.

        References:
            [EQ-NUM-PROFILE-004].

        """
        if solver not in self.solvers:
            raise KeyError(f"unknown solver {solver!r}")
        return sorted(self.budgets[self.solvers.index(solver)])

    def value(self, solver: str, kappa: float) -> float:
        """Return ``d_s(kappa)``: the fraction solved within ``kappa`` gradients.

        Args:
            solver: Solver name.
            kappa: Budget in simplex gradients (``n_p + 1`` evaluations each).

        Returns:
            Profile value in ``[0, 1]``; nondecreasing in ``kappa``.

        Raises:
            KeyError: If the solver is unknown.

        Equation:
            [EQ-NUM-PROFILE-004].

        References:
            [REF-MORE-WILD-2009].

        """
        return _step_value(self._sorted(solver), kappa, self.problems)

    def kappa_at(self, solver: str, fraction: float) -> float:
        """Return the smallest budget with ``d_s(kappa) >= fraction``.

        Args:
            solver: Solver name.
            fraction: Target fraction of problems in ``(0, 1]``.

        Returns:
            Budget in simplex gradients, or ``inf`` if never reached.

        Raises:
            KeyError: If the solver is unknown.
            TypeError: If ``fraction`` is not real.
            ValueError: If ``fraction`` lies outside ``(0, 1]``.

        Equation:
            [EQ-NUM-PROFILE-004].

        References:
            [REF-MORE-WILD-2009].

        """
        target = _required_count(_require_fraction(fraction), self.problems)
        return self._sorted(solver)[target - 1]

    def breakpoints(self, solver: str) -> tuple[tuple[float, float], ...]:
        """Return the jumps ``(kappa, d_s(kappa))`` of the step profile.

        Args:
            solver: Solver name.

        Returns:
            One pair per distinct finite budget, ascending.

        Raises:
            KeyError: If the solver is unknown.

        Equation:
            [EQ-NUM-PROFILE-004].

        References:
            [REF-MORE-WILD-2009].

        """
        return _breakpoints(self._sorted(solver), self.problems)


def data_profile(
    evaluations: Mapping[str, Sequence[Cost]],
    dimensions: Sequence[int] | None = None,
) -> DataProfile:
    """Build Moré-Wild data profiles from evaluations-to-solve counts.

    Args:
        evaluations: Per solver, the number of function evaluations
            ``t_(p,s)`` it needed to solve each problem (common order);
            ``None``, ``inf``, or ``NaN`` marks a problem it did not solve.
        dimensions: Problem dimensions ``n_p``; each count is divided by
            ``n_p + 1``. ``None`` divides by 1 (budgets in raw evaluations).

    Returns:
        The data profiles of all solvers.

    Raises:
        TypeError: If an entry is not a real number or ``None``, or a dimension
            is not an integer.
        ValueError: If the table is empty or ragged, a count is not positive,
            or the dimensions do not match the problems or are negative.

    Equation:
        [EQ-NUM-PROFILE-004].

    References:
        [REF-MORE-WILD-2009].

    """
    solvers, rows, count = _cost_table(evaluations, name="evaluations")
    if dimensions is None:
        divisors = [1] * count
    else:
        if len(dimensions) != count:
            raise ValueError("dimensions needs one entry per problem")
        divisors = []
        for dimension in dimensions:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError("dimensions must be integers")
            if dimension < 0:
                raise ValueError("dimensions must be non-negative")
            divisors.append(dimension + 1)
    budgets = tuple(
        tuple(t / d for t, d in zip(row, divisors, strict=True)) for row in rows
    )
    return DataProfile(solvers=solvers, budgets=budgets, problems=count)


def first_converged_evaluation(
    values: Sequence[float], *, best: float, tolerance: float
) -> int | None:
    """Return the evaluations needed to pass the Moré-Wild convergence test.

    The test accepts the ``k``-th evaluation when
    ``f(x_0) - f(x_k) >= (1 - tau)(f(x_0) - f_L)``, with ``f(x_0)`` the first
    value and ``f_L`` the best value any solver found on the problem.

    Args:
        values: Objective values in evaluation order (``values[0] = f(x_0)``).
        best: Best value ``f_L`` known for the problem.
        tolerance: Accuracy level ``tau`` in ``(0, 1)``.

    Returns:
        The 1-based evaluation count of the first accepted point, or ``None``
        if no point passes.

    Raises:
        ValueError: If ``values`` is empty, contains a NaN, or ``tolerance``
            lies outside ``(0, 1)``.

    Equation:
        [EQ-NUM-PROFILE-004].

    References:
        [REF-MORE-WILD-2009].

    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must lie in (0, 1)")
    if any(math.isnan(v) for v in values) or math.isnan(best):
        raise ValueError("values and best must not be NaN")
    start = values[0]
    required = (1.0 - tolerance) * (start - best)
    for index, value in enumerate(values, start=1):
        if start - value >= required:
            return index
    return None
