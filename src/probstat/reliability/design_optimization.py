"""Reliability-based design optimisation over a design vector.

Sizing a structure or a spacecraft subsystem is the problem

    minimise cost(d)   subject to   beta_j(d) >= beta_target_j,   j = 1..m,

where ``d`` collects deterministic design variables and the means of random
inputs, and ``beta_j(d)`` is the first-order reliability index of the limit
state ``g_j`` at design ``d`` (failure is ``g_j <= 0``, the library-wide
convention of :class:`~probstat.reliability.ModelLimitState`). A *design*
enters a limit state through a factory ``d -> ModelLimitState``: the factory
builds the :class:`~probstat.modeling.ProbabilisticModel` for that design,
putting ``d`` into the means of the random inputs, into ``model.fixed``, or
both.

The module offers three tools.

- :func:`minimize_constrained`: a general inequality-constrained local
  minimiser (augmented Lagrangian with the library's BFGS as inner solver).
  The library had no constrained optimiser; this one is small, transparent
  and tested on problems with known solutions.
- :func:`rbdo_sora`: the decoupled sequential optimisation and reliability
  assessment (SORA) of Du and Chen (2004). Each cycle solves a deterministic
  problem whose constraints are shifted by the distance between the mean and
  the inverse-FORM most-probable target point (performance-measure approach,
  :func:`~probstat.reliability.inverse_form`), then repeats the inverse
  reliability analysis at the new design.
- :func:`rbdo_double_loop`: the nested reliability-index approach, one FORM run
  per constraint evaluation. It is slow and serves as the reference.

:func:`verify_design_reliability` re-checks a final design with an
independent estimator (second-order reliability or importance sampling at the
design point) and reports the difference from FORM.

Assumptions: continuous inputs (FORM, inverse FORM); the limit states are
smooth enough in the design for finite-difference gradients; the optimiser
is local. SORA shifts constraints in *physical* space by ``mu(d) - x*``, which
is the standard formulation; it is exact when the limit state is linear in
the random inputs and the shift does not depend on the design (normal inputs
with design-independent standard deviations), and an approximation that is
repaired by iteration otherwise. Both reliability methods are first order:
the reliability index is only as accurate as FORM (see the verification
helper).

Relation to neighbouring tools: :func:`~probstat.reliability.design_for_target_beta`
solves one design parameter against one limit state by Brent's method; the
functions here handle a design vector, several limit states and a cost.

Guide: docs/foundations/reliability-analysis.md

References:
    [REF-DU-CHEN-2004], [REF-TU-CHOI-PARK-1999],
    [REF-DERKIUREGHIAN-ZHANG-LI-1994], [REF-AOUES-CHATEAUNEUF-2010],
    [REF-NOCEDAL-WRIGHT-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector, norm
from probstat.modeling import standard_normal_cdf, standard_normal_quantile
from probstat.optimization import minimize_bfgs
from probstat.optimization.bounds import Bounds

from .model_based import ModelLimitState, form_hlrf, inverse_form, sorm
from .simulation import design_point_importance_sampling

LimitStateFactory = Callable[[Vector], ModelLimitState]
VERIFICATION_METHODS = ("sorm", "importance")


@dataclass(frozen=True, slots=True)
class ConstrainedResult:
    """Outcome of :func:`minimize_constrained`.

    Args:
        x: Final iterate.
        value: Objective value at ``x``.
        constraint_values: ``c_j(x)``; the constraint ``c_j <= 0`` is met.
        multipliers: Final Lagrange multiplier estimates (non-negative).
        max_violation: ``max(0, max_j c_j(x))``.
        outer_iterations: Augmented-Lagrangian updates.
        evaluations: Objective evaluations (including finite differences).
        constraint_evaluations: Calls of the constraint function.
        converged: Whether the violation and the step tolerances were met.
        message: Termination reason.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 17.4].

    """

    x: Vector
    value: float
    constraint_values: Vector
    multipliers: Vector
    max_violation: float
    outer_iterations: int
    evaluations: int
    constraint_evaluations: int
    converged: bool
    message: str


def _positive(value: object, name: str) -> float:
    """Return a finite positive float.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    checked = require_finite_real(value, name=name)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _positive_int(value: object, name: str) -> int:
    """Return a positive integer.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def minimize_constrained(
    objective: Callable[[Vector], float],
    constraints: Callable[[Vector], Sequence[float]],
    x0: Sequence[float],
    *,
    bounds: Bounds | None = None,
    tolerance: float = 1e-7,
    max_outer_iterations: int = 50,
    max_inner_iterations: int = 200,
    initial_penalty: float = 10.0,
    penalty_growth: float = 10.0,
) -> ConstrainedResult:
    """Minimise ``f(x)`` subject to ``c_j(x) <= 0`` by an augmented Lagrangian.

    The Rockafellar augmented Lagrangian for inequality constraints is

        ``L(x; lam, mu) = f(x) + (1 / (2 mu)) sum_j [ max(0, lam_j + mu c_j(x))^2
        - lam_j^2 ]``,

    minimised over ``x`` (and the optional box) by BFGS with finite-difference
    gradients; the multipliers then move to ``max(0, lam_j + mu c_j)``. The
    penalty ``mu`` is multiplied by ``penalty_growth`` whenever the largest
    violation fails to drop to a quarter of its previous value. The
    iteration stops when ``max_j max(0, c_j) <= tolerance`` and the iterate
    moved by at most ``sqrt(tolerance) (1 + ||x||)``.

    The method finds a local solution of a smooth problem. A constraint that is
    only piecewise smooth (such as an order statistic of sampled values) is
    handled when the kinks are isolated; no global guarantee is given.

    Args:
        objective: Objective ``f``.
        constraints: Function returning the vector ``(c_1(x), ..., c_m(x))``;
            ``c_j <= 0`` is feasible. The length must not change between calls.
        x0: Starting point (inside ``bounds`` when given).
        bounds: Optional ``(low, high)`` per coordinate, handled by the inner
            solver.
        tolerance: Feasibility tolerance.
        max_outer_iterations: Multiplier updates allowed.
        max_inner_iterations: BFGS iteration limit per inner solve.
        initial_penalty: Starting penalty ``mu``.
        penalty_growth: Factor ``> 1`` applied to ``mu`` on slow progress.

    Returns:
        Solution, multipliers, violation, counts, and a convergence flag.

    Raises:
        TypeError: If a callable or an integer argument has the wrong type.
        ValueError: If a numeric argument is out of range, ``constraints``
            returns a non-finite value or changes length, or ``x0`` is empty.

    Equation:
        [EQ-REL-AUGLAG-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 17.4].

    """
    if not callable(objective) or not callable(constraints):
        raise TypeError("objective and constraints must be callable")
    start = as_vector(x0, name="x0")
    if not start:
        raise ValueError("x0 must not be empty")
    tol = _positive(tolerance, "tolerance")
    outer_limit = _positive_int(max_outer_iterations, "max_outer_iterations")
    inner_limit = _positive_int(max_inner_iterations, "max_inner_iterations")
    penalty = _positive(initial_penalty, "initial_penalty")
    growth = _positive(penalty_growth, "penalty_growth")
    if growth <= 1.0:
        raise ValueError("penalty_growth must exceed 1")
    counts = {"objective": 0, "constraints": 0}
    width = [0]

    def constraint_vector(x: Vector) -> Vector:
        """Evaluate and validate the constraint vector.

        References:
            [EQ-REL-AUGLAG-001]. See :func:`minimize_constrained`.

        """
        counts["constraints"] += 1
        values = tuple(float(v) for v in constraints(x))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("constraints returned a non-finite value")
        if width[0] and len(values) != width[0]:
            raise ValueError("constraints changed length between calls")
        width[0] = len(values)
        return values

    def objective_value(x: Vector) -> float:
        """Evaluate the objective with a call count.

        References:
            [EQ-REL-AUGLAG-001]. See :func:`minimize_constrained`.

        """
        counts["objective"] += 1
        return float(objective(x))

    x = start
    first = constraint_vector(x)
    if not first:
        raise ValueError("constraints must return at least one value")
    multipliers = [0.0] * len(first)
    previous_violation = math.inf
    converged = False
    message = "outer iteration limit reached"
    outer = 0
    while outer < outer_limit:
        mu = penalty
        lam = tuple(multipliers)

        def lagrangian(point: Vector, mu: float = mu, lam: Vector = lam) -> float:
            """Return the augmented Lagrangian at ``point``.

            References:
                [EQ-REL-AUGLAG-001]. See :func:`minimize_constrained`.

            """
            value = objective_value(point)
            shifted = constraint_vector(point)
            total = 0.0
            for lj, cj in zip(lam, shifted, strict=True):
                total += max(0.0, lj + mu * cj) ** 2 - lj * lj
            return value + total / (2.0 * mu)

        inner = minimize_bfgs(
            lagrangian,
            x,
            bounds=bounds,
            max_iterations=inner_limit,
            gradient_tolerance=1e-9,
        )
        new_x = inner.x
        values = constraint_vector(new_x)
        multipliers = [
            max(0.0, lj + penalty * cj)
            for lj, cj in zip(multipliers, values, strict=True)
        ]
        violation = max(0.0, max(values))
        step = norm(tuple(a - b for a, b in zip(new_x, x, strict=True)))
        x = new_x
        outer += 1
        if violation <= tol and step <= math.sqrt(tol) * (1.0 + norm(x)):
            converged = True
            message = "converged"
            break
        if violation > 0.25 * previous_violation:
            penalty *= growth
        previous_violation = violation
    final = constraint_vector(x)
    return ConstrainedResult(
        x=x,
        value=float(objective(x)),
        constraint_values=final,
        multipliers=tuple(multipliers),
        max_violation=max(0.0, max(final)),
        outer_iterations=outer,
        evaluations=counts["objective"],
        constraint_evaluations=counts["constraints"],
        converged=converged,
        message=message,
    )


@dataclass(frozen=True, slots=True)
class RBDOIteration:
    """One cycle of a reliability-based design optimisation.

    Args:
        cycle: One-based cycle number.
        design: Design after the deterministic optimisation of the cycle.
        cost: Cost at ``design``.
        mptp_margins: ``g_j`` at the inverse-FORM most-probable target point
            of each limit state at ``design``; negative means the target
            index is not met (SORA only; the double loop stores ``beta_j``
            minus its target).
        step: Euclidean distance between the new and the previous design.

    References:
        [REF-DU-CHEN-2004].

    """

    cycle: int
    design: Vector
    cost: float
    mptp_margins: Vector
    step: float


@dataclass(frozen=True, slots=True)
class RBDOResult:
    """Result of a reliability-based design optimisation.

    Args:
        method: ``"sora"`` or ``"double-loop"``.
        design: Final design vector.
        cost: Cost at ``design``.
        betas: FORM reliability index of every limit state at ``design``.
        target_betas: Requested indices.
        history: One record per cycle (SORA) or outer update (double loop).
        evaluations: Model evaluations charged to limit states.
        cost_evaluations: Cost-function calls.
        converged: Whether the stopping rules were met and every target holds
            within ``feasibility_tolerance``.
        message: Reason for the convergence flag.

    References:
        [REF-DU-CHEN-2004], [REF-AOUES-CHATEAUNEUF-2010].

    """

    method: str
    design: Vector
    cost: float
    betas: Vector
    target_betas: Vector
    history: tuple[RBDOIteration, ...]
    evaluations: int
    cost_evaluations: int
    converged: bool
    message: str


def _targets(value: float | Sequence[float], count: int) -> Vector:
    """Broadcast one target index or validate a list of ``count`` targets.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    if isinstance(value, bool):
        raise TypeError("target_betas must be numbers")
    if isinstance(value, (int, float)):
        return tuple(_positive(value, "target_betas") for _ in range(count))
    items = tuple(_positive(v, "target_betas") for v in value)
    if len(items) != count:
        raise ValueError("target_betas must match the number of limit states")
    return items


def _factories(value: Sequence[LimitStateFactory]) -> tuple[LimitStateFactory, ...]:
    """Validate the sequence of limit-state factories.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """
    items = tuple(value)
    if not items:
        raise ValueError("at least one limit state is required")
    for factory in items:
        if not callable(factory):
            raise TypeError("each limit state must be a callable design -> limit state")
    return items


class _Ledger:
    """Counts model evaluations and cost calls across fresh limit states.

    References:
        [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

    """

    def __init__(self, cost: Callable[[Vector], float]) -> None:
        """Store the cost function and zero the counters.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        self._cost = cost
        self.model_evaluations = 0
        self.cost_evaluations = 0

    def cost(self, design: Vector) -> float:
        """Evaluate the cost and count the call.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        self.cost_evaluations += 1
        return float(self._cost(design))

    def run(self, state: ModelLimitState, action: Callable[[], float]) -> float:
        """Run ``action`` and charge the model evaluations it caused.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        before = state.model.evaluations
        try:
            return action()
        finally:
            self.model_evaluations += state.model.evaluations - before


def _final_betas(
    design: Vector, factories: Sequence[LimitStateFactory], ledger: _Ledger
) -> Vector:
    """Return the FORM index of each limit state at ``design``.

    Raises:
        ValueError: If FORM does not converge for a limit state.

    References:
        [REF-HASOFER-LIND-1974].

    """
    betas: list[float] = []
    for factory in factories:
        state = factory(design)

        def beta_of(state: ModelLimitState = state) -> float:
            """Return the FORM index (raises when FORM fails).

            References:
                [REF-HASOFER-LIND-1974].

            """
            result = form_hlrf(state)
            if not result.converged:
                raise ValueError(f"FORM did not converge for {state.label!r}")
            return result.beta

        betas.append(ledger.run(state, beta_of))
    return tuple(betas)


def rbdo_sora(
    cost: Callable[[Vector], float],
    limit_states: Sequence[LimitStateFactory],
    target_betas: float | Sequence[float],
    initial_design: Sequence[float],
    *,
    bounds: Bounds | None = None,
    max_cycles: int = 20,
    design_tolerance: float = 1e-5,
    feasibility_tolerance: float = 1e-5,
    inner_tolerance: float = 1e-8,
) -> RBDOResult:
    """Decoupled RBDO by sequential optimisation and reliability assessment.

    Cycle ``k`` solves ``min cost(d)`` subject to the shifted deterministic
    constraints ``g_j(mu_j(d) - s_j^k; d) >= 0``, where ``mu_j(d)`` is the
    vector of input means of limit state ``j`` at design ``d`` and ``s_j^k``
    is the shift vector. The first cycle has ``s = 0`` (a deterministic
    design). After each cycle :func:`~probstat.reliability.inverse_form`
    locates the most-probable target point ``x*_j`` of every limit state at
    the new design (the point of minimal ``g`` on the sphere
    ``||u|| = beta_target_j``) and the shift becomes ``s_j = mu_j(d) - x*_j``.
    The loop ends when the design moves by at most
    ``design_tolerance (1 + ||d||)`` and every margin ``g_j(x*_j)`` is at least
    ``-feasibility_tolerance``; ``g_j(x*_j) >= 0`` is the performance-measure
    statement ``beta_j >= beta_target_j``.

    The shift equals the exact FORM distance when the limit state is linear in
    normal inputs with design-independent standard deviations, so a linear
    problem converges after the first reassessment.

    Args:
        cost: Cost function of the design vector.
        limit_states: One factory ``design -> ModelLimitState`` per
            constraint. The factory must return a model whose input names and
            order do not depend on the design.
        target_betas: One positive target index or one per limit state.
        initial_design: Starting design.
        bounds: Optional design bounds ``(low, high)`` per coordinate.
        max_cycles: Maximum number of cycles.
        design_tolerance: Relative design change that counts as stationary.
        feasibility_tolerance: Allowed negative margin at the target point.
        inner_tolerance: Feasibility tolerance of the deterministic solves.

    Returns:
        Design, cost, FORM indices at the design, cycle history, evaluation
        counts, and a convergence flag with its reason.

    Raises:
        TypeError: If a callable or an integer argument has the wrong type.
        ValueError: If a numeric argument is out of range, the limit-state
            list is empty, or FORM fails at the final design.

    Equation:
        [EQ-REL-SORA-001], [EQ-REL-INV-001].

    References:
        [REF-DU-CHEN-2004], [REF-TU-CHOI-PARK-1999],
        [REF-DERKIUREGHIAN-ZHANG-LI-1994].

    """
    if not callable(cost):
        raise TypeError("cost must be callable")
    factories = _factories(limit_states)
    goals = _targets(target_betas, len(factories))
    design = as_vector(initial_design, name="initial_design")
    cycles = _positive_int(max_cycles, "max_cycles")
    design_tol = _positive(design_tolerance, "design_tolerance")
    feas_tol = _positive(feasibility_tolerance, "feasibility_tolerance")
    ledger = _Ledger(cost)
    shifts: list[Vector | None] = [None] * len(factories)
    history: list[RBDOIteration] = []
    converged = False
    message = "cycle limit reached"

    def shifted_constraints(d: Vector) -> Vector:
        """Return ``-g_j`` at the shifted mean point for every limit state.

        References:
            [EQ-REL-SORA-001].

        """
        values: list[float] = []
        for factory, shift in zip(factories, shifts, strict=True):
            state = factory(d)
            means = state.model.inputs.means()
            point = (
                means
                if shift is None
                else tuple(m - s for m, s in zip(means, shift, strict=True))
            )

            def margin(s: ModelLimitState = state, p: Vector = point) -> float:
                """Return ``g`` at the shifted mean point.

                References:
                    [EQ-REL-SORA-001].

                """
                return s.physical(p)

            values.append(-ledger.run(state, margin))
        return tuple(values)

    for cycle in range(1, cycles + 1):
        sub = minimize_constrained(
            ledger.cost,
            shifted_constraints,
            design,
            bounds=bounds,
            tolerance=_positive(inner_tolerance, "inner_tolerance"),
        )
        new_design = sub.x
        margins: list[float] = []
        reassessment_ok = True
        for index, factory in enumerate(factories):
            state = factory(new_design)
            before = state.model.evaluations
            inverse = inverse_form(state, goals[index])
            margin = state.standard(inverse.design_point_u)
            ledger.model_evaluations += state.model.evaluations - before
            reassessment_ok = reassessment_ok and inverse.converged
            means = state.model.inputs.means()
            shifts[index] = tuple(
                m - x for m, x in zip(means, inverse.design_point_x, strict=True)
            )
            margins.append(margin)
        step = norm(tuple(a - b for a, b in zip(new_design, design, strict=True)))
        history.append(
            RBDOIteration(
                cycle=cycle,
                design=new_design,
                cost=float(cost(new_design)),
                mptp_margins=tuple(margins),
                step=step,
            )
        )
        previous_norm = norm(design)
        design = new_design
        if not sub.converged:
            message = "deterministic subproblem did not converge"
            break
        if not reassessment_ok:
            message = "inverse reliability analysis did not converge"
            break
        stationary = step <= design_tol * (1.0 + previous_norm)
        if stationary and min(margins) >= -feas_tol:
            converged = True
            message = "converged"
            break
        if cycle == cycles:
            message = "cycle limit reached"
    betas = _final_betas(design, factories, ledger)
    if converged and any(b < g - 1e-3 for b, g in zip(betas, goals, strict=True)):
        converged = False
        message = "FORM index at the final design is below its target"
    return RBDOResult(
        method="sora",
        design=design,
        cost=float(cost(design)),
        betas=betas,
        target_betas=goals,
        history=tuple(history),
        evaluations=ledger.model_evaluations,
        cost_evaluations=ledger.cost_evaluations,
        converged=converged,
        message=message,
    )


def rbdo_double_loop(
    cost: Callable[[Vector], float],
    limit_states: Sequence[LimitStateFactory],
    target_betas: float | Sequence[float],
    initial_design: Sequence[float],
    *,
    bounds: Bounds | None = None,
    feasibility_tolerance: float = 1e-5,
    inner_tolerance: float = 1e-8,
) -> RBDOResult:
    """Reliability-index approach: optimise with one FORM run per constraint call.

    The constraints are ``beta_target_j - beta_j(d) <= 0`` with ``beta_j``
    from :func:`~probstat.reliability.form_hlrf`, solved by
    :func:`minimize_constrained`. A FORM run that does not converge at a trial
    design raises inside the inner solver, which then shortens its step. The
    cost is a FORM analysis per constraint per trial design (and per
    finite-difference point), which makes this the slow reference for
    :func:`rbdo_sora`.

    Args:
        cost: Cost function of the design vector.
        limit_states: One factory ``design -> ModelLimitState`` per constraint.
        target_betas: One positive target index or one per limit state.
        initial_design: Starting design (FORM must converge there).
        bounds: Optional design bounds.
        feasibility_tolerance: Allowed shortfall of ``beta_j`` below its target
            in the convergence flag.
        inner_tolerance: Feasibility tolerance of the constrained solver.

    Returns:
        The same record as :func:`rbdo_sora`; ``history`` has one entry whose
        ``mptp_margins`` are ``beta_j - beta_target_j``.

    Raises:
        TypeError: If a callable has the wrong type.
        ValueError: If a numeric argument is out of range or FORM fails at the
            starting design.

    Equation:
        [EQ-REL-RBDO-002].

    References:
        [REF-TU-CHOI-PARK-1999], [REF-AOUES-CHATEAUNEUF-2010],
        [REF-HASOFER-LIND-1974].

    """
    if not callable(cost):
        raise TypeError("cost must be callable")
    factories = _factories(limit_states)
    goals = _targets(target_betas, len(factories))
    start = as_vector(initial_design, name="initial_design")
    feas_tol = _positive(feasibility_tolerance, "feasibility_tolerance")
    ledger = _Ledger(cost)

    def index_constraints(d: Vector) -> Vector:
        """Return ``beta_target_j - beta_j(d)`` for every limit state.

        References:
            [EQ-REL-RBDO-002].

        """
        betas = _final_betas(d, factories, ledger)
        return tuple(g - b for g, b in zip(goals, betas, strict=True))

    sub = minimize_constrained(
        ledger.cost,
        index_constraints,
        start,
        bounds=bounds,
        tolerance=_positive(inner_tolerance, "inner_tolerance"),
    )
    betas = _final_betas(sub.x, factories, ledger)
    margins = tuple(b - g for b, g in zip(betas, goals, strict=True))
    feasible = min(margins) >= -feas_tol
    converged = sub.converged and feasible
    if converged:
        message = "converged"
    elif not sub.converged:
        message = f"constrained solver: {sub.message}"
    else:
        message = "a target reliability index is not met"
    step = norm(tuple(a - b for a, b in zip(sub.x, start, strict=True)))
    return RBDOResult(
        method="double-loop",
        design=sub.x,
        cost=float(cost(sub.x)),
        betas=betas,
        target_betas=goals,
        history=(
            RBDOIteration(
                cycle=1,
                design=sub.x,
                cost=sub.value,
                mptp_margins=margins,
                step=step,
            ),
        ),
        evaluations=ledger.model_evaluations,
        cost_evaluations=ledger.cost_evaluations,
        converged=converged,
        message=message,
    )


@dataclass(frozen=True, slots=True)
class DesignVerification:
    """FORM index of one limit state compared with an independent estimate.

    Args:
        limit_state: Label of the limit state.
        target_beta: Requested index.
        form_beta: FORM index at the design.
        form_probability: ``Phi(-form_beta)``.
        check_method: ``"sorm"`` (Hohenbichler-Rackwitz) or ``"importance"``.
        check_probability: Failure probability from the independent method.
        check_beta: Generalised index ``-Phi^-1(check_probability)`` (``inf``
            when no importance-sampling weight fell in the failure domain).
        beta_difference: ``check_beta - form_beta``.
        standard_error: Standard error of ``check_probability`` for
            importance sampling (``0`` for SORM).
        meets_target: Whether ``check_beta >= target_beta``.

    References:
        [REF-DERKIUREGHIAN-2005], [REF-MELCHERS-BECK-2018, Chapter 6].

    """

    limit_state: str
    target_beta: float
    form_beta: float
    form_probability: float
    check_method: str
    check_probability: float
    check_beta: float
    beta_difference: float
    standard_error: float
    meets_target: bool


def verify_design_reliability(
    design: Sequence[float],
    limit_states: Sequence[LimitStateFactory],
    target_betas: float | Sequence[float],
    *,
    method: str = "sorm",
    samples: int = 2000,
    seed: int | RandomStream | None = 0,
) -> tuple[DesignVerification, ...]:
    """Re-check a design with SORM or importance sampling and compare with FORM.

    Args:
        design: Design vector to verify.
        limit_states: One factory ``design -> ModelLimitState`` per constraint.
        target_betas: One positive target index or one per limit state.
        method: ``"sorm"`` (second-order Hohenbichler-Rackwitz probability at
            the FORM design point) or ``"importance"`` (design-point
            importance sampling with ``samples`` draws).
        samples: Importance-sampling draws.
        seed: Seed or stream for importance sampling.

    Returns:
        One :class:`DesignVerification` per limit state.

    Raises:
        TypeError: If a callable has the wrong type.
        ValueError: If ``method`` is unknown, a numeric argument is out of
            range, or FORM does not converge at the design.

    Equation:
        [EQ-REL-BETA-001].

    References:
        [REF-DERKIUREGHIAN-2005], [REF-MELCHERS-BECK-2018, Chapter 6].

    """
    if method not in VERIFICATION_METHODS:
        raise ValueError(f"method must be one of {VERIFICATION_METHODS}")
    point = as_vector(design, name="design")
    factories = _factories(limit_states)
    goals = _targets(target_betas, len(factories))
    rows: list[DesignVerification] = []
    for index, factory in enumerate(factories):
        state = factory(point)
        form = form_hlrf(state)
        if not form.converged:
            raise ValueError(f"FORM did not converge for {state.label!r}")
        if method == "sorm":
            second = sorm(state, form_result=form)
            probability = second.hohenbichler_rackwitz_probability
            error = 0.0
        else:
            stream = resolve_stream(seed).child("verify", index)
            sampled = design_point_importance_sampling(
                state, samples, form_result=form, seed=stream
            )
            probability = sampled.estimate.estimate
            error = sampled.estimate.standard_error
        if probability <= 0.0:
            check_beta = math.inf
        elif probability >= 1.0:
            check_beta = -math.inf
        else:
            check_beta = -standard_normal_quantile(probability)
        rows.append(
            DesignVerification(
                limit_state=state.label,
                target_beta=goals[index],
                form_beta=form.beta,
                form_probability=standard_normal_cdf(-form.beta),
                check_method=method,
                check_probability=probability,
                check_beta=check_beta,
                beta_difference=check_beta - form.beta,
                standard_error=error,
                meets_target=check_beta >= goals[index],
            )
        )
    return tuple(rows)
