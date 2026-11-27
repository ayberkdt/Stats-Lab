"""Correlation (calibration) of a lumped spacecraft thermal network to test data.

Experimental (added in 0.28).

Setting: a lumped-parameter thermal network of ``n`` free nodes (panels, an
electronics box, a bracket, a radiator, a battery, ...) with heat capacities
``C_i`` (J/K), conductive couplings ``G_ij`` (W/K), radiative couplings
``R_ij = epsilon sigma A F`` (W/K^4), internal dissipation ``Q_i(t)`` (W) and
fixed-temperature boundary nodes (a chamber shroud, space sink) [EQ-INV-THERM-001].
Temperatures are absolute (K). The relations are the textbook ones for
conduction between lumped bodies and for the radiation exchange of gray
surfaces [REF-BERGMAN-2011]; the network is an idealization, not a model of
any real spacecraft, and every bundled case is synthetic.

What is implemented:

* Forward model: the steady state by damped Newton iteration on the nonlinear
  energy balance with the exact Jacobian ``dF/dT`` and the exact parameter
  derivative ``dF/dtheta``, hence the exact steady sensitivities
  ``dT/dtheta = -(dF/dT)^-1 dF/dtheta`` [EQ-INV-THERM-002]; the transient by the
  backward Euler scheme with Newton iteration per step and the exact discrete
  forward sensitivity recursion [EQ-INV-THERM-003]; energy-conservation
  residuals (steady and per transient step) [EQ-INV-THERM-004].
* Correlation: ``theta`` scales selected conductances or the radiative
  couplings of a group of links; measured sensor temperatures at several
  steady power levels and/or a transient carry independent Gaussian error with
  a known standard deviation; the prior of ``log theta`` is Gaussian
  (lognormal ``theta``). An optional model-form discrepancy is a persistent
  additive offset ``delta_s ~ N(0, tau^2)`` per sensor, shared by every record
  of that sensor, with an unknown ``tau`` [EQ-INV-THERM-005]. The MAP is found
  by Levenberg-Marquardt with the analytic Jacobian, the Laplace posterior and
  evidence from the Gauss-Newton Hessian [EQ-INV-THERM-006], and the full
  posterior (including ``tau``) by transitional MCMC
  [REF-CHING-CHEN-2007].
* Identifiability: eigenvalues and weakest direction of the likelihood
  Hessian, posterior standard deviations and correlations, shrinkage
  relative to the prior, and the profile likelihood of one parameter
  [EQ-INV-THERM-007]. Two conductances in series with no sensor between them
  enter the steady state only through their sum of resistances, so a steady
  test cannot separate them; a transient test can, through the heat capacity
  between them.
* Prediction: the posterior predictive of an untested (flight) steady case,
  with a different power and boundary temperature, as sample bands at chosen
  nodes, with the discrepancy drawn at sensor nodes when it is modelled
  [EQ-INV-THERM-008]. This is the purpose of correlation: a prediction with
  an honest band, not the fit of the test data.

Common pitfalls:
    - Correlation to steady data at one power level cannot separate parameters
      that act through one combination (series conductances, a conduction and
      a radiation path in parallel at one temperature level). The Laplace
      standard deviations then come mostly from the prior; look at the
      likelihood eigenvalues and the correlation before quoting a parameter.
    - A calibrated parameter absorbs unmodelled physics. If the model lacks a
      coupling, the fitted conductances are biased, the bands are too narrow
      and the flight case is extrapolated with that bias. The discrepancy
      term is a constant offset per sensor: it is a remedy only when the
      unmodelled effect is roughly persistent between the tests and the flight
      case, and it is not a physical model.
    - The Laplace posterior with the discrepancy plugs in the evidence-
      maximizing ``tau`` on a grid and ignores its uncertainty; the TMCMC
      posterior samples ``tau``.
    - The initial state of a transient with ``initial=None`` is the steady
      state under the power at time zero, so it depends on ``theta`` (and the
      sensitivity includes that dependence). A given initial state is
      treated as known exactly.
    - Newton may fail far from the solution of a radiative network (negative
      temperatures are never accepted); the failure is reported
      (:class:`ThermalSolveError`, ``stop_reason``), never hidden.

References:
    [REF-BERGMAN-2011], [REF-BECK-BLACKWELL-STCLAIR-1985],
    [REF-KENNEDY-OHAGAN-2001], [REF-CHING-CHEN-2007].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference.transitional_mcmc import TMCMCResult, transitional_mcmc
from probstat.inverse.profile_likelihood import (
    ProfileLikelihoodResult,
    profile_likelihood,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    inverse_spd,
    log_det_spd,
    lu_solve,
    symmetric_eigen,
)
from probstat.optimization import least_squares

STEFAN_BOLTZMANN = 5.670374419e-8
"""Stefan-Boltzmann constant in W m^-2 K^-4 (CODATA 2018) [REF-BERGMAN-2011]."""

_LOG_TWO_PI = math.log(2.0 * math.pi)


class ThermalSolveError(ValueError):
    """Raised when a Newton solve of the thermal network does not converge.

    References:
        [EQ-INV-THERM-002].

    """


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _vector(values: object, size: int, name: str, *, positive: bool = False) -> Vector:
    """Return a finite real vector of a given length.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the length is wrong, an entry is not finite, or
            ``positive`` and an entry is not strictly positive.

    References:
        [EQ-INV-THERM-001]. No new equation is introduced.

    """
    vec = as_vector(values, name=name)  # type: ignore[arg-type]
    if len(vec) != size:
        raise ValueError(f"{name} has length {len(vec)}, expected {size}")
    if positive and any(v <= 0.0 for v in vec):
        raise ValueError(f"{name} must be strictly positive")
    return vec


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If it is not finite and positive.

    References:
        [EQ-INV-THERM-001]. No new equation is introduced.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _count(value: object, name: str) -> int:
    """Return a positive integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is not positive.

    References:
        [EQ-INV-THERM-001]. No new equation is introduced.

    """
    number = require_non_negative_integer(value, name=name)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThermalLink:
    """One coupling between two nodes.

    Args:
        first: Index of the first node (free nodes come first, then boundary
            nodes).
        second: Index of the second node (different from ``first``).
        value: ``G`` in W/K for a conductive link or ``R = epsilon sigma A F``
            in W/K^4 for a radiative link; strictly positive.

    References:
        [EQ-INV-THERM-001], [REF-BERGMAN-2011].

    """

    first: int
    second: int
    value: float


@dataclass(frozen=True, slots=True)
class ThermalNetwork:
    """Lumped thermal network with free and fixed-temperature nodes.

    The flow into node ``a`` from node ``b`` along a link is
    ``G (T_b - T_a) + R (T_b^4 - T_a^4)`` [EQ-INV-THERM-001]. The first
    ``len(capacities)`` nodes are free (they have a heat capacity and a
    temperature to solve for); the remaining nodes are boundary nodes whose
    temperature is given per test.

    Args:
        names: Node names, free nodes first.
        capacities: Heat capacities ``C_i`` (J/K) of the free nodes, strictly
            positive.
        conductive: Conductive links (W/K).
        radiative: Radiative links (W/K^4).

    Raises:
        TypeError: If an entry has the wrong type.
        ValueError: If an index is out of range, a link joins a node to itself
            or two boundary nodes, a value is not positive, or names repeat.

    References:
        [EQ-INV-THERM-001], [REF-BERGMAN-2011].

    """

    names: tuple[str, ...]
    capacities: Vector
    conductive: tuple[ThermalLink, ...]
    radiative: tuple[ThermalLink, ...]

    def __post_init__(self) -> None:
        """Validate the network.

        Raises:
            TypeError: If an entry has the wrong type.
            ValueError: If the network is inconsistent.

        References:
            [EQ-INV-THERM-001].

        """
        n = len(self.capacities)
        if n < 1:
            raise ValueError("the network needs at least one free node")
        if len(self.names) < n or len(set(self.names)) != len(self.names):
            raise ValueError("names must be distinct and cover every node")
        _vector(self.capacities, n, "capacities", positive=True)
        total = len(self.names)
        for label, links in (
            ("conductive", self.conductive),
            ("radiative", self.radiative),
        ):
            for link in links:
                if not isinstance(link, ThermalLink):
                    raise TypeError(f"{label} entries must be ThermalLink")
                for index in (link.first, link.second):
                    if isinstance(index, bool) or not isinstance(index, int):
                        raise TypeError("link indices must be integers")
                    if not 0 <= index < total:
                        raise ValueError("link index out of range")
                if link.first == link.second:
                    raise ValueError("a link must join two different nodes")
                if link.first >= n and link.second >= n:
                    raise ValueError("a link may not join two boundary nodes")
                _positive(link.value, f"{label} link value")

    @property
    def n_free(self) -> int:
        """Return the number of free nodes.

        References:
            [EQ-INV-THERM-001].

        """
        return len(self.capacities)

    @property
    def n_boundary(self) -> int:
        """Return the number of boundary nodes.

        References:
            [EQ-INV-THERM-001].

        """
        return len(self.names) - len(self.capacities)

    def with_link(
        self,
        kind: Literal["conductive", "radiative"],
        first: int,
        second: int,
        value: float,
    ) -> ThermalNetwork:
        """Return a copy with one more link (e.g. an unmodelled coupling).

        Args:
            kind: ``"conductive"`` or ``"radiative"``.
            first: First node index.
            second: Second node index.
            value: Link value (W/K or W/K^4).

        Returns:
            The extended network.

        Raises:
            ValueError: For an unknown ``kind`` or an invalid link.

        References:
            [EQ-INV-THERM-001].

        """
        link = ThermalLink(first, second, value)
        if kind == "conductive":
            return ThermalNetwork(
                self.names, self.capacities, (*self.conductive, link), self.radiative
            )
        if kind == "radiative":
            return ThermalNetwork(
                self.names, self.capacities, self.conductive, (*self.radiative, link)
            )
        raise ValueError("kind must be 'conductive' or 'radiative'")

    def _all_temperatures(
        self, free: Sequence[float], boundary: Sequence[float]
    ) -> list[float]:
        """Return the free temperatures followed by the boundary temperatures.

        References:
            [EQ-INV-THERM-001]. No new equation is introduced.

        """
        return [*free, *boundary]

    def node_inflows(self, free: Sequence[float], boundary: Sequence[float]) -> Vector:
        """Return the net link heat flow into every node (W).

        Entry ``a`` is ``sum_links flow into a`` and the entries add up to
        zero because each link gives ``+q`` to one node and ``-q`` to the
        other [EQ-INV-THERM-004].

        Args:
            free: Temperatures of the free nodes (K).
            boundary: Temperatures of the boundary nodes (K).

        Returns:
            Inflows of all nodes, free first.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If a length is wrong or an entry is not finite.

        References:
            [EQ-INV-THERM-001], [EQ-INV-THERM-004].

        """
        t = self._all_temperatures(
            _vector(free, self.n_free, "free"),
            _vector(boundary, self.n_boundary, "boundary"),
        )
        out = [0.0] * len(t)
        for link in self.conductive:
            q = link.value * (t[link.second] - t[link.first])
            out[link.first] += q
            out[link.second] -= q
        for link in self.radiative:
            q = link.value * (t[link.second] ** 4 - t[link.first] ** 4)
            out[link.first] += q
            out[link.second] -= q
        return tuple(out)

    def net_heat_flow(
        self, free: Sequence[float], boundary: Sequence[float], power: Sequence[float]
    ) -> Vector:
        """Return ``F(T) = Q + inflow`` of the free nodes (W).

        ``C dT/dt = F(T, t)`` is the transient model and ``F = 0`` the steady
        state [EQ-INV-THERM-001].

        Args:
            free: Free temperatures (K).
            boundary: Boundary temperatures (K).
            power: Internal dissipation ``Q_i`` of the free nodes (W).

        Returns:
            ``F`` of length ``n_free``.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If a length is wrong or an entry is not finite.

        References:
            [EQ-INV-THERM-001].

        """
        q = _vector(power, self.n_free, "power")
        inflow = self.node_inflows(free, boundary)
        return tuple(q[i] + inflow[i] for i in range(self.n_free))

    def boundary_heat_flow(
        self, free: Sequence[float], boundary: Sequence[float]
    ) -> Vector:
        """Return the heat flow into each boundary node (W).

        Args:
            free: Free temperatures (K).
            boundary: Boundary temperatures (K).

        Returns:
            One entry per boundary node (positive = heat leaves the network).

        Raises:
            TypeError: If an entry is not real.
            ValueError: If a length is wrong or an entry is not finite.

        References:
            [EQ-INV-THERM-004].

        """
        return self.node_inflows(free, boundary)[self.n_free :]

    def jacobian_temperature(
        self, free: Sequence[float], boundary: Sequence[float]
    ) -> Matrix:
        """Return the exact Jacobian ``dF/dT`` over the free nodes.

        A link with flow ``q = g(T_second) - g(T_first)`` into ``first`` adds
        ``+a_second`` to ``(first, second)`` and ``-a_first`` to
        ``(first, first)`` (and the negatives to the ``second`` row) with
        ``a = G`` or ``a = 4 R T^3`` [EQ-INV-THERM-002].

        Args:
            free: Free temperatures (K).
            boundary: Boundary temperatures (K).

        Returns:
            ``n_free x n_free`` matrix.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If a length is wrong or an entry is not finite.

        References:
            [EQ-INV-THERM-002].

        """
        n = self.n_free
        t = self._all_temperatures(
            _vector(free, n, "free"), _vector(boundary, self.n_boundary, "boundary")
        )
        jac = [[0.0] * n for _ in range(n)]

        def add(link: ThermalLink, a_first: float, a_second: float) -> None:
            """Accumulate one link into the Jacobian.

            References:
                [EQ-INV-THERM-002]. No new equation is introduced.

            """
            f, s = link.first, link.second
            if f < n:
                jac[f][f] -= a_first
                if s < n:
                    jac[f][s] += a_second
            if s < n:
                jac[s][s] -= a_second
                if f < n:
                    jac[s][f] += a_first

        for link in self.conductive:
            add(link, link.value, link.value)
        for link in self.radiative:
            add(
                link,
                4.0 * link.value * t[link.first] ** 3,
                4.0 * link.value * t[link.second] ** 3,
            )
        return tuple(tuple(row) for row in jac)


# --------------------------------------------------------------------------
# Tests (experiments) and parameter map
# --------------------------------------------------------------------------


def constant_power(power: Sequence[float]) -> Callable[[float], Vector]:
    """Return a power schedule ``t -> Q`` that is constant in time.

    Args:
        power: Dissipation of each free node (W).

    Returns:
        The schedule.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If an entry is not finite.

    References:
        [EQ-INV-THERM-003].

    """
    q = as_vector(power, name="power")

    def schedule(time: float) -> Vector:
        """Return the constant power.

        References:
            [EQ-INV-THERM-003]. No new equation is introduced.

        """
        return q

    return schedule


def step_power(
    before: Sequence[float], after: Sequence[float], switch_time: float
) -> Callable[[float], Vector]:
    """Return a power schedule that switches from ``before`` to ``after``.

    The schedule is ``before`` for ``t <= switch_time`` and ``after`` later.

    Args:
        before: Dissipation up to the switch (W).
        after: Dissipation after the switch (W).
        switch_time: Switch time (s), non-negative.

    Returns:
        The schedule.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If a length differs, an entry is not finite or the time is
            negative.

    References:
        [EQ-INV-THERM-003].

    """
    low = as_vector(before, name="before")
    high = as_vector(after, name="after")
    if len(low) != len(high):
        raise ValueError("before and after must have the same length")
    t0 = require_finite_real(switch_time, name="switch_time")
    if t0 < 0.0:
        raise ValueError("switch_time must be non-negative")

    def schedule(time: float) -> Vector:
        """Return the power at ``time``.

        References:
            [EQ-INV-THERM-003]. No new equation is introduced.

        """
        return low if time <= t0 else high

    return schedule


@dataclass(frozen=True, slots=True)
class SteadyTest:
    """A steady thermal-balance condition: one power level, one boundary set.

    Args:
        power: Dissipation of each free node (W).
        boundary: Boundary temperatures (K), strictly positive.
        label: Optional name.

    References:
        [EQ-INV-THERM-002].

    """

    power: Vector
    boundary: Vector
    label: str = ""


@dataclass(frozen=True, slots=True)
class TransientTest:
    """A transient condition solved by backward Euler.

    Args:
        power: Schedule ``t -> Q`` (see :func:`constant_power`,
            :func:`step_power`), evaluated at the end of every step.
        boundary: Boundary temperatures (K), constant over the test.
        step: Time step ``dt`` (s), positive.
        steps: Number of steps (positive).
        sample_steps: Strictly increasing step indices in ``1..steps`` at
            which the sensors are recorded.
        initial: Initial free temperatures (K), or ``None`` for the steady
            state under ``power(0)``, which depends on ``theta``.
        label: Optional name.

    References:
        [EQ-INV-THERM-003].

    """

    power: Callable[[float], Sequence[float]]
    boundary: Vector
    step: float
    steps: int
    sample_steps: tuple[int, ...]
    initial: Vector | None = None
    label: str = ""


ThermalTest = SteadyTest | TransientTest


@dataclass(frozen=True, slots=True)
class ThermalParameter:
    """A scalar ``theta`` multiplying the nominal values of a group of links.

    The link value used by the model is ``nominal * theta`` [EQ-INV-THERM-001].
    A radiative group with all links of one surface type is an emissivity
    scale.

    Args:
        name: Parameter name.
        kind: ``"conductive"`` or ``"radiative"``.
        links: Indices into the network's ``conductive`` (or ``radiative``)
            tuple; a link belongs to at most one parameter.

    References:
        [EQ-INV-THERM-001].

    """

    name: str
    kind: Literal["conductive", "radiative"]
    links: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SteadyState:
    """Result of a steady Newton solve.

    Args:
        temperatures: Free temperatures (K) at the last iterate.
        converged: Whether ``||F||_inf`` met the tolerance.
        iterations: Newton iterations.
        residual_norm: ``||F||_inf`` (W) at the last iterate.
        stop_reason: ``"converged"``, ``"max_iterations"``,
            ``"line_search_failed"`` or ``"singular_jacobian"``.
        boundary_heat_flow: Heat flow into each boundary node (W).
        energy_residual: ``sum Q - sum boundary_heat_flow`` (W), zero for a
            steady state [EQ-INV-THERM-004].

    References:
        [EQ-INV-THERM-002], [EQ-INV-THERM-004].

    """

    temperatures: Vector
    converged: bool
    iterations: int
    residual_norm: float
    stop_reason: str
    boundary_heat_flow: Vector
    energy_residual: float


@dataclass(frozen=True, slots=True)
class TransientResponse:
    """Result of a backward Euler transient.

    Args:
        times: ``0, dt, ..., steps dt`` (s).
        temperatures: Free temperatures at each time (K).
        converged: Whether every Newton solve converged.
        stop_reason: ``"converged"`` or the failing step's reason.
        newton_iterations: Total Newton iterations.
        max_energy_residual: Largest per-step discrete energy-balance residual
            ``|sum C (T_{k+1}-T_k) - dt (sum Q - sum boundary flow)|`` (J)
            [EQ-INV-THERM-004].

    References:
        [EQ-INV-THERM-003], [EQ-INV-THERM-004].

    """

    times: Vector
    temperatures: tuple[Vector, ...]
    converged: bool
    stop_reason: str
    newton_iterations: int
    max_energy_residual: float


def _infinity_norm(values: Sequence[float]) -> float:
    """Return the maximum absolute entry.

    References:
        [EQ-INV-THERM-002]. No new equation is introduced.

    """
    return max((abs(v) for v in values), default=0.0)


def _newton(
    residual: Callable[[Vector], Vector],
    jacobian: Callable[[Vector], Matrix],
    start: Vector,
    *,
    scale: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[Vector, bool, int, float, str]:
    """Damped Newton iteration with a positivity constraint on ``T``.

    The step is halved until ``T > 0`` and the residual decreases (Armijo with
    ``1e-4``). Returns the iterate, ``converged``, the iteration count, the
    final ``||r||_inf`` and the stop reason.

    References:
        [EQ-INV-THERM-002], [REF-NOCEDAL-WRIGHT-2006].

    """
    x = start
    r = residual(x)
    norm = _infinity_norm(r)
    limit = tolerance * scale
    for iteration in range(max_iterations + 1):
        if norm <= limit:
            return x, True, iteration, norm, "converged"
        if iteration == max_iterations:
            break
        try:
            step = lu_solve(jacobian(x), tuple(-v for v in r))
        except ValueError:
            return x, False, iteration, norm, "singular_jacobian"
        alpha = 1.0
        while True:
            trial = tuple(a + alpha * b for a, b in zip(x, step, strict=True))
            if all(v > 0.0 and math.isfinite(v) for v in trial):
                r_trial = residual(trial)
                norm_trial = _infinity_norm(r_trial)
                if math.isfinite(norm_trial) and (
                    norm_trial <= (1.0 - 1e-4 * alpha) * norm or norm_trial <= limit
                ):
                    break
            alpha *= 0.5
            if alpha < 1e-12:
                return x, False, iteration, norm, "line_search_failed"
        x, r, norm = trial, r_trial, norm_trial
    return x, False, max_iterations, norm, "max_iterations"


def solve_steady_state(
    network: ThermalNetwork,
    power: Sequence[float],
    boundary: Sequence[float],
    *,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> SteadyState:
    """Solve ``F(T) = 0`` for the free temperatures by damped Newton iteration.

    The Jacobian is the exact :meth:`ThermalNetwork.jacobian_temperature`
    [EQ-INV-THERM-002]. The default start is the mean boundary temperature.
    The solution with positive absolute temperatures is found if the iteration
    converges; a non-converged result is returned with its status. For a
    network with a path from every free node to a boundary node the steady
    state is unique (the energy balance is monotone in every temperature).

    Args:
        network: The network (needs at least one boundary node).
        power: Dissipation of the free nodes (W).
        boundary: Boundary temperatures (K), strictly positive.
        initial: Start (K), strictly positive (default: mean boundary).
        tolerance: Relative tolerance on ``||F||_inf`` with the scale
            ``1 + sum |Q|``.
        max_iterations: Newton iteration limit.

    Returns:
        :class:`SteadyState`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an input is invalid or there is no boundary node.

    References:
        [EQ-INV-THERM-002], [EQ-INV-THERM-004], [REF-NOCEDAL-WRIGHT-2006].

    """
    if not isinstance(network, ThermalNetwork):
        raise TypeError("network must be a ThermalNetwork")
    if network.n_boundary < 1:
        raise ValueError("a steady state needs at least one boundary node")
    n = network.n_free
    q = _vector(power, n, "power")
    tb = _vector(boundary, network.n_boundary, "boundary", positive=True)
    tol = _positive(tolerance, "tolerance")
    steps = _count(max_iterations, "max_iterations")
    start = (
        (math.fsum(tb) / len(tb),) * n
        if initial is None
        else _vector(initial, n, "initial", positive=True)
    )

    def residual(x: Vector) -> Vector:
        """Return ``F(x)``.

        References:
            [EQ-INV-THERM-001]. No new equation is introduced.

        """
        return network.net_heat_flow(x, tb, q)

    def jacobian(x: Vector) -> Matrix:
        """Return ``dF/dT(x)``.

        References:
            [EQ-INV-THERM-002]. No new equation is introduced.

        """
        return network.jacobian_temperature(x, tb)

    x, ok, iterations, norm, reason = _newton(
        residual,
        jacobian,
        start,
        scale=1.0 + math.fsum(abs(v) for v in q),
        tolerance=tol,
        max_iterations=steps,
    )
    flow = network.boundary_heat_flow(x, tb)
    return SteadyState(
        x, ok, iterations, norm, reason, flow, math.fsum(q) - math.fsum(flow)
    )


def simulate_transient(
    network: ThermalNetwork,
    test: TransientTest,
    *,
    initial: Sequence[float] | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 50,
) -> TransientResponse:
    """Integrate ``C dT/dt = F(T, t)`` by backward Euler with Newton per step.

    ``C (T_{k+1} - T_k)/dt = F(T_{k+1}, t_{k+1})`` is solved for ``T_{k+1}``
    with the Jacobian ``C/dt - dF/dT`` [EQ-INV-THERM-003]. The scheme is
    first order, A-stable and (here) positivity preserving for the monotone
    energy balance. Summing the free-node equations gives the discrete energy
    identity [EQ-INV-THERM-004], reported as ``max_energy_residual``.

    Args:
        network: The network.
        test: The transient condition (its ``initial`` is ignored).
        initial: Start ``T_0`` (K); required here (use
            :meth:`ThermalModel.simulate` for the steady-state start).
        tolerance: Newton tolerance as in :func:`solve_steady_state`.
        max_iterations: Newton limit per step.

    Returns:
        :class:`TransientResponse`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an input is invalid or ``initial`` is missing.

    References:
        [EQ-INV-THERM-003], [EQ-INV-THERM-004].

    """
    if not isinstance(network, ThermalNetwork):
        raise TypeError("network must be a ThermalNetwork")
    _check_transient(network, test)
    if initial is None:
        raise ValueError("initial is required")
    t0 = _vector(initial, network.n_free, "initial", positive=True)
    response, _ = _transient(network, test, t0, None, tolerance, max_iterations)
    return response


def _check_transient(network: ThermalNetwork, test: TransientTest) -> None:
    """Validate a transient test against a network.

    Raises:
        TypeError: If ``power`` is not callable or ``steps`` is not an integer.
        ValueError: If a field is inconsistent.

    References:
        [EQ-INV-THERM-003]. No new equation is introduced.

    """
    if not callable(test.power):
        raise TypeError("power must be a callable t -> Q")
    _vector(test.boundary, network.n_boundary, "boundary", positive=True)
    _positive(test.step, "step")
    count = _count(test.steps, "steps")
    if not test.sample_steps:
        raise ValueError("sample_steps must not be empty")
    previous = 0
    for s in test.sample_steps:
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("sample_steps must be integers")
        if not previous < s <= count:
            raise ValueError("sample_steps must increase within 1..steps")
        previous = s
    if test.initial is not None:
        _vector(test.initial, network.n_free, "initial", positive=True)


def _transient(
    network: ThermalNetwork,
    test: TransientTest,
    start: Vector,
    sensitivity: tuple[Callable[[Vector], Matrix], Matrix] | None,
    tolerance: float,
    max_iterations: int,
) -> tuple[TransientResponse, list[Matrix]]:
    """Run the backward Euler loop, optionally with the sensitivity recursion.

    ``sensitivity`` is ``(dF_dtheta, S_0)``: a function ``T_all -> dF/dtheta``
    on the free nodes (the nominal-link parameter derivative at the boundary
    temperatures) and the initial sensitivity. The recursion is
    ``(C/dt - F_T) S_{k+1} = (C/dt) S_k + F_theta(T_{k+1})``
    [EQ-INV-THERM-003]. Returns the response and the list of sensitivities
    (empty without ``sensitivity``).

    References:
        [EQ-INV-THERM-003], [EQ-INV-THERM-004].

    """
    n = network.n_free
    dt = test.step
    tb = _vector(test.boundary, network.n_boundary, "boundary", positive=True)
    cap = network.capacities
    diag = tuple(c / dt for c in cap)
    states = [start]
    times = [0.0]
    sens: list[Matrix] = []
    s_prev: Matrix | None = None
    if sensitivity is not None:
        s_prev = sensitivity[1]
        sens.append(s_prev)
    total_iterations = 0
    worst = 0.0
    reason = "converged"
    ok = True
    for k in range(test.steps):
        time = (k + 1) * dt
        q = _vector(test.power(time), n, "power(t)")
        previous = states[-1]

        def residual(x: Vector, previous: Vector = previous, q: Vector = q) -> Vector:
            """Return the backward Euler residual ``C (x - T_k)/dt - F(x)``.

            References:
                [EQ-INV-THERM-003]. No new equation is introduced.

            """
            flow = network.net_heat_flow(x, tb, q)
            return tuple(diag[i] * (x[i] - previous[i]) - flow[i] for i in range(n))

        def jacobian(x: Vector) -> Matrix:
            """Return ``C/dt - dF/dT``.

            References:
                [EQ-INV-THERM-003]. No new equation is introduced.

            """
            ft = network.jacobian_temperature(x, tb)
            return tuple(
                tuple((diag[i] if i == j else 0.0) - ft[i][j] for j in range(n))
                for i in range(n)
            )

        x, ok_step, iterations, _, why = _newton(
            residual,
            jacobian,
            previous,
            scale=1.0 + math.fsum(abs(v) for v in q) + max(diag) * 1e-3,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        total_iterations += iterations
        states.append(x)
        times.append(time)
        if not ok_step:
            ok = False
            reason = f"step {k + 1}: {why}"
            break
        flow_b = math.fsum(network.boundary_heat_flow(x, tb))
        stored = math.fsum(cap[i] * (x[i] - previous[i]) for i in range(n))
        worst = max(worst, abs(stored - dt * (math.fsum(q) - flow_b)))
        if sensitivity is not None and s_prev is not None:
            lhs = jacobian(x)
            f_theta = sensitivity[0](tuple([*x, *tb]))
            p = len(f_theta[0])
            cols = []
            for j in range(p):
                rhs = tuple(diag[i] * s_prev[i][j] + f_theta[i][j] for i in range(n))
                cols.append(lu_solve(lhs, rhs))
            s_prev = tuple(tuple(cols[j][i] for j in range(p)) for i in range(n))
            sens.append(s_prev)
    return (
        TransientResponse(
            tuple(times), tuple(states), ok, reason, total_iterations, worst
        ),
        sens,
    )


# --------------------------------------------------------------------------
# Model with parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorResponse:
    """Sensor values and their ``theta`` sensitivities for one test.

    Args:
        records: One tuple of sensor temperatures (K) per record (one for a
            steady test, one per ``sample_steps`` entry for a transient).
        jacobians: ``d records / d theta`` per record, ``n_sensors x p``;
            empty if not requested.

    References:
        [EQ-INV-THERM-002], [EQ-INV-THERM-003].

    """

    records: tuple[Vector, ...]
    jacobians: tuple[Matrix, ...]


class ThermalModel:
    """A nominal network with scalar scale parameters on groups of links.

    ``theta_k`` multiplies the nominal value of every link of group ``k``
    [EQ-INV-THERM-001]. The model evaluates the steady and transient responses
    and their exact sensitivities in ``theta`` [EQ-INV-THERM-002],
    [EQ-INV-THERM-003].

    References:
        [EQ-INV-THERM-001], [EQ-INV-THERM-002], [EQ-INV-THERM-003].

    """

    __slots__ = ("_network", "_parameters")

    def __init__(
        self, network: ThermalNetwork, parameters: Sequence[ThermalParameter]
    ) -> None:
        """Validate and store the model.

        Args:
            network: Nominal network.
            parameters: Parameter groups (at least one), with distinct names
                and disjoint link sets.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a group is empty, refers to a missing link, shares
                a link with another group or repeats a name.

        References:
            [EQ-INV-THERM-001].

        """
        if not isinstance(network, ThermalNetwork):
            raise TypeError("network must be a ThermalNetwork")
        groups = tuple(parameters)
        if not groups:
            raise ValueError("at least one parameter is required")
        seen: set[tuple[str, int]] = set()
        names: set[str] = set()
        for par in groups:
            if not isinstance(par, ThermalParameter):
                raise TypeError("parameters must be ThermalParameter")
            if par.name in names:
                raise ValueError("parameter names must be distinct")
            names.add(par.name)
            if par.kind not in ("conductive", "radiative"):
                raise ValueError("kind must be 'conductive' or 'radiative'")
            if not par.links:
                raise ValueError("a parameter needs at least one link")
            pool = network.conductive if par.kind == "conductive" else network.radiative
            for index in par.links:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError("link indices must be integers")
                if not 0 <= index < len(pool):
                    raise ValueError("parameter link index out of range")
                key = (par.kind, index)
                if key in seen:
                    raise ValueError("a link may belong to one parameter only")
                seen.add(key)
        self._network = network
        self._parameters = groups

    @property
    def network(self) -> ThermalNetwork:
        """Return the nominal network.

        References:
            [EQ-INV-THERM-001].

        """
        return self._network

    @property
    def parameters(self) -> tuple[ThermalParameter, ...]:
        """Return the parameter groups.

        References:
            [EQ-INV-THERM-001].

        """
        return self._parameters

    @property
    def dimension(self) -> int:
        """Return the number of parameters ``p``.

        References:
            [EQ-INV-THERM-001].

        """
        return len(self._parameters)

    def network_at(self, theta: Sequence[float]) -> ThermalNetwork:
        """Return the network with link values ``nominal * theta_k``.

        Args:
            theta: Positive scales, one per parameter.

        Returns:
            The scaled network.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the length is wrong or an entry is not positive.

        References:
            [EQ-INV-THERM-001].

        """
        th = _vector(theta, self.dimension, "theta", positive=True)
        cond = list(self._network.conductive)
        rad = list(self._network.radiative)
        for value, par in zip(th, self._parameters, strict=True):
            pool = cond if par.kind == "conductive" else rad
            for index in par.links:
                link = pool[index]
                pool[index] = ThermalLink(link.first, link.second, link.value * value)
        return ThermalNetwork(
            self._network.names, self._network.capacities, tuple(cond), tuple(rad)
        )

    def parameter_derivative(self, all_temperatures: Sequence[float]) -> Matrix:
        """Return ``dF/dtheta`` (``n_free x p``) at the nominal-link scaling.

        For a link of group ``k`` with scaled value ``nominal * theta_k`` the
        flow is linear in ``theta_k``, so ``dq/dtheta_k = q / theta_k =
        nominal * (g(T_second) - g(T_first))`` [EQ-INV-THERM-002]. The
        temperatures are those of the solution (free nodes first, then
        boundary).

        Args:
            all_temperatures: Free then boundary temperatures (K).

        Returns:
            The matrix of partial derivatives (W per unit ``theta``).

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-THERM-002].

        """
        n = self._network.n_free
        t = _vector(all_temperatures, len(self._network.names), "all_temperatures")
        out = [[0.0] * self.dimension for _ in range(n)]
        for k, par in enumerate(self._parameters):
            conductive = par.kind == "conductive"
            pool = self._network.conductive if conductive else self._network.radiative
            for index in par.links:
                link = pool[index]
                if conductive:
                    d = t[link.second] - t[link.first]
                else:
                    d = t[link.second] ** 4 - t[link.first] ** 4
                q = link.value * d
                if link.first < n:
                    out[link.first][k] += q
                if link.second < n:
                    out[link.second][k] -= q
        return tuple(tuple(row) for row in out)

    def steady(
        self,
        theta: Sequence[float],
        test: SteadyTest,
        *,
        sensitivity: bool = False,
        tolerance: float = 1e-12,
        max_iterations: int = 100,
    ) -> tuple[SteadyState, Matrix | None]:
        """Solve a steady test and optionally return ``dT/dtheta``.

        ``dT/dtheta = -(dF/dT)^-1 dF/dtheta`` at the solution
        [EQ-INV-THERM-002].

        Args:
            theta: Positive scales.
            test: The steady condition.
            sensitivity: Also return the ``n_free x p`` sensitivity.
            tolerance: Newton tolerance.
            max_iterations: Newton limit.

        Returns:
            The :class:`SteadyState` and the sensitivity (or ``None``).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an input is invalid.
            ThermalSolveError: If a sensitivity is requested and Newton did
                not converge.

        References:
            [EQ-INV-THERM-002].

        """
        net = self.network_at(theta)
        state = solve_steady_state(
            net,
            test.power,
            test.boundary,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if not sensitivity:
            return state, None
        if not state.converged:
            raise ThermalSolveError(f"steady solve failed: {state.stop_reason}")
        t_all = (*state.temperatures, *test.boundary)
        jac = net.jacobian_temperature(state.temperatures, test.boundary)
        f_theta = self.parameter_derivative(t_all)
        n = net.n_free
        p = self.dimension
        cols = [
            lu_solve(jac, tuple(-f_theta[i][k] for i in range(n))) for k in range(p)
        ]
        return state, tuple(tuple(cols[k][i] for k in range(p)) for i in range(n))

    def simulate(
        self,
        theta: Sequence[float],
        test: TransientTest,
        *,
        sensitivity: bool = False,
        tolerance: float = 1e-12,
        max_iterations: int = 50,
    ) -> tuple[TransientResponse, tuple[Matrix, ...]]:
        """Run a transient test, optionally with ``dT_k/dtheta`` per step.

        With ``test.initial = None`` the start is the steady state under
        ``power(0)`` and ``S_0`` is its steady sensitivity
        [EQ-INV-THERM-003].

        Args:
            theta: Positive scales.
            test: The transient condition.
            sensitivity: Also return the sensitivities (``steps + 1`` matrices
                of size ``n_free x p``).
            tolerance: Newton tolerance.
            max_iterations: Newton limit per step.

        Returns:
            The response and the sensitivities (empty if not requested).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an input is invalid.
            ThermalSolveError: If the initial steady state does not converge.

        References:
            [EQ-INV-THERM-003].

        """
        net = self.network_at(theta)
        _check_transient(net, test)
        n = net.n_free
        if test.initial is None:
            state, s0 = self.steady(
                theta,
                SteadyTest(_vector(test.power(0.0), n, "power(0)"), test.boundary),
                sensitivity=sensitivity,
                tolerance=tolerance,
            )
            if not state.converged:
                raise ThermalSolveError(
                    f"initial steady state failed: {state.stop_reason}"
                )
            start = state.temperatures
        else:
            start = _vector(test.initial, n, "initial", positive=True)
            s0 = tuple((0.0,) * self.dimension for _ in range(n))
        derivative = (
            None if not sensitivity or s0 is None else (self.parameter_derivative, s0)
        )
        response, sens = _transient(
            net, test, start, derivative, tolerance, max_iterations
        )
        return response, tuple(sens)

    def sensor_response(
        self,
        theta: Sequence[float],
        test: SteadyTest | TransientTest,
        sensors: Sequence[int],
        *,
        jacobian: bool = False,
    ) -> SensorResponse:
        """Return the sensor records (and ``d/dtheta``) of one test.

        Args:
            theta: Positive scales.
            test: A steady or transient test.
            sensors: Free-node indices that carry a sensor.
            jacobian: Also return ``d record / d theta``.

        Returns:
            :class:`SensorResponse`.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an input is invalid.
            ThermalSolveError: If a solve does not converge.

        References:
            [EQ-INV-THERM-002], [EQ-INV-THERM-003].

        """
        n = self._network.n_free
        idx = _sensor_indices(sensors, n)
        if isinstance(test, SteadyTest):
            state, sens = self.steady(theta, test, sensitivity=jacobian)
            if not state.converged:
                raise ThermalSolveError(f"steady solve failed: {state.stop_reason}")
            rec = tuple(state.temperatures[i] for i in idx)
            jacs = (
                (tuple(sens[i] for i in idx),) if jacobian and sens is not None else ()
            )
            return SensorResponse((rec,), jacs)
        if isinstance(test, TransientTest):
            response, sensitivities = self.simulate(theta, test, sensitivity=jacobian)
            if not response.converged:
                raise ThermalSolveError(f"transient failed: {response.stop_reason}")
            records = tuple(
                tuple(response.temperatures[k][i] for i in idx)
                for k in test.sample_steps
            )
            jac_list = (
                tuple(
                    tuple(sensitivities[k][i] for i in idx) for k in test.sample_steps
                )
                if jacobian
                else ()
            )
            return SensorResponse(records, jac_list)
        raise TypeError("test must be a SteadyTest or a TransientTest")


def _sensor_indices(sensors: Sequence[int], n_free: int) -> tuple[int, ...]:
    """Validate sensor indices (distinct free nodes).

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If the list is empty, repeats or leaves the free nodes.

    References:
        [EQ-INV-THERM-005]. No new equation is introduced.

    """
    idx = tuple(sensors)
    if not idx:
        raise ValueError("at least one sensor is required")
    for i in idx:
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError("sensor indices must be integers")
        if not 0 <= i < n_free:
            raise ValueError("a sensor must sit on a free node")
    if len(set(idx)) != len(idx):
        raise ValueError("sensor indices must be distinct")
    return idx


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------


def simulate_thermal_data(
    model: ThermalModel,
    theta: Sequence[float],
    sensors: Sequence[int],
    tests: Sequence[SteadyTest | TransientTest],
    noise_sd: float,
    *,
    seed: int | RandomStream | None = 0,
) -> Vector:
    """Return noisy synthetic sensor data ``y = G(theta) + e``.

    The data are laid out record by record (one block of sensor values per
    steady test or transient sample, tests in order). The truth model may
    differ from the model that is later calibrated (model-form error)
    [EQ-INV-THERM-005].

    Args:
        model: The model that generates the truth.
        theta: True scales.
        sensors: Sensor node indices.
        tests: Calibration tests.
        noise_sd: Standard deviation of the independent Gaussian error (K);
            ``0`` gives noise-free data.
        seed: Integer seed or stream.

    Returns:
        The data vector.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an input is invalid or ``noise_sd`` is negative.
        ThermalSolveError: If a solve fails.

    References:
        [EQ-INV-THERM-005].

    """
    sd = require_finite_real(noise_sd, name="noise_sd")
    if sd < 0.0:
        raise ValueError("noise_sd must be non-negative")
    gen = resolve_stream(seed).generator()
    values: list[float] = []
    for test in tests:
        for rec in model.sensor_response(theta, test, sensors).records:
            values.extend(v + sd * gen.gauss(0.0, 1.0) for v in rec)
    return tuple(values)


# --------------------------------------------------------------------------
# Correlation problem
# --------------------------------------------------------------------------


class ThermalCorrelationProblem:
    """Bayesian correlation of a :class:`ThermalModel` to sensor data.

    Unknowns ``u = log theta`` (lognormal ``theta`` with median
    ``prior_median`` and log standard deviation ``prior_log_sd``) and, with a
    discrepancy prior, ``log tau`` [EQ-INV-THERM-005], [EQ-INV-THERM-006].
    The likelihood has independent errors of standard deviation ``noise_sd``
    and, if ``tau > 0``, a persistent offset ``delta_s ~ N(0, tau^2)`` per
    sensor shared by all records of that sensor; the offsets are integrated
    out in closed form, so the data covariance is block diagonal over sensors
    with blocks ``sigma^2 I + tau^2 1 1'`` [EQ-INV-THERM-005].

    The data vector has ``R x n_s`` entries (record-major).

    References:
        [EQ-INV-THERM-005], [EQ-INV-THERM-006], [REF-KENNEDY-OHAGAN-2001].

    """

    __slots__ = (
        "_data",
        "_disc",
        "_model",
        "_noise",
        "_prior_log_median",
        "_prior_log_sd",
        "_records",
        "_sensors",
        "_tests",
    )

    def __init__(
        self,
        model: ThermalModel,
        sensors: Sequence[int],
        tests: Sequence[SteadyTest | TransientTest],
        data: Sequence[float],
        *,
        noise_sd: float,
        prior_median: Sequence[float] | None = None,
        prior_log_sd: float | Sequence[float] = 0.5,
        discrepancy_prior: tuple[float, float] | None = None,
    ) -> None:
        """Validate and store the problem.

        Args:
            model: The model to correlate.
            sensors: Free-node indices carrying a sensor.
            tests: Calibration tests (at least one).
            data: Measured sensor temperatures (K), record-major.
            noise_sd: Known standard deviation (K) of the independent error.
            prior_median: Prior medians of ``theta`` (default ones).
            prior_log_sd: Prior standard deviation of ``log theta`` (scalar or
                one per parameter), positive.
            discrepancy_prior: ``(median, log_sd)`` of a lognormal prior of
                ``tau`` (K), or ``None`` for no discrepancy.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If dimensions disagree or a value is invalid.
            ThermalSolveError: If a test cannot be solved at the prior median.

        References:
            [EQ-INV-THERM-005], [EQ-INV-THERM-006].

        """
        if not isinstance(model, ThermalModel):
            raise TypeError("model must be a ThermalModel")
        self._model = model
        self._sensors = _sensor_indices(sensors, model.network.n_free)
        self._tests = tuple(tests)
        if not self._tests:
            raise ValueError("at least one test is required")
        for test in self._tests:
            if isinstance(test, TransientTest):
                _check_transient(model.network, test)
            elif isinstance(test, SteadyTest):
                _vector(test.power, model.network.n_free, "power")
                _vector(
                    test.boundary, model.network.n_boundary, "boundary", positive=True
                )
            else:
                raise TypeError("tests must be SteadyTest or TransientTest")
        self._records = sum(
            1 if isinstance(t, SteadyTest) else len(t.sample_steps) for t in self._tests
        )
        ns = len(self._sensors)
        self._data = _vector(data, self._records * ns, "data")
        self._noise = _positive(noise_sd, "noise_sd")
        p = model.dimension
        median = (1.0,) * p if prior_median is None else prior_median
        self._prior_log_median = tuple(
            math.log(v) for v in _vector(median, p, "prior_median", positive=True)
        )
        if isinstance(prior_log_sd, (int, float)):
            sds: Vector = (_positive(prior_log_sd, "prior_log_sd"),) * p
        else:
            sds = _vector(prior_log_sd, p, "prior_log_sd", positive=True)
        self._prior_log_sd = sds
        if discrepancy_prior is None:
            self._disc: tuple[float, float] | None = None
        else:
            if len(discrepancy_prior) != 2:
                raise ValueError("discrepancy_prior must be (median, log_sd)")
            self._disc = (
                math.log(_positive(discrepancy_prior[0], "discrepancy median")),
                _positive(discrepancy_prior[1], "discrepancy log_sd"),
            )
        # Fail early (before any optimization) if the prior median is unsolvable.
        self.forward(self._prior_log_median)

    @property
    def model(self) -> ThermalModel:
        """Return the model.

        References:
            [EQ-INV-THERM-005].

        """
        return self._model

    @property
    def sensors(self) -> tuple[int, ...]:
        """Return the sensor node indices.

        References:
            [EQ-INV-THERM-005].

        """
        return self._sensors

    @property
    def tests(self) -> tuple[SteadyTest | TransientTest, ...]:
        """Return the calibration tests.

        References:
            [EQ-INV-THERM-005].

        """
        return self._tests

    @property
    def data(self) -> Vector:
        """Return the data vector.

        References:
            [EQ-INV-THERM-005].

        """
        return self._data

    @property
    def noise_sd(self) -> float:
        """Return the measurement standard deviation.

        References:
            [EQ-INV-THERM-005].

        """
        return self._noise

    @property
    def dimension(self) -> int:
        """Return ``p``, the number of thermal parameters.

        References:
            [EQ-INV-THERM-006].

        """
        return self._model.dimension

    @property
    def has_discrepancy(self) -> bool:
        """Return whether ``log tau`` is an unknown.

        References:
            [EQ-INV-THERM-005].

        """
        return self._disc is not None

    @property
    def records(self) -> int:
        """Return the number of records ``R``.

        References:
            [EQ-INV-THERM-005].

        """
        return self._records

    @property
    def prior_log_median(self) -> Vector:
        """Return the prior mean of ``u = log theta``.

        References:
            [EQ-INV-THERM-006].

        """
        return self._prior_log_median

    @property
    def prior_log_sd(self) -> Vector:
        """Return the prior standard deviation of ``u``.

        References:
            [EQ-INV-THERM-006].

        """
        return self._prior_log_sd

    @property
    def discrepancy_prior(self) -> tuple[float, float] | None:
        """Return ``(log median, log sd)`` of ``tau`` or ``None``.

        References:
            [EQ-INV-THERM-005].

        """
        return self._disc

    def forward(self, u: Sequence[float]) -> Vector:
        """Return the stacked sensor data ``G(exp u)``.

        Args:
            u: ``log theta`` (length ``p``).

        Returns:
            Model counterpart of the data.

        Raises:
            ValueError: If ``u`` is invalid.
            ThermalSolveError: If a solve fails.

        References:
            [EQ-INV-THERM-005].

        """
        theta = tuple(math.exp(v) for v in _vector(u, self.dimension, "u"))
        out: list[float] = []
        for test in self._tests:
            for rec in self._model.sensor_response(theta, test, self._sensors).records:
                out.extend(rec)
        return tuple(out)

    def forward_jacobian(self, u: Sequence[float]) -> tuple[Vector, Matrix]:
        """Return ``G(exp u)`` and the exact Jacobian ``dG/du``.

        ``dG/du_k = theta_k dG/dtheta_k`` [EQ-INV-THERM-002],
        [EQ-INV-THERM-003].

        Args:
            u: ``log theta``.

        Returns:
            ``(values, jacobian)`` with ``m`` rows and ``p`` columns.

        Raises:
            ValueError: If ``u`` is invalid.
            ThermalSolveError: If a solve fails.

        References:
            [EQ-INV-THERM-002], [EQ-INV-THERM-003].

        """
        theta = tuple(math.exp(v) for v in _vector(u, self.dimension, "u"))
        values: list[float] = []
        rows: list[Vector] = []
        for test in self._tests:
            resp = self._model.sensor_response(
                theta, test, self._sensors, jacobian=True
            )
            for rec, jac in zip(resp.records, resp.jacobians, strict=True):
                values.extend(rec)
                rows.extend(
                    tuple(theta[k] * row[k] for k in range(self.dimension))
                    for row in jac
                )
        return tuple(values), tuple(rows)

    def _whiten(self, vector: Sequence[float], tau: float) -> Vector:
        """Apply the whitening ``W`` with ``W'W = Sigma^-1`` to a data vector.

        Per sensor the records ``r`` give
        ``(1/sigma) [(I - P) r + sqrt(sigma^2/(sigma^2 + R tau^2)) P r]`` with
        ``P = 1 1'/R`` [EQ-INV-THERM-005].

        References:
            [EQ-INV-THERM-005].

        """
        ns = len(self._sensors)
        r = self._records
        out = [0.0] * (r * ns)
        sigma = self._noise
        shrink = sigma / math.sqrt(sigma * sigma + r * tau * tau)
        for s in range(ns):
            mean = math.fsum(vector[k * ns + s] for k in range(r)) / r
            for k in range(r):
                out[k * ns + s] = (vector[k * ns + s] - mean + shrink * mean) / sigma
        return tuple(out)

    def _tau(self, tau: float | None) -> float:
        """Validate a discrepancy standard deviation (``None`` means zero).

        References:
            [EQ-INV-THERM-005]. No new equation is introduced.

        """
        if tau is None:
            return 0.0
        value = require_finite_real(tau, name="tau")
        if value < 0.0:
            raise ValueError("tau must be non-negative")
        return value

    def whitened_residual(self, u: Sequence[float], tau: float | None = None) -> Vector:
        """Return ``W (y - G(exp u))`` for the discrepancy level ``tau``.

        Args:
            u: ``log theta``.
            tau: Discrepancy standard deviation (K), default zero.

        Returns:
            The whitened data residual.

        Raises:
            ValueError: If an argument is invalid.
            ThermalSolveError: If a solve fails.

        References:
            [EQ-INV-THERM-005].

        """
        t = self._tau(tau)
        g = self.forward(u)
        return self._whiten(tuple(a - b for a, b in zip(self._data, g, strict=True)), t)

    def log_likelihood(self, u: Sequence[float], tau: float | None = None) -> float:
        """Return ``log p(y | theta, tau)`` including all constants.

        ``-1/2 [m log 2 pi + m log sigma^2 + n_s log(1 + R tau^2/sigma^2)]
        - 1/2 ||W r||^2`` [EQ-INV-THERM-005].

        Args:
            u: ``log theta``.
            tau: Discrepancy standard deviation (K), default zero.

        Returns:
            The log-likelihood.

        Raises:
            ValueError: If an argument is invalid.
            ThermalSolveError: If a solve fails.

        References:
            [EQ-INV-THERM-005].

        """
        t = self._tau(tau)
        w = self.whitened_residual(u, t)
        m = len(self._data)
        ns = len(self._sensors)
        log_det = m * math.log(self._noise**2) + ns * math.log(
            1.0 + self._records * t * t / self._noise**2
        )
        return -0.5 * (m * _LOG_TWO_PI + log_det + math.fsum(v * v for v in w))

    def log_prior(self, v: Sequence[float]) -> float:
        """Return the Gaussian log density of ``v = (u, log tau)``.

        Args:
            v: ``u`` of length ``p`` (plus ``log tau`` with a discrepancy).

        Returns:
            ``log N(u; log median, s^2)`` (plus the ``log tau`` term).

        Raises:
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-THERM-006].

        """
        p = self.dimension
        vec = _vector(v, p + (1 if self._disc else 0), "v")
        total = 0.0
        for x, m, s in zip(
            vec[:p], self._prior_log_median, self._prior_log_sd, strict=True
        ):
            total -= 0.5 * ((x - m) / s) ** 2 + math.log(s) + 0.5 * _LOG_TWO_PI
        if self._disc is not None:
            m, s = self._disc
            total -= 0.5 * ((vec[p] - m) / s) ** 2 + math.log(s) + 0.5 * _LOG_TWO_PI
        return total

    def log_posterior_unnormalized(self, v: Sequence[float]) -> float:
        """Return ``log L + log p`` at ``v = (u, log tau)`` (``-inf`` on failure).

        Args:
            v: ``u`` (plus ``log tau`` with a discrepancy).

        Returns:
            The unnormalized log posterior; ``-inf`` if a solve fails.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-THERM-006].

        """
        p = self.dimension
        vec = _vector(v, p + (1 if self._disc else 0), "v")
        tau = math.exp(vec[p]) if self._disc is not None else 0.0
        try:
            return self.log_likelihood(vec[:p], tau) + self.log_prior(vec)
        except ThermalSolveError:
            return -math.inf

    def prior_sample(self, generator: random.Random) -> Vector:
        """Draw ``v = (u, log tau)`` from the prior.

        Args:
            generator: Source of randomness.

        Returns:
            One prior draw.

        References:
            [EQ-INV-THERM-006].

        """
        draw = [
            generator.gauss(m, s)
            for m, s in zip(self._prior_log_median, self._prior_log_sd, strict=True)
        ]
        if self._disc is not None:
            draw.append(generator.gauss(*self._disc))
        return tuple(draw)

    def whitened_jacobian(self, u: Sequence[float], tau: float | None = None) -> Matrix:
        """Return ``-W dG/du``, the Jacobian of :meth:`whitened_residual`.

        Args:
            u: ``log theta``.
            tau: Discrepancy standard deviation (K), default zero.

        Returns:
            ``m x p`` matrix.

        Raises:
            ValueError: If an argument is invalid.
            ThermalSolveError: If a solve fails.

        References:
            [EQ-INV-THERM-005], [EQ-INV-THERM-006].

        """
        t = self._tau(tau)
        _, jac = self.forward_jacobian(u)
        p = self.dimension
        cols = [self._whiten(tuple(-row[k] for row in jac), t) for k in range(p)]
        return tuple(tuple(cols[k][i] for k in range(p)) for i in range(len(jac)))


@dataclass(frozen=True, slots=True)
class DerivativeCheck:
    """Analytic against central-difference Jacobian of the sensor map.

    Args:
        max_abs_error: Largest absolute entry difference (K per unit ``u``).
        max_rel_error: Largest difference over ``max(1, |entry|)``.
        step: Central-difference step in ``u``.

    References:
        [EQ-INV-THERM-002], [EQ-INV-THERM-003].

    """

    max_abs_error: float
    max_rel_error: float
    step: float


def check_thermal_jacobian(
    problem: ThermalCorrelationProblem, u: Sequence[float], *, step: float = 1e-6
) -> DerivativeCheck:
    """Compare ``dG/du`` with central differences of the forward map.

    Args:
        problem: The problem.
        u: Evaluation point ``log theta``.
        step: Central-difference step in ``u`` (positive).

    Returns:
        :class:`DerivativeCheck`.

    Raises:
        TypeError: If ``problem`` has the wrong type.
        ValueError: If ``u`` or ``step`` is invalid.
        ThermalSolveError: If a solve fails.

    References:
        [EQ-INV-THERM-002], [EQ-INV-THERM-003].

    """
    if not isinstance(problem, ThermalCorrelationProblem):
        raise TypeError("problem must be a ThermalCorrelationProblem")
    h = _positive(step, "step")
    point = _vector(u, problem.dimension, "u")
    _, analytic = problem.forward_jacobian(point)
    worst_abs = 0.0
    worst_rel = 0.0
    for k in range(problem.dimension):
        up = tuple(v + (h if i == k else 0.0) for i, v in enumerate(point))
        um = tuple(v - (h if i == k else 0.0) for i, v in enumerate(point))
        gp, gm = problem.forward(up), problem.forward(um)
        for i in range(len(gp)):
            fd = (gp[i] - gm[i]) / (2.0 * h)
            diff = abs(fd - analytic[i][k])
            worst_abs = max(worst_abs, diff)
            worst_rel = max(worst_rel, diff / max(1.0, abs(analytic[i][k])))
    return DerivativeCheck(worst_abs, worst_rel, h)


# --------------------------------------------------------------------------
# MAP, Laplace
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThermalLaplace:
    """MAP and Laplace posterior of ``u = log theta`` at a discrepancy level.

    Args:
        problem: The problem.
        u_map: MAP of ``log theta``.
        theta_map: ``exp(u_map)``.
        tau: Discrepancy standard deviation used (zero without discrepancy).
        covariance: Posterior covariance of ``u``, ``(J'J + S^-1)^-1``
            (``None`` if the Hessian is not positive definite).
        likelihood_hessian: ``J'J`` of the whitened data residual at the MAP.
        log_evidence: Laplace ``log p(y | tau)``
            [EQ-INV-THERM-006] (NaN if no covariance).
        converged: MAP optimizer converged and the Hessian is SPD.
        stop_reason: Optimizer message.
        tau_grid: Discrepancy levels scanned (empty if ``tau`` was given or
            there is no discrepancy).
        tau_scores: ``log evidence + log prior(log tau)`` at each level.

    References:
        [EQ-INV-THERM-006], [REF-NOCEDAL-WRIGHT-2006].

    """

    problem: ThermalCorrelationProblem
    u_map: Vector
    theta_map: Vector
    tau: float
    covariance: Matrix | None
    likelihood_hessian: Matrix
    log_evidence: float
    converged: bool
    stop_reason: str
    tau_grid: Vector
    tau_scores: Vector

    def standard_deviations(self) -> Vector:
        """Return the posterior standard deviations of ``u``.

        Returns:
            One entry per parameter.

        Raises:
            ValueError: If the posterior covariance is not available.

        References:
            [EQ-INV-THERM-006].

        """
        if self.covariance is None:
            raise ValueError("the Hessian is not positive definite")
        return tuple(math.sqrt(self.covariance[i][i]) for i in range(len(self.u_map)))

    def correlation(self) -> Matrix:
        """Return the posterior correlation matrix of ``u``.

        Returns:
            ``p x p`` matrix with unit diagonal.

        Raises:
            ValueError: If the posterior covariance is not available.

        References:
            [EQ-INV-THERM-006], [EQ-INV-THERM-007].

        """
        sd = self.standard_deviations()
        cov = self.covariance
        assert cov is not None
        return tuple(
            tuple(cov[i][j] / (sd[i] * sd[j]) for j in range(len(sd)))
            for i in range(len(sd))
        )

    def sample(
        self, size: int, *, seed: int | RandomStream | None = 0
    ) -> ThermalPosterior:
        """Draw ``u ~ N(u_map, covariance)`` (``tau`` fixed at its plug-in).

        Args:
            size: Number of draws (positive).
            seed: Integer seed or stream.

        Returns:
            :class:`ThermalPosterior` of the draws.

        Raises:
            ValueError: If ``size`` is invalid or no covariance is available.

        References:
            [EQ-INV-THERM-006].

        """
        count = _count(size, "size")
        if self.covariance is None:
            raise ValueError("the Hessian is not positive definite")
        lower = cholesky(self.covariance)
        gen = resolve_stream(seed).generator()
        p = len(self.u_map)
        rows: list[Vector] = []
        for _ in range(count):
            z = [gen.gauss(0.0, 1.0) for _ in range(p)]
            u = [
                self.u_map[i] + math.fsum(lower[i][j] * z[j] for j in range(i + 1))
                for i in range(p)
            ]
            if self.problem.has_discrepancy:
                u.append(math.log(self.tau))
            rows.append(tuple(u))
        return ThermalPosterior(
            problem=self.problem,
            samples=tuple(rows),
            method="laplace",
            converged=self.converged,
            log_evidence=self.log_evidence,
            tmcmc=None,
        )


def _laplace_at(
    problem: ThermalCorrelationProblem,
    start: Vector,
    tau: float,
    max_iterations: int,
    tolerance: float,
) -> ThermalLaplace:
    """Run LM at a fixed ``tau`` and build the Laplace posterior.

    The stacked residual is ``[W (y - G); (u - m)/s]``; its Jacobian is
    ``[-W J; S^-1]`` [EQ-INV-THERM-006].

    References:
        [EQ-INV-THERM-006], [REF-NOCEDAL-WRIGHT-2006].

    """
    p = problem.dimension
    mu = problem.prior_log_median
    sd = problem.prior_log_sd

    def residual(u: Vector) -> Vector:
        """Return the stacked whitened data and prior residual.

        References:
            [EQ-INV-THERM-006]. No new equation is introduced.

        """
        data = problem.whitened_residual(u, tau)
        return (*data, *((u[i] - mu[i]) / sd[i] for i in range(p)))

    def jacobian(u: Vector) -> Matrix:
        """Return the stacked Jacobian.

        References:
            [EQ-INV-THERM-006]. No new equation is introduced.

        """
        top = problem.whitened_jacobian(u, tau)
        bottom = tuple(
            tuple(1.0 / sd[i] if i == j else 0.0 for j in range(p)) for i in range(p)
        )
        return (*top, *bottom)

    fit = least_squares(
        residual,
        start,
        jacobian=jacobian,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    u_map = fit.x
    w_jac = problem.whitened_jacobian(u_map, tau)
    lik_hessian = tuple(
        tuple(math.fsum(row[a] * row[b] for row in w_jac) for b in range(p))
        for a in range(p)
    )
    precision = tuple(
        tuple(
            lik_hessian[a][b] + (1.0 / sd[a] ** 2 if a == b else 0.0) for b in range(p)
        )
        for a in range(p)
    )
    covariance: Matrix | None
    try:
        covariance = inverse_spd(precision)
        log_prior_u = -math.fsum(
            0.5 * ((u_map[i] - mu[i]) / sd[i]) ** 2
            + math.log(sd[i])
            + 0.5 * _LOG_TWO_PI
            for i in range(p)
        )
        evidence = (
            problem.log_likelihood(u_map, tau)
            + log_prior_u
            + 0.5 * p * _LOG_TWO_PI
            - 0.5 * log_det_spd(precision)
        )
        spd = True
    except ValueError:
        covariance = None
        evidence = math.nan
        spd = False
    return ThermalLaplace(
        problem=problem,
        u_map=u_map,
        theta_map=tuple(math.exp(v) for v in u_map),
        tau=tau,
        covariance=covariance,
        likelihood_hessian=lik_hessian,
        log_evidence=evidence,
        converged=bool(fit.converged and spd),
        stop_reason=fit.message,
        tau_grid=(),
        tau_scores=(),
    )


def thermal_laplace(
    problem: ThermalCorrelationProblem,
    initial: Sequence[float] | None = None,
    *,
    tau: float | None = None,
    tau_grid_size: int = 15,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> ThermalLaplace:
    """Find the MAP by Levenberg-Marquardt and return the Laplace posterior.

    Without a discrepancy the problem is fitted once. With a discrepancy prior
    and ``tau=None`` the level ``tau`` is chosen on a grid of
    ``tau_grid_size`` log-spaced values within three prior log standard
    deviations of the prior median by the largest score
    ``log evidence(tau) + log p(log tau)`` (each level re-fitted from the
    previous MAP); the Laplace posterior is that of the best level (a plug-in,
    ``tau`` uncertainty ignored). The best level at a grid end is flagged by
    ``converged=False`` with the message ``"tau at grid end"``. A local optimum
    is returned [EQ-INV-THERM-006].

    Args:
        problem: The problem.
        initial: Start in ``u`` (default the prior median).
        tau: Fixed discrepancy standard deviation (K); only with a
            discrepancy prior or zero.
        tau_grid_size: Grid points for the discrepancy scan (``>= 3``).
        max_iterations: Levenberg-Marquardt iteration limit.
        tolerance: Levenberg-Marquardt tolerance.

    Returns:
        :class:`ThermalLaplace`.

    Raises:
        TypeError: If ``problem`` has the wrong type.
        ValueError: If an option is invalid or ``tau`` is given for a problem
            without a discrepancy prior.
        ThermalSolveError: If the start cannot be solved.

    References:
        [EQ-INV-THERM-006], [REF-NOCEDAL-WRIGHT-2006].

    """
    if not isinstance(problem, ThermalCorrelationProblem):
        raise TypeError("problem must be a ThermalCorrelationProblem")
    steps = _count(max_iterations, "max_iterations")
    start = (
        problem.prior_log_median
        if initial is None
        else _vector(initial, problem.dimension, "initial")
    )
    if not problem.has_discrepancy:
        if tau is not None and problem._tau(tau) != 0.0:
            raise ValueError("tau needs a problem with a discrepancy prior")
        return _laplace_at(problem, start, 0.0, steps, tolerance)
    if tau is not None:
        return _laplace_at(problem, start, problem._tau(tau), steps, tolerance)
    size = _count(tau_grid_size, "tau_grid_size")
    if size < 3:
        raise ValueError("tau_grid_size must be at least 3")
    prior = problem.discrepancy_prior
    assert prior is not None
    centre, spread = prior
    grid = tuple(
        math.exp(centre + spread * (-3.0 + 6.0 * i / (size - 1))) for i in range(size)
    )
    best: ThermalLaplace | None = None
    best_score = -math.inf
    best_index = -1
    scores: list[float] = []
    current = start
    for i, value in enumerate(grid):
        lap = _laplace_at(problem, current, value, steps, tolerance)
        current = lap.u_map
        z = (math.log(value) - centre) / spread
        score = lap.log_evidence - 0.5 * z * z - math.log(spread) - 0.5 * _LOG_TWO_PI
        scores.append(score if math.isfinite(score) else -math.inf)
        if scores[-1] > best_score:
            best, best_score, best_index = lap, scores[-1], i
    if best is None:
        raise ValueError("no discrepancy level gave a finite evidence")
    at_end = best_index in (0, size - 1)
    return ThermalLaplace(
        problem=best.problem,
        u_map=best.u_map,
        theta_map=best.theta_map,
        tau=best.tau,
        covariance=best.covariance,
        likelihood_hessian=best.likelihood_hessian,
        log_evidence=best.log_evidence,
        converged=best.converged and not at_end,
        stop_reason="tau at grid end" if at_end else best.stop_reason,
        tau_grid=grid,
        tau_scores=tuple(scores),
    )


# --------------------------------------------------------------------------
# Posterior samples (Laplace draws or TMCMC)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThermalPosterior:
    """Posterior samples of ``(u, log tau)``.

    Args:
        problem: The problem.
        samples: One tuple per sample: ``u`` (length ``p``) and, with a
            discrepancy, ``log tau`` as the last entry.
        method: ``"laplace"`` or ``"tmcmc"``.
        converged: Whether the producing algorithm converged.
        log_evidence: Evidence estimate (NaN if not available).
        tmcmc: The TMCMC run, if any.

    References:
        [EQ-INV-THERM-006], [REF-CHING-CHEN-2007].

    """

    problem: ThermalCorrelationProblem
    samples: tuple[Vector, ...]
    method: str
    converged: bool
    log_evidence: float
    tmcmc: TMCMCResult | None

    def theta_samples(self) -> tuple[Vector, ...]:
        """Return the samples of ``theta = exp(u)``.

        Returns:
            One tuple of length ``p`` per sample.

        References:
            [EQ-INV-THERM-006].

        """
        p = self.problem.dimension
        return tuple(tuple(math.exp(v) for v in s[:p]) for s in self.samples)

    def tau_samples(self) -> Vector:
        """Return the samples of ``tau`` (empty without a discrepancy).

        Returns:
            One entry per sample, or an empty tuple.

        References:
            [EQ-INV-THERM-005].

        """
        if not self.problem.has_discrepancy:
            return ()
        return tuple(math.exp(s[-1]) for s in self.samples)

    def mean_log(self) -> Vector:
        """Return the sample mean of ``u``.

        Returns:
            One entry per parameter.

        References:
            [EQ-INV-THERM-006].

        """
        p = self.problem.dimension
        n = len(self.samples)
        return tuple(math.fsum(s[i] for s in self.samples) / n for i in range(p))

    def covariance_log(self) -> Matrix:
        """Return the sample covariance of ``u``.

        Returns:
            ``p x p`` matrix (``n - 1`` denominator).

        Raises:
            ValueError: If there are fewer than two samples.

        References:
            [EQ-INV-THERM-006].

        """
        n = len(self.samples)
        if n < 2:
            raise ValueError("at least two samples are required")
        p = self.problem.dimension
        mean = self.mean_log()
        return tuple(
            tuple(
                math.fsum((s[a] - mean[a]) * (s[b] - mean[b]) for s in self.samples)
                / (n - 1)
                for b in range(p)
            )
            for a in range(p)
        )

    def standard_deviations(self) -> Vector:
        """Return the sample standard deviations of ``u``.

        Returns:
            One entry per parameter.

        References:
            [EQ-INV-THERM-006].

        """
        cov = self.covariance_log()
        return tuple(math.sqrt(cov[i][i]) for i in range(len(cov)))

    def correlation(self) -> Matrix:
        """Return the sample correlation matrix of ``u``.

        Returns:
            ``p x p`` matrix.

        References:
            [EQ-INV-THERM-006], [EQ-INV-THERM-007].

        """
        cov = self.covariance_log()
        sd = tuple(math.sqrt(cov[i][i]) for i in range(len(cov)))
        return tuple(
            tuple(cov[i][j] / (sd[i] * sd[j]) for j in range(len(cov)))
            for i in range(len(cov))
        )


def thermal_tmcmc(
    problem: ThermalCorrelationProblem,
    *,
    particles: int = 300,
    seed: int | RandomStream | None = 0,
    burn_in: int = 5,
    cov_target: float = 1.0,
    max_stages: int = 100,
) -> ThermalPosterior:
    """Sample the posterior of ``(log theta, log tau)`` by transitional MCMC.

    The prior is the lognormal prior of :class:`ThermalCorrelationProblem`
    and a failed forward solve counts as zero likelihood
    [REF-CHING-CHEN-2007]. The result carries the :class:`TMCMCResult`;
    check ``converged`` before using the samples.

    Args:
        problem: The problem.
        particles: Number of samples (at least 2).
        seed: Integer seed or stream.
        burn_in: Discarded Metropolis steps per chain and stage.
        cov_target: Target coefficient of variation of the weights.
        max_stages: Maximum number of stages.

    Returns:
        :class:`ThermalPosterior` with ``method="tmcmc"``.

    Raises:
        TypeError: If ``problem`` has the wrong type.
        ValueError: If an option is invalid.

    References:
        [REF-CHING-CHEN-2007], [EQ-INV-THERM-006].

    """
    if not isinstance(problem, ThermalCorrelationProblem):
        raise TypeError("problem must be a ThermalCorrelationProblem")
    names = [f"log_{par.name}" for par in problem.model.parameters]
    if problem.has_discrepancy:
        names.append("log_tau")

    def log_likelihood(v: Vector) -> float:
        """Return ``log L`` at ``v`` (``-inf`` if a solve fails).

        References:
            [EQ-INV-THERM-005]. No new equation is introduced.

        """
        p = problem.dimension
        tau = math.exp(v[p]) if problem.has_discrepancy else 0.0
        try:
            return problem.log_likelihood(v[:p], tau)
        except ThermalSolveError:
            return -math.inf

    result = transitional_mcmc(
        problem.prior_sample,
        problem.log_prior,
        log_likelihood,
        particles=particles,
        burn_in=burn_in,
        cov_target=cov_target,
        max_stages=max_stages,
        seed=seed,
        names=tuple(names),
    )
    return ThermalPosterior(
        problem=problem,
        samples=tuple(tuple(x) for x in result.particles),
        method="tmcmc",
        converged=result.converged,
        log_evidence=result.log_evidence,
        tmcmc=result,
    )


# --------------------------------------------------------------------------
# Identifiability
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    """What the data can and cannot separate.

    Args:
        names: Parameter names.
        likelihood_eigenvalues: Eigenvalues of ``J'J`` (whitened data Jacobian
            in ``u``), ascending. A value near zero means a direction of
            ``u`` that the data do not constrain.
        weakest_direction: Unit eigenvector of the smallest eigenvalue.
        prior_sd: Prior standard deviations of ``u``.
        posterior_sd: Laplace posterior standard deviations of ``u``.
        shrinkage: ``posterior_sd / prior_sd`` (near 1 means the data taught
            nothing about that parameter).
        correlation: Laplace posterior correlation of ``u``.

    References:
        [EQ-INV-THERM-007].

    """

    names: tuple[str, ...]
    likelihood_eigenvalues: Vector
    weakest_direction: Vector
    prior_sd: Vector
    posterior_sd: Vector
    shrinkage: Vector
    correlation: Matrix


def identifiability_report(laplace: ThermalLaplace) -> IdentifiabilityReport:
    """Summarize identifiability from a Laplace posterior.

    Args:
        laplace: A converged :class:`ThermalLaplace`.

    Returns:
        :class:`IdentifiabilityReport`.

    Raises:
        TypeError: If ``laplace`` has the wrong type.
        ValueError: If the posterior covariance is not available.

    References:
        [EQ-INV-THERM-007].

    """
    if not isinstance(laplace, ThermalLaplace):
        raise TypeError("laplace must be a ThermalLaplace")
    values, vectors = symmetric_eigen(laplace.likelihood_hessian)
    sd = laplace.standard_deviations()
    prior = laplace.problem.prior_log_sd
    return IdentifiabilityReport(
        names=tuple(par.name for par in laplace.problem.model.parameters),
        likelihood_eigenvalues=tuple(values),
        weakest_direction=tuple(vectors[i][0] for i in range(len(values))),
        prior_sd=prior,
        posterior_sd=sd,
        shrinkage=tuple(a / b for a, b in zip(sd, prior, strict=True)),
        correlation=laplace.correlation(),
    )


def profile_log_parameter(
    problem: ThermalCorrelationProblem,
    estimate: Sequence[float],
    index: int,
    grid: Sequence[float],
    *,
    tau: float | None = None,
) -> ProfileLikelihoodResult:
    """Profile the data misfit ``1/2 ||W r||^2`` of one ``log theta_k``.

    The objective is the negative log-likelihood up to a constant (the prior
    is not included, so the profile shows what the data alone say). The
    nuisance parameters are re-optimized at every grid value by
    :func:`~probstat.inverse.profile_likelihood`. A flat profile marks a
    parameter the data cannot determine [EQ-INV-THERM-007].

    Args:
        problem: The problem.
        estimate: ``u`` at (or near) the minimizer of the misfit, e.g. the
            MAP.
        index: Parameter index.
        grid: Strictly increasing values of ``u_index`` bracketing the
            estimate.
        tau: Discrepancy level (default zero).

    Returns:
        :class:`~probstat.inverse.ProfileLikelihoodResult`.

    Raises:
        TypeError: If ``problem`` has the wrong type.
        ValueError: If an argument is invalid.

    References:
        [EQ-INV-THERM-007], [REF-RAUE-ETAL-2009].

    """
    if not isinstance(problem, ThermalCorrelationProblem):
        raise TypeError("problem must be a ThermalCorrelationProblem")
    level = problem._tau(tau)

    def nll(u: Vector) -> float:
        """Return ``1/2 ||W r(u)||^2`` (``inf`` if a solve fails).

        References:
            [EQ-INV-THERM-007]. No new equation is introduced.

        """
        try:
            w = problem.whitened_residual(u, level)
        except ThermalSolveError:
            return math.inf
        return 0.5 * math.fsum(v * v for v in w)

    return profile_likelihood(nll, estimate, index, grid)


# --------------------------------------------------------------------------
# Prediction of an untested case
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictiveBand:
    """Posterior predictive of an untested steady case at chosen nodes.

    Args:
        nodes: Predicted free-node indices.
        mean: Predictive mean (K).
        sd: Predictive standard deviation (K).
        lower: Lower band edge (K), the ``(1 - level)/2`` quantile.
        upper: Upper band edge (K), the ``(1 + level)/2`` quantile.
        level: Central probability of the band.
        used: Posterior samples that produced a prediction.
        failed: Samples whose steady solve did not converge (dropped).
        includes_discrepancy: Whether sensor nodes carry a drawn offset.

    References:
        [EQ-INV-THERM-008].

    """

    nodes: tuple[int, ...]
    mean: Vector
    sd: Vector
    lower: Vector
    upper: Vector
    level: float
    used: int
    failed: int
    includes_discrepancy: bool

    def covers(self, values: Sequence[float]) -> tuple[bool, ...]:
        """Return whether each value lies inside its band.

        Args:
            values: One value per node.

        Returns:
            One flag per node.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-THERM-008].

        """
        vec = _vector(values, len(self.nodes), "values")
        return tuple(
            lo <= v <= hi for v, lo, hi in zip(vec, self.lower, self.upper, strict=True)
        )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated quantile of sorted data.

    References:
        [EQ-INV-THERM-008]. No new equation is introduced.

    """
    position = probability * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    frac = position - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def predict_flight(
    posterior: ThermalPosterior,
    flight: SteadyTest,
    nodes: Sequence[int],
    *,
    level: float = 0.9,
    max_samples: int = 400,
    seed: int | RandomStream | None = 0,
) -> PredictiveBand:
    """Return the posterior predictive band of a steady flight case.

    Each (thinned) posterior sample ``(theta, tau)`` is propagated through the
    steady model at the flight power and boundary temperatures, which differ
    from the tests; if the problem has a discrepancy, a fresh offset
    ``delta ~ N(0, tau^2)`` is added at every predicted node that carries a
    sensor (the unmodelled effect at flight is treated as exchangeable with
    that of the tests, a conservative choice, not a physical model). The band
    is for the true temperature, not for a new noisy measurement
    [EQ-INV-THERM-008].

    Args:
        posterior: Posterior samples (Laplace draws or TMCMC).
        flight: The untested steady condition.
        nodes: Free-node indices to predict.
        level: Central probability of the band, in ``(0, 1)``.
        max_samples: Cap on the samples propagated (evenly thinned).
        seed: Integer seed or stream (discrepancy draws).

    Returns:
        :class:`PredictiveBand`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is invalid or every sample fails.

    References:
        [EQ-INV-THERM-008], [REF-KENNEDY-OHAGAN-2001].

    """
    if not isinstance(posterior, ThermalPosterior):
        raise TypeError("posterior must be a ThermalPosterior")
    if not isinstance(flight, SteadyTest):
        raise TypeError("flight must be a SteadyTest")
    problem = posterior.problem
    model = problem.model
    n = model.network.n_free
    idx = _sensor_indices(nodes, n)
    prob = require_finite_real(level, name="level")
    if not 0.0 < prob < 1.0:
        raise ValueError("level must lie in (0, 1)")
    cap = _count(max_samples, "max_samples")
    total = len(posterior.samples)
    stride = max(1, total // cap)
    chosen = posterior.samples[::stride]
    gen = resolve_stream(seed).generator()
    p = problem.dimension
    sensor_set = set(problem.sensors)
    columns: list[list[float]] = [[] for _ in idx]
    failed = 0
    for sample in chosen:
        theta = tuple(math.exp(v) for v in sample[:p])
        state, _ = model.steady(theta, flight)
        if not state.converged:
            failed += 1
            continue
        tau = math.exp(sample[p]) if problem.has_discrepancy else 0.0
        for c, node in enumerate(idx):
            value = state.temperatures[node]
            if tau > 0.0 and node in sensor_set:
                value += gen.gauss(0.0, tau)
            columns[c].append(value)
    if not columns[0]:
        raise ValueError("every posterior sample failed to produce a prediction")
    means, sds, lows, highs = [], [], [], []
    for col in columns:
        k = len(col)
        mean = math.fsum(col) / k
        means.append(mean)
        sds.append(
            math.sqrt(math.fsum((v - mean) ** 2 for v in col) / (k - 1))
            if k > 1
            else 0.0
        )
        ordered = sorted(col)
        lows.append(_quantile(ordered, 0.5 * (1.0 - prob)))
        highs.append(_quantile(ordered, 0.5 * (1.0 + prob)))
    return PredictiveBand(
        nodes=idx,
        mean=tuple(means),
        sd=tuple(sds),
        lower=tuple(lows),
        upper=tuple(highs),
        level=prob,
        used=len(columns[0]),
        failed=failed,
        includes_discrepancy=problem.has_discrepancy,
    )


# --------------------------------------------------------------------------
# Synthetic demonstration case
# --------------------------------------------------------------------------

DEMO_NODES = ("panel", "box", "bracket", "radiator", "battery", "shroud")
"""Node names of :func:`demo_spacecraft_model` (free nodes 0-4, boundary 5)."""

DEMO_SENSORS = (0, 1, 3, 4)
"""Sensor nodes of the demonstration case: panel, box, radiator, battery."""

DEMO_PARAMETER_NAMES = ("G_box_bracket", "G_bracket_radiator", "G_panel_box", "eps_rad")
"""Parameter names of :func:`demo_spacecraft_model`."""


def demo_spacecraft_model(extra_panel_radiator: float = 0.0) -> ThermalModel:
    """Return the bundled synthetic five-node spacecraft network.

    Free nodes: panel (0), electronics box (1), bracket (2), radiator (3),
    battery (4); boundary node 5 is a shroud or sink. Conductive links
    (W/K): panel-box 0.5, box-bracket 2.0, bracket-radiator 1.2, box-battery
    1.5. Radiative links (W/K^4, ``epsilon sigma A F``): panel-shroud
    ``0.8 sigma 2.0``, radiator-shroud ``0.85 sigma 1.2``, box-shroud
    ``0.1 sigma 1.0``. Heat capacities (J/K): 800, 1500, 900, 1000, 1200.
    The parameters are the scales of the box-bracket, bracket-radiator and
    panel-box conductances and of the panel and radiator radiative links (an
    emissivity scale). The bracket carries no sensor, so the two series
    conductances act in the steady state only through their sum of resistances
    [EQ-INV-THERM-007]. The numbers are invented for tests; this is not a
    model of any real vehicle.

    Args:
        extra_panel_radiator: Value (W/K) of an additional unmodelled
            conductive panel-radiator coupling, to create a truth model with
            model-form error (``0`` for none).

    Returns:
        The model.

    Raises:
        ValueError: If ``extra_panel_radiator`` is negative.

    References:
        [EQ-INV-THERM-001], [REF-BERGMAN-2011].

    """
    extra = require_finite_real(extra_panel_radiator, name="extra_panel_radiator")
    if extra < 0.0:
        raise ValueError("extra_panel_radiator must be non-negative")
    s = STEFAN_BOLTZMANN
    network = ThermalNetwork(
        DEMO_NODES,
        (800.0, 1500.0, 900.0, 1000.0, 1200.0),
        (
            ThermalLink(0, 1, 0.5),
            ThermalLink(1, 2, 2.0),
            ThermalLink(2, 3, 1.2),
            ThermalLink(1, 4, 1.5),
        ),
        (
            ThermalLink(0, 5, 0.8 * s * 2.0),
            ThermalLink(3, 5, 0.85 * s * 1.2),
            ThermalLink(1, 5, 0.1 * s * 1.0),
        ),
    )
    if extra > 0.0:
        network = network.with_link("conductive", 0, 3, extra)
    return ThermalModel(
        network,
        (
            ThermalParameter(DEMO_PARAMETER_NAMES[0], "conductive", (1,)),
            ThermalParameter(DEMO_PARAMETER_NAMES[1], "conductive", (2,)),
            ThermalParameter(DEMO_PARAMETER_NAMES[2], "conductive", (0,)),
            ThermalParameter(DEMO_PARAMETER_NAMES[3], "radiative", (0, 1)),
        ),
    )


DEMO_BASE_POWER = (40.0, 60.0, 0.0, 0.0, 20.0)
"""Base dissipation (W) of the demonstration case (panel, box, bracket, ...)."""


def demo_calibration_tests(
    scales: Sequence[float] = (0.6, 1.0, 1.4), shroud: float = 150.0
) -> tuple[SteadyTest, ...]:
    """Return steady thermal-balance tests at several power levels.

    Args:
        scales: Multipliers of :data:`DEMO_BASE_POWER`.
        shroud: Shroud temperature (K).

    Returns:
        One :class:`SteadyTest` per scale.

    Raises:
        ValueError: If an input is invalid.

    References:
        [EQ-INV-THERM-002].

    """
    sink = _positive(shroud, "shroud")
    return tuple(
        SteadyTest(
            tuple(_positive(c, "scale") * q for q in DEMO_BASE_POWER),
            (sink,),
            f"x{c:g}",
        )
        for c in scales
    )


def demo_flight_case() -> SteadyTest:
    """Return the untested flight case of the demonstration network.

    Different power distribution and a different (warmer) sink temperature
    than :func:`demo_calibration_tests`.

    Returns:
        A :class:`SteadyTest`.

    References:
        [EQ-INV-THERM-008].

    """
    return SteadyTest((55.0, 75.0, 0.0, 0.0, 10.0), (190.0,), "flight")


def demo_transient_test(shroud: float = 150.0) -> TransientTest:
    """Return a power-step transient of the demonstration network.

    The network starts in the steady state at half of :data:`DEMO_BASE_POWER`
    and the dissipation steps to the full base power for ``t > 0``; ``120``
    backward Euler steps of ``30 s`` (one hour), sampled every fourth step.
    The heat capacity of the unsensed bracket makes the response depend on
    the two series conductances separately [EQ-INV-THERM-007].

    Args:
        shroud: Shroud temperature (K).

    Returns:
        A :class:`TransientTest`.

    Raises:
        ValueError: If ``shroud`` is not positive.

    References:
        [EQ-INV-THERM-003], [EQ-INV-THERM-007].

    """
    sink = _positive(shroud, "shroud")
    low = tuple(0.5 * q for q in DEMO_BASE_POWER)
    return TransientTest(
        step_power(low, DEMO_BASE_POWER, 0.0),
        (sink,),
        30.0,
        120,
        tuple(range(4, 121, 4)),
        None,
        "power step",
    )
