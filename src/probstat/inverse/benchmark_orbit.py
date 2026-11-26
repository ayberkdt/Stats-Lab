"""Simplified two-body orbit determination: a synthetic range-only benchmark (A01).

Experimental (added in 0.28).

:class:`TwoBodyRangeModel` is a :class:`~probstat.inverse.gauss_newton.
DifferentiableForward` map from the INITIAL state ``x = (r0, v0)`` (and
optionally the gravitational parameter ``mu``) of a point mass in an inertial
two-body field to range observations to fixed inertial stations
[EQ-INV-ORBIT-001], [EQ-INV-ORBIT-004]. :func:`orbit_determination_benchmark`
wraps it into the catalog entry A01 as a
:class:`~probstat.inverse.benchmark_extended.NonlinearBenchmark`.

SCOPE AND UNITS. Everything is SYNTHETIC and deliberately simplified: an
inertial frame, a point-mass central body, NO Earth rotation, NO time systems
or light-time, NO perturbations, NO measurement biases unless the caller adds
them, stations that are fixed points in the inertial frame. Units are
CANONICAL: the length unit is the reference orbit radius, the time unit is
chosen so that ``mu = 1`` (the circular period of unit radius is ``2 pi``).
Nothing here claims realistic tracking performance; it is a geometry and
conditioning laboratory (one station on a short arc versus several stations
on a full revolution).

DERIVATIVE CHOICE (option (b), exact). The state and the sensitivity matrix
``Z = d(r, v)/d(x)`` (the state transition matrix ``Phi`` for the state block,
plus the column ``S = d(r, v)/d mu``) are advanced TOGETHER with one fixed-step
classical RK4 applied to the AUGMENTED system
``s' = f(s)``, ``Z' = F(s) Z + [0, F_mu(s)]`` [EQ-INV-ORBIT-002]. Because a
Runge-Kutta scheme commutes with differentiation with respect to the initial
data (the stage derivatives of the stage states are exactly the stage states of
the variational equation), the computed ``Z`` is the EXACT derivative of the
RK4-discretized map ``x -> s_N(x)`` for the SAME step sizes, up to rounding;
it is not merely an ``O(h^4)`` approximation of the derivative of the
continuous flow. Consequently ``jvp`` and ``vjp`` are exact derivatives of
:meth:`TwoBodyRangeModel.value` and the Taylor remainder test of
:func:`~probstat.inverse.gauss_newton.check_derivatives` is second order down
to rounding. The discretization error of the map itself against the true flow
is ``O(h^4)`` and is what the benchmark's finer data generator removes
(``same_discretization=False``).

Range observations are ``rho_ij = ||r(t_i) - s_j||`` ordered time-major and
station-minor, with the Jacobian ``(r - s)' / rho`` times the position rows of
``Z``; a zero range raises ``ValueError`` (no regularization, no clipping)
[EQ-INV-ORBIT-003].

References:
    [REF-TAPLEY-SCHUTZ-BORN-2004], [REF-HAIRER-NORSETT-WANNER-1993],
    [REF-SEBER-WILD-2003], [REF-NOCEDAL-WRIGHT-2006],
    [REF-KAIPIO-SOMERSALO-2007].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.benchmark_extended import NonlinearBenchmark
from probstat.inverse.gauss_newton import ModelCounts
from probstat.linalg import ScaledIdentityCovariance, Vector

_GEOMETRIES = ("well_observed", "single_station_short_arc")


def _finite_real(value: object, name: str) -> float:
    """Validate a finite real number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: object, name: str) -> float:
    """Validate a finite strictly positive real number.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    number = _finite_real(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    """Validate an integer of at least ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _real_sequence(values: object, name: str) -> tuple[float, ...]:
    """Return a tuple of finite reals from a (non-string) sequence.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of reals")
    return tuple(_finite_real(v, name) for v in values)


class TwoBodyRangeModel:
    """Range observations of a two-body orbit as a differentiable forward map.

    The parameters are ``x = (r0, v0)`` of length ``2 d`` (``d`` = 2 or 3),
    followed by ``mu`` when ``estimate_mu=True``. The dynamics are
    ``r' = v``, ``v' = -mu r / ||r||^3`` [EQ-INV-ORBIT-001] integrated from
    ``t = 0`` with ``steps_per_interval`` classical RK4 steps on each of the
    intervals ``[0, t_1], [t_1, t_2], ...``. The observations are the ranges
    to every station at every time, ordered time-major
    [EQ-INV-ORBIT-003]. Derivatives are the exact derivatives of the
    discretized map (the module docstring, option (b)) [EQ-INV-ORBIT-002].

    ``value`` raises ``ValueError`` when an observed range or the orbit radius
    at a step vanishes. With ``estimate_mu=True`` a non-positive ``mu`` in
    ``x`` makes ``value`` return a vector of ``nan`` (a rejected trial for the
    solvers, never repaired), while ``jvp``/``vjp`` raise ``ValueError``.

    Attributes:
        counts: :class:`~probstat.inverse.gauss_newton.ModelCounts` snapshot
            (``derivative_value`` is always ``0``: derivatives are analytic).
        times: Observation times.
        stations: Station positions.
        mu: Fixed gravitational parameter (used when ``estimate_mu=False``).
        steps_per_interval: RK4 steps per observation interval.

    References:
        [EQ-INV-ORBIT-001], [EQ-INV-ORBIT-002], [EQ-INV-ORBIT-003],
        [EQ-INV-ORBIT-004], [REF-HAIRER-NORSETT-WANNER-1993].

    """

    __slots__ = (
        "_cache",
        "_counts",
        "_d",
        "_estimate_mu",
        "_n",
        "mu",
        "stations",
        "steps_per_interval",
        "times",
    )

    def __init__(
        self,
        times: Sequence[float],
        stations: Sequence[Sequence[float]],
        *,
        mu: float = 1.0,
        estimate_mu: bool = False,
        steps_per_interval: int = 16,
    ) -> None:
        """Validate and store the observation design.

        Args:
            times: Strictly increasing positive observation times (canonical
                units); at least one.
            stations: At least one fixed inertial station position; all of the
                same dimension ``d`` (2 or 3), which is the orbit dimension.
            mu: Gravitational parameter (``> 0``), the fixed value when
                ``estimate_mu=False``.
            estimate_mu: Append ``mu`` to the parameters.
            steps_per_interval: RK4 steps per interval (``>= 1``).

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the times are not positive and increasing, the
                stations are inconsistent or of dimension other than 2 or 3,
                or ``mu`` is not positive.

        References:
            [EQ-INV-ORBIT-001], [EQ-INV-ORBIT-003].

        """
        grid = _real_sequence(times, "times")
        if not grid or grid[0] <= 0.0:
            raise ValueError("times must be non-empty and positive")
        if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
            raise ValueError("times must be strictly increasing")
        if isinstance(stations, (str, bytes)) or not isinstance(stations, Sequence):
            raise TypeError("stations must be a sequence of positions")
        pos = tuple(_real_sequence(s, "station") for s in stations)
        if not pos:
            raise ValueError("at least one station is required")
        dimension = len(pos[0])
        if dimension not in (2, 3) or any(len(s) != dimension for s in pos):
            raise ValueError("stations must all have dimension 2 or 3 (the orbit's)")
        if not isinstance(estimate_mu, bool):
            raise TypeError("estimate_mu must be a bool")
        self.times = grid
        self.stations = pos
        self.mu = _positive(mu, "mu")
        self.steps_per_interval = _positive_int(
            steps_per_interval, "steps_per_interval"
        )
        self._d = dimension
        self._estimate_mu = estimate_mu
        self._n = 2 * dimension + (1 if estimate_mu else 0)
        self._counts = [0, 0, 0]
        self._cache: tuple[Vector, list[list[list[float]]], list[list[float]]] | None
        self._cache = None

    @property
    def input_dimension(self) -> int:
        """Return the parameter dimension ``n = 2 d`` (``+ 1`` with ``mu``).

        References:
            [EQ-INV-ORBIT-004].

        """
        return self._n

    @property
    def output_dimension(self) -> int:
        """Return the number of ranges ``m = len(times) * len(stations)``.

        References:
            [EQ-INV-ORBIT-003].

        """
        return len(self.times) * len(self.stations)

    @property
    def counts(self) -> ModelCounts:
        """Return a snapshot of the evaluation counts.

        References:
            [EQ-INV-ORBIT-004].

        """
        value, jvp, vjp = self._counts
        return ModelCounts(value=value, jvp=jvp, vjp=vjp, derivative_value=0)

    def _check_x(self, x: Sequence[float]) -> tuple[Vector, float]:
        """Validate ``x`` and return it with the effective ``mu``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        point = _real_sequence(x, "x")
        if len(point) != self._n:
            raise ValueError(f"x must have length {self._n}")
        mu = point[-1] if self._estimate_mu else self.mu
        return point, mu

    def _rhs(
        self,
        aug: list[float],
        mu: float,
        with_sensitivity: bool,
    ) -> list[float]:
        """Return the derivative of the augmented state ``(s, Z)``.

        ``Z`` is stored row-major (``2 d`` rows of ``n`` entries) after the
        ``2 d`` state entries.

        References:
            [EQ-INV-ORBIT-001], [EQ-INV-ORBIT-002].

        """
        d = self._d
        r = aug[:d]
        v = aug[d : 2 * d]
        rho2 = math.fsum(c * c for c in r)
        if rho2 == 0.0:
            raise ValueError("orbit radius vanished (collision with the center)")
        rho = math.sqrt(rho2)
        inv3 = 1.0 / (rho2 * rho)
        out = list(v) + [-mu * c * inv3 for c in r]
        if not with_sensitivity:
            return out
        n = self._n
        inv5 = inv3 / rho2
        accel = [
            [
                -mu * ((inv3 if i == j else 0.0) - 3.0 * r[i] * r[j] * inv5)
                for j in range(d)
            ]
            for i in range(d)
        ]
        base = 2 * d
        for row in range(d):
            out.extend(aug[base + (d + row) * n : base + (d + row + 1) * n])
        for row in range(d):
            vals = [
                math.fsum(accel[row][k] * aug[base + k * n + col] for k in range(d))
                for col in range(n)
            ]
            if self._estimate_mu:
                vals[n - 1] += -r[row] * inv3
            out.extend(vals)
        return out

    def _step(
        self,
        aug: list[float],
        h: float,
        mu: float,
        with_sensitivity: bool,
    ) -> list[float]:
        """Advance the augmented state by one classical RK4 step.

        References:
            [EQ-INV-ORBIT-002], [REF-HAIRER-NORSETT-WANNER-1993].

        """
        k1 = self._rhs(aug, mu, with_sensitivity)
        k2 = self._rhs(
            [a + 0.5 * h * k for a, k in zip(aug, k1, strict=True)],
            mu,
            with_sensitivity,
        )
        k3 = self._rhs(
            [a + 0.5 * h * k for a, k in zip(aug, k2, strict=True)],
            mu,
            with_sensitivity,
        )
        k4 = self._rhs(
            [a + h * k for a, k in zip(aug, k3, strict=True)], mu, with_sensitivity
        )
        return [
            a + h / 6.0 * (p + 2.0 * q + 2.0 * s + t)
            for a, p, q, s, t in zip(aug, k1, k2, k3, k4, strict=True)
        ]

    def _propagate(
        self, x: Vector, mu: float, with_sensitivity: bool
    ) -> tuple[list[list[float]], list[list[list[float]]]]:
        """Return the states (and sensitivities) at every observation time.

        References:
            [EQ-INV-ORBIT-001], [EQ-INV-ORBIT-002].

        """
        d = self._d
        n = self._n
        aug = list(x[: 2 * d])
        if with_sensitivity:
            for row in range(2 * d):
                aug.extend(1.0 if col == row else 0.0 for col in range(n))
        states: list[list[float]] = []
        sens: list[list[list[float]]] = []
        t_prev = 0.0
        for t in self.times:
            h = (t - t_prev) / self.steps_per_interval
            for _ in range(self.steps_per_interval):
                aug = self._step(aug, h, mu, with_sensitivity)
            t_prev = t
            states.append(aug[: 2 * d])
            if with_sensitivity:
                base = 2 * d
                sens.append(
                    [aug[base + k * n : base + (k + 1) * n] for k in range(2 * d)]
                )
        return states, sens

    def states(self, x: Sequence[float]) -> tuple[Vector, ...]:
        """Return the propagated states ``(r, v)`` at the observation times.

        Args:
            x: Parameters (length ``input_dimension``).

        Returns:
            One ``2 d`` tuple per observation time.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length or the orbit hits the
                center.

        References:
            [EQ-INV-ORBIT-001].

        """
        point, mu = self._check_x(x)
        states, _ = self._propagate(point, mu, False)
        return tuple(tuple(s) for s in states)

    def sensitivities(self, x: Sequence[float]) -> tuple[tuple[Vector, ...], ...]:
        """Return ``Z_i = d(r, v)(t_i) / d x`` for every observation time.

        The leading ``2 d`` columns form the state transition matrix
        ``Phi(t_i, 0)`` and, with ``estimate_mu``, the last column is the
        ``mu``-sensitivity ``S`` [EQ-INV-ORBIT-002].

        Args:
            x: Parameters (length ``input_dimension``).

        Returns:
            For each time, ``2 d`` rows of length ``input_dimension``.

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length, ``mu <= 0`` or the
                orbit hits the center.

        References:
            [EQ-INV-ORBIT-002].

        """
        point, mu = self._check_x(x)
        if mu <= 0.0:
            raise ValueError("mu must be positive")
        _, sens = self._propagate(point, mu, True)
        return tuple(tuple(tuple(row) for row in z) for z in sens)

    def _range_rows(
        self, state: Sequence[float], station: Sequence[float]
    ) -> tuple[float, list[float]]:
        """Return the range and the unit line-of-sight ``(r - s)/rho``.

        References:
            [EQ-INV-ORBIT-003].

        """
        diff = [a - b for a, b in zip(state[: self._d], station, strict=True)]
        rho = math.sqrt(math.fsum(c * c for c in diff))
        if rho == 0.0:
            raise ValueError("zero range: the orbit passes through a station")
        return rho, [c / rho for c in diff]

    def value(self, x: Sequence[float]) -> Vector:
        """Return the ranges ``rho_ij = ||r(t_i) - s_j||``.

        Args:
            x: Parameters (length ``input_dimension``).

        Returns:
            ``m`` ranges, time-major and station-minor (a vector of ``nan``
            when ``estimate_mu`` and the ``mu`` in ``x`` is not positive).

        Raises:
            TypeError: If ``x`` is not a real sequence.
            ValueError: If ``x`` has the wrong length, the orbit hits the
                center or a range vanishes.

        References:
            [EQ-INV-ORBIT-003].

        """
        point, mu = self._check_x(x)
        self._counts[0] += 1
        if mu <= 0.0:
            return (math.nan,) * self.output_dimension
        states, _ = self._propagate(point, mu, False)
        return tuple(
            self._range_rows(state, station)[0]
            for state in states
            for station in self.stations
        )

    def _linearization(
        self, x: Vector, mu: float
    ) -> tuple[list[list[list[float]]], list[list[float]]]:
        """Return (sensitivities, line-of-sight rows) cached for the last ``x``.

        References:
            [EQ-INV-ORBIT-002], [EQ-INV-ORBIT-003].

        """
        if mu <= 0.0:
            raise ValueError("mu must be positive")
        if self._cache is not None and self._cache[0] == x:
            return self._cache[1], self._cache[2]
        states, sens = self._propagate(x, mu, True)
        rows: list[list[float]] = []
        for state in states:
            for station in self.stations:
                rows.append(self._range_rows(state, station)[1])
        self._cache = (x, sens, rows)
        return sens, rows

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``J(x) v`` from the position rows of the sensitivities.

        Args:
            x: Linearization point (length ``input_dimension``).
            v: Direction (same length).

        Returns:
            ``J(x) v`` (length ``m``): for observation ``(i, j)`` the value
            ``u_ij' [Z_i v]_position`` with ``u_ij = (r - s_j)/rho_ij``.

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, ``mu <= 0``, or a range
                vanishes.

        References:
            [EQ-INV-ORBIT-002], [EQ-INV-ORBIT-003].

        """
        point, mu = self._check_x(x)
        direction = _real_sequence(v, "v")
        if len(direction) != self._n:
            raise ValueError(f"v must have length {self._n}")
        sens, rows = self._linearization(point, mu)
        self._counts[1] += 1
        d = self._d
        ns = len(self.stations)
        out: list[float] = []
        for i, z in enumerate(sens):
            dr = [
                math.fsum(z[k][c] * direction[c] for c in range(self._n))
                for k in range(d)
            ]
            for j in range(ns):
                u = rows[i * ns + j]
                out.append(math.fsum(a * b for a, b in zip(u, dr, strict=True)))
        return tuple(out)

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``J(x)' w`` through the transposed sensitivities.

        No dense finite-difference Jacobian is formed: for each time the
        position-space vector ``g_i = sum_j w_ij u_ij`` is mapped by
        ``Z_i' `` (position rows only).

        Args:
            x: Linearization point (length ``input_dimension``).
            w: Adjoint direction (length ``m``).

        Returns:
            ``J(x)' w`` (length ``input_dimension``).

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: If a length is wrong, ``mu <= 0``, or a range
                vanishes.

        References:
            [EQ-INV-ORBIT-002], [EQ-INV-ORBIT-003].

        """
        point, mu = self._check_x(x)
        weights = _real_sequence(w, "w")
        if len(weights) != self.output_dimension:
            raise ValueError(f"w must have length {self.output_dimension}")
        sens, rows = self._linearization(point, mu)
        self._counts[2] += 1
        d = self._d
        ns = len(self.stations)
        terms: list[list[float]] = [[] for _ in range(self._n)]
        for i, z in enumerate(sens):
            g = [
                math.fsum(weights[i * ns + j] * rows[i * ns + j][k] for j in range(ns))
                for k in range(d)
            ]
            for c in range(self._n):
                terms[c].append(math.fsum(g[k] * z[k][c] for k in range(d)))
        return tuple(math.fsum(t) for t in terms)

    def energy(self, state: Sequence[float], mu: float | None = None) -> float:
        """Return the specific orbital energy ``|v|^2 / 2 - mu / |r|``.

        Args:
            state: ``(r, v)`` of length ``2 d``.
            mu: Gravitational parameter (default: the model's fixed ``mu``).

        Returns:
            The energy, conserved by the continuous flow and, up to the
            ``O(h^4)`` integrator error, by the discretization.

        Raises:
            TypeError: If ``state`` is not a real sequence.
            ValueError: If the length is wrong, ``mu`` is not positive or
                ``r = 0``.

        References:
            [EQ-INV-ORBIT-001].

        """
        s = _real_sequence(state, "state")
        if len(s) != 2 * self._d:
            raise ValueError(f"state must have length {2 * self._d}")
        gm = self.mu if mu is None else _positive(mu, "mu")
        radius = math.sqrt(math.fsum(c * c for c in s[: self._d]))
        if radius == 0.0:
            raise ValueError("orbit radius vanished")
        kinetic = 0.5 * math.fsum(c * c for c in s[self._d :])
        return kinetic - gm / radius


def _reference_orbit(dimension: int) -> Vector:
    """Return the truth initial state: a mildly eccentric prograde orbit.

    ``r0 = (1, 0[, 0])`` and a speed of ``1.05`` (the circular speed is ``1``
    for ``mu = 1``), giving eccentricity ``~0.10`` and period ``~6.9``; in 3D
    the velocity is tilted by an inclination of ``0.3`` rad.

    References:
        [EQ-INV-ORBIT-001].

    """
    speed = 1.05
    if dimension == 2:
        return (1.0, 0.0, 0.0, speed)
    return (1.0, 0.0, 0.0, 0.0, speed * math.cos(0.3), speed * math.sin(0.3))


def _geometry(
    dimension: int, geometry: str
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return the observation times and the stations of a named geometry.

    ``"well_observed"``: three stations at distance ``~4`` around the orbit and
    8 times over about one revolution. ``"single_station_short_arc"``: one
    station and 8 times on roughly a tenth of a revolution.

    References:
        [EQ-INV-ORBIT-003].

    """
    stations: tuple[tuple[float, ...], ...]
    if geometry == "well_observed":
        times = tuple(0.8 * (i + 1) for i in range(8))
        if dimension == 2:
            stations = ((4.0, 0.5), (-1.0, 4.0), (-3.5, -2.5))
        else:
            stations = ((4.0, 0.5, 1.0), (-1.0, 4.0, -1.0), (-3.5, -2.5, 2.0))
    else:
        times = tuple(0.1 * (i + 1) for i in range(8))
        stations = ((4.0, 0.5),) if dimension == 2 else ((4.0, 0.5, 1.0),)
    return times, stations


def orbit_determination_benchmark(
    *,
    dimension: int = 2,
    geometry: str = "well_observed",
    noise_sd: float = 1e-3,
    seed: int | RandomStream | None = 11,
    estimate_mu: bool = False,
    steps_per_interval: int = 12,
    data_refinement: int = 4,
) -> NonlinearBenchmark:
    """Return A01: simplified synthetic range-only orbit determination.

    The unknown is the initial state ``(r0, v0)`` (and ``mu`` with
    ``estimate_mu``) of a mildly eccentric orbit in canonical units
    (``mu = 1``, reference radius 1, period ``~6.9``); the data are noisy
    ranges to fixed inertial stations [EQ-INV-ORBIT-005]. Two geometries:

    * ``"well_observed"``: 3 stations around the orbit, 8 times over one
      revolution (``m = 24``);
    * ``"single_station_short_arc"``: 1 station, 8 times over about a
      sixth of a revolution (``m = 8``); the range-only short arc leaves
      the cross-line-of-sight directions weakly determined (pathology
      ``"weak_observability"``).

    The solver's model integrates with ``steps_per_interval`` RK4 steps per
    interval, the data are generated with ``data_refinement`` times as many
    (``same_discretization=False``); ``data_refinement=1`` gives the inverse
    crime (algebraic verification only). Noise is iid Gaussian with the
    stated absolute standard deviation (canonical length units); it is not
    rescaled and the realized norm is reported. Truth is withheld. This is a
    synthetic geometry laboratory and makes no claim about real tracking.

    Args:
        dimension: Orbit dimension, 2 or 3.
        geometry: ``"well_observed"`` or ``"single_station_short_arc"``.
        noise_sd: Range noise standard deviation (``> 0``).
        seed: Root seed (only the noise is random).
        estimate_mu: Also estimate ``mu`` (truth ``1``).
        steps_per_interval: Solver-model RK4 steps per interval.
        data_refinement: Data generator steps multiplier (``>= 1``).

    Returns:
        :class:`~probstat.inverse.benchmark_extended.NonlinearBenchmark`
        with identifier ``"A01"``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-ORBIT-005], [REF-SEBER-WILD-2003],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    if isinstance(dimension, bool) or dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if geometry not in _GEOMETRIES:
        raise ValueError(f"geometry must be one of {_GEOMETRIES}")
    sigma = _positive(noise_sd, "noise_sd")
    refinement = _positive_int(data_refinement, "data_refinement")
    steps = _positive_int(steps_per_interval, "steps_per_interval")
    times, stations = _geometry(dimension, geometry)
    model = TwoBodyRangeModel(
        times,
        stations,
        estimate_mu=estimate_mu,
        steps_per_interval=steps,
    )
    generator_model = TwoBodyRangeModel(
        times,
        stations,
        estimate_mu=estimate_mu,
        steps_per_interval=steps * refinement,
    )
    truth = _reference_orbit(dimension) + ((1.0,) if estimate_mu else ())
    exact = generator_model.value(truth)
    stream = resolve_stream(seed).child("A01").child("noise")
    draws_rng = stream.generator()
    draws = [draws_rng.gauss(0.0, 1.0) for _ in exact]
    data = tuple(b + sigma * e for b, e in zip(exact, draws, strict=True))
    pathologies = ("nonlinear",) + (
        ("weak_observability", "ill_conditioned")
        if geometry == "single_station_short_arc"
        else ()
    )
    return NonlinearBenchmark(
        identifier="A01",
        name=(
            f"two-body range orbit determination ({dimension}D, {geometry}, "
            f"{len(times)} times x {len(stations)} stations)"
        ),
        model=model,
        data=data,
        noise=ScaledIdentityCovariance(len(exact), sigma * sigma),
        noise_level=sigma,
        realized_noise_norm=sigma * math.hypot(*draws),
        truth=truth,
        exact_data=exact,
        same_discretization=refinement == 1,
        prior_mean=None,
        prior_covariance=None,
        pathologies=pathologies,
        seeds=MappingProxyType({"noise": "A01/noise"}),
    )
