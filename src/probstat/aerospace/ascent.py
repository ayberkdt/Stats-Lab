"""Planar launch-vehicle ascent with dispersion analysis.

A planar (two-dimensional) point-mass ascent model for Monte Carlo dispersion
studies of a multi-stage launch vehicle. All quantities are SI (m, s, kg, N,
Pa, K, rad).

Model (Curtis, *Orbital Mechanics for Engineering Students*; Sutton and
Biblarz, *Rocket Propulsion Elements*; U.S. Standard Atmosphere 1976):

- State ``(theta, h, v, gamma, m)``: downrange central angle, altitude above a
  spherical Earth of radius ``R_E = 6 378 137 m``, speed relative to the
  (non-rotating) Earth, flight-path angle above the local horizontal, mass.
- Forces: thrust along the velocity vector (zero angle of attack, a gravity
  turn), drag opposing the velocity, inverse-square gravity ``mu / r^2``. Lift
  and wind are absent; the Earth does not rotate, so inertial and relative
  velocity coincide.
- Thrust ``F = F_vac - p_a A_e`` with constant mass flow
  ``mdot = F_vac / (Isp_vac g0)``; the delivered specific impulse is
  ``F / (mdot g0)``. Stages burn in sequence, each until its propellant is
  exhausted (burn time ``m_p / mdot`` is known in closed form, so staging and
  burnout are event times hit exactly); a stage's dry mass is jettisoned at its
  burnout, optionally followed by a coast.
- Atmosphere: the U.S. Standard Atmosphere 1976 for geometric altitudes up to
  86 km (piecewise layers with the exact lapse-rate formulas), and an
  isothermal exponential extension above 86 km at the 86 km temperature with
  constant molecular weight. The extension is *not* part of the 1976 model;
  at those altitudes the density is below about 1e-5 kg/m^3 and drag is
  negligible for the vehicles considered here.
- Drag coefficient: constant or tabulated against Mach number (linear
  interpolation; a Mach number outside the table is an error).
- Guidance: vertical flight until ``kick_time``, then an instantaneous pitch-over
  that rotates the velocity vector by ``kick_angle`` from the vertical, then a
  gravity turn ``dgamma/dt = (v/r - g/v) cos(gamma)``.
- Integration: classical fourth-order Runge-Kutta
  (:func:`probstat.numerics.integrate_fixed_step`) on each flight segment with
  the segment length divided into equal steps no longer than ``max_step``.
  Stage burnout and kick times fall exactly on step boundaries, so they carry
  no event-location error. Ground impact is located by bisection of a single
  sub-step from the last state above ground to ``|h| < 1e-6`` m, so the
  impact time inherits only the local RK4 error ``O(max_step^5)``.
  Measured accuracy for the illustrative two-stage vehicle of the tests (perigee
  altitude against a 0.25 s reference): about 38 m at ``max_step = 4`` s, 7.6 m
  at 2 s and 0.8 m at 1 s. The drag table and the atmosphere-layer kinks keep
  the observed order below the formal RK4 order of 4 in the integrand.

Outputs at burnout: altitude, speed, flight-path angle, the osculating orbital
elements (semi-major axis, eccentricity, perigee and apogee altitude) and an
insertion indicator against a target window.

Limits. Illustrative only: the vehicle numbers in this library are synthetic,
not those of any real launcher. A non-rotating spherical Earth omits the
Earth-rotation speed bonus, ``J2`` and oblateness, so perigee altitudes are
valid only for the idealized planet. Gravity-turn flight has no pitch program
beyond the kick and no angle of attack, so aerodynamic loads are not
representative. Linear (RSS) dispersion budgets, see
:func:`dispersion_budget`, fail for nonlinear responses, interactions and
non-Gaussian inputs.

Experimental (added in 0.28).

References:
    [REF-CURTIS-2014], [REF-SUTTON-BIBLARZ-2017], [REF-US-STANDARD-ATMOSPHERE-1976].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from probstat.aerospace.liquid_engine import EnginePerformance
from probstat.foundations.random_streams import resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.numerics import integrate_fixed_step
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "EARTH_MU",
    "EARTH_RADIUS",
    "STANDARD_GRAVITY",
    "AscentResult",
    "AtmosphereState",
    "DispersionBudget",
    "DispersionEllipse",
    "DispersionInterval",
    "DragModel",
    "GuidanceSpec",
    "InsertionWindow",
    "OrbitalElements",
    "StageSpec",
    "VehicleSpec",
    "ascent_model",
    "atmosphere",
    "coast_arc",
    "dispersion_budget",
    "dispersion_ellipse",
    "dispersion_intervals",
    "geometric_from_geopotential",
    "orbital_elements",
    "rocket_equation_delta_v",
    "simulate_ascent",
    "specific_energy",
    "stage_from_engine",
    "vertical_burn_closed_form",
]

EARTH_MU = 3.986004418e14
"""Earth's gravitational parameter ``mu`` in m^3/s^2 (WGS-84 value)."""

EARTH_RADIUS = 6378137.0
"""Spherical-Earth radius in m (WGS-84 equatorial radius)."""

STANDARD_GRAVITY = 9.80665
"""Standard gravity ``g0`` in m/s^2 (exact, conventional)."""

# U.S. Standard Atmosphere 1976 constants (SI).
_R_STAR = 8314.32  # J/(kmol K), universal gas constant of the 1976 model
_M_AIR = 28.9644  # kg/kmol, sea-level mean molecular weight
_P0 = 101325.0  # Pa
_R0_GEOPOTENTIAL = 6356766.0  # m, effective Earth radius for geopotential altitude
_GAMMA_AIR = 1.4
_AIR_R = _R_STAR / _M_AIR  # J/(kg K) specific gas constant, 287.053
_SEA_LEVEL_TEMPERATURE = 288.15  # K
_TOP_GEOPOTENTIAL = 84852.0  # m, geopotential altitude of the 86 km geometric limit

# (base geopotential altitude m, base temperature K, lapse rate K/m)
_LAYER_DEFS: tuple[tuple[float, float, float], ...] = (
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.002),
)
_TOP_TEMPERATURE = 214.65 - 0.002 * (_TOP_GEOPOTENTIAL - 71000.0)  # 186.946 K

_GM_OVER_R = STANDARD_GRAVITY * _M_AIR / _R_STAR


def _layer_table() -> tuple[tuple[float, float, float, float], ...]:
    """Build ``(H_b, T_b, L_b, P_b)`` for every layer, base pressures chained.

    Returns:
        One tuple per layer, including the isothermal extension above the
        86 km geometric altitude as the last entry.

    References:
        [EQ-AERO-ASC-001], [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    rows: list[tuple[float, float, float, float]] = []
    pressure = _P0
    defs = (*_LAYER_DEFS, (_TOP_GEOPOTENTIAL, _TOP_TEMPERATURE, 0.0))
    for index, (base, temp, lapse) in enumerate(defs):
        rows.append((base, temp, lapse, pressure))
        if index + 1 < len(defs):
            top = defs[index + 1][0]
            if lapse != 0.0:
                t_top = temp + lapse * (top - base)
                pressure *= (temp / t_top) ** (_GM_OVER_R / lapse)
            else:
                pressure *= math.exp(-_GM_OVER_R * (top - base) / temp)
    return tuple(rows)


_LAYERS = _layer_table()


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    """Atmospheric state at one altitude.

    Args:
        altitude: Geometric altitude, m.
        temperature: Kinetic temperature, K.
        pressure: Static pressure, Pa.
        density: Density, kg/m^3.
        speed_of_sound: ``sqrt(1.4 R T)``, m/s.
        extended: ``True`` above 86 km, where the isothermal extension (not
            the 1976 model) is used.

    References:
        [EQ-AERO-ASC-001], [REF-US-STANDARD-ATMOSPHERE-1976].

    """

    altitude: float
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float
    extended: bool


def geometric_from_geopotential(geopotential: float) -> float:
    """Convert geopotential altitude to geometric altitude.

    ``h = r0 H / (r0 - H)`` with ``r0 = 6 356 766 m`` (1976 model).

    Args:
        geopotential: Geopotential altitude ``H`` in m, ``0 <= H < r0``.

    Returns:
        Geometric altitude in m.

    Raises:
        TypeError: If the input is not a real number.
        ValueError: If it is negative or not below ``r0``.

    Equation:
        [EQ-AERO-ASC-001].

    References:
        [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    value = require_finite_real(geopotential, name="geopotential")
    if value < 0.0 or value >= _R0_GEOPOTENTIAL:
        raise ValueError("geopotential must satisfy 0 <= H < 6356766 m")
    return _R0_GEOPOTENTIAL * value / (_R0_GEOPOTENTIAL - value)


def _atmosphere_raw(altitude: float) -> tuple[float, float, float]:
    """Return ``(T, p, rho)`` for a validated non-negative geometric altitude.

    Args:
        altitude: Geometric altitude in m, already validated.

    Returns:
        Temperature (K), pressure (Pa), density (kg/m^3).

    References:
        [EQ-AERO-ASC-001], [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    geopotential = _R0_GEOPOTENTIAL * altitude / (_R0_GEOPOTENTIAL + altitude)
    base, t_base, lapse, p_base = _LAYERS[0]
    for row in _LAYERS:
        if geopotential >= row[0]:
            base, t_base, lapse, p_base = row
        else:
            break
    if lapse != 0.0:
        temperature = t_base + lapse * (geopotential - base)
        pressure = p_base * (t_base / temperature) ** (_GM_OVER_R / lapse)
    else:
        temperature = t_base
        pressure = p_base * math.exp(-_GM_OVER_R * (geopotential - base) / t_base)
    return temperature, pressure, pressure / (_AIR_R * temperature)


def atmosphere(altitude: float) -> AtmosphereState:
    """Evaluate the U.S. Standard Atmosphere 1976 at a geometric altitude.

    With geopotential altitude ``H = r0 h / (r0 + h)`` and layer base values
    ``(H_b, T_b, L_b, p_b)``, ``T = T_b + L_b (H - H_b)`` and

    ``p = p_b (T_b / T)^(g0 M / (R* L_b))`` for ``L_b != 0``,
    ``p = p_b exp(-g0 M (H - H_b) / (R* T_b))`` for ``L_b = 0``,

    ``rho = p M / (R* T)``. Seven layers cover 0 to 86 km geometric; above
    that, an isothermal extension at the 86 km temperature (not part of the
    1976 model, flagged by ``extended``).

    Args:
        altitude: Geometric altitude in m, ``>= 0``.

    Returns:
        The :class:`AtmosphereState`.

    Raises:
        TypeError: If ``altitude`` is not a real number.
        ValueError: If it is negative or not finite.

    Equation:
        [EQ-AERO-ASC-001].

    References:
        [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    value = require_finite_real(altitude, name="altitude")
    if value < 0.0:
        raise ValueError("altitude must be non-negative")
    temperature, pressure, density = _atmosphere_raw(value)
    sound = math.sqrt(_GAMMA_AIR * _AIR_R * temperature)
    extended = _R0_GEOPOTENTIAL * value / (_R0_GEOPOTENTIAL + value) > _TOP_GEOPOTENTIAL
    return AtmosphereState(value, temperature, pressure, density, sound, extended)


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive real.

    Args:
        value: Candidate number.
        name: Name used in error messages.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite and strictly positive.

    References:
        [REF-CURTIS-2014]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative real.

    Args:
        value: Candidate number.
        name: Name used in error messages.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite and non-negative.

    References:
        [REF-CURTIS-2014]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class DragModel:
    """Drag coefficient, constant or tabulated against Mach number.

    Build with :meth:`constant` or :meth:`tabulated`. A tabulated model is
    interpolated linearly and raises outside its Mach range (no
    extrapolation).

    Args:
        mach: Strictly increasing Mach numbers (empty for a constant model).
        coefficients: Drag coefficients (one entry for a constant model).

    References:
        [EQ-AERO-ASC-004], [REF-CURTIS-2014].

    """

    mach: tuple[float, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the table.

        Raises:
            ValueError: If the table is inconsistent, not increasing, or has a
                non-positive coefficient.

        References:
            [EQ-AERO-ASC-004]. No equation is evaluated.

        """
        if not self.coefficients:
            raise ValueError("coefficients must not be empty")
        if self.mach:
            if len(self.mach) != len(self.coefficients) or len(self.mach) < 2:
                raise ValueError("a table needs matching mach and coefficients (>= 2)")
            if any(b <= a for a, b in zip(self.mach, self.mach[1:], strict=False)):
                raise ValueError("mach must be strictly increasing")
            if self.mach[0] < 0.0:
                raise ValueError("mach must be non-negative")
        elif len(self.coefficients) != 1:
            raise ValueError("a constant model has exactly one coefficient")
        if any(not (math.isfinite(c) and c >= 0.0) for c in self.coefficients):
            raise ValueError("coefficients must be finite and non-negative")

    @classmethod
    def constant(cls, coefficient: float) -> DragModel:
        """Mach-independent drag coefficient.

        Args:
            coefficient: ``C_D >= 0``.

        Returns:
            A constant model.

        Raises:
            TypeError: If the value is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-ASC-004].

        References:
            [REF-CURTIS-2014].

        """
        return cls((), (_non_negative(coefficient, "coefficient"),))

    @classmethod
    def tabulated(cls, points: Sequence[tuple[float, float]]) -> DragModel:
        """Drag coefficient from ``(Mach, C_D)`` pairs.

        Args:
            points: At least two pairs with strictly increasing Mach.

        Returns:
            A tabulated model.

        Raises:
            TypeError: If entries are not numbers.
            ValueError: If the table is invalid.

        Equation:
            [EQ-AERO-ASC-004].

        References:
            [REF-CURTIS-2014].

        """
        pairs = [tuple(p) for p in points]
        mach = tuple(require_finite_real(p[0], name="mach") for p in pairs)
        coefficients = tuple(require_finite_real(p[1], name="C_D") for p in pairs)
        return cls(mach, coefficients)

    def at(self, mach: float) -> float:
        """Return ``C_D`` at a Mach number.

        Args:
            mach: Mach number, ``>= 0``.

        Returns:
            The drag coefficient.

        Raises:
            ValueError: If a tabulated model is queried outside its range.

        Equation:
            [EQ-AERO-ASC-004].

        References:
            [REF-CURTIS-2014].

        """
        if not self.mach:
            return self.coefficients[0]
        grid = self.mach
        if mach < grid[0] or mach > grid[-1]:
            raise ValueError(
                f"Mach {mach:.4g} is outside the drag table [{grid[0]}, {grid[-1]}]"
            )
        lo, hi = 0, len(grid) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if grid[mid] <= mach:
                lo = mid
            else:
                hi = mid
        w = (mach - grid[lo]) / (grid[hi] - grid[lo])
        return (1.0 - w) * self.coefficients[lo] + w * self.coefficients[hi]

    def scaled(self, factor: float) -> DragModel:
        """Return the model with every coefficient multiplied by ``factor``.

        Args:
            factor: Positive scale.

        Returns:
            The scaled model.

        Raises:
            ValueError: If ``factor`` is not positive and finite.

        Equation:
            [EQ-AERO-ASC-004].

        References:
            [REF-CURTIS-2014].

        """
        k = _positive(factor, "factor")
        return DragModel(self.mach, tuple(k * c for c in self.coefficients))


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One propulsive stage (illustrative numbers only).

    Args:
        dry_mass: Inert mass jettisoned at burnout (kg); for the last stage it
            stays with the payload.
        propellant_mass: Usable propellant (kg), ``> 0``.
        vacuum_thrust: ``F_vac`` (N), ``> 0``.
        vacuum_isp: ``Isp_vac`` (s), ``> 0``; ``mdot = F_vac / (Isp_vac g0)``.
        exit_area: Nozzle exit area ``A_e`` (m^2); the ambient-pressure
            thrust loss is ``p_a A_e``. Must satisfy ``F_vac > p_0 A_e``.
        reference_area: Drag reference area ``S`` (m^2), ``> 0``.
        coast_after: Coast time between this burnout and the next ignition
            (s), ``>= 0``; ignored for the last stage.

    References:
        [EQ-AERO-ASC-003], [REF-SUTTON-BIBLARZ-2017].

    """

    dry_mass: float
    propellant_mass: float
    vacuum_thrust: float
    vacuum_isp: float
    exit_area: float
    reference_area: float
    coast_after: float = 0.0

    def __post_init__(self) -> None:
        """Validate the stage.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If a field is out of range or the engine cannot
                produce positive thrust at sea level.

        References:
            [EQ-AERO-ASC-003]. No equation is evaluated.

        """
        _positive(self.dry_mass, "dry_mass")
        _positive(self.propellant_mass, "propellant_mass")
        _positive(self.vacuum_thrust, "vacuum_thrust")
        _positive(self.vacuum_isp, "vacuum_isp")
        _non_negative(self.exit_area, "exit_area")
        _positive(self.reference_area, "reference_area")
        _non_negative(self.coast_after, "coast_after")
        if self.vacuum_thrust <= _P0 * self.exit_area:
            raise ValueError("vacuum_thrust must exceed p0 * exit_area")

    @property
    def mass_flow(self) -> float:
        """Propellant mass flow ``F_vac / (Isp_vac g0)`` in kg/s.

        References:
            [EQ-AERO-ASC-003].

        """
        return self.vacuum_thrust / (self.vacuum_isp * STANDARD_GRAVITY)

    @property
    def burn_time(self) -> float:
        """Burn time ``m_p / mdot`` in s.

        References:
            [EQ-AERO-ASC-003].

        """
        return self.propellant_mass / self.mass_flow

    def thrust(self, ambient_pressure: float) -> float:
        """Thrust ``F_vac - p_a A_e`` in N.

        Args:
            ambient_pressure: ``p_a`` in Pa, ``>= 0``.

        Returns:
            The thrust.

        Raises:
            ValueError: If the pressure is negative or the result is not
                positive.

        Equation:
            [EQ-AERO-ASC-003].

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        pressure = _non_negative(ambient_pressure, "ambient_pressure")
        value = self.vacuum_thrust - pressure * self.exit_area
        if value <= 0.0:
            raise ValueError("thrust is not positive at this ambient pressure")
        return value

    def specific_impulse(self, ambient_pressure: float) -> float:
        """Delivered ``Isp = F / (mdot g0)`` in s at an ambient pressure.

        Args:
            ambient_pressure: ``p_a`` in Pa.

        Returns:
            The specific impulse.

        Raises:
            ValueError: If the thrust is not positive.

        Equation:
            [EQ-AERO-ASC-003].

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return self.thrust(ambient_pressure) / (self.mass_flow * STANDARD_GRAVITY)


def stage_from_engine(
    performance: EnginePerformance,
    *,
    exit_area: float,
    dry_mass: float,
    propellant_mass: float,
    reference_area: float,
    engines: int = 1,
    coast_after: float = 0.0,
) -> StageSpec:
    """Build a stage from ideal-rocket engine performance.

    Takes the vacuum thrust and vacuum specific impulse computed by
    :func:`probstat.aerospace.liquid_engine.ideal_rocket_performance` and
    clusters ``engines`` identical engines.

    Args:
        performance: Engine performance with ``vacuum_thrust`` and
            ``vacuum_specific_impulse``.
        exit_area: Total exit area of all engines (m^2).
        dry_mass: Stage inert mass (kg).
        propellant_mass: Propellant (kg).
        reference_area: Drag reference area (m^2).
        engines: Number of identical engines, ``>= 1``.
        coast_after: Coast after burnout (s).

    Returns:
        The :class:`StageSpec`.

    Raises:
        TypeError: If ``performance`` is not an ``EnginePerformance`` or
            ``engines`` is not an integer.
        ValueError: If a value is out of range.

    Equation:
        [EQ-AERO-ASC-003].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(performance, EnginePerformance):
        raise TypeError("performance must be an EnginePerformance")
    if isinstance(engines, bool) or not isinstance(engines, int) or engines < 1:
        raise TypeError("engines must be a positive integer")
    return StageSpec(
        dry_mass=dry_mass,
        propellant_mass=propellant_mass,
        vacuum_thrust=engines * performance.vacuum_thrust,
        vacuum_isp=performance.vacuum_specific_impulse,
        exit_area=exit_area,
        reference_area=reference_area,
        coast_after=coast_after,
    )


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    """A multi-stage vehicle with a payload (illustrative numbers only).

    Args:
        stages: One or more :class:`StageSpec`, first stage first.
        payload_mass: Mass carried above the last stage's dry mass (kg), ``>= 0``.

    References:
        [EQ-AERO-ASC-003], [EQ-AERO-ASC-006].

    """

    stages: tuple[StageSpec, ...]
    payload_mass: float

    def __post_init__(self) -> None:
        """Validate the vehicle.

        Raises:
            TypeError: If a stage is not a ``StageSpec``.
            ValueError: If there is no stage or the payload mass is invalid.

        References:
            [EQ-AERO-ASC-003]. No equation is evaluated.

        """
        if not self.stages:
            raise ValueError("a vehicle needs at least one stage")
        if not all(isinstance(s, StageSpec) for s in self.stages):
            raise TypeError("stages must be StageSpec instances")
        _non_negative(self.payload_mass, "payload_mass")

    @property
    def liftoff_mass(self) -> float:
        """Total mass at liftoff in kg.

        References:
            [EQ-AERO-ASC-006].

        """
        return self.payload_mass + sum(
            s.dry_mass + s.propellant_mass for s in self.stages
        )


@dataclass(frozen=True, slots=True)
class GuidanceSpec:
    """Vertical rise, pitch-over kick, then gravity turn.

    Args:
        kick_time: Time of the kick (s), ``>= 0``; ``math.inf`` means purely
            vertical flight.
        kick_angle: Rotation of the velocity vector from the vertical at the
            kick (rad), ``0 <= angle < pi/2``.

    References:
        [EQ-AERO-ASC-002], [REF-CURTIS-2014].

    """

    kick_time: float
    kick_angle: float

    def __post_init__(self) -> None:
        """Validate the guidance.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If the kick time is negative or the angle is outside
                ``[0, pi/2)``.

        References:
            [EQ-AERO-ASC-002]. No equation is evaluated.

        """
        if isinstance(self.kick_time, bool) or not isinstance(
            self.kick_time, int | float
        ):
            raise TypeError("kick_time must be a real number")
        if math.isnan(self.kick_time) or self.kick_time < 0.0:
            raise ValueError("kick_time must be non-negative (or inf)")
        angle = require_finite_real(self.kick_angle, name="kick_angle")
        if not 0.0 <= angle < 0.5 * math.pi:
            raise ValueError("kick_angle must satisfy 0 <= angle < pi/2")

    @classmethod
    def vertical(cls) -> GuidanceSpec:
        """Purely vertical flight (no kick).

        Returns:
            A guidance with ``kick_time = inf``.

        References:
            [EQ-AERO-ASC-002].

        """
        return cls(math.inf, 0.0)


@dataclass(frozen=True, slots=True)
class InsertionWindow:
    """Target window for a successful insertion.

    Success requires ``perigee_altitude >= min_perigee`` and a bound orbit with
    ``apogee_min <= apogee_altitude <= apogee_max``.

    Args:
        min_perigee: Minimum perigee altitude (m).
        apogee_min: Lower apogee altitude bound (m).
        apogee_max: Upper apogee altitude bound (m).

    References:
        [EQ-AERO-ASC-005].

    """

    min_perigee: float
    apogee_min: float
    apogee_max: float

    def __post_init__(self) -> None:
        """Validate the window.

        Raises:
            TypeError: If a bound is not a real number.
            ValueError: If the bounds are not ordered.

        References:
            [EQ-AERO-ASC-005]. No equation is evaluated.

        """
        low = require_finite_real(self.min_perigee, name="min_perigee")
        a_lo = require_finite_real(self.apogee_min, name="apogee_min")
        a_hi = require_finite_real(self.apogee_max, name="apogee_max")
        if not (a_lo <= a_hi):
            raise ValueError("apogee_min must not exceed apogee_max")
        if a_lo < low:
            raise ValueError("apogee_min must not be below min_perigee")


@dataclass(frozen=True, slots=True)
class OrbitalElements:
    """Osculating Keplerian elements of a planar state.

    Args:
        specific_energy: ``E = v^2/2 - mu/r`` (J/kg).
        angular_momentum: ``h = r v cos(gamma)`` (m^2/s).
        semi_major_axis: ``a = -mu / (2E)`` (m); ``inf`` at ``E = 0``, negative
            on a hyperbola.
        eccentricity: ``e = sqrt(1 + 2 E h^2 / mu^2)``.
        perigee_radius: ``r_p = p / (1 + e)`` (m), ``p = h^2/mu``.
        apogee_radius: ``r_a = p / (1 - e)`` (m); ``inf`` when ``e >= 1``.
        perigee_altitude: ``r_p - R_E`` (m); negative when the orbit
            intersects the planet.
        apogee_altitude: ``r_a - R_E`` (m); ``inf`` when unbound.
        bound: Whether ``E < 0``.

    References:
        [EQ-AERO-ASC-005], [REF-CURTIS-2014].

    """

    specific_energy: float
    angular_momentum: float
    semi_major_axis: float
    eccentricity: float
    perigee_radius: float
    apogee_radius: float
    perigee_altitude: float
    apogee_altitude: float
    bound: bool


def specific_energy(altitude: float, speed: float) -> float:
    """Specific orbital energy ``v^2/2 - mu/(R_E + h)`` in J/kg.

    Args:
        altitude: Altitude (m), ``> -R_E``.
        speed: Speed (m/s), ``>= 0``.

    Returns:
        The energy.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-ASC-005].

    References:
        [REF-CURTIS-2014].

    """
    h = require_finite_real(altitude, name="altitude")
    v = _non_negative(speed, "speed")
    if h <= -EARTH_RADIUS:
        raise ValueError("altitude must exceed -R_E")
    return 0.5 * v * v - EARTH_MU / (EARTH_RADIUS + h)


def orbital_elements(
    altitude: float, speed: float, flight_path_angle: float
) -> OrbitalElements:
    """Osculating orbit of a planar state (altitude, speed, flight-path angle).

    With ``r = R_E + h``, ``E = v^2/2 - mu/r``, ``H = r v cos(gamma)``,
    ``p = H^2/mu``: ``e = sqrt(1 + 2 E H^2/mu^2)``, ``r_p = p/(1+e)``,
    ``r_a = p/(1-e)`` (``inf`` if ``e >= 1``) and ``a = -mu/(2E)``. For a
    state with ``H = 0`` (radial) the orbit is degenerate and is rejected.

    Args:
        altitude: Altitude above the spherical Earth (m).
        speed: Speed relative to the non-rotating Earth (m/s), ``> 0``.
        flight_path_angle: Angle above the local horizontal (rad).

    Returns:
        The :class:`OrbitalElements`.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If the speed is not positive, the altitude is invalid or
            the angular momentum vanishes.

    Equation:
        [EQ-AERO-ASC-005].

    References:
        [REF-CURTIS-2014].

    """
    h = require_finite_real(altitude, name="altitude")
    v = _positive(speed, "speed")
    gamma = require_finite_real(flight_path_angle, name="flight_path_angle")
    energy = specific_energy(h, v)
    radius = EARTH_RADIUS + h
    momentum = radius * v * math.cos(gamma)
    if abs(momentum) < 1e-9 * radius * v:
        raise ValueError("angular momentum vanishes: radial trajectory")
    parameter = momentum * momentum / EARTH_MU
    radicand = 1.0 + 2.0 * energy * momentum * momentum / EARTH_MU**2
    if radicand < -1e-12:
        raise ValueError("inconsistent state: negative eccentricity radicand")
    ecc = math.sqrt(max(radicand, 0.0))  # roundoff guard only (checked above)
    r_p = parameter / (1.0 + ecc)
    bound = energy < 0.0
    r_a = parameter / (1.0 - ecc) if ecc < 1.0 else math.inf
    a = -EARTH_MU / (2.0 * energy) if energy != 0.0 else math.inf
    return OrbitalElements(
        energy,
        abs(momentum),
        a,
        ecc,
        r_p,
        r_a,
        r_p - EARTH_RADIUS,
        r_a - EARTH_RADIUS,
        bound,
    )


def rocket_equation_delta_v(
    specific_impulse: float, initial_mass: float, final_mass: float
) -> float:
    """Tsiolkovsky velocity increment ``Isp g0 ln(m0 / mf)`` in m/s.

    Args:
        specific_impulse: ``Isp`` (s), ``> 0``.
        initial_mass: ``m0`` (kg).
        final_mass: ``mf`` (kg), ``0 < mf <= m0``.

    Returns:
        The ideal velocity increment in force-free space.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If masses or impulse are out of range.

    Equation:
        [EQ-AERO-ASC-006].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    isp = _positive(specific_impulse, "specific_impulse")
    m0 = _positive(initial_mass, "initial_mass")
    mf = _positive(final_mass, "final_mass")
    if mf > m0:
        raise ValueError("final_mass must not exceed initial_mass")
    return isp * STANDARD_GRAVITY * math.log(m0 / mf)


def vertical_burn_closed_form(
    specific_impulse: float,
    initial_mass: float,
    mass_flow: float,
    burn_time: float,
    *,
    gravity: float = STANDARD_GRAVITY,
) -> tuple[float, float]:
    """Vacuum vertical burn in uniform gravity: closed-form speed and altitude.

    With ``v_e = Isp g0``, ``m(t) = m0 - mdot t`` and constant ``g``:
    ``v = v_e ln(m0/m) - g t`` and, with ``x = mdot t / m0``,
    ``h = v_e (m0/mdot) [(1 - x) ln(1 - x) + x] - g t^2 / 2``.
    The first term of ``v`` is the Tsiolkovsky increment; ``g t`` is the
    gravity loss.

    Args:
        specific_impulse: ``Isp`` (s).
        initial_mass: ``m0`` (kg).
        mass_flow: ``mdot`` (kg/s).
        burn_time: ``t`` (s), with ``mdot t < m0``.
        gravity: Uniform acceleration ``g`` (m/s^2), ``>= 0``.

    Returns:
        ``(speed, altitude)`` after ``burn_time``, starting from rest at
        ``h = 0``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range or the burn exhausts the
            mass.

    Equation:
        [EQ-AERO-ASC-006].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    isp = _positive(specific_impulse, "specific_impulse")
    m0 = _positive(initial_mass, "initial_mass")
    mdot = _positive(mass_flow, "mass_flow")
    t = _non_negative(burn_time, "burn_time")
    g = _non_negative(gravity, "gravity")
    x = mdot * t / m0
    if x >= 1.0:
        raise ValueError("burn_time exhausts the initial mass")
    ve = isp * STANDARD_GRAVITY
    speed = -ve * math.log1p(-x) - g * t
    altitude = ve * (m0 / mdot) * ((1.0 - x) * math.log1p(-x) + x) - 0.5 * g * t * t
    return speed, altitude


Status = Literal["burnout", "impact", "no_liftoff", "numerical_failure"]


@dataclass(frozen=True, slots=True)
class AscentResult:
    """Outcome of :func:`simulate_ascent`.

    Args:
        status: ``"burnout"`` (all stages burned), ``"impact"`` (ground hit
            first), ``"no_liftoff"`` (thrust-to-weight at liftoff below 1) or
            ``"numerical_failure"``.
        stop_reason: Human-readable explanation.
        time: Time of the final state (s).
        downrange_angle: Central angle travelled (rad); distance is
            ``R_E * angle``.
        altitude: Final altitude (m).
        speed: Final speed (m/s).
        flight_path_angle: Final flight-path angle (rad).
        mass: Final mass (kg).
        elements: Osculating orbit of the final state, ``None`` unless the
            status is ``"burnout"`` and the burnout is not purely radial.
        insertion_success: ``True``/``False`` against the window, ``None``
            without a window or unless the status is ``"burnout"``.
        thrust_delta_v: ``int F/m dt`` (m/s), including the ``p_a A_e`` loss.
        gravity_loss: ``int g sin(gamma) dt`` (m/s).
        drag_loss: ``int D/m dt`` (m/s).
        max_dynamic_pressure: Largest ``rho v^2 / 2`` on the integration grid (Pa).
        steps: Number of RK4 steps taken.
        trajectory: ``(t, h, v, gamma, m)`` grid points if recording was
            requested, else empty.

    References:
        [EQ-AERO-ASC-002], [EQ-AERO-ASC-005].

    """

    status: Status
    stop_reason: str
    time: float
    downrange_angle: float
    altitude: float
    speed: float
    flight_path_angle: float
    mass: float
    elements: OrbitalElements | None
    insertion_success: bool | None
    thrust_delta_v: float
    gravity_loss: float
    drag_loss: float
    max_dynamic_pressure: float
    steps: int
    trajectory: tuple[tuple[float, float, float, float, float], ...]


# State layout of the integrated vector.
_TH, _H, _V, _G, _M, _DVT, _LG, _LD = range(8)


@dataclass(frozen=True, slots=True)
class _Phase:
    """One integration segment with constant engine/guidance mode.

    Args:
        start: Segment start time (s).
        end: Segment end time (s).
        stage: Index of the stage whose area/engine applies.
        powered: Whether the engine burns.
        turning: Whether the gravity turn is active (post-kick).

    References:
        [EQ-AERO-ASC-002]. No equation is evaluated.

    """

    start: float
    end: float
    stage: int
    powered: bool
    turning: bool


def _make_rhs(
    stage: StageSpec | None,
    *,
    powered: bool,
    turning: bool,
    drag: DragModel,
    density_scale: float,
    atmosphere_on: bool,
    uniform_gravity: bool,
) -> Callable[[float, Sequence[float]], Sequence[float]]:
    """Build the right-hand side for one phase.

    Args:
        stage: Stage providing thrust, mass flow and reference area; ``None``
            gives a ballistic arc without drag.
        powered: Whether the engine burns.
        turning: Whether ``dgamma/dt`` follows the gravity-turn law.
        drag: Drag model.
        density_scale: Multiplier on the density.
        atmosphere_on: Whether the atmosphere acts (otherwise vacuum).
        uniform_gravity: Use ``g0`` independent of altitude (verification).

    Returns:
        A function ``f(t, y)`` for :func:`integrate_fixed_step`.

    References:
        [EQ-AERO-ASC-002], [EQ-AERO-ASC-003], [EQ-AERO-ASC-004].

    """
    f_vac = stage.vacuum_thrust if (stage is not None and powered) else 0.0
    mdot = stage.mass_flow if (stage is not None and powered) else 0.0
    exit_area = stage.exit_area if stage is not None else 0.0
    ref_area = stage.reference_area if stage is not None else 0.0

    def rhs(t: float, y: Sequence[float]) -> Sequence[float]:
        """Evaluate the planar equations of motion.

        Args:
            t: Time (unused; the system is autonomous within a phase).
            y: State vector.

        Returns:
            The state derivative.

        References:
            [EQ-AERO-ASC-002].

        """
        h = y[_H]
        v = y[_V]
        gam = y[_G]
        m = y[_M]
        r = EARTH_RADIUS + h
        grav = STANDARD_GRAVITY if uniform_gravity else EARTH_MU / (r * r)
        sin_g = math.sin(gam)
        cos_g = math.cos(gam)
        force = 0.0
        drag_force = 0.0
        if atmosphere_on:
            temp, pressure, rho = _atmosphere_raw(h if h > 0.0 else 0.0)
            force = f_vac - pressure * exit_area if powered else 0.0
            if ref_area > 0.0 and v > 0.0:
                mach = v / math.sqrt(_GAMMA_AIR * _AIR_R * temp)
                drag_force = (
                    0.5 * density_scale * rho * v * v * drag.at(mach) * ref_area
                )
        else:
            force = f_vac
        if turning:
            if v <= 0.0:
                raise ValueError("speed is not positive during the gravity turn")
            d_gam = (v / r - grav / v) * cos_g
        else:
            d_gam = 0.0
        acc_thrust = force / m
        g_term = grav * sin_g
        return (
            v * cos_g / r,
            v * sin_g,
            acc_thrust - drag_force / m - g_term,
            d_gam,
            -mdot,
            acc_thrust,
            g_term,
            drag_force / m,
        )

    return rhs


def _build_phases(vehicle: VehicleSpec, guidance: GuidanceSpec) -> list[_Phase]:
    """Lay out burn, coast and kick segments in time order.

    Args:
        vehicle: The vehicle.
        guidance: The guidance (kick time splits a segment).

    Returns:
        The ordered phases; the kick splits the phase it falls in.

    References:
        [EQ-AERO-ASC-002], [EQ-AERO-ASC-003].

    """
    raw: list[tuple[float, float, int, bool]] = []
    time = 0.0
    count = len(vehicle.stages)
    for index, stage in enumerate(vehicle.stages):
        raw.append((time, time + stage.burn_time, index, True))
        time += stage.burn_time
        if index + 1 < count and stage.coast_after > 0.0:
            raw.append((time, time + stage.coast_after, index + 1, False))
            time += stage.coast_after
    kick = guidance.kick_time
    phases: list[_Phase] = []
    for start, end, index, powered in raw:
        if start < kick < end:
            phases.append(_Phase(start, kick, index, powered, False))
            phases.append(_Phase(kick, end, index, powered, True))
        else:
            phases.append(_Phase(start, end, index, powered, kick <= start))
    return phases


def _locate_impact(
    rhs: Callable[[float, Sequence[float]], Sequence[float]],
    t0: float,
    y0: Sequence[float],
    width: float,
) -> tuple[float, tuple[float, ...]]:
    """Locate ``h = 0`` inside one step by bisection of the sub-step length.

    Args:
        rhs: Right-hand side of the phase.
        t0: Time of the last state above ground.
        y0: That state.
        width: Step length at which ``h < 0`` was found.

    Returns:
        ``(time, state)`` at the crossing, ``|h| < 1e-6`` m or after 80
        bisections.

    References:
        [EQ-AERO-ASC-002], [REF-CURTIS-2014].

    """
    lo, hi = 0.0, width
    best: tuple[float, ...] = tuple(y0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        state = integrate_fixed_step(rhs, t0, y0, step=mid, steps=1).final_state
        if state[_H] >= 0.0:
            lo = mid
            best = tuple(state)
            if state[_H] < 1e-6:
                break
        else:
            hi = mid
    return t0 + lo, best


def _run_phases(
    vehicle: VehicleSpec,
    guidance: GuidanceSpec,
    *,
    drag: DragModel,
    density_scale: float,
    atmosphere_on: bool,
    uniform_gravity: bool,
    max_step: float,
    record: bool,
) -> tuple[
    Status,
    str,
    float,
    list[float],
    int,
    float,
    list[tuple[float, float, float, float, float]],
]:
    """Integrate all phases, handling kick, staging and ground impact.

    Args:
        vehicle: The vehicle.
        guidance: Kick time and angle.
        drag: Drag model.
        density_scale: Multiplier on the density.
        atmosphere_on: Atmosphere switch.
        uniform_gravity: Uniform-gravity switch (verification).
        max_step: Largest RK4 step (s).
        record: Whether to keep the grid points.

    Returns:
        ``(status, reason, time, state, steps, max_q, trajectory)``.

    References:
        [EQ-AERO-ASC-002], [EQ-AERO-ASC-003].

    """
    y: list[float] = [0.0, 0.0, 0.0, 0.5 * math.pi, vehicle.liftoff_mass, 0.0, 0.0, 0.0]
    steps_total = 0
    max_q = 0.0
    trajectory: list[tuple[float, float, float, float, float]] = []
    if record:
        trajectory.append((0.0, 0.0, 0.0, y[_G], y[_M]))
    time = 0.0
    phases = _build_phases(vehicle, guidance)
    last_phase = len(phases) - 1
    previous_stage = 0
    for position, phase in enumerate(phases):
        if phase.stage != previous_stage:
            # Jettison the burned-out stage(s) at the staging instant.
            for gone in range(previous_stage, phase.stage):
                y[_M] -= vehicle.stages[gone].dry_mass
            previous_stage = phase.stage
        if phase.turning and (not _was_turning(phases, position)):
            speed = y[_V]
            if speed < 1.0:
                raise ValueError("pitch kick requires a speed of at least 1 m/s")
            y[_G] = 0.5 * math.pi - guidance.kick_angle
        rhs = _make_rhs(
            vehicle.stages[phase.stage],
            powered=phase.powered,
            turning=phase.turning,
            drag=drag,
            density_scale=density_scale,
            atmosphere_on=atmosphere_on,
            uniform_gravity=uniform_gravity,
        )
        length = phase.end - phase.start
        n = max(1, math.ceil(length / max_step - 1e-12))
        h_step = length / n
        try:
            trajectory_ = integrate_fixed_step(
                rhs, phase.start, y, step=h_step, steps=n, method="rk4"
            )
        except ValueError as error:
            return (
                "numerical_failure",
                str(error),
                time,
                y,
                steps_total,
                max_q,
                trajectory,
            )
        for index, state in enumerate(trajectory_.states):
            if state[_H] < 0.0 and index > 0:
                t_hit, hit = _locate_impact(
                    rhs,
                    trajectory_.times[index - 1],
                    trajectory_.states[index - 1],
                    h_step,
                )
                steps_total += index
                return (
                    "impact",
                    "ground impact before burnout",
                    t_hit,
                    list(hit),
                    steps_total,
                    max_q,
                    trajectory,
                )
            if atmosphere_on:
                rho = _atmosphere_raw(max(state[_H], 0.0))[2] * density_scale
                max_q = max(max_q, 0.5 * rho * state[_V] ** 2)
            if record and index > 0:
                trajectory.append(
                    (
                        trajectory_.times[index],
                        state[_H],
                        state[_V],
                        state[_G],
                        state[_M],
                    )
                )
        steps_total += n
        y = list(trajectory_.final_state)
        time = phase.end
        if position == last_phase:
            break
    return (
        "burnout",
        "all propellant consumed",
        time,
        y,
        steps_total,
        max_q,
        trajectory,
    )


def _was_turning(phases: Sequence[_Phase], position: int) -> bool:
    """Return whether the previous phase was already in the gravity turn.

    Args:
        phases: All phases in order.
        position: Index of the current phase.

    Returns:
        ``True`` if the phase before ``position`` had ``turning`` set.

    References:
        [EQ-AERO-ASC-002]. No equation is evaluated.

    """
    return position > 0 and phases[position - 1].turning


def simulate_ascent(
    vehicle: VehicleSpec,
    guidance: GuidanceSpec,
    *,
    drag: DragModel,
    density_scale: float = 1.0,
    max_step: float = 1.0,
    window: InsertionWindow | None = None,
    atmosphere_on: bool = True,
    uniform_gravity: bool = False,
    record: bool = False,
) -> AscentResult:
    """Integrate the planar ascent from rest on the pad to final burnout.

    Equations [EQ-AERO-ASC-002]: with ``r = R_E + h`` and ``g = mu/r^2``,

    ``theta' = v cos(gamma)/r``, ``h' = v sin(gamma)``,
    ``v' = (F - D)/m - g sin(gamma)``,
    ``gamma' = (v/r - g/v) cos(gamma)`` after the kick (0 before),
    ``m' = -mdot``, with ``D = rho v^2 C_D(M) S / 2`` and
    ``F = F_vac - p_a A_e``. Accuracy: RK4, global error ``O(max_step^4)`` in
    smooth segments; the kick and burnout times coincide with step
    boundaries. The accumulators ``int F/m``, ``int g sin(gamma)`` and
    ``int D/m`` give the exact budget ``v = thrust_dv - gravity_loss -
    drag_loss`` of the integration.

    Args:
        vehicle: The staged vehicle.
        guidance: Kick time and angle.
        drag: Drag coefficient model.
        density_scale: Multiplier on the standard density (``> 0``); pressure
            is not scaled.
        max_step: Largest RK4 step in s (``> 0``).
        window: Optional insertion window for the success indicator.
        atmosphere_on: ``False`` for vacuum (no drag, ``p_a = 0``).
        uniform_gravity: ``True`` uses ``g = g0`` at every altitude (for
            verification against closed forms only).
        record: Keep the grid points in ``trajectory``.

    Returns:
        The :class:`AscentResult`. Physical failures (impact, no liftoff) are
        reported in ``status``, not raised.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a numeric argument is out of range or the kick happens
            before the vehicle reaches 1 m/s.

    Equation:
        [EQ-AERO-ASC-002], [EQ-AERO-ASC-003], [EQ-AERO-ASC-004],
        [EQ-AERO-ASC-005].

    References:
        [REF-CURTIS-2014], [REF-SUTTON-BIBLARZ-2017],
        [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be a VehicleSpec")
    if not isinstance(guidance, GuidanceSpec):
        raise TypeError("guidance must be a GuidanceSpec")
    if not isinstance(drag, DragModel):
        raise TypeError("drag must be a DragModel")
    if window is not None and not isinstance(window, InsertionWindow):
        raise TypeError("window must be an InsertionWindow or None")
    k_rho = _positive(density_scale, "density_scale")
    step = _positive(max_step, "max_step")
    first = vehicle.stages[0]
    sea_level = first.thrust(_P0 if atmosphere_on else 0.0)
    weight = vehicle.liftoff_mass * (
        STANDARD_GRAVITY if uniform_gravity else EARTH_MU / EARTH_RADIUS**2
    )
    if sea_level <= weight:
        return AscentResult(
            "no_liftoff",
            "thrust-to-weight ratio at liftoff is not above 1",
            0.0,
            0.0,
            0.0,
            0.0,
            0.5 * math.pi,
            vehicle.liftoff_mass,
            None,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            (),
        )
    status, reason, time, y, steps, max_q, path = _run_phases(
        vehicle,
        guidance,
        drag=drag,
        density_scale=k_rho,
        atmosphere_on=atmosphere_on,
        uniform_gravity=uniform_gravity,
        max_step=step,
        record=record,
    )
    elements: OrbitalElements | None = None
    success: bool | None = None
    if status == "burnout":
        try:
            elements = orbital_elements(y[_H], y[_V], y[_G])
        except ValueError:
            elements = None  # radial burnout: no Keplerian orbit exists
        if window is not None:
            success = (
                elements is not None
                and elements.bound
                and elements.perigee_altitude >= window.min_perigee
                and window.apogee_min <= elements.apogee_altitude <= window.apogee_max
            )
    return AscentResult(
        status,
        reason,
        time,
        y[_TH],
        y[_H],
        y[_V],
        y[_G],
        y[_M],
        elements,
        success,
        y[_DVT],
        y[_LG],
        y[_LD],
        max_q,
        steps,
        tuple(path),
    )


def coast_arc(
    altitude: float,
    speed: float,
    flight_path_angle: float,
    duration: float,
    *,
    max_step: float = 1.0,
    uniform_gravity: bool = False,
) -> tuple[float, float, float, float]:
    """Propagate a ballistic (thrust-free, drag-free) arc with RK4.

    Uses the same equations as the powered phases with ``F = D = 0``. Specific
    energy and angular momentum are conserved by the exact flow; their drift
    measures the integration error.

    Args:
        altitude: Initial altitude (m).
        speed: Initial speed (m/s), ``> 0``.
        flight_path_angle: Initial flight-path angle (rad).
        duration: Coast time (s), ``> 0``.
        max_step: Largest step (s).
        uniform_gravity: Use ``g0`` at every altitude.

    Returns:
        ``(altitude, speed, flight_path_angle, downrange_angle)`` at the end.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range or the arc fails
            numerically.

    Equation:
        [EQ-AERO-ASC-002].

    References:
        [REF-CURTIS-2014].

    """
    h0 = require_finite_real(altitude, name="altitude")
    v0 = _positive(speed, "speed")
    g0 = require_finite_real(flight_path_angle, name="flight_path_angle")
    t_end = _positive(duration, "duration")
    step = _positive(max_step, "max_step")
    rhs = _make_rhs(
        None,
        powered=False,
        turning=True,
        drag=DragModel.constant(0.0),
        density_scale=1.0,
        atmosphere_on=False,
        uniform_gravity=uniform_gravity,
    )
    n = max(1, math.ceil(t_end / step - 1e-12))
    start = [0.0, h0, v0, g0, 1.0, 0.0, 0.0, 0.0]
    final = integrate_fixed_step(rhs, 0.0, start, step=t_end / n, steps=n).final_state
    return final[_H], final[_V], final[_G], final[_TH]


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_INPUT_UNITS = {
    "thrust_scale": "-",
    "isp_scale": "-",
    "propellant_scale": "-",
    "dry_mass_scale": "-",
    "drag_scale": "-",
    "density_scale": "-",
    "pitch_kick_angle": "rad",
    "pitch_kick_time": "s",
}


@dataclass(frozen=True, slots=True)
class _AscentForward:
    """Picklable forward map of :func:`ascent_model`.

    Args:
        vehicle: Nominal vehicle.
        guidance: Nominal guidance.
        drag: Nominal drag model.
        window: Optional insertion window.
        stages: Indices of stages receiving the thrust/Isp/mass scales.
        max_step: RK4 step bound (s).

    References:
        [EQ-AERO-ASC-007].

    """

    vehicle: VehicleSpec
    guidance: GuidanceSpec
    drag: DragModel
    window: InsertionWindow | None
    stages: tuple[int, ...]
    max_step: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Run one ascent for named parameter values.

        Args:
            parameters: The eight scale/kick parameters of :func:`ascent_model`.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If a parameter is out of its domain.
            RuntimeError: If the ascent does not reach burnout, or the final
                orbit is unbound (no finite apogee).

        Equation:
            [EQ-AERO-ASC-007].

        References:
            [REF-CURTIS-2014].

        """
        thrust_k = parameters["thrust_scale"]
        isp_k = parameters["isp_scale"]
        prop_k = parameters["propellant_scale"]
        dry_k = parameters["dry_mass_scale"]
        new_stages = []
        for index, stage in enumerate(self.vehicle.stages):
            if index in self.stages:
                stage = replace(
                    stage,
                    vacuum_thrust=stage.vacuum_thrust
                    * _positive(thrust_k, "thrust_scale"),
                    vacuum_isp=stage.vacuum_isp * _positive(isp_k, "isp_scale"),
                    propellant_mass=stage.propellant_mass
                    * _positive(prop_k, "propellant_scale"),
                    dry_mass=stage.dry_mass * _positive(dry_k, "dry_mass_scale"),
                )
            new_stages.append(stage)
        vehicle = VehicleSpec(tuple(new_stages), self.vehicle.payload_mass)
        guidance = GuidanceSpec(
            parameters["pitch_kick_time"], parameters["pitch_kick_angle"]
        )
        result = simulate_ascent(
            vehicle,
            guidance,
            drag=self.drag.scaled(parameters["drag_scale"]),
            density_scale=parameters["density_scale"],
            max_step=self.max_step,
            window=self.window,
        )
        elements = result.elements
        if result.status != "burnout" or elements is None:
            raise RuntimeError(f"ascent failed: {result.status}: {result.stop_reason}")
        if not elements.bound:
            raise RuntimeError("final orbit is unbound: no finite apogee")
        out = {
            "perigee_altitude": elements.perigee_altitude,
            "apogee_altitude": elements.apogee_altitude,
            "eccentricity": elements.eccentricity,
            "specific_energy": elements.specific_energy,
            "burnout_altitude": result.altitude,
            "burnout_speed": result.speed,
            "flight_path_angle": result.flight_path_angle,
            "burnout_time": result.time,
            "final_mass": result.mass,
            "gravity_loss": result.gravity_loss,
            "drag_loss": result.drag_loss,
            "max_dynamic_pressure": result.max_dynamic_pressure,
        }
        if result.insertion_success is not None:
            out["insertion_success"] = 1.0 if result.insertion_success else 0.0
        return out


_QUANTITY_TABLE = (
    ("perigee_altitude", "Perigee altitude of the burnout orbit", "m"),
    ("apogee_altitude", "Apogee altitude of the burnout orbit", "m"),
    ("eccentricity", "Eccentricity of the burnout orbit", "-"),
    ("specific_energy", "Specific orbital energy at burnout", "J/kg"),
    ("burnout_altitude", "Altitude at final burnout", "m"),
    ("burnout_speed", "Speed at final burnout", "m/s"),
    ("flight_path_angle", "Flight-path angle at final burnout", "rad"),
    ("burnout_time", "Time of final burnout", "s"),
    ("final_mass", "Mass after final burnout", "kg"),
    ("gravity_loss", "Gravity loss integral", "m/s"),
    ("drag_loss", "Drag loss integral", "m/s"),
    ("max_dynamic_pressure", "Maximum dynamic pressure on the grid", "Pa"),
)


def ascent_model(
    vehicle: VehicleSpec,
    guidance: GuidanceSpec,
    drag: DragModel,
    *,
    thrust_scale: Marginal | float = 1.0,
    isp_scale: Marginal | float = 1.0,
    propellant_scale: Marginal | float = 1.0,
    dry_mass_scale: Marginal | float = 1.0,
    drag_scale: Marginal | float = 1.0,
    density_scale: Marginal | float = 1.0,
    pitch_kick_angle: Marginal | float | None = None,
    pitch_kick_time: Marginal | float | None = None,
    window: InsertionWindow | None = None,
    perturbed_stages: Sequence[int] | None = None,
    max_step: float = 2.0,
    name: str = "ascent",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic ascent model for dispersion analysis.

    Each argument is a distribution (an uncertain input, wrapped as an
    independent :class:`~probstat.modeling.RandomInput`) or a plain number (a
    fixed parameter). The scales are multiplicative factors with nominal
    value 1 applied to every perturbed stage: ``thrust_scale`` on ``F_vac`` at
    fixed ``Isp_vac`` (so the mass flow scales with it), ``isp_scale`` on
    ``Isp_vac`` at fixed ``F_vac`` (mass flow falls as it rises),
    ``propellant_scale`` and ``dry_mass_scale`` on the masses,
    ``drag_scale`` on ``C_D`` and ``density_scale`` on the density. The kick
    angle (rad) and time (s) default to the nominal ``guidance`` as fixed
    parameters. There is no wind. The scale must remain positive, so use
    bounded or well-away-from-zero laws.

    Quantities of interest: ``perigee_altitude``, ``apogee_altitude``,
    ``eccentricity``, ``specific_energy``, ``burnout_altitude``,
    ``burnout_speed``, ``flight_path_angle``, ``burnout_time``, ``final_mass``,
    ``gravity_loss``, ``drag_loss``, ``max_dynamic_pressure`` and, when a
    ``window`` is given, ``insertion_success`` (1 or 0). A run that does not
    reach burnout, or ends on an unbound orbit, fails and is handled by
    ``on_failure``. Vehicle numbers are illustrative.

    Args:
        vehicle: Nominal vehicle.
        guidance: Nominal guidance.
        drag: Nominal drag model.
        thrust_scale: Scale on the vacuum thrust.
        isp_scale: Scale on the vacuum specific impulse.
        propellant_scale: Scale on the propellant mass.
        dry_mass_scale: Scale on the dry mass.
        drag_scale: Scale on the drag coefficient.
        density_scale: Scale on the atmospheric density.
        pitch_kick_angle: Kick angle (rad); ``None`` uses the guidance value.
        pitch_kick_time: Kick time (s); ``None`` uses the guidance value.
        window: Optional insertion window (adds ``insertion_success``).
        perturbed_stages: Stage indices receiving the thrust, Isp and mass
            scales; default all stages.
        max_step: RK4 step bound (s) of every run.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is outside its domain, a stage index is
            invalid or no input is random.

    Equation:
        [EQ-AERO-ASC-007].

    References:
        [REF-CURTIS-2014], [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be a VehicleSpec")
    if not isinstance(guidance, GuidanceSpec):
        raise TypeError("guidance must be a GuidanceSpec")
    if not isinstance(drag, DragModel):
        raise TypeError("drag must be a DragModel")
    if window is not None and not isinstance(window, InsertionWindow):
        raise TypeError("window must be an InsertionWindow or None")
    stages = (
        tuple(range(len(vehicle.stages)))
        if perturbed_stages is None
        else tuple(perturbed_stages)
    )
    if not stages or any(
        isinstance(i, bool)
        or not isinstance(i, int)
        or not 0 <= i < len(vehicle.stages)
        for i in stages
    ):
        raise ValueError("perturbed_stages must be valid stage indices")
    kick_angle: Marginal | float = (
        guidance.kick_angle if pitch_kick_angle is None else pitch_kick_angle
    )
    kick_time: Marginal | float = (
        guidance.kick_time if pitch_kick_time is None else pitch_kick_time
    )
    if isinstance(kick_time, float) and math.isinf(kick_time):
        raise ValueError("a purely vertical guidance has no kick to disperse")
    step = _positive(max_step, "max_step")
    given: dict[str, Marginal | float] = {
        "thrust_scale": thrust_scale,
        "isp_scale": isp_scale,
        "propellant_scale": propellant_scale,
        "dry_mass_scale": dry_mass_scale,
        "drag_scale": drag_scale,
        "density_scale": density_scale,
        "pitch_kick_angle": kick_angle,
        "pitch_kick_time": kick_time,
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, value in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = require_finite_real(value, name=key)
        else:
            random_inputs.append(RandomInput(key, value, unit=_INPUT_UNITS[key]))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    quantities = [QuantityOfInterest(n, d, u) for n, d, u in _QUANTITY_TABLE]
    if window is not None:
        quantities.append(
            QuantityOfInterest(
                "insertion_success", "1 if the burnout orbit is in the window", "-"
            )
        )
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        _AscentForward(vehicle, guidance, drag, window, stages, step),
        tuple(quantities),
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# ---------------------------------------------------------------------------
# Dispersion analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DispersionInterval:
    """Monte Carlo dispersion interval of one quantity.

    Args:
        quantity: Quantity name.
        samples: Number of successful runs.
        mean: Sample mean.
        standard_deviation: Sample standard deviation (``n - 1``).
        coverage: Central coverage, e.g. 0.9973 (3 sigma).
        normal_interval: ``mean -/+ k sigma`` with ``k`` the two-sided normal
            quantile of ``coverage`` (3.0 at 99.73%).
        empirical_interval: Order-statistic quantiles ``((1-c)/2, (1+c)/2)``
            with linear interpolation; unreliable unless ``samples`` greatly
            exceeds ``1/(1 - coverage)``.

    References:
        [EQ-AERO-ASC-008].

    """

    quantity: str
    samples: int
    mean: float
    standard_deviation: float
    coverage: float
    normal_interval: tuple[float, float]
    empirical_interval: tuple[float, float]


def _normal_two_sided_k(coverage: float) -> float:
    """Return ``k`` with ``P(|Z| <= k) = coverage`` by bisection.

    Args:
        coverage: Probability in ``(0, 1)``.

    Returns:
        The two-sided standard-normal half-width.

    References:
        [EQ-AERO-ASC-008].

    """
    lo, hi = 0.0, 12.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2.0)) < coverage:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _quantile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation sample quantile (type 7).

    Args:
        sorted_values: Ascending sample.
        p: Probability in ``[0, 1]``.

    Returns:
        The quantile.

    References:
        [EQ-AERO-ASC-008].

    """
    position = p * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return (1.0 - frac) * sorted_values[lo] + frac * sorted_values[hi]


def _check_coverage(coverage: float) -> float:
    """Validate a coverage probability.

    Args:
        coverage: Candidate probability.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If it is not a real number.
        ValueError: If it is not strictly between 0 and 1.

    References:
        [EQ-AERO-ASC-008]. No equation is evaluated.

    """
    value = require_finite_real(coverage, name="coverage")
    if not 0.0 < value < 1.0:
        raise ValueError("coverage must lie strictly in (0, 1)")
    return value


def _run_samples(
    model: ProbabilisticModel,
    quantities: Sequence[str],
    samples: int,
    seed: int | None,
) -> dict[str, list[float]]:
    """Run plain Monte Carlo and collect named outputs.

    Args:
        model: The probabilistic model.
        quantities: Quantity names to collect.
        samples: Number of draws (``>= 2``).
        seed: Seed for the sampling stream.

    Returns:
        Output lists keyed by quantity name (failed runs under a non-raising
        policy are dropped).

    Raises:
        TypeError: If ``model`` is not a ``ProbabilisticModel``.
        ValueError: If fewer than two runs succeed or a name is unknown.

    References:
        [EQ-AERO-ASC-008].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    unknown = [q for q in quantities if q not in model.quantity_names]
    if unknown:
        raise ValueError(f"unknown quantities: {unknown}")
    stream = resolve_stream(seed)
    points = model.inputs.sample(samples, seed=stream)
    out: dict[str, list[float]] = {q: [] for q in quantities}
    for point in points:
        try:
            result = model.evaluate(point)
        except Exception:
            if model.on_failure == "raise":
                raise
            continue
        for q in quantities:
            out[q].append(float(result[q]))
    if len(out[quantities[0]]) < 2:
        raise ValueError("fewer than two model runs succeeded")
    return out


def dispersion_intervals(
    model: ProbabilisticModel,
    quantities: Sequence[str],
    samples: int,
    *,
    coverage: float = 0.9973,
    seed: int | None = None,
) -> tuple[DispersionInterval, ...]:
    """Monte Carlo dispersion intervals (default 3-sigma, 99.73%).

    Args:
        model: Probabilistic model, for example from :func:`ascent_model`.
        quantities: Names of quantities of interest.
        samples: Number of Monte Carlo runs (``>= 2``).
        coverage: Central coverage in ``(0, 1)``.
        seed: Sampling seed.

    Returns:
        One :class:`DispersionInterval` per quantity, in the given order.

    Raises:
        TypeError: If ``model`` is not a ``ProbabilisticModel``.
        ValueError: If arguments are invalid or fewer than two runs succeed.

    Equation:
        [EQ-AERO-ASC-008].

    References:
        [REF-CURTIS-2014].

    """
    names = tuple(quantities)
    if not names:
        raise ValueError("quantities must not be empty")
    cover = _check_coverage(coverage)
    data = _run_samples(model, names, samples, seed)
    k = _normal_two_sided_k(cover)
    out: list[DispersionInterval] = []
    for q in names:
        values = data[q]
        n = len(values)
        mean = math.fsum(values) / n
        var = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
        sd = math.sqrt(var)
        ordered = sorted(values)
        out.append(
            DispersionInterval(
                q,
                n,
                mean,
                sd,
                cover,
                (mean - k * sd, mean + k * sd),
                (
                    _quantile(ordered, 0.5 * (1.0 - cover)),
                    _quantile(ordered, 0.5 * (1.0 + cover)),
                ),
            )
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class DispersionEllipse:
    """Gaussian dispersion ellipse of two outputs.

    Args:
        center: Sample mean ``(x, y)``.
        semi_axes: ``(major, minor)`` half-lengths ``k sqrt(lambda_i)``.
        angle: Orientation of the major axis from the ``x`` axis (rad).
        radius: Mahalanobis radius ``k = sqrt(-2 ln(1 - coverage))``.
        coverage: Probability content under a bivariate normal law.
        correlation: Sample correlation coefficient.
        empirical_coverage: Fraction of the sample inside the ellipse.

    References:
        [EQ-AERO-ASC-008].

    """

    center: tuple[float, float]
    semi_axes: tuple[float, float]
    angle: float
    radius: float
    coverage: float
    correlation: float
    empirical_coverage: float


def dispersion_ellipse(
    x: Sequence[float], y: Sequence[float], *, coverage: float = 0.9973
) -> DispersionEllipse:
    """Fit a bivariate-Gaussian dispersion ellipse to paired samples.

    The ellipse ``(z - m)^T S^{-1} (z - m) = k^2`` with sample mean ``m`` and
    covariance ``S`` has probability ``coverage`` under a bivariate normal law
    when ``k^2 = -2 ln(1 - coverage)`` (``k = 3.44`` at 99.73%, larger than
    the 1-D value 3).

    Args:
        x: First output samples.
        y: Second output samples, same length ``>= 3``.
        coverage: Gaussian probability content in ``(0, 1)``.

    Returns:
        The :class:`DispersionEllipse`.

    Raises:
        TypeError: If a value is not a real number.
        ValueError: If lengths differ, are too short, or the covariance is
            singular.

    Equation:
        [EQ-AERO-ASC-008].

    References:
        [REF-CURTIS-2014].

    """
    xs = [require_finite_real(v, name="x") for v in x]
    ys = [require_finite_real(v, name="y") for v in y]
    n = len(xs)
    if n != len(ys) or n < 3:
        raise ValueError("x and y need equal length >= 3")
    cover = _check_coverage(coverage)
    mx = math.fsum(xs) / n
    my = math.fsum(ys) / n
    sxx = math.fsum((a - mx) ** 2 for a in xs) / (n - 1)
    syy = math.fsum((b - my) ** 2 for b in ys) / (n - 1)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / (n - 1)
    det = sxx * syy - sxy * sxy
    if sxx <= 0.0 or syy <= 0.0 or det <= 1e-14 * sxx * syy:
        raise ValueError("sample covariance is singular")
    k2 = -2.0 * math.log1p(-cover)
    half_trace = 0.5 * (sxx + syy)
    spread = math.sqrt(max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
    lam1 = half_trace + spread
    lam2 = half_trace - spread
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    inside = sum(
        1
        for a, b in zip(xs, ys, strict=True)
        if (syy * (a - mx) ** 2 - 2.0 * sxy * (a - mx) * (b - my) + sxx * (b - my) ** 2)
        / det
        <= k2
    )
    k = math.sqrt(k2)
    return DispersionEllipse(
        (mx, my),
        (k * math.sqrt(lam1), k * math.sqrt(lam2)),
        angle,
        k,
        cover,
        sxy / math.sqrt(sxx * syy),
        inside / n,
    )


@dataclass(frozen=True, slots=True)
class DispersionBudget:
    """One-at-a-time dispersion budget against Monte Carlo.

    Args:
        quantity: Output name.
        nominal: Output at the input means.
        input_names: Names of the uncertain inputs.
        sigmas: Input standard deviations.
        derivatives: Central finite-difference slopes ``dy/dx_i`` at the means.
        contributions: ``|dy/dx_i| sigma_i`` (linear 1-sigma contribution).
        rss: ``sqrt(sum contributions^2)``, the linear RSS dispersion.
        plus_shift: Actual output change at ``+3 sigma_i`` alone.
        minus_shift: Actual output change at ``-3 sigma_i`` alone.
        monte_carlo_mean: Sample mean of the output.
        monte_carlo_std: Sample standard deviation of the output.
        ratio: ``rss / monte_carlo_std``; near 1 only when the response is
            near-linear with independent inputs.
        mean_shift: ``monte_carlo_mean - nominal``, invisible to the linear
            budget.
        nonlinearity: Largest ``|plus + minus| / (|plus| + |minus|)`` over
            the inputs: asymmetry of the 3-sigma one-at-a-time responses
            (0 for a linear response).

    References:
        [EQ-AERO-ASC-008].

    """

    quantity: str
    nominal: float
    input_names: tuple[str, ...]
    sigmas: tuple[float, ...]
    derivatives: tuple[float, ...]
    contributions: tuple[float, ...]
    rss: float
    plus_shift: tuple[float, ...]
    minus_shift: tuple[float, ...]
    monte_carlo_mean: float
    monte_carlo_std: float
    ratio: float
    mean_shift: float
    nonlinearity: float


def dispersion_budget(
    model: ProbabilisticModel,
    quantity: str,
    samples: int,
    *,
    relative_step: float = 0.25,
    seed: int | None = None,
) -> DispersionBudget:
    """Dispersion budget: RSS of one-at-a-time sensitivities versus Monte Carlo.

    At the vector of input means ``x0`` each uncertain input is perturbed alone
    by ``+/- relative_step * sigma_i`` (central difference) to give the slope
    ``s_i``, and the linear budget is ``RSS = sqrt(sum (s_i sigma_i)^2)``,
    which equals the output standard deviation only if the response is
    linear in the inputs and the inputs are uncorrelated. The same inputs are
    also moved to ``+/- 3 sigma_i`` alone to expose asymmetry, and a plain
    Monte Carlo run with ``samples`` draws gives the reference standard
    deviation and mean.

    Where linear RSS fails: (i) nonlinear responses, where the mean shifts
    by about ``0.5 sum f'' sigma^2`` and the spread differs from the slope
    prediction, visible in ``mean_shift`` and ``nonlinearity``; (ii)
    interactions between inputs, since one-at-a-time slopes see none; (iii)
    correlated or non-Gaussian inputs; (iv) cliffs where a perturbation makes
    the ascent fail (impact), which stops the budget. The cost is
    ``1 + 4 d`` runs plus ``samples``.

    Args:
        model: Probabilistic model with independent inputs.
        quantity: Output name.
        samples: Monte Carlo runs (``>= 2``).
        relative_step: Finite-difference step in standard deviations,
            ``(0, 3]``.
        seed: Seed of the Monte Carlo draw.

    Returns:
        The :class:`DispersionBudget`.

    Raises:
        TypeError: If ``model`` is not a ``ProbabilisticModel``.
        ValueError: If the inputs are dependent, an argument is invalid, or a
            perturbed run fails.

    Equation:
        [EQ-AERO-ASC-008].

    References:
        [REF-CURTIS-2014].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    if quantity not in model.quantity_names:
        raise ValueError(f"unknown quantity {quantity!r}")
    if not model.inputs.is_independent:
        raise ValueError("the RSS budget assumes independent inputs")
    step = require_finite_real(relative_step, name="relative_step")
    if not 0.0 < step <= 3.0:
        raise ValueError("relative_step must lie in (0, 3]")
    x0 = list(model.inputs.means())
    sigmas = tuple(model.inputs.standard_deviations())
    names = model.inputs.names
    base = model.evaluate(tuple(x0))[quantity]

    def shifted(index: int, amount: float) -> float:
        """Evaluate the output with one input moved by ``amount`` sigma.

        Args:
            index: Input index.
            amount: Shift in units of that input's standard deviation.

        Returns:
            The output value.

        References:
            [EQ-AERO-ASC-008].

        """
        x = list(x0)
        x[index] += amount * sigmas[index]
        return float(model.evaluate(tuple(x))[quantity])

    derivatives: list[float] = []
    plus: list[float] = []
    minus: list[float] = []
    for i in range(len(x0)):
        derivatives.append(
            (shifted(i, step) - shifted(i, -step)) / (2.0 * step * sigmas[i])
        )
        plus.append(shifted(i, 3.0) - base)
        minus.append(shifted(i, -3.0) - base)
    contributions = tuple(abs(d) * s for d, s in zip(derivatives, sigmas, strict=True))
    rss = math.sqrt(sum(c * c for c in contributions))
    data = _run_samples(model, (quantity,), samples, seed)[quantity]
    n = len(data)
    mean = math.fsum(data) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in data) / (n - 1))
    asym = [
        abs(p + m) / (abs(p) + abs(m)) if (abs(p) + abs(m)) > 0.0 else 0.0
        for p, m in zip(plus, minus, strict=True)
    ]
    return DispersionBudget(
        quantity,
        float(base),
        names,
        sigmas,
        tuple(derivatives),
        contributions,
        rss,
        tuple(plus),
        tuple(minus),
        mean,
        sd,
        rss / sd if sd > 0.0 else math.inf,
        mean - float(base),
        max(asym) if asym else 0.0,
    )
