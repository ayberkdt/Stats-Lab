"""Atmospheric entry of a capsule: trajectory, heating and heat-shield sizing.

A planar point-mass model of a blunt capsule entering the atmosphere of a
spherical, non-rotating Earth, with the convective stagnation-point heating of
Sutton and Graves, a first-order ablative heat-shield sizing model and the
closed-form ballistic-entry solution of Allen and Eggers as a verification
anchor. All quantities are SI (m, s, kg, W, J, K, rad) unless a name says
otherwise (peak deceleration is reported in multiples of ``g0``).

Model (Allen and Eggers; Sutton and Graves; Curtis; U.S. Standard Atmosphere
1976):

- State ``(theta, h, V, gamma, Q)``: central (downrange) angle, altitude above
  the sphere of radius ``R_E = 6 378 137 m``, speed, flight-path angle above
  the local horizontal (negative while descending) and the accumulated
  convective heat load per unit area ``Q = int q dt`` (J/m^2).
- Aerodynamics: drag acceleration ``D/m = rho V^2 / (2 beta)`` with the
  ballistic coefficient ``beta = m / (C_D A)`` (kg/m^2) and an optional
  constant lift-to-drag ratio ``L/D`` with the lift vector rotated "up" (toward
  increasing ``gamma``, no bank). The total aerodynamic load factor is
  ``(D/m) sqrt(1 + (L/D)^2) / g0``.
- Gravity ``mu / r^2``; the Earth does not rotate, so inertial and relative
  velocity coincide.
- Atmosphere: the 1976 standard atmosphere of
  :func:`probstat.aerospace.ascent.atmosphere` (the isothermal extension above
  86 km is *not* part of the standard model, so the entry interface at about
  120 km lies in the extension) or a single exponential atmosphere
  ``rho = rho_0 exp(-h / H)`` for the closed-form checks. A multiplicative
  ``density_scale`` represents density uncertainty (it scales the density seen
  by both drag and heating).
- Heating: the Sutton-Graves stagnation-point convective correlation
  ``q = k sqrt(rho / R_n) V^3`` with ``k = 1.7483e-4 kg^0.5 / m`` for Earth
  air. Radiative heating, turbulent augmentation, catalytic effects and
  real-gas effects are *not* modelled; the correlation is a convective
  laminar estimate, and at lunar-return speeds (about 11 km/s) radiation is
  not negligible. The ``heating_scale`` input of :func:`entry_model` absorbs
  the correlation's uncertainty.
- Integration: classical fourth-order Runge-Kutta
  (:func:`probstat.numerics.integrate_fixed_step`) on a uniform grid whose step
  is the smaller of ``max_step`` and ``resolution`` scale heights of altitude
  change at the entry rate of descent. Peak deceleration and peak heat flux
  are located on the grid and refined by a parabola through the three grid
  points around the discrete maximum. The terminal altitude is located by
  bisection of one sub-step.

Verification anchor: in an exponential atmosphere, with gravity and curvature
neglected and a constant flight-path angle, the equations integrate in closed
form (Allen and Eggers, :class:`AllenEggersEntry`): ``V(h) = V_e exp(-rho H /
(2 beta sin|gamma|))``, peak load at ``rho = beta sin|gamma| / H`` with value
``V_e^2 sin|gamma| / (2 e H)``, peak heating at ``rho = beta sin|gamma| / (3 H)``
(an altitude ``H ln 3`` above the peak load), and a total heat load
``k V_e^2 sqrt(pi beta H / (R_n sin|gamma|))``. The integrator switch
``idealised=True`` removes gravity and curvature so the numerical solution can
be compared with these formulas to the integration error alone; the same
formulas are compared with the full model only as an order-of-magnitude check
whose tolerance is set by the neglected terms.

Limits. Everything is engineering level. The capsule is a point mass at
constant ``C_D`` and constant ``L/D`` (no angle-of-attack history, no bank
modulation, no guidance), so skip-out and landing-footprint results illustrate
the method rather than describe a flight system. The heat-shield model
(:func:`heat_shield_sizing`) has synthetic material parameters. Nothing here
represents data from a real vehicle; the Apollo-/Orion-like capsule used in
tests and examples is a synthetic case of that order of magnitude.

Experimental (added in 0.28).

References:
    [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971], [REF-CURTIS-2014],
    [REF-US-STANDARD-ATMOSPHERE-1976], [EQ-AERO-ENTRY-001].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.aerospace.ascent import (
    EARTH_MU,
    EARTH_RADIUS,
    STANDARD_GRAVITY,
    _atmosphere_raw,
)
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.numerics import integrate_fixed_step
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "SUTTON_GRAVES_EARTH_K",
    "AllenEggersEntry",
    "EntryCorridor",
    "EntryResult",
    "EntryVehicle",
    "ExponentialAtmosphere",
    "HeatShield",
    "HeatShieldSizing",
    "entry_corridor",
    "entry_model",
    "heat_shield_sizing",
    "simulate_entry",
    "sutton_graves_heat_flux",
]

SUTTON_GRAVES_EARTH_K = 0.1113 / (2.0 * math.sqrt(101325.0))
"""Sutton-Graves constant ``k`` for Earth air, kg^0.5 / m (about 1.7483e-4).

``q [W/m^2] = k sqrt(rho [kg/m^3] / R_n [m]) V^3 [m/s]``. Derived from the
report's heat-transfer coefficient ``K = q sqrt(R_n / p_s) / (h_s - h_w)``
(its equation 33), whose correlated value for air (0.2320 O2, 0.7680 N2 by
mass) is ``K = 0.1113 kg/(s m^1.5 atm^0.5)`` (its table II; average
correlation error 3.3 %, maximum 9.8 %), with the Newtonian stagnation
pressure ``p_s = rho V^2`` (converted to atm by 101325 Pa) and the cold-wall
enthalpy difference ``h_s - h_w = V^2 / 2``. The often quoted 1.7415e-4 is
0.4 % lower. Both differences are far smaller than the correlation error,
which the ``heating_scale`` input of :func:`entry_model` carries.
"""

_H_REF = 7200.0  # m, reference scale height used only to size the time step
_CHUNK = 32  # integration steps between event checks
_STATE_TH, _STATE_H, _STATE_V, _STATE_G, _STATE_Q = range(5)

EntryStatus = Literal["landed", "skip_out", "timeout", "numerical_failure"]


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
        [REF-ALLEN-EGGERS-1958]. No equation is evaluated.

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
        [REF-ALLEN-EGGERS-1958]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _entry_angle(value: object, name: str = "flight_path_angle") -> float:
    """Return a descending flight-path angle in ``[-pi/2, 0)``.

    Args:
        value: Candidate angle in rad.
        name: Name used in error messages.

    Returns:
        The angle as ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If the angle is not in ``[-pi/2, 0)``.

    References:
        [EQ-AERO-ENTRY-001]. No equation is evaluated.

    """
    angle = require_finite_real(value, name=name)
    if not -0.5 * math.pi <= angle < 0.0:
        raise ValueError(f"{name} must satisfy -pi/2 <= angle < 0 (a descent)")
    return angle


@dataclass(frozen=True, slots=True)
class ExponentialAtmosphere:
    """Single-scale-height exponential atmosphere ``rho = rho_0 exp(-h/H)``.

    The atmosphere of the Allen-Eggers solution. The defaults are an
    engineering-level choice (sea-level density of the standard atmosphere and
    a scale height of about 7.2 km, typical of the 40 to 80 km band that
    matters for entry); they are not fitted to the 1976 model.

    Args:
        surface_density: ``rho_0`` in kg/m^3, ``> 0``.
        scale_height: ``H`` in m, ``> 0``.

    References:
        [EQ-AERO-ENTRY-001], [REF-ALLEN-EGGERS-1958].

    """

    surface_density: float = 1.225
    scale_height: float = 7200.0

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            TypeError: If a parameter is not a real number.
            ValueError: If a parameter is not positive.

        References:
            [EQ-AERO-ENTRY-001]. No equation is evaluated.

        """
        _positive(self.surface_density, "surface_density")
        _positive(self.scale_height, "scale_height")

    def density(self, altitude: float) -> float:
        """Return ``rho_0 exp(-h / H)`` in kg/m^3.

        Args:
            altitude: Altitude ``h`` in m.

        Returns:
            The density.

        Raises:
            TypeError: If ``altitude`` is not a real number.

        Equation:
            [EQ-AERO-ENTRY-001].

        References:
            [REF-ALLEN-EGGERS-1958].

        """
        h = require_finite_real(altitude, name="altitude")
        return self.surface_density * math.exp(-h / self.scale_height)


def _density_function(
    atmosphere: ExponentialAtmosphere | Literal["us76"],
) -> Callable[[float], float]:
    """Return the density ``rho(h)`` callable of an atmosphere choice.

    Args:
        atmosphere: ``"us76"`` or an :class:`ExponentialAtmosphere`.

    Returns:
        A function from altitude (m) to density (kg/m^3); altitudes below 0
        are evaluated at 0 for the standard atmosphere.

    Raises:
        TypeError: If the choice is neither of the two.

    References:
        [EQ-AERO-ENTRY-001], [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    if isinstance(atmosphere, ExponentialAtmosphere):
        rho0 = atmosphere.surface_density
        scale = atmosphere.scale_height

        def exponential(h: float) -> float:
            """Exponential density.

            Args:
                h: Altitude in m.

            Returns:
                The density in kg/m^3.

            References:
                [EQ-AERO-ENTRY-001].

            """
            return rho0 * math.exp(-h / scale)

        return exponential
    if atmosphere == "us76":

        def standard(h: float) -> float:
            """Standard-atmosphere density.

            Args:
                h: Altitude in m.

            Returns:
                The density in kg/m^3.

            References:
                [REF-US-STANDARD-ATMOSPHERE-1976].

            """
            return _atmosphere_raw(h if h > 0.0 else 0.0)[2]

        return standard
    raise TypeError("atmosphere must be 'us76' or an ExponentialAtmosphere")


def sutton_graves_heat_flux(
    density: float,
    speed: float,
    nose_radius: float,
    *,
    constant: float = SUTTON_GRAVES_EARTH_K,
) -> float:
    """Convective stagnation-point heat flux ``k sqrt(rho / R_n) V^3`` in W/m^2.

    Args:
        density: Free-stream density ``rho`` in kg/m^3, ``>= 0``.
        speed: Speed ``V`` in m/s, ``>= 0``.
        nose_radius: Effective stagnation-point radius ``R_n`` in m, ``> 0``.
        constant: ``k`` in kg^0.5/m, ``> 0``.

    Returns:
        The cold-wall convective heat flux in W/m^2.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-ENTRY-002].

    References:
        [REF-SUTTON-GRAVES-1971].

    """
    rho = _non_negative(density, "density")
    v = _non_negative(speed, "speed")
    radius = _positive(nose_radius, "nose_radius")
    k = _positive(constant, "constant")
    return k * math.sqrt(rho / radius) * v**3


@dataclass(frozen=True, slots=True)
class EntryVehicle:
    """A capsule described by its ballistic coefficient (synthetic numbers).

    Args:
        ballistic_coefficient: ``beta = m / (C_D A)`` in kg/m^2, ``> 0``.
        nose_radius: Effective stagnation-point radius ``R_n`` in m, ``> 0``
            (for a spherical-cap heat shield, the cap radius).
        lift_to_drag: Constant ``L/D >= 0``; the lift acts "up" (toward
            increasing flight-path angle) with no bank.

    References:
        [EQ-AERO-ENTRY-001], [REF-ALLEN-EGGERS-1958].

    """

    ballistic_coefficient: float
    nose_radius: float
    lift_to_drag: float = 0.0

    def __post_init__(self) -> None:
        """Validate the vehicle.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If a field is out of range.

        References:
            [EQ-AERO-ENTRY-001]. No equation is evaluated.

        """
        _positive(self.ballistic_coefficient, "ballistic_coefficient")
        _positive(self.nose_radius, "nose_radius")
        _non_negative(self.lift_to_drag, "lift_to_drag")

    @classmethod
    def from_mass(
        cls,
        mass: float,
        drag_coefficient: float,
        reference_area: float,
        nose_radius: float,
        lift_to_drag: float = 0.0,
    ) -> EntryVehicle:
        """Build a vehicle from ``beta = m / (C_D A)``.

        Args:
            mass: Entry mass ``m`` in kg, ``> 0``.
            drag_coefficient: ``C_D > 0`` (hypersonic, constant).
            reference_area: Reference area ``A`` in m^2, ``> 0``.
            nose_radius: Stagnation-point radius in m.
            lift_to_drag: Constant ``L/D``.

        Returns:
            The vehicle.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If an argument is out of range.

        Equation:
            [EQ-AERO-ENTRY-001].

        References:
            [REF-ALLEN-EGGERS-1958].

        """
        beta = _positive(mass, "mass") / (
            _positive(drag_coefficient, "drag_coefficient")
            * _positive(reference_area, "reference_area")
        )
        return cls(beta, nose_radius, lift_to_drag)


@dataclass(frozen=True, slots=True)
class HeatShield:
    """Ablative heat-shield parameters (synthetic, engineering level).

    The thickness model of :func:`heat_shield_sizing` needs the following
    material constants; the values a user supplies are placeholders for a
    real material database.

    Args:
        ablator_density: Virgin-ablator density in kg/m^3, ``> 0``.
        heat_of_ablation: Effective heat of ablation in J/kg, ``> 0``: the
            convective heat load absorbed per unit mass lost, including
            blowing and re-radiation benefits.
        recession_safety_factor: Factor ``>= 1`` on the computed recession.
        thermal_diffusivity: Insulation diffusivity ``alpha`` in m^2/s.
        surface_temperature: Char-surface temperature ``T_s`` in K.
        bondline_temperature: Allowable structure/bond-line temperature in K.
        initial_temperature: Initial (cold-soak) temperature in K.
        design_thickness: Installed thickness in m, ``> 0``.

    References:
        [EQ-AERO-ENTRY-006], [EQ-AERO-ENTRY-007].

    """

    ablator_density: float
    heat_of_ablation: float
    recession_safety_factor: float
    thermal_diffusivity: float
    surface_temperature: float
    bondline_temperature: float
    initial_temperature: float
    design_thickness: float

    def __post_init__(self) -> None:
        """Validate the shield.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If a field is out of range or the temperatures are
                not ordered ``T_initial < T_bondline < T_surface``.

        References:
            [EQ-AERO-ENTRY-007]. No equation is evaluated.

        """
        _positive(self.ablator_density, "ablator_density")
        _positive(self.heat_of_ablation, "heat_of_ablation")
        if _positive(self.recession_safety_factor, "recession_safety_factor") < 1.0:
            raise ValueError("recession_safety_factor must be at least 1")
        _positive(self.thermal_diffusivity, "thermal_diffusivity")
        _positive(self.design_thickness, "design_thickness")
        t_s = _positive(self.surface_temperature, "surface_temperature")
        t_b = _positive(self.bondline_temperature, "bondline_temperature")
        t_i = _positive(self.initial_temperature, "initial_temperature")
        if not t_i < t_b < t_s:
            raise ValueError("temperatures must satisfy initial < bondline < surface")


@dataclass(frozen=True, slots=True)
class HeatShieldSizing:
    """Result of :func:`heat_shield_sizing`.

    Args:
        recession: Ablator recession ``s = f Q / (rho_a H_eff)`` in m.
        insulation_thickness: Penetration depth that keeps the bond line below
            its limit in m.
        required_thickness: ``recession + insulation_thickness`` in m.
        margin: ``design_thickness - required_thickness`` in m (negative
            means the design is short).
        heating_time: Equivalent heating time ``Q / q_peak`` in s.

    References:
        [EQ-AERO-ENTRY-006], [EQ-AERO-ENTRY-007], [EQ-AERO-ENTRY-008].

    """

    recession: float
    insulation_thickness: float
    required_thickness: float
    margin: float
    heating_time: float


def _erfc_inverse(p: float) -> float:
    """Return ``z > 0`` with ``erfc(z) = p`` by bisection, ``0 < p < 1``.

    Args:
        p: Target value of ``erfc``.

    Returns:
        The inverse complementary error function value.

    References:
        [EQ-AERO-ENTRY-007].

    """
    lo, hi = 0.0, 8.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def heat_shield_sizing(
    shield: HeatShield,
    heat_load: float,
    peak_heat_flux: float,
    *,
    heat_of_ablation: float | None = None,
) -> HeatShieldSizing:
    """Size an ablative heat shield from the convective heat load (engineering level).

    Two layers are added:

    1. Recession of the ablator, ``s = f Q / (rho_a H_eff)``, with the
       convective heat load ``Q`` (J/m^2), the safety factor ``f``, the
       ablator density ``rho_a`` and the effective heat of ablation ``H_eff``
       (J/kg). All of ``Q`` is assumed absorbed by ablation; blowing and
       re-radiation are folded into ``H_eff``.
    2. An insulation depth from the semi-infinite-solid conduction solution
       with the surface held at the char temperature ``T_s`` for the
       equivalent heating time ``t_h = Q / q_peak``: the temperature at depth
       ``x`` is ``T_i + (T_s - T_i) erfc(x / (2 sqrt(alpha t_h)))``, so the
       bond line stays at or below ``T_b`` when
       ``x >= 2 sqrt(alpha t_h) erfc^-1((T_b - T_i)/(T_s - T_i))``.

    Limits. The model is a first estimate with synthetic material
    parameters. The insulation depth neglects the adiabatic back face (which
    would warm the bond line more) and the motion of the surface, and a
    constant ``T_s`` over the whole pulse is a conservative surface history
    only against the front-face temperature. It is not a thermal analysis.

    Args:
        shield: Material parameters.
        heat_load: Convective heat load ``Q`` in J/m^2, ``>= 0``.
        peak_heat_flux: Peak heat flux ``q_peak`` in W/m^2, ``> 0`` when
            ``heat_load > 0``.
        heat_of_ablation: Override of ``shield.heat_of_ablation`` in J/kg
            (for example a sampled value), ``> 0``.

    Returns:
        The :class:`HeatShieldSizing`.

    Raises:
        TypeError: If ``shield`` is not a :class:`HeatShield` or a number is
            not real.
        ValueError: If a number is out of range.

    Equation:
        [EQ-AERO-ENTRY-006], [EQ-AERO-ENTRY-007], [EQ-AERO-ENTRY-008].

    References:
        [REF-SUTTON-GRAVES-1971], [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(shield, HeatShield):
        raise TypeError("shield must be a HeatShield")
    q_total = _non_negative(heat_load, "heat_load")
    q_peak = _non_negative(peak_heat_flux, "peak_heat_flux")
    h_eff = (
        shield.heat_of_ablation
        if heat_of_ablation is None
        else _positive(heat_of_ablation, "heat_of_ablation")
    )
    if q_total > 0.0 and q_peak <= 0.0:
        raise ValueError("peak_heat_flux must be positive when heat_load is positive")
    recession = (
        shield.recession_safety_factor * q_total / (shield.ablator_density * h_eff)
    )
    heating_time = q_total / q_peak if q_peak > 0.0 else 0.0
    theta = (shield.bondline_temperature - shield.initial_temperature) / (
        shield.surface_temperature - shield.initial_temperature
    )
    insulation = (
        2.0
        * math.sqrt(shield.thermal_diffusivity * heating_time)
        * _erfc_inverse(theta)
    )
    required = recession + insulation
    return HeatShieldSizing(
        recession,
        insulation,
        required,
        shield.design_thickness - required,
        heating_time,
    )


# ---------------------------------------------------------------------------
# Closed-form (Allen-Eggers) solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AllenEggersEntry:
    """Closed-form ballistic entry in an exponential atmosphere.

    Assumptions (all must hold for the formulas to be accurate): exponential
    atmosphere ``rho = rho_0 exp(-h/H)``; constant flight-path angle
    ``gamma`` (straight-line flight); gravity and the centrifugal term
    neglected; constant ``beta`` and no lift. With ``s = sin|gamma|`` and
    ``c = H / (beta s)``:

    - speed ``V = V_e exp(-rho H / (2 beta s))``, where ``V_e`` is the speed at
      infinite altitude;
    - drag deceleration ``a = rho V^2 / (2 beta)``, maximal at
      ``rho = 1/c`` with ``a_max = V_e^2 s / (2 e H)`` and ``V = V_e e^(-1/2)``;
    - heat flux ``q = k sqrt(rho/R_n) V^3``, maximal at ``rho = 1/(3c)``,
      i.e. ``H ln 3`` above the peak-deceleration altitude, with
      ``q_max = k V_e^3 e^(-1/2) sqrt(beta s / (3 H R_n))`` and
      ``V = V_e e^(-1/6)``;
    - heat load from ``rho_1`` to ``rho_2``:
      ``Q = k V_e^2 sqrt(pi beta H / (R_n s)) [erf(sqrt(c rho_2))
      - erf(sqrt(c rho_1))]``, which tends to
      ``k V_e^2 sqrt(pi beta H / (R_n s))`` from space to the ground.

    The steep-versus-shallow trade-off follows: peak deceleration grows with
    ``sin|gamma|`` while the heat load falls as ``1 / sqrt(sin|gamma|)``.

    Args:
        ballistic_coefficient: ``beta`` in kg/m^2.
        flight_path_angle: ``gamma`` in rad, ``-pi/2 <= gamma < 0``.
        entry_speed: Speed at ``entry_altitude`` in m/s.
        nose_radius: ``R_n`` in m.
        atmosphere: The exponential atmosphere.
        sutton_graves_constant: ``k`` in kg^0.5/m.
        entry_altitude: Altitude of ``entry_speed`` in m; ``math.inf`` (the
            default) means ``entry_speed`` is ``V_e``, the speed at infinity.

    References:
        [EQ-AERO-ENTRY-003], [EQ-AERO-ENTRY-004], [EQ-AERO-ENTRY-005],
        [REF-ALLEN-EGGERS-1958].

    """

    ballistic_coefficient: float
    flight_path_angle: float
    entry_speed: float
    nose_radius: float
    atmosphere: ExponentialAtmosphere = ExponentialAtmosphere()
    sutton_graves_constant: float = SUTTON_GRAVES_EARTH_K
    entry_altitude: float = math.inf

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If a parameter is out of range.

        References:
            [EQ-AERO-ENTRY-003]. No equation is evaluated.

        """
        _positive(self.ballistic_coefficient, "ballistic_coefficient")
        _entry_angle(self.flight_path_angle)
        _positive(self.entry_speed, "entry_speed")
        _positive(self.nose_radius, "nose_radius")
        _positive(self.sutton_graves_constant, "sutton_graves_constant")
        if not isinstance(self.atmosphere, ExponentialAtmosphere):
            raise TypeError("atmosphere must be an ExponentialAtmosphere")
        if not math.isinf(self.entry_altitude):
            require_finite_real(self.entry_altitude, name="entry_altitude")

    @property
    def _sin(self) -> float:
        """``sin|gamma|``.

        References:
            [EQ-AERO-ENTRY-003].

        """
        return abs(math.sin(self.flight_path_angle))

    @property
    def _c(self) -> float:
        """``c = H / (beta sin|gamma|)`` in m^3/kg.

        References:
            [EQ-AERO-ENTRY-003].

        """
        return self.atmosphere.scale_height / (self.ballistic_coefficient * self._sin)

    @property
    def infinity_speed(self) -> float:
        """Speed ``V_e`` at infinite altitude in m/s.

        Computed from ``entry_speed`` at ``entry_altitude`` by inverting
        ``V(h)``; equal to ``entry_speed`` when ``entry_altitude`` is infinite.

        References:
            [EQ-AERO-ENTRY-003].

        """
        if math.isinf(self.entry_altitude):
            return self.entry_speed
        rho_start = self.atmosphere.density(self.entry_altitude)
        return self.entry_speed * math.exp(0.5 * self._c * rho_start)

    def speed(self, altitude: float) -> float:
        """Speed ``V_e exp(-c rho / 2)`` in m/s at an altitude.

        Args:
            altitude: Altitude in m.

        Returns:
            The speed.

        Raises:
            TypeError: If ``altitude`` is not a real number.

        Equation:
            [EQ-AERO-ENTRY-003].

        References:
            [REF-ALLEN-EGGERS-1958].

        """
        rho = self.atmosphere.density(altitude)
        return self.infinity_speed * math.exp(-0.5 * self._c * rho)

    def deceleration(self, altitude: float) -> float:
        """Drag deceleration ``rho V^2 / (2 beta)`` in m/s^2 at an altitude.

        Args:
            altitude: Altitude in m.

        Returns:
            The deceleration.

        Raises:
            TypeError: If ``altitude`` is not a real number.

        Equation:
            [EQ-AERO-ENTRY-004].

        References:
            [REF-ALLEN-EGGERS-1958].

        """
        rho = self.atmosphere.density(altitude)
        v = self.speed(altitude)
        return rho * v * v / (2.0 * self.ballistic_coefficient)

    def heat_flux(self, altitude: float) -> float:
        """Sutton-Graves heat flux in W/m^2 along the closed-form trajectory.

        Args:
            altitude: Altitude in m.

        Returns:
            The convective heat flux.

        Raises:
            TypeError: If ``altitude`` is not a real number.

        Equation:
            [EQ-AERO-ENTRY-005].

        References:
            [REF-SUTTON-GRAVES-1971].

        """
        return sutton_graves_heat_flux(
            self.atmosphere.density(altitude),
            self.speed(altitude),
            self.nose_radius,
            constant=self.sutton_graves_constant,
        )

    @property
    def peak_deceleration(self) -> float:
        """Peak drag deceleration ``V_e^2 sin|gamma| / (2 e H)`` in m/s^2.

        References:
            [EQ-AERO-ENTRY-004].

        """
        return (
            self.infinity_speed**2
            * self._sin
            / (2.0 * math.e * self.atmosphere.scale_height)
        )

    @property
    def peak_deceleration_altitude(self) -> float:
        """Altitude ``H ln(rho_0 / rho*)``, ``rho* = beta sin|gamma| / H``, in m.

        References:
            [EQ-AERO-ENTRY-004].

        """
        rho_star = self.ballistic_coefficient * self._sin / self.atmosphere.scale_height
        return self.atmosphere.scale_height * math.log(
            self.atmosphere.surface_density / rho_star
        )

    @property
    def peak_deceleration_speed(self) -> float:
        """Speed ``V_e exp(-1/2)`` at the peak deceleration in m/s.

        References:
            [EQ-AERO-ENTRY-004].

        """
        return self.infinity_speed * math.exp(-0.5)

    @property
    def peak_heat_flux(self) -> float:
        """Peak heat flux ``k V_e^3 e^(-1/2) sqrt(beta s / (3 H R_n))`` in W/m^2.

        References:
            [EQ-AERO-ENTRY-005].

        """
        return (
            self.sutton_graves_constant
            * self.infinity_speed**3
            * math.exp(-0.5)
            * math.sqrt(
                self.ballistic_coefficient
                * self._sin
                / (3.0 * self.atmosphere.scale_height * self.nose_radius)
            )
        )

    @property
    def peak_heat_flux_altitude(self) -> float:
        """Altitude of peak heating, ``H ln 3`` above the peak deceleration, in m.

        References:
            [EQ-AERO-ENTRY-005].

        """
        return (
            self.peak_deceleration_altitude
            + self.atmosphere.scale_height * math.log(3.0)
        )

    @property
    def peak_heat_flux_speed(self) -> float:
        """Speed ``V_e exp(-1/6)`` at peak heating in m/s.

        References:
            [EQ-AERO-ENTRY-005].

        """
        return self.infinity_speed * math.exp(-1.0 / 6.0)

    def heat_load(self, altitude_end: float) -> float:
        """Convective heat load in J/m^2 down to ``altitude_end``.

        Args:
            altitude_end: Final altitude in m (below the entry altitude).

        Returns:
            ``k V_e^2 sqrt(pi beta H / (R_n s)) [erf(sqrt(c rho_end)) -
            erf(sqrt(c rho_start))]`` with ``rho_start = 0`` for an infinite
            entry altitude.

        Raises:
            TypeError: If ``altitude_end`` is not a real number.

        Equation:
            [EQ-AERO-ENTRY-005].

        References:
            [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971].

        """
        rho_end = self.atmosphere.density(altitude_end)
        rho_start = (
            0.0
            if math.isinf(self.entry_altitude)
            else self.atmosphere.density(self.entry_altitude)
        )
        c = self._c
        prefactor = (
            self.sutton_graves_constant
            * self.infinity_speed**2
            * math.sqrt(
                math.pi
                * self.ballistic_coefficient
                * self.atmosphere.scale_height
                / (self.nose_radius * self._sin)
            )
        )
        return prefactor * (
            math.erf(math.sqrt(c * rho_end)) - math.erf(math.sqrt(c * rho_start))
        )


# ---------------------------------------------------------------------------
# Trajectory integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryResult:
    """Outcome of :func:`simulate_entry`.

    Args:
        status: ``"landed"`` (the terminal altitude was reached),
            ``"skip_out"`` (the vehicle climbed back above the entry
            altitude), ``"timeout"`` (``max_time`` exceeded) or
            ``"numerical_failure"``.
        stop_reason: Human-readable explanation.
        time: Time of the final state in s.
        downrange_angle: Central angle travelled in rad.
        downrange: ``R_E * downrange_angle`` in m (arc length at sea level).
        altitude: Final altitude in m.
        speed: Final speed in m/s.
        flight_path_angle: Final flight-path angle in rad.
        peak_deceleration_g: Peak total aerodynamic load factor
            ``(D/m) sqrt(1 + (L/D)^2) / g0``.
        peak_deceleration_altitude: Altitude of that peak in m.
        peak_deceleration_speed: Speed at that peak in m/s.
        peak_heat_flux: Peak Sutton-Graves heat flux in W/m^2.
        peak_heat_flux_altitude: Altitude of that peak in m.
        heat_load: ``int q dt`` in J/m^2 up to the final state.
        peaks_interior: ``False`` when a peak occurred at the last grid point
            (the maximum was not passed before the run ended), in which case
            the peak values are lower bounds.
        steps: Number of integration steps of the uniform grid.
        trajectory: ``(t, h, V, gamma, theta)`` grid points when recording was
            requested, else empty.

    References:
        [EQ-AERO-ENTRY-001], [EQ-AERO-ENTRY-002], [EQ-AERO-ENTRY-011].

    """

    status: EntryStatus
    stop_reason: str
    time: float
    downrange_angle: float
    downrange: float
    altitude: float
    speed: float
    flight_path_angle: float
    peak_deceleration_g: float
    peak_deceleration_altitude: float
    peak_deceleration_speed: float
    peak_heat_flux: float
    peak_heat_flux_altitude: float
    heat_load: float
    peaks_interior: bool
    steps: int
    trajectory: tuple[tuple[float, float, float, float, float], ...]


def _make_entry_rhs(
    density: Callable[[float], float],
    *,
    beta: float,
    nose_radius: float,
    lift_to_drag: float,
    density_scale: float,
    heating_constant: float,
    idealised: bool,
) -> Callable[[float, Sequence[float]], Sequence[float]]:
    """Build the right-hand side of the planar entry equations.

    Args:
        density: Density function ``rho(h)`` in kg/m^3 before scaling.
        beta: Ballistic coefficient in kg/m^2.
        nose_radius: Stagnation-point radius in m.
        lift_to_drag: Constant ``L/D``.
        density_scale: Multiplier on the density.
        heating_constant: Effective Sutton-Graves constant.
        idealised: Drop gravity and the curvature term (verification only).

    Returns:
        A function ``f(t, y)`` for :func:`integrate_fixed_step`.

    References:
        [EQ-AERO-ENTRY-001], [EQ-AERO-ENTRY-002].

    """
    inv_two_beta = 1.0 / (2.0 * beta)
    heat_factor = heating_constant / math.sqrt(nose_radius)

    def rhs(t: float, y: Sequence[float]) -> Sequence[float]:
        """Evaluate the planar entry equations of motion.

        Args:
            t: Time (unused; the system is autonomous).
            y: State ``(theta, h, V, gamma, Q)``.

        Returns:
            The state derivative.

        Raises:
            ValueError: If the speed is not positive.

        References:
            [EQ-AERO-ENTRY-001], [EQ-AERO-ENTRY-002].

        """
        h = y[_STATE_H]
        v = y[_STATE_V]
        gam = y[_STATE_G]
        if v <= 0.0:
            raise ValueError("speed is not positive")
        r = EARTH_RADIUS + h
        rho = density_scale * density(h)
        drag = rho * v * v * inv_two_beta
        sin_g = math.sin(gam)
        cos_g = math.cos(gam)
        if idealised:
            grav = 0.0
            curve = 0.0
        else:
            grav = EARTH_MU / (r * r)
            curve = v / r
        return (
            v * cos_g / r,
            v * sin_g,
            -drag - grav * sin_g,
            lift_to_drag * drag / v + (curve - grav / v) * cos_g,
            heat_factor * math.sqrt(rho) * v**3,
        )

    return rhs


def _parabolic_peak(values: Sequence[float], index: int) -> tuple[float, float]:
    """Refine a discrete interior maximum by a parabola through three points.

    Args:
        values: Samples on a uniform grid.
        index: Interior index of the discrete maximum (``0 < index < n - 1``).

    Returns:
        ``(offset, peak)``: the vertex offset from ``index`` in grid units
        (within ``[-1/2, 1/2]`` at a true discrete maximum) and the value of
        the parabola there.

    References:
        [EQ-AERO-ENTRY-011].

    """
    left = values[index - 1]
    mid = values[index]
    right = values[index + 1]
    curvature = left - 2.0 * mid + right
    if curvature >= 0.0:
        return 0.0, mid
    offset = 0.5 * (left - right) / curvature
    return offset, mid - 0.25 * (left - right) * offset


def _interpolate(values: Sequence[float], index: int, offset: float) -> float:
    """Quadratic interpolation of a uniform-grid series at ``index + offset``.

    Args:
        values: Samples on a uniform grid.
        index: Interior centre index.
        offset: Offset from ``index`` in grid units.

    Returns:
        The interpolated value.

    References:
        [EQ-AERO-ENTRY-011].

    """
    left = values[index - 1]
    mid = values[index]
    right = values[index + 1]
    return (
        mid
        + 0.5 * offset * (right - left)
        + 0.5 * offset * offset * (right - 2.0 * mid + left)
    )


def _locate_terminal(
    rhs: Callable[[float, Sequence[float]], Sequence[float]],
    t0: float,
    y0: Sequence[float],
    width: float,
    target: float,
) -> tuple[float, tuple[float, ...]]:
    """Locate ``h = target`` inside one step by bisection of the sub-step.

    Args:
        rhs: Right-hand side.
        t0: Time of the last state above the target.
        y0: That state.
        width: Step length at which the target was passed.
        target: Terminal altitude in m.

    Returns:
        ``(time, state)`` at the crossing, ``|h - target| < 1e-4`` m or after
        80 bisections.

    References:
        [EQ-AERO-ENTRY-001].

    """
    lo, hi = 0.0, width
    best: tuple[float, ...] = tuple(y0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        state = integrate_fixed_step(rhs, t0, y0, step=mid, steps=1).final_state
        if state[_STATE_H] > target:
            lo = mid
            best = tuple(state)
            if state[_STATE_H] - target < 1e-4:
                break
        else:
            hi = mid
    return t0 + lo, best


def simulate_entry(
    vehicle: EntryVehicle,
    *,
    altitude: float,
    speed: float,
    flight_path_angle: float,
    atmosphere: ExponentialAtmosphere | Literal["us76"] = "us76",
    density_scale: float = 1.0,
    heating_scale: float = 1.0,
    sutton_graves_constant: float = SUTTON_GRAVES_EARTH_K,
    terminal_altitude: float = 10_000.0,
    max_step: float = 2.0,
    resolution: float = 0.2,
    max_time: float = 4000.0,
    idealised: bool = False,
    record: bool = False,
) -> EntryResult:
    """Integrate the planar entry from the entry interface to a terminal altitude.

    With ``r = R_E + h``, ``g = mu / r^2``, ``D/m = rho V^2 / (2 beta)``
    ([EQ-AERO-ENTRY-001]),

    ``theta' = V cos(gamma) / r``, ``h' = V sin(gamma)``,
    ``V' = -D/m - g sin(gamma)``,
    ``gamma' = (L/D)(D/m)/V + (V/r - g/V) cos(gamma)``,

    and the heat load ``Q' = k sqrt(rho / R_n) V^3`` ([EQ-AERO-ENTRY-002]).
    The run stops when the altitude falls to ``terminal_altitude`` ("landed":
    for a capsule, the parachute-deployment interface, not touchdown), when the
    altitude climbs back above the entry altitude ("skip_out"), or when
    ``max_time`` is exceeded ("timeout"). Physical failures are reported in
    ``status``, never raised; invalid arguments raise.

    The step is ``min(max_step, resolution * 7200 m / (V |sin gamma|))`` at the
    entry state, so steep entries get a finer grid. RK4 gives a global error of
    order ``step^4`` in smooth segments; the peak refinement is third order.

    Args:
        vehicle: The capsule.
        altitude: Entry-interface altitude in m (above ``terminal_altitude``).
        speed: Entry speed in m/s, ``> 0``.
        flight_path_angle: Entry flight-path angle in rad, ``-pi/2 <= gamma < 0``.
        atmosphere: ``"us76"`` or an :class:`ExponentialAtmosphere`.
        density_scale: Multiplier on the density (``> 0``).
        heating_scale: Multiplier on the Sutton-Graves constant (``> 0``).
        sutton_graves_constant: ``k`` in kg^0.5/m.
        terminal_altitude: Stop altitude in m, ``>= 0``.
        max_step: Largest step in s.
        resolution: Altitude change per step as a fraction of 7200 m at the
            entry rate of descent, ``> 0``.
        max_time: Time limit in s.
        idealised: Drop gravity and curvature (straight-line flight) for
            comparison with :class:`AllenEggersEntry`; requires ``L/D = 0``.
        record: Keep the grid points in ``trajectory``.

    Returns:
        The :class:`EntryResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a number is out of range or ``idealised`` is combined
            with lift.

    Equation:
        [EQ-AERO-ENTRY-001], [EQ-AERO-ENTRY-002], [EQ-AERO-ENTRY-011].

    References:
        [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971], [REF-CURTIS-2014],
        [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    if not isinstance(vehicle, EntryVehicle):
        raise TypeError("vehicle must be an EntryVehicle")
    h0 = require_finite_real(altitude, name="altitude")
    v0 = _positive(speed, "speed")
    g0 = _entry_angle(flight_path_angle)
    k_rho = _positive(density_scale, "density_scale")
    k_heat = _positive(heating_scale, "heating_scale")
    k_sg = _positive(sutton_graves_constant, "sutton_graves_constant")
    h_end = _non_negative(terminal_altitude, "terminal_altitude")
    if h0 <= h_end:
        raise ValueError("altitude must exceed terminal_altitude")
    step_cap = _positive(max_step, "max_step")
    res = _positive(resolution, "resolution")
    t_max = _positive(max_time, "max_time")
    if not isinstance(idealised, bool) or not isinstance(record, bool):
        raise TypeError("idealised and record must be bool")
    if idealised and vehicle.lift_to_drag != 0.0:
        raise ValueError("idealised straight-line flight requires lift_to_drag = 0")
    density = _density_function(atmosphere)
    rhs = _make_entry_rhs(
        density,
        beta=vehicle.ballistic_coefficient,
        nose_radius=vehicle.nose_radius,
        lift_to_drag=vehicle.lift_to_drag,
        density_scale=k_rho,
        heating_constant=k_sg * k_heat,
        idealised=idealised,
    )
    dt = min(step_cap, res * _H_REF / (v0 * abs(math.sin(g0))))
    load_factor = math.sqrt(1.0 + vehicle.lift_to_drag**2) / (
        2.0 * vehicle.ballistic_coefficient * STANDARD_GRAVITY
    )
    heat_factor = (
        k_sg * k_heat / math.sqrt(vehicle.nose_radius)
    )  # q = heat_factor sqrt(rho) V^3

    def diagnostics(state: Sequence[float]) -> tuple[float, float]:
        """Return ``(load factor in g, heat flux in W/m^2)`` of a state.

        Args:
            state: ``(theta, h, V, gamma, Q)``.

        Returns:
            The load factor and the stagnation-point heat flux.

        References:
            [EQ-AERO-ENTRY-001], [EQ-AERO-ENTRY-002].

        """
        rho = k_rho * density(state[_STATE_H])
        v = state[_STATE_V]
        return load_factor * rho * v * v, heat_factor * math.sqrt(rho) * v**3

    state0: Sequence[float] = (0.0, h0, v0, g0, 0.0)
    grid: list[Sequence[float]] = [state0]
    status: EntryStatus = "landed"
    reason = "terminal altitude reached"
    final: Sequence[float] = state0
    final_time = 0.0
    off_grid_end = False
    running = True
    while running:
        try:
            chunk = integrate_fixed_step(
                rhs, (len(grid) - 1) * dt, grid[-1], step=dt, steps=_CHUNK
            )
        except ValueError as error:
            status, reason = "numerical_failure", str(error)
            final, final_time, running = grid[-1], (len(grid) - 1) * dt, False
            continue
        for index in range(1, _CHUNK + 1):
            current = chunk.states[index]
            time = len(grid) * dt
            if current[_STATE_H] <= h_end:
                final_time, final = _locate_terminal(
                    rhs, chunk.times[index - 1], chunk.states[index - 1], dt, h_end
                )
                off_grid_end, running = True, False
                break
            grid.append(current)
            if current[_STATE_H] > h0:
                status = "skip_out"
                reason = "the vehicle climbed back above the entry altitude"
                final, final_time, running = current, time, False
                break
            if time >= t_max:
                status = "timeout"
                reason = "max_time exceeded before the terminal altitude"
                final, final_time, running = current, time, False
                break
    loads = [diagnostics(s)[0] for s in grid]
    fluxes = [diagnostics(s)[1] for s in grid]
    end_load, end_flux = diagnostics(final)
    heights = [s[_STATE_H] for s in grid]
    speeds = [s[_STATE_V] for s in grid]
    interior = True

    def peak(series: list[float], end_value: float) -> tuple[float, float, float]:
        """Peak of a grid series with its altitude and speed.

        Args:
            series: Samples on the uniform grid.
            end_value: Value at the final state (which is off the uniform
                grid when the run ended at the terminal altitude).

        Returns:
            ``(peak, altitude, speed)``; the parabola refines interior maxima.

        References:
            [EQ-AERO-ENTRY-011].

        """
        nonlocal interior
        best = max(range(len(series)), key=series.__getitem__)
        if off_grid_end and end_value > series[best]:
            interior = False
            return end_value, final[_STATE_H], final[_STATE_V]
        last = len(series) - 1
        if 0 < best < last:
            offset, value = _parabolic_peak(series, best)
            return (
                value,
                _interpolate(heights, best, offset),
                _interpolate(speeds, best, offset),
            )
        if best == 0 or (best == last and not off_grid_end):
            interior = False
        return series[best], heights[best], speeds[best]

    peak_load, load_h, load_v = peak(loads, end_load)
    peak_flux, flux_h, _flux_v = peak(fluxes, end_flux)
    trajectory: tuple[tuple[float, float, float, float, float], ...] = ()
    if record:
        trajectory = tuple(
            (i * dt, s[_STATE_H], s[_STATE_V], s[_STATE_G], s[_STATE_TH])
            for i, s in enumerate(grid)
        )
    return EntryResult(
        status,
        reason,
        final_time,
        final[_STATE_TH],
        EARTH_RADIUS * final[_STATE_TH],
        final[_STATE_H],
        final[_STATE_V],
        final[_STATE_G],
        peak_load,
        load_h,
        load_v,
        peak_flux,
        flux_h,
        final[_STATE_Q],
        interior,
        len(grid) - 1,
        trajectory,
    )


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_INPUT_UNITS = {
    "flight_path_angle": "rad",
    "entry_speed": "m/s",
    "ballistic_coefficient": "kg/m^2",
    "nose_radius": "m",
    "density_scale": "-",
    "heating_scale": "-",
    "heat_of_ablation": "J/kg",
}

_QUANTITY_TABLE = (
    ("peak_deceleration_g", "Peak aerodynamic load factor", "g0"),
    ("peak_deceleration_altitude", "Altitude of the peak load factor", "m"),
    ("peak_heat_flux", "Peak Sutton-Graves convective heat flux", "W/m^2"),
    ("peak_heat_flux_altitude", "Altitude of the peak heat flux", "m"),
    ("heat_load", "Integrated convective heat load", "J/m^2"),
    ("heating_time", "Equivalent heating time Q / q_peak", "s"),
    ("recession", "Ablator recession with safety factor", "m"),
    ("insulation_thickness", "Insulation depth for the bond-line limit", "m"),
    ("required_thickness", "Required heat-shield thickness", "m"),
    ("thickness_margin", "Design thickness minus required thickness", "m"),
    ("downrange", "Downrange arc length at the terminal altitude", "m"),
    ("flight_time", "Time from the entry interface to the terminal altitude", "s"),
    ("terminal_speed", "Speed at the terminal altitude", "m/s"),
)


@dataclass(frozen=True, slots=True)
class _EntryForward:
    """Picklable forward map of :func:`entry_model`.

    Args:
        vehicle: Nominal vehicle (supplies the lift-to-drag ratio).
        shield: Heat-shield parameters.
        altitude: Entry-interface altitude in m.
        atmosphere: Atmosphere choice.
        terminal_altitude: Terminal altitude in m.
        max_step: Step bound in s.
        resolution: Step resolution in scale heights.

    References:
        [EQ-AERO-ENTRY-009].

    """

    vehicle: EntryVehicle
    shield: HeatShield
    altitude: float
    atmosphere: ExponentialAtmosphere | Literal["us76"]
    terminal_altitude: float
    max_step: float
    resolution: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Run one entry for named parameter values.

        Args:
            parameters: The seven parameters of :func:`entry_model`.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If a parameter is out of its domain.
            RuntimeError: If the vehicle does not reach the terminal altitude
                (skip-out, time-out or numerical failure).

        Equation:
            [EQ-AERO-ENTRY-009].

        References:
            [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971].

        """
        vehicle = EntryVehicle(
            parameters["ballistic_coefficient"],
            parameters["nose_radius"],
            self.vehicle.lift_to_drag,
        )
        result = simulate_entry(
            vehicle,
            altitude=self.altitude,
            speed=parameters["entry_speed"],
            flight_path_angle=parameters["flight_path_angle"],
            atmosphere=self.atmosphere,
            density_scale=parameters["density_scale"],
            heating_scale=parameters["heating_scale"],
            terminal_altitude=self.terminal_altitude,
            max_step=self.max_step,
            resolution=self.resolution,
        )
        if result.status != "landed":
            raise RuntimeError(f"entry failed: {result.status}: {result.stop_reason}")
        sizing = heat_shield_sizing(
            self.shield,
            result.heat_load,
            result.peak_heat_flux,
            heat_of_ablation=parameters["heat_of_ablation"],
        )
        return {
            "peak_deceleration_g": result.peak_deceleration_g,
            "peak_deceleration_altitude": result.peak_deceleration_altitude,
            "peak_heat_flux": result.peak_heat_flux,
            "peak_heat_flux_altitude": result.peak_heat_flux_altitude,
            "heat_load": result.heat_load,
            "heating_time": sizing.heating_time,
            "recession": sizing.recession,
            "insulation_thickness": sizing.insulation_thickness,
            "required_thickness": sizing.required_thickness,
            "thickness_margin": sizing.margin,
            "downrange": result.downrange,
            "flight_time": result.time,
            "terminal_speed": result.speed,
        }


def entry_model(
    vehicle: EntryVehicle,
    shield: HeatShield,
    *,
    entry_speed: Marginal | float,
    flight_path_angle: Marginal | float,
    entry_altitude: float = 120_000.0,
    ballistic_coefficient: Marginal | float | None = None,
    nose_radius: Marginal | float | None = None,
    density_scale: Marginal | float = 1.0,
    heating_scale: Marginal | float = 1.0,
    heat_of_ablation: Marginal | float | None = None,
    atmosphere: ExponentialAtmosphere | Literal["us76"] = "us76",
    terminal_altitude: float = 10_000.0,
    max_step: float = 2.0,
    resolution: float = 0.2,
    name: str = "entry",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic entry model for heating and heat-shield analysis.

    Each argument is a distribution (an uncertain input, wrapped as an
    independent :class:`~probstat.modeling.RandomInput`) or a plain number (a
    fixed parameter). ``ballistic_coefficient``, ``nose_radius`` and
    ``heat_of_ablation`` default to the values of ``vehicle`` and ``shield``;
    the ballistic coefficient ``m / (C_D A)`` carries the ``C_D`` uncertainty
    (``beta`` scales as ``1 / C_D``). ``density_scale`` multiplies the
    atmospheric density and ``heating_scale`` multiplies the Sutton-Graves
    constant. The lift-to-drag ratio and the shield's other parameters are
    fixed.

    Quantities of interest: ``peak_deceleration_g``,
    ``peak_deceleration_altitude``, ``peak_heat_flux``,
    ``peak_heat_flux_altitude``, ``heat_load``, ``heating_time``,
    ``recession``, ``insulation_thickness``, ``required_thickness``,
    ``thickness_margin`` (design minus required), ``downrange``,
    ``flight_time`` and ``terminal_speed``. A run that skips out of the
    atmosphere, times out or fails numerically raises ``RuntimeError`` and is
    handled by ``on_failure`` (``"raise"``, ``"skip"`` or ``"record"``), so a
    wide flight-path-angle dispersion that straddles the skip-out boundary is
    a failure, not a silent extrapolation. Vehicle and material numbers are
    synthetic.

    Args:
        vehicle: Nominal vehicle.
        shield: Heat-shield parameters.
        entry_speed: Entry speed in m/s.
        flight_path_angle: Entry flight-path angle in rad (negative).
        entry_altitude: Entry-interface altitude in m (fixed).
        ballistic_coefficient: ``beta`` in kg/m^2; ``None`` uses the vehicle's.
        nose_radius: ``R_n`` in m; ``None`` uses the vehicle's.
        density_scale: Scale on the atmospheric density.
        heating_scale: Scale on the Sutton-Graves constant.
        heat_of_ablation: ``H_eff`` in J/kg; ``None`` uses the shield's.
        atmosphere: ``"us76"`` or an :class:`ExponentialAtmosphere`.
        terminal_altitude: Terminal altitude in m.
        max_step: Largest step in s.
        resolution: Step resolution in scale heights.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is outside its domain or no input is
            random.

    Equation:
        [EQ-AERO-ENTRY-009].

    References:
        [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971], [REF-AU-BECK-2001].

    """
    if not isinstance(vehicle, EntryVehicle):
        raise TypeError("vehicle must be an EntryVehicle")
    if not isinstance(shield, HeatShield):
        raise TypeError("shield must be a HeatShield")
    given: dict[str, Marginal | float] = {
        "flight_path_angle": flight_path_angle,
        "entry_speed": entry_speed,
        "ballistic_coefficient": (
            vehicle.ballistic_coefficient
            if ballistic_coefficient is None
            else ballistic_coefficient
        ),
        "nose_radius": vehicle.nose_radius if nose_radius is None else nose_radius,
        "density_scale": density_scale,
        "heating_scale": heating_scale,
        "heat_of_ablation": (
            shield.heat_of_ablation if heat_of_ablation is None else heat_of_ablation
        ),
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
    forward = _EntryForward(
        vehicle,
        shield,
        _positive(entry_altitude, "entry_altitude"),
        atmosphere,
        _non_negative(terminal_altitude, "terminal_altitude"),
        _positive(max_step, "max_step"),
        _positive(resolution, "resolution"),
    )
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        forward,
        tuple(QuantityOfInterest(n, d, u) for n, d, u in _QUANTITY_TABLE),
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# ---------------------------------------------------------------------------
# Entry corridor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryCorridor:
    """Entry-corridor bounds on the flight-path angle at fixed speed.

    Args:
        shallow_angle: Shallowest capturing angle in rad: the landing side of
            the skip-out boundary found by bisection (negative).
        steep_angle: Steepest angle in rad whose peak load factor does not
            exceed the limit (negative).
        width: ``shallow_angle - steep_angle`` in rad (the corridor width).
        g_limit: The load-factor limit in ``g0``.
        peak_g_at_shallow: Peak load factor at the shallow bound.
        peak_g_at_steep: Peak load factor at the steep bound.
        heat_load_at_shallow: Heat load at the shallow bound in J/m^2.
        heat_load_at_steep: Heat load at the steep bound in J/m^2.
        evaluations: Number of trajectory integrations used.

    References:
        [EQ-AERO-ENTRY-010].

    """

    shallow_angle: float
    steep_angle: float
    width: float
    g_limit: float
    peak_g_at_shallow: float
    peak_g_at_steep: float
    heat_load_at_shallow: float
    heat_load_at_steep: float
    evaluations: int


def entry_corridor(
    vehicle: EntryVehicle,
    *,
    entry_speed: float,
    g_limit: float,
    entry_altitude: float = 120_000.0,
    shallow_bracket: float = math.radians(-1.0),
    steep_bracket: float = math.radians(-40.0),
    tolerance: float = 1e-5,
    atmosphere: ExponentialAtmosphere | Literal["us76"] = "us76",
    density_scale: float = 1.0,
    terminal_altitude: float = 10_000.0,
    max_step: float = 2.0,
    resolution: float = 0.2,
) -> EntryCorridor:
    """Find the skip-out and load-limit bounds of the entry corridor by bisection.

    For a fixed entry speed the flight-path angle is bracketed between a
    shallow angle that fails to capture (status other than ``"landed"``) and a
    steep angle whose peak load factor exceeds ``g_limit``. Two bisections on
    ``|gamma|`` then give

    - the skip-out boundary: the smallest ``|gamma|`` that still lands, and
    - the load-limit boundary: the largest ``|gamma|`` with peak load factor
      at most ``g_limit``,

    each to ``tolerance`` rad. Both rely on monotonicity in ``|gamma|``
    (landing is monotone for a vehicle that flies a single pass, and the peak
    load factor increases with the entry angle, as in the Allen-Eggers law
    ``a_max proportional to sin|gamma|``); the bracket ends are checked and a
    violated assumption raises instead of returning a wrong bound.

    Args:
        vehicle: The capsule.
        entry_speed: Entry speed in m/s.
        g_limit: Allowed peak load factor in ``g0``, ``> 0``.
        entry_altitude: Entry-interface altitude in m.
        shallow_bracket: Shallow bracket end in rad (negative); must fail to
            capture.
        steep_bracket: Steep bracket end in rad (negative, steeper than the
            shallow end); must exceed ``g_limit``.
        tolerance: Bisection tolerance in rad, ``> 0``.
        atmosphere: ``"us76"`` or an exponential atmosphere.
        density_scale: Density multiplier.
        terminal_altitude: Terminal altitude in m.
        max_step: Largest step in s.
        resolution: Step resolution in scale heights.

    Returns:
        The :class:`EntryCorridor`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the brackets are not ordered or do not bracket the
            boundaries (the shallow end captures, or the steep end is still
            below the limit, or the steep end fails to land).

    Equation:
        [EQ-AERO-ENTRY-010].

    References:
        [REF-ALLEN-EGGERS-1958], [REF-CURTIS-2014].

    """
    if not isinstance(vehicle, EntryVehicle):
        raise TypeError("vehicle must be an EntryVehicle")
    limit = _positive(g_limit, "g_limit")
    tol = _positive(tolerance, "tolerance")
    shallow = _entry_angle(shallow_bracket, "shallow_bracket")
    steep = _entry_angle(steep_bracket, "steep_bracket")
    if not steep < shallow:
        raise ValueError("steep_bracket must be steeper than shallow_bracket")
    count = 0

    def run(angle: float) -> EntryResult:
        """Integrate one entry at a flight-path angle.

        Args:
            angle: Flight-path angle in rad.

        Returns:
            The entry result.

        References:
            [EQ-AERO-ENTRY-010].

        """
        nonlocal count
        count += 1
        return simulate_entry(
            vehicle,
            altitude=entry_altitude,
            speed=entry_speed,
            flight_path_angle=angle,
            atmosphere=atmosphere,
            density_scale=density_scale,
            terminal_altitude=terminal_altitude,
            max_step=max_step,
            resolution=resolution,
        )

    if run(shallow).status == "landed":
        raise ValueError("shallow_bracket captures: there is no skip-out boundary")
    steep_run = run(steep)
    if steep_run.status != "landed":
        raise ValueError(f"steep_bracket does not land ({steep_run.status})")
    if steep_run.peak_deceleration_g <= limit:
        raise ValueError("steep_bracket does not exceed g_limit")
    # Skip-out boundary: lo skips, hi lands (angles as positive magnitudes).
    lo, hi = -shallow, -steep
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if run(-mid).status == "landed":
            hi = mid
        else:
            lo = mid
    shallow_angle = -hi
    shallow_run = run(shallow_angle)
    if shallow_run.peak_deceleration_g > limit:
        raise ValueError("the skip-out boundary already exceeds g_limit: no corridor")
    # Load-limit boundary: lo is within the limit, hi exceeds it.
    lo, hi = hi, -steep
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        result = run(-mid)
        if result.status == "landed" and result.peak_deceleration_g <= limit:
            lo = mid
        else:
            hi = mid
    steep_angle = -lo
    steep_run = run(steep_angle)
    return EntryCorridor(
        shallow_angle,
        steep_angle,
        shallow_angle - steep_angle,
        limit,
        shallow_run.peak_deceleration_g,
        steep_run.peak_deceleration_g,
        shallow_run.heat_load,
        steep_run.heat_load,
        count,
    )
