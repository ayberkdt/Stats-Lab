"""Uncontrolled-reentry casualty risk: fragment demise, casualty area, expectation.

A synthetic engineering-level chain from a break-up event to an expected
number of casualties: each fragment released at a break-up altitude follows a
ballistic entry trajectory, is heated by a lumped (single-temperature) heat
balance and either demises or reaches the ground with an impact energy; the
survivors define a casualty area, which times a population density is the
casualty expectation. All quantities are SI (m, s, kg, W, J, K, rad) except
the population density, which is in persons per km^2 as demographic tables
are quoted.

Model:

- Trajectory: the planar point-mass entry equations of
  :mod:`probstat.aerospace.entry` (Allen-Eggers / Curtis form, non-rotating
  spherical Earth, 1976 standard atmosphere or an exponential atmosphere),
  with the fragment's ballistic coefficient ``beta = m / (C_D A)``. The break-up
  state (altitude, speed, flight-path angle) is an input; tumbling is
  represented by a constant mean ``C_D`` and the mean projected area
  ``A = S / 4`` (Cauchy's mean-projection theorem for a convex body of surface
  area ``S``).
- Heating: the Sutton-Graves stagnation correlation
  ``q = k sqrt(rho / R_n) V^3`` ([REF-SUTTON-GRAVES-1971]) with the equivalent
  sphere radius ``R_n = sqrt(S / 4 pi)``, multiplied by a dimensionless
  ``heating_scale`` (uncertainty of the correlation) and a per-fragment
  ``heating_shape_factor``, acting on the mean projected area ``S / 4``
  (:data:`EQ-AERO-CASUALTY-002`). Re-radiation leaves the whole surface at
  ``epsilon sigma S T^4`` (no ambient or view-factor term).
- Lumped thermal response (:data:`EQ-AERO-CASUALTY-001` and ``-003``): the
  absorbed energy ``E`` obeys ``dE/dt = P_in - P_rad``. The temperature is
  ``T = T_0 + E / (m c)`` up to the melt temperature, ``T = T_m`` while the
  heat of fusion is absorbed (``m c (T_m - T_0) <= E <= E_dem``) and the
  fragment *demises* when ``E`` reaches ``E_dem = m [c (T_m - T_0) + L_f]``.
  Beyond ``E_dem`` the code keeps integrating with ``T = T_m + (E - E_dem) /
  (m c)`` only so that the smooth demise margin ``1 - max E / E_dem`` is
  defined; the temperature above ``T_m`` is a bookkeeping continuation, not a
  physical result.
- Assumptions of the lumped model (all of them idealisations, none is
  validated here): uniform temperature (no conduction gradient, so thick
  objects demise too easily and thin shells too early relative to a
  distributed model), no oxidation heat release, no spallation or ablative
  mass loss before demise (a surviving fragment keeps its full mass, which is
  conservative for the impact energy), no break-up cascade during descent,
  constant specific heat, no fragment melting-run-off, no shielding of one
  fragment by another, the whole fragment is at a single trajectory (no
  separate aerodynamic tumbling).
- Impact energy ``E_k = m V_i^2 / 2`` at the ground speed ``V_i``
  (:data:`EQ-AERO-CASUALTY-004`); parachute-like terminal-velocity effects
  are those of the drag law, not of a separate model.
- Casualty area (:data:`EQ-AERO-CASUALTY-005`): a surviving fragment counts
  when ``E_k >= E_min`` (the energy threshold, a user input), with
  ``A_c = (sqrt(A_p) + sqrt(A_f))^2`` where ``A_p`` is a person's
  projected area (a user input) and ``A_f`` the fragment's mean projected
  area; the total is the sum over the surviving fragments times their
  multiplicity. The casualty expectation is ``E_c = A_c rho_pop``
  (:data:`EQ-AERO-CASUALTY-006`) with ``rho_pop`` the mean population density
  (persons per m^2) under the ground track, a user input.

Everything that carries a standard or an empirical value is a user input: the
casualty-expectation threshold (commonly quoted as ``1e-4`` and *to be checked
against the governing standard*, which is not cited here because no standard
was verified for this module), the energy threshold, the person area, the
population density and every material property (melt temperature, heat of
fusion, specific heat, emissivity). The demonstration set returned by
:func:`demonstration_fragments` and :func:`demonstration_materials` is
SYNTHETIC: round numbers of the right order of magnitude, not verified
against a property database and not data of a real object.

The survival of a fragment is a near-threshold switch, so the casualty
expectation is a step function of the uncertain inputs. Gradient-based
reliability methods (FORM) are not reliable for it; :func:`reentry_casualty_model`
therefore also reports the continuous per-fragment ``demise_margin_<name>``
and a ``smooth_casualty_expectation`` for diagnostics. See the module tests
for the measured FORM, subset-simulation and Monte Carlo comparison.

Experimental (added in 0.28).

References:
    [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971], [REF-CURTIS-2014],
    [REF-US-STANDARD-ATMOSPHERE-1976], [EQ-AERO-CASUALTY-001].

"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from probstat.aerospace.entry import (
    SUTTON_GRAVES_EARTH_K,
    ExponentialAtmosphere,
    _density_function,
    _entry_angle,
    _locate_terminal,
    _make_entry_rhs,
)
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.numerics import integrate_fixed_step
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "STEFAN_BOLTZMANN",
    "CasualtyAssessment",
    "Fragment",
    "FragmentFate",
    "Material",
    "assess_reentry_casualty",
    "casualty_area",
    "cylinder_fragment",
    "demonstration_fragments",
    "demonstration_materials",
    "plate_fragment",
    "reentry_casualty_model",
    "simulate_fragment",
    "sphere_fragment",
]

STEFAN_BOLTZMANN = 5.670374419e-8
"""Stefan-Boltzmann constant ``sigma`` in W / (m^2 K^4) (CODATA 2018 value)."""

_H_REF = 7200.0  # m, reference scale height used only to size the time step
_CHUNK = 8  # integration steps between event checks
_RELAX = 2.0  # largest rate*step of the drag relaxation (RK4 stable below 2.78)
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_S_TH, _S_H, _S_V, _S_G, _S_Q, _S_E = range(6)

FragmentStatus = Literal["impact", "skip_out", "timeout", "numerical_failure"]
FragmentShape = Literal["sphere", "cylinder", "plate"]


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


def _name(value: object, label: str) -> str:
    """Return a name made of letters, digits and underscores.

    Args:
        value: Candidate name.
        label: Name used in error messages.

    Returns:
        The name.

    Raises:
        TypeError: If ``value`` is not text.
        ValueError: If it is empty or contains other characters.

    References:
        [EQ-AERO-CASUALTY-007]. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    if _NAME_PATTERN.match(value) is None:
        raise ValueError(f"{label} must be non-empty letters, digits and underscores")
    return value


@dataclass(frozen=True, slots=True)
class Material:
    """Thermal properties of a fragment material (user inputs).

    Args:
        name: Identifier (letters, digits, underscores).
        specific_heat: ``c`` in J/(kg K), ``> 0`` (constant).
        melt_temperature: ``T_m`` in K, ``> 0``.
        heat_of_fusion: ``L_f`` in J/kg, ``> 0``.
        emissivity: Total hemispherical emissivity ``epsilon`` in ``(0, 1]``.

    References:
        [EQ-AERO-CASUALTY-001], [EQ-AERO-CASUALTY-003].

    """

    name: str
    specific_heat: float
    melt_temperature: float
    heat_of_fusion: float
    emissivity: float

    def __post_init__(self) -> None:
        """Validate the properties.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range.

        References:
            [EQ-AERO-CASUALTY-003]. No equation is evaluated.

        """
        _name(self.name, "name")
        _positive(self.specific_heat, "specific_heat")
        _positive(self.melt_temperature, "melt_temperature")
        _positive(self.heat_of_fusion, "heat_of_fusion")
        eps = _positive(self.emissivity, "emissivity")
        if eps > 1.0:
            raise ValueError("emissivity must not exceed 1")


@dataclass(frozen=True, slots=True)
class Fragment:
    """A fragment released at break-up (build with the shape constructors).

    Args:
        name: Identifier (letters, digits, underscores).
        shape: ``"sphere"``, ``"cylinder"`` or ``"plate"`` (a label; the
            geometry enters through ``surface_area``).
        material: The :class:`Material`.
        mass: ``m`` in kg, ``> 0`` (the thermal mass; a hollow tank is its
            shell mass).
        surface_area: Total wetted surface ``S`` in m^2, ``> 0``.
        drag_coefficient: Mean hypersonic ``C_D > 0`` of the tumbling body
            (a user input).
        heating_shape_factor: Dimensionless multiplier ``f_s > 0`` of the
            Sutton-Graves flux acting on the mean projected area (a user
            input, ``1`` keeps the stagnation flux over ``S / 4``).
        count: Number of identical fragments, ``>= 1``.

    References:
        [EQ-AERO-CASUALTY-002], [EQ-AERO-CASUALTY-009].

    """

    name: str
    shape: FragmentShape
    material: Material
    mass: float
    surface_area: float
    drag_coefficient: float = 1.0
    heating_shape_factor: float = 1.0
    count: int = 1

    def __post_init__(self) -> None:
        """Validate the fragment.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range.

        References:
            [EQ-AERO-CASUALTY-009]. No equation is evaluated.

        """
        _name(self.name, "name")
        if self.shape not in ("sphere", "cylinder", "plate"):
            raise ValueError("shape must be 'sphere', 'cylinder' or 'plate'")
        if not isinstance(self.material, Material):
            raise TypeError("material must be a Material")
        _positive(self.mass, "mass")
        _positive(self.surface_area, "surface_area")
        _positive(self.drag_coefficient, "drag_coefficient")
        _positive(self.heating_shape_factor, "heating_shape_factor")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("count must be an integer")
        if self.count < 1:
            raise ValueError("count must be at least 1")

    @property
    def reference_area(self) -> float:
        """Mean projected area ``A = S / 4`` in m^2 (convex, tumbling).

        Returns:
            The area.

        References:
            [EQ-AERO-CASUALTY-009].

        """
        return 0.25 * self.surface_area

    @property
    def ballistic_coefficient(self) -> float:
        """Ballistic coefficient ``beta = m / (C_D A)`` in kg/m^2.

        Returns:
            The coefficient.

        References:
            [EQ-AERO-CASUALTY-009], [REF-ALLEN-EGGERS-1958].

        """
        return self.mass / (self.drag_coefficient * self.reference_area)

    @property
    def nose_radius(self) -> float:
        """Equivalent-sphere radius ``sqrt(S / (4 pi))`` in m for Sutton-Graves.

        Returns:
            The radius.

        References:
            [EQ-AERO-CASUALTY-009], [REF-SUTTON-GRAVES-1971].

        """
        return math.sqrt(self.surface_area / (4.0 * math.pi))


def sphere_fragment(
    name: str,
    material: Material,
    mass: float,
    diameter: float,
    *,
    drag_coefficient: float = 1.0,
    heating_shape_factor: float = 1.0,
    count: int = 1,
) -> Fragment:
    """Build a spherical fragment (tank, ball) from its mass and diameter.

    Args:
        name: Identifier.
        material: The material.
        mass: Thermal mass in kg.
        diameter: Outer diameter ``d`` in m, ``> 0``; ``S = pi d^2``.
        drag_coefficient: Mean ``C_D``.
        heating_shape_factor: Shape factor of the heating.
        count: Multiplicity.

    Returns:
        The :class:`Fragment`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-CASUALTY-009].

    References:
        [REF-SUTTON-GRAVES-1971].

    """
    d = _positive(diameter, "diameter")
    return Fragment(
        name,
        "sphere",
        material,
        mass,
        math.pi * d * d,
        drag_coefficient,
        heating_shape_factor,
        count,
    )


def cylinder_fragment(
    name: str,
    material: Material,
    mass: float,
    diameter: float,
    length: float,
    *,
    drag_coefficient: float = 1.0,
    heating_shape_factor: float = 1.0,
    count: int = 1,
) -> Fragment:
    """Build a closed cylindrical fragment (shaft, pressure bottle).

    Args:
        name: Identifier.
        material: The material.
        mass: Thermal mass in kg.
        diameter: Diameter ``d`` in m, ``> 0``.
        length: Length ``L`` in m, ``> 0``; ``S = pi d L + pi d^2 / 2``.
        drag_coefficient: Mean ``C_D``.
        heating_shape_factor: Shape factor of the heating.
        count: Multiplicity.

    Returns:
        The :class:`Fragment`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-CASUALTY-009].

    References:
        [REF-SUTTON-GRAVES-1971].

    """
    d = _positive(diameter, "diameter")
    length_m = _positive(length, "length")
    surface = math.pi * d * length_m + 0.5 * math.pi * d * d
    return Fragment(
        name,
        "cylinder",
        material,
        mass,
        surface,
        drag_coefficient,
        heating_shape_factor,
        count,
    )


def plate_fragment(
    name: str,
    material: Material,
    mass: float,
    length: float,
    width: float,
    thickness: float,
    *,
    drag_coefficient: float = 1.0,
    heating_shape_factor: float = 1.0,
    count: int = 1,
) -> Fragment:
    """Build a flat-plate fragment (panel, thin skin section).

    Args:
        name: Identifier.
        material: The material.
        mass: Thermal mass in kg.
        length: Length ``L`` in m, ``> 0``.
        width: Width ``W`` in m, ``> 0``.
        thickness: Thickness ``t`` in m, ``> 0``;
            ``S = 2 L W + 2 t (L + W)``.
        drag_coefficient: Mean ``C_D``.
        heating_shape_factor: Shape factor of the heating.
        count: Multiplicity.

    Returns:
        The :class:`Fragment`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-CASUALTY-009].

    References:
        [REF-SUTTON-GRAVES-1971].

    """
    length_m = _positive(length, "length")
    width_m = _positive(width, "width")
    t = _positive(thickness, "thickness")
    surface = 2.0 * length_m * width_m + 2.0 * t * (length_m + width_m)
    return Fragment(
        name,
        "plate",
        material,
        mass,
        surface,
        drag_coefficient,
        heating_shape_factor,
        count,
    )


def demonstration_materials() -> dict[str, Material]:
    """Return three SYNTHETIC materials of aluminium-, titanium- and steel-like order.

    The numbers are round values of the right order of magnitude for a
    light alloy, a titanium alloy and a steel. They were *not* verified
    against a property database in this repository and are not data of any
    real object; replace them with the values of the assessed design.

    Returns:
        Mapping from material name to :class:`Material`.

    Equation:
        [EQ-AERO-CASUALTY-003].

    References:
        [REF-SUTTON-GRAVES-1971].

    """
    return {
        "aluminium_like": Material("aluminium_like", 900.0, 930.0, 3.9e5, 0.3),
        "titanium_like": Material("titanium_like", 540.0, 1900.0, 3.9e5, 0.5),
        "steel_like": Material("steel_like", 500.0, 1700.0, 2.7e5, 0.6),
    }


def demonstration_fragments() -> tuple[Fragment, ...]:
    """Return a SYNTHETIC four-fragment set (tank, panel, shaft, box).

    A titanium-like spherical tank (30 kg, 0.7 m), an aluminium-like panel
    (6 kg, 1.2 m by 0.8 m), a steel-like shaft (12 kg, 0.12 m by 0.9 m) and
    two aluminium-like equipment boxes (4 kg each, 0.3 m cubes treated as a
    sphere of 0.37 m diameter). None is data of a real object.

    Returns:
        The fragments.

    Equation:
        [EQ-AERO-CASUALTY-009].

    References:
        [REF-SUTTON-GRAVES-1971].

    """
    mats = demonstration_materials()
    return (
        sphere_fragment("tank", mats["titanium_like"], 30.0, 0.7),
        plate_fragment("panel", mats["aluminium_like"], 6.0, 1.2, 0.8, 0.003),
        cylinder_fragment("shaft", mats["steel_like"], 12.0, 0.12, 0.9),
        sphere_fragment("box", mats["aluminium_like"], 4.0, 0.37, count=2),
    )


def casualty_area(fragment_area: float, person_area: float) -> float:
    """Return the debris casualty area ``(sqrt(A_p) + sqrt(A_f))^2`` in m^2.

    The person's area ``A_p`` is a user input (an upright person's mean
    projected area; the form is the usual sum of the two characteristic
    lengths squared and is a modelling convention here, not a verified
    standard). For ``A_f = 0`` it reduces to ``A_p``.

    Args:
        fragment_area: Mean projected area ``A_f >= 0`` of the fragment in m^2.
        person_area: Person area ``A_p > 0`` in m^2.

    Returns:
        The casualty area in m^2.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-CASUALTY-005].

    References:
        [EQ-AERO-CASUALTY-005].

    """
    a_f = _non_negative(fragment_area, "fragment_area")
    a_p = _positive(person_area, "person_area")
    return (math.sqrt(a_p) + math.sqrt(a_f)) ** 2


def _temperature(
    energy: float, mass: float, material: Material, initial_temperature: float
) -> float:
    """Return the lumped temperature of a body that absorbed ``energy``.

    Args:
        energy: Absorbed energy ``E`` in J (may be negative only by
            round-off).
        mass: Mass in kg.
        material: The material.
        initial_temperature: ``T_0`` in K.

    Returns:
        ``T_0 + E / (m c)`` below the melt energy, ``T_m`` in the fusion
        plateau and ``T_m + (E - E_dem) / (m c)`` beyond demise.

    References:
        [EQ-AERO-CASUALTY-003].

    """
    heat_capacity = mass * material.specific_heat
    sensible = heat_capacity * (material.melt_temperature - initial_temperature)
    if energy <= sensible:
        return initial_temperature + energy / heat_capacity
    fusion = mass * material.heat_of_fusion
    if energy <= sensible + fusion:
        return material.melt_temperature
    return material.melt_temperature + (energy - sensible - fusion) / heat_capacity


@dataclass(frozen=True, slots=True)
class FragmentFate:
    """Outcome of :func:`simulate_fragment` for one fragment.

    Args:
        name: Fragment name.
        status: ``"impact"`` (reached the ground), ``"skip_out"`` (climbed
            more than 10 km above the break-up altitude), ``"timeout"`` or
            ``"numerical_failure"``; only ``"impact"`` has meaningful fields
            below.
        stop_reason: Explanation.
        demised: ``True`` when the absorbed energy reached ``E_dem``.
        demise_altitude: Linearly interpolated altitude of the first
            crossing in m (``nan`` when it did not demise).
        demise_energy: ``E_dem = m [c (T_m - T_0) + L_f]`` in J.
        peak_energy: ``max E`` on the grid in J.
        demise_margin: ``1 - peak_energy / demise_energy`` (``> 0`` survives,
            continuous in all inputs).
        peak_temperature: ``T`` at the peak energy in K (a bookkeeping
            continuation beyond ``T_m`` when demised).
        impact_speed: Ground speed in m/s.
        impact_energy: ``m V_i^2 / 2`` in J (reported also for demised
            fragments, where it is meaningless).
        flight_time: Time to the ground in s.
        steps: Number of integration steps.
        survived: ``status == "impact"`` and not demised.

    References:
        [EQ-AERO-CASUALTY-001], [EQ-AERO-CASUALTY-004].

    """

    name: str
    status: FragmentStatus
    stop_reason: str
    demised: bool
    demise_altitude: float
    demise_energy: float
    peak_energy: float
    demise_margin: float
    peak_temperature: float
    impact_speed: float
    impact_energy: float
    flight_time: float
    steps: int
    survived: bool


def _fragment_rhs(
    fragment: Fragment,
    density: Callable[[float], float],
    *,
    beta: float,
    heating_scale: float,
    initial_temperature: float,
) -> Callable[[float, Sequence[float]], Sequence[float]]:
    """Build the right-hand side of the entry equations plus the heat balance.

    Args:
        fragment: The fragment.
        density: Density function in kg/m^3.
        beta: Ballistic coefficient in kg/m^2 (already scaled).
        heating_scale: Multiplier of the Sutton-Graves constant.
        initial_temperature: ``T_0`` in K.

    Returns:
        ``f(t, y)`` for the state ``(theta, h, V, gamma, Q, E)``.

    References:
        [EQ-AERO-CASUALTY-001], [EQ-AERO-CASUALTY-002].

    """
    base = _make_entry_rhs(
        density,
        beta=beta,
        nose_radius=fragment.nose_radius,
        lift_to_drag=0.0,
        density_scale=1.0,
        heating_constant=SUTTON_GRAVES_EARTH_K * heating_scale,
        idealised=False,
    )
    area_heat = fragment.heating_shape_factor * fragment.reference_area
    radiator = fragment.material.emissivity * STEFAN_BOLTZMANN * fragment.surface_area
    mass = fragment.mass
    material = fragment.material

    def rhs(t: float, y: Sequence[float]) -> Sequence[float]:
        """Evaluate the state derivative.

        Args:
            t: Time.
            y: State ``(theta, h, V, gamma, Q, E)``.

        Returns:
            The derivative.

        References:
            [EQ-AERO-CASUALTY-001], [EQ-AERO-CASUALTY-002].

        """
        flight = base(t, y)
        temp = _temperature(y[_S_E], mass, material, initial_temperature)
        power = area_heat * flight[4] - radiator * temp**4
        return (*flight, power)

    return rhs


def simulate_fragment(
    fragment: Fragment,
    *,
    breakup_altitude: float,
    flight_path_angle: float,
    breakup_speed: float = 7500.0,
    heating_scale: float = 1.0,
    ballistic_scale: float = 1.0,
    initial_temperature: float = 250.0,
    atmosphere: ExponentialAtmosphere | Literal["us76"] = "us76",
    max_step: float = 2.0,
    resolution: float = 0.2,
    max_time: float = 6000.0,
    record: bool = False,
) -> FragmentFate | tuple[FragmentFate, tuple[tuple[float, ...], ...]]:
    """Fly one fragment to the ground and decide its demise.

    The state ``(theta, h, V, gamma, Q, E)`` is advanced by classical RK4
    (:func:`probstat.numerics.integrate_fixed_step`) with the entry equations
    of :mod:`probstat.aerospace.entry` and the absorbed-energy equation
    ``dE/dt = f_s A q_SG - epsilon sigma S T(E)^4`` ([EQ-AERO-CASUALTY-001],
    [EQ-AERO-CASUALTY-002]). The step is recomputed every 8 steps as the
    smallest of ``max_step``, ``2.0 / lambda`` (``lambda = rho V / beta`` the
    drag relaxation rate, keeping RK4 stable) and ``resolution`` scale heights
    of altitude change. The ground is located by bisection of one sub-step
    (``|h| < 1e-4`` m). Physical failures are reported in ``status``; invalid
    arguments raise.

    Args:
        fragment: The fragment.
        breakup_altitude: Break-up altitude in m, ``> 0``.
        flight_path_angle: Break-up flight-path angle in rad,
            ``-pi/2 <= gamma < 0``.
        breakup_speed: Speed at break-up in m/s (inertial, non-rotating
            Earth), ``> 0``.
        heating_scale: Multiplier of the Sutton-Graves constant, ``> 0``.
        ballistic_scale: Multiplier of the fragment's ballistic coefficient,
            ``> 0``.
        initial_temperature: ``T_0`` in K, ``> 0`` and below the melt
            temperature.
        atmosphere: ``"us76"`` or an :class:`ExponentialAtmosphere`.
        max_step: Largest step in s.
        resolution: Altitude change per step in scale heights of 7200 m.
        max_time: Time limit in s.
        record: Also return the grid ``(t, h, V, gamma, Q, E)`` points.

    Returns:
        The :class:`FragmentFate`; with ``record=True`` a pair of it and the
        grid.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a number is out of range.

    Equation:
        [EQ-AERO-CASUALTY-001], [EQ-AERO-CASUALTY-002], [EQ-AERO-CASUALTY-003],
        [EQ-AERO-CASUALTY-004].

    References:
        [REF-ALLEN-EGGERS-1958], [REF-SUTTON-GRAVES-1971], [REF-CURTIS-2014],
        [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    if not isinstance(fragment, Fragment):
        raise TypeError("fragment must be a Fragment")
    h0 = _positive(breakup_altitude, "breakup_altitude")
    v0 = _positive(breakup_speed, "breakup_speed")
    gamma0 = _entry_angle(flight_path_angle)
    k_heat = _positive(heating_scale, "heating_scale")
    k_beta = _positive(ballistic_scale, "ballistic_scale")
    t0_k = _positive(initial_temperature, "initial_temperature")
    if t0_k >= fragment.material.melt_temperature:
        raise ValueError("initial_temperature must be below the melt temperature")
    step_cap = _positive(max_step, "max_step")
    res = _positive(resolution, "resolution")
    t_max = _positive(max_time, "max_time")
    if not isinstance(record, bool):
        raise TypeError("record must be bool")
    material = fragment.material
    mass = fragment.mass
    e_dem = mass * (
        material.specific_heat * (material.melt_temperature - t0_k)
        + material.heat_of_fusion
    )
    beta = fragment.ballistic_coefficient * k_beta
    density = _density_function(atmosphere)
    rhs = _fragment_rhs(
        fragment,
        density,
        beta=beta,
        heating_scale=k_heat,
        initial_temperature=t0_k,
    )

    def choose_step(state: Sequence[float]) -> float:
        """Return the step for the next chunk from the current state.

        Args:
            state: The state at the start of the chunk.

        Returns:
            The step in s: ``max_step / 2^k`` with the smallest ``k`` that
            respects the stability and resolution limits, so that the step is
            piecewise constant in the inputs and finite differences of the
            output are not polluted by step-size noise.

        References:
            [EQ-AERO-CASUALTY-001].

        """
        v = state[_S_V]
        rate = density(state[_S_H]) * v / beta
        needed = step_cap
        if rate > 0.0:
            needed = min(needed, _RELAX / rate)
        vertical = max(v * abs(math.sin(state[_S_G])), 1.0)
        needed = max(min(needed, res * _H_REF / vertical), 1e-3)
        halvings = max(0, math.ceil(math.log2(step_cap / needed)))
        return step_cap / 2.0**halvings

    state: tuple[float, ...] = (0.0, h0, v0, gamma0, 0.0, 0.0)
    time = 0.0
    nodes: list[tuple[float, tuple[float, ...]]] = [(0.0, state)]
    status: FragmentStatus = "impact"
    reason = "ground reached"
    running = True
    steps = 0
    while running:
        dt = choose_step(state)
        try:
            chunk = integrate_fixed_step(rhs, time, state, step=dt, steps=_CHUNK)
        except ValueError as error:
            status, reason, running = "numerical_failure", str(error), False
            continue
        for index in range(1, _CHUNK + 1):
            current = tuple(chunk.states[index])
            if current[_S_H] <= 0.0:
                t_hit, hit = _locate_terminal(
                    rhs, chunk.times[index - 1], chunk.states[index - 1], dt, 0.0
                )
                nodes.append((t_hit, tuple(hit)))
                state, time = tuple(hit), t_hit
                steps += index
                running = False
                break
            nodes.append((chunk.times[index], current))
            if current[_S_H] > h0 + 10_000.0:
                status = "skip_out"
                reason = "fragment climbed 10 km above the break-up altitude"
                state, time = current, chunk.times[index]
                steps += index
                running = False
                break
            if chunk.times[index] >= t_max:
                status = "timeout"
                reason = "max_time exceeded before the ground"
                state, time = current, chunk.times[index]
                steps += index
                running = False
                break
        else:
            state = tuple(chunk.states[_CHUNK])
            time = chunk.times[_CHUNK]
            steps += _CHUNK
    energies = [node[1][_S_E] for node in nodes]
    peak_index = max(range(len(energies)), key=energies.__getitem__)
    peak_energy = energies[peak_index]
    demised = peak_energy >= e_dem
    demise_altitude = math.nan
    if demised:
        first = next(i for i, e in enumerate(energies) if e >= e_dem)
        if first == 0:
            demise_altitude = nodes[0][1][_S_H]
        else:
            e_lo, e_hi = energies[first - 1], energies[first]
            frac = (e_dem - e_lo) / (e_hi - e_lo)
            h_lo, h_hi = nodes[first - 1][1][_S_H], nodes[first][1][_S_H]
            demise_altitude = h_lo + frac * (h_hi - h_lo)
    speed = state[_S_V]
    fate = FragmentFate(
        name=fragment.name,
        status=status,
        stop_reason=reason,
        demised=demised,
        demise_altitude=demise_altitude,
        demise_energy=e_dem,
        peak_energy=peak_energy,
        demise_margin=1.0 - peak_energy / e_dem,
        peak_temperature=_temperature(peak_energy, mass, material, t0_k),
        impact_speed=speed,
        impact_energy=0.5 * mass * speed * speed,
        flight_time=time,
        steps=steps,
        survived=status == "impact" and not demised,
    )
    if record:
        grid = tuple((t, *s) for t, s in nodes)
        return fate, grid
    return fate


@dataclass(frozen=True, slots=True)
class CasualtyAssessment:
    """Casualty result of a set of fragments.

    Args:
        fates: One :class:`FragmentFate` per fragment, in input order.
        survivors: Number of fragments (with multiplicity) reaching the ground.
        casualty_survivors: Survivors with impact energy at or above the
            threshold, with multiplicity.
        casualty_area_by_fragment: Name to its summed casualty area in m^2
            (``count`` times the single-fragment area; ``0`` when it demised
            or is below the energy threshold).
        total_casualty_area: ``A_c`` in m^2.
        casualty_expectation: ``E_c = A_c rho_pop`` (``rho_pop`` in persons/m^2).
        population_density: The density used, persons per km^2.
        energy_threshold: The impact-energy threshold in J.

    References:
        [EQ-AERO-CASUALTY-005], [EQ-AERO-CASUALTY-006].

    """

    fates: tuple[FragmentFate, ...]
    survivors: int
    casualty_survivors: int
    casualty_area_by_fragment: Mapping[str, float]
    total_casualty_area: float
    casualty_expectation: float
    population_density: float
    energy_threshold: float

    def contributions(self) -> dict[str, float]:
        """Return each fragment's fraction of the total casualty area.

        Returns:
            Name to fraction; all zero when no fragment contributes.

        References:
            [EQ-AERO-CASUALTY-005].

        """
        total = self.total_casualty_area
        return {
            name: (area / total if total > 0.0 else 0.0)
            for name, area in self.casualty_area_by_fragment.items()
        }

    def dominant_fragment(self) -> str | None:
        """Return the fragment with the largest casualty area.

        Returns:
            The name, or ``None`` when the total casualty area is zero.

        References:
            [EQ-AERO-CASUALTY-005].

        """
        if self.total_casualty_area <= 0.0:
            return None
        return max(
            self.casualty_area_by_fragment,
            key=self.casualty_area_by_fragment.__getitem__,
        )


def _check_fragments(fragments: object) -> tuple[Fragment, ...]:
    """Validate a non-empty sequence of fragments with unique names.

    Args:
        fragments: Candidate sequence.

    Returns:
        The fragments as a tuple.

    Raises:
        TypeError: If an element is not a :class:`Fragment`.
        ValueError: If the sequence is empty or names repeat.

    References:
        [EQ-AERO-CASUALTY-007]. No equation is evaluated.

    """
    if not isinstance(fragments, Sequence) or isinstance(fragments, str):
        raise TypeError("fragments must be a sequence of Fragment")
    items = tuple(fragments)
    if not items:
        raise ValueError("fragments must not be empty")
    if not all(isinstance(f, Fragment) for f in items):
        raise TypeError("fragments must be a sequence of Fragment")
    names = [f.name for f in items]
    if len(set(names)) != len(names):
        raise ValueError("fragment names must be unique")
    return items


def _aggregate(
    fates: Sequence[FragmentFate],
    fragments: Sequence[Fragment],
    person_area: float,
    population_density: float,
    energy_threshold: float,
) -> CasualtyAssessment:
    """Combine fragment fates into the casualty area and expectation.

    Args:
        fates: Fates, one per fragment.
        fragments: The fragments (same order).
        person_area: ``A_p`` in m^2.
        population_density: Persons per km^2.
        energy_threshold: Minimum impact energy in J.

    Returns:
        The :class:`CasualtyAssessment`.

    References:
        [EQ-AERO-CASUALTY-005], [EQ-AERO-CASUALTY-006].

    """
    areas: dict[str, float] = {}
    survivors = 0
    casualty_survivors = 0
    for fate, frag in zip(fates, fragments, strict=True):
        area = 0.0
        if fate.survived:
            survivors += frag.count
            if fate.impact_energy >= energy_threshold:
                casualty_survivors += frag.count
                area = frag.count * casualty_area(frag.reference_area, person_area)
        areas[frag.name] = area
    total = sum(areas.values())
    return CasualtyAssessment(
        tuple(fates),
        survivors,
        casualty_survivors,
        areas,
        total,
        total * population_density * 1e-6,
        population_density,
        energy_threshold,
    )


def assess_reentry_casualty(
    fragments: Sequence[Fragment],
    *,
    breakup_altitude: float,
    flight_path_angle: float,
    person_area: float,
    population_density: float,
    energy_threshold: float,
    breakup_speed: float = 7500.0,
    heating_scale: float = 1.0,
    initial_temperature: float = 250.0,
    atmosphere: ExponentialAtmosphere | Literal["us76"] = "us76",
    max_step: float = 2.0,
    resolution: float = 0.2,
    max_time: float = 6000.0,
) -> CasualtyAssessment:
    """Fly every fragment and assemble casualty area and expectation.

    ``A_c = sum_i n_i (sqrt(A_p) + sqrt(A_i))^2`` over the fragments that
    reach the ground undemised with ``m_i V_i^2 / 2 >= E_min``
    ([EQ-AERO-CASUALTY-005]) and ``E_c = A_c rho_pop`` with ``rho_pop`` the
    population density in persons per m^2 ([EQ-AERO-CASUALTY-006]).

    Args:
        fragments: Fragments with unique names.
        breakup_altitude: Break-up altitude in m.
        flight_path_angle: Break-up flight-path angle in rad (negative).
        person_area: Person area ``A_p`` in m^2 (a user input).
        population_density: Mean density in persons per km^2 (a user input).
        energy_threshold: Minimum casualty-causing impact energy ``E_min`` in
            J (a user input, to be checked against the governing standard).
        breakup_speed: Break-up speed in m/s.
        heating_scale: Multiplier of the Sutton-Graves constant.
        initial_temperature: ``T_0`` in K.
        atmosphere: ``"us76"`` or an :class:`ExponentialAtmosphere`.
        max_step: Largest step in s.
        resolution: Step resolution in scale heights.
        max_time: Time limit of each fragment's flight in s.

    Returns:
        The :class:`CasualtyAssessment`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a number is out of range.
        RuntimeError: If a fragment does not reach the ground (skip-out,
            timeout or numerical failure).

    Equation:
        [EQ-AERO-CASUALTY-005], [EQ-AERO-CASUALTY-006].

    References:
        [REF-SUTTON-GRAVES-1971], [REF-ALLEN-EGGERS-1958].

    """
    items = _check_fragments(fragments)
    a_p = _positive(person_area, "person_area")
    rho_pop = _non_negative(population_density, "population_density")
    e_min = _non_negative(energy_threshold, "energy_threshold")
    fates: list[FragmentFate] = []
    for frag in items:
        outcome = simulate_fragment(
            frag,
            breakup_altitude=breakup_altitude,
            flight_path_angle=flight_path_angle,
            breakup_speed=breakup_speed,
            heating_scale=heating_scale,
            initial_temperature=initial_temperature,
            atmosphere=atmosphere,
            max_step=max_step,
            resolution=resolution,
            max_time=max_time,
        )
        if isinstance(outcome, tuple):  # pragma: no cover - record is False
            raise TypeError("unexpected recorded result")
        if outcome.status != "impact":
            raise RuntimeError(
                f"fragment {frag.name!r} failed: {outcome.status}: "
                f"{outcome.stop_reason}"
            )
        fates.append(outcome)
    return _aggregate(fates, items, a_p, rho_pop, e_min)


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CasualtyForward:
    """Picklable forward map of :func:`reentry_casualty_model`.

    Args:
        fragments: The nominal fragments.
        person_area: ``A_p`` in m^2.
        energy_threshold: ``E_min`` in J.
        casualty_threshold: The ``E_c`` limit used for the margin.
        smoothing_width: Width of the logistic smoothing of the demise margin.
        initial_temperature: ``T_0`` in K.
        atmosphere: Atmosphere choice.
        max_step: Step bound in s.
        resolution: Step resolution in scale heights.

    References:
        [EQ-AERO-CASUALTY-007].

    """

    fragments: tuple[Fragment, ...]
    person_area: float
    energy_threshold: float
    casualty_threshold: float
    smoothing_width: float
    initial_temperature: float
    atmosphere: ExponentialAtmosphere | Literal["us76"]
    max_step: float
    resolution: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Run every fragment for named parameter values.

        Args:
            parameters: Values of the model inputs (see
                :func:`reentry_casualty_model`).

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If a scaled material property leaves its domain.
            RuntimeError: If a fragment does not reach the ground.

        Equation:
            [EQ-AERO-CASUALTY-007], [EQ-AERO-CASUALTY-008].

        References:
            [REF-SUTTON-GRAVES-1971], [REF-ALLEN-EGGERS-1958].

        """
        fates: list[FragmentFate] = []
        modified: list[Fragment] = []
        for frag in self.fragments:
            mat = frag.material
            key = mat.name
            scaled = replace(
                mat,
                melt_temperature=mat.melt_temperature
                * parameters[f"melt_temperature_scale_{key}"],
                heat_of_fusion=mat.heat_of_fusion
                * parameters[f"fusion_heat_scale_{key}"],
                emissivity=mat.emissivity * parameters[f"emissivity_scale_{key}"],
            )
            frag_run = replace(frag, material=scaled)
            outcome = simulate_fragment(
                frag_run,
                breakup_altitude=parameters["breakup_altitude"],
                flight_path_angle=parameters["flight_path_angle"],
                breakup_speed=parameters["breakup_speed"],
                heating_scale=parameters["heating_scale"],
                ballistic_scale=parameters[f"ballistic_scale_{frag.name}"],
                initial_temperature=self.initial_temperature,
                atmosphere=self.atmosphere,
                max_step=self.max_step,
                resolution=self.resolution,
            )
            if isinstance(outcome, tuple):  # pragma: no cover - record is False
                raise TypeError("unexpected recorded result")
            if outcome.status != "impact":
                raise RuntimeError(
                    f"fragment {frag.name!r} failed: {outcome.status}: "
                    f"{outcome.stop_reason}"
                )
            fates.append(outcome)
            modified.append(frag_run)
        density = parameters["population_density"]
        result = _aggregate(
            fates, modified, self.person_area, density, self.energy_threshold
        )
        out: dict[str, float] = {
            "survivors": float(result.survivors),
            "casualty_survivors": float(result.casualty_survivors),
            "total_casualty_area": result.total_casualty_area,
            "casualty_expectation": result.casualty_expectation,
            "casualty_margin": self.casualty_threshold - result.casualty_expectation,
            "casualty_ratio": result.casualty_expectation / self.casualty_threshold,
        }
        smooth = 0.0
        for fate, frag in zip(fates, modified, strict=True):
            out[f"demise_margin_{frag.name}"] = fate.demise_margin
            out[f"impact_energy_{frag.name}"] = fate.impact_energy
            out[f"casualty_area_{frag.name}"] = result.casualty_area_by_fragment[
                frag.name
            ]
            if fate.impact_energy >= self.energy_threshold:
                z = fate.demise_margin / self.smoothing_width
                weight = 1.0 / (1.0 + math.exp(-max(min(z, 500.0), -500.0)))
                area = casualty_area(frag.reference_area, self.person_area)
                smooth += weight * frag.count * area
        out["smooth_casualty_expectation"] = smooth * density * 1e-6
        return out


def _random_or_fixed(
    given: Mapping[str, Marginal | float],
    units: Mapping[str, str],
) -> tuple[list[RandomInput], dict[str, float]]:
    """Split named inputs into random inputs and fixed values.

    Args:
        given: Name to a distribution or a number.
        units: Name to a unit text.

    Returns:
        The random inputs and the fixed values.

    Raises:
        TypeError: If an entry is neither a number nor a distribution.

    References:
        [EQ-AERO-CASUALTY-007]. No equation is evaluated.

    """
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, value in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = require_finite_real(value, name=key)
        else:
            random_inputs.append(RandomInput(key, value, unit=units[key]))
    return random_inputs, fixed


def _expand(
    spec: Marginal | float | Mapping[str, Marginal | float],
    prefix: str,
    names: Sequence[str],
    unit: str,
    given: dict[str, Marginal | float],
    units: dict[str, str],
) -> None:
    """Expand a per-material or per-fragment uncertainty specification.

    A single number is fixed for every name, a single distribution becomes one
    independent random input per name, and a mapping gives each name its own
    entry (missing names are fixed at 1).

    Args:
        spec: Number, distribution or mapping from name to either.
        prefix: Parameter-name prefix.
        names: The material or fragment names.
        unit: Unit text.
        given: Output mapping of inputs, updated in place.
        units: Output mapping of units, updated in place.

    Raises:
        TypeError: If ``spec`` has an unsupported type.
        ValueError: If a mapping names an unknown entry.

    References:
        [EQ-AERO-CASUALTY-007]. No equation is evaluated.

    """
    if isinstance(spec, Mapping):
        unknown = set(spec) - set(names)
        if unknown:
            raise ValueError(f"unknown names in {prefix}: {sorted(unknown)}")
        for name in names:
            given[f"{prefix}_{name}"] = spec.get(name, 1.0)
            units[f"{prefix}_{name}"] = unit
        return
    for name in names:
        given[f"{prefix}_{name}"] = spec
        units[f"{prefix}_{name}"] = unit


def reentry_casualty_model(
    fragments: Sequence[Fragment],
    *,
    breakup_altitude: Marginal | float,
    flight_path_angle: Marginal | float,
    person_area: float,
    population_density: Marginal | float,
    energy_threshold: float,
    casualty_threshold: float,
    breakup_speed: Marginal | float = 7500.0,
    heating_scale: Marginal | float = 1.0,
    melt_temperature_scale: Marginal | float | Mapping[str, Marginal | float] = 1.0,
    fusion_heat_scale: Marginal | float | Mapping[str, Marginal | float] = 1.0,
    emissivity_scale: Marginal | float | Mapping[str, Marginal | float] = 1.0,
    ballistic_scale: Marginal | float | Mapping[str, Marginal | float] = 1.0,
    smoothing_width: float = 0.05,
    initial_temperature: float = 250.0,
    atmosphere: ExponentialAtmosphere | Literal["us76"] = "us76",
    max_step: float = 2.0,
    resolution: float = 0.2,
    name: str = "reentry_casualty",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic reentry-casualty model (all numbers synthetic).

    Each uncertain argument is a distribution (wrapped as an independent
    :class:`~probstat.modeling.RandomInput`) or a number (fixed). The scale
    arguments ``melt_temperature_scale``, ``fusion_heat_scale`` and
    ``emissivity_scale`` multiply the properties of each *material* (input
    names ``..._scale_<material>``), ``ballistic_scale`` multiplies the
    ballistic coefficient of each *fragment* (``ballistic_scale_<fragment>``).
    A single distribution yields one independent input per material or
    fragment; a mapping gives each its own entry (omitted names are fixed at
    1); a number is fixed. A scaled emissivity above 1, or any other
    out-of-domain sample, is a model failure handled by ``on_failure``.

    Quantities of interest: ``survivors``, ``casualty_survivors``,
    ``total_casualty_area`` (m^2), ``casualty_expectation`` (``E_c``),
    ``casualty_margin`` (``E_c`` limit minus ``E_c``; failure when
    ``<= 0``), ``casualty_ratio`` (``E_c`` over its limit),
    ``smooth_casualty_expectation`` (a continuous surrogate that replaces the
    survival switch of each fragment by ``1 / (1 + exp(-margin / w))``,
    ``w = smoothing_width``, [EQ-AERO-CASUALTY-008]) and, per fragment,
    ``demise_margin_<f>``, ``impact_energy_<f>`` and ``casualty_area_<f>``.
    Survival is a near-threshold switch, so ``casualty_expectation`` is a step
    function of the inputs and FORM is not trustworthy for it.

    Args:
        fragments: Fragments with unique names.
        breakup_altitude: Break-up altitude in m.
        flight_path_angle: Break-up flight-path angle in rad (negative).
        person_area: Person area ``A_p`` in m^2 (a user input).
        population_density: Mean density under the ground track in persons
            per km^2 (a user input).
        energy_threshold: Minimum casualty-causing impact energy in J (a
            user input, to be checked against the governing standard).
        casualty_threshold: Admissible ``E_c`` (commonly quoted as ``1e-4``;
            a user input, to be checked against the governing standard).
        breakup_speed: Break-up speed in m/s.
        heating_scale: Multiplier of the Sutton-Graves constant.
        melt_temperature_scale: Multiplier(s) of the melt temperature.
        fusion_heat_scale: Multiplier(s) of the heat of fusion.
        emissivity_scale: Multiplier(s) of the emissivity.
        ballistic_scale: Multiplier(s) of the fragment ballistic coefficients.
        smoothing_width: Width ``w > 0`` of the smooth demise weight.
        initial_temperature: ``T_0`` in K.
        atmosphere: ``"us76"`` or an :class:`ExponentialAtmosphere`.
        max_step: Largest step in s.
        resolution: Step resolution in scale heights.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is out of range or no input is random.

    Equation:
        [EQ-AERO-CASUALTY-007], [EQ-AERO-CASUALTY-008].

    References:
        [REF-SUTTON-GRAVES-1971], [REF-ALLEN-EGGERS-1958].

    """
    items = _check_fragments(fragments)
    material_names = sorted({f.material.name for f in items})
    fragment_names = [f.name for f in items]
    a_p = _positive(person_area, "person_area")
    e_min = _non_negative(energy_threshold, "energy_threshold")
    limit = _positive(casualty_threshold, "casualty_threshold")
    width = _positive(smoothing_width, "smoothing_width")
    t0 = _positive(initial_temperature, "initial_temperature")
    given: dict[str, Marginal | float] = {
        "breakup_altitude": breakup_altitude,
        "flight_path_angle": flight_path_angle,
        "breakup_speed": breakup_speed,
        "heating_scale": heating_scale,
        "population_density": population_density,
    }
    units: dict[str, str] = {
        "breakup_altitude": "m",
        "flight_path_angle": "rad",
        "breakup_speed": "m/s",
        "heating_scale": "-",
        "population_density": "persons/km^2",
    }
    _expand(
        melt_temperature_scale,
        "melt_temperature_scale",
        material_names,
        "-",
        given,
        units,
    )
    _expand(fusion_heat_scale, "fusion_heat_scale", material_names, "-", given, units)
    _expand(emissivity_scale, "emissivity_scale", material_names, "-", given, units)
    _expand(ballistic_scale, "ballistic_scale", fragment_names, "-", given, units)
    random_inputs, fixed = _random_or_fixed(given, units)
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    forward = _CasualtyForward(
        items,
        a_p,
        e_min,
        limit,
        width,
        t0,
        atmosphere,
        _positive(max_step, "max_step"),
        _positive(resolution, "resolution"),
    )
    table: list[QuantityOfInterest] = [
        QuantityOfInterest("survivors", "Fragments reaching the ground", "-"),
        QuantityOfInterest(
            "casualty_survivors", "Survivors above the energy threshold", "-"
        ),
        QuantityOfInterest("total_casualty_area", "Total casualty area", "m^2"),
        QuantityOfInterest("casualty_expectation", "Casualty expectation E_c", "-"),
        QuantityOfInterest(
            "casualty_margin", "Casualty-expectation limit minus E_c", "-"
        ),
        QuantityOfInterest("casualty_ratio", "E_c divided by its limit", "-"),
        QuantityOfInterest(
            "smooth_casualty_expectation",
            "E_c with a logistic survival weight (diagnostic)",
            "-",
        ),
    ]
    for frag in items:
        table.extend(
            (
                QuantityOfInterest(
                    f"demise_margin_{frag.name}",
                    f"1 - peak absorbed energy / demise energy of {frag.name}",
                    "-",
                ),
                QuantityOfInterest(
                    f"impact_energy_{frag.name}",
                    f"Impact energy of {frag.name}",
                    "J",
                ),
                QuantityOfInterest(
                    f"casualty_area_{frag.name}",
                    f"Casualty area contributed by {frag.name}",
                    "m^2",
                ),
            )
        )
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        forward,
        tuple(table),
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )
