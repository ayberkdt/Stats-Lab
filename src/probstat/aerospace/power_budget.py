"""Energy-balance reliability of a spacecraft power subsystem (synthetic).

A power subsystem fails at end of life (EOL) when the solar array can no
longer cover the sunlit load plus the recharge of the battery, or when the
battery is discharged deeper in every eclipse than its depth-of-discharge
limit allows. Both margins shrink with age through array degradation and
battery capacity fade, and every ingredient is uncertain. This module builds
the deterministic energy balance of one orbit as a function of mission time
and wraps it as a :class:`probstat.modeling.ProbabilisticModel`, so that the
probability of an EOL shortfall, its sensitivities and the array size needed
for a target reliability are obtained with the library's reliability and
Sobol tools.

Contents:

- :func:`eclipse_fraction`, :func:`critical_beta`, :func:`orbital_period`,
  :func:`eclipse_duration`: circular-orbit geometry with a cylindrical Earth
  shadow (closed forms, derived in :func:`eclipse_fraction`).
- :class:`PowerDesign`: the fixed design parameters (orbit, array losses,
  battery, loads, depth-of-discharge limit); :func:`demonstration_design` is
  *synthetic*.
- :func:`array_power`, :func:`battery_capacity`: array power and battery
  capacity at mission year ``t``.
- :func:`power_budget`: the energy balance of one orbit at year ``t``
  (:class:`PowerBudgetState`), with the power margin, the depth of discharge
  and the combined margin.
- :func:`power_lifetime`: the first year the combined margin turns negative
  (monotone bisection; :class:`PowerLifetime`).
- :func:`power_budget_model`: the probabilistic model with uncertain
  beginning-of-life power, degradation rate, load factor, battery fade rate,
  charge and discharge efficiencies and number of failed array strings.
- :func:`power_failure_probability` (FORM, subset simulation or Monte Carlo
  on the series-system limit state), :func:`component_form_bounds` (FORM of
  each margin and the series bounds), :func:`string_failure_pmf` and
  :func:`binomial_mixture_probability` (exact binomial string failures), and
  :func:`required_bol_power` (monotone root find of the beginning-of-life
  power that meets a target reliability).

Units. SI for geometry, power and time: m, s, rad, W; the energy of one orbit
and the battery capacity are in watt-hours (1 Wh = 3600 J); mission time and
rates are per Julian year (365.25 d). Angles are in radians.

Assumptions and validity.
    Circular orbit around a spherical Earth of radius ``R_E`` (equatorial
    radius of :mod:`probstat.aerospace.ascent`), a cylindrical umbra: no
    penumbra, no finite Sun disk, no atmospheric refraction, the Sun direction
    fixed over one revolution (the beta angle changes slowly). The array tracks
    the Sun with a fixed incidence angle and delivers a constant power over the
    sunlit arc; loads are constant in each phase; the battery is the only
    storage and its voltage dynamics, temperature and charge taper are not
    modelled: a discharge efficiency ``eta_d`` and a charge efficiency
    ``eta_c`` are lumped. The balance is per orbit and assumes the battery is
    fully recharged within the sunlit arc, which is what the margin tests.
    Degradation is a smooth deterministic law of time with an uncertain rate;
    it is not a degradation mechanism model. Linear laws are valid only while
    the remaining factor is positive; outside that range the functions raise
    ``ValueError`` (the lifetime search treats an exhausted array or battery
    as a negative margin).

Coefficients and data are not shipped. Every cell temperature coefficient,
degradation rate, fade rate, efficiency, load and capacity in
:func:`demonstration_design` and in the tests is synthetic and illustrative:
it is not a datasheet value, is not a published degradation coefficient and is
not a statement about any real spacecraft, array or cell. Real values must be
supplied by the user.

Experimental (added in 0.28).

References:
    [EQ-AERO-POWER-001], [REF-CURTIS-2014], [REF-MELCHERS-BECK-2018],
    [REF-SALTELLI-2010].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS
from probstat.aerospace.orbital_decay import SECONDS_PER_YEAR
from probstat.distributions import Binomial
from probstat.foundations.random_streams import RandomStream
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.reliability import (
    ModelLimitState,
    form_hlrf,
    model_monte_carlo,
    subset_simulation,
)
from probstat.uq import FailurePolicy, QuantityOfInterest, UncertaintyNature

__all__ = [
    "BinomialMixture",
    "ComponentBounds",
    "PowerBudgetState",
    "PowerDesign",
    "PowerLifetime",
    "PowerReliabilityEstimate",
    "RequiredPower",
    "array_power",
    "battery_capacity",
    "binomial_mixture_probability",
    "component_form_bounds",
    "critical_beta",
    "demonstration_design",
    "eclipse_duration",
    "eclipse_fraction",
    "orbital_period",
    "power_budget",
    "power_budget_model",
    "power_failure_probability",
    "power_lifetime",
    "required_bol_power",
    "string_failure_pmf",
]

DegradationLaw = Literal["compound", "linear"]
FadeLaw = Literal["linear", "sqrt"]


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _real(value: object, *, name: str) -> float:
    """Return a finite real number (``bool`` rejected).

    References:
        [EQ-AERO-POWER-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    return require_finite_real(value, name=name)


def _positive(value: object, *, name: str) -> float:
    """Return a finite strictly positive real number.

    References:
        [EQ-AERO-POWER-001]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: object, *, name: str) -> float:
    """Return a finite non-negative real number.

    References:
        [EQ-AERO-POWER-001]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _efficiency(value: object, *, name: str) -> float:
    """Return an efficiency in ``(0, 1]``.

    References:
        [EQ-AERO-POWER-005]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return number


def _integer(value: object, *, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum`` (``bool`` rejected).

    References:
        [EQ-AERO-POWER-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _open_probability(value: object, *, name: str) -> float:
    """Return a probability strictly inside ``(0, 1)``.

    References:
        [EQ-AERO-POWER-008]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return number


# --------------------------------------------------------------------------
# Orbit geometry
# --------------------------------------------------------------------------


def orbital_period(altitude: float) -> float:
    """Return the period of a circular orbit, ``T = 2 pi sqrt(r^3 / mu)``.

    Args:
        altitude: Altitude above the equatorial radius (m), positive.

    Returns:
        The period (s).

    Raises:
        TypeError: If ``altitude`` is not a real number.
        ValueError: If ``altitude`` is not positive and finite.

    Equation:
        [EQ-AERO-POWER-002].

    References:
        [EQ-AERO-POWER-002], [REF-CURTIS-2014].

    """
    h = _positive(altitude, name="altitude")
    r = EARTH_RADIUS + h
    return 2.0 * math.pi * math.sqrt(r**3 / EARTH_MU)


def critical_beta(altitude: float) -> float:
    """Return the beta angle above which the orbit has no eclipse.

    With the cylindrical shadow of radius ``R_E`` and orbit radius
    ``r = R_E + h``, the orbit is eclipse free exactly when
    ``|beta| >= beta* = arcsin(R_E / r)``.

    Args:
        altitude: Altitude (m), positive.

    Returns:
        ``beta*`` (rad), in ``(0, pi/2)``.

    Raises:
        TypeError: If ``altitude`` is not a real number.
        ValueError: If ``altitude`` is not positive and finite.

    Equation:
        [EQ-AERO-POWER-001].

    References:
        [EQ-AERO-POWER-001], [REF-CURTIS-2014].

    """
    h = _positive(altitude, name="altitude")
    return math.asin(EARTH_RADIUS / (EARTH_RADIUS + h))


def eclipse_fraction(altitude: float, beta: float) -> float:
    """Return the eclipse fraction of a circular orbit (cylindrical shadow).

    Derivation. Let ``s`` be the unit vector to the Sun and ``beta`` the
    angle between ``s`` and the orbit plane. Place the position
    ``r (cos th e1 + sin th e2)`` with ``e1`` the projection of ``s`` on the
    plane, so ``s . r = r cos(beta) cos(th)``. The spacecraft is in the shadow
    cylinder (radius ``R_E``, axis along ``-s`` through the Earth centre) when
    ``s . r < 0`` and its distance to the axis is below ``R_E``:
    ``r^2 - (s . r)^2 < R_E^2``, i.e. ``cos^2(th) cos^2(beta) >
    1 - (R_E / r)^2``. With ``rho = R_E / r`` the eclipse is the arc
    ``|th - pi| < phi`` with ``cos(phi) = sqrt(1 - rho^2) / cos(beta)``, so
    the eclipse fraction is ``f = phi / pi`` and it is zero when
    ``cos(beta) <= sqrt(1 - rho^2)``, i.e. ``|beta| >= arcsin(rho)``. At
    ``beta = 0``: ``f = arcsin(rho) / pi``.

    Args:
        altitude: Altitude (m), positive.
        beta: Beta angle (rad), ``|beta| <= pi/2``.

    Returns:
        The fraction of the period spent in eclipse, in ``[0, 1/2)``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``altitude <= 0`` or ``|beta| > pi/2``.

    Equation:
        [EQ-AERO-POWER-001].

    References:
        [EQ-AERO-POWER-001], [REF-CURTIS-2014].

    """
    h = _positive(altitude, name="altitude")
    b = _real(beta, name="beta")
    if abs(b) > 0.5 * math.pi:
        raise ValueError("beta must satisfy |beta| <= pi/2")
    rho = EARTH_RADIUS / (EARTH_RADIUS + h)
    cos_b = math.cos(b)
    root = math.sqrt(1.0 - rho * rho)
    if cos_b <= root:
        return 0.0
    return math.acos(root / cos_b) / math.pi


def eclipse_duration(altitude: float, beta: float) -> float:
    """Return the eclipse duration per orbit, ``f T`` (s).

    Args:
        altitude: Altitude (m), positive.
        beta: Beta angle (rad), ``|beta| <= pi/2``.

    Returns:
        The duration (s).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-POWER-001].

    References:
        [EQ-AERO-POWER-001], [EQ-AERO-POWER-002].

    """
    return eclipse_fraction(altitude, beta) * orbital_period(altitude)


# --------------------------------------------------------------------------
# Design, array and battery
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerDesign:
    """Fixed design parameters of the power subsystem (synthetic inputs).

    Attributes:
        altitude: Circular-orbit altitude (m).
        beta: Beta angle used for the eclipse (rad); ``0`` is the longest
            eclipse.
        incidence_angle: Sun incidence angle on the array normal (rad),
            ``|angle| < pi/2``; the cosine loss is ``cos(angle)``.
        temperature_coefficient: Relative power loss per kelvin above the
            reference temperature (1/K), non-negative.
        temperature_offset: Operating minus reference cell temperature (K).
        total_strings: Number of array strings ``N``, at least 1.
        bus_efficiency: Array-to-bus efficiency in ``(0, 1]``.
        battery_capacity_wh: Beginning-of-life battery capacity (Wh).
        degradation_law: ``"compound"`` (``(1-d)^t``) or ``"linear"``
            (``1 - d t``).
        fade_law: ``"linear"`` (``a n``) or ``"sqrt"`` (``a sqrt(n)``) in the
            cycle count ``n``.
        dod_max: Allowed depth of discharge in ``(0, 1]``.
        load_sunlit: Nominal load in sunlight (W), positive.
        load_eclipse: Nominal load in eclipse (W), non-negative.

    Raises:
        TypeError: If a value has an unsupported type.
        ValueError: If a value is out of range.

    References:
        [EQ-AERO-POWER-003], [EQ-AERO-POWER-004].

    """

    altitude: float
    beta: float
    incidence_angle: float
    temperature_coefficient: float
    temperature_offset: float
    total_strings: int
    bus_efficiency: float
    battery_capacity_wh: float
    degradation_law: DegradationLaw
    fade_law: FadeLaw
    dod_max: float
    load_sunlit: float
    load_eclipse: float

    def __post_init__(self) -> None:
        """Validate the design.

        Raises:
            TypeError: If a value has an unsupported type.
            ValueError: If a value is out of range.

        References:
            [EQ-AERO-POWER-003]. No equation is evaluated.

        """
        _positive(self.altitude, name="altitude")
        if abs(_real(self.beta, name="beta")) > 0.5 * math.pi:
            raise ValueError("beta must satisfy |beta| <= pi/2")
        if abs(_real(self.incidence_angle, name="incidence_angle")) >= 0.5 * math.pi:
            raise ValueError("incidence_angle must satisfy |angle| < pi/2")
        _non_negative(self.temperature_coefficient, name="temperature_coefficient")
        _real(self.temperature_offset, name="temperature_offset")
        if 1.0 - self.temperature_coefficient * self.temperature_offset <= 0.0:
            raise ValueError("the temperature factor must be positive")
        _integer(self.total_strings, name="total_strings", minimum=1)
        _efficiency(self.bus_efficiency, name="bus_efficiency")
        _positive(self.battery_capacity_wh, name="battery_capacity_wh")
        if self.degradation_law not in ("compound", "linear"):
            raise ValueError("degradation_law must be 'compound' or 'linear'")
        if self.fade_law not in ("linear", "sqrt"):
            raise ValueError("fade_law must be 'linear' or 'sqrt'")
        _efficiency(self.dod_max, name="dod_max")
        _positive(self.load_sunlit, name="load_sunlit")
        _non_negative(self.load_eclipse, name="load_eclipse")

    @property
    def period(self) -> float:
        """Orbital period (s).

        References:
            [EQ-AERO-POWER-002].

        """
        return orbital_period(self.altitude)

    @property
    def eclipse_seconds(self) -> float:
        """Eclipse duration per orbit (s).

        References:
            [EQ-AERO-POWER-001].

        """
        return eclipse_duration(self.altitude, self.beta)

    @property
    def orbits_per_year(self) -> float:
        """Number of orbits (battery cycles) per Julian year.

        References:
            [EQ-AERO-POWER-002].

        """
        return SECONDS_PER_YEAR / self.period


def demonstration_design() -> PowerDesign:
    """Return a *synthetic* design: 600 km, beta 0, invented parameters.

    Nothing here is a datasheet value or a real spacecraft: 600 km altitude,
    zero beta angle, 10 degree incidence, ``0.2 %/K`` over a 40 K offset, 20
    strings, 0.95 bus efficiency, a 500 Wh battery with linear fade, a depth
    limit of 0.4, a 200 W sunlit and a 150 W eclipse load.

    Returns:
        The design.

    Equation:
        [EQ-AERO-POWER-003].

    References:
        [EQ-AERO-POWER-003].

    """
    return PowerDesign(
        altitude=600.0e3,
        beta=0.0,
        incidence_angle=math.radians(10.0),
        temperature_coefficient=0.002,
        temperature_offset=40.0,
        total_strings=20,
        bus_efficiency=0.95,
        battery_capacity_wh=500.0,
        degradation_law="compound",
        fade_law="linear",
        dod_max=0.4,
        load_sunlit=200.0,
        load_eclipse=150.0,
    )


def _degradation_factor(law: str, rate: float, years: float) -> float:
    """Return the degradation factor ``D(t)`` (may be non-positive for linear).

    References:
        [EQ-AERO-POWER-003]. No equation is evaluated beyond ``D``.

    """
    if law == "compound":
        return float((1.0 - rate) ** years)
    return 1.0 - rate * years


def _fade_factor(law: str, rate: float, cycles: float) -> float:
    """Return the remaining capacity fraction ``1 - fade(n)`` (may be <= 0).

    References:
        [EQ-AERO-POWER-004]. No equation is evaluated beyond the fade.

    """
    if law == "linear":
        return 1.0 - rate * cycles
    return 1.0 - rate * math.sqrt(cycles)


def array_power(
    design: PowerDesign,
    bol_power: float,
    years: float,
    *,
    degradation_rate: float,
    failed_strings: float = 0.0,
) -> float:
    """Return the solar-array power delivered to the bus at year ``t``.

    ``P(t) = P_BOL eta_bus cos(theta) (1 - k_T dT) D(t) (1 - n_f / N)``, with
    ``D = (1 - d)^t`` (compound) or ``1 - d t`` (linear), the incidence angle
    ``theta``, the temperature coefficient ``k_T`` and offset ``dT`` and
    ``n_f`` failed strings of ``N`` (a real number, so a continuous
    surrogate is possible).

    Args:
        design: The :class:`PowerDesign`.
        bol_power: Beginning-of-life array power at normal incidence (W).
        years: Mission time (years), non-negative.
        degradation_rate: ``d`` in ``[0, 1)`` per year (per year of ``D``).
        failed_strings: Number of failed strings, in ``[0, N)``.

    Returns:
        The power (W).

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If an argument is out of range or the degradation factor
            is not positive.

    Equation:
        [EQ-AERO-POWER-003].

    References:
        [EQ-AERO-POWER-003].

    """
    if not isinstance(design, PowerDesign):
        raise TypeError("design must be a PowerDesign")
    p0 = _positive(bol_power, name="bol_power")
    t = _non_negative(years, name="years")
    d = _real(degradation_rate, name="degradation_rate")
    if not 0.0 <= d < 1.0:
        raise ValueError("degradation_rate must lie in [0, 1)")
    nf = _non_negative(failed_strings, name="failed_strings")
    if nf >= design.total_strings:
        raise ValueError("failed_strings must be below total_strings")
    degradation = _degradation_factor(design.degradation_law, d, t)
    if degradation <= 0.0:
        raise ValueError("the linear degradation factor is exhausted")
    temperature = 1.0 - design.temperature_coefficient * design.temperature_offset
    return (
        p0
        * design.bus_efficiency
        * math.cos(design.incidence_angle)
        * temperature
        * degradation
        * (1.0 - nf / design.total_strings)
    )


def battery_capacity(design: PowerDesign, years: float, *, fade_rate: float) -> float:
    """Return the battery capacity at year ``t`` after ``n`` cycles (Wh).

    ``C(t) = C_0 (1 - a n)`` (linear) or ``C_0 (1 - a sqrt(n))`` (square
    root), with ``n = t`` times the orbits per year (one cycle per orbit).

    Args:
        design: The :class:`PowerDesign`.
        years: Mission time (years), non-negative.
        fade_rate: ``a >= 0`` per cycle (linear) or per square root of cycle.

    Returns:
        The capacity (Wh).

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If an argument is out of range or the capacity is
            exhausted.

    Equation:
        [EQ-AERO-POWER-004].

    References:
        [EQ-AERO-POWER-004].

    """
    if not isinstance(design, PowerDesign):
        raise TypeError("design must be a PowerDesign")
    t = _non_negative(years, name="years")
    a = _non_negative(fade_rate, name="fade_rate")
    factor = _fade_factor(design.fade_law, a, t * design.orbits_per_year)
    if factor <= 0.0:
        raise ValueError("the battery capacity is exhausted")
    return design.battery_capacity_wh * factor


# --------------------------------------------------------------------------
# Energy balance
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerBudgetState:
    """Energy balance of one orbit at one mission time.

    Attributes:
        years: Mission time (years).
        array_power: Array power at the bus (W).
        capacity_wh: Battery capacity (Wh).
        eclipse_fraction: Eclipse fraction of the orbit.
        sunlit_seconds: Sunlit arc duration (s).
        eclipse_seconds: Eclipse duration (s).
        available_wh: Array energy over the sunlit arc (Wh).
        required_wh: Sunlit load energy plus the recharge energy (Wh).
        margin_wh: ``available - required`` (Wh).
        power_margin_w: ``margin / sunlit duration`` (W).
        power_margin_ratio: ``margin / required`` (dimensionless).
        dod: Depth of discharge per eclipse.
        dod_margin_ratio: ``(DoD_max - DoD) / DoD_max``.
        combined_margin: ``min(power_margin_ratio, dod_margin_ratio)``.

    References:
        [EQ-AERO-POWER-005], [EQ-AERO-POWER-006].

    """

    years: float
    array_power: float
    capacity_wh: float
    eclipse_fraction: float
    sunlit_seconds: float
    eclipse_seconds: float
    available_wh: float
    required_wh: float
    margin_wh: float
    power_margin_w: float
    power_margin_ratio: float
    dod: float
    dod_margin_ratio: float
    combined_margin: float


def power_budget(
    design: PowerDesign,
    years: float,
    *,
    bol_power: float,
    degradation_rate: float,
    fade_rate: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    load_factor: float = 1.0,
    failed_strings: float = 0.0,
) -> PowerBudgetState:
    """Return the energy balance of one orbit at mission year ``t``.

    With the eclipse energy ``E_e = L_e T_e`` (``L_e`` the eclipse load times
    ``load_factor``), the battery gives ``E_e / eta_d`` and the array must
    return it through the charger, so the required sunlit energy is
    ``E_req = L_s T_s + E_e / (eta_d eta_c)``; the array supplies
    ``E_av = P(t) T_s``. The margin is ``E_av - E_req`` and the depth of
    discharge is ``DoD = E_e / (C(t) eta_d)``. The two dimensionless margins
    ``m_P = (E_av - E_req) / E_req`` and ``m_D = (DoD_max - DoD) / DoD_max``
    are both zero at their limit, and the series-system margin is
    ``m = min(m_P, m_D)``.

    Args:
        design: The :class:`PowerDesign`.
        years: Mission time (years), non-negative.
        bol_power: Beginning-of-life array power (W).
        degradation_rate: Array degradation rate per year, in ``[0, 1)``.
        fade_rate: Battery fade rate (see :func:`battery_capacity`).
        charge_efficiency: ``eta_c`` in ``(0, 1]``.
        discharge_efficiency: ``eta_d`` in ``(0, 1]``.
        load_factor: Positive multiplier of both nominal loads.
        failed_strings: Failed strings, in ``[0, N)``.

    Returns:
        The :class:`PowerBudgetState`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If an argument is out of range or the array or battery
            is exhausted.

    Equation:
        [EQ-AERO-POWER-005], [EQ-AERO-POWER-006].

    References:
        [EQ-AERO-POWER-005], [EQ-AERO-POWER-006].

    """
    if not isinstance(design, PowerDesign):
        raise TypeError("design must be a PowerDesign")
    t = _non_negative(years, name="years")
    eta_c = _efficiency(charge_efficiency, name="charge_efficiency")
    eta_d = _efficiency(discharge_efficiency, name="discharge_efficiency")
    factor = _positive(load_factor, name="load_factor")
    power = array_power(
        design,
        bol_power,
        t,
        degradation_rate=degradation_rate,
        failed_strings=failed_strings,
    )
    capacity = battery_capacity(design, t, fade_rate=fade_rate)
    period = design.period
    ecl_s = design.eclipse_seconds
    sun_s = period - ecl_s
    ecl_wh = factor * design.load_eclipse * ecl_s / 3600.0
    required = factor * design.load_sunlit * sun_s / 3600.0 + ecl_wh / (eta_d * eta_c)
    available = power * sun_s / 3600.0
    margin = available - required
    dod = ecl_wh / (capacity * eta_d)
    power_ratio = margin / required
    dod_ratio = (design.dod_max - dod) / design.dod_max
    return PowerBudgetState(
        years=t,
        array_power=power,
        capacity_wh=capacity,
        eclipse_fraction=ecl_s / period,
        sunlit_seconds=sun_s,
        eclipse_seconds=ecl_s,
        available_wh=available,
        required_wh=required,
        margin_wh=margin,
        power_margin_w=margin * 3600.0 / sun_s,
        power_margin_ratio=power_ratio,
        dod=dod,
        dod_margin_ratio=dod_ratio,
        combined_margin=min(power_ratio, dod_ratio),
    )


@dataclass(frozen=True, slots=True)
class PowerLifetime:
    """First year the combined margin is negative.

    Attributes:
        years: The crossing time, or the horizon when ``survived``.
        survived: ``True`` if the margin is still non-negative at the horizon
            (the lifetime is censored at the horizon).

    References:
        [EQ-AERO-POWER-006].

    """

    years: float
    survived: bool


def power_lifetime(
    design: PowerDesign,
    horizon_years: float,
    *,
    bol_power: float,
    degradation_rate: float,
    fade_rate: float,
    charge_efficiency: float,
    discharge_efficiency: float,
    load_factor: float = 1.0,
    failed_strings: float = 0.0,
    tolerance: float = 1.0e-9,
) -> PowerLifetime:
    """Return the first year the combined margin turns negative.

    The combined margin is non-increasing in time (the array power and the
    battery capacity are non-increasing for non-negative rates), so the sign
    change is located by bisection to ``tolerance`` years. A margin below
    zero at ``t = 0`` gives ``0``. An exhausted array or battery counts as a
    negative margin (zero power gives ``m_P = -1`` exactly; zero capacity
    gives an unbounded depth of discharge).

    Args:
        design: The :class:`PowerDesign`.
        horizon_years: Search horizon (years), positive.
        bol_power: Beginning-of-life array power (W).
        degradation_rate: Array degradation rate per year.
        fade_rate: Battery fade rate.
        charge_efficiency: ``eta_c`` in ``(0, 1]``.
        discharge_efficiency: ``eta_d`` in ``(0, 1]``.
        load_factor: Positive multiplier of both loads.
        failed_strings: Failed strings.
        tolerance: Bracket width at termination (years).

    Returns:
        The :class:`PowerLifetime`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-POWER-006].

    References:
        [EQ-AERO-POWER-006].

    """
    horizon = _positive(horizon_years, name="horizon_years")
    tol = _positive(tolerance, name="tolerance")
    rate_d = _real(degradation_rate, name="degradation_rate")
    rate_f = _real(fade_rate, name="fade_rate")

    def margin(t: float) -> float:
        """Return the combined margin, or ``-1`` if the array or battery is gone.

        References:
            [EQ-AERO-POWER-006]. No equation is evaluated beyond the margin.

        """
        if _degradation_factor(design.degradation_law, rate_d, t) <= 0.0:
            return -1.0
        if _fade_factor(design.fade_law, rate_f, t * design.orbits_per_year) <= 0.0:
            return -1.0
        return power_budget(
            design,
            t,
            bol_power=bol_power,
            degradation_rate=rate_d,
            fade_rate=rate_f,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            load_factor=load_factor,
            failed_strings=failed_strings,
        ).combined_margin

    if margin(0.0) < 0.0:
        return PowerLifetime(0.0, False)
    if margin(horizon) >= 0.0:
        return PowerLifetime(horizon, True)
    lo, hi = 0.0, horizon
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if margin(mid) >= 0.0:
            lo = mid
        else:
            hi = mid
    return PowerLifetime(0.5 * (lo + hi), False)


# --------------------------------------------------------------------------
# Probabilistic model
# --------------------------------------------------------------------------

_MODEL_QUANTITIES = (
    QuantityOfInterest("eol_power_margin_w", "Sunlit power margin at EOL", "W"),
    QuantityOfInterest(
        "eol_power_margin_ratio", "Energy margin over required energy at EOL", "1"
    ),
    QuantityOfInterest("eol_dod", "Depth of discharge per eclipse at EOL", "1"),
    QuantityOfInterest("eol_dod_margin_ratio", "(DoD_max - DoD) / DoD_max at EOL", "1"),
    QuantityOfInterest(
        "combined_margin", "min(power margin ratio, DoD margin ratio) at EOL", "1"
    ),
    QuantityOfInterest("eol_array_power", "Array power at EOL", "W"),
    QuantityOfInterest("eol_capacity_wh", "Battery capacity at EOL", "Wh"),
    QuantityOfInterest(
        "lifetime_years", "First year the combined margin is negative", "yr"
    ),
)

_INPUT_ORDER = (
    ("bol_power", UncertaintyNature.ALEATORY),
    ("degradation_rate", UncertaintyNature.EPISTEMIC),
    ("load_factor", UncertaintyNature.EPISTEMIC),
    ("fade_rate", UncertaintyNature.EPISTEMIC),
    ("charge_efficiency", UncertaintyNature.EPISTEMIC),
    ("discharge_efficiency", UncertaintyNature.EPISTEMIC),
    ("failed_strings", UncertaintyNature.ALEATORY),
)


@dataclass(frozen=True, slots=True)
class _PowerForward:
    """Picklable forward map of :func:`power_budget_model`.

    References:
        [EQ-AERO-POWER-007].

    """

    design: PowerDesign
    eol_years: float
    horizon_years: float
    lifetime: bool

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Return the EOL margins and the lifetime.

        Args:
            parameters: The model inputs of :func:`power_budget_model`.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If an input is out of range.

        Equation:
            [EQ-AERO-POWER-007].

        References:
            [EQ-AERO-POWER-007].

        """
        args = {key: parameters[key] for key, _ in _INPUT_ORDER}
        state = power_budget(
            self.design,
            self.eol_years,
            bol_power=args["bol_power"],
            degradation_rate=args["degradation_rate"],
            fade_rate=args["fade_rate"],
            charge_efficiency=args["charge_efficiency"],
            discharge_efficiency=args["discharge_efficiency"],
            load_factor=args["load_factor"],
            failed_strings=args["failed_strings"],
        )
        out = {
            "eol_power_margin_w": state.power_margin_w,
            "eol_power_margin_ratio": state.power_margin_ratio,
            "eol_dod": state.dod,
            "eol_dod_margin_ratio": state.dod_margin_ratio,
            "combined_margin": state.combined_margin,
            "eol_array_power": state.array_power,
            "eol_capacity_wh": state.capacity_wh,
        }
        if self.lifetime:
            out["lifetime_years"] = power_lifetime(
                self.design,
                self.horizon_years,
                bol_power=args["bol_power"],
                degradation_rate=args["degradation_rate"],
                fade_rate=args["fade_rate"],
                charge_efficiency=args["charge_efficiency"],
                discharge_efficiency=args["discharge_efficiency"],
                load_factor=args["load_factor"],
                failed_strings=args["failed_strings"],
            ).years
        return out


def power_budget_model(
    design: PowerDesign,
    eol_years: float,
    *,
    bol_power: Marginal | float,
    degradation_rate: Marginal | float,
    fade_rate: Marginal | float,
    charge_efficiency: Marginal | float = 0.95,
    discharge_efficiency: Marginal | float = 0.95,
    load_factor: Marginal | float = 1.0,
    failed_strings: Marginal | float = 0.0,
    horizon_years: float | None = None,
    lifetime: bool = True,
    name: str = "power-budget",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a :class:`~probstat.modeling.ProbabilisticModel` of the EOL budget.

    Inputs (each a distribution or a plain number; at least one random):
    ``bol_power`` (W), ``degradation_rate`` (per year), ``fade_rate`` (battery,
    per cycle or per square root of cycle), ``charge_efficiency``,
    ``discharge_efficiency``, ``load_factor`` (multiplies both loads) and
    ``failed_strings`` (a real number; a continuous surrogate of the
    binomial count, see :func:`binomial_mixture_probability` for the exact
    mixture).

    Quantities of interest: ``eol_power_margin_w`` and
    ``eol_power_margin_ratio``, ``eol_dod`` and ``eol_dod_margin_ratio``,
    ``combined_margin = min(power ratio, DoD ratio)``, ``eol_array_power``,
    ``eol_capacity_wh`` and ``lifetime_years`` (first year the combined margin
    is negative, censored at the horizon, default ``2 * eol_years``).

    Series system. The mission fails when the power margin is negative *or*
    the depth of discharge exceeds its limit, i.e. when
    ``min(m_P, m_D) < 0``. The event is
    ``ModelLimitState(model, "combined_margin", 0.0, "below")``: a single
    limit state that is piecewise smooth with a kink where ``m_P = m_D``. FORM
    on the minimum linearizes only one branch at the design point; compare it
    with simulation and with :func:`component_form_bounds`.

    Args:
        design: The :class:`PowerDesign`.
        eol_years: End-of-life mission time (years), positive.
        bol_power: Beginning-of-life array power (W).
        degradation_rate: Array degradation rate per year.
        fade_rate: Battery fade rate.
        charge_efficiency: ``eta_c``.
        discharge_efficiency: ``eta_d``.
        load_factor: Load multiplier.
        failed_strings: Failed array strings.
        horizon_years: Lifetime search horizon (default ``2 * eol_years``).
        lifetime: Whether to compute ``lifetime_years`` (a bisection of about
            30 balance evaluations per model call); ``False`` removes that
            quantity and makes reliability studies about 20 times cheaper.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        The probabilistic model.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is out of range or no input is random.

    Equation:
        [EQ-AERO-POWER-007].

    References:
        [EQ-AERO-POWER-007], [REF-MELCHERS-BECK-2018], [REF-SALTELLI-2010].

    """
    if not isinstance(design, PowerDesign):
        raise TypeError("design must be a PowerDesign")
    eol = _positive(eol_years, name="eol_years")
    horizon = (
        2.0 * eol
        if horizon_years is None
        else _positive(horizon_years, name="horizon_years")
    )
    given: dict[str, Marginal | float] = {
        "bol_power": bol_power,
        "degradation_rate": degradation_rate,
        "fade_rate": fade_rate,
        "charge_efficiency": charge_efficiency,
        "discharge_efficiency": discharge_efficiency,
        "load_factor": load_factor,
        "failed_strings": failed_strings,
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, nature in _INPUT_ORDER:
        value = given[key]
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = _real(value, name=key)
        else:
            random_inputs.append(RandomInput(key, value, nature=nature))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    if "bol_power" in fixed:
        _positive(fixed["bol_power"], name="bol_power")
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        _PowerForward(design, eol, horizon, bool(lifetime)),
        _MODEL_QUANTITIES if lifetime else _MODEL_QUANTITIES[:-1],
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerReliabilityEstimate:
    """Estimate of ``P(combined margin < 0)``.

    Attributes:
        probability: The estimate.
        method: ``"form"``, ``"subset"`` or ``"monte_carlo"``.
        standard_error: Standard error (``nan`` for FORM, which has none).
        converged: Whether the method reported convergence.
        evaluations: Model evaluations.

    References:
        [EQ-AERO-POWER-008], [REF-MELCHERS-BECK-2018].

    """

    probability: float
    method: str
    standard_error: float
    converged: bool
    evaluations: int


def power_failure_probability(
    model: ProbabilisticModel,
    *,
    quantity: str = "combined_margin",
    method: Literal["form", "subset", "monte_carlo"] = "form",
    samples: int = 1000,
    seed: int | RandomStream | None = 0,
    form_relative_step: float = 1.0e-6,
) -> PowerReliabilityEstimate:
    """Estimate ``P(margin < 0)`` for a :func:`power_budget_model`.

    The default quantity ``combined_margin`` is the series-system margin
    ``min(m_P, m_D)``, so the event is "the power margin is negative or the
    depth of discharge exceeds its limit". Use ``quantity="eol_power_margin_ratio"``
    or ``"eol_dod_margin_ratio"`` for one failure mode. FORM linearizes the
    limit state at one design point; on the minimum it follows one branch, so
    the estimate can miss the other mode (compare with simulation).

    Args:
        model: Model of :func:`power_budget_model`.
        quantity: A margin whose negative sign is failure.
        method: ``"form"``, ``"subset"`` (``samples`` per level) or
            ``"monte_carlo"``.
        samples: Sample size of the simulation methods.
        seed: Seed or stream of the simulation methods.
        form_relative_step: Finite-difference step of FORM.

    Returns:
        The :class:`PowerReliabilityEstimate`.

    Raises:
        TypeError: If ``model`` is not a probabilistic model.
        ValueError: If the method is unknown or a value is out of range.

    Equation:
        [EQ-AERO-POWER-008].

    References:
        [EQ-AERO-POWER-008], [REF-HASOFER-LIND-1974], [REF-AU-BECK-2001],
        [REF-MELCHERS-BECK-2018].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    state = ModelLimitState(model, quantity, 0.0, "below")
    if method == "form":
        form = form_hlrf(
            state,
            relative_step=_positive(form_relative_step, name="form_relative_step"),
        )
        return PowerReliabilityEstimate(
            form.probability, "form", math.nan, form.converged, form.evaluations
        )
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    if method == "subset":
        sub = subset_simulation(state, samples_per_level=samples, seed=seed)
        return PowerReliabilityEstimate(
            sub.probability,
            "subset",
            sub.coefficient_of_variation * sub.probability,
            sub.converged,
            sub.evaluations,
        )
    if method == "monte_carlo":
        mc = model_monte_carlo(state, samples, seed=seed)
        p = mc.probability
        return PowerReliabilityEstimate(
            p, "monte_carlo", math.sqrt(p * (1.0 - p) / samples), True, mc.evaluations
        )
    raise ValueError("method must be 'form', 'subset' or 'monte_carlo'")


@dataclass(frozen=True, slots=True)
class ComponentBounds:
    """FORM probabilities of the two modes and the series-system bounds.

    Attributes:
        power: FORM ``P(m_P < 0)``.
        dod: FORM ``P(m_D < 0)``.
        lower: ``max(power, dod)``, a lower bound of the series probability.
        upper: ``min(1, power + dod)``, the first-order (union) upper bound.
        evaluations: Model evaluations.

    References:
        [EQ-AERO-POWER-008].

    """

    power: float
    dod: float
    lower: float
    upper: float
    evaluations: int


def component_form_bounds(
    model: ProbabilisticModel, *, form_relative_step: float = 1.0e-6
) -> ComponentBounds:
    """Return FORM of each failure mode and the series bounds.

    For events ``A`` and ``B``: ``max(P(A), P(B)) <= P(A or B) <= P(A) +
    P(B)``. The component probabilities are FORM estimates, so the bounds
    hold only to the accuracy of FORM on each mode.

    Args:
        model: Model of :func:`power_budget_model`.
        form_relative_step: Finite-difference step of FORM.

    Returns:
        The :class:`ComponentBounds`.

    Raises:
        TypeError: If ``model`` is not a probabilistic model.
        ValueError: If the step is out of range.

    Equation:
        [EQ-AERO-POWER-008].

    References:
        [EQ-AERO-POWER-008], [REF-MELCHERS-BECK-2018].

    """
    a = power_failure_probability(
        model,
        quantity="eol_power_margin_ratio",
        form_relative_step=form_relative_step,
    )
    b = power_failure_probability(
        model, quantity="eol_dod_margin_ratio", form_relative_step=form_relative_step
    )
    return ComponentBounds(
        a.probability,
        b.probability,
        max(a.probability, b.probability),
        min(1.0, a.probability + b.probability),
        a.evaluations + b.evaluations,
    )


def string_failure_pmf(
    total_strings: int, failure_probability: float
) -> tuple[float, ...]:
    """Return the binomial law of the number of failed strings.

    Each of ``N`` strings fails independently with probability ``q`` over the
    mission, so ``P(n_f = k) = C(N, k) q^k (1 - q)^(N-k)``.

    Args:
        total_strings: ``N``, at least 1.
        failure_probability: ``q`` in ``[0, 1]``.

    Returns:
        ``(P(n_f = 0), ..., P(n_f = N))``.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-POWER-009].

    References:
        [EQ-AERO-POWER-009].

    """
    n = _integer(total_strings, name="total_strings", minimum=1)
    q = _real(failure_probability, name="failure_probability")
    if not 0.0 <= q <= 1.0:
        raise ValueError("failure_probability must lie in [0, 1]")
    law = Binomial(n, q)
    return tuple(law.pmf(k) for k in range(n + 1))


@dataclass(frozen=True, slots=True)
class BinomialMixture:
    """Probability mixed over a binomial number of failed strings.

    Attributes:
        probability: ``sum_k P(n_f = k) p(k)`` over the enumerated counts.
        neglected_mass: Binomial mass of the counts that were skipped; the
            truncation error lies in ``[0, neglected_mass]``.
        counts: The enumerated numbers of failed strings.

    References:
        [EQ-AERO-POWER-009].

    """

    probability: float
    neglected_mass: float
    counts: tuple[int, ...]


def binomial_mixture_probability(
    probability_given_failures: Callable[[int], float],
    total_strings: int,
    failure_probability: float,
    *,
    tail: float = 1.0e-9,
) -> BinomialMixture:
    """Return ``P(fail) = sum_k Binomial(k; N, q) P(fail | n_f = k)``.

    The counts whose binomial mass is below ``tail / (N + 1)`` are skipped
    and their total mass is reported, so the truncation error is bounded by
    ``neglected_mass`` (``p(k)`` is a probability).

    Args:
        probability_given_failures: Maps the failed count ``k`` to the
            conditional failure probability, in ``[0, 1]``.
        total_strings: ``N``, at least 1.
        failure_probability: ``q`` in ``[0, 1]``.
        tail: Mass threshold, positive.

    Returns:
        The :class:`BinomialMixture`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If an argument is out of range or a conditional
            probability is outside ``[0, 1]``.

    Equation:
        [EQ-AERO-POWER-009].

    References:
        [EQ-AERO-POWER-009].

    """
    if not callable(probability_given_failures):
        raise TypeError("probability_given_failures must be callable")
    cutoff = _positive(tail, name="tail") / (
        _integer(total_strings, name="total_strings", minimum=1) + 1
    )
    pmf = string_failure_pmf(total_strings, failure_probability)
    total = 0.0
    neglected = 0.0
    used: list[int] = []
    for k, mass in enumerate(pmf):
        if mass < cutoff:
            neglected += mass
            continue
        p = _real(probability_given_failures(k), name="conditional probability")
        if not 0.0 <= p <= 1.0:
            raise ValueError("conditional probability must lie in [0, 1]")
        total += mass * p
        used.append(k)
    return BinomialMixture(total, neglected, tuple(used))


@dataclass(frozen=True, slots=True)
class RequiredPower:
    """Result of :func:`required_bol_power`.

    Attributes:
        bol_power: Smallest tested power with ``P <= target`` (W).
        probability: Estimated ``P`` at ``bol_power``.
        lower_power: Largest tested power with ``P > target`` (W); the true
            requirement lies in ``(lower_power, bol_power]``.
        target: Target failure probability.
        method: Probability method.
        iterations: Bisection iterations.
        evaluations: Probability evaluations.

    References:
        [EQ-AERO-POWER-010].

    """

    bol_power: float
    probability: float
    lower_power: float
    target: float
    method: str
    iterations: int
    evaluations: int


def required_bol_power(
    build: Callable[[float], ProbabilisticModel],
    target_probability: float,
    bracket: tuple[float, float],
    *,
    method: Literal["form", "subset", "monte_carlo"] = "form",
    samples: int = 1000,
    seed: int | RandomStream | None = 0,
    power_tolerance: float = 0.5,
    max_iterations: int = 60,
) -> RequiredPower:
    """Return the smallest nominal beginning-of-life power meeting a target.

    ``build(p)`` returns the model for the nominal BOL power ``p`` (W); for
    fixed other inputs the combined margin is increasing in ``p`` pathwise,
    so with the *same seed* (common random numbers) the Monte Carlo failure
    probability is a non-increasing step function of ``p``, and so is the
    FORM probability up to its approximation. The root of ``P(p) = target``
    is found by bisection, not a secant method, which a step function can
    defeat.

    Args:
        build: Maps a BOL power to a model.
        target_probability: Allowed failure probability, in ``(0, 1)``.
        bracket: ``(low, high)`` powers (W) with ``P(low) > target >=
            P(high)``.
        method: Probability method.
        samples: Sample size of the simulation methods.
        seed: Seed (common random numbers across powers).
        power_tolerance: Bracket width at termination (W).
        max_iterations: Bisection budget.

    Returns:
        The :class:`RequiredPower`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the bracket does not enclose the target.

    Equation:
        [EQ-AERO-POWER-010].

    References:
        [EQ-AERO-POWER-010], [REF-MELCHERS-BECK-2018].

    """
    if not callable(build):
        raise TypeError("build must be callable")
    target = _open_probability(target_probability, name="target_probability")
    tolerance = _positive(power_tolerance, name="power_tolerance")
    budget = _integer(max_iterations, name="max_iterations", minimum=1)
    lo = _real(bracket[0], name="bracket")
    hi = _real(bracket[1], name="bracket")
    if lo >= hi:
        raise ValueError("bracket must satisfy low < high")
    count = 0

    def probability(p: float) -> float:
        """Return the failure probability at nominal BOL power ``p``.

        References:
            [EQ-AERO-POWER-008]. No equation is evaluated.

        """
        nonlocal count
        count += 1
        return power_failure_probability(
            build(p), method=method, samples=samples, seed=seed
        ).probability

    p_lo = probability(lo)
    p_hi = probability(hi)
    if p_lo <= target:
        raise ValueError("P at the lower bracket end already meets the target")
    if p_hi > target:
        raise ValueError("P at the upper bracket end exceeds the target")
    iterations = 0
    while hi - lo > tolerance and iterations < budget:
        mid = 0.5 * (lo + hi)
        p_mid = probability(mid)
        if p_mid <= target:
            hi, p_hi = mid, p_mid
        else:
            lo = mid
        iterations += 1
    return RequiredPower(hi, p_hi, lo, target, method, iterations, count)
