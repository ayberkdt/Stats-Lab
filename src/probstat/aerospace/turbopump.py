"""Turbopump suction performance and cavitation risk of a propellant feed line.

A deliberately simple, units-explicit model of whether a pump inlet keeps the
liquid above its vapour pressure. All quantities are SI (Pa, K, m, m^3/s,
kg/m^3, rad/s, m/s^2). Contents:

- :class:`VapourPressureModel`: ``ln p_v = A - B / (T + C)`` (Antoine form in
  natural logarithms; ``C = 0`` is the integrated Clausius-Clapeyron law with
  constant latent heat and an ideal vapour, built by
  :func:`clausius_clapeyron_model`). The constants are *user supplied*; the
  library ships only :data:`DEMONSTRATION_VAPOUR_MODEL`, a synthetic
  LOX-class law that is not fitted to any real fluid. Temperatures outside
  the stated validity interval are rejected.
- :func:`line_loss_head` (Darcy-Weisbach) and :func:`npsh_available`: the net
  positive suction head available, with a flag when the inlet total head is at
  or below the vapour head.
- :func:`npsh_required_from_suction_specific_speed`,
  :func:`npsh_required_from_thoma` and the unit conversions between the
  dimensional (rpm, m^3/s, m) and dimensionless (rad/s, m^3/s, ``g H``)
  suction specific speeds; :func:`suction_margin`.
- :func:`pump_operating_point`: intersection of a quadratic pump head-flow
  curve with a quadratic system curve, with the shaft power evaluated through
  :func:`probstat.aerospace.liquid_engine.pump_power`.
- :func:`turbopump_cavitation_model`: a
  :class:`~probstat.modeling.ProbabilisticModel` (failure when the margin
  ``NPSH_a / NPSH_r`` drops below one) and
  :func:`minimum_tank_pressure_for_probability` (tank-pressurisation sizing
  by a monotone root find on the FORM reliability index).

Validity and limits. The suction head is a one-dimensional steady budget:
a single liquid column of height ``h`` along the acceleration axis, a
lumped loss coefficient, no ullage-gas thermal collapse, no propellant
sloshing or vortexing, no bubble dynamics, no thermodynamic suppression
head (which in cryogenic liquids lowers the head actually required) and no
transient start-up. The *required* head enters only through a user-supplied
suction specific speed or Thoma number; no value of either is taken from the
literature here, and the numbers in the demonstration case are synthetic and
are not data from any engine, pump or vehicle. The definitions of the suction
specific speed and of the head-breakdown criterion follow the cited sources
but those two documents have not been read for this implementation: the
formulas below are the standard definitions re-derived dimensionally, and the
sources are cited for the concept only (to be checked against the sources).

Experimental (added in 0.28).

References:
    [REF-NASA-SP-8052-1971], [REF-BRENNEN-1994].

"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from probstat.aerospace.liquid_engine import G0, PumpSpec
from probstat.aerospace.liquid_engine import pump_power as _pump_power
from probstat.distributions import LogNormal, Normal
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.optimization import brent_root
from probstat.reliability import ModelLimitState, form_hlrf
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "DEMONSTRATION_VAPOUR_MODEL",
    "NpshBudget",
    "OperatingPoint",
    "PumpDesign",
    "VapourPressureModel",
    "clausius_clapeyron_model",
    "line_loss_head",
    "minimum_tank_pressure_for_probability",
    "npsh_available",
    "npsh_required_from_suction_specific_speed",
    "npsh_required_from_thoma",
    "pump_operating_point",
    "suction_margin",
    "suction_specific_speed",
    "suction_specific_speed_dimensional",
    "suction_specific_speed_from_dimensional",
    "turbopump_cavitation_model",
]

_RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive real.

    Args:
        value: Candidate number.
        name: Argument name for messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite and positive.

    References:
        [REF-PYTHON-DATAMODEL]. A software validator; no equation.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative real.

    Args:
        value: Candidate number.
        name: Argument name for messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite or is negative.

    References:
        [REF-PYTHON-DATAMODEL]. A software validator; no equation.

    """
    number = require_finite_real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


# ---------------------------------------------------------------------------
# Vapour pressure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VapourPressureModel:
    """Vapour-pressure law ``ln p_v = A - B / (T + C)`` on a validity interval.

    Args:
        a: Constant ``A`` (dimensionless; ``p_v`` in Pa).
        b: Constant ``B`` in K, ``> 0``.
        c: Constant ``C`` in K (``0`` for the Clausius-Clapeyron form).
        t_min: Lower validity temperature (K), ``> 0``.
        t_max: Upper validity temperature (K), above ``t_min``.
        label: Free text naming the origin of the constants.

    ``T + C`` must be positive on the whole interval. The constants are the
    caller's responsibility: nothing here is fitted to a real fluid.

    References:
        [EQ-AERO-PUMP-001].

    """

    a: float
    b: float
    c: float
    t_min: float
    t_max: float
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the constants and the interval.

        Raises:
            TypeError: If a field is not real or ``label`` is not text.
            ValueError: If ``b`` is not positive, the interval is empty or
                ``T + C`` is not positive on it.

        References:
            [EQ-AERO-PUMP-001].

        """
        a = require_finite_real(self.a, name="a")
        b = _positive(self.b, "b")
        c = require_finite_real(self.c, name="c")
        lo = _positive(self.t_min, "t_min")
        hi = _positive(self.t_max, "t_max")
        if lo >= hi:
            raise ValueError("t_min must be below t_max")
        if lo + c <= 0.0:
            raise ValueError("t_min + c must be positive")
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        for key, value in (("a", a), ("b", b), ("c", c), ("t_min", lo), ("t_max", hi)):
            object.__setattr__(self, key, value)

    def pressure(self, temperature: float) -> float:
        """Return the vapour pressure ``p_v(T)`` in Pa.

        Args:
            temperature: Liquid temperature in K.

        Returns:
            ``exp(A - B / (T + C))``.

        Raises:
            TypeError: If ``temperature`` is not real.
            ValueError: If ``temperature`` is not finite or lies outside
                ``[t_min, t_max]`` (non-physical for this law).

        Equation:
            [EQ-AERO-PUMP-001].

        References:
            [EQ-AERO-PUMP-001].

        """
        t = require_finite_real(temperature, name="temperature")
        if not self.t_min <= t <= self.t_max:
            raise ValueError(
                f"temperature {t:g} K is outside the validity interval "
                f"[{self.t_min:g}, {self.t_max:g}] K of the vapour-pressure law"
            )
        return math.exp(self.a - self.b / (t + self.c))


def clausius_clapeyron_model(
    *,
    reference_pressure: float,
    reference_temperature: float,
    latent_heat_over_gas_constant: float,
    t_min: float,
    t_max: float,
    label: str = "",
) -> VapourPressureModel:
    """Build the integrated Clausius-Clapeyron law from one reference point.

    From ``d ln p / dT = L / (R T^2)`` with constant specific latent heat ``L``
    over the specific gas constant ``R`` (units K) one gets
    ``ln(p / p_ref) = (L / R) (1 / T_ref - 1 / T)``, i.e. ``A = ln p_ref +
    (L/R) / T_ref``, ``B = L / R``, ``C = 0``.

    Args:
        reference_pressure: ``p_ref`` in Pa, ``> 0``.
        reference_temperature: ``T_ref`` in K, ``> 0``.
        latent_heat_over_gas_constant: ``L / R`` in K, ``> 0`` (user supplied).
        t_min: Lower validity temperature (K).
        t_max: Upper validity temperature (K).
        label: Text naming the origin of the constants.

    Returns:
        The vapour-pressure model.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-PUMP-001].

    References:
        [EQ-AERO-PUMP-001].

    """
    p_ref = _positive(reference_pressure, "reference_pressure")
    t_ref = _positive(reference_temperature, "reference_temperature")
    b = _positive(latent_heat_over_gas_constant, "latent_heat_over_gas_constant")
    return VapourPressureModel(
        a=math.log(p_ref) + b / t_ref,
        b=b,
        c=0.0,
        t_min=t_min,
        t_max=t_max,
        label=label,
    )


DEMONSTRATION_VAPOUR_MODEL = clausius_clapeyron_model(
    reference_pressure=1.0e5,
    reference_temperature=90.0,
    latent_heat_over_gas_constant=820.0,
    t_min=70.0,
    t_max=130.0,
    label=(
        "SYNTHETIC LOX-class law for examples and tests: p_v(90 K) = 1.0e5 Pa "
        "and L/R = 820 K are invented round numbers, not fitted to any fluid"
    ),
)
"""Synthetic vapour-pressure law for examples and tests only."""


# ---------------------------------------------------------------------------
# Net positive suction head available
# ---------------------------------------------------------------------------


def line_loss_head(
    *,
    volumetric_flow: float,
    diameter: float,
    length: float,
    friction_factor: float,
    minor_loss_coefficient: float = 0.0,
) -> float:
    """Return the Darcy-Weisbach loss head ``(f L / D + K) v^2 / (2 g0)`` in m.

    ``v = 4 Q / (pi D^2)`` is the mean line velocity, ``f`` the (Darcy) friction
    factor supplied by the caller (nothing is computed from a roughness
    correlation) and ``K`` the sum of minor-loss coefficients.

    Args:
        volumetric_flow: ``Q`` in m^3/s, ``>= 0``.
        diameter: ``D`` in m, ``> 0``.
        length: ``L`` in m, ``>= 0``.
        friction_factor: Darcy friction factor ``f``, ``>= 0``.
        minor_loss_coefficient: ``K``, ``>= 0``.

    Returns:
        The head loss in metres of liquid column.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-PUMP-002].

    References:
        [EQ-AERO-PUMP-002].

    """
    q = _non_negative(volumetric_flow, "volumetric_flow")
    d = _positive(diameter, "diameter")
    ell = _non_negative(length, "length")
    f = _non_negative(friction_factor, "friction_factor")
    k = _non_negative(minor_loss_coefficient, "minor_loss_coefficient")
    velocity = 4.0 * q / (math.pi * d * d)
    return (f * ell / d + k) * velocity * velocity / (2.0 * G0)


@dataclass(frozen=True, slots=True)
class NpshBudget:
    """Terms of the available suction head, all in metres of liquid column.

    Args:
        pressure_head: ``p_tank / (rho g0)``.
        column_head: ``h a / g0`` (hydrostatic plus acceleration head).
        loss_head: Line loss head (subtracted).
        vapour_head: ``p_v / (rho g0)`` (subtracted).
        npsh_available: ``pressure + column - loss - vapour`` (may be negative).
        flag: ``"ok"`` or ``"inlet_at_or_below_vapour_pressure"`` when
            ``npsh_available <= 0`` (the liquid vaporises before the pump).

    References:
        [EQ-AERO-PUMP-003].

    """

    pressure_head: float
    column_head: float
    loss_head: float
    vapour_head: float
    npsh_available: float
    flag: str


def npsh_available(
    *,
    tank_pressure: float,
    density: float,
    vapour_pressure: float,
    column_height: float,
    axial_acceleration: float,
    loss_head: float,
) -> NpshBudget:
    """Return the net positive suction head available at the pump inlet.

    ``NPSH_a = (p_tank - p_v) / (rho g0) + h a / g0 - h_loss``. The column
    term is the pressure rise of a liquid column of height ``h`` under the
    *proper* acceleration ``a`` along the feed axis (``a = g0`` for a vehicle
    at rest on the pad, the thrust acceleration in flight, ``0`` in free fall):
    ``rho a h / (rho g0)``. ``tank_pressure`` is the absolute pressure at the
    free surface. The head is referenced to the inlet total head, so the
    inlet velocity head is already included (the loss coefficient carries the
    entrance loss).

    Args:
        tank_pressure: Absolute ullage pressure ``p_tank`` in Pa, ``> 0``.
        density: Liquid density ``rho`` in kg/m^3, ``> 0``.
        vapour_pressure: ``p_v`` in Pa, ``> 0``.
        column_height: ``h`` in m, ``>= 0``.
        axial_acceleration: Proper acceleration ``a`` in m/s^2, ``>= 0``.
        loss_head: Line loss head in m, ``>= 0`` (see :func:`line_loss_head`).

    Returns:
        The budget; ``npsh_available`` may be negative and is then flagged.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-PUMP-003].

    References:
        [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

    """
    p = _positive(tank_pressure, "tank_pressure")
    rho = _positive(density, "density")
    pv = _positive(vapour_pressure, "vapour_pressure")
    h = _non_negative(column_height, "column_height")
    a = _non_negative(axial_acceleration, "axial_acceleration")
    loss = _non_negative(loss_head, "loss_head")
    pressure_head = p / (rho * G0)
    vapour_head = pv / (rho * G0)
    column_head = h * a / G0
    value = pressure_head + column_head - loss - vapour_head
    flag = "ok" if value > 0.0 else "inlet_at_or_below_vapour_pressure"
    return NpshBudget(pressure_head, column_head, loss, vapour_head, value, flag)


# ---------------------------------------------------------------------------
# Required head
# ---------------------------------------------------------------------------


def suction_specific_speed(
    *, speed: float, volumetric_flow: float, npsh_required: float
) -> float:
    """Return the dimensionless suction specific speed ``S* = w sqrt(Q) / (g H)^(3/4)``.

    Args:
        speed: Shaft speed ``w`` in rad/s, ``> 0``.
        volumetric_flow: Inlet flow ``Q`` in m^3/s, ``> 0``.
        npsh_required: ``NPSH_r`` in m, ``> 0``.

    Returns:
        ``S*`` (dimensionless; ``w`` in rad/s, ``Q`` in m^3/s,
        ``g H`` in m^2/s^2).

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-PUMP-004].

    References:
        [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

    """
    w = _positive(speed, "speed")
    q = _positive(volumetric_flow, "volumetric_flow")
    h = _positive(npsh_required, "npsh_required")
    return w * math.sqrt(q) / math.pow(G0 * h, 0.75)


def suction_specific_speed_dimensional(s_star: float) -> float:
    """Convert ``S*`` to the dimensional form ``N sqrt(Q) / NPSH^(3/4)``.

    The dimensional variant uses ``N`` in rev/min, ``Q`` in m^3/s and ``NPSH``
    in m, so that ``S = S* g0^(3/4) 60 / (2 pi)``. Its numerical value is
    unit-system dependent and must always be quoted with its units.

    Args:
        s_star: Dimensionless ``S*``, ``> 0``.

    Returns:
        ``S`` in ``rpm (m^3/s)^(1/2) / m^(3/4)``.

    Raises:
        TypeError: If ``s_star`` is not real.
        ValueError: If ``s_star`` is not positive.

    Equation:
        [EQ-AERO-PUMP-004].

    References:
        [EQ-AERO-PUMP-004].

    """
    return _positive(s_star, "s_star") * math.pow(G0, 0.75) * _RPM_PER_RAD_S


def suction_specific_speed_from_dimensional(s_metric: float) -> float:
    """Convert the (rpm, m^3/s, m) suction specific speed to ``S*``.

    Args:
        s_metric: Dimensional ``S``, ``> 0``.

    Returns:
        Dimensionless ``S*`` (inverse of
        :func:`suction_specific_speed_dimensional`).

    Raises:
        TypeError: If ``s_metric`` is not real.
        ValueError: If ``s_metric`` is not positive.

    Equation:
        [EQ-AERO-PUMP-004].

    References:
        [EQ-AERO-PUMP-004].

    """
    return _positive(s_metric, "s_metric") / (math.pow(G0, 0.75) * _RPM_PER_RAD_S)


def npsh_required_from_suction_specific_speed(
    *, speed: float, volumetric_flow: float, s_star: float
) -> float:
    """Return ``NPSH_r = (w sqrt(Q) / S*)^(4/3) / g0`` in m.

    The inversion of :func:`suction_specific_speed`. ``S*`` is a user input
    (a value characteristic of the inducer design); no default is offered.

    Args:
        speed: ``w`` in rad/s, ``> 0``.
        volumetric_flow: ``Q`` in m^3/s, ``> 0``.
        s_star: Dimensionless suction specific speed, ``> 0``.

    Returns:
        The required head in m.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-PUMP-004].

    References:
        [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

    """
    w = _positive(speed, "speed")
    q = _positive(volumetric_flow, "volumetric_flow")
    s = _positive(s_star, "s_star")
    return math.pow(w * math.sqrt(q) / s, 4.0 / 3.0) / G0


def npsh_required_from_thoma(*, thoma_number: float, pump_head: float) -> float:
    """Return ``NPSH_r = sigma H`` in m from a cavitation (Thoma) number.

    Args:
        thoma_number: Critical Thoma number ``sigma``, ``> 0`` (user input).
        pump_head: Pump head ``H`` in m, ``> 0``.

    Returns:
        The required head in m.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-PUMP-005].

    References:
        [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

    """
    return _positive(thoma_number, "thoma_number") * _positive(pump_head, "pump_head")


def suction_margin(available: float, required: float) -> tuple[float, float]:
    """Return the margin ratio ``NPSH_a / NPSH_r`` and the head margin in m.

    Args:
        available: ``NPSH_a`` in m (may be negative).
        required: ``NPSH_r`` in m, ``> 0``.

    Returns:
        ``(ratio, head_margin)`` with ``head_margin = NPSH_a - NPSH_r``;
        cavitation (head breakdown) is expected when the ratio is below one.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If ``required`` is not positive.

    Equation:
        [EQ-AERO-PUMP-006].

    References:
        [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

    """
    a = require_finite_real(available, name="available")
    r = _positive(required, "required")
    return a / r, a - r


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Intersection of a pump curve and a system curve.

    Args:
        volumetric_flow: ``Q`` in m^3/s.
        head: ``H`` in m.
        shaft_power: ``rho g0 Q H / eta`` in W via
            :func:`~probstat.aerospace.liquid_engine.pump_power`.
        iterations: Brent iterations.
        converged: Whether the root search converged.

    References:
        [EQ-AERO-PUMP-007].

    """

    volumetric_flow: float
    head: float
    shaft_power: float
    iterations: int
    converged: bool


def pump_operating_point(
    *,
    pump_coefficients: tuple[float, float, float],
    static_lift: float,
    system_coefficient: float,
    density: float,
    efficiency: float,
) -> OperatingPoint:
    """Solve ``h0 + h1 Q + h2 Q^2 = h_s + k Q^2`` for the operating flow.

    The pump curve is ``H_p(Q) = h0 + h1 Q + h2 Q^2`` (m, with ``Q`` in m^3/s)
    with ``h0 > 0``, ``h1 <= 0`` and ``h2 < 0``, hence strictly decreasing for
    ``Q >= 0``; the system curve ``H_s(Q) = h_s + k Q^2`` (``h_s >= 0``,
    ``k >= 0``) is non-decreasing, so the intersection is unique when
    ``h0 > h_s`` (otherwise there is no flow and a ``ValueError`` is raised).
    It is found by Brent's method on ``[0, Q_0]`` where ``Q_0`` is the
    zero-head flow of the pump curve. Shaft power is
    ``mdot dp / (rho eta)`` with ``mdot = rho Q`` and ``dp = rho g0 H``.

    Args:
        pump_coefficients: ``(h0, h1, h2)`` in m, m/(m^3/s), m/(m^3/s)^2.
        static_lift: ``h_s`` in m.
        system_coefficient: ``k`` in m/(m^3/s)^2.
        density: ``rho`` in kg/m^3.
        efficiency: Pump efficiency in ``(0, 1]``.

    Returns:
        The operating point.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a coefficient is out of range or there is no
            intersection with positive flow.

    Equation:
        [EQ-AERO-PUMP-007].

    References:
        [REF-BRENNEN-1994].

    """
    if len(pump_coefficients) != 3:
        raise ValueError("pump_coefficients must be (h0, h1, h2)")
    h0 = _positive(pump_coefficients[0], "h0")
    h1 = require_finite_real(pump_coefficients[1], name="h1")
    h2 = require_finite_real(pump_coefficients[2], name="h2")
    if h1 > 0.0 or h2 >= 0.0:
        raise ValueError("pump curve must decrease: need h1 <= 0 and h2 < 0")
    hs = _non_negative(static_lift, "static_lift")
    k = _non_negative(system_coefficient, "system_coefficient")
    rho = _positive(density, "density")
    eta = _positive(efficiency, "efficiency")
    if eta > 1.0:
        raise ValueError("efficiency must lie in (0, 1]")
    if hs >= h0:
        raise ValueError("static_lift must be below the shut-off head h0")
    zero_head_flow = 2.0 * h0 / (-h1 + math.sqrt(h1 * h1 - 4.0 * h2 * h0))

    def residual(q: float) -> float:
        """Return ``H_p(Q) - H_s(Q)``.

        Args:
            q: Flow in m^3/s.

        Returns:
            The head difference in m.

        References:
            [EQ-AERO-PUMP-007].

        """
        return h0 + h1 * q + (h2 - k) * q * q - hs

    root = brent_root(residual, 0.0, zero_head_flow, tolerance=1e-14 * zero_head_flow)
    q = float(root.root)
    head = h0 + h1 * q + h2 * q * q
    if head <= 0.0 or q <= 0.0:
        raise ValueError("operating point has non-positive flow or head")
    power = _pump_power(
        PumpSpec(
            mass_flow=rho * q,
            density=rho,
            pressure_rise=rho * G0 * head,
            efficiency=eta,
        )
    )
    return OperatingPoint(q, head, power, root.iterations, bool(root.converged))


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_UNITS = {
    "tank_pressure": "Pa",
    "temperature": "K",
    "axial_acceleration": "m/s^2",
    "loss_coefficient": "-",
    "suction_specific_speed": "-",
    "pump_speed": "rad/s",
}

_QUANTITIES = (
    QuantityOfInterest("npsh_available", "Net positive suction head available", "m"),
    QuantityOfInterest("npsh_required", "Net positive suction head required", "m"),
    QuantityOfInterest("margin", "Ratio NPSH_a / NPSH_r", "-"),
    QuantityOfInterest("head_margin", "NPSH_a - NPSH_r", "m"),
    QuantityOfInterest("vapour_pressure", "Vapour pressure at the temperature", "Pa"),
)


@dataclass(frozen=True, slots=True)
class _CavitationForward:
    """Picklable forward map of :func:`turbopump_cavitation_model`.

    Args:
        vapour: Vapour-pressure law.
        density: ``rho`` in kg/m^3.
        volumetric_flow: ``Q`` in m^3/s.
        column_height: ``h`` in m.
        line_diameter: ``D`` in m.

    References:
        [EQ-AERO-PUMP-008].

    """

    vapour: VapourPressureModel
    density: float
    volumetric_flow: float
    column_height: float
    line_diameter: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the outputs for one realization.

        Args:
            parameters: Values of the six model inputs.

        Returns:
            Available and required head, margin ratio and head margin.

        Raises:
            ValueError: If a draw is non-physical (for example a temperature
                outside the validity interval of the vapour-pressure law).

        Equation:
            [EQ-AERO-PUMP-008].

        References:
            [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

        """
        pv = self.vapour.pressure(parameters["temperature"])
        velocity = 4.0 * self.volumetric_flow / (math.pi * self.line_diameter**2)
        loss = _non_negative(parameters["loss_coefficient"], "loss_coefficient") * (
            velocity * velocity / (2.0 * G0)
        )
        budget = npsh_available(
            tank_pressure=parameters["tank_pressure"],
            density=self.density,
            vapour_pressure=pv,
            column_height=self.column_height,
            axial_acceleration=parameters["axial_acceleration"],
            loss_head=loss,
        )
        required = npsh_required_from_suction_specific_speed(
            speed=parameters["pump_speed"],
            volumetric_flow=self.volumetric_flow,
            s_star=parameters["suction_specific_speed"],
        )
        ratio, head_margin = suction_margin(budget.npsh_available, required)
        return {
            "npsh_available": budget.npsh_available,
            "npsh_required": required,
            "margin": ratio,
            "head_margin": head_margin,
            "vapour_pressure": pv,
        }


def turbopump_cavitation_model(
    *,
    tank_pressure: Marginal | float,
    temperature: Marginal | float,
    axial_acceleration: Marginal | float,
    loss_coefficient: Marginal | float,
    suction_specific_speed: Marginal | float,
    pump_speed: Marginal | float,
    density: float,
    volumetric_flow: float,
    column_height: float,
    line_diameter: float,
    vapour_model: VapourPressureModel = DEMONSTRATION_VAPOUR_MODEL,
    name: str = "turbopump-cavitation",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic pump-inlet cavitation model.

    Uncertain-or-fixed inputs (distribution or number): ``tank_pressure``
    (absolute, Pa), ``temperature`` (K, enters through the vapour pressure),
    ``axial_acceleration`` (proper, m/s^2), ``loss_coefficient`` (total
    ``f L / D + K`` of :func:`line_loss_head`, dimensionless),
    ``suction_specific_speed`` (dimensionless ``S*``, the scatter of the
    required head) and ``pump_speed`` (rad/s). Fixed numbers: ``density``,
    ``volumetric_flow`` (the operating flow; it is not coupled to the pump
    speed here), ``column_height`` and ``line_diameter``. Outputs:
    ``npsh_available``, ``npsh_required`` (m), ``margin = NPSH_a / NPSH_r``,
    ``head_margin`` (m) and ``vapour_pressure`` (Pa). Failure is
    ``margin < 1`` (``ModelLimitState(model, "margin", 1.0, "below")``). A
    draw whose temperature lies outside the validity interval of the vapour
    law, or whose acceleration or loss coefficient is negative, fails the run
    (see ``on_failure``), so use bounded or positive laws. Inputs are
    independent. A negative ``NPSH_a`` is a valid (certainly cavitating)
    outcome and yields a negative margin.

    Args:
        tank_pressure: ``p_tank``.
        temperature: ``T``.
        axial_acceleration: ``a``.
        loss_coefficient: ``f L / D + K``.
        suction_specific_speed: ``S*``.
        pump_speed: ``w``.
        density: ``rho`` (kg/m^3).
        volumetric_flow: ``Q`` (m^3/s).
        column_height: ``h`` (m).
        line_diameter: ``D`` (m).
        vapour_model: Vapour-pressure law (default: the synthetic one).
        name: Model label.
        on_failure: Failure policy.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is outside its domain or no input is
            random.

    Equation:
        [EQ-AERO-PUMP-008].

    References:
        [REF-BRENNEN-1994], [REF-NASA-SP-8052-1971].

    """
    if not isinstance(vapour_model, VapourPressureModel):
        raise TypeError("vapour_model must be a VapourPressureModel")
    forward = _CavitationForward(
        vapour_model,
        _positive(density, "density"),
        _positive(volumetric_flow, "volumetric_flow"),
        _non_negative(column_height, "column_height"),
        _positive(line_diameter, "line_diameter"),
    )
    given: dict[str, Marginal | float] = {
        "tank_pressure": tank_pressure,
        "temperature": temperature,
        "axial_acceleration": axial_acceleration,
        "loss_coefficient": loss_coefficient,
        "suction_specific_speed": suction_specific_speed,
        "pump_speed": pump_speed,
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, value in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            number = require_finite_real(value, name=key)
            if key in ("axial_acceleration", "loss_coefficient"):
                _non_negative(number, key)
            elif key != "temperature":
                _positive(number, key)
            else:
                vapour_model.pressure(number)
            fixed[key] = number
        else:
            random_inputs.append(RandomInput(key, value, unit=_UNITS[key]))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        forward,
        _QUANTITIES,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# ---------------------------------------------------------------------------
# Tank-pressure sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PumpDesign:
    """Mean tank pressure meeting a target cavitation probability by FORM.

    Args:
        tank_pressure: Mean (nominal) absolute tank pressure in Pa.
        probability: FORM probability of ``margin < 1`` at that pressure.
        reliability_index: FORM index at that pressure.
        target_probability: The requested probability.
        iterations: Brent iterations.
        converged: Whether the root search converged.

    References:
        [EQ-AERO-PUMP-009].

    """

    tank_pressure: float
    probability: float
    reliability_index: float
    target_probability: float
    iterations: int
    converged: bool


def minimum_tank_pressure_for_probability(
    *,
    target_probability: float,
    temperature: Marginal | float,
    axial_acceleration: Marginal | float,
    loss_coefficient: Marginal | float,
    suction_specific_speed: Marginal | float,
    pump_speed: Marginal | float,
    density: float,
    volumetric_flow: float,
    column_height: float,
    line_diameter: float,
    pressure_cov: float = 0.03,
    vapour_model: VapourPressureModel = DEMONSTRATION_VAPOUR_MODEL,
    lower: float,
    upper: float,
) -> PumpDesign:
    """Find the smallest mean tank pressure with ``P(margin < 1) = target``.

    The tank pressure is lognormal with mean ``p`` and coefficient of
    variation ``pressure_cov``. A higher ``p`` raises ``NPSH_a`` and leaves
    ``NPSH_r`` unchanged, so the FORM index ``beta(p)`` is non-decreasing and
    ``beta(p) - beta_target`` is solved by Brent's method on
    ``[lower, upper]``. FORM must converge at the bracket ends and both must
    straddle the target (otherwise ``ValueError``/``RuntimeError``); the
    result's probability is the FORM value, not a simulation estimate.

    Args:
        target_probability: Target probability in ``(0, 0.5)``.
        temperature: ``T`` law or number (K).
        axial_acceleration: ``a`` law or number (m/s^2).
        loss_coefficient: Total loss coefficient law or number.
        suction_specific_speed: ``S*`` law or number.
        pump_speed: ``w`` law or number (rad/s).
        density: ``rho`` (kg/m^3).
        volumetric_flow: ``Q`` (m^3/s).
        column_height: ``h`` (m).
        line_diameter: ``D`` (m).
        pressure_cov: Coefficient of variation of the tank pressure.
        vapour_model: Vapour-pressure law.
        lower: Lower pressure bracket (Pa).
        upper: Upper pressure bracket (Pa).

    Returns:
        The design and its achieved FORM probability.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or the bracket does not
            straddle the target.
        RuntimeError: If a FORM search does not converge.

    Equation:
        [EQ-AERO-PUMP-009].

    References:
        [REF-HASOFER-LIND-1974], [REF-MELCHERS-BECK-2018].

    """
    p = require_finite_real(target_probability, name="target_probability")
    if not 0.0 < p < 0.5:
        raise ValueError("target_probability must lie in (0, 0.5)")
    cov = _positive(pressure_cov, "pressure_cov")
    lo = _positive(lower, "lower")
    hi = _positive(upper, "upper")
    if lo >= hi:
        raise ValueError("lower must be below upper")
    target_beta = -Normal(0.0, 1.0).quantile(p)
    sigma = math.sqrt(math.log1p(cov * cov))

    def form_at(mean_pressure: float) -> tuple[float, float]:
        """Return the FORM ``(beta, probability)`` at a mean tank pressure.

        Args:
            mean_pressure: Mean tank pressure in Pa.

        Returns:
            The reliability index and failure probability.

        Raises:
            RuntimeError: If FORM does not converge.

        References:
            [EQ-AERO-PUMP-009].

        """
        mu = math.log(mean_pressure) - 0.5 * sigma * sigma
        model = turbopump_cavitation_model(
            tank_pressure=LogNormal(mu, sigma),
            temperature=temperature,
            axial_acceleration=axial_acceleration,
            loss_coefficient=loss_coefficient,
            suction_specific_speed=suction_specific_speed,
            pump_speed=pump_speed,
            density=density,
            volumetric_flow=volumetric_flow,
            column_height=column_height,
            line_diameter=line_diameter,
            vapour_model=vapour_model,
        )
        result = form_hlrf(ModelLimitState(model, "margin", 1.0, "below"))
        if not result.converged:
            raise RuntimeError(f"FORM did not converge at p = {mean_pressure:g} Pa")
        return result.beta, result.probability

    def residual(mean_pressure: float) -> float:
        """Return ``beta(p) - beta_target``.

        Args:
            mean_pressure: Mean tank pressure in Pa.

        Returns:
            The index residual.

        References:
            [EQ-AERO-PUMP-009].

        """
        return form_at(mean_pressure)[0] - target_beta

    if residual(lo) * residual(hi) > 0.0:
        raise ValueError("the bracket [lower, upper] does not straddle the target")
    root = brent_root(residual, lo, hi, tolerance=1e-9 * hi)
    beta, probability = form_at(float(root.root))
    return PumpDesign(
        float(root.root), probability, beta, p, root.iterations, bool(root.converged)
    )
