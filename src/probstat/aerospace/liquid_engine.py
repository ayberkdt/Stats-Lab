"""Ideal-rocket performance of a liquid engine, with uncertainty propagation.

An engineering-level model of steady chamber/nozzle performance built on
ideal-rocket theory (Sutton and Biblarz, *Rocket Propulsion Elements*): frozen
composition, calorically perfect gas, quasi-one-dimensional isentropic flow,
uniform stagnation conditions in the chamber. All quantities are SI (Pa, K,
m, m^2, kg, kg/kmol for molecular weight, s, N).

Contents:

- :func:`characteristic_velocity`, :func:`exit_mach_from_area_ratio` (bracketed
  Newton with a bisection safeguard and an explicit convergence statement),
  :func:`exit_pressure_ratio`, :func:`thrust_coefficient` (with pressure term),
  :func:`ideal_rocket_performance` (thrust, ``Isp``, mass flow, exit pressure)
  and the flow-separation flag of the Summerfield criterion.
- Corrections: combustion (``c*``) efficiency ``eta_c*`` and the conical-nozzle
  divergence factor ``lambda = (1 + cos alpha) / 2``.
- :class:`MixtureRatioTable`: user-supplied ``O/F -> (Tc, gamma, M)`` data with
  linear or monotone cubic (PCHIP) interpolation and no extrapolation.
  :data:`DEMONSTRATION_MIXTURE_TABLE` is an *illustrative synthetic* table, not
  chemical-equilibrium output and not any real propellant combination.
- :func:`power_balance`: the simplest turbine-power = pump-power balance
  returning the required turbine flow fraction. Optional and heavily
  simplified.
- :func:`liquid_engine_model`: a :class:`~probstat.modeling.ProbabilisticModel`
  so that propagation, Sobol sensitivity, FORM and subset simulation apply
  unchanged.

Validity and limits. The model assumes frozen flow in a perfect gas, which
overestimates ``Isp`` relative to equilibrium chemistry only through the
supplied ``(Tc, gamma, M)``; it contains no heat loss, boundary-layer,
two-phase, finite-rate chemistry, or cycle losses beyond the stated
efficiency factors. Realistic ``gamma`` lies roughly in 1.1 to 1.4 and
expansion ratios in 2 to 400, but only ``gamma > 1`` and ``epsilon >= 1`` are
enforced. The cases in this library are synthetic and are not validated
against any engine.

Experimental (added in 0.28).

References:
    [REF-SUTTON-BIBLARZ-2017], [REF-GORDON-MCBRIDE-1994],
    [REF-FRITSCH-CARLSON-1980].

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "DEMONSTRATION_MIXTURE_TABLE",
    "G0",
    "SUMMERFIELD_RATIO",
    "UNIVERSAL_GAS_CONSTANT",
    "AreaMachSolution",
    "EnginePerformance",
    "MixtureRatioTable",
    "MixtureState",
    "PowerBalance",
    "PumpSpec",
    "area_ratio_from_mach",
    "characteristic_velocity",
    "divergence_factor",
    "exit_mach_from_area_ratio",
    "exit_pressure_ratio",
    "expansion_ratio_from_pressure_ratio",
    "ideal_rocket_performance",
    "limiting_thrust_coefficient",
    "liquid_engine_model",
    "power_balance",
    "pump_power",
    "separation_margin",
    "thrust_coefficient",
    "turbine_specific_work",
]

G0 = 9.80665
"""Standard gravity ``g0`` in m/s^2 (exact, conventional)."""

UNIVERSAL_GAS_CONSTANT = 8314.462618
"""Universal gas constant ``R_u`` in J/(kmol K) (SI 2019 exact value)."""

SUMMERFIELD_RATIO = 0.4
"""Summerfield separation criterion: separation when ``p_e < 0.4 p_a``."""


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
        [REF-SUTTON-BIBLARZ-2017]. No equation is evaluated.

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
        [REF-SUTTON-BIBLARZ-2017]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _gamma(value: object) -> float:
    """Return a validated ratio of specific heats ``gamma > 1``.

    Args:
        value: Candidate ratio of specific heats.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite and greater than one.

    References:
        [EQ-AERO-LRE-002], [REF-SUTTON-BIBLARZ-2017].

    """
    result = require_finite_real(value, name="gamma")
    if result <= 1.0:
        raise ValueError("gamma must be greater than 1")
    return result


def _fraction(value: object, name: str) -> float:
    """Return a real in ``(0, 1]``.

    Args:
        value: Candidate efficiency.
        name: Name used in error messages.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is outside ``(0, 1]``.

    References:
        [EQ-AERO-LRE-005], [REF-SUTTON-BIBLARZ-2017].

    """
    result = require_finite_real(value, name=name)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


def characteristic_velocity(
    gamma: float, molecular_weight: float, chamber_temperature: float
) -> float:
    """Return the ideal characteristic velocity ``c*`` in m/s.

    ``c* = sqrt(R T_c / gamma) / (2/(gamma+1))^((gamma+1) / (2 (gamma-1)))``
    with ``R = R_u / M`` for a perfect gas with stagnation temperature ``T_c``
    and molecular weight ``M`` (kg/kmol); equivalently ``c* = p_c A_t / mdot``
    of choked flow.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        molecular_weight: Mean molecular weight in kg/kmol, ``> 0``.
        chamber_temperature: Chamber stagnation temperature in K, ``> 0``.

    Returns:
        ``c*`` in m/s.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-LRE-001].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    weight = _positive(molecular_weight, "molecular_weight")
    temperature = _positive(chamber_temperature, "chamber_temperature")
    gas_constant = UNIVERSAL_GAS_CONSTANT / weight
    exponent = (g + 1.0) / (2.0 * (g - 1.0))
    root = math.sqrt(gas_constant * temperature / g)
    return root / float((2.0 / (g + 1.0)) ** exponent)


def area_ratio_from_mach(gamma: float, mach: float) -> float:
    """Return the isentropic area ratio ``A/A*`` at Mach number ``M``.

    ``A/A* = (1/M) [ (2/(gamma+1)) (1 + (gamma-1) M^2 / 2) ]^((gamma+1) /
    (2 (gamma-1)))``.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        mach: Mach number, ``> 0``.

    Returns:
        The area ratio, ``>= 1``, equal to one at ``M = 1``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-LRE-002].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    m = _positive(mach, "mach")
    return math.exp(_log_area_ratio(g, m))


def _log_area_ratio(g: float, m: float) -> float:
    """Return ``ln(A/A*)`` for a validated ``gamma`` and Mach number.

    Args:
        g: Ratio of specific heats.
        m: Mach number.

    Returns:
        The natural logarithm of the area ratio.

    References:
        [EQ-AERO-LRE-002].

    """
    p = (g + 1.0) / (2.0 * (g - 1.0))
    inner = 1.0 + 0.5 * (g - 1.0) * m * m
    return -math.log(m) + p * (math.log(2.0 / (g + 1.0)) + math.log(inner))


def _log_area_ratio_slope(g: float, m: float) -> float:
    """Return ``d ln(A/A*) / dM``.

    Args:
        g: Ratio of specific heats.
        m: Mach number.

    Returns:
        The derivative, which is positive for ``M > 1``.

    References:
        [EQ-AERO-LRE-002].

    """
    p = (g + 1.0) / (2.0 * (g - 1.0))
    return -1.0 / m + p * (g - 1.0) * m / (1.0 + 0.5 * (g - 1.0) * m * m)


@dataclass(frozen=True, slots=True)
class AreaMachSolution:
    """Supersonic root of the area-Mach relation.

    Args:
        mach: Exit Mach number ``M_e >= 1``.
        converged: Whether the residual and bracket tests were met.
        iterations: Newton or bisection steps taken.
        residual: ``ln(A/A*(M_e)) - ln(epsilon)`` at the returned Mach number.
        bisection_steps: Steps in which the Newton update left the bracket
            and bisection was used instead.
        stop_reason: ``"exact"`` (``epsilon = 1``), ``"residual"``,
            ``"bracket"``, or ``"max_iterations"``.

    References:
        [EQ-AERO-LRE-002], [REF-SUTTON-BIBLARZ-2017].

    """

    mach: float
    converged: bool
    iterations: int
    residual: float
    bisection_steps: int
    stop_reason: str


def exit_mach_from_area_ratio(
    gamma: float,
    expansion_ratio: float,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 200,
) -> AreaMachSolution:
    """Solve ``A/A*(M) = epsilon`` for the supersonic Mach number.

    The unknown is bracketed (``M = 1`` below, an upper end found by doubling)
    and ``F(M) = ln(A/A*) - ln(epsilon)``, which is strictly increasing for
    ``M > 1``, is solved by Newton's method. A Newton step that leaves the
    bracket is replaced by bisection, so the iteration cannot diverge. The
    result states whether it converged; it never returns a silent guess.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        expansion_ratio: Nozzle area ratio ``A_e/A_t >= 1``.
        tolerance: Absolute tolerance on ``|F|``; also used as a relative
            bracket width.
        max_iterations: Iteration cap.

    Returns:
        An :class:`AreaMachSolution`.

    Raises:
        TypeError: If an argument is not a real number or the cap is not an
            integer.
        ValueError: If ``expansion_ratio < 1`` or another argument is invalid.

    Equation:
        [EQ-AERO-LRE-002].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    ratio = require_finite_real(expansion_ratio, name="expansion_ratio")
    if ratio < 1.0:
        raise ValueError("expansion_ratio must be at least 1")
    tol = _positive(tolerance, "tolerance")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if ratio == 1.0:
        return AreaMachSolution(1.0, True, 0, 0.0, 0, "exact")
    target = math.log(ratio)
    low = 1.0
    high = 2.0
    while _log_area_ratio(g, high) < target:
        low = high
        high *= 2.0
        if high > 1e8:
            raise ValueError("expansion_ratio is too large for the Mach bracket")
    mach = high
    bisections = 0
    reason = "max_iterations"
    for step in range(1, max_iterations + 1):
        residual = _log_area_ratio(g, mach) - target
        if residual > 0.0:
            high = mach
        else:
            low = mach
        slope = _log_area_ratio_slope(g, mach)
        proposal = mach - residual / slope if slope > 0.0 else math.nan
        if not low < proposal < high:
            proposal = 0.5 * (low + high)
            bisections += 1
        mach = proposal
        residual = _log_area_ratio(g, mach) - target
        if abs(residual) <= tol:
            reason = "residual"
        elif high - low <= tol * mach:
            reason = "bracket"
        else:
            continue
        return AreaMachSolution(mach, True, step, residual, bisections, reason)
    residual = _log_area_ratio(g, mach) - target
    return AreaMachSolution(mach, False, max_iterations, residual, bisections, reason)


def exit_pressure_ratio(gamma: float, exit_mach: float) -> float:
    """Return the isentropic pressure ratio ``p_e / p_c``.

    ``p_e/p_c = (1 + (gamma-1) M_e^2 / 2)^(-gamma/(gamma-1))``.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        exit_mach: Exit Mach number, ``>= 0``.

    Returns:
        The static-to-chamber pressure ratio in ``(0, 1]``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-LRE-003].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    m = _non_negative(exit_mach, "exit_mach")
    return float((1.0 + 0.5 * (g - 1.0) * m * m) ** (-g / (g - 1.0)))


def expansion_ratio_from_pressure_ratio(gamma: float, pressure_ratio: float) -> float:
    """Return the area ratio whose exit pressure ratio is ``p_e / p_c``.

    Inverts :func:`exit_pressure_ratio` in closed form for the Mach number and
    then applies the area-Mach relation. Useful for the ideally expanded
    nozzle ``p_e = p_a``.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        pressure_ratio: ``p_e / p_c`` in ``(0, 1]``.

    Returns:
        The expansion ratio ``A_e / A_t >= 1``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-LRE-002], [EQ-AERO-LRE-003].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    ratio = _positive(pressure_ratio, "pressure_ratio")
    if ratio > 1.0:
        raise ValueError("pressure_ratio must not exceed 1")
    mach_squared = 2.0 / (g - 1.0) * (ratio ** (-(g - 1.0) / g) - 1.0)
    if mach_squared <= 0.0:
        return 1.0
    return area_ratio_from_mach(g, math.sqrt(mach_squared))


def divergence_factor(half_angle: float) -> float:
    """Return the conical-nozzle divergence factor ``lambda``.

    ``lambda = (1 + cos alpha) / 2`` is the ratio of the axial momentum of a
    conical exit flow of half-angle ``alpha`` to that of a one-dimensional
    flow.

    Args:
        half_angle: Cone half-angle ``alpha`` in radians, ``0 <= alpha <
            pi/2``.

    Returns:
        ``lambda`` in ``(0.5, 1]``.

    Raises:
        TypeError: If ``half_angle`` is not a real number.
        ValueError: If it is outside ``[0, pi/2)``.

    Equation:
        [EQ-AERO-LRE-006].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    alpha = require_finite_real(half_angle, name="half_angle")
    if not 0.0 <= alpha < 0.5 * math.pi:
        raise ValueError("half_angle must lie in [0, pi/2) radians")
    return 0.5 * (1.0 + math.cos(alpha))


def limiting_thrust_coefficient(gamma: float) -> float:
    """Return the vacuum thrust coefficient of an infinite expansion.

    ``CF_max = sqrt(2 gamma^2 / (gamma-1) (2/(gamma+1))^((gamma+1)/(gamma-1)))``,
    the upper bound of the vacuum ``CF`` of the ideal rocket.

    Args:
        gamma: Ratio of specific heats, ``> 1``.

    Returns:
        The limiting momentum thrust coefficient.

    Raises:
        TypeError: If ``gamma`` is not a real number.
        ValueError: If ``gamma <= 1``.

    Equation:
        [EQ-AERO-LRE-004].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    return math.sqrt(
        2.0 * g * g / (g - 1.0) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
    )


def _momentum_coefficient(g: float, ratio: float) -> float:
    """Return the momentum part of ``CF`` for exit pressure ratio ``ratio``.

    Args:
        g: Ratio of specific heats.
        ratio: ``p_e / p_c``.

    Returns:
        ``sqrt(2 g^2/(g-1) (2/(g+1))^((g+1)/(g-1)) (1 - ratio^((g-1)/g)))``.

    References:
        [EQ-AERO-LRE-004].

    """
    limit = limiting_thrust_coefficient(g)
    return limit * math.sqrt(max(0.0, 1.0 - ratio ** ((g - 1.0) / g)))


def thrust_coefficient(
    gamma: float,
    expansion_ratio: float,
    chamber_pressure: float,
    ambient_pressure: float,
    *,
    divergence: float = 1.0,
) -> float:
    """Return the thrust coefficient ``CF`` including the pressure term.

    ``CF = lambda sqrt(2 g^2/(g-1) (2/(g+1))^((g+1)/(g-1)) (1 - (p_e/p_c)^
    ((g-1)/g))) + (p_e - p_a) epsilon / p_c`` with ``p_e/p_c`` from the
    supersonic Mach number of the area ratio. ``ambient_pressure = 0`` gives
    the vacuum value. The divergence factor multiplies only the momentum term.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        expansion_ratio: ``A_e / A_t >= 1``.
        chamber_pressure: ``p_c`` in Pa, ``> 0``.
        ambient_pressure: ``p_a`` in Pa, ``>= 0``.
        divergence: Divergence factor ``lambda`` in ``(0, 1]``.

    Returns:
        ``CF`` (dimensionless).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain or the area-Mach
            solution did not converge.

    Equation:
        [EQ-AERO-LRE-004].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    pc = _positive(chamber_pressure, "chamber_pressure")
    pa = _non_negative(ambient_pressure, "ambient_pressure")
    lam = _fraction(divergence, "divergence")
    solution = exit_mach_from_area_ratio(g, expansion_ratio)
    if not solution.converged:
        raise ValueError("area-Mach relation did not converge")
    ratio = exit_pressure_ratio(g, solution.mach)
    return lam * _momentum_coefficient(g, ratio) + (ratio - pa / pc) * float(
        expansion_ratio
    )


def separation_margin(exit_pressure: float, ambient_pressure: float) -> float:
    """Return the Summerfield separation margin ``p_e - 0.4 p_a`` in Pa.

    Free-shock flow separation in an over-expanded nozzle is predicted by the
    Summerfield criterion when ``p_e < 0.4 p_a``. A positive margin means no
    separation is predicted; it is finite in vacuum (``p_a = 0``). The
    criterion is an empirical rule of thumb for conical or contoured nozzles,
    not a flow-separation solver, and it ignores nozzle contour, wall
    temperature and transients.

    Args:
        exit_pressure: Nozzle-exit static pressure ``p_e`` in Pa, ``>= 0``.
        ambient_pressure: ``p_a`` in Pa, ``>= 0``.

    Returns:
        ``p_e - 0.4 p_a`` in Pa.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is negative or not finite.

    Equation:
        [EQ-AERO-LRE-007].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    pe = _non_negative(exit_pressure, "exit_pressure")
    pa = _non_negative(ambient_pressure, "ambient_pressure")
    return pe - SUMMERFIELD_RATIO * pa


@dataclass(frozen=True, slots=True)
class EnginePerformance:
    """Steady performance of an ideal rocket with efficiency corrections.

    Args:
        characteristic_velocity: Delivered ``c*`` (``eta_c*`` applied), m/s.
        ideal_characteristic_velocity: Ideal ``c*`` before ``eta_c*``, m/s.
        exit_mach: Exit Mach number.
        exit_pressure: Nozzle-exit static pressure, Pa.
        thrust_coefficient: ``CF`` at the given ambient pressure.
        vacuum_thrust_coefficient: ``CF`` at ``p_a = 0``.
        thrust: ``F = CF p_c A_t`` at the ambient pressure, N.
        vacuum_thrust: Thrust at ``p_a = 0``, N.
        mass_flow: ``mdot = p_c A_t / c*``, kg/s.
        specific_impulse: ``Isp = CF c* / g0`` at the ambient pressure, s.
        vacuum_specific_impulse: ``Isp`` at ``p_a = 0``, s.
        divergence_factor: ``lambda`` used.
        separation_margin: ``p_e - 0.4 p_a`` in Pa (positive: no separation).
        separation_predicted: Whether ``p_e < 0.4 p_a``.
        mach_converged: Whether the area-Mach solve converged.
        mach_iterations: Iterations of the area-Mach solve.

    References:
        [EQ-AERO-LRE-001] to [EQ-AERO-LRE-007], [REF-SUTTON-BIBLARZ-2017].

    """

    characteristic_velocity: float
    ideal_characteristic_velocity: float
    exit_mach: float
    exit_pressure: float
    thrust_coefficient: float
    vacuum_thrust_coefficient: float
    thrust: float
    vacuum_thrust: float
    mass_flow: float
    specific_impulse: float
    vacuum_specific_impulse: float
    divergence_factor: float
    separation_margin: float
    separation_predicted: bool
    mach_converged: bool
    mach_iterations: int


def ideal_rocket_performance(
    *,
    gamma: float,
    molecular_weight: float,
    chamber_temperature: float,
    chamber_pressure: float,
    throat_area: float,
    expansion_ratio: float,
    ambient_pressure: float = 0.0,
    combustion_efficiency: float = 1.0,
    divergence_half_angle: float = 0.0,
) -> EnginePerformance:
    """Compute ideal-rocket performance with ``eta_c*`` and divergence losses.

    Steps: ``c*_ideal`` (EQ-AERO-LRE-001); supersonic exit Mach of the area
    ratio (EQ-AERO-LRE-002); ``p_e`` (EQ-AERO-LRE-003); ``CF`` with the
    divergence factor on the momentum term (EQ-AERO-LRE-004, -006);
    ``c* = eta_c* c*_ideal``, ``mdot = p_c A_t / c*``, ``F = CF p_c A_t``,
    ``Isp = F / (mdot g0) = CF c* / g0`` (EQ-AERO-LRE-005); Summerfield flag
    (EQ-AERO-LRE-007). Pressure changes in the chamber with ``eta_c*`` are not
    modelled; ``p_c`` is an input.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        molecular_weight: kg/kmol, ``> 0``.
        chamber_temperature: ``T_c`` in K, ``> 0``.
        chamber_pressure: ``p_c`` in Pa, ``> 0``.
        throat_area: ``A_t`` in m^2, ``> 0``.
        expansion_ratio: ``A_e / A_t >= 1``.
        ambient_pressure: ``p_a`` in Pa, ``>= 0`` (0 is vacuum).
        combustion_efficiency: ``eta_c*`` in ``(0, 1]``.
        divergence_half_angle: Conical half-angle in rad, ``[0, pi/2)``.

    Returns:
        An :class:`EnginePerformance`.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.
        RuntimeError: If the area-Mach solve did not converge.

    Equation:
        [EQ-AERO-LRE-001] to [EQ-AERO-LRE-007].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = _gamma(gamma)
    pc = _positive(chamber_pressure, "chamber_pressure")
    area = _positive(throat_area, "throat_area")
    pa = _non_negative(ambient_pressure, "ambient_pressure")
    eta = _fraction(combustion_efficiency, "combustion_efficiency")
    lam = divergence_factor(divergence_half_angle)
    ideal = characteristic_velocity(g, molecular_weight, chamber_temperature)
    solution = exit_mach_from_area_ratio(g, expansion_ratio)
    if not solution.converged:
        raise RuntimeError("area-Mach relation did not converge")
    eps = float(expansion_ratio)
    ratio = exit_pressure_ratio(g, solution.mach)
    pe = ratio * pc
    momentum = lam * _momentum_coefficient(g, ratio)
    cf = momentum + (ratio - pa / pc) * eps
    cf_vacuum = momentum + ratio * eps
    cstar = eta * ideal
    margin = separation_margin(pe, pa)
    return EnginePerformance(
        characteristic_velocity=cstar,
        ideal_characteristic_velocity=ideal,
        exit_mach=solution.mach,
        exit_pressure=pe,
        thrust_coefficient=cf,
        vacuum_thrust_coefficient=cf_vacuum,
        thrust=cf * pc * area,
        vacuum_thrust=cf_vacuum * pc * area,
        mass_flow=pc * area / cstar,
        specific_impulse=cf * cstar / G0,
        vacuum_specific_impulse=cf_vacuum * cstar / G0,
        divergence_factor=lam,
        separation_margin=margin,
        separation_predicted=pe < SUMMERFIELD_RATIO * pa,
        mach_converged=solution.converged,
        mach_iterations=solution.iterations,
    )


# ---------------------------------------------------------------------------
# Mixture-ratio tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MixtureState:
    """Chamber gas state at one mixture ratio.

    Args:
        mixture_ratio: Oxidizer-to-fuel mass ratio ``O/F``.
        chamber_temperature: ``T_c`` in K.
        gamma: Ratio of specific heats.
        molecular_weight: Mean molecular weight in kg/kmol.

    References:
        [EQ-AERO-LRE-008], [REF-SUTTON-BIBLARZ-2017].

    """

    mixture_ratio: float
    chamber_temperature: float
    gamma: float
    molecular_weight: float


def _pchip_slopes(x: tuple[float, ...], y: tuple[float, ...]) -> tuple[float, ...]:
    """Return Fritsch-Carlson shape-preserving node slopes.

    Args:
        x: Strictly increasing nodes.
        y: Node values.

    Returns:
        Hermite slopes ``d_k`` that keep the interpolant monotone on every
        interval where the data are monotone.

    References:
        [EQ-AERO-LRE-008], [REF-FRITSCH-CARLSON-1980].

    """
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return (delta[0], delta[0])
    slopes = [0.0] * n
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def edge(h0: float, h1: float, d0: float, d1: float) -> float:
        """Return a one-sided three-point slope limited for monotonicity.

        Args:
            h0: Nearest interval width.
            h1: Next interval width.
            d0: Nearest secant slope.
            d1: Next secant slope.

        Returns:
            The end slope.

        References:
            [REF-FRITSCH-CARLSON-1980].

        """
        value = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if value * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(value) > 3.0 * abs(d0):
            return 3.0 * d0
        return value

    slopes[0] = edge(h[0], h[1], delta[0], delta[1])
    slopes[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    return tuple(slopes)


@dataclass(frozen=True, slots=True)
class MixtureRatioTable:
    """User-supplied table ``O/F -> (T_c, gamma, M)`` with interpolation.

    The table is data, not physics: it should come from a chemical-equilibrium
    code such as CEA ([REF-GORDON-MCBRIDE-1994]) for the propellant, chamber
    pressure and expansion assumptions of the study. Evaluation outside
    ``[O/F_min, O/F_max]`` raises; there is no extrapolation. ``"linear"`` is
    piecewise linear; ``"pchip"`` is the monotone piecewise cubic Hermite
    interpolant of Fritsch and Carlson, which does not overshoot the data
    (useful near the temperature maximum).

    Args:
        mixture_ratio: Strictly increasing ``O/F`` nodes (at least two).
        chamber_temperature: ``T_c`` in K at each node, ``> 0``.
        gamma: Ratio of specific heats at each node, ``> 1``.
        molecular_weight: kg/kmol at each node, ``> 0``.
        method: ``"linear"`` or ``"pchip"``.
        label: Free-text provenance of the data.

    Raises:
        TypeError: If a value is not real or the method is not a string.
        ValueError: If lengths differ, nodes are not increasing, or a value
            is outside its domain.

    References:
        [EQ-AERO-LRE-008], [REF-FRITSCH-CARLSON-1980],
        [REF-GORDON-MCBRIDE-1994].

    """

    mixture_ratio: tuple[float, ...]
    chamber_temperature: tuple[float, ...]
    gamma: tuple[float, ...]
    molecular_weight: tuple[float, ...]
    method: Literal["linear", "pchip"] = "pchip"
    label: str = "user table"

    def __post_init__(self) -> None:
        """Validate and normalize the table.

        Raises:
            TypeError: If entries are not real numbers.
            ValueError: If the table is too short, columns differ in length,
                nodes do not increase, a value is outside its domain, or the
                method is unknown.

        Equation:
            [EQ-AERO-LRE-008].

        References:
            [REF-FRITSCH-CARLSON-1980].

        """
        columns = {
            "mixture_ratio": self.mixture_ratio,
            "chamber_temperature": self.chamber_temperature,
            "gamma": self.gamma,
            "molecular_weight": self.molecular_weight,
        }
        sizes = {len(tuple(v)) for v in columns.values()}
        if len(sizes) != 1:
            raise ValueError("table columns must have equal length")
        if next(iter(sizes)) < 2:
            raise ValueError("table needs at least two nodes")
        if self.method not in ("linear", "pchip"):
            raise ValueError("method must be 'linear' or 'pchip'")
        checks = {
            "mixture_ratio": _positive,
            "chamber_temperature": _positive,
            "gamma": lambda v, n: _gamma(v),
            "molecular_weight": _positive,
        }
        for name, column in columns.items():
            clean = tuple(checks[name](v, name) for v in column)
            object.__setattr__(self, name, clean)
        nodes = self.mixture_ratio
        if any(b <= a for a, b in zip(nodes, nodes[1:], strict=False)):
            raise ValueError("mixture_ratio must be strictly increasing")

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the supported ``(O/F_min, O/F_max)``.

        References:
            [EQ-AERO-LRE-008]. No further equation is evaluated.

        """
        return (self.mixture_ratio[0], self.mixture_ratio[-1])

    def _interpolate(self, values: tuple[float, ...], of: float) -> float:
        """Interpolate one column at ``of`` (already range-checked).

        Args:
            values: Column values at the nodes.
            of: Mixture ratio inside the table range.

        Returns:
            The interpolated value.

        References:
            [EQ-AERO-LRE-008], [REF-FRITSCH-CARLSON-1980].

        """
        x = self.mixture_ratio
        lo, hi = 0, len(x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x[mid] <= of:
                lo = mid
            else:
                hi = mid
        h = x[hi] - x[lo]
        t = (of - x[lo]) / h
        if self.method == "linear":
            return values[lo] + t * (values[hi] - values[lo])
        d = _pchip_slopes(x, values)
        t2, t3 = t * t, t * t * t
        return (
            (2 * t3 - 3 * t2 + 1) * values[lo]
            + (t3 - 2 * t2 + t) * h * d[lo]
            + (-2 * t3 + 3 * t2) * values[hi]
            + (t3 - t2) * h * d[hi]
        )

    def at(self, mixture_ratio: float) -> MixtureState:
        """Return the interpolated chamber state at ``O/F``.

        Args:
            mixture_ratio: Mixture ratio inside the table range.

        Returns:
            A :class:`MixtureState`.

        Raises:
            TypeError: If ``mixture_ratio`` is not a real number.
            ValueError: If it lies outside the table range.

        Equation:
            [EQ-AERO-LRE-008].

        References:
            [REF-FRITSCH-CARLSON-1980].

        """
        of = require_finite_real(mixture_ratio, name="mixture_ratio")
        low, high = self.bounds
        if not low <= of <= high:
            raise ValueError(
                f"mixture_ratio {of:g} is outside the table range [{low:g}, {high:g}]"
            )
        return MixtureState(
            of,
            self._interpolate(self.chamber_temperature, of),
            self._interpolate(self.gamma, of),
            self._interpolate(self.molecular_weight, of),
        )


DEMONSTRATION_MIXTURE_TABLE = MixtureRatioTable(
    mixture_ratio=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    chamber_temperature=(2400.0, 2800.0, 3150.0, 3350.0, 3400.0, 3330.0, 3200.0),
    gamma=(1.24, 1.23, 1.22, 1.21, 1.20, 1.20, 1.19),
    molecular_weight=(9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0),
    method="pchip",
    label=(
        "ILLUSTRATIVE SYNTHETIC DEMONSTRATION TABLE: invented numbers with a "
        "qualitatively plausible shape (temperature peak, molecular weight "
        "rising with O/F). Not CEA output; does not represent any real "
        "propellant combination."
    ),
)
"""Synthetic table for examples and tests only; see its ``label``."""


# ---------------------------------------------------------------------------
# Power balance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PumpSpec:
    """One propellant pump of the power balance.

    Args:
        mass_flow: Pumped mass flow in kg/s, ``> 0``.
        density: Liquid density in kg/m^3, ``> 0``.
        pressure_rise: Discharge minus inlet pressure in Pa, ``> 0``
            (head ``H = dp / (rho g0)``).
        efficiency: Overall pump efficiency in ``(0, 1]`` (mechanical
            losses folded in).

    References:
        [EQ-AERO-LRE-009], [REF-SUTTON-BIBLARZ-2017].

    """

    mass_flow: float
    density: float
    pressure_rise: float
    efficiency: float


def pump_power(pump: PumpSpec) -> float:
    """Return the shaft power ``P = mdot dp / (rho eta_p)`` of one pump in W.

    Args:
        pump: Pump specification.

    Returns:
        Shaft power in W.

    Raises:
        TypeError: If ``pump`` is not a :class:`PumpSpec` or a field is not
            real.
        ValueError: If a field is outside its domain.

    Equation:
        [EQ-AERO-LRE-009].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(pump, PumpSpec):
        raise TypeError("pump must be a PumpSpec")
    flow = _positive(pump.mass_flow, "mass_flow")
    density = _positive(pump.density, "density")
    rise = _positive(pump.pressure_rise, "pressure_rise")
    eta = _fraction(pump.efficiency, "efficiency")
    return flow * rise / (density * eta)


def turbine_specific_work(
    *,
    specific_heat: float,
    inlet_temperature: float,
    pressure_ratio: float,
    gamma: float,
    efficiency: float,
) -> float:
    """Return the turbine shaft work per unit gas mass in J/kg.

    ``w_t = eta_t c_p T_in (1 - PR^(-(gamma-1)/gamma))`` with ``PR`` the
    inlet-to-outlet total pressure ratio of the turbine gas.

    Args:
        specific_heat: ``c_p`` of the turbine gas in J/(kg K), ``> 0``.
        inlet_temperature: Turbine inlet total temperature in K, ``> 0``.
        pressure_ratio: ``PR > 1``.
        gamma: Ratio of specific heats of the turbine gas, ``> 1``.
        efficiency: Turbine efficiency in ``(0, 1]``.

    Returns:
        Specific work in J/kg.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-LRE-009].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    cp = _positive(specific_heat, "specific_heat")
    temperature = _positive(inlet_temperature, "inlet_temperature")
    ratio = require_finite_real(pressure_ratio, name="pressure_ratio")
    if ratio <= 1.0:
        raise ValueError("pressure_ratio must exceed 1")
    g = _gamma(gamma)
    eta = _fraction(efficiency, "efficiency")
    return eta * cp * temperature * (1.0 - float(ratio ** (-(g - 1.0) / g)))


@dataclass(frozen=True, slots=True)
class PowerBalance:
    """Result of the simplified turbine-pump power balance.

    Args:
        pump_power: Total pump shaft power in W.
        turbine_specific_work: Turbine work per kg of turbine gas, J/kg.
        turbine_mass_flow: Gas flow that makes turbine power equal pump
            power, kg/s.
        flow_fraction: ``turbine_mass_flow / engine_mass_flow``.
        feasible: Whether ``flow_fraction < 1`` (a necessary sanity check
            only).

    References:
        [EQ-AERO-LRE-009], [REF-SUTTON-BIBLARZ-2017].

    """

    pump_power: float
    turbine_specific_work: float
    turbine_mass_flow: float
    flow_fraction: float
    feasible: bool


def power_balance(
    pumps: Sequence[PumpSpec],
    engine_mass_flow: float,
    *,
    specific_heat: float,
    inlet_temperature: float,
    pressure_ratio: float,
    gamma: float,
    turbine_efficiency: float,
) -> PowerBalance:
    """Solve ``P_turbine = sum P_pump`` for the required turbine gas flow.

    This is the simplest level of a gas-generator or expander power balance:
    one turbine, perfect-gas expansion, shaft power equal to pump power, no
    bearing or seal losses beyond the efficiencies, no cycle closure (the
    turbine gas flow is not fed back into thrust or ``Isp``). The flow
    fraction relates the turbine flow to the engine total propellant flow.

    Args:
        pumps: Non-empty sequence of pumps.
        engine_mass_flow: Engine total propellant flow in kg/s, ``> 0``.
        specific_heat: Turbine gas ``c_p`` in J/(kg K).
        inlet_temperature: Turbine inlet temperature in K.
        pressure_ratio: Turbine pressure ratio ``> 1``.
        gamma: Turbine gas ratio of specific heats.
        turbine_efficiency: Turbine efficiency in ``(0, 1]``.

    Returns:
        A :class:`PowerBalance`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``pumps`` is empty or a value is outside its domain.

    Equation:
        [EQ-AERO-LRE-009].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    pumps = tuple(pumps)
    if not pumps:
        raise ValueError("pumps must not be empty")
    total = sum(pump_power(p) for p in pumps)
    flow = _positive(engine_mass_flow, "engine_mass_flow")
    work = turbine_specific_work(
        specific_heat=specific_heat,
        inlet_temperature=inlet_temperature,
        pressure_ratio=pressure_ratio,
        gamma=gamma,
        efficiency=turbine_efficiency,
    )
    turbine_flow = total / work
    fraction = turbine_flow / flow
    return PowerBalance(total, work, turbine_flow, fraction, fraction < 1.0)


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "combustion_efficiency": 1.0,
    "ambient_pressure": 101325.0,
    "divergence_half_angle": 0.0,
}

_UNITS = {
    "chamber_pressure": "Pa",
    "mixture_ratio": "-",
    "combustion_efficiency": "-",
    "throat_diameter": "m",
    "expansion_ratio": "-",
    "ambient_pressure": "Pa",
    "divergence_half_angle": "rad",
}

_QUANTITIES = (
    QuantityOfInterest("thrust", "Thrust at the ambient pressure", "N"),
    QuantityOfInterest(
        "specific_impulse", "Specific impulse at the ambient pressure", "s"
    ),
    QuantityOfInterest("vacuum_specific_impulse", "Vacuum specific impulse", "s"),
    QuantityOfInterest("mass_flow", "Propellant mass flow", "kg/s"),
    QuantityOfInterest("exit_pressure", "Nozzle-exit static pressure", "Pa"),
    QuantityOfInterest(
        "separation_margin",
        "Summerfield margin p_e - 0.4 p_a (positive: no separation predicted)",
        "Pa",
    ),
    QuantityOfInterest("characteristic_velocity", "Delivered c*", "m/s"),
)


@dataclass(frozen=True, slots=True)
class _EngineForward:
    """Picklable forward map of :func:`liquid_engine_model`.

    Args:
        table: Mixture-ratio table supplying ``(T_c, gamma, M)``.

    References:
        [EQ-AERO-LRE-001] to [EQ-AERO-LRE-008].

    """

    table: MixtureRatioTable

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the QoIs for one realization of the named inputs.

        Args:
            parameters: Values for chamber_pressure, mixture_ratio,
                combustion_efficiency, throat_diameter, expansion_ratio,
                ambient_pressure and divergence_half_angle.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If an input is outside its domain, for example a
                mixture ratio outside the table range.
            RuntimeError: If the area-Mach solve did not converge.

        Equation:
            [EQ-AERO-LRE-001] to [EQ-AERO-LRE-008].

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        state = self.table.at(parameters["mixture_ratio"])
        diameter = _positive(parameters["throat_diameter"], "throat_diameter")
        result = ideal_rocket_performance(
            gamma=state.gamma,
            molecular_weight=state.molecular_weight,
            chamber_temperature=state.chamber_temperature,
            chamber_pressure=parameters["chamber_pressure"],
            throat_area=0.25 * math.pi * diameter * diameter,
            expansion_ratio=parameters["expansion_ratio"],
            ambient_pressure=parameters["ambient_pressure"],
            combustion_efficiency=parameters["combustion_efficiency"],
            divergence_half_angle=parameters["divergence_half_angle"],
        )
        return {
            "thrust": result.thrust,
            "specific_impulse": result.specific_impulse,
            "vacuum_specific_impulse": result.vacuum_specific_impulse,
            "mass_flow": result.mass_flow,
            "exit_pressure": result.exit_pressure,
            "separation_margin": result.separation_margin,
            "characteristic_velocity": result.characteristic_velocity,
        }


def liquid_engine_model(
    *,
    chamber_pressure: Marginal | float,
    mixture_ratio: Marginal | float,
    throat_diameter: Marginal | float,
    expansion_ratio: Marginal | float,
    combustion_efficiency: Marginal | float = _DEFAULTS["combustion_efficiency"],
    ambient_pressure: Marginal | float = _DEFAULTS["ambient_pressure"],
    divergence_half_angle: Marginal | float = _DEFAULTS["divergence_half_angle"],
    table: MixtureRatioTable = DEMONSTRATION_MIXTURE_TABLE,
    name: str = "liquid-engine",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic liquid-engine performance model.

    Each argument is either a distribution (an uncertain input, wrapped as an
    independent :class:`~probstat.modeling.RandomInput`) or a plain number (a
    fixed parameter). Units: ``chamber_pressure`` and ``ambient_pressure`` in
    Pa, ``throat_diameter`` in m, ``divergence_half_angle`` in rad,
    ``mixture_ratio`` and efficiencies and ``expansion_ratio`` dimensionless.
    ``(T_c, gamma, M)`` come from ``table`` at the sampled mixture ratio, so
    a sample outside the table range fails the run (the failure policy
    ``on_failure`` decides what happens; use bounded laws to avoid it).
    Quantities of interest: ``thrust``, ``specific_impulse``,
    ``vacuum_specific_impulse``, ``mass_flow``, ``exit_pressure``,
    ``separation_margin`` and ``characteristic_velocity``. The returned model
    works unchanged with the library's propagation, Sobol, FORM and subset
    simulation tools. Independence between inputs is assumed.

    Args:
        chamber_pressure: ``p_c`` (Pa).
        mixture_ratio: ``O/F``.
        throat_diameter: ``d_t`` (m); ``A_t = pi d_t^2 / 4``.
        expansion_ratio: ``A_e / A_t``.
        combustion_efficiency: ``eta_c*`` (default 1).
        ambient_pressure: ``p_a`` (Pa, default sea level 101325).
        divergence_half_angle: Cone half-angle (rad, default 0).
        table: Mixture-ratio table; the default is the synthetic
            demonstration table.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is outside its domain or no input is
            random.

    Equation:
        [EQ-AERO-LRE-010].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(table, MixtureRatioTable):
        raise TypeError("table must be a MixtureRatioTable")
    given: dict[str, Marginal | float] = {
        "chamber_pressure": chamber_pressure,
        "mixture_ratio": mixture_ratio,
        "combustion_efficiency": combustion_efficiency,
        "throat_diameter": throat_diameter,
        "expansion_ratio": expansion_ratio,
        "ambient_pressure": ambient_pressure,
        "divergence_half_angle": divergence_half_angle,
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, value in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = require_finite_real(value, name=key)
        else:
            random_inputs.append(RandomInput(key, value, unit=_UNITS[key]))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        _EngineForward(table),
        _QUANTITIES,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )
