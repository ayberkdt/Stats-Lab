"""Lumped (0D) internal ballistics of a solid rocket motor, with uncertainty.

The motor is one control volume of combustion gas fed by a burning grain and
drained through a choked nozzle throat. Burning follows the Saint-Robert /
Vieille law ``r = a (p / p_ref)^n`` with an exponential propellant-temperature
sensitivity, the chamber pressure is either the quasi-steady equilibrium of
gas generation and discharge or the solution of the transient gas
mass balance, and the grain geometries (tubular, end burner, BATES) give the
burning area as an exact function of the web burned. The thrust uses the ideal
thrust coefficient of a fixed-area-ratio nozzle. :func:`solid_motor_model`
wraps the whole chain as a :class:`~probstat.modeling.ProbabilisticModel`
with uncertain propellant, grain-temperature, throat, erosion and case inputs.

Main entry points:
    ``simulate_solid_motor``: pressure-time and thrust-time traces, MEOP,
        total impulse, burn time and average thrust of one motor.
    ``solid_motor_model``: probabilistic model with the quantities of
        interest ``meop``, ``total_impulse``, ``burn_time`` and
        ``burst_margin``.
    ``TubularGrain``, ``EndBurningGrain``, ``BatesGrain``: grain geometries
        with exact burning area, propellant volume and web.
    ``SolidMotorParameters``: propellant, nozzle and case data of one motor.
    ``equilibrium_pressure``, ``burn_rate``, ``klemmung``,
        ``ideal_thrust_coefficient``: the closed-form building blocks.

Units: SI throughout (m, m2, m3, s, kg, kg/m3, Pa, K, N, N s); the
temperature sensitivity ``sigma_p`` is in 1/K.

Assumptions and validity:
    - One lumped chamber with a uniform pressure; the gas is calorically
      perfect with constant ``gamma`` and constant characteristic velocity
      ``c*`` (so the flame temperature is constant), the products leave the
      surface at the flame temperature, and the nozzle is choked at all times
      (an igniter, erosive burning, a pressure-dependent ``c*``, slag,
      two-phase losses, grain deformation and pressure-coupled instabilities
      are not modelled).
    - ``0 <= n < 1``: the equilibrium pressure is proportional to
      ``Kn^(1 / (1 - n))`` and does not exist for ``n >= 1``; such inputs are
      refused. Typical composite propellants have ``0.2 <= n <= 0.5``.
    - The quasi-steady mode needs the chamber filling time
      ``tau = V c* / (R T_c A_t)`` to be short compared with the burn time; it
      has no start-up and no tail-off (the thrust ends abruptly at web
      burnout). The transient mode integrates the gas mass balance and adds
      both, but it does not model the igniter: the chamber starts at
      ``initial_pressure``.
    - Thrust is the ideal ``F = C_F p A_t`` of a nozzle with constant exit
      area (so the area ratio shrinks as the throat erodes) and no flow
      separation; with a non-zero ambient pressure the start-up thrust of the
      transient mode can be slightly negative and is integrated as computed.
    - The demonstration cases are synthetic and are not data from any real
      motor.

Experimental (added in 0.28).

Guide: docs/foundations/aerospace-propulsion-uq.md

References:
    [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-001], [EQ-AERO-SRM-002].

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

from probstat.distributions import LogNormal, Normal, TruncatedNormal, Uniform
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.numerics.ode import Event, Rhs, integrate_fixed_step, integrate_to_event
from probstat.optimization import brent_root
from probstat.uq import QuantityOfInterest

__all__ = [
    "STANDARD_GRAVITY",
    "BatesGrain",
    "EndBurningGrain",
    "GrainGeometry",
    "SolidMotorParameters",
    "SolidMotorResult",
    "TubularGrain",
    "burn_rate",
    "burn_rate_coefficient",
    "chamber_rt",
    "equilibrium_pressure",
    "exit_mach_number",
    "exit_pressure_ratio",
    "ideal_thrust_coefficient",
    "klemmung",
    "simulate_solid_motor",
    "solid_motor_model",
]

STANDARD_GRAVITY = 9.80665
"""Standard gravity ``g_0`` in m/s2, used only for the specific impulse."""

_MODES = ("quasi_steady", "transient")
# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive float.

    Args:
        value: Candidate real number.
        name: Parameter name for the message.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite or not positive.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {number!r}")
    return number


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative float.

    Args:
        value: Candidate real number.
        name: Parameter name for the message.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite or is negative.

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative, got {number!r}")
    return number


def _require_exponent(exponent: object) -> float:
    """Return a burn-rate exponent in ``[0, 1)`` or refuse it.

    Args:
        exponent: Candidate pressure exponent ``n``.

    Returns:
        The exponent as a float.

    Raises:
        TypeError: If ``exponent`` is not a real number.
        ValueError: If ``exponent`` is not finite or lies outside ``[0, 1)``.

    Equation:
        [EQ-AERO-SRM-002].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-002].

    """
    n = require_finite_real(exponent, name="pressure_exponent")
    if not 0.0 <= n < 1.0:
        raise ValueError(
            "pressure_exponent must satisfy 0 <= n < 1 (the equilibrium "
            f"pressure (...)^(1/(1-n)) does not exist for n >= 1), got {n!r}"
        )
    return n


def _require_gamma(gamma: object) -> float:
    """Return a specific-heat ratio ``gamma > 1`` or refuse it.

    Args:
        gamma: Candidate specific-heat ratio.

    Returns:
        The ratio as a float.

    Raises:
        TypeError: If ``gamma`` is not a real number.
        ValueError: If ``gamma`` is not finite or ``gamma <= 1``.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

    """
    value = require_finite_real(gamma, name="gamma")
    if value <= 1.0:
        raise ValueError(f"gamma must exceed 1, got {value!r}")
    return value


# --------------------------------------------------------------------------
# closed-form ballistics
# --------------------------------------------------------------------------


def burn_rate_coefficient(
    coefficient: float,
    temperature_sensitivity: float,
    temperature: float,
    reference_temperature: float,
) -> float:
    """Return the temperature-corrected burn-rate coefficient ``a(T)``.

    ``a(T) = a_0 exp(sigma_p (T - T_ref))`` where ``sigma_p`` is the temperature
    sensitivity of the burn rate at constant pressure (1/K).

    Args:
        coefficient: ``a_0``, burn rate in m/s at ``p_ref`` and ``T_ref``.
        temperature_sensitivity: ``sigma_p`` in 1/K, non-negative.
        temperature: Initial grain temperature ``T`` in K, positive.
        reference_temperature: ``T_ref`` in K, positive.

    Returns:
        ``a(T)`` in m/s.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range or the result overflows.

    Equation:
        [EQ-AERO-SRM-001].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-001].

    """
    a0 = _positive(coefficient, "coefficient")
    sigma = _non_negative(temperature_sensitivity, "temperature_sensitivity")
    t = _positive(temperature, "temperature")
    t_ref = _positive(reference_temperature, "reference_temperature")
    exponent = sigma * (t - t_ref)
    if exponent > 700.0:
        raise ValueError("temperature correction overflows")
    return a0 * math.exp(exponent)


def burn_rate(
    pressure: float,
    coefficient: float,
    exponent: float,
    *,
    reference_pressure: float = 1.0e6,
) -> float:
    """Return the Saint-Robert / Vieille burn rate ``r = a (p / p_ref)^n``.

    Args:
        pressure: Chamber pressure ``p`` in Pa, non-negative.
        coefficient: ``a`` in m/s at ``p_ref`` (already temperature corrected,
            see :func:`burn_rate_coefficient`).
        exponent: Pressure exponent ``n`` with ``0 <= n < 1``.
        reference_pressure: ``p_ref`` in Pa.

    Returns:
        Regression rate of the burning surface normal to itself, in m/s.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-SRM-001].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-001].

    """
    p = _non_negative(pressure, "pressure")
    a = _positive(coefficient, "coefficient")
    n = _require_exponent(exponent)
    p_ref = _positive(reference_pressure, "reference_pressure")
    return float(a * (p / p_ref) ** n)


def klemmung(burning_area: float, throat_area: float) -> float:
    """Return the ratio ``Kn = A_b / A_t`` of burning area to throat area.

    Args:
        burning_area: ``A_b`` in m2, positive.
        throat_area: ``A_t`` in m2, positive.

    Returns:
        The dimensionless Klemmung (area ratio) ``Kn``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-SRM-002].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-002].

    """
    return _positive(burning_area, "burning_area") / _positive(
        throat_area, "throat_area"
    )


def equilibrium_pressure(
    coefficient: float,
    exponent: float,
    characteristic_velocity: float,
    propellant_density: float,
    klemmung_ratio: float,
    *,
    reference_pressure: float = 1.0e6,
) -> float:
    """Return the quasi-steady chamber pressure for a given Klemmung.

    Gas generation ``rho_p A_b r(p)`` equals the choked discharge
    ``p A_t / c*``, which gives
    ``p = (a rho_p c* Kn / p_ref^n)^(1 / (1 - n))`` with ``Kn = A_b / A_t``.

    Args:
        coefficient: ``a`` in m/s at ``p_ref`` (temperature corrected).
        exponent: Pressure exponent ``n`` with ``0 <= n < 1``.
        characteristic_velocity: ``c*`` in m/s, positive.
        propellant_density: ``rho_p`` in kg/m3, positive.
        klemmung_ratio: ``Kn = A_b / A_t``, positive.
        reference_pressure: ``p_ref`` in Pa.

    Returns:
        Equilibrium chamber pressure in Pa.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range (in particular ``n >= 1``,
            for which no equilibrium exists) or the result is not finite.

    Equation:
        [EQ-AERO-SRM-002].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-002].

    """
    a = _positive(coefficient, "coefficient")
    n = _require_exponent(exponent)
    c_star = _positive(characteristic_velocity, "characteristic_velocity")
    rho = _positive(propellant_density, "propellant_density")
    kn = _positive(klemmung_ratio, "klemmung_ratio")
    p_ref = _positive(reference_pressure, "reference_pressure")
    try:
        base = a * rho * c_star * kn / p_ref**n
        result = float(base ** (1.0 / (1.0 - n)))
    except OverflowError as error:
        raise ValueError("equilibrium pressure overflows") from error
    if not math.isfinite(result):
        raise ValueError("equilibrium pressure is not finite")
    return result


def chamber_rt(characteristic_velocity: float, gamma: float) -> float:
    """Return ``R T_c = (c* Gamma(gamma))^2`` of the combustion gas.

    The characteristic velocity of a perfect gas is
    ``c* = sqrt(R T_c) / Gamma`` with the Vandenkerckhove function
    ``Gamma = sqrt(gamma) (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1)))``;
    the product ``R T_c`` (J/kg) is all the lumped model needs of the gas.

    Args:
        characteristic_velocity: ``c*`` in m/s, positive.
        gamma: Specific-heat ratio, above one.

    Returns:
        ``R T_c`` in J/kg (m2/s2).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-SRM-003].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-003].

    """
    c_star = _positive(characteristic_velocity, "characteristic_velocity")
    g = _require_gamma(gamma)
    return (c_star * _vandenkerckhove(g)) ** 2


def _vandenkerckhove(gamma: float) -> float:
    """Return ``Gamma(gamma)``, the Vandenkerckhove function of a perfect gas.

    Args:
        gamma: Specific-heat ratio, above one.

    Returns:
        ``sqrt(gamma) (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1)))``.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

    """
    return float(
        math.sqrt(gamma)
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )


# --------------------------------------------------------------------------
# nozzle
# --------------------------------------------------------------------------


def _log_area_ratio(mach: float, gamma: float) -> float:
    """Return ``ln(A / A*)`` of isentropic flow at Mach number ``mach``.

    Args:
        mach: Mach number, positive.
        gamma: Specific-heat ratio.

    Returns:
        ``ln`` of the area-Mach relation.

    References:
        [EQ-AERO-SRM-006].

    """
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return -math.log(mach) + exponent * (
        math.log(2.0 / (gamma + 1.0)) + math.log1p(0.5 * (gamma - 1.0) * mach * mach)
    )


def exit_mach_number(gamma: float, expansion_ratio: float) -> float:
    """Return the supersonic exit Mach number of a nozzle of area ratio ``eps``.

    Solves the isentropic area-Mach relation
    ``eps = (1 / M) [(2 / (gamma + 1)) (1 + (gamma - 1) M^2 / 2)]^((gamma + 1) /
    (2 (gamma - 1)))`` on ``M >= 1`` by Brent's method.

    Args:
        gamma: Specific-heat ratio, above one.
        expansion_ratio: ``eps = A_e / A_t >= 1``.

    Returns:
        Exit Mach number (``1`` for ``eps = 1``).

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.
        RuntimeError: If the root search does not converge.

    Equation:
        [EQ-AERO-SRM-006].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

    """
    g = _require_gamma(gamma)
    eps = require_finite_real(expansion_ratio, name="expansion_ratio")
    if eps < 1.0:
        raise ValueError(f"expansion_ratio must be at least 1, got {eps!r}")
    return _exit_mach(g, eps)


@lru_cache(maxsize=512)
def _exit_mach(gamma: float, eps: float) -> float:
    """Return the cached supersonic root of the area-Mach relation.

    Args:
        gamma: Specific-heat ratio, above one.
        eps: Validated expansion ratio, at least one.

    Returns:
        Exit Mach number.

    Raises:
        RuntimeError: If the root search does not converge.

    References:
        [EQ-AERO-SRM-006].

    """
    if eps == 1.0:
        return 1.0
    target = math.log(eps)
    upper = 2.0
    while _log_area_ratio(upper, gamma) < target:
        upper *= 2.0
        if upper > 1.0e6:
            raise ValueError("expansion_ratio is too large for the area-Mach solver")
    result = brent_root(
        lambda mach: _log_area_ratio(mach, gamma) - target,
        1.0,
        upper,
        tolerance=1e-14,
    )
    if not result.converged:
        raise RuntimeError("exit Mach number search did not converge")
    return result.root


def exit_pressure_ratio(gamma: float, expansion_ratio: float) -> float:
    """Return ``p_e / p_c`` of the isentropic nozzle of area ratio ``eps``.

    ``p_e / p_c = (1 + (gamma - 1) M_e^2 / 2)^(-gamma / (gamma - 1))`` with the
    supersonic exit Mach number of :func:`exit_mach_number`.

    Args:
        gamma: Specific-heat ratio, above one.
        expansion_ratio: ``eps = A_e / A_t >= 1``.

    Returns:
        Exit-to-chamber pressure ratio in ``(0, 1)``; it does not depend on
        the chamber pressure for a fixed ``eps``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-SRM-006].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

    """
    return _nozzle_terms(_require_gamma(gamma), _eps_checked(expansion_ratio))[0]


def _eps_checked(expansion_ratio: object) -> float:
    """Return a validated expansion ratio ``eps >= 1``.

    Args:
        expansion_ratio: Candidate area ratio.

    Returns:
        The ratio as a float.

    Raises:
        TypeError: If the ratio is not a real number.
        ValueError: If the ratio is not finite or below one.

    References:
        [EQ-AERO-SRM-006].

    """
    eps = require_finite_real(expansion_ratio, name="expansion_ratio")
    if eps < 1.0:
        raise ValueError(f"expansion_ratio must be at least 1, got {eps!r}")
    return eps


@lru_cache(maxsize=512)
def _nozzle_terms(gamma: float, eps: float) -> tuple[float, float]:
    """Return ``(p_e / p_c, C_F momentum term)`` of a nozzle of area ratio ``eps``.

    Args:
        gamma: Specific-heat ratio, above one.
        eps: Expansion ratio, at least one.

    Returns:
        The exit pressure ratio and
        ``Gamma sqrt(2 gamma / (gamma - 1) (1 - (p_e / p_c)^((gamma - 1) / gamma)))``.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

    """
    mach = _exit_mach(gamma, eps)
    pi_e = (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (-gamma / (gamma - 1.0))
    momentum = _vandenkerckhove(gamma) * math.sqrt(
        2.0 * gamma / (gamma - 1.0) * (1.0 - pi_e ** ((gamma - 1.0) / gamma))
    )
    return pi_e, momentum


def ideal_thrust_coefficient(
    gamma: float,
    expansion_ratio: float,
    chamber_pressure: float,
    ambient_pressure: float = 0.0,
) -> float:
    """Return the ideal thrust coefficient ``C_F`` of a fixed-area-ratio nozzle.

    ``C_F = Gamma sqrt(2 gamma / (gamma - 1) [1 - (p_e / p_c)^((gamma - 1) /
    gamma)]) + (p_e / p_c - p_a / p_c) eps`` with ``p_e / p_c`` from
    :func:`exit_pressure_ratio`; the thrust is ``F = C_F p_c A_t``. It is
    maximal where ``p_e = p_a`` (optimum expansion).

    Args:
        gamma: Specific-heat ratio, above one.
        expansion_ratio: ``eps = A_e / A_t >= 1``.
        chamber_pressure: ``p_c`` in Pa, positive.
        ambient_pressure: ``p_a`` in Pa, non-negative (zero for vacuum).

    Returns:
        ``C_F`` (dimensionless); it is negative if the ambient pressure is
        high enough, where the ideal model (no separation) is not meaningful.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-SRM-006].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

    """
    g = _require_gamma(gamma)
    eps = _eps_checked(expansion_ratio)
    pc = _positive(chamber_pressure, "chamber_pressure")
    pa = _non_negative(ambient_pressure, "ambient_pressure")
    pi_e, momentum = _nozzle_terms(g, eps)
    return momentum + (pi_e - pa / pc) * eps


# --------------------------------------------------------------------------
# grain geometries
# --------------------------------------------------------------------------


def _classify(values: Sequence[float], tolerance: float) -> str:
    """Classify a burning-area history as neutral, progressive or regressive.

    Args:
        values: Burning areas at equally spaced webs.
        tolerance: Relative variation (of the largest value) treated as flat.

    Returns:
        One of ``"neutral"``, ``"progressive"``, ``"regressive"``,
        ``"progressive-regressive"``, ``"regressive-progressive"`` or
        ``"mixed"``.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

    """
    top = max(values)
    if max(values) - min(values) <= tolerance * top:
        return "neutral"
    slack = tolerance * top / len(values)
    signs: list[int] = []
    for before, after in zip(values, values[1:], strict=False):
        step = after - before
        sign = 1 if step > slack else (-1 if step < -slack else 0)
        if sign and (not signs or signs[-1] != sign):
            signs.append(sign)
    if signs == [1]:
        return "progressive"
    if signs == [-1]:
        return "regressive"
    if signs == [1, -1]:
        return "progressive-regressive"
    if signs == [-1, 1]:
        return "regressive-progressive"
    return "mixed"


class GrainGeometry(ABC):
    """Burning-surface geometry of a propellant grain as a function of web burned.

    A concrete grain provides the total web ``w``, the burning area
    ``A_b(e)`` and the remaining propellant volume ``V_p(e)`` for the web
    burned ``e`` (the distance every burning surface has receded, in m). The
    two are consistent: ``dV_p / de = -A_b`` on ``0 <= e <= w``.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

    """

    __slots__ = ()

    @property
    @abstractmethod
    def web(self) -> float:
        """Return the web thickness: the web burned at burn-out, in m.

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """

    @abstractmethod
    def burning_area(self, web_burned: float) -> float:
        """Return the burning area ``A_b(e)`` in m2.

        Args:
            web_burned: Web burned ``e`` in m, non-negative. ``e = web``
                gives the limiting value at burn-out; ``e > web`` gives ``0``
                (the grain is consumed).

        Returns:
            Burning area in m2.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If ``web_burned`` is negative or not finite.

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """

    @abstractmethod
    def propellant_volume(self, web_burned: float) -> float:
        """Return the remaining propellant volume ``V_p(e)`` in m3.

        Args:
            web_burned: Web burned ``e`` in m, non-negative; ``e >= web``
                gives ``0``.

        Returns:
            Propellant volume in m3.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If ``web_burned`` is negative or not finite.

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """

    def classification(self, *, tolerance: float = 0.02, samples: int = 201) -> str:
        """Classify the burning-area history over the web.

        The area is sampled at ``samples`` equally spaced webs. It is
        ``"neutral"`` when it varies by at most ``tolerance`` of its maximum,
        ``"progressive"`` / ``"regressive"`` when it only increases /
        decreases (beyond the same tolerance), and ``"progressive-regressive"``,
        ``"regressive-progressive"`` or ``"mixed"`` otherwise.

        Args:
            tolerance: Relative flat band, in ``(0, 1)``.
            samples: Number of sample webs, at least three.

        Returns:
            The class name.

        Raises:
            ValueError: If ``tolerance`` or ``samples`` is invalid.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        tol = require_finite_real(tolerance, name="tolerance")
        if not 0.0 < tol < 1.0:
            raise ValueError("tolerance must lie in (0, 1)")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
            raise ValueError("samples must be an integer of at least 3")
        web = self.web
        values = [self.burning_area(web * i / (samples - 1)) for i in range(samples)]
        return _classify(values, tol)

    def _checked(self, web_burned: object) -> float:
        """Return a validated web burned.

        Args:
            web_burned: Candidate web burned.

        Returns:
            The value as a float.

        Raises:
            TypeError: If the value is not a real number.
            ValueError: If it is negative or not finite.

        References:
            [EQ-AERO-SRM-004].

        """
        return _non_negative(web_burned, "web_burned")


@dataclass(frozen=True, slots=True)
class TubularGrain(GrainGeometry):
    """Cylindrical tube burning on its bore (and optionally its outside).

    The two end faces are inhibited. With ``outer_burning=False`` only the
    bore burns outward: ``A_b = 2 pi (r_i + e) L`` (progressive) and the web is
    ``r_o - r_i``. With ``outer_burning=True`` the outer surface burns inward
    as well: ``A_b = 2 pi L (r_i + r_o)`` (exactly neutral) and the web is
    ``(r_o - r_i) / 2``.

    Args:
        inner_radius: Bore radius ``r_i`` in m, positive.
        outer_radius: Outer radius ``r_o`` in m, greater than ``r_i``.
        length: Length ``L`` in m, positive.
        outer_burning: Whether the outer cylindrical surface also burns.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

    """

    inner_radius: float
    outer_radius: float
    length: float
    outer_burning: bool = False

    def __post_init__(self) -> None:
        """Validate the dimensions.

        Raises:
            TypeError: If a dimension is not a real number.
            ValueError: If a dimension is not positive or ``r_o <= r_i``.

        References:
            [EQ-AERO-SRM-004].

        """
        ri = _positive(self.inner_radius, "inner_radius")
        ro = _positive(self.outer_radius, "outer_radius")
        length = _positive(self.length, "length")
        if ro <= ri:
            raise ValueError("outer_radius must exceed inner_radius")
        object.__setattr__(self, "inner_radius", ri)
        object.__setattr__(self, "outer_radius", ro)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "outer_burning", bool(self.outer_burning))

    @property
    def web(self) -> float:
        """Return the web thickness in m.

        References:
            [EQ-AERO-SRM-004].

        """
        full = self.outer_radius - self.inner_radius
        return 0.5 * full if self.outer_burning else full

    def burning_area(self, web_burned: float) -> float:
        """Return ``A_b(e)`` in m2.

        Args:
            web_burned: Web burned ``e`` in m.

        Returns:
            ``2 pi (r_i + e) L``, or ``2 pi (r_i + r_o) L`` when the outer
            surface burns too; ``0`` beyond the web.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        e = self._checked(web_burned)
        if e > self.web:
            return 0.0
        if self.outer_burning:
            return 2.0 * math.pi * self.length * (self.inner_radius + self.outer_radius)
        return 2.0 * math.pi * (self.inner_radius + e) * self.length

    def propellant_volume(self, web_burned: float) -> float:
        """Return ``V_p(e)`` in m3.

        Args:
            web_burned: Web burned ``e`` in m.

        Returns:
            ``pi L [(r_o - e_o)^2 - (r_i + e)^2]`` with ``e_o = e`` if the
            outer surface burns and ``0`` otherwise; ``0`` beyond the web.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        e = self._checked(web_burned)
        if e >= self.web:
            return 0.0
        outer = self.outer_radius - (e if self.outer_burning else 0.0)
        return math.pi * self.length * (outer**2 - (self.inner_radius + e) ** 2)


@dataclass(frozen=True, slots=True)
class EndBurningGrain(GrainGeometry):
    """Cylindrical end burner: one end face burns, everything else is inhibited.

    ``A_b = pi R^2`` for all ``e`` (exactly neutral), the web is the grain
    length ``L`` and ``V_p = pi R^2 (L - e)``.

    Args:
        radius: Grain radius ``R`` in m, positive.
        length: Grain length ``L`` in m, positive.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

    """

    radius: float
    length: float

    def __post_init__(self) -> None:
        """Validate the dimensions.

        Raises:
            TypeError: If a dimension is not a real number.
            ValueError: If a dimension is not positive.

        References:
            [EQ-AERO-SRM-004].

        """
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        object.__setattr__(self, "length", _positive(self.length, "length"))

    @property
    def web(self) -> float:
        """Return the web thickness (the grain length) in m.

        References:
            [EQ-AERO-SRM-004].

        """
        return self.length

    def burning_area(self, web_burned: float) -> float:
        """Return ``A_b = pi R^2`` in m2 (zero beyond the web).

        Args:
            web_burned: Web burned ``e`` in m.

        Returns:
            Burning area in m2.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        e = self._checked(web_burned)
        return 0.0 if e > self.length else math.pi * self.radius**2

    def propellant_volume(self, web_burned: float) -> float:
        """Return ``V_p = pi R^2 (L - e)`` in m3 (zero beyond the web).

        Args:
            web_burned: Web burned ``e`` in m.

        Returns:
            Propellant volume in m3.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        e = self._checked(web_burned)
        return 0.0 if e >= self.length else math.pi * self.radius**2 * (self.length - e)


@dataclass(frozen=True, slots=True)
class BatesGrain(GrainGeometry):
    """Stack of BATES segments burning on the bore and on both end faces.

    Each segment is a tube of outer diameter ``D``, core diameter ``d_0`` and
    length ``L``; its outer surface is inhibited (bonded to the case). With
    ``u = d_0 + 2 e`` the burning area and propellant volume of ``N``
    segments are ``A_b = N [pi u (L - 2 e) + (pi / 2)(D^2 - u^2)]`` and
    ``V_p = N (pi / 4)(D^2 - u^2)(L - 2 e)``. The web is
    ``min((D - d_0) / 2, L / 2)``: the radial web or the half length, whichever
    is consumed first (the sliver left at burn-out is neglected).

    Args:
        outer_diameter: ``D`` in m, positive.
        core_diameter: ``d_0`` in m, positive and below ``D``.
        segment_length: ``L`` in m, positive.
        segments: Number of segments ``N >= 1``.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

    """

    outer_diameter: float
    core_diameter: float
    segment_length: float
    segments: int = 1

    def __post_init__(self) -> None:
        """Validate the dimensions and the segment count.

        Raises:
            TypeError: If a dimension is not a real number or ``segments`` is
                not an integer.
            ValueError: If a dimension is not positive, ``d_0 >= D`` or
                ``segments < 1``.

        References:
            [EQ-AERO-SRM-004].

        """
        outer = _positive(self.outer_diameter, "outer_diameter")
        core = _positive(self.core_diameter, "core_diameter")
        length = _positive(self.segment_length, "segment_length")
        if isinstance(self.segments, bool) or not isinstance(self.segments, int):
            raise TypeError("segments must be an integer")
        if self.segments < 1:
            raise ValueError("segments must be at least 1")
        if core >= outer:
            raise ValueError("core_diameter must be smaller than outer_diameter")
        object.__setattr__(self, "outer_diameter", outer)
        object.__setattr__(self, "core_diameter", core)
        object.__setattr__(self, "segment_length", length)

    @property
    def web(self) -> float:
        """Return ``min((D - d_0) / 2, L / 2)`` in m.

        References:
            [EQ-AERO-SRM-004].

        """
        return min(
            0.5 * (self.outer_diameter - self.core_diameter),
            0.5 * self.segment_length,
        )

    def burning_area(self, web_burned: float) -> float:
        """Return ``A_b(e)`` of all segments in m2 (zero beyond the web).

        Args:
            web_burned: Web burned ``e`` in m.

        Returns:
            ``N [pi u (L - 2 e) + (pi / 2)(D^2 - u^2)]`` with
            ``u = d_0 + 2 e``.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        e = self._checked(web_burned)
        if e > self.web:
            return 0.0
        u = self.core_diameter + 2.0 * e
        length = self.segment_length - 2.0 * e
        bore = math.pi * u * length
        ends = 0.5 * math.pi * (self.outer_diameter**2 - u * u)
        return self.segments * (bore + ends)

    def propellant_volume(self, web_burned: float) -> float:
        """Return ``V_p(e)`` of all segments in m3 (zero beyond the web).

        Args:
            web_burned: Web burned ``e`` in m.

        Returns:
            ``N (pi / 4)(D^2 - u^2)(L - 2 e)`` with ``u = d_0 + 2 e``.

        Raises:
            TypeError: If ``web_burned`` is not a real number.
            ValueError: If it is negative or not finite.

        Equation:
            [EQ-AERO-SRM-004].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-004].

        """
        e = self._checked(web_burned)
        if e >= self.web:
            return 0.0
        u = self.core_diameter + 2.0 * e
        return (
            self.segments
            * 0.25
            * math.pi
            * (self.outer_diameter**2 - u * u)
            * (self.segment_length - 2.0 * e)
        )


# --------------------------------------------------------------------------
# motor data and results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolidMotorParameters:
    """Propellant, nozzle and case data of one solid motor (SI units).

    Args:
        burn_rate_coefficient: ``a_0`` in m/s: burn rate at ``p_ref`` and
            ``T_ref``.
        pressure_exponent: ``n`` with ``0 <= n < 1``.
        characteristic_velocity: ``c*`` in m/s.
        propellant_density: ``rho_p`` in kg/m3.
        throat_diameter: Initial throat diameter in m.
        chamber_volume: Internal case volume in m3 (grain plus free gas
            volume); it must exceed the initial propellant volume of the
            grain it is used with.
        expansion_ratio: Initial nozzle area ratio ``eps_0 = A_e / A_t(0)``;
            the exit area stays fixed while the throat erodes.
        gamma: Specific-heat ratio of the combustion gas, above one.
        temperature_sensitivity: ``sigma_p`` in 1/K, non-negative.
        grain_temperature: Initial grain temperature in K.
        reference_temperature: ``T_ref`` in K.
        reference_pressure: ``p_ref`` in Pa.
        erosion_rate: Throat radius recession rate in m/s, non-negative.
        ambient_pressure: ``p_a`` in Pa (zero for vacuum).

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-001], [EQ-AERO-SRM-005].

    """

    burn_rate_coefficient: float
    pressure_exponent: float
    characteristic_velocity: float
    propellant_density: float
    throat_diameter: float
    chamber_volume: float
    expansion_ratio: float
    gamma: float
    temperature_sensitivity: float = 0.0
    grain_temperature: float = 293.15
    reference_temperature: float = 293.15
    reference_pressure: float = 1.0e6
    erosion_rate: float = 0.0
    ambient_pressure: float = 101325.0

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If a field is out of range (in particular
                ``pressure_exponent`` outside ``[0, 1)``).

        References:
            [EQ-AERO-SRM-001], [EQ-AERO-SRM-002].

        """
        fields = {
            "burn_rate_coefficient": _positive(
                self.burn_rate_coefficient, "burn_rate_coefficient"
            ),
            "pressure_exponent": _require_exponent(self.pressure_exponent),
            "characteristic_velocity": _positive(
                self.characteristic_velocity, "characteristic_velocity"
            ),
            "propellant_density": _positive(
                self.propellant_density, "propellant_density"
            ),
            "throat_diameter": _positive(self.throat_diameter, "throat_diameter"),
            "chamber_volume": _positive(self.chamber_volume, "chamber_volume"),
            "expansion_ratio": _eps_checked(self.expansion_ratio),
            "gamma": _require_gamma(self.gamma),
            "temperature_sensitivity": _non_negative(
                self.temperature_sensitivity, "temperature_sensitivity"
            ),
            "grain_temperature": _positive(self.grain_temperature, "grain_temperature"),
            "reference_temperature": _positive(
                self.reference_temperature, "reference_temperature"
            ),
            "reference_pressure": _positive(
                self.reference_pressure, "reference_pressure"
            ),
            "erosion_rate": _non_negative(self.erosion_rate, "erosion_rate"),
            "ambient_pressure": _non_negative(
                self.ambient_pressure, "ambient_pressure"
            ),
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    @property
    def effective_coefficient(self) -> float:
        """Return ``a(T)`` at the initial grain temperature, in m/s.

        Equation:
            [EQ-AERO-SRM-001].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-001].

        """
        return burn_rate_coefficient(
            self.burn_rate_coefficient,
            self.temperature_sensitivity,
            self.grain_temperature,
            self.reference_temperature,
        )

    def throat_area(self, time: float = 0.0) -> float:
        """Return the throat area ``pi (d_t / 2 + epsilon t)^2`` in m2.

        Args:
            time: Time since ignition in s, non-negative.

        Returns:
            Throat area including the linear radial erosion.

        Raises:
            TypeError: If ``time`` is not a real number.
            ValueError: If ``time`` is negative or not finite.

        Equation:
            [EQ-AERO-SRM-005].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-005].

        """
        t = _non_negative(time, "time")
        radius = 0.5 * self.throat_diameter + self.erosion_rate * t
        return math.pi * radius * radius

    @property
    def exit_area(self) -> float:
        """Return the fixed nozzle exit area ``eps_0 A_t(0)`` in m2.

        Equation:
            [EQ-AERO-SRM-006].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-006].

        """
        return self.expansion_ratio * self.throat_area(0.0)

    def equilibrium_pressure(self, burning_area: float, time: float = 0.0) -> float:
        """Return the quasi-steady pressure for a burning area at ``time``.

        Args:
            burning_area: ``A_b`` in m2, positive.
            time: Time since ignition in s (sets the eroded throat).

        Returns:
            Equilibrium chamber pressure in Pa.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If an argument is out of range.

        Equation:
            [EQ-AERO-SRM-002].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-002].

        """
        return equilibrium_pressure(
            self.effective_coefficient,
            self.pressure_exponent,
            self.characteristic_velocity,
            self.propellant_density,
            klemmung(burning_area, self.throat_area(time)),
            reference_pressure=self.reference_pressure,
        )


@dataclass(frozen=True, slots=True)
class SolidMotorResult:
    """Pressure-time and thrust-time traces and summary of one motor run.

    Args:
        mode: ``"quasi_steady"`` or ``"transient"``.
        times: Trace times in s (strictly increasing).
        pressures: Chamber pressure at each time, in Pa.
        thrusts: Thrust at each time, in N.
        webs_burned: Web burned at each time, in m.
        burning_areas: Burning area at each time, in m2 (zero in the tail).
        throat_areas: Eroded throat area at each time, in m2.
        burn_time: Time of web burn-out in s (the time reached when the run
            did not converge).
        action_time: End time of the trace in s: equal to ``burn_time`` in the
            quasi-steady mode, the end of the tail-off in the transient mode.
        meop: Peak chamber pressure of the run in Pa (the realized maximum
            operating pressure; its upper quantile over the input uncertainty
            is the design MEOP).
        total_impulse: ``integral F dt`` over the trace in N s (trapezoid
            rule on the trace times).
        average_thrust: ``total_impulse / action_time`` in N.
        propellant_mass: Mass of propellant burned in kg.
        converged: Whether web burn-out was reached.
        stop_reason: ``"web_burnout"`` or ``"step_limit"``.
        steps: Number of integration steps taken in the burning phase.

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-007].

    """

    mode: str
    times: tuple[float, ...]
    pressures: tuple[float, ...]
    thrusts: tuple[float, ...]
    webs_burned: tuple[float, ...]
    burning_areas: tuple[float, ...]
    throat_areas: tuple[float, ...]
    burn_time: float
    action_time: float
    meop: float
    total_impulse: float
    average_thrust: float
    propellant_mass: float
    converged: bool
    stop_reason: str
    steps: int

    @property
    def specific_impulse(self) -> float:
        """Return ``I_sp = I_t / (m_p g_0)`` in s.

        Equation:
            [EQ-AERO-SRM-007].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-007].

        """
        return self.total_impulse / (self.propellant_mass * STANDARD_GRAVITY)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------


class _Motor:
    """Right-hand sides and events of one motor, for the ODE integrator.

    Args:
        grain: Grain geometry.
        parameters: Motor data.
        dilatation: Whether the transient balance keeps the volume-growth term.

    References:
        [EQ-AERO-SRM-002], [EQ-AERO-SRM-003].

    """

    def __init__(
        self, grain: GrainGeometry, parameters: SolidMotorParameters, dilatation: bool
    ) -> None:
        """Cache the constants used in every right-hand-side call.

        Raises:
            ValueError: If the chamber is not larger than the propellant.

        References:
            [EQ-AERO-SRM-003].

        """
        self.grain = grain
        self.params = parameters
        self.dilatation = dilatation
        self.web = grain.web
        self.coefficient = parameters.effective_coefficient
        self.exponent = parameters.pressure_exponent
        self.rho = parameters.propellant_density
        self.c_star = parameters.characteristic_velocity
        self.p_ref = parameters.reference_pressure
        self.rt = chamber_rt(self.c_star, parameters.gamma)
        self.pressure_factor = self.coefficient * self.rho * self.c_star
        self.pressure_factor /= self.p_ref**self.exponent
        self.pressure_power = 1.0 / (1.0 - self.exponent)
        self.throat_radius = 0.5 * parameters.throat_diameter
        self.erosion = parameters.erosion_rate
        self.initial_volume = grain.propellant_volume(0.0)
        if parameters.chamber_volume <= self.initial_volume:
            raise ValueError(
                "chamber_volume must exceed the initial propellant volume "
                f"({self.initial_volume!r} m3)"
            )

    def throat(self, time: float) -> float:
        """Return the eroded throat area in m2 without re-validating.

        References:
            [EQ-AERO-SRM-005].

        """
        radius = self.throat_radius + self.erosion * time
        return math.pi * radius * radius

    def equilibrium(self, area: float, time: float) -> float:
        """Return the quasi-steady pressure of [EQ-AERO-SRM-002] in Pa.

        References:
            [EQ-AERO-SRM-002].

        """
        return float(
            (self.pressure_factor * area / self.throat(time)) ** self.pressure_power
        )

    def free_volume(self, web_burned: float) -> float:
        """Return the free gas volume ``V_c - V_p(e)`` in m3.

        References:
            [EQ-AERO-SRM-003].

        """
        return self.params.chamber_volume - self.grain.propellant_volume(web_burned)

    def quasi_rhs(self, time: float, state: Sequence[float]) -> tuple[float]:
        """Return ``de/dt = r(p_eq(e, t))`` for the state ``[e]``.

        References:
            [EQ-AERO-SRM-001], [EQ-AERO-SRM-002].

        """
        e = min(state[0], self.web)
        area = self.grain.burning_area(e)
        pressure = self.equilibrium(area, time)
        return (self.coefficient * (pressure / self.p_ref) ** self.exponent,)

    def burnout_quasi(self, time: float, state: Sequence[float]) -> float:
        """Return the remaining web ``w - e`` (event at burn-out).

        References:
            [EQ-AERO-SRM-004].

        """
        return self.web - state[0]

    def transient_rhs(self, time: float, state: Sequence[float]) -> tuple[float, float]:
        """Return ``(dp/dt, de/dt)`` of the gas mass balance for ``[p, e]``.

        References:
            [EQ-AERO-SRM-003], [EQ-AERO-SRM-001].

        """
        pressure, e = state
        if pressure < 0.0:
            raise ValueError("chamber pressure became negative")
        e = min(e, self.web)
        area = self.grain.burning_area(e)
        volume = self.free_volume(e)
        rate = self.coefficient * (pressure / self.p_ref) ** self.exponent
        generation = self.rho * area * rate
        discharge = pressure * self.throat(time) / self.c_star
        slope = self.rt / volume * (generation - discharge)
        if self.dilatation:
            slope -= pressure * area * rate / volume
        return (slope, rate)

    def burnout_transient(self, time: float, state: Sequence[float]) -> float:
        """Return the remaining web ``w - e`` for the state ``[p, e]``.

        References:
            [EQ-AERO-SRM-004].

        """
        return self.web - state[1]

    def tail_rhs(self, time: float, state: Sequence[float]) -> tuple[float]:
        """Return ``dp/dt`` after burn-out (no generation, constant volume).

        References:
            [EQ-AERO-SRM-003].

        """
        volume = self.free_volume(self.web)
        rate = self.rt * self.throat(time) / (volume * self.c_star)
        return (-rate * state[0],)


def _nozzle_state(
    parameters: SolidMotorParameters, throat_area: float
) -> tuple[float, float, float]:
    """Return ``(p_e / p_c, momentum term, eps)`` at the current throat area.

    Args:
        parameters: Motor data.
        throat_area: Current (eroded) throat area in m2.

    Returns:
        The nozzle terms of the ideal thrust coefficient.

    References:
        [EQ-AERO-SRM-006].

    """
    eps = parameters.exit_area / throat_area
    # a heavily eroded throat can in principle exceed the exit area
    eps = max(eps, 1.0)
    pi_e, momentum = _nozzle_terms(parameters.gamma, eps)
    return pi_e, momentum, eps


def _thrust(parameters: SolidMotorParameters, pressure: float, time: float) -> float:
    """Return the ideal thrust ``C_F p A_t`` in N.

    Args:
        parameters: Motor data.
        pressure: Chamber pressure in Pa.
        time: Time in s (sets the eroded throat).

    Returns:
        Thrust in N.

    References:
        [EQ-AERO-SRM-006].

    """
    area = parameters.throat_area(time)
    pi_e, momentum, eps = _nozzle_state(parameters, area)
    coefficient = momentum + eps * pi_e
    return area * (pressure * coefficient - eps * parameters.ambient_pressure)


def _trapezoid(times: Sequence[float], values: Sequence[float]) -> float:
    """Return the trapezoid-rule integral of ``values`` over ``times``.

    Args:
        times: Increasing abscissae.
        values: Ordinates at ``times``.

    Returns:
        The composite trapezoid integral.

    References:
        [EQ-AERO-SRM-007].

    """
    return sum(
        0.5 * (times[i + 1] - times[i]) * (values[i] + values[i + 1])
        for i in range(len(times) - 1)
    )


def _tail_step(motor: _Motor, time: float) -> float:
    """Return a stable tail-off step: a quarter of the chamber emptying time.

    Args:
        motor: Motor right-hand sides.
        time: Time of burn-out in s.

    Returns:
        Step in s.

    References:
        [EQ-AERO-SRM-003].

    """
    tau = (
        motor.free_volume(motor.web)
        * motor.c_star
        / (motor.rt * motor.params.throat_area(time))
    )
    return 0.25 * tau


def _tail_pressure_floor(
    parameters: SolidMotorParameters, peak: float, fraction: float, time: float
) -> float:
    """Return the pressure at which the tail-off trace ends.

    The larger of ``fraction * peak`` and the pressure where the ideal thrust
    ``C_F p A_t`` falls to zero against the ambient pressure.

    Args:
        parameters: Motor data.
        peak: Peak chamber pressure in Pa.
        fraction: Cut-off fraction of the peak.
        time: Time of burn-out in s.

    Returns:
        Cut-off pressure in Pa.

    References:
        [EQ-AERO-SRM-006], [EQ-AERO-SRM-007].

    """
    pi_e, momentum, eps = _nozzle_state(parameters, parameters.throat_area(time))
    zero_thrust = eps * parameters.ambient_pressure / (momentum + eps * pi_e)
    return max(fraction * peak, zero_thrust)


def simulate_solid_motor(
    grain: GrainGeometry,
    parameters: SolidMotorParameters,
    *,
    mode: str = "quasi_steady",
    steps: int = 200,
    step: float | None = None,
    initial_pressure: float = 101325.0,
    tail_fraction: float = 0.05,
    dilatation: bool = True,
    max_steps: int = 200_000,
) -> SolidMotorResult:
    """Simulate one motor burn and return its pressure and thrust traces.

    ``mode="quasi_steady"`` integrates only the web burned,
    ``de/dt = r(p_eq(e, t))`` with the equilibrium pressure
    ([EQ-AERO-SRM-002]) at the current burning area and eroded throat. The
    trace ends at web burn-out (abrupt cut-off; no start-up or tail-off).
    ``mode="transient"`` integrates the pressure form of the gas mass balance
    ``dp/dt = (R T_c / V)(rho_p A_b r - p A_t / c*) - (p / V) A_b r``
    ([EQ-AERO-SRM-003]) together with ``de/dt = r`` from ``initial_pressure``,
    then the tail-off ``dp/dt = -R T_c A_t p / (V c*)`` until the pressure
    reaches the larger of ``tail_fraction`` times the peak and the pressure
    where the ideal thrust is zero. Both use the fixed-step Runge-Kutta
    integrator of :func:`probstat.numerics.ode.integrate_to_event`
    ([EQ-NUM-ODE-001]) with the web burn-out located by the cubic Hermite
    event ([EQ-NUM-ODE-002]).

    Args:
        grain: Grain geometry.
        parameters: Motor data; ``parameters.chamber_volume`` must exceed the
            initial propellant volume.
        mode: ``"quasi_steady"`` or ``"transient"``.
        steps: Approximate number of steps over the expected burn time
            (sets the step of the quasi-steady mode; the transient step is
            the smaller of this and a quarter of the chamber filling time
            ``V c* / (R T_c A_t)``, which keeps the explicit scheme stable).
        step: Explicit step in s (overrides the automatic one).
        initial_pressure: Chamber pressure at ignition in Pa, positive
            (transient mode only; the igniter is not modelled).
        tail_fraction: Tail-off cut-off as a fraction of the peak pressure,
            in ``(0, 1)`` (transient mode only).
        dilatation: Keep the ``-(p / V) dV / dt`` term of the transient
            balance; without it the steady state is exactly the closed form
            of [EQ-AERO-SRM-002].
        max_steps: Step limit of the burning phase.

    Returns:
        :class:`SolidMotorResult`; ``converged=False`` with
        ``stop_reason="step_limit"`` if the web was not consumed within
        ``max_steps`` (the trace is then the partial solution).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, the chamber is not larger
            than the propellant, or a state becomes non-finite.

    Equation:
        [EQ-AERO-SRM-001], [EQ-AERO-SRM-002], [EQ-AERO-SRM-003],
        [EQ-AERO-SRM-005], [EQ-AERO-SRM-006], [EQ-AERO-SRM-007].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-NUM-ODE-001], [EQ-NUM-ODE-002].

    """
    if not isinstance(grain, GrainGeometry):
        raise TypeError("grain must be a GrainGeometry")
    if not isinstance(parameters, SolidMotorParameters):
        raise TypeError("parameters must be SolidMotorParameters")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 10:
        raise ValueError("steps must be an integer of at least 10")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    p_init = _positive(initial_pressure, "initial_pressure")
    fraction = require_finite_real(tail_fraction, name="tail_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    motor = _Motor(grain, parameters, bool(dilatation))
    web = motor.web
    p_start = parameters.equilibrium_pressure(grain.burning_area(0.0))
    r_start = motor.coefficient * (p_start / motor.p_ref) ** motor.exponent
    h = web / r_start / steps
    if mode == "transient":
        tau = (
            motor.free_volume(0.0)
            * motor.c_star
            / (motor.rt * parameters.throat_area(0.0))
        )
        h = min(h, 0.25 * tau)
    if step is not None:
        h = _positive(step, "step")
    converged = True
    rhs: Rhs
    event: Event
    if mode == "quasi_steady":
        rhs, event, y0 = motor.quasi_rhs, motor.burnout_quasi, [0.0]
    else:
        rhs, event, y0 = motor.transient_rhs, motor.burnout_transient, [p_init, 0.0]
    try:
        solution = integrate_to_event(
            rhs, y0, event, step=h, max_steps=max_steps, record=True
        )
        path = list(solution.path)
        del path[-2]  # the grid point beyond burn-out; the located event follows
        taken = solution.steps
    except RuntimeError:
        converged = False
        trajectory = integrate_fixed_step(rhs, 0.0, y0, step=h, steps=max_steps)
        path = list(zip(trajectory.times, trajectory.states, strict=True))
        taken = max_steps
    times = [t for t, _ in path]
    webs = [min(state[-1], web) for _, state in path]
    areas = [grain.burning_area(e) for e in webs]
    if mode == "quasi_steady":
        pressures = [
            motor.equilibrium(area, t) for area, t in zip(areas, times, strict=True)
        ]
    else:
        pressures = [state[0] for _, state in path]
    burn_time = times[-1]
    burned_web = webs[-1]
    if mode == "transient" and converged:
        peak = max(pressures)
        floor = _tail_pressure_floor(parameters, peak, fraction, burn_time)
        if pressures[-1] > floor:
            tail = integrate_to_event(
                motor.tail_rhs,
                [pressures[-1]],
                lambda t, y: y[0] - floor,
                step=_tail_step(motor, burn_time),
                start=burn_time,
                max_steps=max_steps,
                record=True,
            )
            tail_path = list(tail.path)
            del tail_path[-2]
            for t, state in tail_path[1:]:
                times.append(t)
                pressures.append(state[0])
                webs.append(web)
                areas.append(0.0)
    throats = [motor.throat(t) for t in times]
    thrusts = [_thrust(parameters, p, t) for p, t in zip(pressures, times, strict=True)]
    impulse = _trapezoid(times, thrusts)
    action = times[-1]
    mass = motor.rho * (motor.initial_volume - grain.propellant_volume(burned_web))
    return SolidMotorResult(
        mode=mode,
        times=tuple(times),
        pressures=tuple(pressures),
        thrusts=tuple(thrusts),
        webs_burned=tuple(webs),
        burning_areas=tuple(areas),
        throat_areas=tuple(throats),
        burn_time=burn_time,
        action_time=action,
        meop=max(pressures),
        total_impulse=impulse,
        average_thrust=impulse / action,
        propellant_mass=mass,
        converged=converged,
        stop_reason="web_burnout" if converged else "step_limit",
        steps=taken,
    )


# --------------------------------------------------------------------------
# probabilistic model
# --------------------------------------------------------------------------


class _SolidMotorForward:
    """Forward map of :func:`solid_motor_model`: named inputs to the four QoIs.

    Args:
        grain: Grain geometry (fixed).
        nominal: Motor data whose uncertain fields are replaced per call.
        mode: Simulation mode.
        steps: Step count passed to :func:`simulate_solid_motor`.

    References:
        [EQ-AERO-SRM-007].

    """

    def __init__(
        self,
        grain: GrainGeometry,
        nominal: SolidMotorParameters,
        mode: str,
        steps: int,
    ) -> None:
        """Store the fixed motor description.

        References:
            [EQ-AERO-SRM-007].

        """
        self.grain = grain
        self.nominal = nominal
        self.mode = mode
        self.steps = steps

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Run the motor for one realization of the uncertain inputs.

        Args:
            parameters: Values of ``a0``, ``n``, ``c_star``, ``rho_p``,
                ``sigma_p``, ``T_grain``, ``d_throat``, ``erosion_rate`` and
                ``p_burst``.

        Returns:
            ``meop`` (Pa), ``total_impulse`` (N s), ``burn_time`` (s) and
            ``burst_margin = p_burst - meop`` (Pa).

        Raises:
            RuntimeError: If the burn did not reach web burn-out.
            ValueError: If a realized input is out of range.

        Equation:
            [EQ-AERO-SRM-007].

        References:
            [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-007].

        """
        realized = replace(
            self.nominal,
            burn_rate_coefficient=parameters["a0"],
            pressure_exponent=parameters["n"],
            characteristic_velocity=parameters["c_star"],
            propellant_density=parameters["rho_p"],
            temperature_sensitivity=parameters["sigma_p"],
            grain_temperature=parameters["T_grain"],
            throat_diameter=parameters["d_throat"],
            erosion_rate=parameters["erosion_rate"],
            expansion_ratio=self.nominal.exit_area
            / (0.25 * math.pi * parameters["d_throat"] ** 2),
        )
        result = simulate_solid_motor(
            self.grain, realized, mode=self.mode, steps=self.steps
        )
        if not result.converged:
            raise RuntimeError("solid-motor burn did not reach web burn-out")
        return {
            "meop": result.meop,
            "total_impulse": result.total_impulse,
            "burn_time": result.burn_time,
            "burst_margin": parameters["p_burst"] - result.meop,
        }


def solid_motor_model(
    grain: GrainGeometry,
    nominal: SolidMotorParameters,
    burst_pressure: float,
    *,
    burn_rate_cov: float = 0.04,
    exponent_sd: float = 0.02,
    c_star_cov: float = 0.01,
    density_cov: float = 0.01,
    sensitivity_cov: float = 0.15,
    temperature_bounds: tuple[float, float] = (273.15, 323.15),
    throat_cov: float = 0.002,
    erosion_spread: float = 0.5,
    burst_cov: float = 0.05,
    mode: str = "quasi_steady",
    steps: int = 60,
) -> ProbabilisticModel:
    """Return a probabilistic model of a solid motor with nine uncertain inputs.

    Independent marginals (all spreads are assumed engineering values, not
    data): ``a0`` lognormal with median ``a_0`` and log-scale deviation
    ``burn_rate_cov``; ``n`` normal truncated to ``[0, 0.95]`` with mean ``n``
    and standard deviation ``exponent_sd``; ``c_star`` and ``rho_p`` normal
    with relative deviations ``c_star_cov`` and ``density_cov``; ``sigma_p``
    normal truncated at zero with relative deviation ``sensitivity_cov``;
    ``T_grain`` uniform on ``temperature_bounds`` (K); ``d_throat`` normal
    with relative deviation ``throat_cov``; ``erosion_rate`` uniform on
    ``(1 - s, 1 + s)`` times the nominal rate with ``s = erosion_spread``;
    ``p_burst`` lognormal with median ``burst_pressure`` and log-scale
    deviation ``burst_cov``. The nozzle exit area is fixed hardware (that of
    ``nominal``), so the expansion ratio varies with the realized throat.
    Quantities of interest: ``meop`` (Pa), ``total_impulse`` (N s),
    ``burn_time`` (s) and ``burst_margin = p_burst - meop`` (Pa), so that
    ``burst_margin <= 0`` is a case burst.

    Args:
        grain: Grain geometry (not random).
        nominal: Nominal motor data; ``temperature_sensitivity`` and
            ``erosion_rate`` must be positive and the nominal exponent must
            lie in ``(0, 0.95)``. The other uncertain fields are replaced by
            the random inputs.
        burst_pressure: Median case burst pressure in Pa, positive.
        burn_rate_cov: Log-scale deviation of ``a0``.
        exponent_sd: Standard deviation of ``n``.
        c_star_cov: Relative deviation of ``c*``.
        density_cov: Relative deviation of ``rho_p``.
        sensitivity_cov: Relative deviation of ``sigma_p``.
        temperature_bounds: ``(low, high)`` of the grain temperature in K.
        throat_cov: Relative deviation of the throat diameter.
        erosion_spread: Half-width of the erosion-rate range, in ``(0, 1)``.
        burst_cov: Log-scale deviation of the burst pressure.
        mode: ``"quasi_steady"`` (fast, default) or ``"transient"``.
        steps: Step count of :func:`simulate_solid_motor`.

    Returns:
        :class:`~probstat.modeling.ProbabilisticModel` named
        ``"solid_motor"``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-SRM-007].

    References:
        [REF-SUTTON-BIBLARZ-2017], [EQ-AERO-SRM-001], [EQ-AERO-SRM-007].

    """
    if not isinstance(grain, GrainGeometry):
        raise TypeError("grain must be a GrainGeometry")
    if not isinstance(nominal, SolidMotorParameters):
        raise TypeError("nominal must be SolidMotorParameters")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    burst = _positive(burst_pressure, "burst_pressure")
    if nominal.temperature_sensitivity <= 0.0 or nominal.erosion_rate <= 0.0:
        raise ValueError(
            "temperature_sensitivity and erosion_rate of nominal must be positive "
            "(they are uncertain inputs with relative spreads)"
        )
    if not 0.0 < nominal.pressure_exponent < 0.95:
        raise ValueError("nominal pressure_exponent must lie in (0, 0.95)")
    low = require_finite_real(temperature_bounds[0], name="temperature_bounds[0]")
    high = require_finite_real(temperature_bounds[1], name="temperature_bounds[1]")
    if not 0.0 < low < high:
        raise ValueError("temperature_bounds must satisfy 0 < low < high")
    spread = _positive(erosion_spread, "erosion_spread")
    if spread >= 1.0:
        raise ValueError("erosion_spread must be below 1")
    deviations = {
        "burn_rate_cov": burn_rate_cov,
        "exponent_sd": exponent_sd,
        "c_star_cov": c_star_cov,
        "density_cov": density_cov,
        "sensitivity_cov": sensitivity_cov,
        "throat_cov": throat_cov,
        "burst_cov": burst_cov,
    }
    for name, value in deviations.items():
        _positive(value, name)
    inputs = JointInput.independent(
        RandomInput(
            "a0",
            LogNormal(math.log(nominal.burn_rate_coefficient), burn_rate_cov),
            "burn-rate coefficient at p_ref, T_ref",
            "m/s",
        ),
        RandomInput(
            "n",
            TruncatedNormal(
                nominal.pressure_exponent, exponent_sd, lower=0.0, upper=0.95
            ),
            "burn-rate pressure exponent",
        ),
        RandomInput(
            "c_star",
            Normal(
                nominal.characteristic_velocity,
                c_star_cov * nominal.characteristic_velocity,
            ),
            "characteristic velocity",
            "m/s",
        ),
        RandomInput(
            "rho_p",
            Normal(
                nominal.propellant_density, density_cov * nominal.propellant_density
            ),
            "propellant density",
            "kg/m3",
        ),
        RandomInput(
            "sigma_p",
            TruncatedNormal(
                nominal.temperature_sensitivity,
                sensitivity_cov * nominal.temperature_sensitivity,
                lower=0.0,
            ),
            "temperature sensitivity of the burn rate",
            "1/K",
        ),
        RandomInput("T_grain", Uniform(low, high), "initial grain temperature", "K"),
        RandomInput(
            "d_throat",
            Normal(nominal.throat_diameter, throat_cov * nominal.throat_diameter),
            "initial throat diameter",
            "m",
        ),
        RandomInput(
            "erosion_rate",
            Uniform(
                (1.0 - spread) * nominal.erosion_rate,
                (1.0 + spread) * nominal.erosion_rate,
            ),
            "throat radius erosion rate",
            "m/s",
        ),
        RandomInput(
            "p_burst", LogNormal(math.log(burst), burst_cov), "burst pressure", "Pa"
        ),
    )
    quantities = (
        QuantityOfInterest("meop", "peak chamber pressure of the burn", "Pa"),
        QuantityOfInterest("total_impulse", "integral of thrust over the burn", "N s"),
        QuantityOfInterest("burn_time", "time of web burn-out", "s"),
        QuantityOfInterest(
            "burst_margin", "burst pressure minus peak chamber pressure", "Pa"
        ),
    )
    return ProbabilisticModel(
        inputs,
        _SolidMotorForward(grain, nominal, mode, steps),
        quantities,
        name="solid_motor",
    )
