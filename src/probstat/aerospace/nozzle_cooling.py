"""Regenerative cooling and wall-temperature uncertainty of a thrust chamber.

A one-dimensional (axial) steady heat-transfer model of a regeneratively
cooled liquid-rocket chamber and nozzle, built so that the library's
propagation, sensitivity, reliability and surrogate tools apply to it. All
quantities are SI (Pa, K, m, m^2, kg/s, W, W/m^2, W/(m K), Pa s, J/(kg K)).

Physics, station by station along the nozzle axis ``x`` (injector face at the
smallest ``x``, nozzle exit at the largest):

- **Local flow.** The area ratio ``A/A_t = (r/r_t)^2`` of a given contour fixes
  the local Mach number by the isentropic area-Mach relation, on the
  subsonic branch upstream of the throat and the supersonic branch
  downstream (the supersonic root reuses
  :func:`~probstat.aerospace.liquid_engine.exit_mach_from_area_ratio`).
- **Gas side.** The Bartz correlation [EQ-AERO-COOL-001] with the
  boundary-layer property-variation factor ``sigma`` [EQ-AERO-COOL-002] gives
  the film coefficient ``h_g``; the driving temperature is the adiabatic-wall
  (recovery) temperature ``T_aw`` [EQ-AERO-COOL-003].
- **Wall and coolant side.** Plane-wall conduction ``t / k_w`` and the
  Dittus-Boelter coolant coefficient [EQ-AERO-COOL-005] in a channel of
  hydraulic diameter ``D_h`` close the series thermal-resistance network, which
  is solved exactly at each station (a bracketed, safeguarded Newton method
  on the gas-side wall temperature, because ``sigma`` depends on it)
  [EQ-AERO-COOL-004].
- **Coolant marching.** The coolant bulk temperature follows from the energy
  balance ``m_c c_p dT_b = q' ds`` [EQ-AERO-COOL-006]. It is advanced with an
  *implicit trapezoidal* rule, solved by fixed-point iteration, so the heat
  absorbed by the coolant equals the trapezoidal integral of the wall heat
  flux times the perimeter to the iteration tolerance, by construction.
  ``flow="counter"`` (default) injects the coolant at the nozzle exit and
  flows toward the injector, the usual regenerative arrangement;
  ``flow="co"`` injects it at the injector face.

Probabilistic use: :func:`nozzle_cooling_model` returns a
:class:`~probstat.modeling.ProbabilisticModel` whose uncertain inputs are the
Bartz and coolant-side multipliers (correlation uncertainty), wall
conductivity and thickness, coolant mass flow, chamber pressure, ``c*`` and
the coolant inlet temperature. Outputs are the peak gas-side wall
temperature and its axial location, the coolant outlet temperature, the total
heat load, the margin to a wall-temperature limit, the peak heat flux and the
coolant-side wall margin (the boiling or coking flag).

Validity and limits. Quasi-steady, axisymmetric, frozen-composition gas with
constant stagnation properties; plane wall referred to the gas-side area
(curvature error of order ``t / (2 r)``); a lumped coolant-side area
enhancement with unit fin efficiency (optimistic); constant coolant
properties (no temperature dependence of ``mu``, ``k`` or ``c_p``); no
coolant pressure drop, no radiation, no film or curvature corrections to the
coolant side, no two-phase heat transfer, no thermal-barrier coating. The
Dittus-Boelter correlation is flagged invalid below ``Re = 1e4`` or outside
``0.6 <= Pr <= 160``. The Bartz correlation is a correlation: its uncertainty
(typically tens of percent) is the reason for the multiplier inputs. The cases
in this library are synthetic and order-of-magnitude only (a regeneratively
cooled kerosene-class chamber); they are not data from any real engine.

Experimental (added in 0.28).

References:
    [REF-BARTZ-1957], [REF-BERGMAN-2011], [REF-SUTTON-BIBLARZ-2017].

"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from probstat.aerospace.liquid_engine import (
    UNIVERSAL_GAS_CONSTANT,
    area_ratio_from_mach,
    exit_mach_from_area_ratio,
)
from probstat.foundations.validation import require_finite_real
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "BARTZ_COEFFICIENT",
    "DITTUS_BOELTER_COEFFICIENT",
    "ChannelGeometry",
    "CoolantProperties",
    "CoolingSolution",
    "GasProperties",
    "NozzleContour",
    "StationProfile",
    "WallCorrection",
    "WallNetworkSolution",
    "adiabatic_wall_temperature",
    "bartz_base_coefficient",
    "bartz_heat_transfer_coefficient",
    "bartz_property_correction",
    "conical_contour",
    "coolant_side_coefficient",
    "local_mach",
    "nozzle_cooling_model",
    "solve_regenerative_cooling",
    "solve_wall_network",
    "station_profile",
]

BARTZ_COEFFICIENT = 0.026
"""Leading constant of the Bartz correlation (dimensionless)."""

DITTUS_BOELTER_COEFFICIENT = 0.023
"""Leading constant of the Dittus-Boelter correlation (dimensionless)."""


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


def _integer(value: object, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``.

    Args:
        value: Candidate integer (booleans rejected).
        name: Argument name for messages.
        minimum: Smallest allowed value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < minimum``.

    References:
        [REF-PYTHON-DATAMODEL]. A software validator; no equation.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


# ---------------------------------------------------------------------------
# Property containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GasProperties:
    """Combustion-gas properties at the chamber stagnation state.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        molecular_weight: Mean molecular weight (kg/kmol), ``> 0``.
        chamber_temperature: Stagnation temperature ``T_c`` (K), ``> 0``.
        viscosity: Dynamic viscosity ``mu`` at stagnation (Pa s), ``> 0``.
        prandtl: Prandtl number; ``None`` selects the Eucken estimate
            ``Pr = 4 gamma / (9 gamma - 5)``.
        recovery_factor: Recovery factor ``r``; ``None`` selects the
            turbulent value ``Pr^(1/3)``.
        viscosity_exponent: ``omega`` in ``mu ~ T^omega`` used by the Bartz
            property-variation factor (default 0.6).

    Attributes:
        specific_heat: ``c_p = gamma R / (gamma - 1)`` with ``R = R_u / M``
            (J/(kg K)).
        prandtl_number: The Prandtl number in use.
        recovery: The recovery factor in use.

    References:
        [EQ-AERO-COOL-001], [EQ-AERO-COOL-003], [REF-BARTZ-1957],
        [REF-SUTTON-BIBLARZ-2017].

    """

    gamma: float
    molecular_weight: float
    chamber_temperature: float
    viscosity: float
    prandtl: float | None = None
    recovery_factor: float | None = None
    viscosity_exponent: float = 0.6

    def __post_init__(self) -> None:
        """Validate the gas properties.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a value is outside its domain.

        References:
            [EQ-AERO-COOL-001], [REF-BARTZ-1957].

        """
        gamma = require_finite_real(self.gamma, name="gamma")
        if gamma <= 1.0:
            raise ValueError("gamma must exceed 1")
        object.__setattr__(self, "gamma", gamma)
        for name in ("molecular_weight", "chamber_temperature", "viscosity"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.prandtl is not None:
            object.__setattr__(self, "prandtl", _positive(self.prandtl, "prandtl"))
        if self.recovery_factor is not None:
            recovery = _positive(self.recovery_factor, "recovery_factor")
            if recovery > 1.2:
                raise ValueError("recovery_factor must not exceed 1.2")
            object.__setattr__(self, "recovery_factor", recovery)
        omega = require_finite_real(self.viscosity_exponent, name="viscosity_exponent")
        if not 0.0 <= omega < 4.0:
            raise ValueError("viscosity_exponent must lie in [0, 4)")
        object.__setattr__(self, "viscosity_exponent", omega)

    @property
    def specific_heat(self) -> float:
        """Return ``c_p = gamma R / (gamma - 1)`` in J/(kg K).

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        gas_constant = UNIVERSAL_GAS_CONSTANT / self.molecular_weight
        return self.gamma * gas_constant / (self.gamma - 1.0)

    @property
    def prandtl_number(self) -> float:
        """Return the Prandtl number (Eucken estimate when not given).

        References:
            [EQ-AERO-COOL-001], [REF-BARTZ-1957].

        """
        if self.prandtl is not None:
            return self.prandtl
        return 4.0 * self.gamma / (9.0 * self.gamma - 5.0)

    @property
    def recovery(self) -> float:
        """Return the recovery factor (``Pr^(1/3)`` when not given).

        References:
            [EQ-AERO-COOL-003], [REF-BERGMAN-2011].

        """
        if self.recovery_factor is not None:
            return self.recovery_factor
        return float(self.prandtl_number ** (1.0 / 3.0))


@dataclass(frozen=True, slots=True)
class CoolantProperties:
    """Constant coolant properties and the wall limit on the coolant side.

    Args:
        specific_heat: ``c_p`` (J/(kg K)), ``> 0``.
        conductivity: Thermal conductivity ``k`` (W/(m K)), ``> 0``.
        viscosity: Dynamic viscosity ``mu`` (Pa s), ``> 0``.
        max_wall_temperature: Coolant-side wall temperature (K) above which
            the boiling or coking flag is raised, ``> 0``.
        heating_exponent: Prandtl exponent ``n`` of Dittus-Boelter (0.4 for
            a heated fluid).

    Attributes:
        prandtl: ``Pr = c_p mu / k``.

    References:
        [EQ-AERO-COOL-005], [REF-BERGMAN-2011].

    """

    specific_heat: float
    conductivity: float
    viscosity: float
    max_wall_temperature: float
    heating_exponent: float = 0.4

    def __post_init__(self) -> None:
        """Validate the coolant properties.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a value is outside its domain.

        References:
            [EQ-AERO-COOL-005], [REF-BERGMAN-2011].

        """
        for name in (
            "specific_heat",
            "conductivity",
            "viscosity",
            "max_wall_temperature",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        exponent = require_finite_real(self.heating_exponent, name="heating_exponent")
        if not 0.0 <= exponent <= 1.0:
            raise ValueError("heating_exponent must lie in [0, 1]")
        object.__setattr__(self, "heating_exponent", exponent)

    @property
    def prandtl(self) -> float:
        """Return the coolant Prandtl number ``c_p mu / k``.

        References:
            [EQ-AERO-COOL-005].

        """
        return self.specific_heat * self.viscosity / self.conductivity


@dataclass(frozen=True, slots=True)
class ChannelGeometry:
    """Rectangular cooling channels of constant cross section.

    Args:
        count: Number of parallel channels, ``>= 1``.
        width: Channel width ``w`` (m), ``> 0``.
        height: Channel height ``b`` (m), ``> 0``.
        area_ratio: Lumped ratio of coolant-wetted area to gas-side area
            (fin enhancement at unit fin efficiency), ``> 0``.

    Attributes:
        flow_area: Total flow area ``N w b`` (m^2).
        hydraulic_diameter: ``D_h = 4 w b / (2 (w + b))`` (m).

    References:
        [EQ-AERO-COOL-005], [REF-BERGMAN-2011].

    """

    count: int
    width: float
    height: float
    area_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Validate the channel geometry.

        Raises:
            TypeError: If a value has the wrong type.
            ValueError: If a value is outside its domain.

        References:
            [EQ-AERO-COOL-005].

        """
        object.__setattr__(self, "count", _integer(self.count, "count", 1))
        for name in ("width", "height", "area_ratio"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    @property
    def flow_area(self) -> float:
        """Return the total flow cross-section ``N w b`` in m^2.

        References:
            [EQ-AERO-COOL-005].

        """
        return self.count * self.width * self.height

    @property
    def hydraulic_diameter(self) -> float:
        """Return the hydraulic diameter ``2 w b / (w + b)`` in m.

        References:
            [EQ-AERO-COOL-005].

        """
        return 2.0 * self.width * self.height / (self.width + self.height)


# ---------------------------------------------------------------------------
# Contour and stations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NozzleContour:
    """Axisymmetric chamber and nozzle contour as a piecewise-linear table.

    The radius must be non-increasing up to the throat (its minimum) and
    non-decreasing after it, so that every station has an unambiguous
    subsonic or supersonic branch.

    Args:
        x: Strictly increasing axial positions (m); the injector face is the
            first point.
        radius: Gas-side radius at each ``x`` (m), positive.
        curvature_radius: Wall radius of curvature at the throat used by the
            Bartz factor ``(D_t / r_c)^0.1`` (m), positive.

    Attributes:
        throat_index: Index of the minimum radius.
        throat_radius: ``r_t`` (m).
        x_throat: Axial position of the throat (m).

    References:
        [EQ-AERO-COOL-001], [EQ-AERO-COOL-007], [REF-BARTZ-1957].

    """

    x: tuple[float, ...]
    radius: tuple[float, ...]
    curvature_radius: float
    throat_index: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        """Validate the table and locate the throat.

        Raises:
            TypeError: If entries are not real numbers.
            ValueError: If the table is too short, ``x`` is not strictly
                increasing, a radius is not positive, or the contour is not
                convergent-divergent.

        References:
            [EQ-AERO-COOL-007].

        """
        xs = tuple(require_finite_real(v, name="x") for v in self.x)
        rs = tuple(_positive(v, "radius") for v in self.radius)
        if len(xs) != len(rs) or len(xs) < 5:
            raise ValueError("x and radius must have the same length, at least 5")
        if any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
            raise ValueError("x must be strictly increasing")
        index = rs.index(min(rs))
        if any(b > a for a, b in zip(rs[: index + 1], rs[1 : index + 1], strict=False)):
            raise ValueError("radius must be non-increasing up to the throat")
        if any(b < a for a, b in zip(rs[index:], rs[index + 1 :], strict=False)):
            raise ValueError("radius must be non-decreasing after the throat")
        if index in (0, len(rs) - 1):
            raise ValueError("the throat must be an interior point of the contour")
        object.__setattr__(self, "x", xs)
        object.__setattr__(self, "radius", rs)
        object.__setattr__(
            self, "curvature_radius", _positive(self.curvature_radius, "curvature")
        )
        object.__setattr__(self, "throat_index", index)

    @property
    def throat_radius(self) -> float:
        """Return the throat radius ``r_t`` in m.

        References:
            [EQ-AERO-COOL-007].

        """
        return self.radius[self.throat_index]

    @property
    def x_throat(self) -> float:
        """Return the axial position of the throat in m.

        References:
            [EQ-AERO-COOL-007].

        """
        return self.x[self.throat_index]

    def radius_at(self, position: float) -> float:
        """Return the radius at an axial position by linear interpolation.

        Args:
            position: Axial position (m) inside ``[x[0], x[-1]]``.

        Returns:
            The gas-side radius (m).

        Raises:
            ValueError: If ``position`` lies outside the contour.

        References:
            [EQ-AERO-COOL-007].

        """
        p = require_finite_real(position, name="position")
        if p < self.x[0] or p > self.x[-1]:
            raise ValueError("position lies outside the contour")
        k = min(max(bisect_right(self.x, p) - 1, 0), len(self.x) - 2)
        t = (p - self.x[k]) / (self.x[k + 1] - self.x[k])
        return self.radius[k] + t * (self.radius[k + 1] - self.radius[k])

    def stations(self, count: int) -> tuple[float, ...]:
        """Return ``count`` axial stations, uniform on each side of the throat.

        The throat is always a station; the upstream and downstream parts get
        a number of intervals proportional to their length.

        Args:
            count: Total number of stations, at least 5.

        Returns:
            Increasing axial positions (m).

        Raises:
            TypeError: If ``count`` is not an integer.
            ValueError: If ``count < 5``.

        References:
            [EQ-AERO-COOL-006].

        """
        n = _integer(count, "count", 5)
        x0, xt, x1 = self.x[0], self.x_throat, self.x[-1]
        intervals = n - 1
        up = min(max(round(intervals * (xt - x0) / (x1 - x0)), 1), intervals - 1)
        down = intervals - up
        left = [x0 + (xt - x0) * k / up for k in range(up)]
        right = [xt + (x1 - xt) * k / down for k in range(down + 1)]
        right[-1] = x1
        return tuple(left + right)


def conical_contour(
    *,
    throat_radius: float,
    chamber_radius: float,
    cylinder_length: float,
    expansion_ratio: float,
    convergence_half_angle: float = math.radians(30.0),
    divergence_half_angle: float = math.radians(15.0),
    upstream_arc_ratio: float = 1.5,
    downstream_arc_ratio: float = 0.382,
) -> NozzleContour:
    """Build a conical nozzle contour with circular throat arcs.

    The contour is a cylinder of radius ``R_ch``, a converging cone, an
    upstream throat arc of radius ``1.5 r_t``, a downstream throat arc of
    radius ``0.382 r_t`` and a diverging cone to the exit radius
    ``r_t sqrt(epsilon)``. The cones and arcs join tangentially; the corner
    between the cylinder and the converging cone is sharp (not rounded) and
    produces a slope jump in the table. The Bartz curvature radius
    is taken as the mean of the two arc radii. The table is dense enough
    (about 3300 points) that its linear interpolation differs from the exact
    contour by far less than any modelling error.

    Args:
        throat_radius: ``r_t`` (m).
        chamber_radius: Cylinder radius (m), larger than the end of the
            upstream arc.
        cylinder_length: Length of the cylinder (m), positive.
        expansion_ratio: ``A_e / A_t``, large enough that the exit radius
            exceeds the end of the downstream arc.
        convergence_half_angle: Converging cone half-angle (rad).
        divergence_half_angle: Diverging cone half-angle (rad).
        upstream_arc_ratio: Upstream arc radius over ``r_t``.
        downstream_arc_ratio: Downstream arc radius over ``r_t``.

    Returns:
        A :class:`NozzleContour` whose first point is the injector face at
        ``x = 0``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is outside its domain or the geometry
            cannot be closed.

    Equation:
        [EQ-AERO-COOL-007].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    rt = _positive(throat_radius, "throat_radius")
    rch = _positive(chamber_radius, "chamber_radius")
    length = _positive(cylinder_length, "cylinder_length")
    eps = require_finite_real(expansion_ratio, name="expansion_ratio")
    if eps <= 1.0:
        raise ValueError("expansion_ratio must exceed 1")
    th_c = _positive(convergence_half_angle, "convergence_half_angle")
    th_d = _positive(divergence_half_angle, "divergence_half_angle")
    if th_c >= math.pi / 2 or th_d >= math.pi / 2:
        raise ValueError("half angles must be below pi/2")
    ru = _positive(upstream_arc_ratio, "upstream_arc_ratio") * rt
    rd = _positive(downstream_arc_ratio, "downstream_arc_ratio") * rt
    r_a = rt + ru * (1.0 - math.cos(th_c))
    if rch <= r_a:
        raise ValueError("chamber_radius must exceed the end of the upstream arc")
    r_c = rt + rd * (1.0 - math.cos(th_d))
    r_e = rt * math.sqrt(eps)
    if r_e <= r_c:
        raise ValueError("expansion_ratio is too small for the downstream arc")
    pts: list[tuple[float, float]] = []

    def add(
        count: int, fx: Callable[[float], float], fr: Callable[[float], float]
    ) -> None:
        """Append ``count`` samples of a parametrised segment.

        Args:
            count: Number of samples (parameter ``s`` runs over ``[0, 1]``).
            fx: Callable ``s -> x``.
            fr: Callable ``s -> r``.

        References:
            [EQ-AERO-COOL-007].

        """
        for k in range(count):
            s = k / count
            pts.append((fx(s), fr(s)))

    x_a = -ru * math.sin(th_c)
    x_b = x_a - (rch - r_a) / math.tan(th_c)
    x_0 = x_b - length
    add(400, lambda s: x_0 + s * length, lambda s: rch)
    add(600, lambda s: x_b + s * (x_a - x_b), lambda s: rch + s * (r_a - rch))
    add(
        400,
        lambda s: -ru * math.sin(th_c * (1.0 - s)),
        lambda s: rt + ru * (1.0 - math.cos(th_c * (1.0 - s))),
    )
    add(
        400,
        lambda s: rd * math.sin(th_d * s),
        lambda s: rt + rd * (1.0 - math.cos(th_d * s)),
    )
    x_c = rd * math.sin(th_d)
    x_e = x_c + (r_e - r_c) / math.tan(th_d)
    add(1500, lambda s: x_c + s * (x_e - x_c), lambda s: r_c + s * (r_e - r_c))
    pts.append((x_e, r_e))
    xs = tuple(p[0] - x_0 for p in pts)
    rs = tuple(p[1] for p in pts)
    return NozzleContour(xs, rs, 0.5 * (ru + rd))


def local_mach(gamma: float, area_ratio: float, *, supersonic: bool) -> float:
    """Return the isentropic Mach number at a local area ratio ``A/A_t``.

    Supersonic branch: the bracketed Newton root of
    :func:`~probstat.aerospace.liquid_engine.exit_mach_from_area_ratio`.
    Subsonic branch: bisection on ``M in (0, 1]`` of the strictly decreasing
    ``A/A*(M)``.

    Args:
        gamma: Ratio of specific heats, ``> 1``.
        area_ratio: ``A / A_t >= 1``.
        supersonic: Choose the supersonic (``True``) or subsonic root.

    Returns:
        The Mach number; exactly 1 at ``A / A_t = 1``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``area_ratio < 1`` or ``gamma <= 1``.
        RuntimeError: If the supersonic solve does not converge.

    Equation:
        [EQ-AERO-COOL-007].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    g = require_finite_real(gamma, name="gamma")
    if g <= 1.0:
        raise ValueError("gamma must exceed 1")
    ratio = require_finite_real(area_ratio, name="area_ratio")
    if ratio < 1.0:
        raise ValueError("area_ratio must be at least 1")
    if ratio == 1.0:
        return 1.0
    if supersonic:
        solution = exit_mach_from_area_ratio(g, ratio)
        if not solution.converged:
            raise RuntimeError("supersonic area-Mach solve did not converge")
        return solution.mach
    low, high = 1e-9, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if area_ratio_from_mach(g, mid) > ratio:
            low = mid
        else:
            high = mid
        if high - low <= 1e-15 * high:
            break
    return 0.5 * (low + high)


@dataclass(frozen=True, slots=True)
class StationProfile:
    """Axial stations with local geometry and Mach number.

    Args:
        contour: The contour the stations were taken from.
        gamma: Ratio of specific heats used for the Mach numbers.
        x: Station positions (m), increasing.
        radius: Gas-side radius at each station (m).
        area_ratio: ``A / A_t = (r / r_t)^2`` at each station.
        mach: Local Mach number at each station.
        segment_length: Chord length ``ds`` between consecutive stations (m);
            one entry fewer than stations.
        throat_index: Index of the throat station.

    References:
        [EQ-AERO-COOL-007].

    """

    contour: NozzleContour
    gamma: float
    x: tuple[float, ...]
    radius: tuple[float, ...]
    area_ratio: tuple[float, ...]
    mach: tuple[float, ...]
    segment_length: tuple[float, ...]
    throat_index: int


def station_profile(
    contour: NozzleContour, gamma: float, count: int = 81
) -> StationProfile:
    """Place stations on a contour and compute the local Mach numbers.

    Args:
        contour: The nozzle contour.
        gamma: Ratio of specific heats, ``> 1``.
        count: Number of stations (at least 5).

    Returns:
        A :class:`StationProfile` with the throat as a station.

    Raises:
        TypeError: If ``contour`` is not a :class:`NozzleContour`.
        ValueError: If ``gamma`` or ``count`` is invalid.

    Equation:
        [EQ-AERO-COOL-007].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(contour, NozzleContour):
        raise TypeError("contour must be a NozzleContour")
    xs = contour.stations(count)
    rs = tuple(contour.radius_at(v) for v in xs)
    rt = contour.throat_radius
    xt = contour.x_throat
    ratios = tuple(max((r / rt) ** 2, 1.0) for r in rs)
    machs = tuple(
        local_mach(gamma, a, supersonic=v > xt) for a, v in zip(ratios, xs, strict=True)
    )
    ds = tuple(
        math.hypot(xs[i + 1] - xs[i], rs[i + 1] - rs[i]) for i in range(len(xs) - 1)
    )
    return StationProfile(contour, gamma, xs, rs, ratios, machs, ds, xs.index(xt))


# ---------------------------------------------------------------------------
# Gas-side relations
# ---------------------------------------------------------------------------


def adiabatic_wall_temperature(gas: GasProperties, mach: float) -> float:
    """Return the adiabatic-wall (recovery) temperature ``T_aw``.

    ``T_aw = T_c (1 + r (gamma - 1) M^2 / 2) / (1 + (gamma - 1) M^2 / 2)``;
    it equals ``T_c`` at ``M = 0`` and decreases toward ``r T_c``-like values
    as the flow accelerates.

    Args:
        gas: Gas properties (supplies ``T_c``, ``gamma`` and ``r``).
        mach: Local Mach number, ``>= 0``.

    Returns:
        ``T_aw`` in K.

    Raises:
        TypeError: If ``gas`` has the wrong type or ``mach`` is not real.
        ValueError: If ``mach < 0``.

    Equation:
        [EQ-AERO-COOL-003].

    References:
        [REF-BARTZ-1957], [REF-BERGMAN-2011].

    """
    if not isinstance(gas, GasProperties):
        raise TypeError("gas must be a GasProperties")
    m = require_finite_real(mach, name="mach")
    if m < 0.0:
        raise ValueError("mach must be non-negative")
    z = 1.0 + 0.5 * (gas.gamma - 1.0) * m * m
    return gas.chamber_temperature * (1.0 + (z - 1.0) * gas.recovery) / z


def bartz_base_coefficient(
    *,
    gas: GasProperties,
    chamber_pressure: float,
    characteristic_velocity: float,
    throat_diameter: float,
    curvature_radius: float,
    area_ratio: float,
) -> float:
    """Return the Bartz coefficient without the property factor, ``h_g / sigma``.

    ``h_g/sigma = [0.026 / D_t^0.2 (mu^0.2 c_p / Pr^0.6)
    (p_c / c*)^0.8 (D_t / r_c)^0.1] (A_t / A)^0.9`` in SI units: ``p_c`` in
    Pa and ``c*`` in m/s, so that ``p_c / c* = mdot / A_t`` is a mass flux
    (kg/(m^2 s)); the constant 0.026 is then dimensionless and the result is
    in W/(m^2 K). With the curvature and area factors removed this is the
    pipe-flow form ``0.026 (k/D) Re^0.8 Pr^0.4`` evaluated at the throat mass
    flux, which fixes the units. Gas properties are stagnation values.

    Args:
        gas: Gas properties (``mu``, ``c_p``, ``Pr``).
        chamber_pressure: ``p_c`` (Pa), ``> 0``.
        characteristic_velocity: ``c*`` (m/s), ``> 0``.
        throat_diameter: ``D_t`` (m), ``> 0``.
        curvature_radius: Throat wall radius of curvature ``r_c`` (m).
        area_ratio: Local ``A / A_t >= 1``.

    Returns:
        ``h_g / sigma`` in W/(m^2 K).

    Raises:
        TypeError: If ``gas`` has the wrong type or a number is not real.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-COOL-001].

    References:
        [REF-BARTZ-1957], [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(gas, GasProperties):
        raise TypeError("gas must be a GasProperties")
    pc = _positive(chamber_pressure, "chamber_pressure")
    cstar = _positive(characteristic_velocity, "characteristic_velocity")
    dt = _positive(throat_diameter, "throat_diameter")
    rc = _positive(curvature_radius, "curvature_radius")
    ratio = require_finite_real(area_ratio, name="area_ratio")
    if ratio < 1.0:
        raise ValueError("area_ratio must be at least 1")
    return float(
        BARTZ_COEFFICIENT
        / dt**0.2
        * gas.viscosity**0.2
        * gas.specific_heat
        / gas.prandtl_number**0.6
        * (pc / cstar) ** 0.8
        * (dt / rc) ** 0.1
        / ratio**0.9
    )


def bartz_property_correction(
    gas: GasProperties, mach: float, wall_temperature: float
) -> float:
    """Return Bartz's boundary-layer property-variation factor ``sigma``.

    ``sigma = 1 / { [ (T_w / (2 T_c)) Z + 1/2 ]^(0.8 - omega/5)
    Z^(omega/5) }`` with ``Z = 1 + (gamma - 1) M^2 / 2``. A cold wall
    (``T_w << T_c``) gives ``sigma > 1`` because the near-wall gas is denser
    than at the stagnation state.

    Args:
        gas: Gas properties (``gamma``, ``T_c``, ``omega``).
        mach: Local Mach number, ``>= 0``.
        wall_temperature: Gas-side wall temperature ``T_w`` (K), ``> 0``.

    Returns:
        The dimensionless factor ``sigma``.

    Raises:
        TypeError: If ``gas`` has the wrong type or a number is not real.
        ValueError: If ``mach < 0`` or ``wall_temperature <= 0``.

    Equation:
        [EQ-AERO-COOL-002].

    References:
        [REF-BARTZ-1957].

    """
    if not isinstance(gas, GasProperties):
        raise TypeError("gas must be a GasProperties")
    m = require_finite_real(mach, name="mach")
    if m < 0.0:
        raise ValueError("mach must be non-negative")
    tw = _positive(wall_temperature, "wall_temperature")
    z = 1.0 + 0.5 * (gas.gamma - 1.0) * m * m
    omega = gas.viscosity_exponent
    s = 0.5 * tw / gas.chamber_temperature * z + 0.5
    return float(1.0 / (s ** (0.8 - 0.2 * omega) * z ** (0.2 * omega)))


def bartz_heat_transfer_coefficient(
    *,
    gas: GasProperties,
    chamber_pressure: float,
    characteristic_velocity: float,
    throat_diameter: float,
    curvature_radius: float,
    area_ratio: float,
    mach: float,
    wall_temperature: float,
    multiplier: float = 1.0,
) -> float:
    """Return the gas-side coefficient ``h_g = m_B (h_g / sigma) sigma``.

    Args:
        gas: Gas properties.
        chamber_pressure: ``p_c`` (Pa).
        characteristic_velocity: ``c*`` (m/s).
        throat_diameter: ``D_t`` (m).
        curvature_radius: Throat curvature radius ``r_c`` (m).
        area_ratio: Local ``A / A_t >= 1``.
        mach: Local Mach number.
        wall_temperature: Gas-side wall temperature (K).
        multiplier: Dimensionless uncertainty multiplier ``m_B > 0`` of the
            correlation (1 = nominal Bartz).

    Returns:
        ``h_g`` in W/(m^2 K).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-COOL-001], [EQ-AERO-COOL-002].

    References:
        [REF-BARTZ-1957].

    """
    base = bartz_base_coefficient(
        gas=gas,
        chamber_pressure=chamber_pressure,
        characteristic_velocity=characteristic_velocity,
        throat_diameter=throat_diameter,
        curvature_radius=curvature_radius,
        area_ratio=area_ratio,
    )
    m_b = _positive(multiplier, "multiplier")
    return m_b * base * bartz_property_correction(gas, mach, wall_temperature)


# ---------------------------------------------------------------------------
# Coolant side
# ---------------------------------------------------------------------------


def coolant_side_coefficient(
    channel: ChannelGeometry,
    coolant: CoolantProperties,
    mass_flow: float,
    *,
    multiplier: float = 1.0,
) -> tuple[float, float]:
    """Return the Dittus-Boelter coolant coefficient and Reynolds number.

    ``Re = G D_h / mu`` with ``G = mdot / (N w b)``,
    ``Nu = 0.023 Re^0.8 Pr^n`` and ``h_c = Nu k / D_h``. The returned
    coefficient already includes the lumped wetted-area ratio and the
    multiplier, so it multiplies the gas-side area directly:
    ``h_eff = m_c phi h_c``.

    Args:
        channel: Channel geometry.
        coolant: Coolant properties.
        mass_flow: Total coolant mass flow ``mdot`` (kg/s), ``> 0``.
        multiplier: Uncertainty multiplier ``m_c > 0`` of the coolant side.

    Returns:
        ``(h_eff, Re)`` in W/(m^2 K) referred to the gas-side area, and the
        channel Reynolds number.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its domain.

    Equation:
        [EQ-AERO-COOL-005].

    References:
        [REF-BERGMAN-2011].

    """
    if not isinstance(channel, ChannelGeometry):
        raise TypeError("channel must be a ChannelGeometry")
    if not isinstance(coolant, CoolantProperties):
        raise TypeError("coolant must be a CoolantProperties")
    flow = _positive(mass_flow, "mass_flow")
    m_c = _positive(multiplier, "multiplier")
    dh = channel.hydraulic_diameter
    reynolds = flow / channel.flow_area * dh / coolant.viscosity
    nusselt = (
        DITTUS_BOELTER_COEFFICIENT
        * reynolds**0.8
        * coolant.prandtl**coolant.heating_exponent
    )
    h = nusselt * coolant.conductivity / dh
    return m_c * channel.area_ratio * h, reynolds


# ---------------------------------------------------------------------------
# Thermal-resistance network at one station
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WallCorrection:
    """Stagnation data needed to evaluate ``sigma`` inside the wall solve.

    Args:
        chamber_temperature: ``T_c`` (K).
        stagnation_factor: ``Z = 1 + (gamma - 1) M^2 / 2`` at the station.
        viscosity_exponent: ``omega`` of ``mu ~ T^omega``.

    References:
        [EQ-AERO-COOL-002].

    """

    chamber_temperature: float
    stagnation_factor: float
    viscosity_exponent: float = 0.6


@dataclass(frozen=True, slots=True)
class WallNetworkSolution:
    """Exact solution of the series resistance network at one station.

    Args:
        heat_flux: Heat flux ``q`` through the wall (W/m^2, gas-side area).
        gas_wall_temperature: ``T_wg`` (K).
        coolant_wall_temperature: ``T_wc`` (K).
        gas_coefficient: Effective ``h_g`` at the solution (W/(m^2 K)).
        iterations: Newton iterations (0 for the closed form).
        converged: Whether the tolerance was met.

    References:
        [EQ-AERO-COOL-004].

    """

    heat_flux: float
    gas_wall_temperature: float
    coolant_wall_temperature: float
    gas_coefficient: float
    iterations: int
    converged: bool


def _network_residual(
    t: float,
    taw: float,
    tb: float,
    base: float,
    r_series: float,
    a: float,
    m_exp: float,
    z_pow: float,
) -> tuple[float, float, float]:
    """Return ``(f, f', h)`` of the wall-temperature equation at ``t``.

    Args:
        t: Gas-side wall temperature (K).
        taw: Recovery temperature (K).
        tb: Coolant bulk temperature (K).
        base: Coefficient ``h_g / sigma`` (W/(m^2 K)).
        r_series: Wall plus coolant resistance (m^2 K/W).
        a: ``Z / (2 T_c)``.
        m_exp: Exponent ``0.8 - omega/5`` of the bracket.
        z_pow: Factor ``Z^(-omega/5)``.

    Returns:
        ``f = h (T_aw - t) - (t - T_b) / R_s``, its derivative and ``h``.

    References:
        [EQ-AERO-COOL-004].

    """
    s = a * t + 0.5
    h = base * z_pow * s ** (-m_exp)
    f = h * (taw - t) - (t - tb) / r_series
    df = -h * a * m_exp / s * (taw - t) - h - 1.0 / r_series
    return f, df, h


def solve_wall_network(
    *,
    recovery_temperature: float,
    coolant_temperature: float,
    gas_coefficient: float,
    wall_resistance: float,
    coolant_coefficient: float,
    correction: WallCorrection | None = None,
    guess: float | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 100,
) -> WallNetworkSolution:
    """Solve ``q = h_g (T_aw - T_wg) = (T_wg - T_wc)/R_w = h_c (T_wc - T_b)``.

    Without ``correction`` the gas coefficient is a constant and the network
    has the closed form ``q = (T_aw - T_b) / (1/h_g + R_w + 1/h_c)``. With a
    :class:`WallCorrection`, ``gas_coefficient`` is the Bartz coefficient
    without the property factor and ``h_g = base sigma(T_wg)`` depends on the
    unknown, so ``f(T) = h_g(T)(T_aw - T) - (T - T_b)/(R_w + 1/h_c) = 0`` is
    solved. ``f`` is strictly decreasing on ``[T_b, T_aw]`` with
    ``f(T_b) > 0 > f(T_aw)``, so the root is unique; a safeguarded Newton
    method (bisection whenever a step leaves the bracket) finds it.

    Args:
        recovery_temperature: ``T_aw`` (K).
        coolant_temperature: ``T_b`` (K), below ``T_aw``.
        gas_coefficient: ``h_g`` (constant case) or ``h_g / sigma``.
        wall_resistance: ``R_w = t / k_w`` (m^2 K/W), ``>= 0``.
        coolant_coefficient: Effective ``h_c`` (W/(m^2 K)), ``> 0``.
        correction: Property-variation data, or ``None`` for the closed form.
        guess: Starting gas-side wall temperature for Newton (K).
        tolerance: Relative tolerance on the temperature update.
        max_iterations: Iteration cap.

    Returns:
        A :class:`WallNetworkSolution`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its domain or ``T_b >= T_aw``.

    Equation:
        [EQ-AERO-COOL-004].

    References:
        [REF-BERGMAN-2011], [REF-BARTZ-1957].

    """
    taw = _positive(recovery_temperature, "recovery_temperature")
    tb = _positive(coolant_temperature, "coolant_temperature")
    base = _positive(gas_coefficient, "gas_coefficient")
    r_wall = require_finite_real(wall_resistance, name="wall_resistance")
    if r_wall < 0.0:
        raise ValueError("wall_resistance must be non-negative")
    h_c = _positive(coolant_coefficient, "coolant_coefficient")
    tol = _positive(tolerance, "tolerance")
    cap = _integer(max_iterations, "max_iterations", 1)
    if tb >= taw:
        raise ValueError("coolant_temperature must be below recovery_temperature")
    r_series = r_wall + 1.0 / h_c
    if correction is None:
        flux = (taw - tb) / (1.0 / base + r_series)
        twg = taw - flux / base
        return WallNetworkSolution(flux, twg, twg - flux * r_wall, base, 0, True)
    if not isinstance(correction, WallCorrection):
        raise TypeError("correction must be a WallCorrection or None")
    tc = _positive(correction.chamber_temperature, "chamber_temperature")
    z = _positive(correction.stagnation_factor, "stagnation_factor")
    t, h, converged, iterations = _wall_newton(
        taw, tb, base, r_series, z, tc, correction.viscosity_exponent, guess, tol, cap
    )
    flux = h * (taw - t)
    return WallNetworkSolution(flux, t, t - flux * r_wall, h, iterations, converged)


def _wall_newton(
    taw: float,
    tb: float,
    base: float,
    r_series: float,
    z: float,
    tc: float,
    omega: float,
    guess: float | None,
    tol: float,
    cap: int,
) -> tuple[float, float, bool, int]:
    """Solve the wall-temperature equation without argument validation.

    Args:
        taw: Recovery temperature (K), above ``tb``.
        tb: Coolant bulk temperature (K).
        base: Coefficient ``h_g / sigma`` (W/(m^2 K)).
        r_series: Wall plus coolant resistance (m^2 K/W).
        z: Stagnation factor ``Z``.
        tc: Chamber temperature (K).
        omega: Viscosity exponent.
        guess: Starting temperature (K) or ``None`` for the bracket midpoint.
        tol: Relative tolerance on the temperature update.
        cap: Iteration cap.

    Returns:
        ``(T_wg, h_g, converged, iterations)``.

    References:
        [EQ-AERO-COOL-004].

    """
    a = z / (2.0 * tc)
    m_exp = 0.8 - 0.2 * omega
    z_pow = z ** (-0.2 * omega)
    low, high = tb, taw
    t = 0.5 * (low + high) if guess is None else min(max(guess, low), high)
    converged = False
    iterations = 0
    for iterations in range(1, cap + 1):  # noqa: B007
        f, df, _ = _network_residual(t, taw, tb, base, r_series, a, m_exp, z_pow)
        if f > 0.0:
            low = t
        else:
            high = t
        step = t - f / df
        if abs(step - t) <= tol * taw:
            t = step
            converged = True
            break
        if not low < step < high:
            step = 0.5 * (low + high)
        t = step
    _, _, h = _network_residual(t, taw, tb, base, r_series, a, m_exp, z_pow)
    return t, h, converged, iterations


# ---------------------------------------------------------------------------
# Marching solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoolingSolution:
    """Axial solution of the regenerative-cooling problem.

    Per-station tuples follow the order of ``x`` (increasing, injector to
    exit), whatever the coolant direction.

    Args:
        x: Station positions (m).
        mach: Local Mach number.
        recovery_temperature: ``T_aw`` (K).
        gas_coefficient: Effective ``h_g`` at the wall temperature (W/(m^2 K)).
        heat_flux: Wall heat flux ``q`` (W/m^2).
        gas_wall_temperature: ``T_wg`` (K).
        coolant_wall_temperature: ``T_wc`` (K).
        coolant_temperature: Bulk coolant temperature ``T_b`` (K).
        linear_heat_rate: ``q' = 2 pi r q`` (W per metre of arc length).
        coolant_coefficient: Effective ``h_c`` (W/(m^2 K)).
        coolant_reynolds: Channel Reynolds number.
        flow: ``"counter"`` or ``"co"``.
        total_heat_load: Trapezoidal integral of ``q'`` over arc length (W).
        coolant_heat_gain: ``mdot c_p (T_out - T_in)`` (W).
        coolant_outlet_temperature: ``T_out`` (K).
        peak_wall_temperature: Maximum ``T_wg`` over the stations (K).
        peak_wall_index: Station index of that maximum.
        peak_heat_flux: Maximum ``q`` over the stations (W/m^2).
        peak_heat_flux_index: Station index of that maximum.
        max_coolant_wall_temperature: Maximum ``T_wc`` (K).
        boiling_flag: ``max T_wc`` exceeds the coolant wall limit.
        correlation_valid: Dittus-Boelter validity (``Re >= 1e4`` and
            ``0.6 <= Pr <= 160``).
        throat_index: Throat station index.
        converged: Every wall solve and every marching step converged.
        stop_reason: ``"converged"`` or a description of the failure.

    References:
        [EQ-AERO-COOL-004], [EQ-AERO-COOL-006], [EQ-AERO-COOL-008].

    """

    x: tuple[float, ...]
    mach: tuple[float, ...]
    recovery_temperature: tuple[float, ...]
    gas_coefficient: tuple[float, ...]
    heat_flux: tuple[float, ...]
    gas_wall_temperature: tuple[float, ...]
    coolant_wall_temperature: tuple[float, ...]
    coolant_temperature: tuple[float, ...]
    linear_heat_rate: tuple[float, ...]
    coolant_coefficient: float
    coolant_reynolds: float
    flow: str
    total_heat_load: float
    coolant_heat_gain: float
    coolant_outlet_temperature: float
    peak_wall_temperature: float
    peak_wall_index: int
    peak_heat_flux: float
    peak_heat_flux_index: int
    max_coolant_wall_temperature: float
    boiling_flag: bool
    correlation_valid: bool
    throat_index: int
    converged: bool
    stop_reason: str


def solve_regenerative_cooling(
    profile: StationProfile,
    *,
    gas: GasProperties,
    coolant: CoolantProperties,
    channel: ChannelGeometry,
    chamber_pressure: float,
    characteristic_velocity: float,
    coolant_mass_flow: float,
    coolant_inlet_temperature: float,
    wall_conductivity: float,
    wall_thickness: float,
    bartz_multiplier: float = 1.0,
    coolant_multiplier: float = 1.0,
    flow: Literal["counter", "co"] = "counter",
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> CoolingSolution:
    """March the coolant along the chamber and solve every station exactly.

    At each station the wall network [EQ-AERO-COOL-004] is solved for the
    local bulk temperature ``T_b``. Between consecutive stations ``j`` (upstream
    in the coolant direction) and ``k`` the bulk temperature obeys the
    implicit trapezoidal rule ``T_b,k = T_b,j + ds (q'_j + q'_k(T_b,k)) /
    (2 mdot c_p)`` [EQ-AERO-COOL-006], solved by fixed-point iteration (a
    contraction because ``q'`` decreases weakly with ``T_b``). Consequently
    ``mdot c_p (T_out - T_in)`` equals the trapezoidal integral of ``q'``
    over arc length to the iteration tolerance.

    Args:
        profile: Stations from :func:`station_profile`.
        gas: Gas properties.
        coolant: Coolant properties.
        channel: Channel geometry.
        chamber_pressure: ``p_c`` (Pa).
        characteristic_velocity: ``c*`` (m/s).
        coolant_mass_flow: Total coolant flow (kg/s).
        coolant_inlet_temperature: ``T_in`` (K).
        wall_conductivity: ``k_w`` (W/(m K)).
        wall_thickness: ``t`` (m).
        bartz_multiplier: Multiplier on the Bartz coefficient.
        coolant_multiplier: Multiplier on the coolant coefficient.
        flow: ``"counter"``: coolant enters at the nozzle exit and flows
            toward the injector; ``"co"``: enters at the injector face.
        tolerance: Absolute tolerance (K, relative to ``max(1, T)``) of the
            marching fixed point.
        max_iterations: Cap on fixed-point iterations per step.

    Returns:
        A :class:`CoolingSolution`; ``converged`` is False if a wall solve
        or a marching step hit its cap.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is outside its domain or the coolant
            reaches the recovery temperature.

    Equation:
        [EQ-AERO-COOL-004], [EQ-AERO-COOL-006].

    References:
        [REF-BARTZ-1957], [REF-BERGMAN-2011].

    """
    if not isinstance(profile, StationProfile):
        raise TypeError("profile must be a StationProfile")
    if not isinstance(gas, GasProperties):
        raise TypeError("gas must be a GasProperties")
    if flow not in ("counter", "co"):
        raise ValueError("flow must be 'counter' or 'co'")
    flow_rate = _positive(coolant_mass_flow, "coolant_mass_flow")
    t_in = _positive(coolant_inlet_temperature, "coolant_inlet_temperature")
    k_w = _positive(wall_conductivity, "wall_conductivity")
    thick = _positive(wall_thickness, "wall_thickness")
    m_b = _positive(bartz_multiplier, "bartz_multiplier")
    tol = _positive(tolerance, "tolerance")
    cap = _integer(max_iterations, "max_iterations", 1)
    pc = _positive(chamber_pressure, "chamber_pressure")
    cstar = _positive(characteristic_velocity, "characteristic_velocity")
    h_c, reynolds = coolant_side_coefficient(
        channel, coolant, flow_rate, multiplier=coolant_multiplier
    )
    r_wall = thick / k_w
    r_series = r_wall + 1.0 / h_c
    contour = profile.contour
    dt = 2.0 * contour.throat_radius
    n = len(profile.x)
    mcp = flow_rate * coolant.specific_heat
    omega = gas.viscosity_exponent
    tc = gas.chamber_temperature
    # Per-station constants.
    zs = [1.0 + 0.5 * (gas.gamma - 1.0) * m * m for m in profile.mach]
    taws = [tc * (1.0 + (z - 1.0) * gas.recovery) / z for z in zs]
    bases = [
        m_b
        * bartz_base_coefficient(
            gas=gas,
            chamber_pressure=pc,
            characteristic_velocity=cstar,
            throat_diameter=dt,
            curvature_radius=contour.curvature_radius,
            area_ratio=a,
        )
        for a in profile.area_ratio
    ]
    perimeters = [2.0 * math.pi * r for r in profile.radius]
    twg = [0.0] * n
    twc = [0.0] * n
    tb = [0.0] * n
    flux = [0.0] * n
    hg = [0.0] * n
    ok = True
    reason = "converged"

    def station(i: int, bulk: float) -> float:
        """Solve station ``i`` for the bulk temperature and return ``q'``.

        Args:
            i: Station index.
            bulk: Coolant bulk temperature at the station (K).

        Returns:
            The linear heat rate ``q' = 2 pi r q`` (W/m).

        Raises:
            ValueError: If the coolant is not colder than the recovery
                temperature.

        References:
            [EQ-AERO-COOL-004].

        """
        nonlocal ok, reason
        if bulk >= taws[i]:
            raise ValueError("coolant reached the recovery temperature")
        t, h, done, _ = _wall_newton(
            taws[i],
            bulk,
            bases[i],
            r_series,
            zs[i],
            tc,
            omega,
            twg[i] if twg[i] > 0.0 else None,
            1e-12,
            100,
        )
        if not done and ok:
            ok = False
            reason = "wall_network_not_converged"
        q = h * (taws[i] - t)
        twg[i] = t
        twc[i] = t - q * r_wall
        flux[i] = q
        hg[i] = h
        tb[i] = bulk
        return q * perimeters[i]

    order = list(range(n - 1, -1, -1)) if flow == "counter" else list(range(n))
    rate_prev = station(order[0], t_in)
    total = 0.0
    for j, k in zip(order, order[1:], strict=False):
        ds = profile.segment_length[min(j, k)]
        c = 0.5 * ds / mcp
        guess = tb[j] + 2.0 * c * rate_prev
        converged_step = False
        for _ in range(cap):
            rate = station(k, guess)
            new = tb[j] + c * (rate_prev + rate)
            done = abs(new - guess) <= tol * max(1.0, abs(new))
            guess = new
            if done:
                converged_step = True
                break
        tb[k] = guess
        if not converged_step and ok:
            ok = False
            reason = "marching_step_not_converged"
        total += 0.5 * ds * (rate_prev + rate)
        rate_prev = rate
    out_index = order[-1]
    t_out = tb[out_index]
    gain = mcp * (t_out - t_in)
    heat_rate = tuple(f * p for f, p in zip(flux, perimeters, strict=True))
    peak_i = max(range(n), key=lambda i: twg[i])
    flux_i = max(range(n), key=lambda i: flux[i])
    pr = coolant.prandtl
    return CoolingSolution(
        x=profile.x,
        mach=profile.mach,
        recovery_temperature=tuple(taws),
        gas_coefficient=tuple(hg),
        heat_flux=tuple(flux),
        gas_wall_temperature=tuple(twg),
        coolant_wall_temperature=tuple(twc),
        coolant_temperature=tuple(tb),
        linear_heat_rate=heat_rate,
        coolant_coefficient=h_c,
        coolant_reynolds=reynolds,
        flow=flow,
        total_heat_load=total,
        coolant_heat_gain=gain,
        coolant_outlet_temperature=t_out,
        peak_wall_temperature=twg[peak_i],
        peak_wall_index=peak_i,
        peak_heat_flux=flux[flux_i],
        peak_heat_flux_index=flux_i,
        max_coolant_wall_temperature=max(twc),
        boiling_flag=max(twc) > coolant.max_wall_temperature,
        correlation_valid=reynolds >= 1e4 and 0.6 <= pr <= 160.0,
        throat_index=profile.throat_index,
        converged=ok,
        stop_reason=reason,
    )


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_UNITS = {
    "bartz_multiplier": "-",
    "coolant_multiplier": "-",
    "wall_conductivity": "W/(m K)",
    "wall_thickness": "m",
    "coolant_mass_flow": "kg/s",
    "chamber_pressure": "Pa",
    "characteristic_velocity": "m/s",
    "coolant_inlet_temperature": "K",
}

_QUANTITIES = (
    QuantityOfInterest("peak_wall_temperature", "Peak gas-side wall temperature", "K"),
    QuantityOfInterest(
        "peak_wall_location", "Axial position of the peak wall temperature", "m"
    ),
    QuantityOfInterest("coolant_outlet_temperature", "Coolant outlet temperature", "K"),
    QuantityOfInterest("total_heat_load", "Total heat absorbed by the coolant", "W"),
    QuantityOfInterest(
        "wall_temperature_margin",
        "Wall-temperature limit minus peak gas-side wall temperature",
        "K",
    ),
    QuantityOfInterest("peak_heat_flux", "Peak wall heat flux", "W/m^2"),
    QuantityOfInterest(
        "coolant_wall_margin",
        "Coolant-side wall limit minus peak coolant-side wall temperature "
        "(negative: boiling or coking flag)",
        "K",
    ),
)


@dataclass(frozen=True, slots=True)
class _CoolingForward:
    """Picklable forward map of :func:`nozzle_cooling_model`.

    Args:
        profile: Precomputed stations and Mach numbers.
        gas: Gas properties.
        coolant: Coolant properties.
        channel: Channel geometry.
        wall_limit: Gas-side wall-temperature limit (K).
        flow: Coolant direction.

    References:
        [EQ-AERO-COOL-009].

    """

    profile: StationProfile
    gas: GasProperties
    coolant: CoolantProperties
    channel: ChannelGeometry
    wall_limit: float
    flow: Literal["counter", "co"]

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the quantities of interest for one realization.

        Args:
            parameters: Values of the eight model inputs.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If an input is non-physical (non-positive).
            RuntimeError: If a wall solve or marching step did not converge.

        Equation:
            [EQ-AERO-COOL-009].

        References:
            [REF-BARTZ-1957].

        """
        sol = solve_regenerative_cooling(
            self.profile,
            gas=self.gas,
            coolant=self.coolant,
            channel=self.channel,
            chamber_pressure=parameters["chamber_pressure"],
            characteristic_velocity=parameters["characteristic_velocity"],
            coolant_mass_flow=parameters["coolant_mass_flow"],
            coolant_inlet_temperature=parameters["coolant_inlet_temperature"],
            wall_conductivity=parameters["wall_conductivity"],
            wall_thickness=parameters["wall_thickness"],
            bartz_multiplier=parameters["bartz_multiplier"],
            coolant_multiplier=parameters["coolant_multiplier"],
            flow=self.flow,
        )
        if not sol.converged:
            raise RuntimeError(f"cooling solve failed: {sol.stop_reason}")
        return {
            "peak_wall_temperature": sol.peak_wall_temperature,
            "peak_wall_location": sol.x[sol.peak_wall_index],
            "coolant_outlet_temperature": sol.coolant_outlet_temperature,
            "total_heat_load": sol.total_heat_load,
            "wall_temperature_margin": self.wall_limit - sol.peak_wall_temperature,
            "peak_heat_flux": sol.peak_heat_flux,
            "coolant_wall_margin": (
                self.coolant.max_wall_temperature - sol.max_coolant_wall_temperature
            ),
        }


def nozzle_cooling_model(
    *,
    profile: StationProfile,
    gas: GasProperties,
    coolant: CoolantProperties,
    channel: ChannelGeometry,
    chamber_pressure: Marginal | float,
    characteristic_velocity: Marginal | float,
    coolant_mass_flow: Marginal | float,
    coolant_inlet_temperature: Marginal | float,
    wall_conductivity: Marginal | float,
    wall_thickness: Marginal | float,
    bartz_multiplier: Marginal | float = 1.0,
    coolant_multiplier: Marginal | float = 1.0,
    wall_temperature_limit: float = 900.0,
    flow: Literal["counter", "co"] = "counter",
    name: str = "nozzle-cooling",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a probabilistic regenerative-cooling model.

    Each uncertain argument is a distribution (wrapped as an independent
    :class:`~probstat.modeling.RandomInput`) or a plain number (fixed).
    Units: ``chamber_pressure`` Pa, ``characteristic_velocity`` m/s,
    ``coolant_mass_flow`` kg/s, ``coolant_inlet_temperature`` K,
    ``wall_conductivity`` W/(m K), ``wall_thickness`` m; the two multipliers
    are dimensionless and model the relative uncertainty of the Bartz and
    Dittus-Boelter correlations. Quantities of interest:
    ``peak_wall_temperature`` (K), ``peak_wall_location`` (m),
    ``coolant_outlet_temperature`` (K), ``total_heat_load`` (W),
    ``wall_temperature_margin`` (K, ``limit - peak``), ``peak_heat_flux``
    (W/m^2) and ``coolant_wall_margin`` (K; negative means the boiling or
    coking flag). A non-physical draw (for example a negative thickness) or a
    solver failure fails the run; ``on_failure`` decides what then happens, so
    use bounded laws. Independence between inputs is assumed. Gas, coolant,
    channel and contour are fixed.

    Args:
        profile: Stations and Mach numbers from :func:`station_profile`.
        gas: Gas properties.
        coolant: Coolant properties.
        channel: Channel geometry.
        chamber_pressure: ``p_c`` (Pa).
        characteristic_velocity: ``c*`` (m/s).
        coolant_mass_flow: Total coolant mass flow (kg/s).
        coolant_inlet_temperature: ``T_in`` (K).
        wall_conductivity: ``k_w`` (W/(m K)).
        wall_thickness: ``t`` (m).
        bartz_multiplier: Bartz uncertainty multiplier (default 1).
        coolant_multiplier: Coolant-side multiplier (default 1).
        wall_temperature_limit: Admissible gas-side wall temperature (K).
        flow: ``"counter"`` or ``"co"``.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is outside its domain or no input is
            random.

    Equation:
        [EQ-AERO-COOL-009].

    References:
        [REF-BARTZ-1957], [REF-BERGMAN-2011].

    """
    if not isinstance(profile, StationProfile):
        raise TypeError("profile must be a StationProfile")
    if not isinstance(gas, GasProperties):
        raise TypeError("gas must be a GasProperties")
    if not isinstance(coolant, CoolantProperties):
        raise TypeError("coolant must be a CoolantProperties")
    if not isinstance(channel, ChannelGeometry):
        raise TypeError("channel must be a ChannelGeometry")
    if flow not in ("counter", "co"):
        raise ValueError("flow must be 'counter' or 'co'")
    limit = _positive(wall_temperature_limit, "wall_temperature_limit")
    given: dict[str, Marginal | float] = {
        "bartz_multiplier": bartz_multiplier,
        "coolant_multiplier": coolant_multiplier,
        "wall_conductivity": wall_conductivity,
        "wall_thickness": wall_thickness,
        "coolant_mass_flow": coolant_mass_flow,
        "chamber_pressure": chamber_pressure,
        "characteristic_velocity": characteristic_velocity,
        "coolant_inlet_temperature": coolant_inlet_temperature,
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, value in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = _positive(value, key)
        else:
            random_inputs.append(RandomInput(key, value, unit=_UNITS[key]))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        _CoolingForward(profile, gas, coolant, channel, limit, flow),
        _QUANTITIES,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )
