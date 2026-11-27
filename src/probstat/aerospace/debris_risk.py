"""Micrometeoroid and orbital debris (MMOD) penetration risk for spacecraft design.

Spacecraft survivability against MMOD is assessed with a Poisson model: the
surfaces ``i`` of area ``A_i`` are exposed for a time ``T`` to a cumulative flux
``F_i(>d)`` of particles larger than a diameter ``d`` (per unit area and unit
time); a particle penetrates when its diameter exceeds the *critical diameter*
``d_c`` of the shield, given by a ballistic limit equation (BLE) that depends
on the shield, the impact angle and the impact speed. The expected number of
penetrations is ``N = sum_i A_i T F_i(>d_c,i)`` and the probability of no
penetration is ``PNP = exp(-N)``.

Contents:

- :class:`PowerLawFlux` and :class:`TabulatedFlux`: user-supplied cumulative
  flux laws; :func:`demonstration_flux` is a *synthetic* power law.
- :class:`PowerLawBallisticLimit` (single wall) and
  :class:`DualWallBallisticLimit` with three speed regimes (ballistic,
  linear-interpolated transition, hypervelocity) built from
  :class:`DualWallRegime`: *parameterised families* whose coefficients the
  user must take from a source; :func:`demonstration_single_wall_limit` and
  :func:`demonstration_dual_wall_limit` are *synthetic* illustrations.
- :class:`SingleWallShield`, :class:`DualWallShield`, :class:`Surface` and
  :class:`ImpactEnvironment` (binned impact angle and speed);
  :func:`isotropic_angle_bins`.
- :func:`penetration_risk`: the risk algebra ``N`` and ``PNP``.
- Uncertainty: :class:`EpistemicFactors` (lognormal multiplicative factors on
  the flux and on ``d_c``), :func:`pnp_distribution` (nested: the Poisson part
  is analytic for every epistemic draw), :func:`mean_pnp_quadrature` and the
  Jensen gap, :func:`mmod_risk_model` (a
  :class:`probstat.modeling.ProbabilisticModel` for Sobol, FORM and subset
  simulation), and :func:`simulate_penetration_counts` (aleatory counts).
- :func:`required_thickness`: the thickness that meets a PNP requirement,
  nominally or with epistemic margin.
- :func:`update_flux_factor`: Gamma-Poisson updating of the flux factor from
  impact counts on returned surfaces and the posterior predictive of PNP.

Units. Lengths in m, densities in kg/m^3, speeds in m/s, angles in rad (from
the surface normal), areas in m^2. The flux rate and the exposure time must use
the same time unit (years in the demonstration objects). The coefficient of a
ballistic limit equation absorbs the unit system, so a published coefficient
set is valid only in the units of its source; convert before use.

Assumptions and validity.
    Impacts form a homogeneous Poisson process in time with independent
    marks; particle size is independent of the impact angle and speed within a
    surface (the angle and speed bins carry weights that sum to one and apply
    to every size). The angle/speed integral is discretised by a weighted bin
    sum (midpoint rule on bins that the user chooses; the isotropic helper uses
    equal-probability midpoints), so the result converges with the number of
    bins and is not exact for a coarse grid. Surfaces do not shield one another
    unless the user encodes that in the surface flux. The power-law ballistic
    limit has no physical validity range of its own: the user must keep the
    arguments within the range of the source equation. Penetration is taken as
    ``d >= d_c`` (no probabilistic ballistic limit, no secondary damage, no
    penetration-to-failure model). The epistemic factors multiply the flux and
    the critical diameter; they do not represent a shift in the *shape* of the
    flux or of the BLE.

Coefficients and data are not shipped. The ballistic-limit coefficients and
the flux in the ``demonstration_*`` functions are synthetic and illustrative,
are not taken from any published equation, and are not ORDEM, MASTER or any
other environment model; published coefficient sets and environment fluxes must
be taken from the source by the user. Every number in the tests and the
documentation is synthetic; none is a statement about a real spacecraft.

Experimental (added in 0.28).

References:
    [REF-CHRISTIANSEN-1993], [REF-COUR-PALAIS-1987], [REF-WHIPPLE-1947],
    [REF-NASA-PRA-2011], [REF-GELMAN-ETAL-2013], [REF-SALTELLI-2010].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from probstat.distributions import Gamma
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.modeling.inputs import Marginal, gauss_hermite
from probstat.optimization import brent_root
from probstat.uq import FailurePolicy, QuantityOfInterest, UncertaintyNature

__all__ = [
    "CumulativeFlux",
    "DualWallBallisticLimit",
    "DualWallRegime",
    "DualWallShield",
    "EpistemicFactors",
    "FluxFactorPosterior",
    "ImpactEnvironment",
    "PNPDistribution",
    "PenetrationRisk",
    "PowerLawBallisticLimit",
    "PowerLawFlux",
    "QuadraturePNP",
    "SingleWallShield",
    "Surface",
    "TabulatedFlux",
    "ThicknessDesign",
    "baseline_impact_count",
    "demonstration_dual_wall_limit",
    "demonstration_flux",
    "demonstration_single_wall_limit",
    "gamma_prior_from_mean_cv",
    "isotropic_angle_bins",
    "mean_pnp_gamma_flux",
    "mean_pnp_quadrature",
    "mmod_risk_model",
    "penetration_risk",
    "pnp_distribution",
    "required_thickness",
    "simulate_penetration_counts",
    "update_flux_factor",
]

_HALF_PI = 0.5 * math.pi


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _real(value: object, *, name: str) -> float:
    """Return a finite real number (``bool`` rejected).

    References:
        [EQ-AERO-MMOD-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: object, *, name: str) -> float:
    """Return a finite strictly positive real number.

    References:
        [EQ-AERO-MMOD-001]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: object, *, name: str) -> float:
    """Return a finite non-negative real number.

    References:
        [EQ-AERO-MMOD-001]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _angle(value: object, *, name: str = "angle") -> float:
    """Return an impact angle in ``[0, pi/2)`` rad from the surface normal.

    References:
        [EQ-AERO-MMOD-002]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if not 0.0 <= number < _HALF_PI:
        raise ValueError(f"{name} must lie in [0, pi/2) rad from the normal")
    return number


def _integer(value: object, *, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum`` (``bool`` rejected).

    References:
        [EQ-AERO-MMOD-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _open_probability(value: object, *, name: str) -> float:
    """Return a probability strictly inside ``(0, 1)``.

    References:
        [EQ-AERO-MMOD-006]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return number


def _empirical_quantile(sorted_values: Sequence[float], level: float) -> float:
    """Linear-interpolation (type 7) quantile of an ascending sample.

    References:
        [EQ-AERO-MMOD-006]. Sample quantile; no further source.

    """
    count = len(sorted_values)
    position = level * (count - 1)
    low = int(math.floor(position))
    high = min(low + 1, count - 1)
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


# --------------------------------------------------------------------------
# Cumulative flux laws
# --------------------------------------------------------------------------


@runtime_checkable
class CumulativeFlux(Protocol):
    """A cumulative flux law ``F(>d)`` (particles larger than ``d``)."""

    def cumulative(self, diameter: float) -> float:
        """Return ``F(>d)`` for a particle diameter ``d`` in m.

        References:
            [EQ-AERO-MMOD-003]. Protocol member; no equation is evaluated.

        """
        ...


@dataclass(frozen=True, slots=True)
class PowerLawFlux:
    """Power-law cumulative flux ``F(>d) = F_ref (d / d_ref)^(-q)``.

    The flux is user supplied; it is not an environment model. An optional
    ``maximum_diameter`` truncates the population: ``F(>d) = 0`` for
    ``d >= maximum_diameter`` (a discontinuity, documented, not smoothed).

    Args:
        reference_flux: ``F_ref`` in m^-2 per time unit at ``d_ref``.
        reference_diameter: ``d_ref`` in m.
        exponent: ``q > 0`` (flux falls as ``d^-q``).
        maximum_diameter: Optional largest particle diameter in m.

    References:
        [EQ-AERO-MMOD-003].

    """

    reference_flux: float
    reference_diameter: float
    exponent: float
    maximum_diameter: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the parameters.

        Raises:
            TypeError: If a parameter is not a real number.
            ValueError: If a parameter is non-finite or out of range.

        References:
            [EQ-AERO-MMOD-003]. No equation is evaluated.

        """
        object.__setattr__(
            self,
            "reference_flux",
            _positive(self.reference_flux, name="reference_flux"),
        )
        object.__setattr__(
            self,
            "reference_diameter",
            _positive(self.reference_diameter, name="reference_diameter"),
        )
        object.__setattr__(self, "exponent", _positive(self.exponent, name="exponent"))
        if self.maximum_diameter is not None:
            object.__setattr__(
                self,
                "maximum_diameter",
                _positive(self.maximum_diameter, name="maximum_diameter"),
            )

    def cumulative(self, diameter: float) -> float:
        """Return ``F(>d)``.

        Args:
            diameter: Particle diameter in m, positive.

        Returns:
            The cumulative flux, ``0`` at or above ``maximum_diameter``.

        Raises:
            TypeError: If ``diameter`` is not a real number.
            ValueError: If ``diameter`` is not positive and finite.

        Equation:
            [EQ-AERO-MMOD-003].

        References:
            [REF-NASA-PRA-2011].

        """
        d = _positive(diameter, name="diameter")
        if self.maximum_diameter is not None and d >= self.maximum_diameter:
            return 0.0
        return float(
            self.reference_flux * (d / self.reference_diameter) ** (-self.exponent)
        )


@dataclass(frozen=True, slots=True)
class TabulatedFlux:
    """Tabulated cumulative flux with log-log linear interpolation.

    Between nodes ``F`` is a power law (exact for a power-law table). Outside
    the table a diameter is rejected unless ``extrapolate`` is true, in which
    case the end segment's power law is continued.

    Args:
        diameters: Strictly increasing node diameters in m (at least two).
        fluxes: Positive, non-increasing cumulative fluxes at the nodes.
        extrapolate: Continue the end power laws beyond the table.

    References:
        [EQ-AERO-MMOD-003].

    """

    diameters: tuple[float, ...]
    fluxes: tuple[float, ...]
    extrapolate: bool = False

    def __post_init__(self) -> None:
        """Validate the table.

        Raises:
            TypeError: If entries are not real numbers.
            ValueError: If the table has fewer than two nodes, the lengths
                differ, the diameters are not increasing, or the fluxes are
                not positive and non-increasing.

        References:
            [EQ-AERO-MMOD-003]. No equation is evaluated.

        """
        diam = tuple(_positive(d, name="diameters") for d in self.diameters)
        flux = tuple(_positive(f, name="fluxes") for f in self.fluxes)
        if len(diam) < 2 or len(diam) != len(flux):
            raise ValueError("need at least two nodes and equal-length tables")
        if any(b <= a for a, b in zip(diam, diam[1:], strict=False)):
            raise ValueError("diameters must be strictly increasing")
        if any(b > a for a, b in zip(flux, flux[1:], strict=False)):
            raise ValueError("a cumulative flux must be non-increasing")
        object.__setattr__(self, "diameters", diam)
        object.__setattr__(self, "fluxes", flux)
        object.__setattr__(self, "extrapolate", bool(self.extrapolate))

    def cumulative(self, diameter: float) -> float:
        """Return ``F(>d)`` by log-log interpolation.

        Args:
            diameter: Particle diameter in m, positive.

        Returns:
            The cumulative flux.

        Raises:
            ValueError: If ``diameter`` is outside the table and
                ``extrapolate`` is false.

        Equation:
            [EQ-AERO-MMOD-003].

        References:
            [REF-NASA-PRA-2011].

        """
        d = _positive(diameter, name="diameter")
        xs = self.diameters
        if (d < xs[0] or d > xs[-1]) and not self.extrapolate:
            raise ValueError("diameter outside the tabulated range")
        if d <= xs[0]:
            i = 0
        elif d >= xs[-1]:
            i = len(xs) - 2
        else:
            i = 0
            while xs[i + 1] < d:
                i += 1
        slope = math.log(self.fluxes[i + 1] / self.fluxes[i]) / math.log(
            xs[i + 1] / xs[i]
        )
        return float(self.fluxes[i] * (d / xs[i]) ** slope)


def demonstration_flux() -> PowerLawFlux:
    """Return a SYNTHETIC illustrative power-law flux (not ORDEM or MASTER).

    ``F(>d) = 2e-3 (d / 1 mm)^-2.5`` per m^2 per year. The numbers were chosen
    to give risks of order 1e-2 for small shielded areas and carry no
    empirical meaning.

    Returns:
        The demonstration flux.

    Equation:
        [EQ-AERO-MMOD-003].

    References:
        [REF-NASA-PRA-2011].

    """
    return PowerLawFlux(2.0e-3, 1.0e-3, 2.5)


# --------------------------------------------------------------------------
# Ballistic limit equations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PowerLawBallisticLimit:
    """Single-wall ballistic limit as a parameterised power-law family.

    ``d_c = K t^a rho_p^b rho_t^c V^d cos(theta)^e`` with the coefficient
    ``K`` and the exponents ``a, b, c, d, e`` supplied by the user (units and
    validity range of the source equation apply). **No published coefficient
    set is built in; take one from the source** (for example the single-wall
    equation of Cour-Palais [REF-COUR-PALAIS-1987]) and check its units.

    Args:
        coefficient: ``K > 0``.
        thickness_exponent: ``a``.
        projectile_density_exponent: ``b``.
        target_density_exponent: ``c``.
        velocity_exponent: ``d``.
        angle_exponent: ``e`` (``> 0`` makes oblique impacts less severe in
            the sense ``d_c`` shrinks as the angle from the normal grows).

    References:
        [EQ-AERO-MMOD-004], [REF-COUR-PALAIS-1987].

    """

    coefficient: float
    thickness_exponent: float
    projectile_density_exponent: float
    target_density_exponent: float
    velocity_exponent: float
    angle_exponent: float

    def __post_init__(self) -> None:
        """Validate the coefficient and exponents.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a value is non-finite or the coefficient is not
                positive.

        References:
            [EQ-AERO-MMOD-004]. No equation is evaluated.

        """
        object.__setattr__(
            self, "coefficient", _positive(self.coefficient, name="coefficient")
        )
        for key in (
            "thickness_exponent",
            "projectile_density_exponent",
            "target_density_exponent",
            "velocity_exponent",
            "angle_exponent",
        ):
            object.__setattr__(self, key, _real(getattr(self, key), name=key))

    def critical_diameter(
        self,
        *,
        thickness: float,
        projectile_density: float,
        target_density: float,
        velocity: float,
        angle: float,
    ) -> float:
        """Return the critical projectile diameter ``d_c`` in m.

        Args:
            thickness: Wall thickness ``t`` (m).
            projectile_density: ``rho_p`` (kg/m^3).
            target_density: ``rho_t`` (kg/m^3).
            velocity: Impact speed ``V`` (m/s).
            angle: Impact angle from the normal (rad), in ``[0, pi/2)``.

        Returns:
            ``d_c`` in m.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If an argument is out of range.

        Equation:
            [EQ-AERO-MMOD-004].

        References:
            [REF-COUR-PALAIS-1987].

        """
        t = _positive(thickness, name="thickness")
        rp = _positive(projectile_density, name="projectile_density")
        rt = _positive(target_density, name="target_density")
        v = _positive(velocity, name="velocity")
        theta = _angle(angle)
        return float(
            self.coefficient
            * t**self.thickness_exponent
            * rp**self.projectile_density_exponent
            * rt**self.target_density_exponent
            * v**self.velocity_exponent
            * math.cos(theta) ** self.angle_exponent
        )


@dataclass(frozen=True, slots=True)
class DualWallRegime:
    """One speed regime of a dual-wall (Whipple) ballistic limit, as a family.

    ``d_c = K t_b^a t_w^b S^c rho_p^d rho_w^e V^f cos(theta)^g`` with the
    bumper thickness ``t_b``, rear-wall thickness ``t_w``, standoff ``S``,
    projectile density ``rho_p``, rear-wall density ``rho_w``, impact speed
    ``V`` and angle ``theta`` from the normal. The coefficient and the
    exponents are user supplied; the bumper material is absorbed in ``K``.

    Args:
        coefficient: ``K > 0``.
        bumper_thickness_exponent: ``a``.
        wall_thickness_exponent: ``b``.
        spacing_exponent: ``c``.
        projectile_density_exponent: ``d``.
        wall_density_exponent: ``e``.
        velocity_exponent: ``f``.
        angle_exponent: ``g``.

    References:
        [EQ-AERO-MMOD-005], [REF-CHRISTIANSEN-1993], [REF-WHIPPLE-1947].

    """

    coefficient: float
    bumper_thickness_exponent: float
    wall_thickness_exponent: float
    spacing_exponent: float
    projectile_density_exponent: float
    wall_density_exponent: float
    velocity_exponent: float
    angle_exponent: float

    def __post_init__(self) -> None:
        """Validate the coefficient and exponents.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a value is non-finite or the coefficient is not
                positive.

        References:
            [EQ-AERO-MMOD-005]. No equation is evaluated.

        """
        object.__setattr__(
            self, "coefficient", _positive(self.coefficient, name="coefficient")
        )
        for key in (
            "bumper_thickness_exponent",
            "wall_thickness_exponent",
            "spacing_exponent",
            "projectile_density_exponent",
            "wall_density_exponent",
            "velocity_exponent",
            "angle_exponent",
        ):
            object.__setattr__(self, key, _real(getattr(self, key), name=key))

    def critical_diameter(
        self,
        *,
        bumper_thickness: float,
        wall_thickness: float,
        spacing: float,
        projectile_density: float,
        wall_density: float,
        velocity: float,
        angle: float,
    ) -> float:
        """Evaluate this regime's power law (no regime selection).

        Args:
            bumper_thickness: ``t_b`` (m).
            wall_thickness: ``t_w`` (m).
            spacing: ``S`` (m).
            projectile_density: ``rho_p`` (kg/m^3).
            wall_density: ``rho_w`` (kg/m^3).
            velocity: ``V`` (m/s).
            angle: Angle from the normal (rad), in ``[0, pi/2)``.

        Returns:
            ``d_c`` in m.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If an argument is out of range.

        Equation:
            [EQ-AERO-MMOD-005].

        References:
            [REF-CHRISTIANSEN-1993].

        """
        theta = _angle(angle)
        return float(
            self.coefficient
            * _positive(bumper_thickness, name="bumper_thickness")
            ** self.bumper_thickness_exponent
            * _positive(wall_thickness, name="wall_thickness")
            ** self.wall_thickness_exponent
            * _positive(spacing, name="spacing") ** self.spacing_exponent
            * _positive(projectile_density, name="projectile_density")
            ** self.projectile_density_exponent
            * _positive(wall_density, name="wall_density") ** self.wall_density_exponent
            * _positive(velocity, name="velocity") ** self.velocity_exponent
            * math.cos(theta) ** self.angle_exponent
        )


@dataclass(frozen=True, slots=True)
class DualWallBallisticLimit:
    """Three-regime dual-wall ballistic limit with user-supplied coefficients.

    With impact speed ``V`` and the boundaries ``V_b < V_h``::

        V <= V_b:        d_c = d_ballistic(V)
        V >= V_h:        d_c = d_hypervelocity(V)
        V_b < V < V_h:   d_c = d_b(V_b) + (V - V_b)/(V_h - V_b)
                               * (d_h(V_h) - d_b(V_b))

    The transition is the straight line in ``V`` joining the two regimes at
    the boundary speeds, so ``d_c`` is continuous in ``V`` at both boundaries
    by construction (at fixed angle and geometry). The speed that selects the
    regime is the impact speed ``V`` itself; use the angle exponents to
    represent a normal-velocity dependence if the source equation needs it.
    No published coefficient set is built in.

    Args:
        ballistic: Low-speed regime.
        hypervelocity: High-speed regime.
        ballistic_limit_velocity: ``V_b`` (m/s).
        hypervelocity_limit_velocity: ``V_h`` (m/s), larger than ``V_b``.

    References:
        [EQ-AERO-MMOD-005], [REF-CHRISTIANSEN-1993].

    """

    ballistic: DualWallRegime
    hypervelocity: DualWallRegime
    ballistic_limit_velocity: float
    hypervelocity_limit_velocity: float

    def __post_init__(self) -> None:
        """Validate the regimes and boundary speeds.

        Raises:
            TypeError: If a regime or speed has the wrong type.
            ValueError: If the boundary speeds are not ``0 < V_b < V_h``.

        References:
            [EQ-AERO-MMOD-005]. No equation is evaluated.

        """
        if not isinstance(self.ballistic, DualWallRegime) or not isinstance(
            self.hypervelocity, DualWallRegime
        ):
            raise TypeError("regimes must be DualWallRegime instances")
        low = _positive(self.ballistic_limit_velocity, name="ballistic_limit_velocity")
        high = _positive(
            self.hypervelocity_limit_velocity, name="hypervelocity_limit_velocity"
        )
        if high <= low:
            raise ValueError(
                "hypervelocity_limit_velocity must exceed the ballistic one"
            )
        object.__setattr__(self, "ballistic_limit_velocity", low)
        object.__setattr__(self, "hypervelocity_limit_velocity", high)

    def regime(
        self, velocity: float
    ) -> Literal["ballistic", "transition", "hypervelocity"]:
        """Return the speed regime at ``velocity``.

        Args:
            velocity: Impact speed (m/s).

        Returns:
            ``"ballistic"`` (``V <= V_b``), ``"hypervelocity"`` (``V >= V_h``)
            or ``"transition"``.

        Equation:
            [EQ-AERO-MMOD-005].

        References:
            [REF-CHRISTIANSEN-1993].

        """
        v = _positive(velocity, name="velocity")
        if v <= self.ballistic_limit_velocity:
            return "ballistic"
        if v >= self.hypervelocity_limit_velocity:
            return "hypervelocity"
        return "transition"

    def critical_diameter(
        self,
        *,
        bumper_thickness: float,
        wall_thickness: float,
        spacing: float,
        projectile_density: float,
        wall_density: float,
        velocity: float,
        angle: float,
    ) -> float:
        """Return the critical diameter in m for the selected regime.

        Args:
            bumper_thickness: ``t_b`` (m).
            wall_thickness: ``t_w`` (m).
            spacing: ``S`` (m).
            projectile_density: ``rho_p`` (kg/m^3).
            wall_density: ``rho_w`` (kg/m^3).
            velocity: ``V`` (m/s).
            angle: Angle from the normal (rad), in ``[0, pi/2)``.

        Returns:
            ``d_c`` in m.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If an argument is out of range.

        Equation:
            [EQ-AERO-MMOD-005].

        References:
            [REF-CHRISTIANSEN-1993].

        """
        common = {
            "bumper_thickness": bumper_thickness,
            "wall_thickness": wall_thickness,
            "spacing": spacing,
            "projectile_density": projectile_density,
            "wall_density": wall_density,
            "angle": angle,
        }
        v = _positive(velocity, name="velocity")
        v_low = self.ballistic_limit_velocity
        v_high = self.hypervelocity_limit_velocity
        if v <= v_low:
            return self.ballistic.critical_diameter(velocity=v, **common)
        if v >= v_high:
            return self.hypervelocity.critical_diameter(velocity=v, **common)
        d_low = self.ballistic.critical_diameter(velocity=v_low, **common)
        d_high = self.hypervelocity.critical_diameter(velocity=v_high, **common)
        return d_low + (v - v_low) / (v_high - v_low) * (d_high - d_low)


def demonstration_single_wall_limit() -> PowerLawBallisticLimit:
    """Return a SYNTHETIC illustrative single-wall limit (not a published BLE).

    ``d_c = 50 t rho_p^-0.5 rho_t^0.5 V^-0.5 cos(theta)^0.5`` in SI units. The
    coefficient and exponents are invented for demonstration so that
    ``d_c`` is about half the wall thickness at 10 km/s for equal densities;
    they are not those of any published equation.

    Returns:
        The demonstration limit.

    Equation:
        [EQ-AERO-MMOD-004].

    References:
        [REF-COUR-PALAIS-1987].

    """
    return PowerLawBallisticLimit(50.0, 1.0, -0.5, 0.5, -0.5, 0.5)


def demonstration_dual_wall_limit() -> DualWallBallisticLimit:
    """Return a SYNTHETIC illustrative dual-wall limit (not a published BLE).

    Regime boundaries 3 km/s and 7 km/s. Both regimes are invented power laws
    chosen so that the critical diameter is a few millimetres for a 1 mm
    bumper, 3 mm wall and 0.1 m standoff; every coefficient and exponent is
    for demonstration only.

    Returns:
        The demonstration limit.

    Equation:
        [EQ-AERO-MMOD-005].

    References:
        [REF-CHRISTIANSEN-1993].

    """
    ballistic = DualWallRegime(40.0, 0.0, 1.0, 0.0, -0.5, 0.5, -0.5, 0.5)
    hyper = DualWallRegime(1.2, 0.3, 0.6, 0.3, -0.5, 0.5, 0.0, 0.5)
    return DualWallBallisticLimit(ballistic, hyper, 3000.0, 7000.0)


# --------------------------------------------------------------------------
# Shields, surfaces, environment
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SingleWallShield:
    """A single wall of thickness ``t`` and density ``rho_t``.

    Args:
        limit: Single-wall ballistic limit.
        thickness: Wall thickness (m).
        target_density: Wall density (kg/m^3).

    References:
        [EQ-AERO-MMOD-004].

    """

    limit: PowerLawBallisticLimit
    thickness: float
    target_density: float

    def __post_init__(self) -> None:
        """Validate the shield.

        Raises:
            TypeError: If ``limit`` has the wrong type or a value is not real.
            ValueError: If a value is not positive.

        References:
            [EQ-AERO-MMOD-004]. No equation is evaluated.

        """
        if not isinstance(self.limit, PowerLawBallisticLimit):
            raise TypeError("limit must be a PowerLawBallisticLimit")
        object.__setattr__(
            self, "thickness", _positive(self.thickness, name="thickness")
        )
        object.__setattr__(
            self,
            "target_density",
            _positive(self.target_density, name="target_density"),
        )

    def critical_diameter(
        self, velocity: float, angle: float, projectile_density: float
    ) -> float:
        """Return ``d_c`` (m) for one impact condition.

        Args:
            velocity: Impact speed (m/s).
            angle: Angle from the normal (rad).
            projectile_density: Projectile density (kg/m^3).

        Returns:
            The critical diameter in m.

        Equation:
            [EQ-AERO-MMOD-004].

        References:
            [REF-COUR-PALAIS-1987].

        """
        return self.limit.critical_diameter(
            thickness=self.thickness,
            projectile_density=projectile_density,
            target_density=self.target_density,
            velocity=velocity,
            angle=angle,
        )


@dataclass(frozen=True, slots=True)
class DualWallShield:
    """A bumper, a standoff and a rear wall.

    Args:
        limit: Dual-wall ballistic limit.
        bumper_thickness: ``t_b`` (m).
        wall_thickness: ``t_w`` (m).
        spacing: Standoff ``S`` (m).
        wall_density: Rear-wall density (kg/m^3).

    References:
        [EQ-AERO-MMOD-005].

    """

    limit: DualWallBallisticLimit
    bumper_thickness: float
    wall_thickness: float
    spacing: float
    wall_density: float

    def __post_init__(self) -> None:
        """Validate the shield.

        Raises:
            TypeError: If ``limit`` has the wrong type or a value is not real.
            ValueError: If a value is not positive.

        References:
            [EQ-AERO-MMOD-005]. No equation is evaluated.

        """
        if not isinstance(self.limit, DualWallBallisticLimit):
            raise TypeError("limit must be a DualWallBallisticLimit")
        for key in ("bumper_thickness", "wall_thickness", "spacing", "wall_density"):
            object.__setattr__(self, key, _positive(getattr(self, key), name=key))

    def critical_diameter(
        self, velocity: float, angle: float, projectile_density: float
    ) -> float:
        """Return ``d_c`` (m) for one impact condition.

        Args:
            velocity: Impact speed (m/s).
            angle: Angle from the normal (rad).
            projectile_density: Projectile density (kg/m^3).

        Returns:
            The critical diameter in m.

        Equation:
            [EQ-AERO-MMOD-005].

        References:
            [REF-CHRISTIANSEN-1993].

        """
        return self.limit.critical_diameter(
            bumper_thickness=self.bumper_thickness,
            wall_thickness=self.wall_thickness,
            spacing=self.spacing,
            projectile_density=projectile_density,
            wall_density=self.wall_density,
            velocity=velocity,
            angle=angle,
        )


Shield = SingleWallShield | DualWallShield


@dataclass(frozen=True, slots=True)
class Surface:
    """An exposed surface: area, shield and the flux that reaches it.

    Args:
        name: Non-empty label.
        area: Exposed area ``A_i`` (m^2).
        shield: A :class:`SingleWallShield` or :class:`DualWallShield`.
        flux: Cumulative flux law ``F_i(>d)`` seen by this surface (per m^2
            per time unit); ram, wake and shadowing differences are encoded
            here by the user.

    References:
        [EQ-AERO-MMOD-001].

    """

    name: str
    area: float
    shield: SingleWallShield | DualWallShield
    flux: CumulativeFlux

    def __post_init__(self) -> None:
        """Validate the surface.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If the name is empty or the area is not positive.

        References:
            [EQ-AERO-MMOD-001]. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "area", _positive(self.area, name="area"))
        if not isinstance(self.shield, SingleWallShield | DualWallShield):
            raise TypeError("shield must be a SingleWallShield or DualWallShield")
        if not callable(getattr(self.flux, "cumulative", None)):
            raise TypeError("flux must provide cumulative(diameter)")


@dataclass(frozen=True, slots=True)
class ImpactEnvironment:
    """Binned impact angle and speed with a projectile density.

    The weights ``w[j][k]`` are the fraction of impacting particles (of any
    size) with angle ``angles[j]`` and speed ``velocities[k]``; they sum to
    one. The integral over the angle and speed distributions is the weighted
    sum over the bins.

    Args:
        angles: Bin angles from the surface normal (rad, in ``[0, pi/2)``).
        velocities: Bin impact speeds (m/s).
        weights: ``len(angles)`` rows of ``len(velocities)`` non-negative
            weights summing to one.
        projectile_density: Particle density (kg/m^3), common to all bins.

    References:
        [EQ-AERO-MMOD-002].

    """

    angles: tuple[float, ...]
    velocities: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    projectile_density: float

    def __post_init__(self) -> None:
        """Validate the bins and weights.

        Raises:
            TypeError: If an entry is not a real number.
            ValueError: If shapes disagree, a weight is negative or the
                weights do not sum to one within ``1e-9``.

        References:
            [EQ-AERO-MMOD-002]. No equation is evaluated.

        """
        angles = tuple(_angle(a) for a in self.angles)
        speeds = tuple(_positive(v, name="velocity") for v in self.velocities)
        rows = tuple(
            tuple(_non_negative(w, name="weight") for w in r) for r in self.weights
        )
        if not angles or not speeds:
            raise ValueError("angles and velocities must be non-empty")
        if len(rows) != len(angles) or any(len(r) != len(speeds) for r in rows):
            raise ValueError("weights must be len(angles) x len(velocities)")
        total = math.fsum(w for r in rows for w in r)
        if abs(total - 1.0) > 1.0e-9:
            raise ValueError(f"weights must sum to 1 (got {total!r})")
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "velocities", speeds)
        object.__setattr__(self, "weights", rows)
        object.__setattr__(
            self,
            "projectile_density",
            _positive(self.projectile_density, name="projectile_density"),
        )

    @classmethod
    def single_bin(
        cls, angle: float, velocity: float, projectile_density: float
    ) -> ImpactEnvironment:
        """Return a one-bin environment (all particles at one angle and speed).

        Args:
            angle: Angle from the normal (rad).
            velocity: Impact speed (m/s).
            projectile_density: Particle density (kg/m^3).

        Returns:
            The environment with weight one.

        Equation:
            [EQ-AERO-MMOD-002].

        References:
            [REF-NASA-PRA-2011].

        """
        return cls((angle,), (velocity,), ((1.0,),), projectile_density)

    @classmethod
    def product(
        cls,
        angles: Sequence[float],
        angle_weights: Sequence[float],
        velocities: Sequence[float],
        velocity_weights: Sequence[float],
        projectile_density: float,
    ) -> ImpactEnvironment:
        """Return the environment with independent angle and speed laws.

        Args:
            angles: Bin angles (rad).
            angle_weights: Angle-bin probabilities summing to one.
            velocities: Bin speeds (m/s).
            velocity_weights: Speed-bin probabilities summing to one.
            projectile_density: Particle density (kg/m^3).

        Returns:
            The product-weight environment ``w_jk = wa_j wv_k``.

        Raises:
            ValueError: If lengths disagree or a weight set does not sum to
                one.

        Equation:
            [EQ-AERO-MMOD-002].

        References:
            [REF-NASA-PRA-2011].

        """
        if len(angles) != len(angle_weights) or len(velocities) != len(
            velocity_weights
        ):
            raise ValueError("each bin set needs one weight per bin")
        for label, weights in (
            ("angle", angle_weights),
            ("velocity", velocity_weights),
        ):
            if abs(math.fsum(weights) - 1.0) > 1.0e-9:
                raise ValueError(f"{label} weights must sum to 1")
        rows = tuple(tuple(wa * wv for wv in velocity_weights) for wa in angle_weights)
        return cls(tuple(angles), tuple(velocities), rows, projectile_density)

    def bins(self) -> tuple[tuple[float, float, float], ...]:
        """Return the non-empty bins as ``(weight, velocity, angle)``.

        Returns:
            Tuples for every bin with positive weight.

        Equation:
            [EQ-AERO-MMOD-002].

        References:
            [REF-NASA-PRA-2011].

        """
        return tuple(
            (w, v, a)
            for a, row in zip(self.angles, self.weights, strict=True)
            for v, w in zip(self.velocities, row, strict=True)
            if w > 0.0
        )


def isotropic_angle_bins(count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return equal-probability angle bins for an isotropic flux on a flat plate.

    For a flux that is isotropic over the visible hemisphere the angle ``theta``
    from the normal has density ``2 sin(theta) cos(theta)``, i.e.
    ``P(theta <= x) = sin^2 x``. The ``k``-th of ``count`` equal-probability
    bins is represented by the angle at its probability midpoint,
    ``theta_k = asin(sqrt((k + 1/2) / count))``, with weight ``1/count``
    (midpoint rule in probability; ``E[cos theta] = 2/3`` is recovered as
    ``count`` grows).

    Args:
        count: Number of bins, at least 1.

    Returns:
        ``(angles, weights)``.

    Raises:
        TypeError: If ``count`` is not an integer.
        ValueError: If ``count < 1``.

    Equation:
        [EQ-AERO-MMOD-002].

    References:
        [REF-NASA-PRA-2011].

    """
    n = _integer(count, name="count", minimum=1)
    angles = tuple(math.asin(math.sqrt((k + 0.5) / n)) for k in range(n))
    return angles, (1.0 / n,) * n


# --------------------------------------------------------------------------
# Risk algebra
# --------------------------------------------------------------------------

_Table = tuple[tuple[float, float], ...]


def _critical_table(surface: Surface, environment: ImpactEnvironment) -> _Table:
    """Return ``(weight, d_c)`` for every non-empty bin of one surface.

    References:
        [EQ-AERO-MMOD-002].

    """
    rho = environment.projectile_density
    return tuple(
        (w, surface.shield.critical_diameter(v, a, rho))
        for w, v, a in environment.bins()
    )


def _surface_n(
    table: _Table,
    surface: Surface,
    exposure_time: float,
    flux_factor: float,
    limit_factor: float,
) -> float:
    """Return ``A T f sum_b w_b F(>g d_c,b)`` for one surface.

    References:
        [EQ-AERO-MMOD-002], [EQ-AERO-MMOD-006].

    """
    inner = math.fsum(w * surface.flux.cumulative(limit_factor * d) for w, d in table)
    return surface.area * exposure_time * flux_factor * inner


@dataclass(frozen=True, slots=True)
class PenetrationRisk:
    """Result of the Poisson penetration-risk algebra.

    Attributes:
        names: Surface labels.
        per_surface: ``N_i`` per surface.
        expected_penetrations: ``N = sum N_i``.
        pnp: ``exp(-N)``, the probability of no penetration.
        penetration_probability: ``1 - exp(-N)`` (by ``-expm1``).
        exposure_time: ``T`` (the flux time unit).
        bins: Number of angle/speed bins used.
        flux_factor: Epistemic flux factor applied.
        limit_factor: Epistemic critical-diameter factor applied.

    References:
        [EQ-AERO-MMOD-001].

    """

    names: tuple[str, ...]
    per_surface: tuple[float, ...]
    expected_penetrations: float
    pnp: float
    penetration_probability: float
    exposure_time: float
    bins: int
    flux_factor: float
    limit_factor: float


def penetration_risk(
    surfaces: Sequence[Surface],
    environment: ImpactEnvironment,
    exposure_time: float,
    *,
    flux_factor: float = 1.0,
    limit_factor: float = 1.0,
) -> PenetrationRisk:
    """Return ``N = sum_i A_i T f sum_b w_b F_i(>g d_c,ib)`` and ``PNP = exp(-N)``.

    The penetrations form a Poisson process, so the count is Poisson with mean
    ``N`` and ``PNP = P(count = 0) = exp(-N)``. The surfaces' counts add
    (independent thinning), hence ``PNP`` is the product of the surface PNPs.
    ``flux_factor`` ``f`` scales the flux and ``limit_factor`` ``g`` scales
    every critical diameter (both 1 for the nominal case).

    Args:
        surfaces: One or more surfaces.
        environment: Binned impact angle and speed law.
        exposure_time: ``T`` in the flux time unit, positive.
        flux_factor: ``f > 0``.
        limit_factor: ``g > 0``.

    Returns:
        The risk summary.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a value is out of range or ``surfaces`` is empty.

    Equation:
        [EQ-AERO-MMOD-001], [EQ-AERO-MMOD-002].

    References:
        [REF-NASA-PRA-2011].

    """
    items = _check_surfaces(surfaces)
    if not isinstance(environment, ImpactEnvironment):
        raise TypeError("environment must be an ImpactEnvironment")
    time = _positive(exposure_time, name="exposure_time")
    f = _positive(flux_factor, name="flux_factor")
    g = _positive(limit_factor, name="limit_factor")
    parts = tuple(
        _surface_n(_critical_table(s, environment), s, time, f, g) for s in items
    )
    total = math.fsum(parts)
    return PenetrationRisk(
        tuple(s.name for s in items),
        parts,
        total,
        math.exp(-total),
        -math.expm1(-total),
        time,
        len(environment.bins()),
        f,
        g,
    )


def _check_surfaces(surfaces: Sequence[Surface]) -> tuple[Surface, ...]:
    """Return the surfaces as a validated non-empty tuple.

    References:
        [EQ-AERO-MMOD-001]. No equation is evaluated.

    """
    items = tuple(surfaces)
    if not items:
        raise ValueError("at least one surface is required")
    if any(not isinstance(s, Surface) for s in items):
        raise TypeError("surfaces must be Surface instances")
    return items


def simulate_penetration_counts(
    expected_penetrations: float,
    size: int,
    *,
    seed: int | RandomStream | None = 0,
) -> tuple[int, ...]:
    """Draw aleatory penetration counts, Poisson with mean ``N``.

    Uses sequential inversion; ``N`` is limited to 100 so the method stays
    accurate (it is a risk model for rare events).

    Args:
        expected_penetrations: ``N``, in ``[0, 100]``.
        size: Number of mission replicates, at least 1.
        seed: Integer seed or stream.

    Returns:
        The counts.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``N`` is outside ``[0, 100]`` or ``size < 1``.

    Equation:
        [EQ-AERO-MMOD-001].

    References:
        [REF-NASA-PRA-2011].

    """
    n = _non_negative(expected_penetrations, name="expected_penetrations")
    if n > 100.0:
        raise ValueError("expected_penetrations must not exceed 100")
    count = _integer(size, name="size", minimum=1)
    rng = resolve_stream(seed).child("counts").generator()
    out: list[int] = []
    p0 = math.exp(-n)
    for _ in range(count):
        u = rng.random()
        k, pmf = 0, p0
        cdf = pmf
        while u > cdf and k < 10_000:
            k += 1
            pmf *= n / k
            cdf += pmf
        out.append(k)
    return tuple(out)


# --------------------------------------------------------------------------
# Epistemic uncertainty
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpistemicFactors:
    """Lognormal multiplicative factors on the flux and on the critical diameter.

    ``f = flux_median exp(sigma_f Z_1)`` multiplies the flux and
    ``g = limit_median exp(sigma_g Z_2)`` multiplies every critical diameter,
    with independent standard-normal ``Z``. ``sigma = 0`` means that factor is
    known. The mean of ``f`` is ``flux_median exp(sigma_f^2 / 2)``.

    Args:
        flux_sigma: ``sigma_f >= 0`` (log standard deviation).
        limit_sigma: ``sigma_g >= 0``.
        flux_median: Median of ``f`` (default 1).
        limit_median: Median of ``g`` (default 1).

    References:
        [EQ-AERO-MMOD-006].

    """

    flux_sigma: float
    limit_sigma: float = 0.0
    flux_median: float = 1.0
    limit_median: float = 1.0

    def __post_init__(self) -> None:
        """Validate the factor parameters.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a sigma is negative or a median not positive.

        References:
            [EQ-AERO-MMOD-006]. No equation is evaluated.

        """
        for key in ("flux_sigma", "limit_sigma"):
            object.__setattr__(self, key, _non_negative(getattr(self, key), name=key))
        for key in ("flux_median", "limit_median"):
            object.__setattr__(self, key, _positive(getattr(self, key), name=key))

    @property
    def flux_mean(self) -> float:
        """Return ``E[f] = median exp(sigma^2 / 2)``.

        References:
            [EQ-AERO-MMOD-006]. Lognormal mean.

        """
        return self.flux_median * math.exp(0.5 * self.flux_sigma**2)

    @property
    def limit_mean(self) -> float:
        """Return ``E[g] = median exp(sigma^2 / 2)``.

        References:
            [EQ-AERO-MMOD-006]. Lognormal mean.

        """
        return self.limit_median * math.exp(0.5 * self.limit_sigma**2)


@dataclass(frozen=True, slots=True)
class PNPDistribution:
    """Distribution of PNP over the epistemic factors (Poisson part analytic).

    Attributes:
        samples: Number of epistemic draws.
        mean_pnp: ``mean_k exp(-N_k)`` (the expected PNP).
        mean_pnp_standard_error: Monte Carlo standard error of ``mean_pnp``.
        std_pnp: Sample standard deviation of PNP over the epistemic draws.
        pnp_at_mean_inputs: ``exp(-N(E f, E g))``.
        pnp_at_median_inputs: ``exp(-N(med f, med g))`` (the nominal case).
        jensen_gap: ``mean_pnp - pnp_at_mean_inputs``. Its sign is not fixed
            when ``N`` is convex in an input (see :func:`pnp_distribution`).
        pnp_at_mean_n: ``exp(-mean_k N_k) = exp(-E[N])``.
        poisson_jensen_gap: ``mean_pnp - pnp_at_mean_n``; non-negative
            always (Jensen's inequality for the convex ``exp(-n)``).
        level: Credibility level of the quantities below.
        expected_penetrations_quantile: ``N`` quantile at ``level`` (the
            epistemic-margin ``N``).
        pnp_lower: PNP exceeded with probability ``level`` (the ``1-level``
            quantile).
        requirement: PNP requirement used.
        probability_below_requirement: Fraction of epistemic draws with
            ``PNP < requirement`` (probability that the requirement is missed).
        pnp_sorted: The sorted PNP draws.

    References:
        [EQ-AERO-MMOD-006].

    """

    samples: int
    mean_pnp: float
    mean_pnp_standard_error: float
    std_pnp: float
    pnp_at_mean_inputs: float
    pnp_at_median_inputs: float
    jensen_gap: float
    pnp_at_mean_n: float
    poisson_jensen_gap: float
    level: float
    expected_penetrations_quantile: float
    pnp_lower: float
    requirement: float
    probability_below_requirement: float
    pnp_sorted: tuple[float, ...] = field(repr=False)


def _draw_factors(
    factors: EpistemicFactors, samples: int, seed: int | RandomStream | None
) -> tuple[tuple[float, float], ...]:
    """Draw ``samples`` pairs ``(f, g)`` from the lognormal factors.

    References:
        [EQ-AERO-MMOD-006].

    """
    stream = resolve_stream(seed)
    rng_f = stream.child("flux-factor").generator()
    rng_g = stream.child("limit-factor").generator()
    return tuple(
        (
            factors.flux_median * math.exp(factors.flux_sigma * rng_f.gauss(0.0, 1.0)),
            factors.limit_median
            * math.exp(factors.limit_sigma * rng_g.gauss(0.0, 1.0)),
        )
        for _ in range(samples)
    )


def pnp_distribution(
    surfaces: Sequence[Surface],
    environment: ImpactEnvironment,
    exposure_time: float,
    factors: EpistemicFactors,
    *,
    samples: int = 4000,
    level: float = 0.9,
    requirement: float = 0.95,
    seed: int | RandomStream | None = 0,
) -> PNPDistribution:
    """Propagate the epistemic factors to a distribution of PNP (nested).

    The outer loop draws ``(f_k, g_k)``; for each draw the aleatory Poisson
    part is analytic, ``PNP_k = exp(-N_k)``, so no inner sampling is needed.
    ``mean_pnp`` is the expected PNP, ``pnp_at_mean_inputs`` the PNP computed
    at the mean factors; their difference is ``jensen_gap``. Two effects
    enter. Because ``exp(-n)`` is convex, ``E[exp(-N)] >= exp(-E[N])`` always
    (``poisson_jensen_gap >= 0``). But ``E[N]`` itself differs from
    ``N(E f, E g)`` when ``N`` is non-linear in the inputs: ``N`` is linear in
    ``f`` and convex in ``g`` (``F(>g d)`` falls as ``g^-q`` for a power-law
    flux), so ``E[N] >= N(E f, E g)`` and that second term pulls
    ``jensen_gap`` down. With uncertainty on ``f`` alone ``jensen_gap > 0``;
    with uncertainty on ``g`` it can be negative. ``poisson_jensen_gap``
    isolates the first effect and is the one that is always non-negative.

    Args:
        surfaces: Surfaces.
        environment: Impact angle/speed bins.
        exposure_time: ``T``.
        factors: Epistemic lognormal factors.
        samples: Number of draws, at least 10.
        level: Credibility level in ``(0, 1)`` for the margin quantities.
        requirement: PNP requirement in ``(0, 1)``.
        seed: Integer seed or stream.

    Returns:
        The distribution summary.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-MMOD-006].

    References:
        [REF-GELMAN-ETAL-2013], [REF-SMITH-UQ-2024].

    """
    items = _check_surfaces(surfaces)
    if not isinstance(environment, ImpactEnvironment):
        raise TypeError("environment must be an ImpactEnvironment")
    if not isinstance(factors, EpistemicFactors):
        raise TypeError("factors must be EpistemicFactors")
    time = _positive(exposure_time, name="exposure_time")
    count = _integer(samples, name="samples", minimum=10)
    lev = _open_probability(level, name="level")
    req = _open_probability(requirement, name="requirement")
    tables = tuple(_critical_table(s, environment) for s in items)

    def total_n(f: float, g: float) -> float:
        """Return ``N`` for one factor pair.

        References:
            [EQ-AERO-MMOD-006].

        """
        return math.fsum(
            _surface_n(t, s, time, f, g) for t, s in zip(tables, items, strict=True)
        )

    draws = _draw_factors(factors, count, seed)
    n_values = sorted(total_n(f, g) for f, g in draws)
    pnp_values = [math.exp(-n) for n in n_values]
    mean = math.fsum(pnp_values) / count
    variance = math.fsum((p - mean) ** 2 for p in pnp_values) / (count - 1)
    at_mean = math.exp(-total_n(factors.flux_mean, factors.limit_mean))
    at_mean_n = math.exp(-math.fsum(n_values) / count)
    at_median = math.exp(-total_n(factors.flux_median, factors.limit_median))
    pnp_sorted = tuple(sorted(pnp_values))
    return PNPDistribution(
        samples=count,
        mean_pnp=mean,
        mean_pnp_standard_error=math.sqrt(variance / count),
        std_pnp=math.sqrt(variance),
        pnp_at_mean_inputs=at_mean,
        pnp_at_median_inputs=at_median,
        jensen_gap=mean - at_mean,
        pnp_at_mean_n=at_mean_n,
        poisson_jensen_gap=mean - at_mean_n,
        level=lev,
        expected_penetrations_quantile=_empirical_quantile(n_values, lev),
        pnp_lower=_empirical_quantile(pnp_sorted, 1.0 - lev),
        requirement=req,
        probability_below_requirement=sum(1 for p in pnp_values if p < req) / count,
        pnp_sorted=pnp_sorted,
    )


@dataclass(frozen=True, slots=True)
class QuadraturePNP:
    """Deterministic (Gauss-Hermite) moments of PNP over the epistemic factors.

    Attributes:
        mean_pnp: ``E[exp(-N(f, g))]``.
        std_pnp: Standard deviation of ``exp(-N)``.
        pnp_at_mean_inputs: ``exp(-N(E f, E g))``.
        pnp_at_median_inputs: ``exp(-N(med f, med g))``.
        jensen_gap: ``mean_pnp - pnp_at_mean_inputs`` (sign not fixed; see
            :func:`pnp_distribution`).
        pnp_at_mean_n: ``exp(-E[N])``.
        poisson_jensen_gap: ``mean_pnp - pnp_at_mean_n >= 0``.
        order: Gauss-Hermite nodes per uncertain dimension.

    References:
        [EQ-AERO-MMOD-007].

    """

    mean_pnp: float
    std_pnp: float
    pnp_at_mean_inputs: float
    pnp_at_median_inputs: float
    jensen_gap: float
    pnp_at_mean_n: float
    poisson_jensen_gap: float
    order: int


def mean_pnp_quadrature(
    surfaces: Sequence[Surface],
    environment: ImpactEnvironment,
    exposure_time: float,
    factors: EpistemicFactors,
    *,
    order: int = 24,
) -> QuadraturePNP:
    """Return ``E[PNP]`` and its Jensen gap by tensor Gauss-Hermite quadrature.

    ``E[exp(-N(f, g))]`` with ``f, g`` lognormal is integrated over the two
    standard-normal scores (a dimension with ``sigma = 0`` uses one node). The
    rule is deterministic, so it is the noise-free reference for
    :func:`pnp_distribution`. Accuracy needs ``N`` smooth in the scores; check
    by doubling ``order`` for large ``sigma``.

    Args:
        surfaces: Surfaces.
        environment: Impact bins.
        exposure_time: ``T``.
        factors: Epistemic factors.
        order: Nodes per dimension, 2 to 80.

    Returns:
        Moments of PNP and the Jensen gap.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-MMOD-007].

    References:
        [REF-GOLUB-WELSCH-1969], [REF-GELMAN-ETAL-2013].

    """
    items = _check_surfaces(surfaces)
    if not isinstance(environment, ImpactEnvironment):
        raise TypeError("environment must be an ImpactEnvironment")
    if not isinstance(factors, EpistemicFactors):
        raise TypeError("factors must be EpistemicFactors")
    time = _positive(exposure_time, name="exposure_time")
    nodes = _integer(order, name="order", minimum=2)
    tables = tuple(_critical_table(s, environment) for s in items)

    def total_n(f: float, g: float) -> float:
        """Return ``N`` for one factor pair.

        References:
            [EQ-AERO-MMOD-007].

        """
        return math.fsum(
            _surface_n(t, s, time, f, g) for t, s in zip(tables, items, strict=True)
        )

    z_all, w_all = gauss_hermite(nodes)
    z_f, w_f = (z_all, w_all) if factors.flux_sigma > 0.0 else ((0.0,), (1.0,))
    z_g, w_g = (z_all, w_all) if factors.limit_sigma > 0.0 else ((0.0,), (1.0,))
    first = 0.0
    second = 0.0
    mean_n = 0.0
    for zf, wf in zip(z_f, w_f, strict=True):
        f = factors.flux_median * math.exp(factors.flux_sigma * zf)
        for zg, wg in zip(z_g, w_g, strict=True):
            g = factors.limit_median * math.exp(factors.limit_sigma * zg)
            n = total_n(f, g)
            p = math.exp(-n)
            first += wf * wg * p
            second += wf * wg * p * p
            mean_n += wf * wg * n
    at_mean = math.exp(-total_n(factors.flux_mean, factors.limit_mean))
    at_median = math.exp(-total_n(factors.flux_median, factors.limit_median))
    at_mean_n = math.exp(-mean_n)
    return QuadraturePNP(
        mean_pnp=first,
        std_pnp=math.sqrt(max(second - first * first, 0.0)),
        pnp_at_mean_inputs=at_mean,
        pnp_at_median_inputs=at_median,
        jensen_gap=first - at_mean,
        pnp_at_mean_n=at_mean_n,
        poisson_jensen_gap=first - at_mean_n,
        order=nodes,
    )


def mean_pnp_gamma_flux(
    expected_penetrations: float, shape: float, rate: float
) -> float:
    """Return ``E[exp(-N0 f)]`` for a Gamma-distributed flux factor, exactly.

    With ``f ~ Gamma(shape a, rate b)`` the Laplace transform gives
    ``E[exp(-N0 f)] = (b / (b + N0))^a``, the PNP averaged over the factor
    (also the posterior predictive of PNP, see :class:`FluxFactorPosterior`).

    Args:
        expected_penetrations: Nominal ``N0`` (at ``f = 1``), non-negative.
        shape: ``a > 0``.
        rate: ``b > 0``.

    Returns:
        The mean PNP.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-MMOD-010].

    References:
        [REF-GELMAN-ETAL-2013].

    """
    n0 = _non_negative(expected_penetrations, name="expected_penetrations")
    a = _positive(shape, name="shape")
    b = _positive(rate, name="rate")
    return math.exp(a * (math.log(b) - math.log(b + n0)))


# --------------------------------------------------------------------------
# Probabilistic model
# --------------------------------------------------------------------------

_MODEL_QUANTITIES = (
    QuantityOfInterest(
        "expected_penetrations", "Expected number of penetrations N", "1"
    ),
    QuantityOfInterest("pnp", "Probability of no penetration exp(-N)", "1"),
)


@dataclass(frozen=True, slots=True)
class _MMODForward:
    """Picklable forward map of :func:`mmod_risk_model`.

    Args:
        surfaces: Surfaces.
        tables: Precomputed ``(weight, d_c)`` tables, one per surface.

    References:
        [EQ-AERO-MMOD-011].

    """

    surfaces: tuple[Surface, ...]
    tables: tuple[_Table, ...]

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Return ``N`` and ``PNP`` for one realisation of the inputs.

        Args:
            parameters: ``flux_factor``, ``limit_factor``, ``exposure_time``.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If an input is not positive.

        Equation:
            [EQ-AERO-MMOD-011].

        References:
            [REF-SMITH-UQ-2024].

        """
        f = _positive(parameters["flux_factor"], name="flux_factor")
        g = _positive(parameters["limit_factor"], name="limit_factor")
        t = _positive(parameters["exposure_time"], name="exposure_time")
        n = math.fsum(
            _surface_n(tab, s, t, f, g)
            for tab, s in zip(self.tables, self.surfaces, strict=True)
        )
        return {"expected_penetrations": n, "pnp": math.exp(-n)}


def mmod_risk_model(
    surfaces: Sequence[Surface],
    environment: ImpactEnvironment,
    *,
    flux_factor: Marginal | float,
    limit_factor: Marginal | float,
    exposure_time: Marginal | float,
    name: str = "mmod-risk",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a :class:`~probstat.modeling.ProbabilisticModel` of MMOD risk.

    Inputs are the epistemic flux factor ``f``, the epistemic critical-diameter
    factor ``g`` and the exposure time ``T`` (each a distribution or a plain
    number; at least one must be random). ``flux_factor`` and ``limit_factor``
    are tagged epistemic, ``exposure_time`` aleatory. Quantities of interest:
    ``expected_penetrations`` (``N``) and ``pnp`` (``exp(-N)``). The Poisson
    randomness of the impacts is *not* a model input: it is analytic inside
    ``pnp``. The object works with the library's Sobol, FORM and subset
    simulation tools; for example ``ModelLimitState(model, "pnp", 0.95,
    "below")`` is the event that the PNP requirement is missed.

    Args:
        surfaces: Surfaces.
        environment: Impact bins.
        flux_factor: ``f`` (distribution or number).
        limit_factor: ``g``.
        exposure_time: ``T`` in the flux time unit.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        The probabilistic model.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is out of range or no input is random.

    Equation:
        [EQ-AERO-MMOD-011].

    References:
        [REF-SMITH-UQ-2024], [REF-SALTELLI-2010].

    """
    items = _check_surfaces(surfaces)
    if not isinstance(environment, ImpactEnvironment):
        raise TypeError("environment must be an ImpactEnvironment")
    given: dict[str, tuple[Marginal | float, UncertaintyNature]] = {
        "flux_factor": (flux_factor, UncertaintyNature.EPISTEMIC),
        "limit_factor": (limit_factor, UncertaintyNature.EPISTEMIC),
        "exposure_time": (exposure_time, UncertaintyNature.ALEATORY),
    }
    random_inputs: list[RandomInput] = []
    fixed: dict[str, float] = {}
    for key, (value, nature) in given.items():
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = _positive(value, name=key)
        else:
            random_inputs.append(RandomInput(key, value, nature=nature))
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    forward = _MMODForward(items, tuple(_critical_table(s, environment) for s in items))
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        forward,
        _MODEL_QUANTITIES,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# --------------------------------------------------------------------------
# Shield design trade
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ThicknessDesign:
    """Result of :func:`required_thickness`.

    Attributes:
        thickness: Smallest design variable meeting the requirement (m).
        basis: ``"nominal"``, ``"quantile"`` or ``"mean_pnp"``.
        requirement: PNP requirement.
        level: Credibility level (``"quantile"`` basis) else ``nan``.
        nominal_pnp: PNP at the nominal factors at ``thickness``.
        achieved_metric: The basis metric at ``thickness``: ``exp(-N_q)`` for
            ``"quantile"``, the mean PNP for ``"mean_pnp"``, the nominal PNP
            for ``"nominal"``; it equals the requirement at the root.
        iterations: Brent iterations.
        converged: Whether the root search converged.

    References:
        [EQ-AERO-MMOD-008].

    """

    thickness: float
    basis: str
    requirement: float
    level: float
    nominal_pnp: float
    achieved_metric: float
    iterations: int
    converged: bool


def required_thickness(
    build: Callable[[float], Sequence[Surface]],
    environment: ImpactEnvironment,
    exposure_time: float,
    requirement: float,
    *,
    bracket: tuple[float, float],
    basis: Literal["nominal", "quantile", "mean_pnp"] = "nominal",
    factors: EpistemicFactors | None = None,
    level: float = 0.9,
    samples: int = 4000,
    seed: int | RandomStream | None = 0,
    tolerance: float = 1.0e-10,
) -> ThicknessDesign:
    """Return the design variable ``t`` that just meets a PNP requirement.

    ``build(t)`` returns the surfaces for design variable ``t`` (a thickness in
    m; the caller chooses what it controls). The requirement ``PNP >= R`` is
    ``N <= -ln R``. The three bases are

    - ``"nominal"``: ``N`` at ``f = g = 1`` (no epistemic margin);
    - ``"quantile"``: the ``level`` quantile of ``N`` over the epistemic
      factors (for example the 90th percentile of ``N``);
    - ``"mean_pnp"``: the epistemic mean of PNP.

    The epistemic basis uses one fixed set of draws (common random numbers) so
    the criterion is a deterministic, piecewise-continuous function of ``t``.
    A root of the criterion is located by Brent's method on ``log t`` inside
    ``bracket``; this assumes the criterion decreases with ``t`` (a thicker
    wall does not raise ``d_c``-based risk) and that the bracket encloses the
    root: both are *checked at the endpoints only*, a non-monotone ``build`` is
    not detected in between.

    Args:
        build: Maps ``t`` to the surfaces.
        environment: Impact bins.
        exposure_time: ``T``.
        requirement: Required PNP in ``(0, 1)``.
        bracket: ``(t_low, t_high)``, ``0 < t_low < t_high``.
        basis: See above.
        factors: Epistemic factors (required for the non-nominal bases).
        level: Credibility level for ``"quantile"``.
        samples: Epistemic draws, at least 10.
        seed: Integer seed or stream.
        tolerance: Brent tolerance on ``log t``.

    Returns:
        The design.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is invalid, ``factors`` is missing for an
            epistemic basis, or the bracket does not enclose a root (the
            requirement is already met at ``t_low`` or not met at ``t_high``).

    Equation:
        [EQ-AERO-MMOD-008].

    References:
        [REF-NASA-PRA-2011], [REF-GELMAN-ETAL-2013].

    """
    if not callable(build):
        raise TypeError("build must be callable")
    if not isinstance(environment, ImpactEnvironment):
        raise TypeError("environment must be an ImpactEnvironment")
    time = _positive(exposure_time, name="exposure_time")
    req = _open_probability(requirement, name="requirement")
    low = _positive(bracket[0], name="bracket[0]")
    high = _positive(bracket[1], name="bracket[1]")
    if high <= low:
        raise ValueError("bracket must satisfy 0 < low < high")
    if basis not in ("nominal", "quantile", "mean_pnp"):
        raise ValueError("basis must be 'nominal', 'quantile' or 'mean_pnp'")
    lev = _open_probability(level, name="level")
    draws: tuple[tuple[float, float], ...] = ((1.0, 1.0),)
    if basis != "nominal":
        if not isinstance(factors, EpistemicFactors):
            raise ValueError("factors are required for an epistemic basis")
        draws = _draw_factors(
            factors, _integer(samples, name="samples", minimum=10), seed
        )
    target = -math.log(req)

    def n_values(t: float) -> list[float]:
        """Return ``N`` at design variable ``t`` for every draw.

        References:
            [EQ-AERO-MMOD-008].

        """
        items = _check_surfaces(build(t))
        tables = tuple(_critical_table(s, environment) for s in items)
        return [
            math.fsum(
                _surface_n(tab, s, time, f, g)
                for tab, s in zip(tables, items, strict=True)
            )
            for f, g in draws
        ]

    def metric(t: float) -> float:
        """Return the criterion value, positive when the requirement is missed.

        References:
            [EQ-AERO-MMOD-008].

        """
        values = n_values(t)
        if basis == "nominal":
            return values[0] - target
        if basis == "quantile":
            return _empirical_quantile(sorted(values), lev) - target
        return req - math.fsum(math.exp(-n) for n in values) / len(values)

    def on_log(s: float) -> float:
        """Return the criterion at ``t = exp(s)``.

        References:
            [EQ-AERO-MMOD-008].

        """
        return metric(math.exp(s))

    m_low, m_high = metric(low), metric(high)
    if m_low <= 0.0:
        raise ValueError("requirement already met at the lower bracket end")
    if m_high > 0.0:
        raise ValueError("requirement not met at the upper bracket end")
    root = brent_root(on_log, math.log(low), math.log(high), tolerance=tolerance)
    t_star = math.exp(root.root)
    values = n_values(t_star)
    nominal = _check_surfaces(build(t_star))
    nominal_n = penetration_risk(nominal, environment, time).pnp
    if basis == "nominal":
        achieved = math.exp(-values[0])
    elif basis == "quantile":
        achieved = math.exp(-_empirical_quantile(sorted(values), lev))
    else:
        achieved = math.fsum(math.exp(-n) for n in values) / len(values)
    return ThicknessDesign(
        thickness=t_star,
        basis=basis,
        requirement=req,
        level=lev if basis == "quantile" else math.nan,
        nominal_pnp=nominal_n,
        achieved_metric=achieved,
        iterations=root.iterations,
        converged=root.converged,
    )


# --------------------------------------------------------------------------
# Bayesian updating of the flux factor
# --------------------------------------------------------------------------


def gamma_prior_from_mean_cv(mean: float, cv: float) -> tuple[float, float]:
    """Return the Gamma ``(shape, rate)`` of a given mean and variation.

    ``shape = 1/cv^2`` and ``rate = shape/mean``. Use it to build a conjugate
    prior for the flux factor with mean ``mean`` (1 for an unbiased model) and
    relative uncertainty ``cv``.

    Args:
        mean: Prior mean, positive.
        cv: Coefficient of variation, positive.

    Returns:
        ``(shape, rate)``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-MMOD-009].

    References:
        [REF-GELMAN-ETAL-2013].

    """
    m = _positive(mean, name="mean")
    c = _positive(cv, name="cv")
    shape = 1.0 / (c * c)
    return shape, shape / m


def baseline_impact_count(
    area: float, exposure_time: float, flux: CumulativeFlux, diameter: float
) -> float:
    """Return the baseline expected count ``A T F(>d)`` of impacts above ``d``.

    The count of impact features larger than the detection size ``d`` on a
    returned surface has mean ``f`` times this baseline when the flux model is
    scaled by ``f``; it is the exposure term of :func:`update_flux_factor`.

    Args:
        area: Inspected area (m^2).
        exposure_time: Exposure in the flux time unit.
        flux: Cumulative flux law.
        diameter: Detection (feature) particle diameter (m).

    Returns:
        ``A T F(>d)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a value is not positive.

    Equation:
        [EQ-AERO-MMOD-009].

    References:
        [REF-NASA-PRA-2011].

    """
    a = _positive(area, name="area")
    t = _positive(exposure_time, name="exposure_time")
    return a * t * flux.cumulative(diameter)


@dataclass(frozen=True, slots=True)
class FluxFactorPosterior:
    """Gamma posterior ``Gamma(a, b)`` of the flux factor.

    Attributes:
        prior_shape: Prior shape ``a_0``.
        prior_rate: Prior rate ``b_0``.
        shape: Posterior shape ``a_0 + sum n_j``.
        rate: Posterior rate ``b_0 + sum mu_j``.
        total_count: ``sum n_j``.
        total_exposure: ``sum mu_j`` (baseline expected counts).

    References:
        [EQ-AERO-MMOD-009], [EQ-AERO-MMOD-010].

    """

    prior_shape: float
    prior_rate: float
    shape: float
    rate: float
    total_count: int
    total_exposure: float

    @property
    def mean(self) -> float:
        """Return the posterior mean ``a / b``.

        References:
            [EQ-AERO-MMOD-009].

        """
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        """Return the posterior variance ``a / b^2``.

        References:
            [EQ-AERO-MMOD-009].

        """
        return self.shape / (self.rate * self.rate)

    def quantile(self, probability: float) -> float:
        """Return a posterior quantile of the flux factor.

        Args:
            probability: Level in ``(0, 1)``.

        Returns:
            The quantile.

        Equation:
            [EQ-AERO-MMOD-009].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        p = _open_probability(probability, name="probability")
        return Gamma(self.shape, self.rate).quantile(p)

    def pnp_mean(self, expected_penetrations: float) -> float:
        """Return the posterior predictive PNP ``(b/(b+N0))^a`` in closed form.

        Args:
            expected_penetrations: Nominal ``N0`` of the future mission (at
                ``f = 1``).

        Returns:
            ``E[exp(-N0 f)]`` over the posterior.

        Equation:
            [EQ-AERO-MMOD-010].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        return mean_pnp_gamma_flux(expected_penetrations, self.shape, self.rate)

    def pnp_credible_bound(
        self, expected_penetrations: float, credibility: float
    ) -> float:
        """Return the PNP exceeded with posterior probability ``credibility``.

        ``PNP = exp(-N0 f)`` is decreasing in ``f``, so ``PNP >= exp(-N0 f_c)``
        with probability ``c`` when ``f_c`` is the posterior ``c``-quantile.

        Args:
            expected_penetrations: Nominal ``N0``.
            credibility: Level in ``(0, 1)``.

        Returns:
            The lower credible bound of PNP.

        Equation:
            [EQ-AERO-MMOD-010].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        n0 = _non_negative(expected_penetrations, name="expected_penetrations")
        return math.exp(-n0 * self.quantile(credibility))

    def predictive_count_pmf(self, count: int, baseline_expected: float) -> float:
        """Return the negative-binomial predictive ``P(K = k)`` of a future count.

        With ``K | f ~ Poisson(mu f)`` and ``f ~ Gamma(a, b)``,
        ``P(K = k) = Gamma(a+k) / (k! Gamma(a)) (b/(b+mu))^a (mu/(b+mu))^k``.

        Args:
            count: ``k >= 0``.
            baseline_expected: ``mu`` (baseline expected count, ``f = 1``).

        Returns:
            The predictive probability.

        Equation:
            [EQ-AERO-MMOD-010].

        References:
            [REF-GELMAN-ETAL-2013].

        """
        k = _integer(count, name="count", minimum=0)
        mu = _positive(baseline_expected, name="baseline_expected")
        a, b = self.shape, self.rate
        log_p = (
            math.lgamma(a + k)
            - math.lgamma(k + 1)
            - math.lgamma(a)
            + a * (math.log(b) - math.log(b + mu))
            + k * (math.log(mu) - math.log(b + mu))
        )
        return math.exp(log_p)


def update_flux_factor(
    prior_shape: float,
    prior_rate: float,
    counts: Sequence[int],
    baseline_expected: Sequence[float],
) -> FluxFactorPosterior:
    """Update the flux factor from impact counts on returned surfaces.

    Model: ``n_j | f ~ Poisson(f mu_j)`` independently, ``f ~ Gamma(a_0, b_0)``.
    The conjugate posterior is ``Gamma(a_0 + sum n_j, b_0 + sum mu_j)``, where
    ``mu_j`` is the baseline expected count of surface ``j`` (see
    :func:`baseline_impact_count`). The inference is conditional on the flux
    shape (the size slope) being right and on a complete count at the detection
    size; a biased detection threshold biases ``f``.

    Args:
        prior_shape: ``a_0 > 0``.
        prior_rate: ``b_0 > 0``.
        counts: Observed counts ``n_j >= 0`` (integers).
        baseline_expected: Baseline expected counts ``mu_j > 0``.

    Returns:
        The posterior.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If lengths differ, a list is empty, or a value is out of
            range.

    Equation:
        [EQ-AERO-MMOD-009].

    References:
        [REF-GELMAN-ETAL-2013].

    """
    a0 = _positive(prior_shape, name="prior_shape")
    b0 = _positive(prior_rate, name="prior_rate")
    n = tuple(_integer(c, name="counts", minimum=0) for c in counts)
    mu = tuple(_positive(m, name="baseline_expected") for m in baseline_expected)
    if not n or len(n) != len(mu):
        raise ValueError("counts and baseline_expected must be equal-length, non-empty")
    total_n = sum(n)
    total_mu = math.fsum(mu)
    return FluxFactorPosterior(a0, b0, a0 + total_n, b0 + total_mu, total_n, total_mu)
