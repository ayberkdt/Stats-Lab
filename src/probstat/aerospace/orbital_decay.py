"""Orbital lifetime under atmospheric drag, with uncertainty, for disposal rules.

A post-mission disposal rule such as "reenter within ``N`` years" is a
statement about the *lifetime* of an orbit, and that lifetime depends
exponentially on the drag environment, which is dominated by the solar cycle
and by an uncertain density model. This module integrates the drag decay of a
near-circular or slightly eccentric orbit, models the thermospheric density as
a (piecewise) exponential profile whose reference density and scale height
depend on a solar-activity index, and wraps the result as a
:class:`probstat.modeling.ProbabilisticModel` so that the probability of
missing the deadline, its sensitivities and the highest compliant altitude can
be computed with the library's reliability and Sobol tools.

Contents:

- :class:`SyntheticThermosphere`: piecewise-exponential density with a
  power-law dependence of the reference density and of the scale heights on
  the activity index; :func:`demonstration_thermosphere` is *synthetic*.
- :class:`SolarActivity`: a synthetic sinusoidal 11-year cycle times a
  log-normal noise factor; :func:`demonstration_solar_activity` is
  *synthetic*.
- :func:`ballistic_coefficient`: ``B = m / (C_D A)``.
- :func:`orbital_decay`: time integration of the circular decay equation or of
  the orbit-averaged ``(a, e)`` equations until the reentry altitude, with an
  adaptive step and a bisection on the final step; :class:`DecayResult`.
- :func:`static_lifetime` (quadrature at a frozen activity) and
  :func:`exponential_lifetime_bounds` (closed form for a constant scale
  height, as rigorous lower/upper bounds): the oracles of the tests.
- :func:`orbital_lifetime_model`: the probabilistic model with uncertain
  ballistic coefficient (``C_D``, area-to-mass ratio, attitude factor),
  initial altitude, solar-cycle phase and amplitude, density error factor and
  optional solar-noise knots; outputs ``lifetime_years`` and
  ``compliance_margin_years``.
- :func:`noncompliance_probability` (FORM, subset simulation or Monte Carlo)
  and :func:`maximum_compliant_altitude` (monotone root find of the highest
  initial altitude whose probability of non-compliance meets a target).

Units. SI throughout (m, s, kg, kg/m^3, m^2/kg); lifetimes and the solar
cycle in Julian years (365.25 d). The activity index is dimensionless in the
sense that only its ratio to ``reference_activity`` matters.

Assumptions and validity.
    Drag is the only perturbation: no oblateness, third body, radiation
    pressure, thrust or attitude dynamics. The atmosphere does not rotate
    (the drag speed is the inertial speed) and is spherically symmetric, with
    no diurnal, latitudinal or geomagnetic variation. ``R_E`` and ``mu`` are
    the constants of :mod:`probstat.aerospace.ascent`. The circular equation
    holds for ``e = 0``; the eccentric option uses the orbit average over one
    revolution with the activity frozen during that revolution, so the period
    must be short compared with the solar-cycle and decay time scales, and the
    quadrature in eccentric anomaly needs more nodes when ``a e`` is large
    compared with the scale height (check the ``quadrature_points``
    convergence). Reentry is declared when the perigee
    altitude reaches ``reentry_altitude``; the decay below that altitude is
    not modelled. The density model error is a single multiplicative factor.
    A circular orbit still up at the integration horizon is flagged by
    ``reentered = False``: its lifetime is censored at the horizon, or, with
    ``static_tail=True``, continued by the frozen-activity lifetime from the
    final altitude (an approximation beyond the horizon).

Coefficients and data are not shipped. Every scale height, density, activity
parameter and noise value produced by the ``demonstration_*`` functions and in
the tests is synthetic and illustrative: it is not fitted to any measurement,
is not NRLMSISE, Jacchia or any other empirical thermosphere model, and is not
real solar-cycle data. Published models and data must be supplied by the user.

Experimental (added in 0.28).

References:
    [REF-KING-HELE-1987], [REF-CURTIS-2014], [REF-MELCHERS-BECK-2018],
    [REF-SALTELLI-2010].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS
from probstat.distributions import Normal
from probstat.foundations.random_streams import RandomStream, resolve_stream
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
    "DecayResult",
    "LifetimeBounds",
    "MaximumAltitude",
    "NoncomplianceEstimate",
    "SECONDS_PER_YEAR",
    "SolarActivity",
    "SyntheticThermosphere",
    "ballistic_coefficient",
    "demonstration_solar_activity",
    "demonstration_thermosphere",
    "exponential_lifetime_bounds",
    "maximum_compliant_altitude",
    "noncompliance_probability",
    "orbital_decay",
    "orbital_lifetime_model",
    "static_lifetime",
    "synthetic_noise_values",
]

SECONDS_PER_YEAR = 365.25 * 86400.0
"""Julian year in seconds."""

_TWO_PI = 2.0 * math.pi

# Five-point Gauss-Legendre rule on [-1, 1].
_GL_NODES = (
    -0.9061798459386640,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.9061798459386640,
)
_GL_WEIGHTS = (
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
)


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------


def _real(value: object, *, name: str) -> float:
    """Return a finite real number (``bool`` rejected).

    References:
        [EQ-AERO-DECAY-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    return require_finite_real(value, name=name)


def _positive(value: object, *, name: str) -> float:
    """Return a finite strictly positive real number.

    References:
        [EQ-AERO-DECAY-001]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: object, *, name: str) -> float:
    """Return a finite non-negative real number.

    References:
        [EQ-AERO-DECAY-001]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _integer(value: object, *, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum`` (``bool`` rejected).

    References:
        [EQ-AERO-DECAY-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _open_probability(value: object, *, name: str) -> float:
    """Return a probability strictly inside ``(0, 1)``.

    References:
        [EQ-AERO-DECAY-007]. No equation is evaluated.

    """
    number = _real(value, name=name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return number


# --------------------------------------------------------------------------
# Synthetic thermosphere and solar activity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticThermosphere:
    """Piecewise-exponential density with an activity-dependent profile.

    Layer ``i`` starts at ``base_altitudes[i]`` and has the scale height
    ``H_i s^q`` with ``s = F / F_ref`` (activity ratio) and ``q =
    scale_height_exponent``. The density at the first base altitude is
    ``rho_0 s^p`` (``p = density_exponent``) and the profile is continuous
    across layer boundaries for every ``F``::

        rho(h, F) = rho_0 s^p exp(-sum_i  (min(h, z_{i+1}) - z_i)^+ / (H_i s^q))

    Below the first base altitude the first layer is extrapolated. The
    relation of the density and of the scale heights to the activity is a
    *user-supplied, synthetic demonstration relation*: ``p`` and ``q`` are
    free parameters, not a fit to any thermosphere model.

    Args:
        base_altitudes: Strictly increasing layer base altitudes (m).
        scale_heights: Scale height of each layer at ``F_ref`` (m).
        reference_density: Density at ``base_altitudes[0]`` at ``F_ref``
            (kg/m^3).
        reference_activity: ``F_ref`` (same unit as the activity index).
        density_exponent: ``p`` (reference density ``~ s^p``).
        scale_height_exponent: ``q`` (scale heights ``~ s^q``).

    References:
        [EQ-AERO-DECAY-002], [REF-KING-HELE-1987].

    """

    base_altitudes: tuple[float, ...]
    scale_heights: tuple[float, ...]
    reference_density: float
    reference_activity: float = 1.0
    density_exponent: float = 0.0
    scale_height_exponent: float = 0.0

    def __post_init__(self) -> None:
        """Validate the layers.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If the layers are empty, not ordered, or not positive.

        References:
            [EQ-AERO-DECAY-002]. No equation is evaluated.

        """
        bases = tuple(_real(v, name="base_altitudes") for v in self.base_altitudes)
        heights = tuple(_positive(v, name="scale_heights") for v in self.scale_heights)
        if not bases or len(bases) != len(heights):
            raise ValueError("base_altitudes and scale_heights need equal length >= 1")
        if any(b1 <= b0 for b0, b1 in zip(bases, bases[1:], strict=False)):
            raise ValueError("base_altitudes must be strictly increasing")
        object.__setattr__(self, "base_altitudes", bases)
        object.__setattr__(self, "scale_heights", heights)
        object.__setattr__(
            self,
            "reference_density",
            _positive(self.reference_density, name="reference_density"),
        )
        object.__setattr__(
            self,
            "reference_activity",
            _positive(self.reference_activity, name="reference_activity"),
        )
        object.__setattr__(
            self,
            "density_exponent",
            _real(self.density_exponent, name="density_exponent"),
        )
        object.__setattr__(
            self,
            "scale_height_exponent",
            _real(self.scale_height_exponent, name="scale_height_exponent"),
        )

    def _layer(self, altitude: float) -> int:
        """Return the index of the layer that contains ``altitude``.

        References:
            [EQ-AERO-DECAY-002]. No equation is evaluated.

        """
        index = 0
        for i, base in enumerate(self.base_altitudes):
            if altitude >= base:
                index = i
            else:
                break
        return index

    def density(self, altitude: float, activity: float) -> float:
        """Return the density (kg/m^3) at an altitude and an activity.

        Args:
            altitude: Geometric altitude (m).
            activity: Activity index ``F > 0``.

        Returns:
            ``rho(h, F)``.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If the activity is not positive.

        Equation:
            [EQ-AERO-DECAY-002].

        References:
            [REF-KING-HELE-1987].

        """
        h = _real(altitude, name="altitude")
        s = _positive(activity, name="activity") / self.reference_activity
        return self._density(h, s)

    def _density(self, altitude: float, ratio: float) -> float:
        """Return the density for a validated altitude and activity ratio.

        References:
            [EQ-AERO-DECAY-002]. No equation is evaluated.

        """
        scale = ratio**self.scale_height_exponent
        bases = self.base_altitudes
        heights = self.scale_heights
        last = len(bases) - 1
        exponent = 0.0
        for i in range(last + 1):
            if i == last or altitude <= bases[i + 1]:
                exponent += (altitude - bases[i]) / (heights[i] * scale)
                break
            exponent += (bases[i + 1] - bases[i]) / (heights[i] * scale)
        return float(
            self.reference_density * ratio**self.density_exponent * math.exp(-exponent)
        )

    def scale_height(self, altitude: float, activity: float) -> float:
        """Return the local scale height (m) at an altitude and an activity.

        Args:
            altitude: Geometric altitude (m).
            activity: Activity index ``F > 0``.

        Returns:
            ``H_i s^q`` of the layer containing the altitude.

        Raises:
            TypeError: If an argument is not a real number.
            ValueError: If the activity is not positive.

        Equation:
            [EQ-AERO-DECAY-002].

        References:
            [REF-KING-HELE-1987].

        """
        h = _real(altitude, name="altitude")
        s = _positive(activity, name="activity") / self.reference_activity
        return float(self.scale_heights[self._layer(h)] * s**self.scale_height_exponent)


def demonstration_thermosphere() -> SyntheticThermosphere:
    """Return a *synthetic* three-layer thermosphere for demonstrations.

    Round numbers chosen only to give drag lifetimes of years to decades at
    400-700 km; they are not fitted to any measurement and are not an
    empirical model.

    Returns:
        The synthetic profile: layers from 150, 300 and 500 km with scale
        heights 30, 55 and 90 km, ``rho_0 = 2e-9 kg/m^3`` at 150 km,
        ``p = 1`` and ``q = 0.5``.

    Equation:
        [EQ-AERO-DECAY-002].

    References:
        [REF-KING-HELE-1987]. Synthetic values, not taken from the book.

    """
    return SyntheticThermosphere(
        base_altitudes=(150.0e3, 300.0e3, 500.0e3),
        scale_heights=(30.0e3, 55.0e3, 90.0e3),
        reference_density=2.0e-9,
        reference_activity=1.0,
        density_exponent=1.0,
        scale_height_exponent=0.5,
    )


@dataclass(frozen=True, slots=True)
class SolarActivity:
    """Synthetic solar-activity index: sinusoidal cycle times log-normal noise.

    ``F(t) = mean (1 + A sin(2 pi (t / P + phase))) exp(sigma n(t))`` with
    ``t`` in years, ``A in [0, 1)``, ``n`` the piecewise-linear interpolation
    of the standard-normal ``noise_values`` placed at ``0, Delta, 2 Delta,
    ...`` (held constant after the last knot). With ``noise_sigma = 0`` or no
    values the factor is one. The noise factor has median one; ``F`` is
    always positive. All parameters are user inputs and the process is
    synthetic: it is not a solar-cycle record.

    Args:
        mean: Cycle mean ``F_mean > 0``.
        amplitude: Relative amplitude ``A``, ``0 <= A < 1``.
        period_years: Cycle period ``P`` (years).
        phase: Cycle phase in cycles (``0.25`` starts at the maximum).
        noise_sigma: Log-normal noise scale ``sigma >= 0``.
        noise_values: Standard-normal knot values ``n_j``.
        noise_interval_years: Knot spacing ``Delta`` (years).

    References:
        [EQ-AERO-DECAY-003].

    """

    mean: float = 1.0
    amplitude: float = 0.0
    period_years: float = 11.0
    phase: float = 0.0
    noise_sigma: float = 0.0
    noise_values: tuple[float, ...] = ()
    noise_interval_years: float = 1.0

    def __post_init__(self) -> None:
        """Validate the process parameters.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a value is out of range.

        References:
            [EQ-AERO-DECAY-003]. No equation is evaluated.

        """
        object.__setattr__(self, "mean", _positive(self.mean, name="mean"))
        amplitude = _real(self.amplitude, name="amplitude")
        if not 0.0 <= amplitude < 1.0:
            raise ValueError("amplitude must lie in [0, 1)")
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(
            self, "period_years", _positive(self.period_years, name="period_years")
        )
        object.__setattr__(self, "phase", _real(self.phase, name="phase"))
        object.__setattr__(
            self, "noise_sigma", _non_negative(self.noise_sigma, name="noise_sigma")
        )
        object.__setattr__(
            self,
            "noise_values",
            tuple(_real(v, name="noise_values") for v in self.noise_values),
        )
        object.__setattr__(
            self,
            "noise_interval_years",
            _positive(self.noise_interval_years, name="noise_interval_years"),
        )

    def index(self, years: float) -> float:
        """Return the activity index ``F`` at a time in years.

        Args:
            years: Time since the start of the integration (years), ``>= 0``.

        Returns:
            ``F(t) > 0``.

        Raises:
            TypeError: If ``years`` is not a real number.
            ValueError: If ``years`` is negative.

        Equation:
            [EQ-AERO-DECAY-003].

        References:
            [REF-KING-HELE-1987]. Synthetic process.

        """
        t = _non_negative(years, name="years")
        return self._index(t)

    def _index(self, years: float) -> float:
        """Return ``F`` at a validated time in years.

        References:
            [EQ-AERO-DECAY-003]. No equation is evaluated.

        """
        value = self.mean
        if self.amplitude != 0.0:
            value *= 1.0 + self.amplitude * math.sin(
                _TWO_PI * (years / self.period_years + self.phase)
            )
        knots = self.noise_values
        if self.noise_sigma != 0.0 and knots:
            position = years / self.noise_interval_years
            j = int(position)
            if j >= len(knots) - 1:
                noise = knots[-1]
            else:
                w = position - j
                noise = (1.0 - w) * knots[j] + w * knots[j + 1]
            value *= math.exp(self.noise_sigma * noise)
        return value


def demonstration_solar_activity(
    *,
    phase: float = 0.0,
    amplitude: float = 0.4,
    noise_sigma: float = 0.0,
    noise_values: Sequence[float] = (),
    noise_interval_years: float = 1.0,
) -> SolarActivity:
    """Return a *synthetic* 11-year activity cycle with mean one.

    Args:
        phase: Cycle phase in cycles.
        amplitude: Relative amplitude.
        noise_sigma: Log-normal noise scale.
        noise_values: Standard-normal knot values.
        noise_interval_years: Knot spacing (years).

    Returns:
        A :class:`SolarActivity` with ``mean = 1`` and ``period = 11`` years.

    Equation:
        [EQ-AERO-DECAY-003].

    References:
        [REF-KING-HELE-1987]. Synthetic values, not solar-cycle data.

    """
    return SolarActivity(
        mean=1.0,
        amplitude=amplitude,
        period_years=11.0,
        phase=phase,
        noise_sigma=noise_sigma,
        noise_values=tuple(noise_values),
        noise_interval_years=noise_interval_years,
    )


def synthetic_noise_values(
    count: int, seed: int | RandomStream | None = 0
) -> tuple[float, ...]:
    """Draw standard-normal knot values for :class:`SolarActivity`.

    Args:
        count: Number of knots.
        seed: Integer seed or random stream.

    Returns:
        ``count`` independent standard-normal draws.

    Raises:
        TypeError: If ``count`` is not an integer.
        ValueError: If ``count`` is negative.

    Equation:
        [EQ-AERO-DECAY-003].

    References:
        [REF-KING-HELE-1987]. Synthetic noise.

    """
    n = _integer(count, name="count", minimum=0)
    generator = resolve_stream(seed).generator()
    return tuple(generator.gauss(0.0, 1.0) for _ in range(n))


def ballistic_coefficient(mass: float, drag_coefficient: float, area: float) -> float:
    """Return the ballistic coefficient ``B = m / (C_D A)`` in kg/m^2.

    Args:
        mass: Mass ``m`` (kg).
        drag_coefficient: Drag coefficient ``C_D`` (dimensionless).
        area: Reference (mean projected) area ``A`` (m^2).

    Returns:
        ``B``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not positive.

    Equation:
        [EQ-AERO-DECAY-001].

    References:
        [REF-KING-HELE-1987], [REF-CURTIS-2014].

    """
    return _positive(mass, name="mass") / (
        _positive(drag_coefficient, name="drag_coefficient")
        * _positive(area, name="area")
    )


# --------------------------------------------------------------------------
# Decay integration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecayResult:
    """Result of :func:`orbital_decay`.

    Attributes:
        lifetime_years: Time to reach the reentry altitude (years); the
            horizon when the orbit did not reenter (censored; see ``static_tail``).
        reentered: Whether the reentry altitude was reached.
        stop_reason: ``"reentry"``, ``"horizon"`` or ``"max_steps"``.
        final_altitude: Altitude (circular) or perigee altitude at the end (m).
        final_eccentricity: Eccentricity at the end.
        steps: Accepted integration steps.
        times_years: Recorded times (empty unless ``record``).
        altitudes: Recorded altitudes or perigee altitudes (m).
        eccentricities: Recorded eccentricities.

    References:
        [EQ-AERO-DECAY-001], [EQ-AERO-DECAY-006].

    """

    lifetime_years: float
    reentered: bool
    stop_reason: str
    final_altitude: float
    final_eccentricity: float
    steps: int
    times_years: tuple[float, ...] = ()
    altitudes: tuple[float, ...] = ()
    eccentricities: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class _Context:
    """Validated inputs of one decay integration.

    References:
        [EQ-AERO-DECAY-001], [EQ-AERO-DECAY-006].

    """

    thermosphere: SyntheticThermosphere
    solar: SolarActivity | None
    constant_activity: float
    density_factor: float
    ballistic: float
    points: int
    clock_offset: float = 0.0

    def ratio(self, seconds: float) -> float:
        """Return the activity ratio ``F / F_ref`` at a time in seconds.

        References:
            [EQ-AERO-DECAY-003]. No equation is evaluated.

        """
        if self.solar is None:
            activity = self.constant_activity
        else:
            activity = self.solar._index(
                (seconds + self.clock_offset) / SECONDS_PER_YEAR
            )
        return activity / self.thermosphere.reference_activity

    def density(self, altitude: float, ratio: float) -> float:
        """Return the density including the model-error factor.

        References:
            [EQ-AERO-DECAY-002]. No equation is evaluated.

        """
        return self.density_factor * self.thermosphere._density(altitude, ratio)

    def circular_rate(self, seconds: float, radius: float) -> float:
        """Return ``da/dt`` (m/s) of a circular orbit of radius ``a``.

        References:
            [EQ-AERO-DECAY-001]. No equation is evaluated.

        """
        rho = self.density(radius - EARTH_RADIUS, self.ratio(seconds))
        return -math.sqrt(EARTH_MU * radius) * rho / self.ballistic

    def averaged_rates(self, seconds: float, a: float, e: float) -> tuple[float, float]:
        """Return orbit-averaged ``(da/dt, de/dt)`` (m/s, 1/s).

        The averages over one revolution are computed by the periodic
        trapezoid rule in the eccentric anomaly ``E`` with ``points`` nodes.

        References:
            [EQ-AERO-DECAY-006]. No equation is evaluated.

        """
        ratio = self.ratio(seconds)
        n = self.points
        speed_scale = math.sqrt(EARTH_MU / a)
        da_sum = 0.0
        de_sum = 0.0
        for k in range(n):
            big_e = _TWO_PI * (k + 0.5) / n
            cos_e = math.cos(big_e)
            r_over_a = 1.0 - e * cos_e
            rho = self.density(a * r_over_a - EARTH_RADIUS, ratio)
            v = speed_scale * math.sqrt((1.0 + e * cos_e) / r_over_a)
            da_sum += rho * v**3 * r_over_a
            de_sum += rho * v * cos_e
        da = -da_sum / n * a * a / (EARTH_MU * self.ballistic)
        de = -de_sum / n * (1.0 - e * e) / self.ballistic
        return da, de


def _rk4_circular(ctx: _Context, t: float, a: float, k1: float, dt: float) -> float:
    """Advance the circular radius by one RK4 step given ``k1 = f(t, a)``.

    References:
        [EQ-AERO-DECAY-001]. Classical Runge-Kutta; no new equation.

    """
    k2 = ctx.circular_rate(t + 0.5 * dt, a + 0.5 * dt * k1)
    k3 = ctx.circular_rate(t + 0.5 * dt, a + 0.5 * dt * k2)
    k4 = ctx.circular_rate(t + dt, a + dt * k3)
    return a + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rk4_eccentric(
    ctx: _Context, t: float, a: float, e: float, k1: tuple[float, float], dt: float
) -> tuple[float, float]:
    """Advance ``(a, e)`` by one RK4 step given ``k1 = f(t, a, e)``.

    References:
        [EQ-AERO-DECAY-006]. Classical Runge-Kutta; no new equation.

    """
    h = 0.5 * dt
    k2 = ctx.averaged_rates(t + h, a + h * k1[0], e + h * k1[1])
    k3 = ctx.averaged_rates(t + h, a + h * k2[0], e + h * k2[1])
    k4 = ctx.averaged_rates(t + dt, a + dt * k3[0], e + dt * k3[1])
    a_new = a + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    e_new = e + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    return a_new, e_new


def orbital_decay(
    initial_altitude: float,
    ballistic: float,
    thermosphere: SyntheticThermosphere,
    solar: SolarActivity | float | None = None,
    *,
    eccentricity: float = 0.0,
    reentry_altitude: float = 120.0e3,
    horizon_years: float = 100.0,
    density_factor: float = 1.0,
    altitude_fraction: float = 0.05,
    max_step_years: float | None = None,
    quadrature_points: int = 64,
    max_steps: int = 200000,
    record: bool = False,
    static_tail: bool = False,
) -> DecayResult:
    """Integrate the drag decay of an orbit until the reentry altitude.

    For ``eccentricity = 0`` the circular equation
    ``da/dt = -sqrt(mu a) rho(a - R_E, F(t)) / B`` is integrated. For
    ``eccentricity > 0`` the *orbit-averaged* equations for ``(a, e)`` (derived
    in the module notes from the energy and angular-momentum balance of a
    drag acceleration ``-rho v v / (2 B)``) are integrated, with ``initial_altitude``
    the *perigee* altitude: ``a = (R_E + h_p) / (1 - e)``. Time stepping is
    classical RK4 with ``dt = min(dt_max, f H / |rate|)``, where ``H`` is the
    local scale height and ``rate`` the perigee-radius rate; the step that
    crosses the reentry altitude is shortened by bisection so the lifetime is
    located to about ``1e-10`` of a step. The relation of the activity to the
    density is that of ``thermosphere`` (synthetic).

    Args:
        initial_altitude: Initial altitude (circular) or perigee altitude (m).
        ballistic: Ballistic coefficient ``B = m / (C_D A)`` (kg/m^2).
        thermosphere: Density profile.
        solar: A :class:`SolarActivity` process, a constant activity index, or
            ``None`` for the reference activity.
        eccentricity: Initial eccentricity ``0 <= e < 1``.
        reentry_altitude: Reentry (perigee) altitude (m).
        horizon_years: Integration horizon (years); a longer lifetime is
            censored.
        density_factor: Multiplicative density-model error factor ``> 0``.
        altitude_fraction: Step bound in scale heights, ``f``.
        max_step_years: Maximum step (years); default ``min(0.7, P/16)`` with
            the solar period ``P`` (``0.7`` without a cycle).
        quadrature_points: Eccentric-anomaly nodes of the orbit average.
        max_steps: Step budget.
        record: Keep the trajectory in the result.
        static_tail: For a circular orbit that has not reentered at the
            horizon, add :func:`static_lifetime` from the final altitude at
            the mean activity (the reference activity, or the cycle mean) to
            the lifetime instead of censoring it. The result is then a smooth
            continuation, flagged by ``reentered = False`` and
            ``stop_reason = "horizon_extrapolated"``; it is an approximation
            (frozen activity, no noise, no cycle) that matters only beyond the
            horizon. Ignored for ``eccentricity > 0``.

    Returns:
        The :class:`DecayResult`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If ``initial_altitude`` (perigee) is not above
            ``reentry_altitude``, ``ballistic`` is not positive, or another
            value is out of range.

    Equation:
        [EQ-AERO-DECAY-001] (circular), [EQ-AERO-DECAY-006] (averaged).

    References:
        [REF-KING-HELE-1987], [REF-CURTIS-2014].

    """
    h0 = _real(initial_altitude, name="initial_altitude")
    b = _positive(ballistic, name="ballistic")
    if not isinstance(thermosphere, SyntheticThermosphere):
        raise TypeError("thermosphere must be a SyntheticThermosphere")
    e0 = _real(eccentricity, name="eccentricity")
    if not 0.0 <= e0 < 1.0:
        raise ValueError("eccentricity must lie in [0, 1)")
    h_re = _real(reentry_altitude, name="reentry_altitude")
    if h0 <= h_re:
        raise ValueError("initial_altitude must exceed reentry_altitude")
    horizon = _positive(horizon_years, name="horizon_years") * SECONDS_PER_YEAR
    factor = _positive(density_factor, name="density_factor")
    fraction = _positive(altitude_fraction, name="altitude_fraction")
    points = _integer(quadrature_points, name="quadrature_points", minimum=8)
    budget = _integer(max_steps, name="max_steps", minimum=1)
    if isinstance(solar, SolarActivity):
        process: SolarActivity | None = solar
        constant = 0.0
        natural = min(0.7, solar.period_years / 16.0)
    elif solar is None:
        process = None
        constant = thermosphere.reference_activity
        natural = 0.7
    else:
        process = None
        constant = _positive(solar, name="solar")
        natural = 0.7
    dt_max = (
        natural
        if max_step_years is None
        else _positive(max_step_years, name="max_step_years")
    ) * SECONDS_PER_YEAR
    ctx = _Context(thermosphere, process, constant, factor, b, points)
    if e0 == 0.0:
        circular = _integrate_circular(
            ctx, EARTH_RADIUS + h0, h_re, horizon, fraction, dt_max, budget, record
        )
        if static_tail and circular.stop_reason == "horizon":
            if process is not None:
                tail_activity = process.mean
            else:
                tail_activity = constant
            tail = static_lifetime(
                circular.final_altitude,
                b,
                thermosphere,
                tail_activity,
                reentry_altitude=h_re,
                density_factor=factor,
            )
            return replace(
                circular,
                lifetime_years=circular.lifetime_years + tail,
                stop_reason="horizon_extrapolated",
            )
        return circular
    return _integrate_eccentric(
        ctx,
        (EARTH_RADIUS + h0) / (1.0 - e0),
        e0,
        h_re,
        horizon,
        fraction,
        dt_max,
        budget,
        record,
    )


def _integrate_circular(
    ctx: _Context,
    a0: float,
    h_re: float,
    horizon: float,
    fraction: float,
    dt_max: float,
    budget: int,
    record: bool,
) -> DecayResult:
    """Integrate the circular decay equation (see :func:`orbital_decay`).

    References:
        [EQ-AERO-DECAY-001]. Classical Runge-Kutta; no new equation.

    """
    a_re = EARTH_RADIUS + h_re
    t = 0.0
    a = a0
    steps = 0
    times: list[float] = [0.0] if record else []
    alts: list[float] = [a0 - EARTH_RADIUS] if record else []
    while True:
        if steps >= budget:
            return _finish("max_steps", t, a - EARTH_RADIUS, 0.0, steps, times, alts)
        k1 = ctx.circular_rate(t, a)
        ratio = ctx.ratio(t)
        scale = (
            ctx.thermosphere.scale_heights[ctx.thermosphere._layer(a - EARTH_RADIUS)]
            * ratio**ctx.thermosphere.scale_height_exponent
        )
        dt = min(dt_max, fraction * scale / abs(k1)) if k1 != 0.0 else dt_max
        if t + dt >= horizon:
            dt = horizon - t
        a_new = _rk4_circular(ctx, t, a, k1, dt)
        if a_new <= a_re:
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _rk4_circular(ctx, t, a, k1, mid) <= a_re:
                    hi = mid
                else:
                    lo = mid
            t += 0.5 * (lo + hi)
            steps += 1
            if record:
                times.append(t / SECONDS_PER_YEAR)
                alts.append(h_re)
            return DecayResult(
                t / SECONDS_PER_YEAR,
                True,
                "reentry",
                h_re,
                0.0,
                steps,
                tuple(times),
                tuple(alts),
                tuple(0.0 for _ in times),
            )
        t += dt
        a = a_new
        steps += 1
        if record:
            times.append(t / SECONDS_PER_YEAR)
            alts.append(a - EARTH_RADIUS)
        if t >= horizon:
            return _finish("horizon", t, a - EARTH_RADIUS, 0.0, steps, times, alts)


def _finish(
    reason: str,
    t: float,
    altitude: float,
    eccentricity: float,
    steps: int,
    times: list[float],
    alts: list[float],
    eccs: list[float] | None = None,
) -> DecayResult:
    """Build the result of an orbit that did not reenter.

    References:
        [EQ-AERO-DECAY-001]. No equation is evaluated.

    """
    return DecayResult(
        t / SECONDS_PER_YEAR,
        False,
        reason,
        altitude,
        eccentricity,
        steps,
        tuple(times),
        tuple(alts),
        tuple(eccs) if eccs is not None else tuple(0.0 for _ in times),
    )


def _integrate_eccentric(
    ctx: _Context,
    a0: float,
    e0: float,
    h_re: float,
    horizon: float,
    fraction: float,
    dt_max: float,
    budget: int,
    record: bool,
) -> DecayResult:
    """Integrate the orbit-averaged ``(a, e)`` equations.

    The orbit is declared circular (``e = 0``) when ``e`` reaches ``1e-12`` or
    would become negative within a step; the step is limited so that this
    happens only through the physical asymptotic approach ``e -> 0``.

    References:
        [EQ-AERO-DECAY-006]. Classical Runge-Kutta; no new equation.

    """
    q_re = EARTH_RADIUS + h_re
    t = 0.0
    a, e = a0, e0
    steps = 0
    times: list[float] = [0.0] if record else []
    alts: list[float] = [a0 * (1.0 - e0) - EARTH_RADIUS] if record else []
    eccs: list[float] = [e0] if record else []
    while True:
        perigee = a * (1.0 - e)
        if steps >= budget:
            return _finish(
                "max_steps", t, perigee - EARTH_RADIUS, e, steps, times, alts, eccs
            )
        if e <= 1.0e-12:
            return _shifted_tail(
                ctx, t, a, h_re, horizon, fraction, dt_max, budget - steps, steps
            )
        k1 = ctx.averaged_rates(t, a, e)
        perigee_rate = k1[0] * (1.0 - e) - a * k1[1]
        ratio = ctx.ratio(t)
        scale = (
            ctx.thermosphere.scale_heights[
                ctx.thermosphere._layer(perigee - EARTH_RADIUS)
            ]
            * ratio**ctx.thermosphere.scale_height_exponent
        )
        dt = dt_max
        if perigee_rate != 0.0:
            dt = min(dt, fraction * scale / abs(perigee_rate))
        if k1[1] < 0.0:
            dt = min(dt, 0.2 * e / abs(k1[1]))
        if t + dt >= horizon:
            dt = horizon - t
        a_new, e_new = _rk4_eccentric(ctx, t, a, e, k1, dt)
        if e_new < 0.0:
            raise ValueError("eccentricity became negative: reduce altitude_fraction")
        if a_new * (1.0 - e_new) <= q_re:
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                a_m, e_m = _rk4_eccentric(ctx, t, a, e, k1, mid)
                if a_m * (1.0 - e_m) <= q_re:
                    hi = mid
                else:
                    lo = mid
            t += 0.5 * (lo + hi)
            steps += 1
            if record:
                times.append(t / SECONDS_PER_YEAR)
                alts.append(h_re)
                eccs.append(e_new)
            return DecayResult(
                t / SECONDS_PER_YEAR,
                True,
                "reentry",
                h_re,
                e_new,
                steps,
                tuple(times),
                tuple(alts),
                tuple(eccs),
            )
        t += dt
        a, e = a_new, e_new
        steps += 1
        if record:
            times.append(t / SECONDS_PER_YEAR)
            alts.append(a * (1.0 - e) - EARTH_RADIUS)
            eccs.append(e)
        if t >= horizon:
            return _finish(
                "horizon",
                t,
                a * (1.0 - e) - EARTH_RADIUS,
                e,
                steps,
                times,
                alts,
                eccs,
            )


def _shifted_tail(
    ctx: _Context,
    t: float,
    a: float,
    h_re: float,
    horizon: float,
    fraction: float,
    dt_max: float,
    remaining: int,
    steps: int,
) -> DecayResult:
    """Continue a circularised orbit at the elapsed time ``t``.

    The circular integrator starts its clock at zero, so the solar process is
    evaluated at the elapsed time plus the integrator clock
    (``clock_offset``); the lifetime is reported on the original clock.

    References:
        [EQ-AERO-DECAY-001]. Classical Runge-Kutta; no new equation.

    """
    new_ctx = replace(ctx, clock_offset=ctx.clock_offset + t)
    result = _integrate_circular(
        new_ctx, a, h_re, horizon - t, fraction, dt_max, remaining, False
    )
    return DecayResult(
        t / SECONDS_PER_YEAR + result.lifetime_years,
        result.reentered,
        result.stop_reason,
        result.final_altitude,
        0.0,
        steps + result.steps,
    )


# --------------------------------------------------------------------------
# Static-activity oracles
# --------------------------------------------------------------------------


def static_lifetime(
    initial_altitude: float,
    ballistic: float,
    thermosphere: SyntheticThermosphere,
    activity: float | None = None,
    *,
    reentry_altitude: float = 120.0e3,
    density_factor: float = 1.0,
) -> float:
    """Return the circular-orbit lifetime (years) at a frozen activity.

    With a constant activity the decay equation separates:
    ``T = B int_{a_e}^{a_0} da / (sqrt(mu a) rho(a - R_E))``. The integral is
    evaluated by composite five-point Gauss-Legendre panels, split at the
    layer boundaries with panel width at most half a local scale height; it
    is independent of the time stepping of :func:`orbital_decay` and serves
    as its oracle.

    Args:
        initial_altitude: Initial altitude (m).
        ballistic: Ballistic coefficient ``B`` (kg/m^2).
        thermosphere: Density profile.
        activity: Constant activity index (default the reference activity).
        reentry_altitude: Reentry altitude (m).
        density_factor: Density error factor.

    Returns:
        The lifetime in years.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the initial altitude does not exceed the reentry
            altitude or a value is not positive where required.

    Equation:
        [EQ-AERO-DECAY-004].

    References:
        [REF-KING-HELE-1987], [REF-CURTIS-2014].

    """
    h0 = _real(initial_altitude, name="initial_altitude")
    b = _positive(ballistic, name="ballistic")
    h_re = _real(reentry_altitude, name="reentry_altitude")
    if h0 <= h_re:
        raise ValueError("initial_altitude must exceed reentry_altitude")
    if not isinstance(thermosphere, SyntheticThermosphere):
        raise TypeError("thermosphere must be a SyntheticThermosphere")
    factor = _positive(density_factor, name="density_factor")
    f = (
        thermosphere.reference_activity
        if activity is None
        else _positive(activity, name="activity")
    )
    ratio = f / thermosphere.reference_activity
    cuts = [h_re]
    cuts.extend(z for z in thermosphere.base_altitudes if h_re < z < h0)
    cuts.append(h0)
    total = 0.0
    for lo, hi in zip(cuts, cuts[1:], strict=False):
        local = thermosphere.scale_heights[thermosphere._layer(0.5 * (lo + hi))]
        local *= ratio**thermosphere.scale_height_exponent
        panels = max(1, math.ceil((hi - lo) / (0.5 * local)))
        width = (hi - lo) / panels
        for p in range(panels):
            mid = lo + (p + 0.5) * width
            for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
                h = mid + 0.5 * width * node
                rho = factor * thermosphere._density(h, ratio)
                total += (
                    0.5
                    * width
                    * weight
                    / (math.sqrt(EARTH_MU * (EARTH_RADIUS + h)) * rho)
                )
    return b * total / SECONDS_PER_YEAR


@dataclass(frozen=True, slots=True)
class LifetimeBounds:
    """Closed-form lifetime bounds of :func:`exponential_lifetime_bounds`.

    Attributes:
        lower_years: Lifetime with ``sqrt(mu a)`` frozen at the initial radius
            (the fastest decay rate along the path), a lower bound.
        upper_years: Lifetime with ``sqrt(mu a)`` frozen at the reentry radius
            (the slowest rate), an upper bound.
        midpoint_years: Closed form with ``sqrt(mu a)`` at the mean radius.

    References:
        [EQ-AERO-DECAY-005].

    """

    lower_years: float
    upper_years: float
    midpoint_years: float


def exponential_lifetime_bounds(
    initial_altitude: float,
    ballistic: float,
    reference_altitude: float,
    reference_density: float,
    scale_height: float,
    *,
    reentry_altitude: float = 120.0e3,
) -> LifetimeBounds:
    """Closed-form circular lifetime in an isothermal exponential atmosphere.

    For ``rho = rho_r exp(-(h - h_r)/H)`` and ``sqrt(mu a)`` frozen at ``c``,
    ``da/dt = -c rho / B`` integrates to
    ``T = (H B / c)(1/rho(h_e) - 1/rho(h_0))``. Because ``sqrt(mu a)``
    decreases along the path, freezing it at the initial radius gives a
    lower bound and at the reentry radius an upper bound; the true lifetime
    lies between and the relative gap is about
    ``(h_0 - h_e) / (2 (R_E + h))``.

    Args:
        initial_altitude: Initial altitude ``h_0`` (m).
        ballistic: Ballistic coefficient ``B`` (kg/m^2).
        reference_altitude: Altitude ``h_r`` of the reference density (m).
        reference_density: Density ``rho_r`` at ``h_r`` (kg/m^3).
        scale_height: Constant scale height ``H`` (m).
        reentry_altitude: Reentry altitude ``h_e`` (m).

    Returns:
        The :class:`LifetimeBounds`.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If a value is not positive or ``h_0 <= h_e``.

    Equation:
        [EQ-AERO-DECAY-005].

    References:
        [REF-KING-HELE-1987], [REF-CURTIS-2014].

    """
    h0 = _real(initial_altitude, name="initial_altitude")
    h_e = _real(reentry_altitude, name="reentry_altitude")
    if h0 <= h_e:
        raise ValueError("initial_altitude must exceed reentry_altitude")
    b = _positive(ballistic, name="ballistic")
    h_r = _real(reference_altitude, name="reference_altitude")
    rho_r = _positive(reference_density, name="reference_density")
    big_h = _positive(scale_height, name="scale_height")
    inv_diff = (math.exp((h0 - h_r) / big_h) - math.exp((h_e - h_r) / big_h)) / rho_r

    def at(radius: float) -> float:
        """Lifetime in years with ``sqrt(mu a)`` frozen at ``radius``.

        References:
            [EQ-AERO-DECAY-005]. No equation is evaluated.

        """
        return big_h * b / math.sqrt(EARTH_MU * radius) * inv_diff / SECONDS_PER_YEAR

    return LifetimeBounds(
        lower_years=at(EARTH_RADIUS + h0),
        upper_years=at(EARTH_RADIUS + h_e),
        midpoint_years=at(EARTH_RADIUS + 0.5 * (h0 + h_e)),
    )


# --------------------------------------------------------------------------
# Probabilistic model
# --------------------------------------------------------------------------

_MODEL_QUANTITIES = (
    QuantityOfInterest(
        "lifetime_years", "Orbital lifetime (static tail beyond the horizon)", "yr"
    ),
    QuantityOfInterest("compliance_margin_years", "Deadline minus lifetime", "yr"),
    QuantityOfInterest(
        "log_lifetime",
        "Natural logarithm of lifetime_years (nearly linear in log inputs)",
        "1",
    ),
    QuantityOfInterest(
        "reentered", "1 if the reentry altitude was reached within the horizon", "1"
    ),
)


@dataclass(frozen=True, slots=True)
class _LifetimeForward:
    """Picklable forward map of :func:`orbital_lifetime_model`.

    References:
        [EQ-AERO-DECAY-007].

    """

    thermosphere: SyntheticThermosphere
    deadline_years: float
    horizon_years: float
    reentry_altitude: float
    eccentricity: float
    noise_knots: int
    noise_sigma: float
    solar_mean: float
    solar_period_years: float
    noise_interval_years: float
    altitude_fraction: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Return the lifetime, the compliance margin and the reentry flag.

        Args:
            parameters: The model inputs of :func:`orbital_lifetime_model`.

        Returns:
            The quantities of interest.

        Raises:
            ValueError: If an input is out of range.

        Equation:
            [EQ-AERO-DECAY-007].

        References:
            [REF-KING-HELE-1987].

        """
        cd = _positive(parameters["drag_coefficient"], name="drag_coefficient")
        a_over_m = _positive(parameters["area_to_mass"], name="area_to_mass")
        attitude = _positive(parameters["attitude_factor"], name="attitude_factor")
        solar = SolarActivity(
            mean=self.solar_mean,
            amplitude=_real(parameters["solar_amplitude"], name="solar_amplitude"),
            period_years=self.solar_period_years,
            phase=_real(parameters["solar_phase"], name="solar_phase"),
            noise_sigma=self.noise_sigma,
            noise_values=tuple(
                _real(parameters[f"solar_noise_{j}"], name="solar_noise")
                for j in range(self.noise_knots)
            ),
            noise_interval_years=self.noise_interval_years,
        )
        result = orbital_decay(
            _real(parameters["initial_altitude"], name="initial_altitude"),
            1.0 / (cd * attitude * a_over_m),
            self.thermosphere,
            solar,
            eccentricity=self.eccentricity,
            reentry_altitude=self.reentry_altitude,
            horizon_years=self.horizon_years,
            density_factor=_positive(
                parameters["density_factor"], name="density_factor"
            ),
            altitude_fraction=self.altitude_fraction,
            static_tail=True,
        )
        life = result.lifetime_years
        return {
            "lifetime_years": life,
            "compliance_margin_years": self.deadline_years - life,
            "log_lifetime": math.log(life),
            "reentered": 1.0 if result.reentered else 0.0,
        }


_INPUT_ORDER = (
    ("initial_altitude", UncertaintyNature.ALEATORY),
    ("drag_coefficient", UncertaintyNature.EPISTEMIC),
    ("area_to_mass", UncertaintyNature.EPISTEMIC),
    ("attitude_factor", UncertaintyNature.ALEATORY),
    ("solar_phase", UncertaintyNature.ALEATORY),
    ("solar_amplitude", UncertaintyNature.EPISTEMIC),
    ("density_factor", UncertaintyNature.EPISTEMIC),
)


def orbital_lifetime_model(
    thermosphere: SyntheticThermosphere,
    deadline_years: float,
    *,
    initial_altitude: Marginal | float,
    drag_coefficient: Marginal | float,
    area_to_mass: Marginal | float,
    attitude_factor: Marginal | float = 1.0,
    solar_phase: Marginal | float = 0.0,
    solar_amplitude: Marginal | float = 0.4,
    density_factor: Marginal | float = 1.0,
    solar_mean: float = 1.0,
    solar_period_years: float = 11.0,
    noise_sigma: float = 0.0,
    noise_knots: int = 0,
    horizon_years: float | None = None,
    reentry_altitude: float = 120.0e3,
    eccentricity: float = 0.0,
    altitude_fraction: float = 0.05,
    name: str = "orbital-lifetime",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build a :class:`~probstat.modeling.ProbabilisticModel` of the lifetime.

    Inputs (each a distribution or a plain number; at least one random):

    - ``initial_altitude`` (m; perigee altitude if ``eccentricity > 0``),
    - ``drag_coefficient`` ``C_D``, ``area_to_mass`` ``A/m`` (m^2/kg) and
      ``attitude_factor`` (multiplies the area; mean projected over
      reference area): ``B = 1 / (C_D f_att A/m)``,
    - ``solar_phase`` (cycles) and ``solar_amplitude`` ``A in [0, 1)`` of the
      synthetic cycle,
    - ``density_factor`` (multiplicative density-model error),
    - ``solar_noise_0 .. solar_noise_{k-1}`` (standard normal, created when
      ``noise_knots = k > 0``): the knots of the log-normal activity noise of
      scale ``noise_sigma`` spread uniformly over the horizon.

    Quantities of interest: ``lifetime_years`` (beyond the horizon extended by
    the static tail of :func:`orbital_decay`),
    ``compliance_margin_years = deadline - lifetime``, ``log_lifetime`` and
    ``reentered``. The
    event of non-compliance is ``ModelLimitState(model, "lifetime_years",
    deadline, "above")``. Inputs outside the physical range (for example an
    amplitude above one drawn from an unbounded law) make the forward map
    raise ``ValueError``, handled by ``on_failure``. The default horizon is
    twice the deadline. An orbit still up at the horizon (``reentered = 0``;
    only circular orbits) gets the static-tail continuation of
    :func:`orbital_decay`, so the lifetime stays smooth and strictly
    monotone instead of being censored; the continuation is approximate, but
    it only affects values far beyond the deadline, never the event
    ``lifetime > deadline``.

    Args:
        thermosphere: Synthetic density profile.
        deadline_years: Disposal deadline (years).
        initial_altitude: Initial altitude (m).
        drag_coefficient: ``C_D``.
        area_to_mass: ``A/m`` (m^2/kg).
        attitude_factor: Area multiplier.
        solar_phase: Cycle phase (cycles).
        solar_amplitude: Relative cycle amplitude.
        density_factor: Density error factor.
        solar_mean: Cycle mean activity.
        solar_period_years: Cycle period (years).
        noise_sigma: Log-normal noise scale.
        noise_knots: Number of noise knots (0 disables the noise).
        horizon_years: Horizon (default ``2 * deadline_years``).
        reentry_altitude: Reentry altitude (m).
        eccentricity: Fixed initial eccentricity (0 = circular).
        altitude_fraction: Step bound in scale heights.
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        The probabilistic model.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a fixed value is out of range or no input is random.

    Equation:
        [EQ-AERO-DECAY-007].

    References:
        [REF-KING-HELE-1987], [REF-SALTELLI-2010], [REF-MELCHERS-BECK-2018].

    """
    if not isinstance(thermosphere, SyntheticThermosphere):
        raise TypeError("thermosphere must be a SyntheticThermosphere")
    deadline = _positive(deadline_years, name="deadline_years")
    horizon = (
        2.0 * deadline
        if horizon_years is None
        else _positive(horizon_years, name="horizon_years")
    )
    knots = _integer(noise_knots, name="noise_knots", minimum=0)
    sigma = _non_negative(noise_sigma, name="noise_sigma")
    if knots == 1:
        raise ValueError("noise_knots must be 0 or at least 2")
    _positive(solar_mean, name="solar_mean")
    _positive(solar_period_years, name="solar_period_years")
    h_re = _real(reentry_altitude, name="reentry_altitude")
    e0 = _real(eccentricity, name="eccentricity")
    if not 0.0 <= e0 < 1.0:
        raise ValueError("eccentricity must lie in [0, 1)")
    given: dict[str, Marginal | float] = {
        "initial_altitude": initial_altitude,
        "drag_coefficient": drag_coefficient,
        "area_to_mass": area_to_mass,
        "attitude_factor": attitude_factor,
        "solar_phase": solar_phase,
        "solar_amplitude": solar_amplitude,
        "density_factor": density_factor,
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
    for j in range(knots):
        random_inputs.append(
            RandomInput(
                f"solar_noise_{j}",
                Normal(0.0, 1.0),
                nature=UncertaintyNature.ALEATORY,
            )
        )
    if not random_inputs:
        raise ValueError("at least one input must be uncertain")
    if fixed.get("initial_altitude") is not None and fixed["initial_altitude"] <= h_re:
        raise ValueError("initial_altitude must exceed reentry_altitude")
    forward = _LifetimeForward(
        thermosphere=thermosphere,
        deadline_years=deadline,
        horizon_years=horizon,
        reentry_altitude=h_re,
        eccentricity=e0,
        noise_knots=knots,
        noise_sigma=sigma,
        solar_mean=float(solar_mean),
        solar_period_years=float(solar_period_years),
        noise_interval_years=horizon / (knots - 1) if knots > 1 else 1.0,
        altitude_fraction=_positive(altitude_fraction, name="altitude_fraction"),
    )
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        forward,
        _MODEL_QUANTITIES,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


# --------------------------------------------------------------------------
# Compliance probability and the highest compliant altitude
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NoncomplianceEstimate:
    """Estimate of ``P(lifetime > deadline)``.

    Attributes:
        probability: The estimate.
        method: ``"form"``, ``"subset"`` or ``"monte_carlo"``.
        standard_error: Standard error (``nan`` for FORM, which has none).
        converged: Whether the method reported convergence.
        evaluations: Model evaluations.

    References:
        [EQ-AERO-DECAY-007], [REF-MELCHERS-BECK-2018].

    """

    probability: float
    method: str
    standard_error: float
    converged: bool
    evaluations: int


def noncompliance_probability(
    model: ProbabilisticModel,
    deadline_years: float,
    *,
    method: Literal["form", "subset", "monte_carlo"] = "form",
    samples: int = 1000,
    seed: int | RandomStream | None = 0,
    form_relative_step: float = 1.0e-4,
) -> NoncomplianceEstimate:
    """Estimate ``P(lifetime > deadline)`` for an :func:`orbital_lifetime_model`.

    The failure event is ``lifetime_years >= deadline``, evaluated on the
    equivalent monotone quantity ``log_lifetime >= ln(deadline)`` (the same
    limit-state surface, but a better scaled function for the HL-RF
    iteration). FORM uses the
    finite-difference HL-RF search with a coarse relative step (default
    ``1e-4``) because the integrator's adaptive step makes the lifetime smooth
    only to about ``1e-8``; the first-order approximation is exact only for a
    half-space in standard-normal space, so compare with simulation.

    Args:
        model: Model with a ``lifetime_years`` output.
        deadline_years: Disposal deadline (years).
        method: ``"form"``, ``"subset"`` (``samples`` per level) or
            ``"monte_carlo"``.
        samples: Sample size of the simulation methods.
        seed: Seed or stream of the simulation methods.
        form_relative_step: Finite-difference step of FORM.

    Returns:
        The :class:`NoncomplianceEstimate`.

    Raises:
        TypeError: If ``model`` is not a probabilistic model.
        ValueError: If the method is unknown or a value is out of range.

    Equation:
        [EQ-AERO-DECAY-007].

    References:
        [REF-HASOFER-LIND-1974], [REF-AU-BECK-2001], [REF-MELCHERS-BECK-2018].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    deadline = _positive(deadline_years, name="deadline_years")
    state = ModelLimitState(model, "log_lifetime", math.log(deadline), "above")
    if method == "form":
        form = form_hlrf(
            state,
            relative_step=_positive(form_relative_step, name="form_relative_step"),
        )
        return NoncomplianceEstimate(
            form.probability, "form", math.nan, form.converged, form.evaluations
        )
    count = _integer(samples, name="samples", minimum=2)
    if method == "subset":
        sub = subset_simulation(state, samples_per_level=count, seed=seed)
        return NoncomplianceEstimate(
            sub.probability,
            "subset",
            sub.coefficient_of_variation * sub.probability,
            sub.converged,
            sub.evaluations,
        )
    if method == "monte_carlo":
        mc = model_monte_carlo(state, count, seed=seed)
        p = mc.probability
        return NoncomplianceEstimate(
            p, "monte_carlo", math.sqrt(p * (1.0 - p) / count), True, mc.evaluations
        )
    raise ValueError("method must be 'form', 'subset' or 'monte_carlo'")


@dataclass(frozen=True, slots=True)
class MaximumAltitude:
    """Result of :func:`maximum_compliant_altitude`.

    Attributes:
        altitude: Highest tested altitude with ``P <= target`` (m).
        probability: Estimated ``P`` at ``altitude``.
        upper_altitude: Lowest tested altitude with ``P > target`` (m); the
            true maximum lies in ``[altitude, upper_altitude)``.
        target: Target probability.
        method: Probability method.
        iterations: Bisection iterations.
        evaluations: Probability evaluations.

    References:
        [EQ-AERO-DECAY-008].

    """

    altitude: float
    probability: float
    upper_altitude: float
    target: float
    method: str
    iterations: int
    evaluations: int


def maximum_compliant_altitude(
    build: Callable[[float], ProbabilisticModel],
    deadline_years: float,
    target_probability: float,
    bracket: tuple[float, float],
    *,
    method: Literal["form", "subset", "monte_carlo"] = "form",
    samples: int = 1000,
    seed: int | RandomStream | None = 0,
    altitude_tolerance: float = 500.0,
    max_iterations: int = 60,
) -> MaximumAltitude:
    """Return the highest initial altitude with ``P(non-compliance) <= target``.

    ``build(h)`` returns the lifetime model for the nominal initial altitude
    ``h`` (m); changing the area-to-mass ratio inside ``build`` models a drag
    sail. For fixed other inputs the lifetime of a drag-only orbit is
    non-decreasing in the initial altitude, because two solutions of the
    scalar decay equation cannot cross, so with the *same seed* (common
    random numbers) the Monte Carlo probability is a non-decreasing step
    function of ``h``, and so is the FORM probability up to its
    approximation. The root of ``P(h) = target`` is therefore found by
    bisection (not a secant method, which a step function can defeat) to
    ``altitude_tolerance``.

    Args:
        build: Maps an altitude to a model.
        deadline_years: Disposal deadline (years).
        target_probability: Allowed ``P(lifetime > deadline)``, in ``(0, 1)``.
        bracket: ``(low, high)`` altitudes (m) with ``P(low) <= target <
            P(high)``.
        method: Probability method.
        samples: Sample size of the simulation methods.
        seed: Seed (common random numbers across altitudes).
        altitude_tolerance: Bracket width at termination (m).
        max_iterations: Bisection budget.

    Returns:
        The :class:`MaximumAltitude`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the bracket does not enclose the target.

    Equation:
        [EQ-AERO-DECAY-008].

    References:
        [REF-KING-HELE-1987], [REF-MELCHERS-BECK-2018].

    """
    if not callable(build):
        raise TypeError("build must be callable")
    target = _open_probability(target_probability, name="target_probability")
    tolerance = _positive(altitude_tolerance, name="altitude_tolerance")
    budget = _integer(max_iterations, name="max_iterations", minimum=1)
    lo = _real(bracket[0], name="bracket")
    hi = _real(bracket[1], name="bracket")
    if lo >= hi:
        raise ValueError("bracket must satisfy low < high")
    count = 0

    def probability(h: float) -> float:
        """Return the non-compliance probability at nominal altitude ``h``.

        References:
            [EQ-AERO-DECAY-007]. No equation is evaluated.

        """
        nonlocal count
        count += 1
        return noncompliance_probability(
            build(h), deadline_years, method=method, samples=samples, seed=seed
        ).probability

    p_lo = probability(lo)
    p_hi = probability(hi)
    if p_lo > target:
        raise ValueError("P at the lower bracket end already exceeds the target")
    if p_hi <= target:
        raise ValueError("P at the upper bracket end does not exceed the target")
    iterations = 0
    while hi - lo > tolerance and iterations < budget:
        mid = 0.5 * (lo + hi)
        p_mid = probability(mid)
        if p_mid <= target:
            lo, p_lo = mid, p_mid
        else:
            hi = mid
        iterations += 1
    return MaximumAltitude(lo, p_lo, hi, target, method, iterations, count)
