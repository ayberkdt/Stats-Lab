"""Wind-load dispersion of a planar launch-vehicle ascent (synthetic wind).

The module evaluates the aerodynamic load indicator ``q alpha`` of a
launch vehicle flying through a random horizontal wind profile, and wraps it as
a :class:`~probstat.modeling.ProbabilisticModel` so that FORM, subset
simulation, Monte Carlo, Sobol analysis and extreme-value fits act on it. All
quantities are SI (m, s, kg, Pa, rad).

**Everything here is synthetic.** The wind field is a demonstration random
process (a mean profile with a jet-stream-like peak plus a Gaussian
perturbation), *not* a climatology; the parameter values of
:data:`SYNTHETIC_DEMONSTRATION_WIND` and every load limit used in the tests are
invented to exercise the algorithms and say nothing about any real launch
site, season or vehicle.

Model.

- Wind field (EQ-AERO-WIND-001): the horizontal wind in the trajectory plane,
  positive along the downrange direction (a tail wind), is
  ``w(h) = a [ w_jet exp(-(h - h_jet)^2 / (2 s^2)) + sigma sum_k sqrt(lam_k)
  xi_k phi_k(h) ] T(h)`` with ``phi_0 = 1``, ``phi_k = sqrt(2) cos(k pi h / H)``,
  spectral weights ``lam_k`` proportional to a squared-exponential spectrum
  ``exp(-(k pi ell / H)^2 / 2)`` normalised to sum 1, independent standard
  normal coordinates ``xi_k`` and a smooth taper ``T`` that is 1 below
  ``0.75 H`` and falls to 0 at the top altitude ``H``. The factor ``a`` is a
  global wind amplitude (1 by default). The ``xi_k`` are the only random
  wind inputs, so a standard-normal-space method (FORM, subset simulation)
  acts on them directly. The cosine series is the exact Karhunen-Loeve
  expansion of the stationary field it defines (covariance
  ``sigma^2 sum lam_k phi_k(h) phi_k(h')``), with a pointwise variance that
  oscillates around ``sigma^2`` and doubles near ``h = 0`` and ``h = H`` (the
  usual reflection effect of a cosine basis).
- Relative flow (EQ-AERO-WIND-002): with inertial speed ``v`` and flight-path
  angle ``gamma`` the air-relative velocity is
  ``(v cos(gamma) - w, v sin(gamma))`` (the Earth does not rotate in the
  parent ascent model). The body axis is assumed to lie along the *inertial*
  velocity, the zero-incidence gravity-turn attitude of
  :mod:`probstat.aerospace.ascent`. The angle of attack is the angle from this
  axis to the relative velocity,
  ``alpha = atan2(w sin(gamma), v - w cos(gamma))``, and
  ``V_rel^2 = v^2 - 2 v w cos(gamma) + w^2``.
- Loads (EQ-AERO-WIND-003): ``q = rho V_rel^2 / 2`` and the structural
  indicator ``q |alpha|`` (Pa rad), the standard first measure of the
  bending-moment demand on a slender body; its maximum over the flight and the
  altitude of that maximum are reported.
- Small-wind limit (EQ-AERO-WIND-004): ``alpha = w sin(gamma) / v + O(w^2)`` and
  ``q = q_0 + O(w)``, so ``q |alpha| = q_0 |w| sin(gamma) / v + O(w^2)``,
  proportional to the wind amplitude ``a``.

Approximation (stated, not hidden). The trajectory is the *no-wind* flight of
:func:`probstat.aerospace.ascent.simulate_ascent` (drag and thrust follow the
inertial speed), and the wind enters only the load evaluation along that
trajectory: the vehicle is assumed to be flown, by an ideal autopilot, on the
nominal gravity-turn path with its body axis along the inertial velocity. The
neglected trajectory perturbation due to the wind-modified drag is
``O(w)`` and multiplies a quantity (``alpha``) that is itself ``O(w)``, so
it changes ``q |alpha|`` at order ``w^2`` only (an argument, not a measured
bound). There is no load relief, lift, gust response, structural dynamics or
Earth rotation. The load maximum is taken on the integration grid of the
ascent (``max_step``), so it is a lower bound on the continuous maximum and the
model output is piecewise smooth, not smooth, in its inputs. Instants with
speed below ``min_speed`` (the pad and the first seconds of rise, where
``alpha`` is not meaningful and the vehicle is still restrained) are excluded
from the ``q |alpha|`` maximum, though not from ``q``.

Experimental (added in 0.28).

References:
    [REF-CURTIS-2014], [REF-SUTTON-BIBLARZ-2017],
    [REF-US-STANDARD-ATMOSPHERE-1976], [REF-GHANEM-SPANOS-1991],
    [REF-COLES-2001].

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from probstat.aerospace.ascent import (
    DragModel,
    GuidanceSpec,
    VehicleSpec,
    atmosphere,
    simulate_ascent,
)
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
from probstat.risk import fit_gev, fit_gpd
from probstat.uq import FailurePolicy, QuantityOfInterest

__all__ = [
    "SYNTHETIC_DEMONSTRATION_WIND",
    "SyntheticWindField",
    "WindExtremeStudy",
    "WindLoadHistory",
    "WindReliabilityStudy",
    "relative_flow",
    "small_wind_slope",
    "wind_extreme_study",
    "wind_load_history",
    "wind_load_limit_state",
    "wind_loads_model",
    "wind_reliability_study",
]


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive float.

    Args:
        value: Candidate real number.
        name: Argument name for messages.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If it is not finite and positive.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative float.

    Args:
        value: Candidate real number.
        name: Argument name for messages.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If it is negative or not finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


# ---------------------------------------------------------------------------
# Synthetic wind field
# ---------------------------------------------------------------------------

_TAPER_START = 0.75  # fraction of the top altitude where the taper begins


@dataclass(frozen=True, slots=True)
class SyntheticWindField:
    """Synthetic random horizontal wind profile ``w(h)`` (not a climatology).

    ``w(h) = a T(h) [ w_jet exp(-(h - h_jet)^2 / (2 s^2)) + sigma sum_k
    sqrt(lam_k) xi_k phi_k(h) ]`` with the cosine basis and spectral weights
    of the module docstring. Positive wind is a tail wind along the downrange
    direction. All parameters are inputs of a demonstration; see
    :data:`SYNTHETIC_DEMONSTRATION_WIND` for a labelled set.

    Args:
        jet_speed: Peak speed ``w_jet`` of the mean profile (m/s); any sign.
        jet_altitude: Altitude ``h_jet`` of the mean peak (m), ``>= 0``.
        jet_width: Gaussian width ``s`` of the mean peak (m), ``> 0``.
        fluctuation_std: Root-mean-square perturbation ``sigma`` (m/s), ``>= 0``.
        correlation_length: Length scale ``ell`` of the squared-exponential
            spectrum (m), ``> 0``.
        top_altitude: Altitude ``H`` (m) above which the wind is zero, ``> 0``.
        terms: Number ``K`` of cosine terms (random coordinates), ``>= 1``.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is out of range, or the truncation retains
            less than 95 % of the spectral weight (increase ``terms`` or
            ``correlation_length``).

    Equation:
        [EQ-AERO-WIND-001].

    References:
        [REF-GHANEM-SPANOS-1991, Chapter 2].

    """

    jet_speed: float
    jet_altitude: float
    jet_width: float
    fluctuation_std: float
    correlation_length: float
    top_altitude: float
    terms: int = 8

    def __post_init__(self) -> None:
        """Validate the parameters and the spectral truncation.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range or the truncation is too
                coarse for the correlation length.

        References:
            [EQ-AERO-WIND-001]. No equation is evaluated.

        """
        require_finite_real(self.jet_speed, name="jet_speed")
        _non_negative(self.jet_altitude, "jet_altitude")
        _positive(self.jet_width, "jet_width")
        _non_negative(self.fluctuation_std, "fluctuation_std")
        _positive(self.correlation_length, "correlation_length")
        _positive(self.top_altitude, "top_altitude")
        if isinstance(self.terms, bool) or not isinstance(self.terms, int):
            raise TypeError("terms must be an integer")
        if self.terms < 1:
            raise ValueError("terms must be at least 1")
        if self.retained_spectral_fraction < 0.95:
            raise ValueError(
                "the cosine truncation keeps less than 95 % of the spectral "
                "weight: increase terms or correlation_length"
            )

    def _raw_weights(self, count: int) -> tuple[float, ...]:
        """Return unnormalised spectral weights ``exp(-(k pi ell / H)^2 / 2)``.

        Args:
            count: Number of weights ``k = 0..count-1``.

        Returns:
            The weights.

        References:
            [EQ-AERO-WIND-001].

        """
        scale = math.pi * self.correlation_length / self.top_altitude
        return tuple(math.exp(-0.5 * (k * scale) ** 2) for k in range(count))

    @property
    def retained_spectral_fraction(self) -> float:
        """Fraction of the spectral weight kept by ``terms`` cosines.

        Computed against a 2000-term reference sum; a value near 1 means the
        truncated expansion represents the intended correlation length.

        References:
            [EQ-AERO-WIND-001].

        """
        total = math.fsum(self._raw_weights(2000))
        return math.fsum(self._raw_weights(self.terms)) / total

    @property
    def weights(self) -> tuple[float, ...]:
        """Normalised spectral weights ``lam_k`` (sum 1) of the kept terms.

        References:
            [EQ-AERO-WIND-001].

        """
        raw = self._raw_weights(self.terms)
        total = math.fsum(raw)
        return tuple(r / total for r in raw)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        """Names ``wind_1 .. wind_K`` of the standard-normal coordinates.

        References:
            [EQ-AERO-WIND-001].

        """
        return tuple(f"wind_{k + 1}" for k in range(self.terms))

    def taper(self, altitude: float) -> float:
        """Smooth taper ``T(h)``: 1 below ``0.75 H``, 0 at and above ``H``.

        Args:
            altitude: Altitude (m); negative values are treated as 0.

        Returns:
            The taper in ``[0, 1]``.

        Equation:
            [EQ-AERO-WIND-001].

        References:
            [REF-GHANEM-SPANOS-1991].

        """
        h = require_finite_real(altitude, name="altitude")
        start = _TAPER_START * self.top_altitude
        if h <= start:
            return 1.0
        if h >= self.top_altitude:
            return 0.0
        t = (h - start) / (self.top_altitude - start)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def mean(self, altitude: float) -> float:
        """Mean wind ``E[w(h)]`` (m/s) at amplitude 1.

        Args:
            altitude: Altitude (m).

        Returns:
            The tapered Gaussian jet profile.

        Equation:
            [EQ-AERO-WIND-001].

        References:
            [REF-GHANEM-SPANOS-1991].

        """
        h = require_finite_real(altitude, name="altitude")
        z = (h - self.jet_altitude) / self.jet_width
        return self.jet_speed * math.exp(-0.5 * z * z) * self.taper(h)

    def perturbation(self, altitude: float, coordinates: Sequence[float]) -> float:
        """Random part ``sigma T(h) sum_k sqrt(lam_k) xi_k phi_k(h)`` (m/s).

        Args:
            altitude: Altitude (m).
            coordinates: The ``K`` standard-normal coordinates ``xi_k``.

        Returns:
            The perturbation at ``altitude``.

        Raises:
            ValueError: If ``coordinates`` has the wrong length or a
                non-finite entry.

        Equation:
            [EQ-AERO-WIND-001].

        References:
            [REF-GHANEM-SPANOS-1991, Chapter 2].

        """
        h = require_finite_real(altitude, name="altitude")
        if len(coordinates) != self.terms:
            raise ValueError(f"coordinates must have {self.terms} entries")
        xi = [require_finite_real(c, name="coordinate") for c in coordinates]
        taper = self.taper(h)
        if taper == 0.0 or self.fluctuation_std == 0.0:
            return 0.0
        lam = self.weights
        arg = math.pi * min(max(h, 0.0), self.top_altitude) / self.top_altitude
        total = math.sqrt(lam[0]) * xi[0]
        for k in range(1, self.terms):
            total += math.sqrt(2.0 * lam[k]) * xi[k] * math.cos(k * arg)
        return self.fluctuation_std * taper * total

    def speed(
        self,
        altitude: float,
        coordinates: Sequence[float],
        amplitude: float = 1.0,
    ) -> float:
        """Wind speed ``w(h) = a (mean + perturbation)`` in m/s.

        Args:
            altitude: Altitude (m).
            coordinates: The ``K`` standard-normal coordinates.
            amplitude: Global factor ``a`` (any finite real; 0 is no wind).

        Returns:
            The signed horizontal wind (positive is a tail wind).

        Equation:
            [EQ-AERO-WIND-001].

        References:
            [REF-GHANEM-SPANOS-1991].

        """
        a = require_finite_real(amplitude, name="amplitude")
        return a * (self.mean(altitude) + self.perturbation(altitude, coordinates))

    def profile(
        self,
        altitudes: Sequence[float],
        coordinates: Sequence[float],
        amplitude: float = 1.0,
    ) -> tuple[float, ...]:
        """Wind speeds at several altitudes.

        Args:
            altitudes: Altitudes (m).
            coordinates: The ``K`` standard-normal coordinates.
            amplitude: Global factor ``a``.

        Returns:
            One wind speed per altitude.

        Equation:
            [EQ-AERO-WIND-001].

        References:
            [REF-GHANEM-SPANOS-1991].

        """
        return tuple(self.speed(h, coordinates, amplitude) for h in altitudes)


SYNTHETIC_DEMONSTRATION_WIND = SyntheticWindField(
    jet_speed=35.0,
    jet_altitude=12000.0,
    jet_width=3500.0,
    fluctuation_std=6.0,
    correlation_length=5000.0,
    top_altitude=40000.0,
    terms=8,
)
"""SYNTHETIC demonstration wind: a 35 m/s jet at 12 km plus 6 m/s fluctuations.

Invented numbers for exercising the algorithms; not a measured or published
wind climatology and not tied to any site or season.
"""


# ---------------------------------------------------------------------------
# Relative flow and load history
# ---------------------------------------------------------------------------


def relative_flow(
    speed: float, flight_path_angle: float, wind: float
) -> tuple[float, float]:
    """Air-relative speed and angle of attack for a horizontal wind.

    With the body axis along the inertial velocity (angle ``gamma`` above the
    horizontal) and a horizontal wind ``w``, the air-relative velocity is
    ``(v cos(gamma) - w, v sin(gamma))`` and

    ``V_rel = sqrt(v^2 - 2 v w cos(gamma) + w^2)``,
    ``alpha = atan2(w sin(gamma), v - w cos(gamma))``.

    ``alpha`` is signed: positive when the relative flow comes from below the
    axis side for a positive (tail) wind in ascending flight.

    Args:
        speed: Inertial speed ``v`` (m/s), ``>= 0``.
        flight_path_angle: ``gamma`` (rad).
        wind: Horizontal wind ``w`` (m/s), positive along downrange.

    Returns:
        ``(V_rel, alpha)`` in m/s and rad.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``speed`` is negative or an argument is not finite.

    Equation:
        [EQ-AERO-WIND-002].

    References:
        [REF-CURTIS-2014].

    """
    v = _non_negative(speed, "speed")
    gamma = require_finite_real(flight_path_angle, name="flight_path_angle")
    w = require_finite_real(wind, name="wind")
    c = math.cos(gamma)
    s = math.sin(gamma)
    relative = math.hypot(v * c - w, v * s)
    alpha = math.atan2(w * s, v - w * c)
    return relative, alpha


@dataclass(frozen=True, slots=True)
class WindLoadHistory:
    """Wind loads along one ascent trajectory.

    Args:
        times: Grid times (s).
        altitudes: Altitudes (m).
        speeds: Inertial speeds (m/s).
        winds: Wind speed at each altitude (m/s).
        relative_speeds: Air-relative speeds (m/s).
        angles_of_attack: Signed angles of attack (rad).
        dynamic_pressures: ``q = rho V_rel^2 / 2`` (Pa) at every grid point.
        q_alpha: ``q |alpha|`` (Pa rad), zero at points below ``min_speed``.
        max_dynamic_pressure: Maximum of ``q`` (Pa).
        altitude_of_max_q: Altitude of that maximum (m).
        max_q_alpha: Maximum of ``q |alpha|`` over the evaluated points (Pa rad).
        altitude_of_max_q_alpha: Altitude of that maximum (m).
        time_of_max_q_alpha: Time of that maximum (s).
        evaluated_points: Number of points at or above ``min_speed``.

    Equation:
        [EQ-AERO-WIND-003].

    References:
        [REF-CURTIS-2014], [REF-US-STANDARD-ATMOSPHERE-1976].

    """

    times: tuple[float, ...]
    altitudes: tuple[float, ...]
    speeds: tuple[float, ...]
    winds: tuple[float, ...]
    relative_speeds: tuple[float, ...]
    angles_of_attack: tuple[float, ...]
    dynamic_pressures: tuple[float, ...]
    q_alpha: tuple[float, ...]
    max_dynamic_pressure: float
    altitude_of_max_q: float
    max_q_alpha: float
    altitude_of_max_q_alpha: float
    time_of_max_q_alpha: float
    evaluated_points: int


def wind_load_history(
    trajectory: Sequence[tuple[float, float, float, float, float]],
    field: SyntheticWindField,
    coordinates: Sequence[float],
    *,
    amplitude: float = 1.0,
    min_speed: float = 50.0,
    density_scale: float = 1.0,
) -> WindLoadHistory:
    """Evaluate ``q``, ``alpha`` and ``q |alpha|`` along a recorded trajectory.

    For each grid point ``(t, h, v, gamma, m)`` the wind is ``w(h)``
    (:meth:`SyntheticWindField.speed`), the flow follows :func:`relative_flow`
    and ``q = rho(h) V_rel^2 / 2`` with the 1976 standard-atmosphere density.
    With zero wind (``amplitude = 0``) ``q`` equals the ``rho v^2 / 2`` of
    :func:`probstat.aerospace.ascent.simulate_ascent` at the same grid points
    bit for bit, because the same atmosphere routine is called.

    Args:
        trajectory: ``(t, h, v, gamma, m)`` points as returned by
            ``simulate_ascent(..., record=True).trajectory``.
        field: The wind field.
        coordinates: Standard-normal coordinates of the field.
        amplitude: Global wind factor.
        min_speed: Points slower than this (m/s) are excluded from the
            ``q |alpha|`` maximum.
        density_scale: Multiplier on the density (``> 0``).

    Returns:
        The :class:`WindLoadHistory`.

    Raises:
        TypeError: If ``field`` is not a :class:`SyntheticWindField`.
        ValueError: If the trajectory is empty, an argument is out of range or
            no point reaches ``min_speed``.

    Equation:
        [EQ-AERO-WIND-002], [EQ-AERO-WIND-003].

    References:
        [REF-CURTIS-2014], [REF-US-STANDARD-ATMOSPHERE-1976].

    """
    if not isinstance(field, SyntheticWindField):
        raise TypeError("field must be a SyntheticWindField")
    if not trajectory:
        raise ValueError("trajectory must not be empty")
    floor = _non_negative(min_speed, "min_speed")
    k_rho = _positive(density_scale, "density_scale")
    times: list[float] = []
    alts: list[float] = []
    speeds: list[float] = []
    winds: list[float] = []
    rel: list[float] = []
    alphas: list[float] = []
    qs: list[float] = []
    qas: list[float] = []
    best_qa = -1.0
    best_index = -1
    best_q = -1.0
    best_q_index = 0
    evaluated = 0
    for index, (t, h, v, gamma, _mass) in enumerate(trajectory):
        altitude = max(h, 0.0)
        wind = field.speed(altitude, coordinates, amplitude)
        v_rel, alpha = relative_flow(v, gamma, wind)
        rho = atmosphere(altitude).density * k_rho
        q = 0.5 * rho * v_rel * v_rel
        if wind == 0.0:
            # No wind: V_rel^2 must be exactly v^2 (hypot rounds differently).
            q = 0.5 * rho * v * v
        qa = q * abs(alpha) if v >= floor else 0.0
        if v >= floor:
            evaluated += 1
            if qa > best_qa:
                best_qa = qa
                best_index = index
        if q > best_q:
            best_q = q
            best_q_index = index
        times.append(t)
        alts.append(h)
        speeds.append(v)
        winds.append(wind)
        rel.append(v_rel)
        alphas.append(alpha)
        qs.append(q)
        qas.append(qa)
    if best_index < 0:
        raise ValueError("no trajectory point reaches min_speed")
    return WindLoadHistory(
        times=tuple(times),
        altitudes=tuple(alts),
        speeds=tuple(speeds),
        winds=tuple(winds),
        relative_speeds=tuple(rel),
        angles_of_attack=tuple(alphas),
        dynamic_pressures=tuple(qs),
        q_alpha=tuple(qas),
        max_dynamic_pressure=best_q,
        altitude_of_max_q=alts[best_q_index],
        max_q_alpha=best_qa,
        altitude_of_max_q_alpha=alts[best_index],
        time_of_max_q_alpha=times[best_index],
        evaluated_points=evaluated,
    )


def small_wind_slope(
    trajectory: Sequence[tuple[float, float, float, float, float]],
    field: SyntheticWindField,
    coordinates: Sequence[float],
    *,
    min_speed: float = 50.0,
    density_scale: float = 1.0,
) -> float:
    """First-order slope ``d(max q |alpha|)/d a`` at zero wind amplitude.

    From ``q |alpha| = q_0 |w| sin(gamma) / v + O(w^2)`` the maximum over the
    trajectory is, for a small amplitude ``a``, ``a * max_i q_{0,i} |w_i|
    sin(gamma_i) / v_i`` with ``w_i`` the wind at amplitude 1. This function
    evaluates that closed form from the trajectory alone (no
    :func:`relative_flow` call), as an independent oracle of the numerical
    slope of :func:`wind_load_history`.

    Args:
        trajectory: Recorded ``(t, h, v, gamma, m)`` points.
        field: The wind field.
        coordinates: Standard-normal coordinates.
        min_speed: Points slower than this (m/s) are skipped.
        density_scale: Multiplier on the density.

    Returns:
        The slope in Pa rad per unit amplitude.

    Raises:
        TypeError: If ``field`` is not a :class:`SyntheticWindField`.
        ValueError: If an argument is invalid or no point reaches
            ``min_speed``.

    Equation:
        [EQ-AERO-WIND-004].

    References:
        [REF-CURTIS-2014].

    """
    if not isinstance(field, SyntheticWindField):
        raise TypeError("field must be a SyntheticWindField")
    floor = _non_negative(min_speed, "min_speed")
    k_rho = _positive(density_scale, "density_scale")
    best = -1.0
    for _t, h, v, gamma, _m in trajectory:
        if v < floor:
            continue
        altitude = max(h, 0.0)
        wind = field.speed(altitude, coordinates)
        q0 = 0.5 * atmosphere(altitude).density * k_rho * v * v
        best = max(best, q0 * abs(wind) * abs(math.sin(gamma)) / v)
    if best < 0.0:
        raise ValueError("no trajectory point reaches min_speed")
    return best


# ---------------------------------------------------------------------------
# Probabilistic model
# ---------------------------------------------------------------------------

_QUANTITY_TABLE = (
    ("max_dynamic_pressure", "Maximum q = rho V_rel^2 / 2 over the flight", "Pa"),
    ("altitude_of_max_q", "Altitude of the maximum dynamic pressure", "m"),
    ("max_q_alpha", "Maximum of q |alpha| over the flight", "Pa rad"),
    ("altitude_of_max_q_alpha", "Altitude of the maximum q |alpha|", "m"),
    ("q_alpha_margin", "Design limit minus the maximum q |alpha|", "Pa rad"),
)


@dataclass(frozen=True, slots=True)
class _WindLoadsForward:
    """Picklable forward map of :func:`wind_loads_model`.

    Args:
        vehicle: Nominal vehicle.
        guidance: Nominal guidance.
        drag: Nominal drag model.
        field: Wind field.
        limit: Design ``q |alpha|`` limit (Pa rad).
        max_step: RK4 step bound (s).
        min_speed: Speed floor of the ``q |alpha|`` maximum (m/s).

    References:
        [EQ-AERO-WIND-005].

    """

    vehicle: VehicleSpec
    guidance: GuidanceSpec
    drag: DragModel
    field: SyntheticWindField
    limit: float
    max_step: float
    min_speed: float

    def __call__(self, parameters: Mapping[str, float]) -> dict[str, float]:
        """Run one ascent and evaluate its wind loads.

        Args:
            parameters: ``wind_1 .. wind_K``, ``thrust_scale``,
                ``drag_scale`` and ``wind_amplitude``.

        Returns:
            The quantities of interest of :func:`wind_loads_model`.

        Raises:
            ValueError: If a scale is not positive.
            RuntimeError: If the ascent does not reach burnout.

        Equation:
            [EQ-AERO-WIND-005].

        References:
            [REF-CURTIS-2014].

        """
        thrust_k = _positive(parameters["thrust_scale"], "thrust_scale")
        drag_k = _positive(parameters["drag_scale"], "drag_scale")
        stages = tuple(
            replace(s, vacuum_thrust=s.vacuum_thrust * thrust_k)
            for s in self.vehicle.stages
        )
        vehicle = VehicleSpec(stages, self.vehicle.payload_mass)
        result = simulate_ascent(
            vehicle,
            self.guidance,
            drag=self.drag.scaled(drag_k),
            max_step=self.max_step,
            record=True,
        )
        if result.status != "burnout":
            raise RuntimeError(f"ascent failed: {result.status}: {result.stop_reason}")
        coordinates = [parameters[name] for name in self.field.coordinate_names]
        history = wind_load_history(
            result.trajectory,
            self.field,
            coordinates,
            amplitude=parameters["wind_amplitude"],
            min_speed=self.min_speed,
        )
        return {
            "max_dynamic_pressure": history.max_dynamic_pressure,
            "altitude_of_max_q": history.altitude_of_max_q,
            "max_q_alpha": history.max_q_alpha,
            "altitude_of_max_q_alpha": history.altitude_of_max_q_alpha,
            "q_alpha_margin": self.limit - history.max_q_alpha,
        }


def wind_loads_model(
    vehicle: VehicleSpec,
    guidance: GuidanceSpec,
    drag: DragModel,
    field: SyntheticWindField,
    *,
    q_alpha_limit: float,
    thrust_scale: Marginal | float = 1.0,
    drag_scale: Marginal | float = 1.0,
    wind_amplitude: Marginal | float = 1.0,
    max_step: float = 2.0,
    min_speed: float = 50.0,
    name: str = "wind_loads",
    on_failure: FailurePolicy = "raise",
) -> ProbabilisticModel:
    """Build the probabilistic wind-load model of a planar ascent.

    Random inputs: the ``K`` standard-normal wind coordinates ``wind_1 ..
    wind_K`` of ``field`` (always random) and, if given as distributions,
    ``thrust_scale`` (multiplier of ``F_vac`` at fixed ``Isp``, so the mass
    flow scales with it), ``drag_scale`` (multiplier of ``C_D``) and
    ``wind_amplitude`` (global wind factor, a stand-in for seasonal severity).
    Numbers given instead of distributions are fixed parameters.

    Quantities of interest: ``max_dynamic_pressure`` (Pa),
    ``altitude_of_max_q`` (m), ``max_q_alpha`` (Pa rad),
    ``altitude_of_max_q_alpha`` (m) and ``q_alpha_margin = limit -
    max_q_alpha`` (Pa rad). Failure of the limit state is ``q_alpha_margin <=
    0``. The trajectory is the no-wind flight of the (dispersed) vehicle and
    the wind acts only on the load (see the module docstring); the
    ``q |alpha|`` maximum is taken on the integration grid, so the outputs are
    piecewise smooth in the inputs. The limit and all numbers are synthetic.

    Args:
        vehicle: Nominal vehicle (illustrative numbers).
        guidance: Nominal guidance.
        drag: Nominal drag model.
        field: The synthetic wind field.
        q_alpha_limit: Design limit of ``q |alpha|`` (Pa rad), ``> 0``.
        thrust_scale: Thrust multiplier (distribution or number).
        drag_scale: Drag-coefficient multiplier (distribution or number).
        wind_amplitude: Global wind amplitude (distribution or number).
        max_step: Largest RK4 step of each ascent (s).
        min_speed: Speed floor of the ``q |alpha|`` maximum (m/s).
        name: Model label.
        on_failure: Failure policy of the model.

    Returns:
        A :class:`~probstat.modeling.ProbabilisticModel`.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If a number is out of range.

    Equation:
        [EQ-AERO-WIND-005].

    References:
        [REF-CURTIS-2014], [REF-GHANEM-SPANOS-1991].

    """
    if not isinstance(vehicle, VehicleSpec):
        raise TypeError("vehicle must be a VehicleSpec")
    if not isinstance(guidance, GuidanceSpec):
        raise TypeError("guidance must be a GuidanceSpec")
    if not isinstance(drag, DragModel):
        raise TypeError("drag must be a DragModel")
    if not isinstance(field, SyntheticWindField):
        raise TypeError("field must be a SyntheticWindField")
    limit = _positive(q_alpha_limit, "q_alpha_limit")
    step = _positive(max_step, "max_step")
    floor = _non_negative(min_speed, "min_speed")
    random_inputs: list[RandomInput] = [
        RandomInput(n, Normal(0.0, 1.0), unit="-") for n in field.coordinate_names
    ]
    fixed: dict[str, float] = {}
    for key, value, unit in (
        ("thrust_scale", thrust_scale, "-"),
        ("drag_scale", drag_scale, "-"),
        ("wind_amplitude", wind_amplitude, "-"),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{key} must be a number or a distribution")
        if isinstance(value, int | float):
            fixed[key] = require_finite_real(value, name=key)
        else:
            random_inputs.append(RandomInput(key, value, unit=unit))
    quantities = tuple(QuantityOfInterest(n, d, u) for n, d, u in _QUANTITY_TABLE)
    return ProbabilisticModel(
        JointInput.independent(*random_inputs),
        _WindLoadsForward(vehicle, guidance, drag, field, limit, step, floor),
        quantities,
        name=name,
        fixed=fixed,
        on_failure=on_failure,
    )


def wind_load_limit_state(model: ProbabilisticModel) -> ModelLimitState:
    """Limit state ``q_alpha_margin <= 0`` of a :func:`wind_loads_model`.

    Args:
        model: A model built by :func:`wind_loads_model`.

    Returns:
        The :class:`~probstat.reliability.ModelLimitState` that fails when
        the maximum ``q |alpha|`` reaches the design limit.

    Raises:
        TypeError: If ``model`` is not a probabilistic model.
        KeyError: If the model lacks the ``q_alpha_margin`` quantity.

    Equation:
        [EQ-AERO-WIND-005].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 3].

    """
    return ModelLimitState(model, "q_alpha_margin", 0.0, "below")


# ---------------------------------------------------------------------------
# Reliability and extreme-value views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindReliabilityStudy:
    """FORM, subset simulation and Monte Carlo for one wind-load limit state.

    Args:
        form_probability: First-order probability ``Phi(-beta)``.
        form_beta: Reliability index.
        form_converged: Whether the FORM search converged.
        subset_probability: Subset-simulation estimate.
        subset_cov_upper: Upper-bound coefficient of variation of that estimate.
        subset_converged: Whether the subset run reached the failure level.
        mc_probability: Crude Monte Carlo estimate.
        mc_standard_error: Plug-in standard error of the Monte Carlo estimate.
        mc_samples: Monte Carlo sample size.
        evaluations: Model evaluations of FORM, subset and Monte Carlo.

    Equation:
        [EQ-AERO-WIND-006].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5], [REF-AU-BECK-2001].

    """

    form_probability: float
    form_beta: float
    form_converged: bool
    subset_probability: float
    subset_cov_upper: float
    subset_converged: bool
    mc_probability: float
    mc_standard_error: float
    mc_samples: int
    evaluations: int


def wind_reliability_study(
    model: ProbabilisticModel,
    *,
    mc_samples: int = 2000,
    samples_per_level: int = 300,
    seed: int | RandomStream | None = None,
) -> WindReliabilityStudy:
    """Estimate ``P(max q |alpha| >= limit)`` by FORM, subset simulation and MC.

    The three estimators use the same limit state
    (:func:`wind_load_limit_state`). The maximum over the flight makes the
    limit state piecewise smooth, so the FORM linearisation can be inadequate;
    the study returns all three numbers so the caller can compare them.

    Args:
        model: A :func:`wind_loads_model`.
        mc_samples: Crude Monte Carlo sample size.
        samples_per_level: Subset-simulation samples per level.
        seed: Root seed or stream (subset and Monte Carlo use child streams).

    Returns:
        The :class:`WindReliabilityStudy`.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: If a sample size is invalid.

    Equation:
        [EQ-AERO-WIND-006].

    References:
        [REF-MELCHERS-BECK-2018, Chapter 5], [REF-AU-BECK-2001].

    """
    state = wind_load_limit_state(model)
    base = resolve_stream(seed)
    form = form_hlrf(state)
    subset = subset_simulation(
        state, samples_per_level=samples_per_level, seed=base.child("subset")
    )
    mc = model_monte_carlo(state, mc_samples, seed=base.child("mc"))
    return WindReliabilityStudy(
        form_probability=form.probability,
        form_beta=form.beta,
        form_converged=form.converged,
        subset_probability=subset.probability,
        subset_cov_upper=subset.coefficient_of_variation_upper,
        subset_converged=subset.converged,
        mc_probability=mc.probability,
        mc_standard_error=mc.standard_error,
        mc_samples=mc_samples,
        evaluations=form.evaluations + subset.evaluations + mc.evaluations,
    )


@dataclass(frozen=True, slots=True)
class WindExtremeStudy:
    """Extreme-value view of the flight-maximum ``q |alpha|`` over wind profiles.

    Args:
        samples: Number of synthetic wind profiles (model runs).
        mean: Sample mean of ``max q alpha`` (Pa rad).
        quantiles: Empirical ``(0.5, 0.9, 0.99)`` quantiles (Pa rad).
        threshold: Peaks-over-threshold level ``u`` (Pa rad).
        exceedances: Number of samples above ``u``.
        gpd_sigma: Fitted GPD scale (Pa rad).
        gpd_xi: Fitted GPD shape (a negative value means a bounded tail).
        gpd_converged: Whether the GPD maximum-likelihood fit converged.
        gpd_probability: Tail estimate ``zeta_u S_GPD(limit)`` of
            ``P(max q alpha > limit)``.
        gev_block_size: Block size ``b`` of the block-maxima fit.
        gev_xi: Fitted GEV shape of the block maxima.
        gev_probability: ``1 - F_GEV(limit)^(1/b)``, the per-profile
            exceedance probability implied by the block-maxima fit.
        empirical_probability: Fraction of samples above the limit.
        limit: The limit ``L`` (Pa rad) the tail estimates refer to.

    Equation:
        [EQ-AERO-WIND-006].

    References:
        [REF-COLES-2001, Chapters 3 and 4].

    """

    samples: int
    mean: float
    quantiles: tuple[float, float, float]
    threshold: float
    exceedances: int
    gpd_sigma: float
    gpd_xi: float
    gpd_converged: bool
    gpd_probability: float
    gev_block_size: int
    gev_xi: float
    gev_probability: float
    empirical_probability: float
    limit: float


def _empirical_quantile(sorted_values: Sequence[float], p: float) -> float:
    """Return the linearly interpolated empirical quantile.

    Args:
        sorted_values: Ascending sample.
        p: Probability in ``[0, 1]``.

    Returns:
        The quantile.

    References:
        [REF-COLES-2001].

    """
    position = p * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return (1.0 - fraction) * sorted_values[low] + fraction * sorted_values[high]


def wind_extreme_study(
    model: ProbabilisticModel,
    limit: float,
    *,
    samples: int = 2000,
    threshold_quantile: float = 0.9,
    block_size: int = 20,
    seed: int | RandomStream | None = None,
) -> WindExtremeStudy:
    """Fit GPD and GEV tails to the flight maximum ``q |alpha|`` over profiles.

    Draws ``samples`` independent joint input vectors of the model (each a
    synthetic wind profile with its vehicle dispersion), runs the model and
    takes ``Y = max q alpha`` (one number per profile). (i) A generalized
    Pareto law is fitted by maximum likelihood to the excesses above the
    ``threshold_quantile`` empirical quantile ``u``, giving the tail estimate
    ``P(Y > L) = zeta_u (1 + xi (L - u) / sigma)^(-1/xi)`` with ``zeta_u`` the
    fraction above ``u`` (EQ-AERO-WIND-006). (ii) A GEV law is fitted to the
    maxima of consecutive blocks of ``block_size`` profiles; since the block
    maximum has distribution ``F^b``, the per-profile exceedance probability is
    ``1 - F_GEV(L)^(1/b)``. The empirical frequency of ``Y > L`` is returned
    for comparison. The tail fits extrapolate beyond the data only as far as the
    fitted shape is trustworthy; compare them with the reliability estimates of
    :func:`wind_reliability_study`.

    Args:
        model: A :func:`wind_loads_model` providing ``max_q_alpha``.
        limit: Level ``L`` (Pa rad) at which the exceedance probability is
            estimated, above the POT threshold.
        samples: Number of profiles, at least 100.
        threshold_quantile: Empirical quantile level of ``u``, in ``(0.5, 1)``.
        block_size: Block size of the GEV fit, ``>= 2`` with at least 20 blocks.
        seed: Seed or stream.

    Returns:
        The :class:`WindExtremeStudy`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range, the limit is not above
            the threshold or the fits cannot be made.

    Equation:
        [EQ-AERO-WIND-006].

    References:
        [REF-COLES-2001, Chapters 3 and 4], [REF-PICKANDS-1975],
        [REF-BALKEMA-DEHAAN-1974].

    """
    if not isinstance(model, ProbabilisticModel):
        raise TypeError("model must be a ProbabilisticModel")
    level = _positive(limit, "limit")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise ValueError("samples must be an integer of at least 100")
    quantile_level = require_finite_real(threshold_quantile, name="threshold_quantile")
    if not 0.5 < quantile_level < 1.0:
        raise ValueError("threshold_quantile must lie in (0.5, 1)")
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer")
    if block_size < 2 or samples // block_size < 20:
        raise ValueError("block_size must be >= 2 with at least 20 blocks")
    points = model.inputs.sample(samples, seed=seed)
    batch = model.evaluate_many(points)
    values = tuple(batch["max_q_alpha"])
    ordered = sorted(values)
    threshold = _empirical_quantile(ordered, quantile_level)
    if level <= threshold:
        raise ValueError("limit must exceed the peaks-over-threshold level")
    excesses = [v - threshold for v in values if v > threshold]
    if len(excesses) < 10:
        raise ValueError("fewer than 10 exceedances of the threshold")
    gpd = fit_gpd(excesses, method="mle", threshold=threshold)
    zeta = len(excesses) / len(values)
    p_gpd = zeta * gpd.distribution.sf(level)
    blocks = [
        max(values[i : i + block_size])
        for i in range(0, len(values) - block_size + 1, block_size)
    ]
    gev = fit_gev(blocks)
    cdf = gev.distribution.cdf(level)
    p_gev = 1.0 - cdf ** (1.0 / block_size)
    empirical = sum(1 for v in values if v > level) / len(values)
    return WindExtremeStudy(
        samples=len(values),
        mean=math.fsum(values) / len(values),
        quantiles=(
            _empirical_quantile(ordered, 0.5),
            _empirical_quantile(ordered, 0.9),
            _empirical_quantile(ordered, 0.99),
        ),
        threshold=threshold,
        exceedances=len(excesses),
        gpd_sigma=gpd.sigma,
        gpd_xi=gpd.xi,
        gpd_converged=gpd.converged,
        gpd_probability=p_gpd,
        gev_block_size=block_size,
        gev_xi=gev.xi,
        gev_probability=p_gev,
        empirical_probability=empirical,
        limit=level,
    )
