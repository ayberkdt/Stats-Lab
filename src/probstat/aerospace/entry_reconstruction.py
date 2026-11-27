"""Post-flight reconstruction of an atmospheric entry from synthetic flight data.

After an entry the flight team reconstructs, from the sensed deceleration (an
accelerometer) and sparse tracking fixes (radar altitude and speed), what the
atmosphere really was and what the vehicle's ballistic coefficient was
(Karlgaard et al. solved this problem, with a far richer sensor suite and
model, for a Mars entry; here only the problem *setting* is borrowed). This
module builds a small, fully synthetic version of that problem and solves it
twice, so the two answers can be checked against each other and against the
known truth.

Physical model (SI; planar point mass on a spherical, non-rotating Earth; the
same equations of motion as :mod:`probstat.aerospace.entry`):

``h' = V sin(gamma)``, ``V' = -rho V^2 / (2 beta) - g sin(gamma)``,
``gamma' = (V / r - g / V) cos(gamma)`` with ``r = R_E + h``, ``g = mu / r^2``
and a drag-only vehicle (``L/D = 0``). The density is a known base profile
times an unknown smooth multiplicative correction,
``rho(h) = rho_base(h) exp(c(h))``, so ``c`` is the *log-density correction*
(dimensionless; ``c = 0`` is the base atmosphere). The accelerometer senses the
aerodynamic (drag) acceleration ``a = rho V^2 / (2 beta)`` in m/s^2 (gravity is
not sensed) with additive Gaussian noise. Radar fixes ``(h, V)`` are
optional. An optional "density anchor" is a direct, noisy observation of
``c`` at a given altitude (an in-situ sounding), used only to show what breaks
the degeneracy below.

Two methods:

``reconstruct_entry_filter``
    Augmented-state extended or unscented Kalman filter followed by the
    Rauch-Tung-Striebel smoother
    (:func:`probstat.filtering.extended_kalman_filter`,
    :func:`~probstat.filtering.unscented_kalman_filter`,
    :func:`~probstat.filtering.rts_smoother`) on the state
    ``(h, V, gamma, ln beta - ln beta_ref, c)``. The correction ``c`` is a
    mean-reverting random walk *in altitude* (Ornstein-Uhlenbeck, stationary
    standard deviation ``s_c``, correlation length ``ell``): over an altitude
    step ``dh`` it decays by ``exp(-|dh| / ell)`` and receives the matching
    noise, which is exactly an exponential-kernel Gaussian process sampled
    along the descending trajectory.
``reconstruct_entry_batch``
    ``c`` as a natural cubic spline through uniformly spaced node values over a
    fixed altitude range, with
    the same exponential-kernel Gaussian-process prior, plus
    ``ln beta - ln beta_ref`` with a Gaussian prior, estimated by
    Gauss-Newton (:func:`probstat.inverse.gauss_newton_krylov` on the
    prior-whitened coordinates) with a finite-difference Jacobian of the
    simulated measurement history, and a Laplace posterior ``F (I + J' W J)^-1
    F'`` from the Gauss-Newton Hessian at the estimate.

Identifiability. With ``L/D = 0`` and a known entry state the *entire*
trajectory ``(h, V, gamma)(t)`` depends on the density and the ballistic
coefficient only through the ratio ``rho / beta`` ([EQ-AERO-RECON-002]), so
the accelerometer, the radar fixes and any other kinematic measurement
constrain ``c(h) - ln beta`` and nothing else. The ratio is well determined,
the split is not: the correlation between ``ln beta`` and ``c`` tends to
``+1`` and the individual posterior standard deviations tend to the prior ones
unless information from *outside* the kinematics is added: a prior on ``beta``
(for example from the known mass and drag coefficient), or a density anchor.
Radar fixes sharpen the ratio but, in this model, do not break the split;
:meth:`BatchReconstruction.identifiability` measures all of this on the
Laplace covariance. (With lift, or with a moving angle of attack, the
kinematics would separate density from ``beta``; that is outside this model.)

Limits. The truth generator, the filter and the batch method share one set of
equations of motion (RK4, different step sizes), so agreement validates the
estimation machinery, not the physics. The truth correction is a smooth random
Fourier field (an approximate squared-exponential Gaussian process), while both
estimators assume an exponential-kernel prior, which is a deliberate prior
mismatch. The filter and the Laplace posterior are Gaussian approximations; no
check of their nonlinearity error beyond the stated agreement tests is made.
Altitude is assumed to decrease monotonically along the flight (the correction
is a function of altitude, and the filter indexes it along the descent). All
numbers produced by this module are synthetic; nothing represents a real
vehicle or flight.

Experimental (added in 0.28).

References:
    [REF-KARLGAARD-KUTTY-SCHOENENBERGER-MUNK-2014], [REF-ALLEN-EGGERS-1958],
    [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965],
    [REF-JULIER-UHLMANN-2004], [REF-RASMUSSEN-WILLIAMS-2006],
    [REF-US-STANDARD-ATMOSPHERE-1976], [EQ-AERO-RECON-001].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

from probstat.aerospace.ascent import EARTH_MU, EARTH_RADIUS
from probstat.aerospace.entry import ExponentialAtmosphere, _density_function
from probstat.filtering import (
    GaussianState,
    KalmanFilterResult,
    NonlinearStateSpace,
    extended_kalman_filter,
    rts_smoother,
    unscented_kalman_filter,
)
from probstat.foundations.random_streams import resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inverse.gauss_newton import CallbackForward, gauss_newton_krylov
from probstat.linalg import (
    DiagonalCovariance,
    Matrix,
    Vector,
    cholesky,
    inverse_spd,
    matmul,
    solve_spd,
    transpose,
)

__all__ = [
    "BatchReconstruction",
    "EntryMeasurements",
    "EntryReconstructionSetup",
    "EntryTruth",
    "FilterReconstruction",
    "IdentifiabilityReport",
    "band_coverage",
    "reconstruct_entry_batch",
    "reconstruct_entry_filter",
    "simulate_entry_measurements",
    "synthesize_entry_truth",
]

_FD_STEP = 1e-5  # forward-difference step of the batch Jacobian (log units)
_FOURIER_TERMS = 24  # random Fourier features of the synthetic truth field


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
        [EQ-AERO-RECON-001]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    """Return an integer at least ``minimum``.

    Args:
        value: Candidate integer.
        name: Name used in error messages.
        minimum: Smallest allowed value.

    Returns:
        The value.

    Raises:
        TypeError: If ``value`` is not an ``int`` (``bool`` is rejected).
        ValueError: If it is smaller than ``minimum``.

    References:
        [EQ-AERO-RECON-001]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class EntryReconstructionSetup:
    """Scenario, sensors and priors of a synthetic entry reconstruction.

    The defaults describe a synthetic capsule entering the exponential base
    atmosphere at a shallow angle; they do not represent a real vehicle.

    Args:
        base_atmosphere: ``"us76"`` or an
            :class:`~probstat.aerospace.entry.ExponentialAtmosphere`; the
            known part of the density.
        entry_altitude: Entry-interface altitude in m (known).
        entry_speed: Entry speed in m/s (known).
        flight_path_angle: Entry flight-path angle in rad, negative (known).
        ballistic_coefficient: Prior-median ``beta_ref`` in kg/m^2, ``> 0``.
        ballistic_log_sd: Prior standard deviation of ``ln beta`` (``> 0``).
        duration: Length of the record in s.
        sample_interval: Accelerometer period in s.
        substeps: RK4 sub-steps per sample used by the *estimators*.
        truth_substeps: RK4 sub-steps per sample used for the synthetic truth.
        accel_sigma: Accelerometer noise standard deviation in m/s^2.
        radar_every: ``None`` for no radar, else a radar fix every this many
            accelerometer samples.
        radar_height_sigma: Radar altitude noise sd in m.
        radar_speed_sigma: Radar speed noise sd in m/s.
        density_log_sd: Prior standard deviation ``s_c`` of ``c``.
        correlation_length: Prior correlation length ``ell`` of ``c`` in m.
        altitude_range: ``(lowest, highest)`` altitude in m of the batch
            basis; ``c`` is held constant outside.
        nodes: Number of spline nodes of the batch method (``>= 3``).
        anchor_altitudes: Altitudes in m of optional direct density
            observations (batch method only).
        anchor_sigma: Noise sd of an anchor (in ``c`` units).
        initial_state_sd: Filter initial standard deviations of
            ``(h [m], V [m/s], gamma [rad])``.
        filter_process_noise: Per-step standard deviation of small model-error
            noise on ``(h [m], V [m/s], gamma [rad])`` in the filter.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is out of range.

    Equation:
        [EQ-AERO-RECON-001].

    References:
        [REF-ALLEN-EGGERS-1958], [REF-KARLGAARD-KUTTY-SCHOENENBERGER-MUNK-2014].

    """

    base_atmosphere: ExponentialAtmosphere | Literal["us76"] = field(
        default_factory=ExponentialAtmosphere
    )
    entry_altitude: float = 100_000.0
    entry_speed: float = 7_500.0
    flight_path_angle: float = -0.1
    ballistic_coefficient: float = 300.0
    ballistic_log_sd: float = 0.3
    duration: float = 150.0
    sample_interval: float = 1.0
    substeps: int = 2
    truth_substeps: int = 8
    accel_sigma: float = 0.5
    radar_every: int | None = None
    radar_height_sigma: float = 30.0
    radar_speed_sigma: float = 1.0
    density_log_sd: float = 0.3
    correlation_length: float = 10_000.0
    altitude_range: tuple[float, float] = (5_000.0, 110_000.0)
    nodes: int = 16
    anchor_altitudes: tuple[float, ...] = ()
    anchor_sigma: float = 0.05
    initial_state_sd: tuple[float, float, float] = (1.0, 0.1, 1e-5)
    filter_process_noise: tuple[float, float, float] = (0.2, 0.02, 2e-6)

    def __post_init__(self) -> None:
        """Validate the scenario.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If a field is out of range.

        References:
            [EQ-AERO-RECON-001]. No equation is evaluated.

        """
        if not isinstance(self.base_atmosphere, ExponentialAtmosphere):
            if self.base_atmosphere != "us76":
                raise TypeError("base_atmosphere must be 'us76' or exponential")
        _positive(self.entry_altitude, "entry_altitude")
        _positive(self.entry_speed, "entry_speed")
        angle = require_finite_real(self.flight_path_angle, name="flight_path_angle")
        if not -0.5 * math.pi <= angle < 0.0:
            raise ValueError("flight_path_angle must lie in [-pi/2, 0)")
        for name in (
            "ballistic_coefficient",
            "ballistic_log_sd",
            "duration",
            "sample_interval",
            "accel_sigma",
            "radar_height_sigma",
            "radar_speed_sigma",
            "density_log_sd",
            "correlation_length",
            "anchor_sigma",
        ):
            _positive(getattr(self, name), name)
        _positive_int(self.substeps, "substeps")
        _positive_int(self.truth_substeps, "truth_substeps")
        _positive_int(self.nodes, "nodes", 3)
        if self.radar_every is not None:
            _positive_int(self.radar_every, "radar_every")
        if self.duration < self.sample_interval:
            raise ValueError("duration must be at least one sample_interval")
        low, high = self.altitude_range
        if not 0.0 <= require_finite_real(low, name="altitude_range") < high:
            raise ValueError("altitude_range must satisfy 0 <= lowest < highest")
        require_finite_real(high, name="altitude_range")
        for altitude in self.anchor_altitudes:
            if not low <= require_finite_real(altitude, name="anchor") <= high:
                raise ValueError("anchor altitudes must lie inside altitude_range")
        for name in ("initial_state_sd", "filter_process_noise"):
            values = getattr(self, name)
            if len(values) != 3:
                raise ValueError(f"{name} must have three entries")
            for value in values:
                _positive(value, name)

    @property
    def samples(self) -> int:
        """Number of accelerometer samples ``N = floor(duration / interval)``.

        References:
            [EQ-AERO-RECON-001].

        """
        return int(self.duration / self.sample_interval + 1e-9)

    @property
    def node_altitudes(self) -> Vector:
        """Evenly spaced node altitudes of the batch spline in m.

        References:
            [EQ-AERO-RECON-006].

        """
        low, high = self.altitude_range
        step = (high - low) / (self.nodes - 1)
        return tuple(low + j * step for j in range(self.nodes))


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def _scaled_density(
    altitude: float, base: Callable[[float], float], factor: float
) -> float:
    """Return ``factor * rho_base(altitude)``.

    Args:
        altitude: Altitude in m.
        base: Base density function.
        factor: Multiplicative factor ``exp(c)``.

    Returns:
        The density in kg/m^3.

    References:
        [EQ-AERO-RECON-001].

    """
    return factor * base(altitude)


def _corrected_density(
    altitude: float,
    base: Callable[[float], float],
    correction: Callable[[float], float],
) -> float:
    """Return ``rho_base(h) exp(c(h))`` for a correction function.

    Args:
        altitude: Altitude in m.
        base: Base density function.
        correction: Log-density correction ``c(h)``.

    Returns:
        The density in kg/m^3.

    References:
        [EQ-AERO-RECON-001].

    """
    return base(altitude) * math.exp(correction(altitude))


def _derivative(
    h: float, v: float, g: float, density: Callable[[float], float], i2b: float
) -> tuple[float, float, float]:
    """Return ``(h', V', gamma')`` of the drag-only planar entry.

    Args:
        h: Altitude in m.
        v: Speed in m/s (positive).
        g: Flight-path angle in rad.
        density: Density function in kg/m^3.
        i2b: ``1 / (2 beta)`` in m^2/kg.

    Returns:
        The state derivative.

    Equation:
        [EQ-AERO-RECON-001].

    References:
        [REF-ALLEN-EGGERS-1958].

    """
    r = EARTH_RADIUS + h
    grav = EARTH_MU / (r * r)
    drag = density(h) * v * v * i2b
    return (
        v * math.sin(g),
        -drag - grav * math.sin(g),
        (v / r - grav / v) * math.cos(g),
    )


def _advance(
    h: float,
    v: float,
    g: float,
    step: float,
    substeps: int,
    density: Callable[[float], float],
    i2b: float,
) -> tuple[float, float, float] | None:
    """Advance the state by ``substeps`` classical RK4 steps.

    Args:
        h: Altitude in m.
        v: Speed in m/s.
        g: Flight-path angle in rad.
        step: Sub-step in s.
        substeps: Number of sub-steps.
        density: Density function.
        i2b: ``1 / (2 beta)``.

    Returns:
        The new ``(h, V, gamma)``, or ``None`` if the state left the valid
        domain (altitude or speed not positive, or not finite).

    Equation:
        [EQ-AERO-RECON-001].

    References:
        [REF-ALLEN-EGGERS-1958].

    """
    for _ in range(substeps):
        k1 = _derivative(h, v, g, density, i2b)
        k2 = _derivative(
            h + 0.5 * step * k1[0],
            v + 0.5 * step * k1[1],
            g + 0.5 * step * k1[2],
            density,
            i2b,
        )
        k3 = _derivative(
            h + 0.5 * step * k2[0],
            v + 0.5 * step * k2[1],
            g + 0.5 * step * k2[2],
            density,
            i2b,
        )
        k4 = _derivative(
            h + step * k3[0], v + step * k3[1], g + step * k3[2], density, i2b
        )
        h += step / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        v += step / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        g += step / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        if not (h > 0.0 and v > 0.0 and math.isfinite(h + v + g)):
            return None
    return h, v, g


def _history(
    setup: EntryReconstructionSetup,
    density: Callable[[float], float],
    i2b: float,
    substeps: int,
) -> tuple[list[float], list[float], list[float], list[float]] | None:
    """Integrate the whole record and return the sampled history.

    Args:
        setup: The scenario.
        density: Density function in kg/m^3.
        i2b: ``1 / (2 beta)``.
        substeps: RK4 sub-steps per sample.

    Returns:
        ``(h, V, gamma, a)`` lists at the sample times ``k * dt``,
        ``k = 0..N`` (``a`` is the sensed drag acceleration), or ``None`` if
        the trajectory left the valid domain.

    Equation:
        [EQ-AERO-RECON-001], [EQ-AERO-RECON-002].

    References:
        [REF-ALLEN-EGGERS-1958].

    """
    dt = setup.sample_interval
    h, v, g = setup.entry_altitude, setup.entry_speed, setup.flight_path_angle
    hs, vs, gs = [h], [v], [g]
    acc = [density(h) * v * v * i2b]
    for _ in range(setup.samples):
        state = _advance(h, v, g, dt / substeps, substeps, density, i2b)
        if state is None:
            return None
        h, v, g = state
        hs.append(h)
        vs.append(v)
        gs.append(g)
        acc.append(density(h) * v * v * i2b)
    return hs, vs, gs, acc


# ---------------------------------------------------------------------------
# Synthetic truth and data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryTruth:
    """A synthetic true entry: trajectory, atmosphere correction and ``beta``.

    The correction is the random Fourier field
    ``c(h) = A sqrt(2 / M) sum_m cos(w_m h + phi_m)`` with ``w_m ~ N(0,
    1/L^2)`` and ``phi_m ~ U(0, 2 pi)``, an approximate sample of a
    squared-exponential Gaussian process of standard deviation ``A`` and
    length scale ``L`` (exact as ``M -> infinity``).

    Args:
        times: Sample times in s, ``k * dt``, ``k = 0..N``.
        altitude: True altitude in m at the sample times.
        speed: True speed in m/s.
        flight_path_angle: True flight-path angle in rad.
        deceleration: True sensed drag acceleration in m/s^2.
        log_ballistic_coefficient: True ``ln beta`` (``beta`` in kg/m^2).
        amplitude: Field standard deviation ``A``.
        frequencies: Angular wavenumbers ``w_m`` in rad/m.
        phases: Phases ``phi_m`` in rad.

    References:
        [EQ-AERO-RECON-001], [REF-RASMUSSEN-WILLIAMS-2006].

    """

    times: tuple[float, ...]
    altitude: tuple[float, ...]
    speed: tuple[float, ...]
    flight_path_angle: tuple[float, ...]
    deceleration: tuple[float, ...]
    log_ballistic_coefficient: float
    amplitude: float
    frequencies: tuple[float, ...]
    phases: tuple[float, ...]

    def log_density_correction(self, altitude: float) -> float:
        """Return the true correction ``c(h)``.

        Args:
            altitude: Altitude in m.

        Returns:
            ``c(h)`` (dimensionless).

        Raises:
            TypeError: If ``altitude`` is not a real number.

        Equation:
            [EQ-AERO-RECON-001].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006].

        """
        h = require_finite_real(altitude, name="altitude")
        total = math.fsum(
            math.cos(w * h + p)
            for w, p in zip(self.frequencies, self.phases, strict=True)
        )
        return self.amplitude * math.sqrt(2.0 / len(self.frequencies)) * total


def synthesize_entry_truth(
    setup: EntryReconstructionSetup,
    *,
    seed: int,
    length_scale: float | None = None,
    amplitude: float | None = None,
    ballistic_offset: float | None = 0.0,
) -> EntryTruth:
    """Simulate a synthetic true entry through a perturbed atmosphere.

    Args:
        setup: The scenario.
        seed: Seed of the random field (and of the offset draw).
        length_scale: Length scale ``L`` in m of the correction field
            (default ``setup.correlation_length``).
        amplitude: Field standard deviation (default ``setup.density_log_sd``).
        ballistic_offset: ``ln beta_true - ln beta_ref``; ``None`` draws it
            from the prior ``N(0, ballistic_log_sd^2)``.

    Returns:
        The :class:`EntryTruth`, integrated with ``setup.truth_substeps``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the true trajectory leaves the valid domain (for
            example the vehicle reaches the ground within the record).

    Equation:
        [EQ-AERO-RECON-001].

    References:
        [REF-ALLEN-EGGERS-1958], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    if not isinstance(setup, EntryReconstructionSetup):
        raise TypeError("setup must be an EntryReconstructionSetup")
    scale = _positive(
        setup.correlation_length if length_scale is None else length_scale,
        "length_scale",
    )
    amp = _positive(
        setup.density_log_sd if amplitude is None else amplitude, "amplitude"
    )
    stream = resolve_stream(seed)
    rng = stream.child("field").generator()
    frequencies = tuple(rng.gauss(0.0, 1.0 / scale) for _ in range(_FOURIER_TERMS))
    phases = tuple(rng.uniform(0.0, 2.0 * math.pi) for _ in range(_FOURIER_TERMS))
    if ballistic_offset is None:
        offset = stream.child("beta").generator().gauss(0.0, setup.ballistic_log_sd)
    else:
        offset = require_finite_real(ballistic_offset, name="ballistic_offset")
    log_beta = math.log(setup.ballistic_coefficient) + offset
    shell = EntryTruth(
        (), (), (), (), (), log_beta, amp, frequencies, phases
    )  # field only, to evaluate c(h)
    base = _density_function(setup.base_atmosphere)
    density = partial(
        _corrected_density, base=base, correction=shell.log_density_correction
    )
    history = _history(setup, density, 0.5 * math.exp(-log_beta), setup.truth_substeps)
    if history is None:
        raise ValueError("the true trajectory left the valid domain")
    hs, vs, gs, acc = history
    times = tuple(k * setup.sample_interval for k in range(len(hs)))
    return EntryTruth(
        times, tuple(hs), tuple(vs), tuple(gs), tuple(acc), log_beta, amp,
        frequencies, phases,
    )  # fmt: skip


@dataclass(frozen=True, slots=True)
class EntryMeasurements:
    """Synthetic flight data of one entry.

    Entry ``k`` of the per-sample tuples belongs to the time ``(k + 1) dt``.

    Args:
        acceleration: Sensed drag acceleration in m/s^2, ``None`` where the
            sample was dropped.
        radar: ``(altitude [m], speed [m/s])`` fixes, ``None`` where there
            is none.
        anchors: Noisy direct observations of ``c`` at
            ``setup.anchor_altitudes``.
        noise_free: ``True`` if no noise was added (verification only).

    References:
        [EQ-AERO-RECON-001].

    """

    acceleration: tuple[float | None, ...]
    radar: tuple[tuple[float, float] | None, ...]
    anchors: tuple[float, ...]
    noise_free: bool


def simulate_entry_measurements(
    setup: EntryReconstructionSetup,
    truth: EntryTruth,
    *,
    seed: int,
    noise: bool = True,
    dropout_band: tuple[float, float] | None = None,
) -> EntryMeasurements:
    """Generate noisy sensor data from a synthetic truth.

    Args:
        setup: The scenario (sensor rates and noise levels).
        truth: The synthetic truth.
        seed: Noise seed.
        noise: ``False`` returns exact values (verification only).
        dropout_band: ``(lowest, highest)`` true altitude in m inside which
            the accelerometer and radar produce no data (a failure case).

    Returns:
        The :class:`EntryMeasurements`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the truth does not match the setup or the band is
            invalid.

    Equation:
        [EQ-AERO-RECON-001].

    References:
        [REF-KARLGAARD-KUTTY-SCHOENENBERGER-MUNK-2014].

    """
    if not isinstance(setup, EntryReconstructionSetup) or not isinstance(
        truth, EntryTruth
    ):
        raise TypeError("setup and truth must be the module's dataclasses")
    if len(truth.times) != setup.samples + 1:
        raise ValueError("truth does not match the setup's record length")
    if dropout_band is not None and not dropout_band[0] < dropout_band[1]:
        raise ValueError("dropout_band must satisfy lowest < highest")
    stream = resolve_stream(seed)
    rng_a = stream.child("accelerometer").generator()
    rng_r = stream.child("radar").generator()
    rng_c = stream.child("anchor").generator()
    scale = 1.0 if noise else 0.0
    acc: list[float | None] = []
    radar: list[tuple[float, float] | None] = []
    for k in range(1, setup.samples + 1):
        h = truth.altitude[k]
        dropped = dropout_band is not None and dropout_band[0] <= h <= dropout_band[1]
        a_noise = rng_a.gauss(0.0, setup.accel_sigma) * scale
        acc.append(None if dropped else truth.deceleration[k] + a_noise)
        fix: tuple[float, float] | None = None
        if setup.radar_every is not None and k % setup.radar_every == 0:
            fix = (
                h + rng_r.gauss(0.0, setup.radar_height_sigma) * scale,
                truth.speed[k] + rng_r.gauss(0.0, setup.radar_speed_sigma) * scale,
            )
        radar.append(None if dropped else fix)
    anchors = tuple(
        truth.log_density_correction(a) + rng_c.gauss(0.0, setup.anchor_sigma) * scale
        for a in setup.anchor_altitudes
    )
    return EntryMeasurements(tuple(acc), tuple(radar), anchors, not noise)


# ---------------------------------------------------------------------------
# Method A: augmented-state filter and smoother
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilterReconstruction:
    """Result of :func:`reconstruct_entry_filter`.

    Profiles are indexed by sample ``k = 0..N`` (time ``k dt``) and are
    functions of the *smoothed* trajectory; ``c`` is the correction at the
    altitude reached at that time.

    Args:
        method: ``"ekf"`` or ``"ukf"``.
        times: Sample times in s.
        altitude: Smoothed altitude in m.
        speed: Smoothed speed in m/s.
        log_density_mean: Smoothed mean of ``c``.
        log_density_sd: Smoothed standard deviation of ``c``.
        filtered_log_density_mean: Forward-filter mean of ``c``.
        filtered_log_density_sd: Forward-filter standard deviation of ``c``.
        log_ballistic_offset_mean: Smoothed mean of ``ln beta - ln beta_ref``
            (a constant, taken at the first sample).
        log_ballistic_offset_sd: Its smoothed standard deviation.
        log_ballistic_density_correlation: Smoothed correlation between the
            ballistic offset and ``c`` at each sample.
        log_ratio_sd: Smoothed standard deviation of ``c - b`` (the
            density-over-ballistic-coefficient log ratio that the kinematics
            determine) at each sample.
        mean_nis: Mean normalized innovation squared per observation
            dimension (about 1 for a consistent filter).
        base_density: Base density in kg/m^3 at the smoothed altitudes.

    References:
        [EQ-AERO-RECON-003], [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965].

    """

    method: str
    times: tuple[float, ...]
    altitude: tuple[float, ...]
    speed: tuple[float, ...]
    log_density_mean: tuple[float, ...]
    log_density_sd: tuple[float, ...]
    filtered_log_density_mean: tuple[float, ...]
    filtered_log_density_sd: tuple[float, ...]
    log_ballistic_offset_mean: float
    log_ballistic_offset_sd: float
    log_ballistic_density_correlation: tuple[float, ...]
    log_ratio_sd: tuple[float, ...]
    mean_nis: float
    base_density: tuple[float, ...]

    def density_band(
        self, z: float = 1.96
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Return the density profile ``rho_base exp(c +- z sd)`` in kg/m^3.

        Args:
            z: Half-width of the band in standard deviations of ``c``.

        Returns:
            ``(lower, median, upper)`` along the trajectory.

        Raises:
            TypeError: If ``z`` is not a real number.
            ValueError: If ``z`` is not positive.

        Equation:
            [EQ-AERO-RECON-006].

        References:
            [REF-KALMAN-1960].

        """
        half = _positive(z, "z")
        lower = tuple(
            r * math.exp(m - half * s)
            for r, m, s in zip(
                self.base_density,
                self.log_density_mean,
                self.log_density_sd,
                strict=True,
            )
        )
        mid = tuple(
            r * math.exp(m)
            for r, m in zip(self.base_density, self.log_density_mean, strict=True)
        )
        upper = tuple(
            r * math.exp(m + half * s)
            for r, m, s in zip(
                self.base_density,
                self.log_density_mean,
                self.log_density_sd,
                strict=True,
            )
        )
        return lower, mid, upper


def _filter_transition(
    state: Vector,
    *,
    setup: EntryReconstructionSetup,
    base: Callable[[float], float],
) -> Vector:
    """Noise-free filter transition of the scaled augmented state.

    The state is ``(h [km], V [km/s], gamma, b, c)`` with
    ``b = ln beta - ln beta_ref``. Over one sample the trajectory is advanced
    with the density ``rho_base(h) exp(c)`` (``h`` the altitude of each RK4
    stage, ``c`` held constant over the sample), and ``c`` decays by
    ``exp(-|dh| / ell)`` with the altitude step ``dh`` of the sample.

    Args:
        state: The scaled state before the step.
        setup: The scenario.
        base: Base density function.

    Returns:
        The scaled state after one sample interval.

    Raises:
        ValueError: If the propagated trajectory leaves the valid domain.

    Equation:
        [EQ-AERO-RECON-003].

    References:
        [REF-ALLEN-EGGERS-1958].

    """
    h, v, g = state[0] * 1000.0, state[1] * 1000.0, state[2]
    i2b = 0.5 * math.exp(-(math.log(setup.ballistic_coefficient) + state[3]))
    density = partial(_scaled_density, base=base, factor=math.exp(state[4]))
    new = _advance(
        h, v, g, setup.sample_interval / setup.substeps, setup.substeps, density, i2b
    )
    if new is None:
        raise ValueError("the filter trajectory left the valid domain")
    decay = math.exp(-abs(new[0] - h) / setup.correlation_length)
    return (new[0] / 1000.0, new[1] / 1000.0, new[2], state[3], decay * state[4])


def _filter_observation(
    state: Vector,
    *,
    base: Callable[[float], float],
    beta_ref: float,
    use_acc: bool,
    use_radar: bool,
) -> Vector:
    """Noise-free observation of the scaled state.

    Args:
        state: The scaled state.
        base: Base density function.
        beta_ref: ``beta_ref`` in kg/m^2.
        use_acc: Include the sensed deceleration (m/s^2).
        use_radar: Include radar altitude and speed (km and km/s).

    Returns:
        The stacked observation (a single zero if both flags are false).

    Equation:
        [EQ-AERO-RECON-002].

    References:
        [REF-ALLEN-EGGERS-1958].

    """
    out: list[float] = []
    if use_acc or not use_radar:
        v = state[1] * 1000.0
        rho = base(state[0] * 1000.0) * math.exp(state[4])
        out.append(rho * v * v * 0.5 / (beta_ref * math.exp(state[3])))
    if use_radar:
        out.extend((state[0], state[1]))
    return tuple(out)


def _filter_observation_jacobian(
    state: Vector,
    *,
    base: Callable[[float], float],
    beta_ref: float,
    use_acc: bool,
    use_radar: bool,
) -> Matrix:
    """Analytic Jacobian of :func:`_filter_observation`.

    Args:
        state: The scaled state.
        base: Base density function.
        beta_ref: ``beta_ref`` in kg/m^2.
        use_acc: Include the sensed deceleration.
        use_radar: Include radar altitude and speed.

    Returns:
        The Jacobian, one row per observation component.

    Equation:
        [EQ-AERO-RECON-002].

    References:
        [REF-ALLEN-EGGERS-1958].

    """
    rows: list[tuple[float, ...]] = []
    if use_acc or not use_radar:
        h = state[0] * 1000.0
        v = state[1] * 1000.0
        a = base(h) * math.exp(state[4]) * v * v * 0.5 / (beta_ref * math.exp(state[3]))
        step = 10.0
        dlog = (math.log(base(h + step)) - math.log(base(h - step))) / (2.0 * step)
        rows.append((a * dlog * 1000.0, 2.0 * a / state[1], 0.0, -a, a))
    if use_radar:
        rows.append((1.0, 0.0, 0.0, 0.0, 0.0))
        rows.append((0.0, 1.0, 0.0, 0.0, 0.0))
    return tuple(rows)


def reconstruct_entry_filter(
    setup: EntryReconstructionSetup,
    measurements: EntryMeasurements,
    *,
    method: Literal["ekf", "ukf"] = "ekf",
) -> FilterReconstruction:
    """Reconstruct the entry with an augmented-state filter and RTS smoother.

    State ``x = (h, V, gamma, b, c)`` (scaled to km and km/s internally) with
    ``b = ln beta - ln beta_ref``. Per sample the model is
    ``x_k = f(x_{k-1}) + w_k`` ([EQ-AERO-RECON-003]): the trajectory is the
    RK4 flow of [EQ-AERO-RECON-001] with the density ``rho_base(h) e^c``, ``b``
    is constant, and ``c`` is an Ornstein-Uhlenbeck process in altitude,
    ``c' = a c + sqrt(s_c^2 (1 - a^2)) eps`` with ``a = exp(-|dh| / ell)``. The
    process covariance therefore depends on the altitude step; it is
    recomputed at every sample from the predicted mean (the filter is run one
    sample at a time and the per-sample records are assembled into one
    :class:`~probstat.filtering.KalmanFilterResult` for the smoother). The
    observation is the sensed drag acceleration
    ``a = rho_base e^c V^2 / (2 beta_ref e^b)`` plus optional radar fixes
    ``(h, V)`` ([EQ-AERO-RECON-002]). Small additive process noise
    (``setup.filter_process_noise``) on ``(h, V, gamma)`` absorbs the
    integration-error difference between filter and truth.

    Args:
        setup: The scenario.
        measurements: Flight data from :func:`simulate_entry_measurements`.
        method: ``"ekf"`` (first-order linearization) or ``"ukf"`` (unscented).

    Returns:
        The :class:`FilterReconstruction`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the data length does not match the setup, the method
            is unknown, or the filter trajectory leaves the valid domain.

    Equation:
        [EQ-AERO-RECON-002], [EQ-AERO-RECON-003].

    References:
        [REF-KALMAN-1960], [REF-RAUCH-TUNG-STRIEBEL-1965],
        [REF-JULIER-UHLMANN-2004], [REF-KARLGAARD-KUTTY-SCHOENENBERGER-MUNK-2014].

    """
    if not isinstance(setup, EntryReconstructionSetup) or not isinstance(
        measurements, EntryMeasurements
    ):
        raise TypeError("setup and measurements must be the module's dataclasses")
    if method not in ("ekf", "ukf"):
        raise ValueError("method must be 'ekf' or 'ukf'")
    n = setup.samples
    if len(measurements.acceleration) != n or len(measurements.radar) != n:
        raise ValueError("measurements do not match the setup's record length")
    base = _density_function(setup.base_atmosphere)
    beta_ref = setup.ballistic_coefficient
    transition = partial(_filter_transition, setup=setup, base=base)
    scale = (1000.0, 1000.0, 1.0)
    diagonal = (
        *((setup.initial_state_sd[i] / scale[i]) ** 2 for i in range(3)),
        setup.ballistic_log_sd**2,
        setup.density_log_sd**2,
    )
    p0 = tuple(
        tuple(diagonal[i] if i == j else 0.0 for j in range(5)) for i in range(5)
    )
    mean: Vector = (
        setup.entry_altitude / 1000.0,
        setup.entry_speed / 1000.0,
        setup.flight_path_angle,
        0.0,
        0.0,
    )
    cov: Matrix = p0
    filtered = [GaussianState(mean, cov)]
    predicted: list[GaussianState] = []
    cross: list[Matrix] = []
    innovations: list[Vector | None] = []
    innovation_covs: list[Matrix | None] = []
    gains: list[Matrix | None] = []
    terms: list[float] = []
    nis_total, nis_dims = 0.0, 0
    q_proc = tuple((setup.filter_process_noise[i] / scale[i]) ** 2 for i in range(3))
    for k in range(n):
        look = transition(mean)
        decay = math.exp(-abs(look[0] - mean[0]) * 1000.0 / setup.correlation_length)
        process = tuple(
            tuple(
                (
                    q_proc[i]
                    if i < 3 and i == j
                    else setup.density_log_sd**2 * (1.0 - decay**2)
                    if i == 4 and j == 4
                    else 0.0
                )
                for j in range(5)
            )
            for i in range(5)
        )
        accel = measurements.acceleration[k]
        fix = measurements.radar[k]
        use_acc, use_radar = accel is not None, fix is not None
        if use_acc or not use_radar:
            obs_values: list[float] = [] if accel is None else [accel]
            variances = [setup.accel_sigma**2]
        else:
            obs_values, variances = [], []
        if fix is not None:
            obs_values.extend((fix[0] / 1000.0, fix[1] / 1000.0))
            variances.extend(
                (
                    (setup.radar_height_sigma / 1000.0) ** 2,
                    (setup.radar_speed_sigma / 1000.0) ** 2,
                )
            )
        observation: Sequence[float] | None = obs_values if obs_values else None
        model = NonlinearStateSpace(
            transition,
            partial(
                _filter_observation,
                base=base,
                beta_ref=beta_ref,
                use_acc=use_acc,
                use_radar=use_radar,
            ),
            process,
            tuple(
                tuple(variances[i] if i == j else 0.0 for j in range(len(variances)))
                for i in range(len(variances))
            ),
            mean,
            cov,
            5,
            len(variances),
            observation_jacobian_function=partial(
                _filter_observation_jacobian,
                base=base,
                beta_ref=beta_ref,
                use_acc=use_acc,
                use_radar=use_radar,
            ),
        )
        step = (
            extended_kalman_filter(model, [observation])
            if method == "ekf"
            else unscented_kalman_filter(model, [observation])
        )
        state = step.filtered[1]
        filtered.append(state)
        predicted.append(step.predicted[0])
        cross.append(step.cross_covariances[0])
        innov = step.innovations[0]
        innov_cov = step.innovation_covariances[0]
        innovations.append(innov)
        innovation_covs.append(innov_cov)
        gains.append(step.gains[0])
        terms.append(step.log_likelihood_terms[0])
        if innov is not None and innov_cov is not None:
            weighted = solve_spd(innov_cov, innov)
            nis_total += sum(a * b for a, b in zip(innov, weighted, strict=True))
            nis_dims += len(innov)
        mean, cov = state.mean, state.covariance
    record = KalmanFilterResult(
        tuple(filtered),
        tuple(predicted),
        tuple(cross),
        tuple(innovations),
        tuple(innovation_covs),
        tuple(gains),
        tuple(terms),
        "extended" if method == "ekf" else "unscented",
    )
    smooth = rts_smoother(record).smoothed
    heights = tuple(s.mean[0] * 1000.0 for s in smooth)
    sd_c = tuple(math.sqrt(max(s.covariance[4][4], 0.0)) for s in smooth)
    sd_b = tuple(math.sqrt(max(s.covariance[3][3], 0.0)) for s in smooth)
    corr = tuple(
        s.covariance[3][4] / (a * b) if a > 0.0 and b > 0.0 else 0.0
        for s, a, b in zip(smooth, sd_b, sd_c, strict=True)
    )
    ratio_sd = tuple(
        math.sqrt(
            max(s.covariance[4][4] - 2.0 * s.covariance[3][4] + s.covariance[3][3], 0.0)
        )
        for s in smooth
    )
    return FilterReconstruction(
        method,
        tuple(k * setup.sample_interval for k in range(n + 1)),
        heights,
        tuple(s.mean[1] * 1000.0 for s in smooth),
        tuple(s.mean[4] for s in smooth),
        sd_c,
        tuple(s.mean[4] for s in filtered),
        tuple(math.sqrt(max(s.covariance[4][4], 0.0)) for s in filtered),
        smooth[0].mean[3],
        sd_b[0],
        corr,
        ratio_sd,
        nis_total / nis_dims if nis_dims else float("nan"),
        tuple(base(h) for h in heights),
    )


# ---------------------------------------------------------------------------
# Method B: batch inverse problem
# ---------------------------------------------------------------------------


def _spline_matrix(nodes: Vector) -> Matrix:
    """Return ``M`` with ``y'' = M y`` for the natural cubic spline on nodes.

    The natural spline through uniformly spaced nodes ``x_i`` (spacing ``d``)
    has second derivatives ``y''_i`` that solve the tridiagonal system
    ``y''_{i-1} + 4 y''_i + y''_{i+1} = (6 / d^2)(y_{i-1} - 2 y_i + y_{i+1})``
    with ``y''_0 = y''_{n-1} = 0``. The interpolant is linear in the node
    values, so ``M`` (computed once) turns any value vector into its second
    derivatives.

    Args:
        nodes: Ascending, uniformly spaced node altitudes (at least three).

    Returns:
        The ``n`` by ``n`` matrix ``M``.

    Equation:
        [EQ-AERO-RECON-006].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    n = len(nodes)
    d = nodes[1] - nodes[0]
    interior = n - 2
    tri = tuple(
        tuple(
            4.0 if i == j else 1.0 if abs(i - j) == 1 else 0.0 for j in range(interior)
        )
        for i in range(interior)
    )
    inverse = inverse_spd(tri)
    scale = 6.0 / (d * d)
    rows = [[0.0] * n]
    for i in range(interior):
        row = [0.0] * n
        for m in range(interior):
            weight = inverse[i][m] * scale
            row[m] += weight
            row[m + 1] -= 2.0 * weight
            row[m + 2] += weight
        rows.append(row)
    rows.append([0.0] * n)
    return tuple(tuple(row) for row in rows)


def _spline_row(nodes: Vector, matrix: Matrix, altitude: float) -> Vector:
    """Return the weights of ``c(altitude)`` on the node values.

    The altitude is clamped to the node range (the correction is held
    constant outside it).

    Args:
        nodes: Ascending, uniformly spaced node altitudes.
        matrix: Output of :func:`_spline_matrix`.
        altitude: Altitude in m.

    Returns:
        ``w`` with ``c(h) = w . c_nodes``.

    Equation:
        [EQ-AERO-RECON-006].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    n = len(nodes)
    d = nodes[1] - nodes[0]
    h = min(max(altitude, nodes[0]), nodes[-1])
    j = min(max(int((h - nodes[0]) / d), 0), n - 2)
    a = (nodes[j + 1] - h) / d
    b = (h - nodes[j]) / d
    ca = (a**3 - a) * d * d / 6.0
    cb = (b**3 - b) * d * d / 6.0
    row = [ca * matrix[j][i] + cb * matrix[j + 1][i] for i in range(n)]
    row[j] += a
    row[j + 1] += b
    return tuple(row)


def _spline_value(
    nodes: Vector,
    values: Sequence[float],
    second: Sequence[float],
    altitude: float,
) -> float:
    """Evaluate the natural cubic spline correction ``c(h)``.

    Args:
        nodes: Ascending, uniformly spaced node altitudes.
        values: Node values of ``c``.
        second: Node second derivatives ``M c``.
        altitude: Altitude in m; clamped to the node range.

    Returns:
        ``c(h)``.

    Equation:
        [EQ-AERO-RECON-006].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    d = nodes[1] - nodes[0]
    h = min(max(altitude, nodes[0]), nodes[-1])
    j = min(max(int((h - nodes[0]) / d), 0), len(nodes) - 2)
    a = (nodes[j + 1] - h) / d
    b = (h - nodes[j]) / d
    return (
        a * values[j]
        + b * values[j + 1]
        + ((a**3 - a) * second[j] + (b**3 - b) * second[j + 1]) * d * d / 6.0
    )


@dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    """Posterior summary of the density/ballistic-coefficient split.

    Args:
        altitude: Altitude in m at which ``c`` was read.
        prior_log_ballistic_sd: Prior sd of ``ln beta``.
        prior_log_density_sd: Prior sd of ``c``.
        log_ballistic_sd: Posterior sd of ``ln beta``.
        log_density_sd: Posterior sd of ``c(altitude)``.
        ratio_sd: Posterior sd of ``c(altitude) - ln beta`` (the ratio the
            kinematics determine, [EQ-AERO-RECON-002]).
        correlation: Posterior correlation of ``ln beta`` and ``c(altitude)``.

    References:
        [EQ-AERO-RECON-007].

    """

    altitude: float
    prior_log_ballistic_sd: float
    prior_log_density_sd: float
    log_ballistic_sd: float
    log_density_sd: float
    ratio_sd: float
    correlation: float


@dataclass(frozen=True, slots=True)
class BatchReconstruction:
    """Result of :func:`reconstruct_entry_batch`.

    Args:
        nodes: Node altitudes in m.
        node_values: Estimated ``c`` at the nodes.
        log_ballistic_offset: Estimated ``ln beta - ln beta_ref``.
        covariance: Laplace posterior covariance of
            ``(b, c_1, ..., c_n)`` (``(n + 1)`` square).
        prior_log_ballistic_sd: Prior sd of ``ln beta``.
        prior_log_density_sd: Prior sd ``s_c`` of ``c``.
        converged: ``True`` if Gauss-Newton stopped on ``"gradient"``,
            ``"step"`` or ``"discrepancy"``.
        stop_reason: The Gauss-Newton stop reason.
        iterations: Accepted Gauss-Newton iterations.
        objective: Final value of the penalized misfit.
        whitened_residual_rms: ``||W (G - y)|| / sqrt(m)`` at the estimate
            (about 1 when the model fits the noise).
        data_count: Number of scalar data used.
        correlation_length: Prior ``ell`` in m.
        spline_matrix: Matrix ``M`` mapping node values to node second
            derivatives of the natural cubic spline.

    References:
        [EQ-AERO-RECON-004], [EQ-AERO-RECON-005].

    """

    nodes: tuple[float, ...]
    node_values: tuple[float, ...]
    log_ballistic_offset: float
    covariance: Matrix
    prior_log_ballistic_sd: float
    prior_log_density_sd: float
    converged: bool
    stop_reason: str
    iterations: int
    objective: float
    whitened_residual_rms: float
    data_count: int
    correlation_length: float
    spline_matrix: Matrix

    def _row(self, altitude: float) -> Vector:
        """Return the weights of ``c(altitude)`` on ``(b, c_1..c_n)``.

        Args:
            altitude: Altitude in m.

        Returns:
            The sensitivity row.

        Equation:
            [EQ-AERO-RECON-006].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006].

        """
        return (0.0, *_spline_row(self.nodes, self.spline_matrix, altitude))

    def log_density_correction(
        self, altitudes: Sequence[float]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the posterior mean and sd of ``c`` at altitudes.

        Args:
            altitudes: Altitudes in m.

        Returns:
            ``(means, sds)`` of the correction ``c(h)``.

        Raises:
            TypeError: If an altitude is not a real number.

        Equation:
            [EQ-AERO-RECON-005], [EQ-AERO-RECON-006].

        References:
            [REF-BUI-THANH-ETAL-2013].

        """
        means: list[float] = []
        sds: list[float] = []
        for altitude in altitudes:
            row = self._row(require_finite_real(altitude, name="altitude"))
            means.append(
                sum(w * c for w, c in zip(row[1:], self.node_values, strict=True))
            )
            var = sum(
                row[i] * sum(self.covariance[i][j] * row[j] for j in range(len(row)))
                for i in range(len(row))
            )
            sds.append(math.sqrt(max(var, 0.0)))
        return tuple(means), tuple(sds)

    def identifiability(self, altitude: float) -> IdentifiabilityReport:
        """Measure how well ``ln beta`` and ``c(altitude)`` are separated.

        Args:
            altitude: Altitude in m at which ``c`` is read.

        Returns:
            The :class:`IdentifiabilityReport` from the Laplace covariance.

        Raises:
            TypeError: If ``altitude`` is not a real number.

        Equation:
            [EQ-AERO-RECON-007].

        References:
            [REF-BUI-THANH-ETAL-2013].

        """
        row = self._row(require_finite_real(altitude, name="altitude"))
        size = len(row)
        cov_row = [
            sum(self.covariance[i][j] * row[j] for j in range(size))
            for i in range(size)
        ]
        var_c = sum(row[i] * cov_row[i] for i in range(size))
        var_b = self.covariance[0][0]
        cov_bc = cov_row[0]
        var_ratio = var_c - 2.0 * cov_bc + var_b
        sd_b, sd_c = math.sqrt(max(var_b, 0.0)), math.sqrt(max(var_c, 0.0))
        corr = cov_bc / (sd_b * sd_c) if sd_b > 0.0 and sd_c > 0.0 else 0.0
        return IdentifiabilityReport(
            altitude,
            self.prior_log_ballistic_sd,
            self.prior_log_density_sd,
            sd_b,
            sd_c,
            math.sqrt(max(var_ratio, 0.0)),
            corr,
        )


def _prior_factor(setup: EntryReconstructionSetup, beta_sd: float) -> Matrix:
    """Return the lower Cholesky factor of the joint prior covariance.

    The prior of ``(b, c_1..c_n)`` is block diagonal: ``N(0, beta_sd^2)`` for
    ``b`` and the exponential-kernel Gaussian process
    ``Cov(c_i, c_j) = s_c^2 exp(-|h_i - h_j| / ell)`` at the nodes.

    Args:
        setup: The scenario.
        beta_sd: Prior sd of ``ln beta``.

    Returns:
        Lower-triangular ``F`` with ``Sigma_prior = F F'``.

    Raises:
        ValueError: If the covariance is not positive definite.

    Equation:
        [EQ-AERO-RECON-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006].

    """
    nodes = setup.node_altitudes
    size = setup.nodes + 1
    matrix = [[0.0] * size for _ in range(size)]
    matrix[0][0] = beta_sd**2
    for i in range(setup.nodes):
        for j in range(setup.nodes):
            matrix[1 + i][1 + j] = setup.density_log_sd**2 * math.exp(
                -abs(nodes[i] - nodes[j]) / setup.correlation_length
            )
    return cholesky(tuple(tuple(row) for row in matrix))


def reconstruct_entry_batch(
    setup: EntryReconstructionSetup,
    measurements: EntryMeasurements,
    *,
    use_radar: bool = True,
    use_anchors: bool = True,
    max_iterations: int = 30,
) -> BatchReconstruction:
    """Reconstruct the entry as a batch Bayesian inverse problem.

    Unknowns ``u = (b, c_1..c_n)`` with the Gaussian prior
    ``N(0, F F')`` ([EQ-AERO-RECON-004]). The forward map ``G(u)`` integrates
    the entry with ``rho = rho_base e^{c(h)}`` (``c`` the natural cubic spline through
    the node values, constant outside the range) and ``beta = beta_ref e^b``,
    and returns the sensed deceleration at the available samples, the radar
    fixes and the anchors ([EQ-AERO-RECON-002]). With the whitened coordinates
    ``u = F z`` the MAP point minimizes
    ``1/2 ||W (G(F z) - y)||^2 + 1/2 ||z||^2`` and is found by
    :func:`probstat.inverse.gauss_newton_krylov` with a forward-difference
    Jacobian (step ``1e-5`` in the log-scaled unknowns). The Laplace
    covariance is ``F (I + J' W' W J)^-1 F'`` with ``J = dG/dz`` at the
    estimate (the Gauss-Newton Hessian, [EQ-AERO-RECON-005]); the Laplace
    approximation is exact only for a linear ``G``.

    Args:
        setup: The scenario.
        measurements: Flight data.
        use_radar: Use the radar fixes if present.
        use_anchors: Use the density anchors if present.
        max_iterations: Gauss-Newton iteration cap.

    Returns:
        The :class:`BatchReconstruction`. Non-convergence is reported in
        ``converged`` and ``stop_reason``, never hidden.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the data length does not match the setup, there is no
            datum, or the prior is not positive definite.

    Equation:
        [EQ-AERO-RECON-004], [EQ-AERO-RECON-005].

    References:
        [REF-BUI-THANH-ETAL-2013], [REF-KAIPIO-SOMERSALO-2007],
        [REF-TARANTOLA-2005], [REF-KARLGAARD-KUTTY-SCHOENENBERGER-MUNK-2014].

    """
    if not isinstance(setup, EntryReconstructionSetup) or not isinstance(
        measurements, EntryMeasurements
    ):
        raise TypeError("setup and measurements must be the module's dataclasses")
    n = setup.samples
    if len(measurements.acceleration) != n or len(measurements.radar) != n:
        raise ValueError("measurements do not match the setup's record length")
    base = _density_function(setup.base_atmosphere)
    nodes = setup.node_altitudes
    spline = _spline_matrix(nodes)
    acc_index = [k for k, a in enumerate(measurements.acceleration) if a is not None]
    radar_index = (
        [k for k, r in enumerate(measurements.radar) if r is not None]
        if use_radar
        else []
    )
    anchors = measurements.anchors if use_anchors else ()
    data: list[float] = [measurements.acceleration[k] for k in acc_index]  # type: ignore[misc]
    variances = [setup.accel_sigma**2] * len(acc_index)
    for k in radar_index:
        fix = measurements.radar[k]
        assert fix is not None  # noqa: S101 - narrowed by construction
        data.extend(fix)
        variances.extend((setup.radar_height_sigma**2, setup.radar_speed_sigma**2))
    data.extend(anchors)
    variances.extend([setup.anchor_sigma**2] * len(anchors))
    if not data:
        raise ValueError("no measurement is available")
    beta_ref = setup.ballistic_coefficient
    factor = _prior_factor(setup, setup.ballistic_log_sd)
    size = setup.nodes + 1
    nan_output = tuple(float("nan") for _ in data)

    def forward(u: Vector) -> Vector:
        """Evaluate the measurement map ``G(u)``.

        Args:
            u: Unknowns ``(b, c_1..c_n)``.

        Returns:
            The predicted data (NaN if the trajectory is invalid).

        Equation:
            [EQ-AERO-RECON-002].

        References:
            [REF-ALLEN-EGGERS-1958].

        """
        values = u[1:]
        second = [sum(m * v for m, v in zip(r, values, strict=True)) for r in spline]
        correction = partial(_spline_value, nodes, values, second)
        density = partial(_corrected_density, base=base, correction=correction)
        history = _history(
            setup, density, 0.5 / (beta_ref * math.exp(u[0])), setup.substeps
        )
        if history is None:
            return nan_output
        hs, vs, _gs, acc = history
        out = [acc[k + 1] for k in acc_index]
        for k in radar_index:
            out.extend((hs[k + 1], vs[k + 1]))
        out.extend(
            _spline_value(nodes, values, second, a)
            for a in setup.anchor_altitudes[: len(anchors)]
        )
        return tuple(out)

    def jacobian_u(u: Vector) -> list[Vector]:
        """Forward-difference Jacobian ``dG/du`` as a list of columns.

        Args:
            u: Unknowns.

        Returns:
            One column per unknown.

        Equation:
            [EQ-AERO-RECON-004].

        References:
            [REF-NOCEDAL-WRIGHT-2006].

        """
        base_value = forward(u)
        columns: list[Vector] = []
        for i in range(size):
            shifted = list(u)
            shifted[i] += _FD_STEP
            moved = forward(tuple(shifted))
            columns.append(
                tuple(
                    (a - b) / _FD_STEP for a, b in zip(moved, base_value, strict=True)
                )
            )
        return columns

    cache: dict[Vector, Matrix] = {}

    def jacobian_z(z: Vector) -> Matrix:
        """Jacobian ``dG/dz = (dG/du) F`` with a one-entry cache.

        Args:
            z: Whitened unknowns.

        Returns:
            Rows of the Jacobian, one per datum.

        Equation:
            [EQ-AERO-RECON-004].

        References:
            [REF-BUI-THANH-ETAL-2013].

        """
        if z in cache:
            return cache[z]
        u = tuple(sum(factor[i][j] * z[j] for j in range(i + 1)) for i in range(size))
        columns = jacobian_u(u)
        ju = tuple(tuple(columns[i][r] for i in range(size)) for r in range(len(data)))
        result = matmul(ju, factor)
        cache.clear()
        cache[z] = result
        return result

    def value_z(z: Vector) -> Vector:
        """Measurement map in whitened coordinates ``G(F z)``.

        Args:
            z: Whitened unknowns.

        Returns:
            The predicted data.

        Equation:
            [EQ-AERO-RECON-004].

        References:
            [REF-BUI-THANH-ETAL-2013].

        """
        return forward(
            tuple(sum(factor[i][j] * z[j] for j in range(i + 1)) for i in range(size))
        )

    model = CallbackForward(size, len(data), value_z, jacobian=jacobian_z)
    result = gauss_newton_krylov(
        model,
        data,
        noise=DiagonalCovariance(variances),
        regularization=1.0,
        initial=(0.0,) * size,
        max_iterations=max_iterations,
        gradient_tolerance=1e-7,
    )
    z_hat = result.estimate
    jz = jacobian_z(z_hat)
    weighted = tuple(
        tuple(entry / math.sqrt(var) for entry in row)
        for row, var in zip(jz, variances, strict=True)
    )
    hessian = matmul(transpose(weighted), weighted)
    hessian = tuple(
        tuple(hessian[i][j] + (1.0 if i == j else 0.0) for j in range(size))
        for i in range(size)
    )
    posterior_z = inverse_spd(hessian)
    covariance = matmul(matmul(factor, posterior_z), transpose(factor))
    u_hat = tuple(
        sum(factor[i][j] * z_hat[j] for j in range(i + 1)) for i in range(size)
    )
    return BatchReconstruction(
        nodes,
        tuple(u_hat[1:]),
        u_hat[0],
        covariance,
        setup.ballistic_log_sd,
        setup.density_log_sd,
        result.converged,
        result.stop_reason,
        result.iterations,
        result.objective,
        result.whitened_residual_norm / math.sqrt(len(data)),
        len(data),
        setup.correlation_length,
        spline,
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def band_coverage(
    truth: EntryTruth,
    altitudes: Sequence[float],
    mean: Sequence[float],
    sd: Sequence[float],
    *,
    z: float = 1.96,
    altitude_window: tuple[float, float] | None = None,
) -> tuple[int, int]:
    """Count how many true corrections lie inside a reconstructed band.

    The band is ``mean +- z sd`` of the log-density correction at the given
    altitudes (for example the measurement altitudes of the true flight).

    Args:
        truth: The synthetic truth.
        altitudes: Altitudes in m.
        mean: Reconstructed means of ``c``.
        sd: Reconstructed standard deviations of ``c``.
        z: Half-width in standard deviations.
        altitude_window: Optional ``(lowest, highest)`` altitude restricting
            the points counted.

    Returns:
        ``(inside, total)``.

    Raises:
        TypeError: If ``truth`` is not an :class:`EntryTruth`.
        ValueError: If the sequences differ in length or ``z`` is not
            positive.

    Equation:
        [EQ-AERO-RECON-007].

    References:
        [REF-PROBSTAT-VALIDATION].

    """
    if not isinstance(truth, EntryTruth):
        raise TypeError("truth must be an EntryTruth")
    if not len(altitudes) == len(mean) == len(sd):
        raise ValueError("altitudes, mean and sd must have equal length")
    half = _positive(z, "z")
    inside = total = 0
    for h, m, s in zip(altitudes, mean, sd, strict=True):
        if altitude_window is not None and not (
            altitude_window[0] <= h <= altitude_window[1]
        ):
            continue
        total += 1
        if abs(truth.log_density_correction(h) - m) <= half * s:
            inside += 1
    return inside, total
