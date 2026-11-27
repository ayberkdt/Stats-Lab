"""Inverse heat conduction: surface heat flux from embedded thermocouples.

Experimental (added in 0.28). The classic inverse heat conduction problem
(IHCP) as it arises for thermal-protection systems and rocket-nozzle walls:
estimate the time history ``q(t)`` of the heat flux entering a one-dimensional
slab at ``x = 0`` from noisy temperatures measured by thermocouples embedded
at depths ``x_s``.

All quantities are SI: lengths in m, times in s, temperatures (rises above the
uniform initial temperature) in K, heat flux in W/m^2, conductivity in
W/(m K), density in kg/m^3, specific heat in J/(kg K), diffusivity in m^2/s,
heat-transfer coefficient in W/(m^2 K). The model is linear, so properties are
constants: it is valid only while the temperature rise is small enough for
the property variation to be negligible, and for one-dimensional conduction
(no lateral losses, no ablation, no pyrolysis gas flow, no contact
resistance). The sensor is modelled as a point reading of the local
temperature (no thermocouple intrusion error). Property values given in this
module are SYNTHETIC and illustrative; they are not data for any real heat
shield, nozzle or vehicle.

Forward model [EQ-INV-IHCP-001]. Cell-centred finite volumes on ``nx`` equal
cells of width ``dx = L / nx`` and implicit Euler with step ``dt``:
``M theta^k = (rho c dx / dt) theta^(k-1) + e_0 q^k`` with the symmetric
tridiagonal ``M`` (conductance ``k / dx`` between cell centres, flux ``q``
entering cell ``0``, insulated or convective back face). The flux is
piecewise constant on ``n_q`` intervals of length ``Delta`` (each holding
``substeps`` time steps); readings are taken at the end of every
``sample_every``-th interval and the sensor temperature is the linear
interpolation between cell centres. :class:`SlabFluxOperator` is the
matrix-free :class:`probstat.linalg.LinearOperator` ``q -> y`` whose
``rmatvec`` is the EXACT discrete transpose, a backward recursion with one
tridiagonal solve per step [EQ-INV-IHCP-002].

Verification oracles. The semi-infinite-solid response to a constant flux
[EQ-INV-IHCP-003], the finite insulated-slab series derived from the
eigenfunction expansion [EQ-INV-IHCP-004], and the exact discrete energy
balance [EQ-INV-IHCP-005].

Because the model is linear and time invariant, with readings at interval
ends the operator is a block lower-triangular Toeplitz matrix whose entries are
increments ``phi_s(n) = Phi_s(n) - Phi_s(n - 1)`` of the unit-step response
``Phi_s`` (Beck's sensitivity coefficients) [EQ-INV-IHCP-006].

Inverse methods:

* :func:`beck_sequential_estimate`: Beck's sequential function specification
  with ``r`` future time steps [EQ-INV-IHCP-007]; the estimator is linear and
  :func:`beck_linear_map` returns its matrix, from which the noise gain
  and the bias follow [EQ-INV-IHCP-008];
* :func:`tikhonov_flux_estimate`: whole-domain Tikhonov with a zeroth- or
  first-order penalty in time, the regularization parameter chosen by GCV,
  the discrepancy principle or the L-curve through
  :mod:`probstat.inverse.parameter_selection` [EQ-INV-IHCP-009];
* :func:`bayesian_flux_posterior`: the exact linear-Gaussian posterior with a
  Matern Gaussian-process prior on ``q(t)`` and pointwise credible bands
  [EQ-INV-IHCP-010];
* :func:`cgls_flux_estimate`: CGLS stopped by the discrepancy principle
  [EQ-INV-IHCP-012].

Resolution. A sensor at depth ``x`` sees a flux impulse delayed by about
``x^2 / (2 alpha)`` and blurred over a comparable time
(:func:`impulse_response_peak_time`, [EQ-INV-IHCP-011]);
:func:`averaging_kernel` and :func:`kernel_resolution` quantify the lag and the
width of the Tikhonov estimate, and :func:`sensitivity_summary` tabulates the
sensitivity-coefficient magnitude against depth.

References:
    [REF-BECK-BLACKWELL-STCLAIR-1985], [REF-CARSLAW-JAEGER-1959],
    [REF-HANSEN-2008], [REF-KAIPIO-SOMERSALO-2007], [REF-RASMUSSEN-WILLIAMS-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import repeat
from operator import mul

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.linear_gaussian import (
    GaussianLinearPosterior,
    linear_gaussian_posterior,
)
from probstat.inverse.linear_problem import LinearInverseProblem
from probstat.inverse.parameter_selection import (
    RegularizationPath,
    logarithmic_grid,
    regularization_path,
    select_regularization,
)
from probstat.inverse.pde_models import TridiagonalFactorization
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    FirstDifferenceOperator,
    IdentityOperator,
    LinearOperator,
    Matrix,
    OperatorBase,
    ScaledIdentityCovariance,
    Vector,
    as_vector,
    cgls,
    cholesky,
    cholesky_solve,
    materialize,
    require_operator,
    symmetric_eigen,
)

__all__ = [
    "BeckEstimate",
    "EnergyBalance",
    "EqualNoiseResolution",
    "FluxEstimate",
    "FluxPosterior",
    "KernelResolution",
    "SensitivityRow",
    "SlabFluxOperator",
    "SlabProperties",
    "SyntheticIHCPData",
    "averaging_kernel",
    "bayesian_flux_posterior",
    "beck_linear_map",
    "beck_noise_gain",
    "beck_sequential_estimate",
    "cgls_flux_estimate",
    "impulse_response_peak_time",
    "insulated_slab_flux_response",
    "kernel_resolution",
    "matern_covariance",
    "resolution_at_noise_level",
    "select_gp_hyperparameters",
    "semi_infinite_flux_response",
    "sensitivity_summary",
    "step_pulse",
    "synthetic_ablator_slab",
    "synthetic_ihcp_data",
    "tikhonov_flux_estimate",
    "triangular_pulse",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _real(name: str, value: object) -> float:
    """Validate a finite real number.

    Args:
        name: Argument name for messages.
        value: Candidate.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite.

    References:
        [EQ-INV-IHCP-001]. Input validation only.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(name: str, value: object) -> float:
    """Validate a strictly positive finite real.

    Args:
        name: Argument name for messages.
        value: Candidate.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not positive and finite.

    References:
        [EQ-INV-IHCP-001]. Input validation only.

    """
    number = _real(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _integer(name: str, value: object, minimum: int) -> int:
    """Validate an integer lower bound.

    Args:
        name: Argument name for messages.
        value: Candidate.
        minimum: Smallest admissible value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < minimum``.

    References:
        [EQ-INV-IHCP-001]. Input validation only.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _normal_quantile(probability: float) -> float:
    """Return the standard normal quantile by bisection on ``erf``.

    Args:
        probability: Probability in ``(0.5, 1)``.

    Returns:
        ``z`` with ``Phi(z) = probability`` to about ``1e-15``.

    References:
        [EQ-INV-IHCP-010]. The normal distribution function is
        ``Phi(z) = (1 + erf(z / sqrt(2))) / 2``.

    """
    low, high = 0.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Closed-form oracles
# ---------------------------------------------------------------------------


def semi_infinite_flux_response(
    conductivity: float,
    diffusivity: float,
    depth: float,
    time: float,
    flux: float = 1.0,
) -> float:
    """Temperature rise of a semi-infinite solid under a constant surface flux.

    ``theta(x, t) = (2 q / k) sqrt(alpha t / pi) exp(-x^2 / (4 alpha t))
    - (q x / k) erfc(x / (2 sqrt(alpha t)))`` for ``t > 0`` and ``0`` at
    ``t = 0`` [EQ-INV-IHCP-003]. It is the exact solution of
    ``theta_t = alpha theta_xx``, ``-k theta_x(0, t) = q``, ``theta(x, 0) = 0``,
    ``theta -> 0`` as ``x -> infinity``, and the early-time limit of the finite
    slab (valid until the thermal wave of depth ``~ 4 sqrt(alpha t)`` reaches
    the back face).

    Args:
        conductivity: ``k`` in W/(m K).
        diffusivity: ``alpha`` in m^2/s.
        depth: ``x >= 0`` in m.
        time: ``t >= 0`` in s.
        flux: ``q`` in W/m^2 (default ``1``, giving the step response).

    Returns:
        The temperature rise in K.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``k`` or ``alpha`` is not positive, or ``x`` or ``t`` is
            negative or non-finite.

    References:
        [EQ-INV-IHCP-003], [REF-CARSLAW-JAEGER-1959].

    """
    k = _positive("conductivity", conductivity)
    alpha = _positive("diffusivity", diffusivity)
    x = _real("depth", depth)
    t = _real("time", time)
    q = _real("flux", flux)
    if x < 0.0 or t < 0.0:
        raise ValueError("depth and time must be non-negative")
    if t == 0.0:
        return 0.0
    root = math.sqrt(alpha * t)
    return (2.0 * q / k) * (root / math.sqrt(math.pi)) * math.exp(
        -(x * x) / (4.0 * alpha * t)
    ) - (q * x / k) * math.erfc(x / (2.0 * root))


def insulated_slab_flux_response(
    conductivity: float,
    diffusivity: float,
    thickness: float,
    depth: float,
    time: float,
    flux: float = 1.0,
    *,
    terms: int = 400,
) -> float:
    """Temperature rise of an insulated slab under a constant surface flux.

    Series solution of ``theta_t = alpha theta_xx`` on ``(0, L)`` with
    ``-k theta_x(0) = q``, ``theta_x(L) = 0``, ``theta(x, 0) = 0``
    [EQ-INV-IHCP-004]:

    ``theta = q alpha t / (k L) + q (x - L)^2 / (2 k L) - q L / (6 k)
    - (2 q L / (k pi^2)) sum_(n>=1) n^-2 exp(-n^2 pi^2 alpha t / L^2)
    cos(n pi x / L)``.

    The first terms are a particular solution that grows linearly in time;
    the constant ``-q L / (6 k)`` makes the remainder have zero mean so that
    it expands in the decaying cosine modes, whose coefficients follow from
    ``int_0^L (x - L)^2 cos(n pi x / L) dx = 2 L^3 / (n pi)^2``. At ``t = 0``
    the series sums to zero (checked in the tests). The truncated tail is
    bounded by ``(2 q L / (k pi^2)) sum_(n>terms) n^-2``, about
    ``2 q L / (k pi^2 terms)``, and shrinks exponentially for ``t > 0``.

    Args:
        conductivity: ``k`` in W/(m K).
        diffusivity: ``alpha`` in m^2/s.
        thickness: ``L`` in m.
        depth: ``x`` in ``[0, L]``.
        time: ``t >= 0`` in s.
        flux: ``q`` in W/m^2.
        terms: Number of retained modes.

    Returns:
        The temperature rise in K.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-IHCP-004], [REF-CARSLAW-JAEGER-1959].

    """
    k = _positive("conductivity", conductivity)
    alpha = _positive("diffusivity", diffusivity)
    big_l = _positive("thickness", thickness)
    x = _real("depth", depth)
    t = _real("time", time)
    q = _real("flux", flux)
    count = _integer("terms", terms, 1)
    if not 0.0 <= x <= big_l:
        raise ValueError("depth must lie in [0, thickness]")
    if t < 0.0:
        raise ValueError("time must be non-negative")
    total = 0.0
    for n in range(1, count + 1):
        total += (
            math.exp(-((n * math.pi) ** 2) * alpha * t / big_l**2)
            * math.cos(n * math.pi * x / big_l)
            / (n * n)
        )
    return (
        q * alpha * t / (k * big_l)
        + q * (x - big_l) ** 2 / (2.0 * k * big_l)
        - q * big_l / (6.0 * k)
        - 2.0 * q * big_l / (k * math.pi**2) * total
    )


def impulse_response_peak_time(depth: float, diffusivity: float) -> float:
    """Time of the maximum of the semi-infinite impulse response at a depth.

    The response of a semi-infinite solid to a flux impulse is
    ``proportional to t^(-1/2) exp(-x^2 / (4 alpha t))``; setting the
    logarithmic derivative to zero gives the peak time ``x^2 / (2 alpha)``
    [EQ-INV-IHCP-011]. It is the time lag (and the order of the smoothing
    time) with which a sensor at depth ``x`` sees a flux event.

    Args:
        depth: ``x > 0`` in m.
        diffusivity: ``alpha > 0`` in m^2/s.

    Returns:
        ``x^2 / (2 alpha)`` in s.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If an argument is not positive and finite.

    References:
        [EQ-INV-IHCP-011], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    x = _positive("depth", depth)
    alpha = _positive("diffusivity", diffusivity)
    return x * x / (2.0 * alpha)


# ---------------------------------------------------------------------------
# Slab, forward operator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlabProperties:
    """Constant thermophysical properties of a one-dimensional slab (SI).

    Attributes:
        conductivity: ``k`` in W/(m K).
        density: ``rho`` in kg/m^3.
        specific_heat: ``c`` in J/(kg K).
        thickness: ``L`` in m (heated face ``x = 0``, back face ``x = L``).
        back_coefficient: Back-face heat-transfer coefficient ``h`` in
            W/(m^2 K) to an ambient at the initial temperature; ``0`` is an
            insulated back face.

    References:
        [EQ-INV-IHCP-001], [REF-CARSLAW-JAEGER-1959].

    """

    conductivity: float
    density: float
    specific_heat: float
    thickness: float
    back_coefficient: float = 0.0

    def __post_init__(self) -> None:
        """Validate the properties.

        Raises:
            TypeError: If a property is not a real number.
            ValueError: If a property is not positive (the back coefficient
                must be non-negative) or not finite.

        References:
            [EQ-INV-IHCP-001]. Input validation only.

        """
        _positive("conductivity", self.conductivity)
        _positive("density", self.density)
        _positive("specific_heat", self.specific_heat)
        _positive("thickness", self.thickness)
        if _real("back_coefficient", self.back_coefficient) < 0.0:
            raise ValueError("back_coefficient must be non-negative")

    @property
    def diffusivity(self) -> float:
        """Return ``alpha = k / (rho c)`` in m^2/s.

        Returns:
            The thermal diffusivity.

        References:
            [EQ-INV-IHCP-001].

        """
        return self.conductivity / (self.density * self.specific_heat)

    @property
    def effusivity(self) -> float:
        """Return ``sqrt(k rho c)`` in W s^(1/2) / (m^2 K).

        Returns:
            The thermal effusivity; the early-time surface temperature rise
            under a flux ``q`` is ``2 q sqrt(t / pi) / effusivity``.

        References:
            [EQ-INV-IHCP-003].

        """
        return math.sqrt(self.conductivity * self.density * self.specific_heat)


def synthetic_ablator_slab(*, back_coefficient: float = 0.0) -> SlabProperties:
    """Return a SYNTHETIC carbon-phenolic-like slab for illustrations and tests.

    ``k = 1.0 W/(m K)``, ``rho = 1450 kg/m^3``, ``c = 1250 J/(kg K)``,
    ``L = 0.02 m``, hence ``alpha = 5.5e-7 m^2/s``. The values are of the
    order of published ablator-class properties but are NOT data for any real
    heat shield, nozzle or vehicle, and temperature dependence, pyrolysis and
    recession are ignored.

    Args:
        back_coefficient: Back-face ``h`` in W/(m^2 K); ``0`` is insulated.

    Returns:
        The slab properties.

    Raises:
        TypeError: If ``back_coefficient`` is not a real number.
        ValueError: If it is negative or not finite.

    References:
        [EQ-INV-IHCP-001]. A synthetic case, not a data source.

    """
    return SlabProperties(1.0, 1450.0, 1250.0, 0.02, back_coefficient)


@dataclass(frozen=True, slots=True)
class EnergyBalance:
    """Discrete energy balance per unit area of one simulation (J/m^2).

    Attributes:
        input_energy: ``dt sum_k q^k``, energy supplied by the surface flux.
        stored_energy: ``rho c dx sum_i theta_i`` at the final time.
        lost_energy: ``dt sum_k G_b theta_back^k`` leaving through the back
            face (zero for an insulated back face).
        relative_residual: ``|input - stored - lost| / max(|input|, tiny)``,
            zero up to rounding for the implicit-Euler scheme.

    References:
        [EQ-INV-IHCP-005].

    """

    input_energy: float
    stored_energy: float
    lost_energy: float
    relative_residual: float


class SlabFluxOperator(OperatorBase):
    """Linear map from the surface heat flux to embedded-sensor temperatures.

    Unknown: ``n_q = flux_intervals`` values of the surface flux ``q_j``
    (W/m^2), piecewise constant on intervals of length ``interval_length``.
    Output: temperature rises (K) at every ``sample_every``-th interval end
    and each sensor, time-major (row ``m * n_sensors + s``). The forward map is
    the implicit-Euler finite-volume march of [EQ-INV-IHCP-001]; ``rmatvec``
    is the EXACT transpose, the backward recursion
    ``nu^k = M^-1 (H_k' y_k + c nu^(k+1))``, ``(A' y)_j = sum_(k in j)
    nu^k_0``, one tridiagonal solve per step [EQ-INV-IHCP-002]. The
    initial temperature is uniform (zero rise).

    Attributes:
        slab: The :class:`SlabProperties`.
        sensors: Sensor depths in m.
        cells: Number of cells ``nx``.
        substeps: Time steps per flux interval.
        interval_length: Flux interval ``Delta`` in s.
        sample_every: Readings every this many flux intervals.

    References:
        [EQ-INV-IHCP-001], [EQ-INV-IHCP-002], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """

    __slots__ = (
        "_dt",
        "_dx",
        "_factorization",
        "_flux_intervals",
        "_heat_capacity_rate",
        "_interp",
        "_n_reads",
        "_n_sensors",
        "_tail",
        "cells",
        "interval_length",
        "sample_every",
        "sensors",
        "slab",
        "substeps",
    )

    def __init__(
        self,
        slab: SlabProperties,
        sensors: Sequence[float],
        *,
        flux_intervals: int,
        interval_length: float,
        cells: int = 40,
        substeps: int = 4,
        sample_every: int = 1,
    ) -> None:
        """Validate the configuration and factorize the step matrix.

        Args:
            slab: Slab properties.
            sensors: Depths ``x_s`` in m, each in ``[dx / 2, L - dx / 2]``
                (between the first and last cell centres).
            flux_intervals: Number ``n_q >= 1`` of flux intervals.
            interval_length: ``Delta > 0`` in s.
            cells: Number ``nx >= 2`` of cells.
            substeps: Time steps per flux interval (``>= 1``).
            sample_every: Reading period in flux intervals; it must divide
                ``flux_intervals``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If an argument is out of range, a sensor lies outside
                the cell-centre range or ``sample_every`` does not divide
                ``flux_intervals``.

        References:
            [EQ-INV-IHCP-001].

        """
        if not isinstance(slab, SlabProperties):
            raise TypeError("slab must be a SlabProperties")
        nx = _integer("cells", cells, 2)
        n_q = _integer("flux_intervals", flux_intervals, 1)
        sub = _integer("substeps", substeps, 1)
        every = _integer("sample_every", sample_every, 1)
        delta = _positive("interval_length", interval_length)
        if n_q % every != 0:
            raise ValueError("sample_every must divide flux_intervals")
        if isinstance(sensors, (str, bytes)):
            raise TypeError("sensors must be a sequence of depths")
        depths = tuple(_real("sensor depth", s) for s in sensors)
        if not depths:
            raise ValueError("at least one sensor is required")
        dx = slab.thickness / nx
        for depth in depths:
            if not 0.5 * dx <= depth <= slab.thickness - 0.5 * dx:
                raise ValueError(
                    f"sensor depth {depth} must lie in [dx/2, L - dx/2] = "
                    f"[{0.5 * dx}, {slab.thickness - 0.5 * dx}]"
                )
        self.slab = slab
        self.sensors = depths
        self.cells = nx
        self.substeps = sub
        self.interval_length = delta
        self.sample_every = every
        self._flux_intervals = n_q
        self._n_sensors = len(depths)
        self._n_reads = n_q // every
        self._dx = dx
        self._dt = delta / sub
        self._heat_capacity_rate = slab.density * slab.specific_heat * dx / self._dt
        conductance = slab.conductivity / dx
        h = slab.back_coefficient
        tail = 0.0 if h == 0.0 else 1.0 / (dx / (2.0 * slab.conductivity) + 1.0 / h)
        self._tail = tail
        diagonal = [self._heat_capacity_rate + 2.0 * conductance] * nx
        diagonal[0] -= conductance
        diagonal[-1] += tail - conductance
        off = [-conductance] * (nx - 1)
        self._factorization = TridiagonalFactorization(off, diagonal, off)
        interp: list[tuple[int, float]] = []
        for depth in depths:
            position = depth / dx - 0.5
            left = min(max(int(math.floor(position)), 0), nx - 2)
            interp.append((left, position - left))
        self._interp = tuple(interp)
        super().__init__(
            self._n_reads * self._n_sensors, n_q, name="slab flux operator"
        )

    @property
    def flux_intervals(self) -> int:
        """Return the number ``n_q`` of flux intervals.

        Returns:
            ``n_q``.

        References:
            [EQ-INV-IHCP-001].

        """
        return self._flux_intervals

    @property
    def reading_count(self) -> int:
        """Return the number of reading times per sensor.

        Returns:
            ``n_q / sample_every``.

        References:
            [EQ-INV-IHCP-001].

        """
        return self._n_reads

    @property
    def reading_times(self) -> tuple[float, ...]:
        """Return the reading times in s.

        Returns:
            ``(m * sample_every * Delta)`` for ``m = 1 .. n_reads``.

        References:
            [EQ-INV-IHCP-001].

        """
        return tuple(
            m * self.sample_every * self.interval_length
            for m in range(1, self._n_reads + 1)
        )

    @property
    def flux_times(self) -> tuple[float, ...]:
        """Return the midpoint time of every flux interval in s.

        Returns:
            ``((j + 1/2) Delta)`` for ``j = 0 .. n_q - 1``.

        References:
            [EQ-INV-IHCP-001].

        """
        return tuple(
            (j + 0.5) * self.interval_length for j in range(self._flux_intervals)
        )

    @property
    def time_step(self) -> float:
        """Return the time step ``dt = Delta / substeps`` in s.

        Returns:
            ``dt``.

        References:
            [EQ-INV-IHCP-001].

        """
        return self._dt

    def _march(self, flux: Sequence[float]) -> Iterator[tuple[int, list[float]]]:
        """Yield ``(k, theta^k)`` for every time step of the implicit-Euler march.

        Args:
            flux: Interval fluxes ``q_j`` (length ``n_q``).

        Yields:
            The step index ``k = 1 .. K`` and the cell temperatures.

        References:
            [EQ-INV-IHCP-001].

        """
        theta = [0.0] * self.cells
        total = self._flux_intervals * self.substeps
        capacity = self._heat_capacity_rate
        substeps = self.substeps
        solve = self._factorization.solve
        for k in range(1, total + 1):
            rhs = list(map(mul, repeat(capacity), theta))
            rhs[0] += flux[(k - 1) // substeps]
            theta = solve(rhs)
            yield k, theta

    def _sensor_values(self, theta: Sequence[float]) -> list[float]:
        """Interpolate the cell temperatures to the sensor depths.

        Args:
            theta: Cell temperatures.

        Returns:
            One value per sensor.

        References:
            [EQ-INV-IHCP-001].

        """
        out = []
        for left, weight in self._interp:
            out.append((1.0 - weight) * theta[left] + weight * theta[left + 1])
        return out

    def _forward(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply ``A`` by implicit-Euler marching.

        Args:
            vector: Interval fluxes.

        Returns:
            Sensor temperature rises.

        References:
            [EQ-INV-IHCP-001].

        """
        out = [0.0] * (self._n_reads * self._n_sensors)
        stride = self.sample_every * self.substeps
        for k, theta in self._march(vector):
            if k % stride == 0:
                row = (k // stride - 1) * self._n_sensors
                out[row : row + self._n_sensors] = self._sensor_values(theta)
        return out

    def _transpose(self, vector: tuple[float, ...]) -> Sequence[float]:
        """Apply the exact transpose by the backward recursion.

        Args:
            vector: Observation-space vector.

        Returns:
            ``A' y``.

        References:
            [EQ-INV-IHCP-002].

        """
        stride = self.sample_every * self.substeps
        total = self._flux_intervals * self.substeps
        result = [0.0] * self._flux_intervals
        nu = [0.0] * self.cells
        capacity = self._heat_capacity_rate
        substeps = self.substeps
        solve = self._factorization.solve
        interp = tuple((left, weight, 1.0 - weight) for left, weight in self._interp)
        for k in range(total, 0, -1):
            lam = list(map(mul, repeat(capacity), nu))
            if k % stride == 0:
                row = (k // stride - 1) * self._n_sensors
                for s, (left, weight, rest) in enumerate(interp):
                    lam[left] += rest * vector[row + s]
                    lam[left + 1] += weight * vector[row + s]
            nu = solve(lam)
            result[(k - 1) // substeps] += nu[0]
        return result

    def field_history(self, flux: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        """Return the cell temperatures at every time step, ``k = 0 .. K``.

        Args:
            flux: Interval fluxes (length ``n_q``).

        Returns:
            ``K + 1`` rows of ``nx`` cell temperature rises (row ``0`` is the
            zero initial state).

        Raises:
            TypeError: If ``flux`` is not a real vector.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-IHCP-001].

        """
        q = self._flux(flux)
        history = [tuple([0.0] * self.cells)]
        for _, theta in self._march(q):
            history.append(tuple(theta))
        return tuple(history)

    def _flux(self, flux: Sequence[float]) -> Vector:
        """Validate a flux vector.

        Args:
            flux: Candidate fluxes.

        Returns:
            The vector.

        Raises:
            ValueError: If the length is not ``n_q``.

        References:
            [EQ-INV-IHCP-001]. Input validation only.

        """
        q = as_vector(flux, name="flux")
        if len(q) != self._flux_intervals:
            raise ValueError(f"flux must have length {self._flux_intervals}")
        return q

    def energy_balance(self, flux: Sequence[float]) -> EnergyBalance:
        """Return the discrete energy balance of one simulation.

        With implicit Euler, summing the cell equations over cells and steps
        gives exactly ``dt sum_k q^k = rho c dx sum_i theta_i^K
        + dt sum_k G_b theta_(nx-1)^k`` (``G_b`` the back-face conductance)
        [EQ-INV-IHCP-005].

        Args:
            flux: Interval fluxes (length ``n_q``).

        Returns:
            :class:`EnergyBalance`.

        Raises:
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-IHCP-005].

        """
        q = self._flux(flux)
        lost = 0.0
        final: list[float] = []
        supplied = 0.0
        for k, theta in self._march(q):
            supplied += q[(k - 1) // self.substeps]
            lost += self._tail * theta[-1]
            final = theta
        input_energy = self._dt * supplied
        lost_energy = self._dt * lost
        stored = self.slab.density * self.slab.specific_heat * self._dx * sum(final)
        residual = abs(input_energy - stored - lost_energy) / max(
            abs(input_energy), 1e-300
        )
        return EnergyBalance(input_energy, stored, lost_energy, residual)

    def step_response(self) -> tuple[tuple[float, ...], ...]:
        """Return the unit-step response ``Phi_s(m)`` per sensor (K per W/m^2).

        ``Phi_s(m)`` is the temperature at sensor ``s`` and the ``m``-th
        reading time when a unit flux is applied from ``t = 0`` on, the
        discrete counterpart of ``semi_infinite_flux_response`` with
        ``q = 1`` [EQ-INV-IHCP-006]. It is ``A`` applied to the all-ones flux.

        Returns:
            ``Phi[s][m - 1]`` for ``m = 1 .. n_reads``.

        References:
            [EQ-INV-IHCP-006].

        """
        y = self.matvec([1.0] * self._flux_intervals)
        ns = self._n_sensors
        return tuple(
            tuple(y[m * ns + s] for m in range(self._n_reads)) for s in range(ns)
        )

    def sensitivity_matrix(self) -> Matrix:
        """Return the dense block lower-triangular Toeplitz matrix of ``A``.

        Valid for ``sample_every = 1``. Entry ``(m, s ; j)`` (reading ``m`` is
        1-based) is ``phi_s(m - j)`` for ``j < m`` with
        ``phi_s(n) = Phi_s(n) - Phi_s(n - 1)``, zero otherwise
        [EQ-INV-IHCP-006]. It is built from one forward run, not ``n_q``.

        Returns:
            The ``(n_reads n_sensors) x n_q`` matrix.

        Raises:
            ValueError: If ``sample_every != 1``.

        References:
            [EQ-INV-IHCP-006].

        """
        if self.sample_every != 1:
            raise ValueError("sensitivity_matrix needs sample_every = 1")
        phi = self.impulse_coefficients()
        ns = self._n_sensors
        n = self._flux_intervals
        rows: list[tuple[float, ...]] = []
        for m in range(n):
            for s in range(ns):
                rows.append(tuple(phi[s][m - j] if j <= m else 0.0 for j in range(n)))
        return tuple(rows)

    def impulse_coefficients(self) -> tuple[tuple[float, ...], ...]:
        """Return ``phi_s(n)``, the response to one flux interval of unit flux.

        ``phi_s(n) = Phi_s(n) - Phi_s(n - 1)`` (``Phi_s(0) = 0``) is the
        sensor temperature ``n`` intervals after a unit flux held for one
        interval [EQ-INV-IHCP-006].

        Returns:
            ``phi[s][n - 1]`` for ``n = 1 .. n_reads`` (``sample_every = 1``).

        Raises:
            ValueError: If ``sample_every != 1``.

        References:
            [EQ-INV-IHCP-006].

        """
        if self.sample_every != 1:
            raise ValueError("impulse_coefficients needs sample_every = 1")
        step = self.step_response()
        return tuple(
            tuple(row[i] - (row[i - 1] if i > 0 else 0.0) for i in range(len(row)))
            for row in step
        )


# ---------------------------------------------------------------------------
# Pulses and synthetic data
# ---------------------------------------------------------------------------


def triangular_pulse(
    time: float, peak: float, start: float, apex: float, end: float
) -> float:
    """Return a triangular heat-flux pulse ``q(t)`` in W/m^2.

    Zero before ``start``, linear up to ``peak`` at ``apex``, linear down to
    zero at ``end``, zero afterwards.

    Args:
        time: ``t`` in s.
        peak: Peak flux in W/m^2.
        start: Start time.
        apex: Time of the peak (``start < apex < end``).
        end: End time.

    Returns:
        ``q(t)``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``start < apex < end`` fails.

    References:
        [EQ-INV-IHCP-001]. A test waveform, not data.

    """
    t = _real("time", time)
    p = _real("peak", peak)
    a, b, c = _real("start", start), _real("apex", apex), _real("end", end)
    if not a < b < c:
        raise ValueError("need start < apex < end")
    if t <= a or t >= c:
        return 0.0
    if t <= b:
        return p * (t - a) / (b - a)
    return p * (c - t) / (c - b)


def step_pulse(time: float, level: float, start: float, end: float) -> float:
    """Return a rectangular (step) heat-flux pulse ``q(t)`` in W/m^2.

    Args:
        time: ``t`` in s.
        level: Flux level in W/m^2 on ``[start, end)``.
        start: Start time.
        end: End time (``end > start``).

    Returns:
        ``level`` on ``[start, end)``, else ``0``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``end <= start``.

    References:
        [EQ-INV-IHCP-001]. A test waveform, not data.

    """
    t = _real("time", time)
    lev = _real("level", level)
    a, b = _real("start", start), _real("end", end)
    if b <= a:
        raise ValueError("need end > start")
    return lev if a <= t < b else 0.0


@dataclass(frozen=True, slots=True)
class SyntheticIHCPData:
    """Synthetic sensor data generated on a finer model (no inverse crime).

    Attributes:
        clean: Noise-free sensor temperature rises (K), time-major.
        noisy: ``clean`` plus independent Gaussian noise.
        noise_std: Noise standard deviation ``sigma`` (K).
        truth: Mean of the fine-grid true flux over each coarse interval
            (W/m^2), the target of the estimators.
        seed: Seed used for the noise.

    References:
        [EQ-INV-IHCP-001].

    """

    clean: Vector
    noisy: Vector
    noise_std: float
    truth: Vector
    seed: int


def synthetic_ihcp_data(
    slab: SlabProperties,
    sensors: Sequence[float],
    flux: Callable[[float], float],
    *,
    flux_intervals: int,
    interval_length: float,
    refinement: int = 4,
    fine_cells: int = 120,
    fine_substeps: int = 4,
    noise_fraction: float = 0.01,
    seed: int | RandomStream = 0,
) -> SyntheticIHCPData:
    """Generate data from a finer mesh than the inversion model.

    The fine operator uses ``refinement`` times more flux intervals (each with
    ``fine_substeps`` steps) and ``fine_cells`` cells; the true flux
    ``flux(t)`` is sampled at the fine interval midpoints. Readings are taken
    at the coarse interval ends. The noise standard deviation is
    ``noise_fraction`` times the largest clean reading, a convention of this
    function, not a property of real thermocouples.

    Args:
        slab: Slab properties.
        sensors: Sensor depths in m.
        flux: True flux ``q(t)`` in W/m^2.
        flux_intervals: Number of coarse intervals ``n_q``.
        interval_length: Coarse interval ``Delta`` in s.
        refinement: Fine intervals per coarse one (``>= 1``).
        fine_cells: Fine cells.
        fine_substeps: Time steps per fine interval.
        noise_fraction: Noise standard deviation relative to the peak reading
            (``>= 0``).
        seed: Seed or stream for the noise.

    Returns:
        :class:`SyntheticIHCPData`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-IHCP-001], [REF-KAIPIO-SOMERSALO-2007].

    """
    n_q = _integer("flux_intervals", flux_intervals, 1)
    delta = _positive("interval_length", interval_length)
    ratio = _integer("refinement", refinement, 1)
    fraction = _real("noise_fraction", noise_fraction)
    if fraction < 0.0:
        raise ValueError("noise_fraction must be non-negative")
    fine = SlabFluxOperator(
        slab,
        sensors,
        flux_intervals=n_q * ratio,
        interval_length=delta / ratio,
        cells=fine_cells,
        substeps=fine_substeps,
        sample_every=ratio,
    )
    fine_flux = [
        _real("flux(t)", flux((j + 0.5) * delta / ratio)) for j in range(n_q * ratio)
    ]
    clean = tuple(fine.matvec(fine_flux))
    truth = tuple(
        math.fsum(fine_flux[i * ratio : (i + 1) * ratio]) / ratio for i in range(n_q)
    )
    sigma = fraction * max(abs(v) for v in clean)
    stream = seed if isinstance(seed, RandomStream) else resolve_stream(seed)
    generator = stream.generator()
    noisy = tuple(v + generator.gauss(0.0, sigma) for v in clean)
    return SyntheticIHCPData(
        clean, noisy, sigma, truth, seed if isinstance(seed, int) else -1
    )


# ---------------------------------------------------------------------------
# Beck's sequential function specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeckEstimate:
    """Result of Beck's sequential function specification.

    Attributes:
        estimate: ``q_hat_j`` (W/m^2) for the ``n_q`` flux intervals.
        future_steps: ``r``, the number of future time steps used.
        sensitivity: ``X_(s, l) = Phi_s(l)`` for ``l = 1 .. r`` per sensor, the
            sensitivity coefficients of the stage estimator.
        gain: ``sum_(s,l) X_(s,l)^2`` of the first stage (K^2 per (W/m^2)^2);
            the variance of the first-stage estimate is ``sigma^2 / gain``.

    References:
        [EQ-INV-IHCP-007].

    """

    estimate: Vector
    future_steps: int
    sensitivity: tuple[Vector, ...]
    gain: float


def _beck_core(
    phi: Sequence[Sequence[float]],
    step: Sequence[Sequence[float]],
    data: Sequence[float],
    future: int,
) -> list[float]:
    """Run the sequential estimator on one data vector.

    Args:
        phi: ``phi[s][n - 1]`` impulse coefficients.
        step: ``step[s][l - 1]`` step response.
        data: Time-major data ``y[m * n_s + s]``.
        future: ``r``.

    Returns:
        The estimated fluxes.

    Raises:
        ValueError: If a stage has zero sensitivity.

    References:
        [EQ-INV-IHCP-007].

    """
    ns = len(phi)
    nq = len(phi[0])
    predicted = [0.0] * (nq * ns)
    estimate: list[float] = []
    for j in range(nq):
        last = min(j + future, nq)
        numerator = 0.0
        denominator = 0.0
        for m in range(j, last):
            lag = m - j
            for s in range(ns):
                x = step[s][lag]
                numerator += x * (data[m * ns + s] - predicted[m * ns + s])
                denominator += x * x
        if denominator <= 0.0:
            raise ValueError("zero sensitivity: the sensors do not respond")
        q = numerator / denominator
        estimate.append(q)
        for m in range(j, nq):
            for s in range(ns):
                predicted[m * ns + s] += phi[s][m - j] * q
    return estimate


def _check_beck_operator(operator: SlabFluxOperator, future_steps: int) -> int:
    """Validate the operator and the number of future steps.

    Args:
        operator: Slab operator.
        future_steps: ``r``.

    Returns:
        ``r`` as an integer.

    Raises:
        TypeError: If the operator or ``r`` has the wrong type.
        ValueError: If ``sample_every != 1`` or ``r`` exceeds ``n_q``.

    References:
        [EQ-INV-IHCP-007]. Input validation only.

    """
    if not isinstance(operator, SlabFluxOperator):
        raise TypeError("operator must be a SlabFluxOperator")
    if operator.sample_every != 1:
        raise ValueError("Beck's method needs readings at every flux interval")
    r = _integer("future_steps", future_steps, 1)
    if r > operator.flux_intervals:
        raise ValueError("future_steps must not exceed the number of flux intervals")
    return r


def beck_sequential_estimate(
    operator: SlabFluxOperator,
    data: Sequence[float],
    *,
    future_steps: int = 3,
) -> BeckEstimate:
    """Estimate the flux history by sequential function specification.

    At stage ``j`` the flux is assumed constant, ``q_j = ... = q_(j+r-1) = q``,
    and the earlier estimates ``q_hat_i`` (``i < j``) are fixed. With
    ``X_(s,l) = Phi_s(l)`` the sensor response to a unit flux held ``l``
    intervals and the residual of the past
    ``e_(s,l) = y_(j+l, s) - sum_(i<j) phi_s(j + l - i) q_hat_i``, the
    least-squares estimate is ``q_hat_j = sum X e / sum X^2`` over
    ``l = 1 .. r`` and all sensors [EQ-INV-IHCP-007]. Only ``q_hat_j`` is
    kept and the window moves on. Near the end fewer than ``r`` readings
    exist and the window is truncated. ``r = 1`` is exact inversion of one
    reading (maximum noise amplification); larger ``r`` stabilizes the estimate
    at the price of a time lag and smoothing (bias). Equal, uncorrelated
    noise on all readings is assumed.

    Args:
        operator: The :class:`SlabFluxOperator` (``sample_every = 1``).
        data: Time-major sensor temperatures of length ``n_q n_sensors``.
        future_steps: ``r`` (``1 <= r <= n_q``).

    Returns:
        :class:`BeckEstimate`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the data length or ``r`` is invalid, or the sensors do
            not respond.

    References:
        [EQ-INV-IHCP-007], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    r = _check_beck_operator(operator, future_steps)
    y = as_vector(data, name="data")
    if len(y) != operator.shape[0]:
        raise ValueError(f"data must have length {operator.shape[0]}")
    phi = operator.impulse_coefficients()
    step = operator.step_response()
    estimate = _beck_core(phi, step, y, r)
    sensitivity = tuple(tuple(step[s][:r]) for s in range(len(step)))
    gain = math.fsum(v * v for row in sensitivity for v in row)
    return BeckEstimate(tuple(estimate), r, sensitivity, gain)


def beck_linear_map(operator: SlabFluxOperator, *, future_steps: int = 3) -> Matrix:
    """Return the matrix ``G`` of the linear estimator ``q_hat = G y``.

    The sequential estimator is linear in the data, so applying it to the unit
    vectors yields ``G`` (``n_q x m``). The estimator's mean for a true flux
    ``q`` is ``G A q`` (bias ``(G A - I) q``) and its covariance for noise
    ``sigma^2 I`` is ``sigma^2 G G'`` [EQ-INV-IHCP-008].

    Args:
        operator: The :class:`SlabFluxOperator` (``sample_every = 1``).
        future_steps: ``r``.

    Returns:
        ``G`` as a tuple of ``n_q`` rows of length ``m``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``r`` is invalid or the sensors do not respond.

    References:
        [EQ-INV-IHCP-008], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    r = _check_beck_operator(operator, future_steps)
    phi = operator.impulse_coefficients()
    step = operator.step_response()
    m = operator.shape[0]
    columns = []
    for i in range(m):
        unit = [0.0] * m
        unit[i] = 1.0
        columns.append(_beck_core(phi, step, unit, r))
    return tuple(tuple(columns[i][j] for i in range(m)) for j in range(len(columns[0])))


def beck_noise_gain(operator: SlabFluxOperator, *, future_steps: int = 3) -> float:
    """Return the RMS standard deviation of ``q_hat`` for unit noise.

    ``sqrt(trace(G G') / n_q)``: the noise standard deviation in the estimated
    flux (W/m^2) per kelvin of sensor noise, averaged over the flux
    intervals [EQ-INV-IHCP-008].

    Args:
        operator: The :class:`SlabFluxOperator` (``sample_every = 1``).
        future_steps: ``r``.

    Returns:
        The RMS amplification in (W/m^2) / K.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``r`` is invalid or the sensors do not respond.

    References:
        [EQ-INV-IHCP-008], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    g = beck_linear_map(operator, future_steps=future_steps)
    return math.sqrt(math.fsum(v * v for row in g for v in row) / len(g))


# ---------------------------------------------------------------------------
# Tikhonov, CGLS
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FluxEstimate:
    """A regularized flux estimate with the diagnostics of its selection.

    Attributes:
        estimate: The flux (W/m^2), or ``None`` when the selection rule found
            no admissible parameter (see ``status``).
        method: ``"tikhonov0"``, ``"tikhonov1"`` or ``"cgls"``.
        regularization: Chosen ``lambda`` (Tikhonov) or ``None``.
        status: The selection status of
            :class:`~probstat.inverse.parameter_selection.RegularizationSelection`
            (``"interior"``, ``"boundary"``, ``"unattainable"``, ...) or, for
            CGLS, ``"discrepancy"`` / ``"discrepancy_unmet"``.
        criterion: Selection criterion name.
        degrees_of_freedom: Effective degrees of freedom of the Tikhonov
            solution, or ``None``.
        residual_rms: Root-mean-square data residual (K), or ``None`` when no
            estimate exists.
        iterations: CGLS iterations, else ``None``.
        path: The regularization path for Tikhonov (L-curve and criteria),
            else ``None``.

    References:
        [EQ-INV-IHCP-009], [EQ-INV-IHCP-012].

    """

    estimate: Vector | None
    method: str
    regularization: float | None
    status: str
    criterion: str
    degrees_of_freedom: float | None
    residual_rms: float | None
    iterations: int | None
    path: RegularizationPath | None


def _dense_operator(operator: object) -> tuple[DenseOperator, Matrix]:
    """Materialize an operator or matrix once.

    Args:
        operator: A linear operator or dense matrix.

    Returns:
        The dense operator and its matrix.

    Raises:
        TypeError: If ``operator`` is not an operator or matrix.

    References:
        [EQ-INV-IHCP-009]. Input validation only.

    """
    if isinstance(operator, SlabFluxOperator) and operator.sample_every == 1:
        matrix = operator.sensitivity_matrix()
    else:
        matrix = materialize(require_operator(operator, name="operator"))
    return DenseOperator(matrix), matrix


def tikhonov_flux_estimate(
    operator: object,
    data: Sequence[float],
    noise_std: float,
    *,
    order: int = 0,
    criterion: str = "gcv",
    regularizations: Sequence[float] | None = None,
    tau: float = 1.0,
) -> FluxEstimate:
    """Whole-domain Tikhonov estimate with a zeroth- or first-order penalty.

    Minimizes ``||A q - y||^2 / sigma^2 + lambda^2 ||L q||^2`` with
    ``L = I`` (``order = 0``, penalizes the flux magnitude) or the first
    difference in time (``order = 1``, penalizes jumps and favours smooth
    ``q(t)``) [EQ-INV-IHCP-009]. ``lambda`` is chosen on a logarithmic grid
    from ``1e-7`` to ``10`` times the Frobenius norm of ``A / sigma`` by
    ``"gcv"``, ``"discrepancy"`` (target ``tau sqrt(m)`` in whitened units)
    or ``"l_curve"``, through
    :func:`~probstat.inverse.parameter_selection.regularization_path`. If the
    rule finds no admissible parameter the estimate is ``None`` and the status
    says why; nothing is substituted.

    Args:
        operator: ``A`` (a :class:`SlabFluxOperator`, a linear operator or a
            dense matrix).
        data: Sensor temperatures.
        noise_std: Noise standard deviation ``sigma`` in K (``> 0``).
        order: ``0`` or ``1``.
        criterion: ``"gcv"``, ``"discrepancy"`` or ``"l_curve"``.
        regularizations: Optional increasing positive grid of ``lambda``.
        tau: Discrepancy safety factor (``>= 1``).

    Returns:
        :class:`FluxEstimate`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-IHCP-009], [REF-HANSEN-2008], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    if criterion not in ("gcv", "discrepancy", "l_curve"):
        raise ValueError("criterion must be 'gcv', 'discrepancy' or 'l_curve'")
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    sigma = _positive("noise_std", noise_std)
    tau_value = _real("tau", tau)
    if tau_value < 1.0:
        raise ValueError("tau must be at least 1")
    dense, matrix = _dense_operator(operator)
    m, n = dense.shape
    y = as_vector(data, name="data")
    if len(y) != m:
        raise ValueError(f"data must have length {m}")
    penalty: LinearOperator
    if order == 0:
        penalty = IdentityOperator(n)
    else:
        if n < 2:
            raise ValueError("a first-order penalty needs at least two unknowns")
        penalty = FirstDifferenceOperator(n, boundary="none")
    problem = LinearInverseProblem.create(
        dense,
        y,
        noise=ScaledIdentityCovariance(m, sigma * sigma),
        penalty=penalty,
    )
    scale = math.sqrt(math.fsum(v * v for row in matrix for v in row)) / sigma
    if regularizations is None:
        grid = logarithmic_grid(1e-7 * scale, 10.0 * scale, 36)
    else:
        grid = tuple(_positive("regularization", v) for v in regularizations)
    path = regularization_path(problem, grid, noise_norm=math.sqrt(m), tau=tau_value)
    selection = select_regularization(path, criterion)
    method = f"tikhonov{order}"
    if selection.index is None or selection.estimate is None:
        return FluxEstimate(
            None, method, None, selection.status, criterion, None, None, None, path
        )
    rms = _rms_residual(matrix, selection.estimate, y)
    return FluxEstimate(
        selection.estimate,
        method,
        selection.regularization,
        selection.status,
        criterion,
        path.degrees_of_freedom[selection.index],
        rms,
        None,
        path,
    )


def _rms_residual(
    matrix: Matrix, estimate: Sequence[float], data: Sequence[float]
) -> float:
    """Return the root-mean-square residual ``||A q - y|| / sqrt(m)``.

    Args:
        matrix: Dense ``A``.
        estimate: ``q``.
        data: ``y``.

    Returns:
        The RMS residual (K).

    References:
        [EQ-INV-IHCP-009].

    """
    total = 0.0
    for row, y in zip(matrix, data, strict=True):
        total += (math.fsum(a * b for a, b in zip(row, estimate, strict=True)) - y) ** 2
    return math.sqrt(total / len(data))


def cgls_flux_estimate(
    operator: object,
    data: Sequence[float],
    noise_std: float,
    *,
    tau: float = 1.0,
    max_iterations: int | None = None,
) -> FluxEstimate:
    """CGLS with the discrepancy principle as stopping rule.

    Iterates CGLS from ``q = 0`` and stops at the first iteration with
    ``||A q_k - y|| <= tau sigma sqrt(m)`` [EQ-INV-IHCP-012]; the iteration
    number is the regularization parameter. Only operator actions are used.
    If the target is never met within ``max_iterations`` the estimate is the
    last iterate and ``status = "discrepancy_unmet"``.

    Args:
        operator: ``A`` (a :class:`SlabFluxOperator`, operator or matrix).
        data: Sensor temperatures.
        noise_std: Noise standard deviation ``sigma`` in K.
        tau: Safety factor (``>= 1``).
        max_iterations: Iteration cap (default ``2 n_q``).

    Returns:
        :class:`FluxEstimate` with ``method = "cgls"``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-IHCP-012], [REF-HANSEN-2008], [REF-HESTENES-STIEFEL-1952].

    """
    sigma = _positive("noise_std", noise_std)
    tau_value = _real("tau", tau)
    if tau_value < 1.0:
        raise ValueError("tau must be at least 1")
    a = require_operator(operator, name="operator")
    y = as_vector(data, name="data")
    if len(y) != a.shape[0]:
        raise ValueError(f"data must have length {a.shape[0]}")
    target = tau_value * sigma * math.sqrt(len(y))
    result = cgls(
        a,
        y,
        residual_tolerance=0.0,
        normal_tolerance=0.0,
        max_iterations=max_iterations,
        callback=lambda k, x, rnorm: rnorm <= target,
    )
    met = result.stop_reason == "callback"
    fitted = a.matvec(result.solution)
    rms = math.sqrt(
        math.fsum((f - v) ** 2 for f, v in zip(fitted, y, strict=True)) / len(y)
    )
    return FluxEstimate(
        result.solution,
        "cgls",
        None,
        "discrepancy" if met else "discrepancy_unmet",
        "discrepancy",
        None,
        rms,
        result.iterations,
        None,
    )


# ---------------------------------------------------------------------------
# Gaussian-process prior and Bayesian posterior
# ---------------------------------------------------------------------------


def matern_covariance(
    times: Sequence[float],
    *,
    amplitude: float,
    length_scale: float,
    smoothness: float = 0.5,
) -> Matrix:
    """Return the Matern covariance matrix of a stationary prior on ``q(t)``.

    ``k(r) = s^2 exp(-r / l)`` for ``smoothness = 1/2`` (Ornstein-Uhlenbeck),
    ``s^2 (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)`` for ``3/2`` and
    ``s^2 (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)`` for
    ``5/2``, with ``r = |t_i - t_j|`` [EQ-INV-IHCP-010]. The OU prior has
    continuous but non-differentiable paths and accommodates step-like
    fluxes better than the smoother kernels.

    Args:
        times: Interval midpoint times in s.
        amplitude: Prior standard deviation ``s`` in W/m^2 (``> 0``).
        length_scale: ``l > 0`` in s.
        smoothness: ``0.5``, ``1.5`` or ``2.5``.

    Returns:
        The symmetric ``n x n`` covariance matrix.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-IHCP-010], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    t = as_vector(times, name="times")
    if not t:
        raise ValueError("times must not be empty")
    s = _positive("amplitude", amplitude)
    ell = _positive("length_scale", length_scale)
    nu = _real("smoothness", smoothness)
    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("smoothness must be 0.5, 1.5 or 2.5")
    rows = []
    for a in t:
        row = []
        for b in t:
            r = abs(a - b) / ell
            if nu == 0.5:
                value = math.exp(-r)
            elif nu == 1.5:
                value = (1.0 + math.sqrt(3.0) * r) * math.exp(-math.sqrt(3.0) * r)
            else:
                value = (1.0 + math.sqrt(5.0) * r + 5.0 * r * r / 3.0) * math.exp(
                    -math.sqrt(5.0) * r
                )
            row.append(s * s * value)
        rows.append(tuple(row))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class FluxPosterior:
    """Gaussian posterior of the flux history with pointwise credible bands.

    Attributes:
        mean: Posterior mean (W/m^2).
        std: Posterior standard deviation per interval.
        lower: ``mean - z std`` with ``z`` the normal quantile of ``level``.
        upper: ``mean + z std``.
        level: Nominal two-sided probability of the band.
        amplitude: Prior amplitude ``s``.
        length_scale: Prior length scale ``l`` (s).
        smoothness: Matern smoothness ``nu``.
        log_evidence: ``log p(y)`` of the data under the prior.
        posterior: The underlying
            :class:`~probstat.inverse.linear_gaussian.GaussianLinearPosterior`.

    References:
        [EQ-INV-IHCP-010].

    """

    mean: Vector
    std: Vector
    lower: Vector
    upper: Vector
    level: float
    amplitude: float
    length_scale: float
    smoothness: float
    log_evidence: float
    posterior: GaussianLinearPosterior

    def coverage(self, truth: Sequence[float]) -> float:
        """Return the fraction of intervals whose band contains ``truth``.

        Args:
            truth: True fluxes (length ``n_q``).

        Returns:
            Pointwise coverage in ``[0, 1]``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-IHCP-010].

        """
        t = as_vector(truth, name="truth")
        if len(t) != len(self.mean):
            raise ValueError("truth has the wrong length")
        inside = sum(
            1
            for lo, v, hi in zip(self.lower, t, self.upper, strict=True)
            if lo <= v <= hi
        )
        return inside / len(t)


def bayesian_flux_posterior(
    operator: object,
    data: Sequence[float],
    noise_std: float,
    times: Sequence[float],
    *,
    amplitude: float,
    length_scale: float,
    smoothness: float = 0.5,
    prior_mean: float = 0.0,
    level: float = 0.95,
) -> FluxPosterior:
    """Exact linear-Gaussian posterior of ``q`` under a Matern prior.

    Prior ``q ~ N(m0 1, C)`` with ``C`` from :func:`matern_covariance`, noise
    ``N(0, sigma^2 I)``; the posterior mean, the marginal standard deviations
    and the evidence are those of
    :func:`~probstat.inverse.linear_gaussian.linear_gaussian_posterior`
    [EQ-INV-IHCP-010]. The band ``mean +- z std`` covers the true flux with
    probability ``level`` pointwise only if the prior and the noise model are
    right; model error and a mismatched prior reduce the coverage.

    Args:
        operator: ``A`` (a :class:`SlabFluxOperator`, operator or matrix).
        data: Sensor temperatures.
        noise_std: ``sigma`` in K.
        times: Interval midpoint times (length ``n_q``).
        amplitude: Prior standard deviation ``s`` in W/m^2.
        length_scale: Prior length scale in s.
        smoothness: ``0.5``, ``1.5`` or ``2.5``.
        prior_mean: Constant prior mean ``m0`` in W/m^2.
        level: Band probability in ``(0, 1)``.

    Returns:
        :class:`FluxPosterior`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or a data-space variance is
            negative from cancellation.

    References:
        [EQ-INV-IHCP-010], [REF-RASMUSSEN-WILLIAMS-2006],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    sigma = _positive("noise_std", noise_std)
    p = _real("level", level)
    if not 0.0 < p < 1.0:
        raise ValueError("level must lie in (0, 1)")
    m0 = _real("prior_mean", prior_mean)
    dense, _ = _dense_operator(operator)
    m, n = dense.shape
    y = as_vector(data, name="data")
    if len(y) != m:
        raise ValueError(f"data must have length {m}")
    t = as_vector(times, name="times")
    if len(t) != n:
        raise ValueError(f"times must have length {n}")
    cov = matern_covariance(
        t, amplitude=amplitude, length_scale=length_scale, smoothness=smoothness
    )
    posterior = linear_gaussian_posterior(
        dense,
        y,
        noise=ScaledIdentityCovariance(m, sigma * sigma),
        prior_mean=[m0] * n,
        prior_covariance=DenseCovariance(cov),
    )
    variances = posterior.marginal_variances()
    std = tuple(math.sqrt(v) for v in variances)
    z = _normal_quantile(0.5 * (1.0 + p))
    mean = posterior.mean
    return FluxPosterior(
        mean,
        std,
        tuple(a - z * s for a, s in zip(mean, std, strict=True)),
        tuple(a + z * s for a, s in zip(mean, std, strict=True)),
        p,
        float(amplitude),
        float(length_scale),
        float(smoothness),
        posterior.log_evidence,
        posterior,
    )


def select_gp_hyperparameters(
    operator: object,
    data: Sequence[float],
    noise_std: float,
    times: Sequence[float],
    *,
    amplitudes: Sequence[float],
    length_scales: Sequence[float],
    smoothness: float = 0.5,
    prior_mean: float = 0.0,
    level: float = 0.95,
) -> FluxPosterior:
    """Return the posterior with the grid hyperparameters of maximal evidence.

    Evaluates :func:`bayesian_flux_posterior` for every pair and keeps the one
    with the largest ``log p(y)`` (type-II maximum likelihood on a grid)
    [EQ-INV-IHCP-010]. Using the data twice (hyperparameters and posterior)
    makes the bands slightly optimistic.

    Args:
        operator: ``A``.
        data: Sensor temperatures.
        noise_std: ``sigma`` in K.
        times: Interval midpoint times.
        amplitudes: Candidate prior standard deviations (W/m^2).
        length_scales: Candidate length scales (s).
        smoothness: Matern smoothness.
        prior_mean: Constant prior mean.
        level: Band probability.

    Returns:
        The best :class:`FluxPosterior`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a candidate list is empty or a value is invalid.

    References:
        [EQ-INV-IHCP-010], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    amps = as_vector(amplitudes, name="amplitudes")
    scales = as_vector(length_scales, name="length_scales")
    if not amps or not scales:
        raise ValueError("amplitudes and length_scales must not be empty")
    dense, _ = _dense_operator(operator)
    best: FluxPosterior | None = None
    for s in amps:
        for ell in scales:
            candidate = bayesian_flux_posterior(
                dense,
                data,
                noise_std,
                times,
                amplitude=s,
                length_scale=ell,
                smoothness=smoothness,
                prior_mean=prior_mean,
                level=level,
            )
            if best is None or candidate.log_evidence > best.log_evidence:
                best = candidate
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def averaging_kernel(
    operator: object,
    noise_std: float,
    regularization: float,
    *,
    order: int = 0,
) -> Matrix:
    """Return the Tikhonov averaging-kernel (resolution) matrix.

    ``R = (A'A / sigma^2 + lambda^2 L'L)^-1 A'A / sigma^2``: the noise-free
    estimate for a true flux ``q`` is ``R q``, so row ``i`` shows how the
    estimate at interval ``i`` averages the true flux
    [EQ-INV-IHCP-009]. ``R = I`` is perfect resolution.

    Args:
        operator: ``A`` (a :class:`SlabFluxOperator`, operator or matrix).
        noise_std: ``sigma`` in K.
        regularization: ``lambda > 0``.
        order: Penalty order (``0`` or ``1``).

    Returns:
        The ``n x n`` matrix ``R``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or the normal matrix is not
            positive definite.

    References:
        [EQ-INV-IHCP-009], [REF-HANSEN-2008].

    """
    sigma = _positive("noise_std", noise_std)
    lam = _positive("regularization", regularization)
    if order not in (0, 1):
        raise ValueError("order must be 0 or 1")
    _, a = _dense_operator(operator)
    n = len(a[0])
    gram = [
        [
            math.fsum(a[r][i] * a[r][j] for r in range(len(a))) / sigma**2
            for j in range(n)
        ]
        for i in range(n)
    ]
    normal = [row[:] for row in gram]
    for i in range(n):
        if order == 0:
            normal[i][i] += lam * lam
        else:
            # L'L for the first difference: tridiagonal(-1, 2, -1) with 1 at the ends.
            normal[i][i] += lam * lam * (1.0 if i in (0, n - 1) else 2.0)
            if i + 1 < n:
                normal[i][i + 1] -= lam * lam
                normal[i + 1][i] -= lam * lam
    try:
        lower = cholesky(tuple(tuple(r) for r in normal))
    except ValueError as error:
        raise ValueError(
            "the regularized normal matrix is not positive definite"
        ) from error
    columns = [
        cholesky_solve(lower, tuple(gram[i][j] for i in range(n))) for j in range(n)
    ]
    return tuple(tuple(columns[j][i] for j in range(n)) for i in range(n))


@dataclass(frozen=True, slots=True)
class KernelResolution:
    """Time lag and width of an averaging kernel.

    Attributes:
        lag: Mean over the examined rows of the kernel centroid minus the row
            time (s); positive when the estimate at ``t`` reflects the flux at
            later times.
        width: Mean kernel standard deviation about its centroid (s).
        gain: Mean row sum of ``R`` (``1`` for an unbiased level; ``< 1``
            shrinkage toward zero).
        rows: Number of rows examined.

    References:
        [EQ-INV-IHCP-009].

    """

    lag: float
    width: float
    gain: float
    rows: int


def kernel_resolution(
    kernel: Matrix,
    interval_length: float,
    *,
    rows: Sequence[int] | None = None,
) -> KernelResolution:
    """Summarize the lag and width of an averaging kernel.

    For row ``i`` with ``w_j = R_ij`` the centroid is
    ``c_i = sum_j j w_j / sum_j w_j`` and the width
    ``sqrt(sum_j w_j (j - c_i)^2 / sum_j w_j)``, both in units of
    ``interval_length``; the summary averages over ``rows`` (by default the
    middle half of the record, away from the edge effects) [EQ-INV-IHCP-009].

    Args:
        kernel: Averaging kernel ``R`` (square).
        interval_length: ``Delta`` in s.
        rows: Row indices to average (default the middle half).

    Returns:
        :class:`KernelResolution`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the kernel is not square, a row index is invalid, or a
            row sums to zero.

    References:
        [EQ-INV-IHCP-009], [REF-HANSEN-2008].

    """
    n = len(kernel)
    if n == 0 or any(len(r) != n for r in kernel):
        raise ValueError("kernel must be a non-empty square matrix")
    delta = _positive("interval_length", interval_length)
    chosen = list(range(n // 4, n - n // 4)) if rows is None else list(rows)
    if not chosen or any(not 0 <= i < n for i in chosen):
        raise ValueError("rows must be valid, non-empty row indices")
    lags: list[float] = []
    widths: list[float] = []
    gains: list[float] = []
    for i in chosen:
        w = kernel[i]
        total = math.fsum(w)
        if total == 0.0:
            raise ValueError("a kernel row sums to zero")
        centroid = math.fsum(j * v for j, v in enumerate(w)) / total
        spread = math.fsum(v * (j - centroid) ** 2 for j, v in enumerate(w)) / total
        lags.append((centroid - i) * delta)
        widths.append(math.sqrt(max(spread, 0.0)) * delta)
        gains.append(total)
    count = len(chosen)
    return KernelResolution(
        math.fsum(lags) / count,
        math.fsum(widths) / count,
        math.fsum(gains) / count,
        count,
    )


@dataclass(frozen=True, slots=True)
class EqualNoiseResolution:
    """Resolution of the zeroth-order Tikhonov estimate at a fixed noise level.

    Attributes:
        regularization: The ``lambda`` whose estimator has the target noise.
        noise_std: Achieved RMS standard deviation of ``q_hat`` (W/m^2).
        resolution: :class:`KernelResolution` of the averaging kernel.

    References:
        [EQ-INV-IHCP-009], [EQ-INV-IHCP-011].

    """

    regularization: float
    noise_std: float
    resolution: KernelResolution


def resolution_at_noise_level(
    operator: object,
    noise_std: float,
    interval_length: float,
    target_std: float,
) -> EqualNoiseResolution:
    """Compare sensors at equal noise: the resolution that a noise level buys.

    For the zeroth-order Tikhonov estimator with ``Gram = A'A / sigma^2 =
    V diag(g) V'`` the averaging kernel is ``R = V diag(g / (g + lambda^2)) V'``
    and the covariance of the estimate has trace
    ``sum_i g_i / (g_i + lambda^2)^2``; the RMS standard deviation per interval
    ``sqrt(trace / n)`` (W/m^2) decreases monotonically in ``lambda``. This
    function finds by bisection in ``log lambda`` the ``lambda`` for which it
    equals ``target_std`` and returns the kernel's lag and width there. At a
    common noise level a deeper sensor needs more smoothing, so its width grows
    [EQ-INV-IHCP-009], [EQ-INV-IHCP-011].

    Args:
        operator: ``A`` (a :class:`SlabFluxOperator`, operator or matrix).
        noise_std: ``sigma`` in K.
        interval_length: ``Delta`` in s.
        target_std: Target RMS standard deviation of ``q_hat`` in W/m^2.

    Returns:
        :class:`EqualNoiseResolution`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the target is not attainable between
            ``lambda = 1e-12 sqrt(g_max)`` and ``1e6 sqrt(g_max)``.

    References:
        [EQ-INV-IHCP-009], [EQ-INV-IHCP-011], [REF-HANSEN-2008].

    """
    sigma = _positive("noise_std", noise_std)
    delta = _positive("interval_length", interval_length)
    target = _positive("target_std", target_std)
    _, a = _dense_operator(operator)
    n = len(a[0])
    gram = tuple(
        tuple(math.fsum(row[i] * row[j] for row in a) / sigma**2 for j in range(n))
        for i in range(n)
    )
    values, vectors = symmetric_eigen(gram)
    top = max(values)

    def deviation(lam: float) -> float:
        """Return the RMS standard deviation of the estimate at ``lambda``.

        Args:
            lam: ``lambda``.

        Returns:
            ``sqrt(sum g / (g + lambda^2)^2 / n)``.

        References:
            [EQ-INV-IHCP-009].

        """
        return math.sqrt(math.fsum(g / (g + lam * lam) ** 2 for g in values) / n)

    low, high = 1e-12 * math.sqrt(top), 1e6 * math.sqrt(top)
    if deviation(low) <= target or deviation(high) > target:
        raise ValueError("target_std is not attainable by any lambda in the bracket")
    for _ in range(80):
        middle = math.sqrt(low * high)
        if deviation(middle) > target:
            low = middle
        else:
            high = middle
    lam = math.sqrt(low * high)
    filters = [g / (g + lam * lam) for g in values]
    kernel = tuple(
        tuple(
            math.fsum(vectors[i][k] * filters[k] * vectors[j][k] for k in range(n))
            for j in range(n)
        )
        for i in range(n)
    )
    return EqualNoiseResolution(lam, deviation(lam), kernel_resolution(kernel, delta))


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    """Sensitivity-coefficient summary of one sensor depth.

    Attributes:
        depth: Sensor depth ``x`` (m).
        peak_coefficient: ``max_n phi(n)``, the largest response to a unit
            flux held for one interval (K per W/m^2).
        peak_time: Time ``n Delta`` of that maximum (s).
        step_at_end: ``Phi`` at the last reading, the response to a unit flux
            held from ``t = 0`` (K per W/m^2).
        theory_peak_time: ``x^2 / (2 alpha)`` of the continuous impulse
            response (s).

    References:
        [EQ-INV-IHCP-006], [EQ-INV-IHCP-011].

    """

    depth: float
    peak_coefficient: float
    peak_time: float
    step_at_end: float
    theory_peak_time: float


def sensitivity_summary(
    slab: SlabProperties,
    depths: Sequence[float],
    *,
    flux_intervals: int,
    interval_length: float,
    cells: int = 40,
    substeps: int = 4,
) -> tuple[SensitivityRow, ...]:
    """Tabulate the sensitivity coefficients against sensor depth.

    The peak coefficient falls roughly like ``exp(-x^2 / (4 alpha t))``-limited
    ``1 / sqrt(t)`` growth, i.e. quickly with depth, while the peak time grows
    like ``x^2 / (2 alpha)``: a deep sensor responds late and weakly, so the
    flux is both delayed and noise-limited [EQ-INV-IHCP-006],
    [EQ-INV-IHCP-011].

    Args:
        slab: Slab properties.
        depths: Sensor depths in m.
        flux_intervals: Number of flux intervals.
        interval_length: ``Delta`` in s.
        cells: Cells of the model.
        substeps: Steps per flux interval.

    Returns:
        One :class:`SensitivityRow` per depth.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    References:
        [EQ-INV-IHCP-006], [EQ-INV-IHCP-011], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    rows = []
    for depth in depths:
        op = SlabFluxOperator(
            slab,
            [depth],
            flux_intervals=flux_intervals,
            interval_length=interval_length,
            cells=cells,
            substeps=substeps,
        )
        phi = op.impulse_coefficients()[0]
        step = op.step_response()[0]
        best = max(range(len(phi)), key=phi.__getitem__)
        rows.append(
            SensitivityRow(
                float(depth),
                phi[best],
                (best + 1) * interval_length,
                step[-1],
                impulse_response_peak_time(depth, slab.diffusivity),
            )
        )
    return tuple(rows)
