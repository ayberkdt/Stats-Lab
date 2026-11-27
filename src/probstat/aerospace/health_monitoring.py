"""Hybrid physics-plus-statistics digital twin for liquid-engine health.

Experimental (added in 0.28).

A small discrete-time plant couples chamber-pressure physics (first-order lag
towards the steady value ``p_c = eta c*_ideal mdot / A_t``, with ``c*_ideal``
from :func:`probstat.aerospace.liquid_engine.characteristic_velocity`) with a
slowly degrading characteristic-velocity efficiency ``eta`` that follows a
random walk with a random-walk drift. Chamber pressure and mass flow are
measured with Gaussian noise. The statistical layer then does four jobs:

1. **Joint state-parameter estimation.** The state ``[p, m, eta, r]``
   (normalised pressure, normalised flow, efficiency, efficiency change per
   step) is estimated by the library's extended and unscented Kalman filters
   (:func:`estimate_health`). In the linear special case where the flow is
   frozen at its commanded value the model is linear-Gaussian and both
   filters must equal :func:`probstat.filtering.kalman_filter`
   (:func:`linear_special_case`, :func:`linear_as_nonlinear`).
   Consistency is judged by NEES/NIS Monte-Carlo tests
   (:func:`filter_consistency_study`).
2. **Fault detection on the normalised innovations.** A two-sided CUSUM
   (:func:`cusum_monitor`) with reference value ``k`` and threshold ``h`` and
   a generalised likelihood-ratio (GLR) test for a step change of the
   innovation mean (:func:`glr_mean_shift`). The in-control average run
   length of the CUSUM is given by Siegmund's approximation
   (:func:`cusum_arl_siegmund`) and, independently, by a numerical solution
   of the Brook-Evans integral equation (:func:`cusum_arl_numerical`).
   :func:`detection_operating_characteristic` measures false-alarm rate and
   detection delay against the threshold over predeclared seeds.
3. **Prognostics.** The efficiency is a Wiener process with drift; the time
   until it first reaches a failure threshold is inverse Gaussian
   (:func:`wiener_first_passage_cdf`). :func:`predict_rul` turns the filtered
   ``(eta, r)`` belief into a predictive remaining-useful-life distribution
   (:class:`RulDistribution`), and :func:`rul_coverage_study` checks its
   calibration against the simulated truth.
4. **Assessment.** :func:`assess_engine_health` runs the whole chain on one
   record.

All numbers in this module, its tests and its documentation are synthetic and
illustrative; none is data from a real engine. Units are SI (Pa, K, m^2,
kg/s, s); the filter works in non-dimensional coordinates, with pressure
divided by ``p_ref = c*_ideal mdot_nominal / A_t`` (the chamber pressure of a
perfect engine at nominal flow) and flow by ``mdot_nominal``, so that the
steady pressure is simply ``p~ = eta m~``.

Assumptions and validity.
    Perfect-gas choked-flow relation ``p_c = c* mdot / A_t`` at a fixed throat
    area; the chamber is a single first-order lag (no combustion delay, no
    acoustics); efficiency loss is the only fault physics; process and sensor
    noise are zero-mean, white and Gaussian with known standard deviations;
    the true degradation is a Wiener process with drift whose diffusion
    equals the filter's ``eta_process_std`` (a *matched* synthetic study). The
    EKF and UKF are Gaussian approximations; the CUSUM average-run-length
    formulas assume independent ``N(0, 1)`` observations, which the
    innovations of a consistent filter are only approximately. The
    inverse-Gaussian law is the continuous-monitoring first-passage law;
    discretely observed paths cross later, which
    :func:`predict_rul` corrects with a barrier shift of ``0.5826 sigma``
    (``DISCRETE_MONITORING_SHIFT``, a first-order Siegmund-type
    approximation).

Common pitfalls:
    - ``eta_process_std`` and the other ``*_std`` values are standard
      deviations per step, not variances.
    - Times returned by :class:`RulDistribution` are in seconds; the
      statistics of the filter are per step (``step_seconds``).
    - A CUSUM threshold calibrated on iid ``N(0, 1)`` data is not the
      threshold of the same detector on filter innovations; measure it with
      :func:`detection_operating_characteristic`.

References:
    [REF-SUTTON-BIBLARZ-2017], [REF-KALMAN-1960], [REF-JULIER-UHLMANN-2004],
    [REF-PAGE-1954], [REF-BROOK-EVANS-1972], [REF-SIEGMUND-1979],
    [REF-WILLSKY-JONES-1976], [REF-CHHIKARA-FOLKS-1989],
    [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.aerospace.liquid_engine import characteristic_velocity
from probstat.filtering import (
    GaussianState,
    KalmanFilterResult,
    LinearGaussianStateSpace,
    NeesResult,
    NonlinearStateSpace,
    UnscentedWeights,
    extended_kalman_filter,
    nees_test_from_filters,
    nis_test_from_filter,
    unscented_kalman_filter,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, lu_solve

__all__ = [
    "DISCRETE_MONITORING_SHIFT",
    "ConsistencyStudy",
    "CusumResult",
    "DetectionStudy",
    "EnginePhmSpec",
    "GlrResult",
    "HealthAssessment",
    "OperatingPoint",
    "PlantSimulation",
    "RulCalibration",
    "RulDistribution",
    "RulStudy",
    "StepFault",
    "assess_engine_health",
    "augmented_model",
    "cusum_arl_numerical",
    "cusum_arl_siegmund",
    "cusum_monitor",
    "detection_operating_characteristic",
    "estimate_health",
    "filter_consistency_study",
    "glr_false_alarm_bound",
    "glr_mean_shift",
    "inverse_gaussian_cdf",
    "linear_as_nonlinear",
    "linear_special_case",
    "normalized_innovations",
    "predict_rul",
    "rul_calibration",
    "rul_coverage_study",
    "sample_inverse_gaussian",
    "simulate_plant",
    "wiener_first_passage_cdf",
    "wiener_first_passage_quantile",
]

#: Barrier shift (in units of ``sigma sqrt(step)``) for discrete monitoring.
DISCRETE_MONITORING_SHIFT = 0.5826

_SIEGMUND_SHIFT = 1.166
_STATE_SIZE = 4


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive real.

    Args:
        value: Candidate number.
        name: Name used in error messages.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite and positive.

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
        ValueError: If it is negative or not finite.

    References:
        [REF-SUTTON-BIBLARZ-2017]. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _count(value: object, name: str, minimum: int = 1) -> int:
    """Return an integer not below ``minimum``.

    Args:
        value: Candidate integer.
        name: Name used in error messages.
        minimum: Smallest admissible value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an ``int`` (``bool`` is rejected).
        ValueError: If it is below ``minimum``.

    References:
        [REF-SUTTON-BIBLARZ-2017]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _series(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Return a non-empty tuple of finite reals.

    Args:
        values: Candidate sequence.
        name: Name used in error messages.

    Returns:
        The values as a tuple of floats.

    Raises:
        TypeError: If an entry is not a real number.
        ValueError: If the sequence is empty or an entry is not finite.

    References:
        [REF-PAGE-1954]. No equation is evaluated.

    """
    data = tuple(require_finite_real(v, name=name) for v in values)
    if not data:
        raise ValueError(f"{name} must be non-empty")
    return data


# --------------------------------------------------------------------------
# plant specification and simulation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnginePhmSpec:
    """Synthetic engine, noise, and prior for the health-monitoring twin.

    Pressure and flow are normalised: ``p~ = p / p_ref`` with
    ``p_ref = c*_ideal mdot_nominal / A_t`` and ``m~ = mdot / mdot_nominal``.
    All ``*_std`` values are per-step standard deviations in those
    non-dimensional coordinates (``rate_process_std`` and the efficiency
    quantities are absolute, since ``eta`` is already dimensionless).

    Args:
        gamma: Ratio of specific heats (dimensionless, ``> 1``).
        molecular_weight: Mean molecular weight, kg/kmol.
        chamber_temperature: Stagnation temperature, K.
        throat_area: Throat area ``A_t``, m^2.
        nominal_mass_flow: Commanded propellant flow, kg/s.
        step_seconds: Sample interval ``dt``, s.
        pressure_time_constant: Chamber lag time constant, s.
        flow_time_constant: Flow-system lag time constant, s.
        pressure_process_std: Process noise of normalised pressure.
        flow_process_std: Process noise of normalised flow.
        eta_process_std: Wiener diffusion of ``eta`` per step.
        rate_process_std: Random-walk std of the efficiency change per step.
        pressure_noise_std: Pressure-sensor noise (normalised).
        flow_noise_std: Flow-sensor noise (normalised).
        initial_eta: Prior mean of ``eta``.
        initial_rate: Prior mean of the efficiency change per step.
        initial_eta_std: Prior std of ``eta``.
        initial_rate_std: Prior std of the rate.
        initial_pressure_std: Prior std of normalised pressure and flow.

    Raises:
        TypeError: If a value is not a real number.
        ValueError: If a value is outside its domain.

    Equation:
        [EQ-AERO-PHM-001].

    References:
        [REF-SUTTON-BIBLARZ-2017], [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    gamma: float = 1.2
    molecular_weight: float = 22.0
    chamber_temperature: float = 3400.0
    throat_area: float = 0.01
    nominal_mass_flow: float = 20.0
    step_seconds: float = 1.0
    pressure_time_constant: float = 2.0
    flow_time_constant: float = 3.0
    pressure_process_std: float = 2e-3
    flow_process_std: float = 2e-3
    eta_process_std: float = 5e-4
    rate_process_std: float = 2e-6
    pressure_noise_std: float = 5e-3
    flow_noise_std: float = 5e-3
    initial_eta: float = 0.98
    initial_rate: float = -3e-4
    initial_eta_std: float = 5e-3
    initial_rate_std: float = 6e-5
    initial_pressure_std: float = 5e-3

    def __post_init__(self) -> None:
        """Validate the specification once.

        Raises:
            TypeError: If a value is not a real number.
            ValueError: If a value is outside its domain.

        References:
            [REF-SUTTON-BIBLARZ-2017]. No equation is evaluated.

        """
        for name in (
            "molecular_weight",
            "chamber_temperature",
            "throat_area",
            "nominal_mass_flow",
            "step_seconds",
            "pressure_time_constant",
            "flow_time_constant",
            "pressure_noise_std",
            "flow_noise_std",
            "eta_process_std",
            "initial_eta_std",
            "initial_rate_std",
            "initial_pressure_std",
        ):
            _positive(getattr(self, name), name)
        for name in ("pressure_process_std", "flow_process_std", "rate_process_std"):
            _non_negative(getattr(self, name), name)
        require_finite_real(self.initial_rate, name="initial_rate")
        eta = require_finite_real(self.initial_eta, name="initial_eta")
        if not 0.0 < eta <= 1.5:
            raise ValueError("initial_eta must be in (0, 1.5]")
        if require_finite_real(self.gamma, name="gamma") <= 1.0:
            raise ValueError("gamma must exceed 1")

    @property
    def ideal_characteristic_velocity(self) -> float:
        """Return the ideal ``c*`` (``eta = 1``) in m/s.

        Equation:
            [EQ-AERO-LRE-001].

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return characteristic_velocity(
            self.gamma, self.molecular_weight, self.chamber_temperature
        )

    @property
    def reference_pressure(self) -> float:
        """Return ``p_ref = c*_ideal mdot_nominal / A_t`` in Pa.

        Equation:
            [EQ-AERO-PHM-001].

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return (
            self.ideal_characteristic_velocity
            * self.nominal_mass_flow
            / self.throat_area
        )

    @property
    def pressure_coefficient(self) -> float:
        """Return the pressure-lag coefficient ``a = 1 - exp(-dt / tau_p)``.

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return -math.expm1(-self.step_seconds / self.pressure_time_constant)

    @property
    def flow_coefficient(self) -> float:
        """Return the flow-lag coefficient ``b = 1 - exp(-dt / tau_m)``.

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return -math.expm1(-self.step_seconds / self.flow_time_constant)

    @property
    def initial_mean(self) -> Vector:
        """Return the prior mean ``[eta0, 1, eta0, r0]``.

        References:
            [REF-KALMAN-1960].

        """
        eta = float(self.initial_eta)
        return (eta, 1.0, eta, float(self.initial_rate))

    @property
    def initial_covariance(self) -> Matrix:
        """Return the diagonal prior covariance.

        References:
            [REF-KALMAN-1960].

        """
        s = self.initial_pressure_std**2
        return _diagonal((s, s, self.initial_eta_std**2, self.initial_rate_std**2))

    @property
    def process_covariance(self) -> Matrix:
        """Return the diagonal process covariance ``Q``.

        References:
            [REF-KALMAN-1960].

        """
        return _diagonal(
            (
                self.pressure_process_std**2,
                self.flow_process_std**2,
                self.eta_process_std**2,
                self.rate_process_std**2,
            )
        )

    @property
    def observation_covariance(self) -> Matrix:
        """Return the diagonal sensor covariance ``R``.

        References:
            [REF-KALMAN-1960].

        """
        return _diagonal((self.pressure_noise_std**2, self.flow_noise_std**2))

    def pressure_pa(self, normalized_pressure: float) -> float:
        """Convert a normalised pressure to Pa.

        Args:
            normalized_pressure: ``p / p_ref``.

        Returns:
            Chamber pressure in Pa.

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return normalized_pressure * self.reference_pressure


def _diagonal(values: Sequence[float]) -> Matrix:
    """Return a diagonal matrix as a tuple of row tuples.

    Args:
        values: Diagonal entries.

    Returns:
        The square matrix.

    References:
        [REF-KALMAN-1960]. No equation is evaluated.

    """
    n = len(values)
    return tuple(
        tuple(float(values[i]) if i == j else 0.0 for j in range(n)) for i in range(n)
    )


def _transition(spec: EnginePhmSpec, x: Sequence[float]) -> Vector:
    """Return the noise-free one-step map of the augmented state.

    Args:
        spec: Plant specification.
        x: State ``[p, m, eta, r]``.

    Returns:
        ``[p + a (eta m - p), m + b (1 - m), eta + r, r]``.

    Equation:
        [EQ-AERO-PHM-001], [EQ-AERO-PHM-002].

    References:
        [REF-SUTTON-BIBLARZ-2017].

    """
    p, m, eta, r = x
    a = spec.pressure_coefficient
    b = spec.flow_coefficient
    return (p + a * (eta * m - p), m + b * (1.0 - m), eta + r, r)


@dataclass(frozen=True, slots=True)
class StepFault:
    """An abrupt efficiency step (injector or throat damage) in the truth.

    Args:
        index: One-based step ``k`` at which ``x_k`` receives the jump.
        size: Additive change of ``eta`` (negative for a loss).

    Raises:
        TypeError: If ``index`` is not an integer or ``size`` is not real.
        ValueError: If ``index < 1`` or ``size`` is not finite.

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    index: int
    size: float

    def __post_init__(self) -> None:
        """Validate the fault.

        Raises:
            TypeError: If a field has the wrong type.
            ValueError: If ``index < 1`` or ``size`` is not finite.

        References:
            [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008]. No equation is evaluated.

        """
        _count(self.index, "index", 1)
        require_finite_real(self.size, name="size")


@dataclass(frozen=True, slots=True)
class PlantSimulation:
    """One simulated record of the synthetic engine.

    Args:
        states: True states ``x_0..x_T`` as ``(p, m, eta, r)`` (normalised).
        observations: Noisy ``(pressure, flow)`` pairs ``y_1..y_T``.
        fault: The injected :class:`StepFault`, if any.

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    states: tuple[Vector, ...]
    observations: tuple[Vector, ...]
    fault: StepFault | None


def simulate_plant(
    spec: EnginePhmSpec,
    steps: int,
    *,
    seed: int | RandomStream | None = None,
    fault: StepFault | None = None,
) -> PlantSimulation:
    """Simulate the synthetic engine, drawing the initial truth from the prior.

    ``x_k = f(x_(k-1)) + w_k`` with the plant map of :func:`augmented_model`
    and ``w_k ~ N(0, Q)``; an optional :class:`StepFault` adds a jump to
    ``eta`` after the transition (the filter is not told). ``x_0`` is drawn
    from the filter prior, so the study is matched and a consistent filter
    must pass NEES tests.

    Args:
        spec: Plant specification.
        steps: Number of steps ``T`` (``>= 1``).
        seed: Integer seed or stream (children ``initial``, ``process``,
            ``measurement`` keep the noise sources separate, so a fault run
            and a fault-free run with one seed share all noise).
        fault: Optional abrupt efficiency step.

    Returns:
        True states and observations.

    Raises:
        TypeError: If ``spec`` or ``fault`` has the wrong type.
        ValueError: If ``steps < 1``.

    Equation:
        [EQ-AERO-PHM-001], [EQ-AERO-PHM-002].

    References:
        [REF-SUTTON-BIBLARZ-2017], [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    if not isinstance(spec, EnginePhmSpec):
        raise TypeError("spec must be an EnginePhmSpec")
    if fault is not None and not isinstance(fault, StepFault):
        raise TypeError("fault must be a StepFault or None")
    n = _count(steps, "steps")
    stream = resolve_stream(seed)
    g_init = stream.child("initial").generator().gauss
    g_proc = stream.child("process").generator().gauss
    g_meas = stream.child("measurement").generator().gauss
    init_cov = spec.initial_covariance
    proc_std = tuple(math.sqrt(spec.process_covariance[i][i]) for i in range(4))
    state = tuple(
        spec.initial_mean[i] + math.sqrt(init_cov[i][i]) * g_init(0.0, 1.0)
        for i in range(_STATE_SIZE)
    )
    states = [state]
    observations: list[Vector] = []
    for k in range(1, n + 1):
        mean = _transition(spec, state)
        noisy = tuple(mean[i] + proc_std[i] * g_proc(0.0, 1.0) for i in range(4))
        if fault is not None and k == fault.index:
            noisy = (noisy[0], noisy[1], noisy[2] + fault.size, noisy[3])
        state = noisy
        states.append(state)
        observations.append(
            (
                state[0] + spec.pressure_noise_std * g_meas(0.0, 1.0),
                state[1] + spec.flow_noise_std * g_meas(0.0, 1.0),
            )
        )
    return PlantSimulation(tuple(states), tuple(observations), fault)


# --------------------------------------------------------------------------
# filter models
# --------------------------------------------------------------------------


def augmented_model(spec: EnginePhmSpec) -> NonlinearStateSpace:
    """Return the augmented joint state-parameter model ``[p, m, eta, r]``.

    ``p' = p + a (eta m - p)``, ``m' = m + b (1 - m)``, ``eta' = eta + r``,
    ``r' = r``, each plus Gaussian noise; observations ``(p, m)``. The
    pressure equation is the discretised first-order chamber lag towards the
    choked-flow steady value ``eta m`` (normalised). Analytic Jacobians are
    supplied.

    Args:
        spec: Plant specification.

    Returns:
        A :class:`probstat.filtering.NonlinearStateSpace`.

    Raises:
        TypeError: If ``spec`` is not an :class:`EnginePhmSpec`.

    Equation:
        [EQ-AERO-PHM-001], [EQ-AERO-PHM-002].

    References:
        [REF-SUTTON-BIBLARZ-2017], [REF-JULIER-UHLMANN-2004].

    """
    if not isinstance(spec, EnginePhmSpec):
        raise TypeError("spec must be an EnginePhmSpec")
    a = spec.pressure_coefficient
    b = spec.flow_coefficient

    def transition(x: Vector) -> Vector:
        """Return the plant map ``f(x)``.

        Args:
            x: State.

        Returns:
            Next noise-free state.

        References:
            [REF-SUTTON-BIBLARZ-2017].

        """
        return _transition(spec, x)

    def jacobian(x: Vector) -> Matrix:
        """Return ``df/dx``.

        Args:
            x: State.

        Returns:
            The ``4 x 4`` Jacobian.

        References:
            [REF-JULIER-UHLMANN-2004].

        """
        _, m, eta, _ = x
        return (
            (1.0 - a, a * eta, a * m, 0.0),
            (0.0, 1.0 - b, 0.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        )

    def observe(x: Vector) -> Vector:
        """Return the sensor map ``h(x) = (p, m)``.

        Args:
            x: State.

        Returns:
            Noise-free sensor values.

        References:
            [REF-KALMAN-1960].

        """
        return (x[0], x[1])

    def observe_jacobian(x: Vector) -> Matrix:
        """Return ``dh/dx``.

        Args:
            x: State (unused; ``h`` is linear).

        Returns:
            The constant ``2 x 4`` Jacobian.

        References:
            [REF-KALMAN-1960].

        """
        del x
        return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))

    return NonlinearStateSpace(
        transition,
        observe,
        spec.process_covariance,
        spec.observation_covariance,
        spec.initial_mean,
        spec.initial_covariance,
        _STATE_SIZE,
        2,
        transition_jacobian_function=jacobian,
        observation_jacobian_function=observe_jacobian,
    )


def linear_special_case(spec: EnginePhmSpec) -> LinearGaussianStateSpace:
    """Return the linear-Gaussian model with the flow frozen at its command.

    With ``m = 1`` the state ``[p, eta, r]`` evolves linearly:
    ``p' = (1 - a) p + a eta``, ``eta' = eta + r``, ``r' = r``, and only the
    pressure is observed. Both nonlinear filters must reproduce
    :func:`probstat.filtering.kalman_filter` on it.

    Args:
        spec: Plant specification.

    Returns:
        The exact linear-Gaussian model.

    Raises:
        TypeError: If ``spec`` is not an :class:`EnginePhmSpec`.

    Equation:
        [EQ-AERO-PHM-002].

    References:
        [REF-KALMAN-1960], [REF-SUTTON-BIBLARZ-2017].

    """
    if not isinstance(spec, EnginePhmSpec):
        raise TypeError("spec must be an EnginePhmSpec")
    a = spec.pressure_coefficient
    q = spec.process_covariance
    s = spec.initial_pressure_std**2
    return LinearGaussianStateSpace(
        transition_matrix=((1.0 - a, a, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        process_covariance=_diagonal((q[0][0], q[2][2], q[3][3])),
        observation_matrix=((1.0, 0.0, 0.0),),
        observation_covariance=((spec.pressure_noise_std**2,),),
        initial_mean=(spec.initial_eta, spec.initial_eta, spec.initial_rate),
        initial_covariance=_diagonal(
            (s, spec.initial_eta_std**2, spec.initial_rate_std**2)
        ),
    )


def linear_as_nonlinear(model: LinearGaussianStateSpace) -> NonlinearStateSpace:
    """Wrap a linear-Gaussian model as a nonlinear model (same ``f``, ``h``).

    Used to run the EKF and UKF on a case with a known exact answer.

    Args:
        model: Linear-Gaussian model without control input.

    Returns:
        A :class:`probstat.filtering.NonlinearStateSpace` evaluating
        ``F x`` and ``H x`` with analytic Jacobians ``F`` and ``H``.

    Raises:
        TypeError: If ``model`` is not a ``LinearGaussianStateSpace``.
        ValueError: If the model has a control matrix.

    Equation:
        [EQ-AERO-PHM-002].

    References:
        [REF-KALMAN-1960], [REF-JULIER-UHLMANN-2004].

    """
    if not isinstance(model, LinearGaussianStateSpace):
        raise TypeError("model must be a LinearGaussianStateSpace")
    if model.control_matrix is not None:
        raise ValueError("a control matrix is not supported")
    f_mat = model.transition_matrix
    h_mat = model.observation_matrix

    def matvec(matrix: Matrix, x: Vector) -> Vector:
        """Return ``matrix x``.

        Args:
            matrix: Matrix.
            x: Vector.

        Returns:
            The product.

        References:
            [REF-KALMAN-1960].

        """
        return tuple(
            math.fsum(r * v for r, v in zip(row, x, strict=True)) for row in matrix
        )

    return NonlinearStateSpace(
        lambda x: matvec(f_mat, x),
        lambda x: matvec(h_mat, x),
        model.process_covariance,
        model.observation_covariance,
        model.initial_mean,
        model.initial_covariance,
        len(f_mat),
        len(h_mat),
        transition_jacobian_function=lambda x: f_mat,
        observation_jacobian_function=lambda x: h_mat,
    )


def estimate_health(
    spec: EnginePhmSpec,
    observations: Sequence[Sequence[float]],
    *,
    method: str = "ekf",
    weights: UnscentedWeights | None = None,
) -> KalmanFilterResult:
    """Estimate ``[p, m, eta, r]`` jointly with the EKF or the UKF.

    Args:
        spec: Plant specification (defines the filter model and prior).
        observations: ``(pressure, flow)`` pairs ``y_1..y_T`` (normalised).
        method: ``"ekf"`` or ``"ukf"``.
        weights: Unscented scaling for ``"ukf"``.

    Returns:
        The library filter record; ``filtered[k]`` holds the belief about
        ``[p, m, eta, r]`` after observation ``k``.

    Raises:
        TypeError: If ``spec`` has the wrong type.
        ValueError: If ``method`` is unknown or ``observations`` is empty.

    Equation:
        [EQ-AERO-PHM-002].

    References:
        [REF-KALMAN-1960], [REF-JULIER-UHLMANN-2004].

    """
    model = augmented_model(spec)
    data = tuple(tuple(float(v) for v in y) for y in observations)
    if method == "ekf":
        return extended_kalman_filter(model, data)
    if method == "ukf":
        return unscented_kalman_filter(model, data, weights=weights)
    raise ValueError("method must be 'ekf' or 'ukf'")


def normalized_innovations(
    result: KalmanFilterResult, *, channel: int = 0
) -> tuple[float, ...]:
    """Return the marginally standardised innovations of one sensor channel.

    ``z_k = nu_k[c] / sqrt(S_k[c][c])``. For a consistent filter ``z_k`` is
    ``N(0, 1)`` and white.

    Args:
        result: Filter record without missing observations.
        channel: Sensor index (0 pressure, 1 flow).

    Returns:
        The standardised innovation series.

    Raises:
        ValueError: If an observation is missing or ``channel`` is invalid.

    Equation:
        [EQ-AERO-PHM-003].

    References:
        [REF-KALMAN-1960], [REF-PAGE-1954].

    """
    c = _count(channel, "channel", 0)
    out: list[float] = []
    for nu, cov in zip(result.innovations, result.innovation_covariances, strict=True):
        if nu is None or cov is None:
            raise ValueError("missing observations are not supported")
        if c >= len(nu):
            raise ValueError("channel out of range")
        out.append(nu[c] / math.sqrt(cov[c][c]))
    return tuple(out)


# --------------------------------------------------------------------------
# CUSUM
# --------------------------------------------------------------------------


def _cusum_series(
    z: Sequence[float], reference: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the upper and lower CUSUM statistics (no resets on alarm).

    Args:
        z: Standardised observations.
        reference: Reference value ``k``.

    Returns:
        ``(S^+, S^-)`` with ``S^+_i = max(0, S^+_(i-1) + z_i - k)`` and
        ``S^-_i = max(0, S^-_(i-1) - z_i - k)``.

    Equation:
        [EQ-AERO-PHM-003].

    References:
        [REF-PAGE-1954].

    """
    up = 0.0
    lo = 0.0
    ups: list[float] = []
    los: list[float] = []
    for value in z:
        up = max(0.0, up + value - reference)
        lo = max(0.0, lo - value - reference)
        ups.append(up)
        los.append(lo)
    return tuple(ups), tuple(los)


@dataclass(frozen=True, slots=True)
class CusumResult:
    """Two-sided CUSUM monitor output.

    Args:
        upper: Upper statistic ``S^+`` per sample (detects an upward shift).
        lower: Lower statistic ``S^-`` per sample (detects a downward shift).
        reference: Reference value ``k``.
        threshold: Decision threshold ``h``.
        alarm_index: Zero-based index of the first sample with
            ``max(S^+, S^-) > h``, or ``None``.
        alarm_side: ``"upper"``, ``"lower"`` or ``None``.

    References:
        [REF-PAGE-1954].

    """

    upper: tuple[float, ...]
    lower: tuple[float, ...]
    reference: float
    threshold: float
    alarm_index: int | None
    alarm_side: str | None


def cusum_monitor(
    z: Sequence[float], *, reference: float = 0.5, threshold: float = 5.0
) -> CusumResult:
    """Run a two-sided CUSUM on standardised observations.

    Statistic: ``S^+_i = max(0, S^+_(i-1) + z_i - k)`` and
    ``S^-_i = max(0, S^-_(i-1) - z_i - k)`` with ``S_0 = 0``; alarm at the
    first ``i`` with ``max(S^+_i, S^-_i) > h``. ``k = delta / 2`` is optimal
    for a mean shift of ``delta`` standard deviations. In control and for
    iid ``N(0, 1)`` data the one-sided average run length is approximately
    ``(exp(2 k b) - 2 k b - 1) / (2 k^2)`` with ``b = h + 1.166``
    (:func:`cusum_arl_siegmund`); the two-sided run length obeys
    ``1/ARL = 1/ARL^+ + 1/ARL^-`` approximately.

    Args:
        z: Standardised innovations (mean 0, variance 1 in control).
        reference: Reference value ``k >= 0``.
        threshold: Threshold ``h > 0``.

    Returns:
        The statistics and the first alarm.

    Raises:
        TypeError: If an input is not real.
        ValueError: If ``z`` is empty, ``reference < 0`` or ``threshold <= 0``.

    Equation:
        [EQ-AERO-PHM-003], [EQ-AERO-PHM-004].

    References:
        [REF-PAGE-1954].

    """
    data = _series(z, "z")
    k = _non_negative(reference, "reference")
    h = _positive(threshold, "threshold")
    ups, los = _cusum_series(data, k)
    alarm: int | None = None
    side: str | None = None
    for i, (u, v) in enumerate(zip(ups, los, strict=True)):
        if u > h or v > h:
            alarm = i
            side = "upper" if u >= v else "lower"
            break
    return CusumResult(ups, los, k, h, alarm, side)


def cusum_arl_siegmund(
    reference: float,
    threshold: float,
    *,
    shift: float = 0.0,
    two_sided: bool = False,
) -> float:
    """Return Siegmund's approximate average run length of the CUSUM.

    For iid ``N(shift, 1)`` data and the one-sided upper CUSUM started at
    zero, with ``Delta = shift - k`` and ``b = h + 1.166``,
    ``ARL ~= (exp(-2 Delta b) - 1 + 2 Delta b) / (2 Delta^2)`` (limit
    ``b^2`` as ``Delta -> 0``). The constant ``1.166`` corrects the
    Brownian-motion result for the overshoot of the discrete walk. The
    two-sided value combines the sides as ``1/ARL = 1/ARL^+ + 1/ARL^-`` with
    the lower side seeing shift ``-shift``. Accurate to a few percent for
    ``h >= 3`` (checked against :func:`cusum_arl_numerical`); not exact.

    Args:
        reference: Reference value ``k >= 0``.
        threshold: Threshold ``h > 0``.
        shift: True mean of the observations in sigma units (0 in control).
        two_sided: Combine both sides.

    Returns:
        The approximate average run length in samples.

    Raises:
        TypeError: If an input is not real.
        ValueError: If an input is out of range or the result exceeds the
            floating-point range.

    Equation:
        [EQ-AERO-PHM-004].

    References:
        [REF-PAGE-1954], [REF-SIEGMUND-1979].

    """
    k = _non_negative(reference, "reference")
    h = _positive(threshold, "threshold")
    mu = require_finite_real(shift, name="shift")
    b = h + _SIEGMUND_SHIFT

    def one_sided(mean: float) -> float:
        """Return the one-sided approximation for mean ``mean``.

        Args:
            mean: Observation mean in sigma units.

        Returns:
            The one-sided average run length.

        Raises:
            ValueError: If the result overflows.

        References:
            [REF-PAGE-1954].

        """
        delta = mean - k
        if abs(delta) < 1e-9:
            return b * b
        exponent = -2.0 * delta * b
        if exponent > 700.0:
            raise ValueError("average run length exceeds the floating-point range")
        return (math.exp(exponent) - 1.0 + 2.0 * delta * b) / (2.0 * delta * delta)

    if not two_sided:
        return one_sided(mu)
    return 1.0 / (1.0 / one_sided(mu) + 1.0 / one_sided(-mu))


def cusum_arl_numerical(
    reference: float, threshold: float, *, shift: float = 0.0, cells: int = 120
) -> float:
    """Return the one-sided CUSUM average run length from its integral equation.

    The run length ``L(u)`` from statistic value ``u`` obeys
    ``L(u) = 1 + L(0) Phi(k - u - mu) + int_0^h L(v) phi(v - u + k - mu) dv``
    for iid ``N(mu, 1)`` data (Brook-Evans type). It is solved by the
    midpoint (Nystrom) rule on ``cells`` cells plus the atom at ``0``; the
    error is ``O((h / cells)^2)``.

    Args:
        reference: Reference value ``k >= 0``.
        threshold: Threshold ``h > 0``.
        shift: Observation mean ``mu`` in sigma units.
        cells: Number of quadrature cells (``>= 10``).

    Returns:
        ``L(0)``, the average run length from a zero start.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If an input is out of range or the system is singular.

    Equation:
        [EQ-AERO-PHM-004].

    References:
        [REF-PAGE-1954], [REF-BROOK-EVANS-1972].

    """
    k = _non_negative(reference, "reference")
    h = _positive(threshold, "threshold")
    mu = require_finite_real(shift, name="shift")
    n = _count(cells, "cells", 10)
    width = h / n
    nodes = [0.0] + [(j + 0.5) * width for j in range(n)]
    root = math.sqrt(2.0 * math.pi)
    matrix: list[list[float]] = []
    for i, u in enumerate(nodes):
        row = [0.0] * (n + 1)
        row[0] = 0.5 * math.erfc(-(k - u - mu) / math.sqrt(2.0))
        for j in range(n):
            d = nodes[j + 1] - u + k - mu
            row[j + 1] = width * math.exp(-0.5 * d * d) / root
        matrix.append([(1.0 if i == c else 0.0) - row[c] for c in range(n + 1)])
    solution = lu_solve(
        tuple(tuple(r) for r in matrix), tuple(1.0 for _ in range(n + 1))
    )
    return float(solution[0])


# --------------------------------------------------------------------------
# GLR
# --------------------------------------------------------------------------


def _glr_series(z: Sequence[float], window: int) -> tuple[float, ...]:
    """Return the windowed GLR statistic at every sample.

    Args:
        z: Standardised observations.
        window: Maximal look-back length ``M``.

    Returns:
        ``g_n = max_(1<=L<=min(M,n)) (z_(n-L+1) + ... + z_n)^2 / (2 L)``.

    Equation:
        [EQ-AERO-PHM-005].

    References:
        [REF-PAGE-1954].

    """
    prefix = [0.0]
    for value in z:
        prefix.append(prefix[-1] + value)
    out: list[float] = []
    for n in range(1, len(prefix)):
        best = 0.0
        for length in range(1, min(window, n) + 1):
            total = prefix[n] - prefix[n - length]
            best = max(best, total * total / (2.0 * length))
        out.append(best)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class GlrResult:
    """Windowed GLR step-change test output.

    Args:
        statistic: ``g_n`` at every sample.
        threshold: Decision threshold ``h``.
        window: Look-back window ``M``.
        alarm_index: Zero-based index of the first sample with ``g_n > h``.
        change_index: Estimated zero-based index of the first shifted sample
            at the alarm (``None`` without alarm).
        magnitude: Estimated mean shift in sigma units at the alarm.

    References:
        [REF-PAGE-1954].

    """

    statistic: tuple[float, ...]
    threshold: float
    window: int
    alarm_index: int | None
    change_index: int | None
    magnitude: float | None


def glr_mean_shift(
    z: Sequence[float], *, threshold: float, window: int = 40
) -> GlrResult:
    """Run the windowed GLR test for a step change of the mean.

    For unit-variance data the log-likelihood ratio of a mean ``theta`` that
    starts at sample ``n - L + 1`` against ``theta = 0`` is maximised over
    ``theta`` by the sample mean of the last ``L`` samples, giving
    ``g_n = max_L (sum of last L)^2 / (2 L)``. Under the no-change hypothesis
    each term is ``chi2(1) / 2``, so a single pair ``(n, L)`` exceeds ``h``
    with probability ``erfc(sqrt(h))``; a union bound over ``M`` window
    lengths and ``T`` samples gives :func:`glr_false_alarm_bound`.

    Args:
        z: Standardised innovations.
        threshold: Threshold ``h > 0``.
        window: Maximal look-back ``M >= 1``.

    Returns:
        The statistic series, the first alarm, and the maximum-likelihood
        change time and size at that alarm.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If ``z`` is empty, ``threshold <= 0`` or ``window < 1``.

    Equation:
        [EQ-AERO-PHM-005].

    References:
        [REF-PAGE-1954], [REF-WILLSKY-JONES-1976].

    """
    data = _series(z, "z")
    h = _positive(threshold, "threshold")
    m = _count(window, "window")
    stat = _glr_series(data, m)
    alarm = next((i for i, g in enumerate(stat) if g > h), None)
    if alarm is None:
        return GlrResult(stat, h, m, None, None, None)
    n = alarm + 1
    prefix = [0.0]
    for value in data[:n]:
        prefix.append(prefix[-1] + value)
    best_length = 1
    best = -1.0
    for length in range(1, min(m, n) + 1):
        total = prefix[n] - prefix[n - length]
        value = total * total / (2.0 * length)
        if value > best:
            best = value
            best_length = length
    shift = (prefix[n] - prefix[n - best_length]) / best_length
    return GlrResult(stat, h, m, alarm, n - best_length, shift)


def glr_false_alarm_bound(threshold: float, window: int, steps: int) -> float:
    """Return the union bound on a false GLR alarm within ``steps`` samples.

    ``min(1, steps * window * erfc(sqrt(h)))``; conservative because the
    ``(n, L)`` pairs are strongly dependent.

    Args:
        threshold: Threshold ``h > 0``.
        window: Look-back window ``M``.
        steps: Number of samples ``T``.

    Returns:
        An upper bound on the probability of at least one false alarm.

    Raises:
        TypeError: If an input has the wrong type.
        ValueError: If an input is out of range.

    Equation:
        [EQ-AERO-PHM-005].

    References:
        [REF-PAGE-1954], [REF-WILLSKY-JONES-1976].

    """
    h = _positive(threshold, "threshold")
    m = _count(window, "window")
    t = _count(steps, "steps")
    return min(1.0, t * m * math.erfc(math.sqrt(h)))


# --------------------------------------------------------------------------
# inverse Gaussian / Wiener first passage
# --------------------------------------------------------------------------


def _log_normal_cdf(x: float) -> float:
    """Return ``log Phi(x)`` accurately in the lower tail.

    Args:
        x: Argument.

    Returns:
        The log of the standard normal cdf (asymptotic series below -20).

    References:
        [REF-PAGE-1954]. No equation is evaluated.

    """
    if x > -20.0:
        return math.log(0.5 * math.erfc(-x / math.sqrt(2.0)))
    u = 1.0 / (x * x)
    series = 1.0 - u + 3.0 * u * u - 15.0 * u**3 + 105.0 * u**4
    return (
        -0.5 * x * x - math.log(-x) - 0.5 * math.log(2.0 * math.pi) + math.log(series)
    )


def inverse_gaussian_cdf(t: float, mean: float, shape: float) -> float:
    """Return the cdf of the inverse Gaussian law ``IG(mean, shape)``.

    ``F(t) = Phi(sqrt(lam/t) (t/mu - 1)) + exp(2 lam/mu) Phi(-sqrt(lam/t)
    (t/mu + 1))``, with the second term evaluated in log space.

    Args:
        t: Time (``<= 0`` gives 0).
        mean: Mean ``mu > 0``.
        shape: Shape ``lambda > 0``.

    Returns:
        ``P(T <= t)``.

    Raises:
        TypeError: If an input is not real.
        ValueError: If ``mean`` or ``shape`` is not positive.

    Equation:
        [EQ-AERO-PHM-006].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008], [REF-CHHIKARA-FOLKS-1989].

    """
    mu = _positive(mean, "mean")
    lam = _positive(shape, "shape")
    time = require_finite_real(t, name="t") if not math.isinf(t) else t
    if time <= 0.0:
        return 0.0
    if math.isinf(time):
        return 1.0
    s = math.sqrt(lam / time)
    first = 0.5 * math.erfc(-s * (time / mu - 1.0) / math.sqrt(2.0))
    second = math.exp(
        min(700.0, 2.0 * lam / mu + _log_normal_cdf(-s * (time / mu + 1.0)))
    )
    return min(1.0, first + second)


def wiener_first_passage_cdf(
    t: float, distance: float, drift: float, volatility: float
) -> float:
    """Return ``P(T <= t)`` for the first passage of a Wiener process.

    A process ``X_t = x_0 - drift t + volatility W_t`` started ``distance``
    above an absorbing barrier first reaches it at ``T ~ IG(distance/drift,
    distance^2/volatility^2)``.

    Args:
        t: Time.
        distance: ``x_0 - barrier > 0``.
        drift: Rate of decrease ``> 0`` (per unit time).
        volatility: Diffusion ``sigma > 0`` (per square-root time).

    Returns:
        The first-passage cdf at ``t``.

    Raises:
        TypeError: If an input is not real.
        ValueError: If ``distance``, ``drift`` or ``volatility`` is not
            positive.

    Equation:
        [EQ-AERO-PHM-006].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008], [REF-CHHIKARA-FOLKS-1989].

    """
    d = _positive(distance, "distance")
    mu = _positive(drift, "drift")
    s = _positive(volatility, "volatility")
    return inverse_gaussian_cdf(t, d / mu, d * d / (s * s))


def wiener_first_passage_quantile(
    probability: float, distance: float, drift: float, volatility: float
) -> float:
    """Return the quantile of the Wiener first-passage time (bisection).

    Args:
        probability: Level in ``(0, 1)``.
        distance: ``x_0 - barrier > 0``.
        drift: Rate of decrease ``> 0``.
        volatility: Diffusion ``> 0``.

    Returns:
        The time ``t`` with ``P(T <= t) = probability``.

    Raises:
        TypeError: If an input is not real.
        ValueError: If an input is out of range.

    Equation:
        [EQ-AERO-PHM-006].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    p = require_finite_real(probability, name="probability")
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    d = _positive(distance, "distance")
    mu = _positive(drift, "drift")
    s = _positive(volatility, "volatility")
    return _ig_quantile(p, d / mu, d * d / (s * s))


def _ig_quantile(p: float, mean: float, shape: float) -> float:
    """Return the inverse Gaussian quantile by bracketing and bisection.

    Args:
        p: Probability in ``(0, 1)``.
        mean: Mean.
        shape: Shape.

    Returns:
        The quantile.

    Raises:
        ValueError: If the bracket does not close.

    Equation:
        [EQ-AERO-PHM-006].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    lo = 0.0
    hi = mean
    for _ in range(2000):
        if inverse_gaussian_cdf(hi, mean, shape) >= p:
            break
        lo = hi
        hi *= 2.0
    else:
        raise ValueError("quantile bracket failed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if inverse_gaussian_cdf(mid, mean, shape) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * hi:
            break
    return 0.5 * (lo + hi)


def sample_inverse_gaussian(
    generator: random.Random, mean: float, shape: float
) -> float:
    """Draw one inverse Gaussian variate (Michael-Schucany-Haas transform).

    Args:
        generator: Random generator from a :class:`RandomStream`.
        mean: Mean ``> 0``.
        shape: Shape ``> 0``.

    Returns:
        A sample of ``IG(mean, shape)``.

    Raises:
        TypeError: If ``generator`` is not a ``random.Random``.
        ValueError: If ``mean`` or ``shape`` is not positive.

    Equation:
        [EQ-AERO-PHM-006].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    if not isinstance(generator, random.Random):
        raise TypeError("generator must be a random.Random")
    mu = _positive(mean, "mean")
    lam = _positive(shape, "shape")
    nu = generator.gauss(0.0, 1.0)
    y = nu * nu
    x = mu + mu * mu * y / (2.0 * lam)
    x -= mu / (2.0 * lam) * math.sqrt(4.0 * mu * lam * y + mu * mu * y * y)
    if generator.random() <= mu / (mu + x):
        return x
    return mu * mu / x


# --------------------------------------------------------------------------
# remaining useful life
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RulDistribution:
    """Predictive distribution of the remaining useful life.

    The law is a mixture over the filtered ``(eta, r)`` belief of
    inverse-Gaussian first-passage times. Draws whose rate is not negative
    never fail (``non_degrading``); draws already below the threshold have a
    zero life (``already_failed``).

    Args:
        failure_threshold: Failure level of ``eta``.
        step_seconds: Seconds per step.
        volatility: Wiener diffusion per step.
        components: ``(mean, shape)`` in steps of each degrading draw (equal
            weights ``1 / draws``).
        draws: Number of belief draws.
        already_failed: Fraction of draws at or below the threshold.
        non_degrading: Fraction of draws with a non-negative rate.
        discrete_correction: Whether the barrier was shifted for sampling.
        eta_mean: Filtered mean of ``eta``.
        rate_mean: Filtered mean of the rate ``r`` (per step).

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    failure_threshold: float
    step_seconds: float
    volatility: float
    components: tuple[tuple[float, float], ...]
    draws: int
    already_failed: float
    non_degrading: float
    discrete_correction: bool
    eta_mean: float
    rate_mean: float

    @property
    def plug_in_mean_seconds(self) -> float:
        """Return the point-estimate life ``(eta - threshold) / |r| dt``.

        Returns:
            ``inf`` if the filtered rate is not negative.

        Equation:
            [EQ-AERO-PHM-006].

        References:
            [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

        """
        if self.rate_mean >= 0.0:
            return math.inf
        distance = self.eta_mean - self.failure_threshold
        return max(0.0, distance / -self.rate_mean * self.step_seconds)

    def cdf(self, seconds: float) -> float:
        """Return ``P(RUL <= seconds)`` (improper if some draws never fail).

        Args:
            seconds: Time in seconds.

        Returns:
            The predictive cdf.

        Raises:
            TypeError: If ``seconds`` is not real.

        Equation:
            [EQ-AERO-PHM-007].

        References:
            [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

        """
        t = require_finite_real(seconds, name="seconds")
        if t < 0.0:
            return 0.0
        steps = t / self.step_seconds
        total = math.fsum(
            inverse_gaussian_cdf(steps, mu, lam) for mu, lam in self.components
        )
        return self.already_failed + total / self.draws

    def quantile(self, probability: float) -> float:
        """Return the predictive quantile in seconds.

        Args:
            probability: Level in ``(0, 1)``.

        Returns:
            The quantile; ``0`` below the already-failed mass and ``inf``
            when the level exceeds the mass of eventually failing draws.

        Raises:
            TypeError: If ``probability`` is not real.
            ValueError: If it is outside ``(0, 1)``.

        Equation:
            [EQ-AERO-PHM-007].

        References:
            [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 < p < 1.0:
            raise ValueError("probability must be in (0, 1)")
        if p <= self.already_failed:
            return 0.0
        if p >= 1.0 - self.non_degrading:
            return math.inf
        lo = 0.0
        hi = max((mu for mu, _ in self.components), default=1.0)
        for _ in range(2000):
            if self.cdf(hi * self.step_seconds) >= p:
                break
            lo = hi
            hi *= 2.0
        else:
            raise ValueError("quantile bracket failed")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid * self.step_seconds) < p:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-10 * hi:
                break
        return 0.5 * (lo + hi) * self.step_seconds

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        """Return the central predictive interval in seconds.

        Args:
            level: Coverage in ``(0, 1)``.

        Returns:
            ``(quantile((1-level)/2), quantile((1+level)/2))``.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.

        Equation:
            [EQ-AERO-PHM-007].

        References:
            [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

        """
        lv = require_finite_real(level, name="level")
        if not 0.0 < lv < 1.0:
            raise ValueError("level must be in (0, 1)")
        return self.quantile((1.0 - lv) / 2.0), self.quantile((1.0 + lv) / 2.0)


def predict_rul(
    spec: EnginePhmSpec,
    state: GaussianState,
    *,
    failure_threshold: float,
    draws: int = 300,
    seed: int | RandomStream | None = 0,
    discrete_correction: bool = True,
) -> RulDistribution:
    """Predict the remaining useful life from a filtered ``(eta, r)`` belief.

    Draws ``(eta_0, r)`` from the Gaussian belief (components 2 and 3 of the
    filtered state, with their covariance). Given a draw with ``r < 0`` and
    ``eta_0`` above the threshold, the future efficiency is a Wiener process
    with drift ``|r|`` per step and diffusion ``sigma = eta_process_std``, so
    the first-passage time in steps is ``IG(d / |r|, d^2 / sigma^2)`` with
    ``d = eta_0 - threshold`` (plus ``0.5826 sigma`` when
    ``discrete_correction`` compensates for monitoring on the step grid).
    The predictive cdf is the equal-weight mixture of these exact cdfs.

    Args:
        spec: Plant specification (``eta_process_std``, ``step_seconds``).
        state: Filtered belief about ``[p, m, eta, r]``.
        failure_threshold: ``eta`` level that defines failure.
        draws: Number of belief draws ``>= 1``.
        seed: Seed or stream for the belief draws.
        discrete_correction: Shift the barrier for discrete monitoring.

    Returns:
        The predictive :class:`RulDistribution`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the state size is not 4 or an input is out of range.

    Equation:
        [EQ-AERO-PHM-006], [EQ-AERO-PHM-007].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008], [REF-KALMAN-1960].

    """
    if not isinstance(spec, EnginePhmSpec):
        raise TypeError("spec must be an EnginePhmSpec")
    if not isinstance(state, GaussianState):
        raise TypeError("state must be a GaussianState")
    if len(state.mean) != _STATE_SIZE:
        raise ValueError("state must have four components")
    threshold = require_finite_real(failure_threshold, name="failure_threshold")
    n = _count(draws, "draws")
    gen = resolve_stream(seed).generator()
    m_e, m_r = state.mean[2], state.mean[3]
    p_ee = state.covariance[2][2]
    p_er = state.covariance[2][3]
    p_rr = state.covariance[3][3]
    if p_ee <= 0.0:
        raise ValueError("filtered eta variance must be positive")
    l11 = math.sqrt(p_ee)
    l21 = p_er / l11
    l22 = math.sqrt(max(p_rr - l21 * l21, 0.0))
    sigma = spec.eta_process_std
    shift = DISCRETE_MONITORING_SHIFT * sigma if discrete_correction else 0.0
    components: list[tuple[float, float]] = []
    failed = 0
    flat = 0
    for _ in range(n):
        z1 = gen.gauss(0.0, 1.0)
        z2 = gen.gauss(0.0, 1.0)
        eta0 = m_e + l11 * z1
        rate = m_r + l21 * z1 + l22 * z2
        if eta0 <= threshold:
            failed += 1
        elif rate >= 0.0:
            flat += 1
        else:
            d = eta0 - threshold + shift
            components.append((d / -rate, d * d / (sigma * sigma)))
    return RulDistribution(
        threshold,
        spec.step_seconds,
        sigma,
        tuple(components),
        n,
        failed / n,
        flat / n,
        discrete_correction,
        m_e,
        m_r,
    )


@dataclass(frozen=True, slots=True)
class RulCalibration:
    """Coverage of central predictive intervals by the true life.

    Args:
        levels: Nominal central-interval levels.
        coverage: Observed fraction of probability-integral-transform values
            inside each central interval.
        band: Three-sigma binomial half-width ``3 sqrt(l (1 - l) / n)``.
        pit_mean: Mean PIT (``0.5`` if calibrated).
        pit_variance: PIT variance (``1/12`` if calibrated).
        runs: Number of PIT values.

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    levels: tuple[float, ...]
    coverage: tuple[float, ...]
    band: tuple[float, ...]
    pit_mean: float
    pit_variance: float
    runs: int

    @property
    def within_band(self) -> bool:
        """Return whether every coverage is within its three-sigma band.

        References:
            [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

        """
        return all(
            abs(c - lv) <= b
            for c, lv, b in zip(self.coverage, self.levels, self.band, strict=True)
        )


def rul_calibration(
    pit_values: Sequence[float], levels: Sequence[float] = (0.5, 0.8, 0.9)
) -> RulCalibration:
    """Summarise the calibration of predictive RUL distributions.

    If the predictive cdf is calibrated, ``PIT = F(true RUL)`` is uniform, so
    the central interval of level ``l`` covers the truth with probability
    ``l``.

    Args:
        pit_values: PIT values in ``[0, 1]``.
        levels: Nominal levels in ``(0, 1)``.

    Returns:
        Observed coverages with binomial bands.

    Raises:
        ValueError: If there is no PIT value, a PIT is outside ``[0, 1]`` or
            a level is outside ``(0, 1)``.

    Equation:
        [EQ-AERO-PHM-007].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    pits = _series(pit_values, "pit_values")
    if any(not 0.0 <= v <= 1.0 for v in pits):
        raise ValueError("PIT values must lie in [0, 1]")
    lv = tuple(require_finite_real(x, name="level") for x in levels)
    if not lv or any(not 0.0 < x < 1.0 for x in lv):
        raise ValueError("levels must be non-empty and in (0, 1)")
    n = len(pits)
    cover = tuple(
        sum(1 for v in pits if (1.0 - x) / 2.0 <= v <= (1.0 + x) / 2.0) / n for x in lv
    )
    band = tuple(3.0 * math.sqrt(x * (1.0 - x) / n) for x in lv)
    mean = math.fsum(pits) / n
    var = math.fsum((v - mean) ** 2 for v in pits) / n
    return RulCalibration(lv, cover, band, mean, var, n)


# --------------------------------------------------------------------------
# studies
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsistencyStudy:
    """Monte-Carlo consistency of the joint state-parameter filter.

    Args:
        nees: Chi-square NEES test over the runs.
        nis_consistent_fraction: Fraction of runs whose time-average NIS test
            returned ``"consistent"`` (about the confidence level if the
            filter is consistent).
        mean_nis: Mean of the per-run average NIS (about ``2`` for two
            sensors).
        runs: Number of runs.
        steps: Steps per run.
        method: ``"ekf"`` or ``"ukf"``.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """

    nees: NeesResult
    nis_consistent_fraction: float
    mean_nis: float
    runs: int
    steps: int
    method: str


def filter_consistency_study(
    spec: EnginePhmSpec,
    *,
    runs: int,
    steps: int,
    method: str = "ekf",
    seed: int = 0,
    confidence: float = 0.95,
) -> ConsistencyStudy:
    """Measure NEES and NIS consistency over matched Monte-Carlo runs.

    Run ``i`` uses the stream ``child("consistency", i)`` of ``seed``; truth
    is drawn from the filter prior, so a consistent filter has average NEES
    inside the chi-square band (library NEES/NIS tests).

    Args:
        spec: Plant specification.
        runs: Number of independent runs ``>= 2``.
        steps: Steps per run.
        method: ``"ekf"`` or ``"ukf"``.
        seed: Master seed.
        confidence: Two-sided acceptance probability.

    Returns:
        NEES result and NIS summary.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a count is too small or ``method`` is unknown.

    Equation:
        [EQ-AERO-PHM-002].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001], [REF-JULIER-UHLMANN-2004].

    """
    n_runs = _count(runs, "runs", 2)
    n_steps = _count(steps, "steps")
    master = resolve_stream(seed)
    results: list[KalmanFilterResult] = []
    truths: list[tuple[Vector, ...]] = []
    for i in range(n_runs):
        sim = simulate_plant(spec, n_steps, seed=master.child("consistency", i))
        results.append(estimate_health(spec, sim.observations, method=method))
        truths.append(sim.states)
    nees = nees_test_from_filters(results, truths, confidence=confidence)
    nis = [nis_test_from_filter(r, confidence=confidence) for r in results]
    good = sum(1 for t in nis if t.verdict == "consistent") / n_runs
    return ConsistencyStudy(
        nees,
        good,
        math.fsum(t.statistic for t in nis) / n_runs,
        n_runs,
        n_steps,
        method,
    )


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Detector performance at one threshold.

    Args:
        threshold: Decision threshold.
        false_alarm_runs: Fault-free runs with at least one alarm.
        exposure_steps: Total fault-free samples observed.
        arl0_estimate: ``exposure / alarms`` (``inf`` without alarms); each
            run is stopped at its first alarm, so this is the exponential
            maximum-likelihood estimate of the in-control run length.
        detection_probability: Fraction of fault runs alarmed at or after
            the fault with no earlier alarm.
        early_alarms: Fault runs alarmed before the fault.
        mean_delay_steps: Mean detection delay in steps (``None`` if none).
        delays: Detection delays of the detected runs, in steps.

    References:
        [REF-PAGE-1954].

    """

    threshold: float
    false_alarm_runs: int
    exposure_steps: int
    arl0_estimate: float
    detection_probability: float
    early_alarms: int
    mean_delay_steps: float | None
    delays: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DetectionStudy:
    """Operating characteristic of a fault detector over predeclared seeds.

    Args:
        detector: ``"cusum"`` or ``"glr"``.
        fault_size: Injected step in ``eta``.
        fault_index: One-based step of the fault.
        runs: Runs per class (fault-free and faulty).
        steps: Samples per run.
        points: One :class:`OperatingPoint` per threshold.

    References:
        [REF-PAGE-1954].

    """

    detector: str
    fault_size: float
    fault_index: int
    runs: int
    steps: int
    points: tuple[OperatingPoint, ...]


def detection_operating_characteristic(
    spec: EnginePhmSpec,
    *,
    thresholds: Sequence[float],
    fault_size: float,
    fault_index: int = 60,
    runs: int = 20,
    steps: int = 160,
    detector: str = "cusum",
    reference: float = 0.5,
    window: int = 40,
    seed: int = 0,
) -> DetectionStudy:
    """Measure false-alarm rate and detection delay against the threshold.

    Run ``i`` uses stream ``child("detect", i)`` of ``seed`` for a fault-free
    record and for the faulty record (same noise, so the two differ only by
    the fault: common random numbers). The EKF innovations of the pressure
    channel are standardised and passed to the detector.

    Args:
        spec: Plant specification.
        thresholds: Thresholds ``h`` to evaluate.
        fault_size: Step injected into ``eta`` at ``fault_index``.
        fault_index: One-based fault step (``< steps``).
        runs: Number of seeds ``>= 1``.
        steps: Samples per run.
        detector: ``"cusum"`` or ``"glr"``.
        reference: CUSUM reference value ``k``.
        window: GLR window.
        seed: Master seed (predeclare it).

    Returns:
        The operating characteristic.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``detector`` is unknown, ``fault_index >= steps`` or
            a count is invalid.

    Equation:
        [EQ-AERO-PHM-003], [EQ-AERO-PHM-005].

    References:
        [REF-PAGE-1954], [REF-KALMAN-1960].

    """
    ths = tuple(_positive(h, "threshold") for h in thresholds)
    if not ths:
        raise ValueError("thresholds must be non-empty")
    if detector not in ("cusum", "glr"):
        raise ValueError("detector must be 'cusum' or 'glr'")
    n_runs = _count(runs, "runs")
    n_steps = _count(steps, "steps", 2)
    k0 = _count(fault_index, "fault_index")
    if k0 >= n_steps:
        raise ValueError("fault_index must be below steps")
    size = require_finite_real(fault_size, name="fault_size")
    ref = _non_negative(reference, "reference")
    win = _count(window, "window")
    master = resolve_stream(seed)

    def statistic(z: tuple[float, ...]) -> tuple[float, ...]:
        """Return the detector statistic series.

        Args:
            z: Standardised innovations.

        Returns:
            The running decision statistic.

        References:
            [REF-PAGE-1954].

        """
        if detector == "glr":
            return _glr_series(z, win)
        ups, los = _cusum_series(z, ref)
        return tuple(max(u, v) for u, v in zip(ups, los, strict=True))

    null_series: list[tuple[float, ...]] = []
    fault_series: list[tuple[float, ...]] = []
    for i in range(n_runs):
        stream = master.child("detect", i)
        for fault, store in ((None, null_series), (StepFault(k0, size), fault_series)):
            sim = simulate_plant(spec, n_steps, seed=stream, fault=fault)
            result = estimate_health(spec, sim.observations)
            store.append(statistic(normalized_innovations(result)))

    def first(series: tuple[float, ...], h: float) -> int | None:
        """Return the first index above ``h``.

        Args:
            series: Statistic series.
            h: Threshold.

        Returns:
            Zero-based index or ``None``.

        References:
            [REF-PAGE-1954].

        """
        return next((j for j, v in enumerate(series) if v > h), None)

    points: list[OperatingPoint] = []
    for h in ths:
        alarms = 0
        exposure = 0
        for series in null_series:
            idx = first(series, h)
            if idx is None:
                exposure += len(series)
            else:
                alarms += 1
                exposure += idx + 1
        early = 0
        delays: list[int] = []
        for series in fault_series:
            idx = first(series, h)
            if idx is None:
                continue
            if idx < k0 - 1:
                early += 1
            else:
                delays.append(idx - (k0 - 1))
        points.append(
            OperatingPoint(
                h,
                alarms,
                exposure,
                exposure / alarms if alarms else math.inf,
                len(delays) / n_runs,
                early,
                math.fsum(delays) / len(delays) if delays else None,
                tuple(delays),
            )
        )
    return DetectionStudy(detector, size, k0, n_runs, n_steps, tuple(points))


@dataclass(frozen=True, slots=True)
class RulStudy:
    """Calibration of predictive RUL distributions over predeclared seeds.

    Args:
        calibration: Coverage summary of the PIT values.
        pits: PIT of the true life under each predictive cdf.
        true_rul_steps: Realised remaining life of each scored run, steps.
        median_rul_seconds: Predictive median of each scored run, seconds.
        censored: Runs whose truth did not fail within the horizon.
        already_failed: Runs whose truth was below the threshold when
            observation stopped.
        runs: Runs attempted.

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    calibration: RulCalibration
    pits: tuple[float, ...]
    true_rul_steps: tuple[int, ...]
    median_rul_seconds: tuple[float, ...]
    censored: int
    already_failed: int
    runs: int


def rul_coverage_study(
    spec: EnginePhmSpec,
    *,
    failure_threshold: float,
    observe_steps: int,
    horizon_steps: int,
    runs: int,
    seed: int = 0,
    draws: int = 200,
    discrete_correction: bool = True,
    levels: Sequence[float] = (0.5, 0.8, 0.9),
) -> RulStudy:
    """Score the predictive RUL distribution against the simulated truth.

    For each run a record of ``observe_steps + horizon_steps`` steps is
    simulated, the EKF sees only the first ``observe_steps`` observations,
    and the true remaining life is the first later step at which the true
    ``eta`` is at or below the threshold. The PIT evaluates the predictive
    cdf at ``(n - 1/2) dt`` for a crossing at step ``n`` after the last
    observation (the mid-point of the step in which it occurred). Runs
    without a crossing in the horizon are counted as censored and excluded;
    keep the horizon long enough that this is rare.

    Args:
        spec: Plant specification.
        failure_threshold: ``eta`` level that defines failure.
        observe_steps: Number of observed steps.
        horizon_steps: Extra simulated steps to find the true crossing.
        runs: Number of seeds.
        seed: Master seed (predeclare it).
        draws: Belief draws per predictive distribution.
        discrete_correction: Shift the barrier for discrete monitoring.
        levels: Central-interval levels for the coverage summary.

    Returns:
        PIT values and the coverage summary.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a count is invalid or no run could be scored.

    Equation:
        [EQ-AERO-PHM-007].

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    thr = require_finite_real(failure_threshold, name="failure_threshold")
    n_obs = _count(observe_steps, "observe_steps")
    n_hor = _count(horizon_steps, "horizon_steps")
    n_runs = _count(runs, "runs")
    master = resolve_stream(seed)
    pits: list[float] = []
    truths: list[int] = []
    medians: list[float] = []
    censored = 0
    failed = 0
    for i in range(n_runs):
        stream = master.child("rul", i)
        sim = simulate_plant(spec, n_obs + n_hor, seed=stream)
        if sim.states[n_obs][2] <= thr:
            failed += 1
            continue
        crossing = next(
            (k for k in range(n_obs + 1, n_obs + n_hor + 1) if sim.states[k][2] <= thr),
            None,
        )
        if crossing is None:
            censored += 1
            continue
        result = estimate_health(spec, sim.observations[:n_obs])
        rul = predict_rul(
            spec,
            result.filtered[n_obs],
            failure_threshold=thr,
            draws=draws,
            seed=stream.child("belief"),
            discrete_correction=discrete_correction,
        )
        steps_left = crossing - n_obs
        pits.append(rul.cdf((steps_left - 0.5) * spec.step_seconds))
        truths.append(steps_left)
        medians.append(rul.quantile(0.5))
    if not pits:
        raise ValueError("no run could be scored")
    return RulStudy(
        rul_calibration(pits, levels),
        tuple(pits),
        tuple(truths),
        tuple(medians),
        censored,
        failed,
        n_runs,
    )


@dataclass(frozen=True, slots=True)
class HealthAssessment:
    """End-to-end assessment of one engine record.

    Args:
        filter: EKF record of the joint estimator.
        innovations: Standardised pressure innovations.
        cusum: Two-sided CUSUM output.
        glr: GLR step-change output.
        rul: Predictive remaining useful life at the last observation.

    References:
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """

    filter: KalmanFilterResult
    innovations: tuple[float, ...]
    cusum: CusumResult
    glr: GlrResult
    rul: RulDistribution


def assess_engine_health(
    spec: EnginePhmSpec,
    observations: Sequence[Sequence[float]],
    *,
    failure_threshold: float,
    cusum_reference: float = 0.5,
    cusum_threshold: float = 6.0,
    glr_threshold: float = 8.0,
    glr_window: int = 40,
    draws: int = 300,
    seed: int = 0,
) -> HealthAssessment:
    """Run estimation, detection and prognosis on one observation record.

    Args:
        spec: Plant specification.
        observations: ``(pressure, flow)`` pairs (normalised).
        failure_threshold: ``eta`` failure level.
        cusum_reference: CUSUM reference ``k``.
        cusum_threshold: CUSUM threshold ``h``.
        glr_threshold: GLR threshold.
        glr_window: GLR window.
        draws: Belief draws for the RUL law.
        seed: Seed for the RUL draws.

    Returns:
        Filter record, detector outputs and the RUL distribution.

    Raises:
        TypeError: If ``spec`` has the wrong type.
        ValueError: If the record is empty or a setting is out of range.

    Equation:
        [EQ-AERO-PHM-002], [EQ-AERO-PHM-003], [EQ-AERO-PHM-005],
        [EQ-AERO-PHM-007].

    References:
        [REF-KALMAN-1960], [REF-PAGE-1954],
        [REF-SAXENA-GOEBEL-SIMON-EKLUND-2008].

    """
    result = estimate_health(spec, observations)
    z = normalized_innovations(result)
    return HealthAssessment(
        result,
        z,
        cusum_monitor(z, reference=cusum_reference, threshold=cusum_threshold),
        glr_mean_shift(z, threshold=glr_threshold, window=glr_window),
        predict_rul(
            spec,
            result.filtered[-1],
            failure_threshold=failure_threshold,
            draws=draws,
            seed=seed,
        ),
    )
