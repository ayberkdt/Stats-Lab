"""Single-axis gyro and star-tracker attitude estimation and a pointing budget.

A spacecraft axis carries a rate gyro and a star tracker. The gyro reading is
the true rate plus a slowly drifting bias plus white *angle random walk*; the
star tracker measures the attitude with white noise. A two-state Kalman filter
``[attitude error, gyro bias]`` blends them. This module states that model with
its exact discretisation, derives and implements a closed form of the
steady-state attitude covariance, checks it against the Riccati solution,
recovers the gyro noise parameters from an overlapping Allan deviation and
assembles a pointing error budget on the library's uncertainty-budget tools.

Contents:

- :class:`GyroStarTrackerModel` and :func:`attitude_process_noise`: the model
  and its exact discretised process noise ``Q``;
  :func:`attitude_state_space` builds the
  :class:`~probstat.filtering.LinearGaussianStateSpace` of the filter.
- :func:`steady_state_attitude_closed_form`: the analytic steady state, derived
  in the docstring; :func:`steady_state_attitude_riccati` the numerical
  reference (a doubling iteration, or the library's Riccati recursion).
- :func:`simulate_attitude_error`, :func:`simulate_gyro_rate`: synthetic
  truths, star-tracker measurements and gyro readings.
- :func:`overlapping_allan_deviation`, :func:`fit_allan_noise`: the Allan
  estimator and the two-component noise fit.
- :func:`pointing_budget`: knowledge, control jitter, thermal-elastic and
  star-tracker alignment errors as a
  :class:`~probstat.uq.UncertaintyBudget` with correlations, compared with a
  Monte Carlo propagation and with the probability of meeting a requirement;
  :func:`knowledge_sobol_ranking` and :func:`grade_requirement_probability`
  rank and propagate the sensor grades.

Model (SI units: angles rad, times s, rates rad/s). The attitude ``theta`` obeys
``theta' = gyro - b - eta_v`` where the gyro reading is ``omega + b + eta_v``,
``eta_v`` is white with spectral density ``sigma_v^2`` (units rad^2/s, that is
``sigma_v`` in rad/sqrt(s)) and the bias ``b`` is a random walk,
``b' = eta_u`` with spectral density ``sigma_u^2`` (rad^2/s^3, ``sigma_u`` in
rad/s/sqrt(s)). Because the known part of the rate cancels in the error state,
the filter works on the error ``x = [theta, b]``. The star tracker returns
``y_k = theta_k + v_k`` every ``dt`` with ``v_k`` white of standard deviation
``sigma_n`` (rad, per sample).

Assumptions and validity. One axis, small angles (linear kinematics), a
constant bias random walk, white noises, time-invariant parameters, a star
tracker that never drops out and has no latency, and independent noises. The
closed form is exact for that model only: it is no statement about a vehicle
with cross-axis coupling, a bias with correlated (Gauss-Markov) drift, a
temperature-dependent scale factor or a non-white star-tracker noise. The
Allan fit assumes just the two components (white rate noise and rate random
walk); flicker (bias instability), quantisation or a rate ramp are not modelled
and show up as a poor fit rather than as a parameter. Every case in the tests
is synthetic; no number here describes a real sensor or spacecraft.

Experimental (added in 0.28).

References:
    [REF-KALMAN-1960], [REF-FARRENKOPF-1978], [REF-JULIER-UHLMANN-2004].
    Farrenkopf's paper was not read: the steady state implemented here is
    derived in :func:`steady_state_attitude_closed_form` and verified
    numerically, not quoted from it.

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.distributions import Normal
from probstat.filtering import (
    LinearGaussianStateSpace,
    steady_state_kalman,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector, cholesky, cholesky_semidefinite
from probstat.sensitivity import estimate_sobol_indices
from probstat.uq import BudgetEntry, UncertaintyBudget, monte_carlo_budget

__all__ = [
    "ARCSEC",
    "POINTING_SOURCES",
    "AllanFit",
    "AllanResult",
    "AttitudeSimulation",
    "GradeRequirementResult",
    "GradeSobolResult",
    "GyroStarTrackerModel",
    "PointingBudgetResult",
    "SensorGrades",
    "SteadyStateAttitude",
    "allan_model_variance",
    "attitude_process_noise",
    "attitude_state_space",
    "fit_allan_noise",
    "grade_requirement_probability",
    "knowledge_sobol_ranking",
    "overlapping_allan_deviation",
    "pointing_budget",
    "simulate_attitude_error",
    "simulate_gyro_rate",
    "steady_state_attitude_closed_form",
    "steady_state_attitude_riccati",
]

ARCSEC = math.pi / 648000.0
"""One arcsecond in radians (``pi / (180 * 3600)``)."""

POINTING_SOURCES: tuple[str, str, str, str] = (
    "attitude_knowledge",
    "control_jitter",
    "thermal_elastic",
    "tracker_alignment",
)
"""Names of the four pointing error sources, in budget order."""


def _positive(value: object, name: str) -> float:
    """Return a finite strictly positive real.

    Args:
        value: Candidate number.
        name: Name used in error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is not finite and positive.

    References:
        [REF-KALMAN-1960]. Validation only.

    """
    result = require_finite_real(value, name=name)
    if not result > 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value: object, name: str) -> float:
    """Return a finite non-negative real.

    Args:
        value: Candidate number.
        name: Name used in error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If it is negative or not finite.

    References:
        [REF-KALMAN-1960]. Validation only.

    """
    result = require_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _count(value: object, name: str, minimum: int) -> int:
    """Return an integer at least ``minimum``.

    Args:
        value: Candidate integer (booleans rejected).
        name: Name used in error messages.
        minimum: Smallest accepted value.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is below ``minimum``.

    References:
        [REF-KALMAN-1960]. Validation only.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


# --------------------------------------------------------------------------
# Model, exact discretisation and the filter's state space.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GyroStarTrackerModel:
    """Single-axis gyro plus star-tracker model (SI units).

    Args:
        sigma_v: Angle random walk density of the gyro, rad/sqrt(s); the gyro
            rate noise has spectral density ``sigma_v**2``. Non-negative.
        sigma_u: Rate random walk density of the bias, rad/s/sqrt(s);
            non-negative. ``sigma_v`` and ``sigma_u`` are not both zero.
        sigma_n: Star-tracker attitude noise standard deviation per sample,
            rad. Positive.
        dt: Sample interval, s. Positive.

    Raises:
        TypeError: If a field is not a real number.
        ValueError: If a field is out of range or the process noise is zero.

    Equation:
        [EQ-AERO-ATT-001].

    References:
        [REF-KALMAN-1960], [REF-FARRENKOPF-1978].

    """

    sigma_v: float
    sigma_u: float
    sigma_n: float
    dt: float

    def __post_init__(self) -> None:
        """Validate the four parameters.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If a field is out of range or ``sigma_v = sigma_u = 0``.

        References:
            [EQ-AERO-ATT-001]. Validation only.

        """
        sv = _non_negative(self.sigma_v, "sigma_v")
        su = _non_negative(self.sigma_u, "sigma_u")
        if sv == 0.0 and su == 0.0:
            raise ValueError("sigma_v and sigma_u must not both be zero")
        object.__setattr__(self, "sigma_v", sv)
        object.__setattr__(self, "sigma_u", su)
        object.__setattr__(self, "sigma_n", _positive(self.sigma_n, "sigma_n"))
        object.__setattr__(self, "dt", _positive(self.dt, "dt"))


def attitude_process_noise(model: GyroStarTrackerModel) -> Matrix:
    """Return the exact discretised process noise of the two-state model.

    With ``F(s) = [[1, -s], [0, 1]]`` and the continuous noise ``w = [-eta_v,
    eta_u]`` of density ``diag(sigma_v^2, sigma_u^2)``,
    ``Q = integral_0^dt F(s) Qc F(s)' ds``, which gives

    ``Q11 = sigma_v^2 dt + sigma_u^2 dt^3 / 3``,
    ``Q12 = -sigma_u^2 dt^2 / 2``, ``Q22 = sigma_u^2 dt``.

    Args:
        model: The gyro and star-tracker model.

    Returns:
        The symmetric ``2 x 2`` matrix ``Q`` in rad^2 and its cross units.

    Raises:
        TypeError: If ``model`` is not a :class:`GyroStarTrackerModel`.

    Equation:
        [EQ-AERO-ATT-001].

    References:
        [REF-KALMAN-1960], [EQ-AERO-ATT-001].

    """
    if not isinstance(model, GyroStarTrackerModel):
        raise TypeError("model must be a GyroStarTrackerModel")
    sv2 = model.sigma_v**2
    su2 = model.sigma_u**2
    dt = model.dt
    q11 = sv2 * dt + su2 * dt**3 / 3.0
    q12 = -su2 * dt**2 / 2.0
    q22 = su2 * dt
    return ((q11, q12), (q12, q22))


def attitude_state_space(
    model: GyroStarTrackerModel,
    *,
    initial_std: tuple[float, float] = (1.0e-3, 1.0e-5),
) -> LinearGaussianStateSpace:
    """Return the filter model ``x_k = F x_(k-1) + w_k``, ``y_k = H x_k + v_k``.

    ``F = [[1, -dt], [0, 1]]``, ``H = [1, 0]``, ``Q`` from
    :func:`attitude_process_noise`, ``R = sigma_n^2``, prior mean zero and
    prior covariance ``diag(initial_std^2)``.

    Args:
        model: The gyro and star-tracker model.
        initial_std: Prior standard deviations of attitude (rad) and bias
            (rad/s).

    Returns:
        The linear-Gaussian state-space model for
        :func:`probstat.filtering.kalman_filter`.

    Raises:
        TypeError: If ``model`` has the wrong type or a value is not real.
        ValueError: If an initial standard deviation is not positive.

    Equation:
        [EQ-AERO-ATT-001].

    References:
        [REF-KALMAN-1960], [EQ-AERO-ATT-001].

    """
    q = attitude_process_noise(model)
    s0 = _positive(initial_std[0], "initial_std[0]")
    s1 = _positive(initial_std[1], "initial_std[1]")
    return LinearGaussianStateSpace(
        transition_matrix=((1.0, -model.dt), (0.0, 1.0)),
        process_covariance=q,
        observation_matrix=((1.0, 0.0),),
        observation_covariance=((model.sigma_n**2,),),
        initial_mean=(0.0, 0.0),
        initial_covariance=((s0 * s0, 0.0), (0.0, s1 * s1)),
    )


# --------------------------------------------------------------------------
# Steady state.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SteadyStateAttitude:
    """Steady-state covariances of the gyro and star-tracker filter.

    Args:
        predicted_covariance: ``P^-``, covariance before the star-tracker
            update (``2 x 2``, attitude then bias).
        filtered_covariance: ``P``, covariance after the update.
        gain: Kalman gain ``K = P^- H' / (H P^- H' + R)`` (two entries).
        attitude_std_predicted: ``sqrt(P^-_11)``, rad.
        attitude_std_filtered: ``sqrt(P_11)``, rad: the attitude knowledge
            error after the update.
        bias_std_filtered: ``sqrt(P_22)``, rad/s.
        method: ``"closed_form"``, ``"doubling"`` or ``"iteration"``.
        iterations: Iterations used (0 for the closed form).
        converged: Whether the numerical method met its tolerance (always true
            for the closed form).

    Equation:
        [EQ-AERO-ATT-002], [EQ-AERO-ATT-003].

    References:
        [REF-KALMAN-1960], [REF-FARRENKOPF-1978].

    """

    predicted_covariance: Matrix
    filtered_covariance: Matrix
    gain: Vector
    attitude_std_predicted: float
    attitude_std_filtered: float
    bias_std_filtered: float
    method: str
    iterations: int
    converged: bool


def _from_predicted(
    predicted: Matrix, sigma_n: float, method: str, iterations: int, converged: bool
) -> SteadyStateAttitude:
    """Build the steady-state record from a predicted covariance.

    Args:
        predicted: ``P^-`` (``2 x 2`` symmetric).
        sigma_n: Star-tracker standard deviation.
        method: Label of the method that produced ``predicted``.
        iterations: Iterations used.
        converged: Convergence flag.

    Returns:
        The record with ``P = P^- - P^- H' S^-1 H P^-`` and the gain.

    References:
        [EQ-AERO-ATT-003].

    """
    r = sigma_n**2
    p11, p12, p22 = predicted[0][0], predicted[0][1], predicted[1][1]
    s = p11 + r
    a = p11 * r / s
    c = p12 * r / s
    d = p22 - p12 * p12 / s
    return SteadyStateAttitude(
        predicted_covariance=((p11, p12), (p12, p22)),
        filtered_covariance=((a, c), (c, d)),
        gain=(p11 / s, p12 / s),
        attitude_std_predicted=math.sqrt(p11),
        attitude_std_filtered=math.sqrt(a),
        bias_std_filtered=math.sqrt(max(d, 0.0)),
        method=method,
        iterations=iterations,
        converged=converged,
    )


def steady_state_attitude_closed_form(
    model: GyroStarTrackerModel,
) -> SteadyStateAttitude:
    """Return the analytic steady-state covariance of the two-state filter.

    Derivation (all steps are algebra on the Riccati fixed point; the result is
    verified numerically, not quoted). Write ``q = sigma_u^2 dt``,
    ``R = sigma_n^2``, ``Q11, Q12, Q22`` as in :func:`attitude_process_noise`,
    ``P^- = [[p11, p12], [p12, p22]]`` and ``S = p11 + R``. The update gives
    ``P = [[p11 R / S, p12 R / S], [p12 R / S, p22 - p12^2 / S]]`` and the
    prediction ``P^- = F P F' + Q`` with ``F = [[1, -dt], [0, 1]]`` reads

    ``p22 = p22 - p12^2 / S + Q22``,
    ``p12 = p12 R / S - dt (p22 - p12^2 / S) + Q12``,
    ``p11 = p11 R / S - 2 dt p12 R / S + dt^2 (p22 - p12^2 / S) + Q11``.

    The first line gives ``p12^2 = q S``; the posterior bias variance is
    non-negative and ``Q12 < 0``, so the second line forces ``p12 < 0``,
    ``p12 = -sqrt(q S)``. Eliminating ``p22 - p12^2/S`` between the second and
    third lines leaves, with ``c0 = Q11 + dt Q12 = sigma_v^2 dt -
    sigma_u^2 dt^3 / 6`` and ``s = sqrt(S)``,

    ``s^4 - dt sqrt(q) s^3 - (2 R + c0) s^2 - dt sqrt(q) R s + R^2 = 0``.

    With ``t = s / sqrt(R)`` and ``beta = dt sqrt(q / R) = sigma_u dt^(3/2) /
    sigma_n`` this is the reciprocal quartic ``t^4 - beta t^3 - (2 + c0/R)
    t^2 - beta t + 1 = 0``. The substitution ``z = t + 1/t`` turns it into the
    quadratic ``z^2 - beta z - (4 + c0/R) = 0``. Its larger root is
    ``z = 2 + w`` with

    ``w = (beta - 4 + sqrt(beta^2 / 3 + 16 + 4 sigma_v^2 dt / R)) / 2``

    (evaluated as ``2 g / (4 - beta + sqrt(.))`` for ``beta < 4``, with
    ``g = beta (2 - beta / 6) + sigma_v^2 dt / R``, to avoid cancellation), and
    ``t = (z + sqrt(z^2 - 4)) / 2`` is the root with ``t >= 1`` (``S >= R``).
    Then, with ``sqrt(z^2 - 4) = sqrt(w (w + 4))``:

    ``p11 = R t sqrt(w (w + 4))`` (predicted attitude variance),
    ``p12 = -sqrt(q R) t``,
    ``P_11 = p11 / t^2`` (filtered attitude variance),
    ``P_12 = p12 / t^2``,
    ``P_22 = sqrt(q) p11 / (sqrt(R) t dt) - q / 2`` and ``p22 = P_22 + q``.

    The reciprocal quartic has a single root with ``t > 1`` for the larger
    ``z`` root, which is the stabilising solution. Limits: for ``sigma_u = 0``
    the bias is eventually known (``P_22 = 0``) and ``p11 ~ sigma_n sigma_v
    sqrt(dt)`` for ``sigma_v^2 dt << R``, the Kalman-Bucy value of a pure
    integrator. The relation to Farrenkopf's published result was not
    checked.

    Args:
        model: The gyro and star-tracker model.

    Returns:
        The :class:`SteadyStateAttitude` with ``method="closed_form"``.
        ``attitude_std_filtered`` is the post-update (knowledge) standard
        deviation; ``attitude_std_predicted`` the pre-update one.

    Raises:
        TypeError: If ``model`` is not a :class:`GyroStarTrackerModel`.

    Equation:
        [EQ-AERO-ATT-002], [EQ-AERO-ATT-003].

    References:
        [REF-KALMAN-1960], [REF-FARRENKOPF-1978].

    """
    if not isinstance(model, GyroStarTrackerModel):
        raise TypeError("model must be a GyroStarTrackerModel")
    dt = model.dt
    r = model.sigma_n**2
    q = model.sigma_u**2 * dt
    beta = dt * math.sqrt(q / r)
    drive = model.sigma_v**2 * dt / r
    root = math.sqrt(beta * beta / 3.0 + 16.0 + 4.0 * drive)
    if beta >= 4.0:
        w = 0.5 * (beta - 4.0 + root)
    else:
        w = (beta * (2.0 - beta / 6.0) + drive) * 2.0 / (4.0 - beta + root)
    spread = math.sqrt(w * (w + 4.0))
    t = 0.5 * (w + 2.0 + spread)
    p11 = r * t * spread
    p12 = -math.sqrt(q * r) * t
    f11 = p11 / (t * t)
    f12 = p12 / (t * t)
    f22 = math.sqrt(q) * p11 / (math.sqrt(r) * t * dt) - 0.5 * q
    f22 = max(f22, 0.0)
    s = r * t * t
    return SteadyStateAttitude(
        predicted_covariance=((p11, p12), (p12, f22 + q)),
        filtered_covariance=((f11, f12), (f12, f22)),
        gain=(p11 / s, p12 / s),
        attitude_std_predicted=math.sqrt(p11),
        attitude_std_filtered=math.sqrt(f11),
        bias_std_filtered=math.sqrt(f22),
        method="closed_form",
        iterations=0,
        converged=True,
    )


def _mm(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two ``2 x 2`` matrices.

    Args:
        a: Left factor.
        b: Right factor.

    Returns:
        The product.

    References:
        [REF-KALMAN-1960]. Linear algebra only.

    """
    return tuple(
        tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] for j in range(2)) for i in range(2)
    )


def _tr(a: Matrix) -> Matrix:
    """Transpose a ``2 x 2`` matrix.

    Args:
        a: The matrix.

    Returns:
        Its transpose.

    References:
        [REF-KALMAN-1960]. Linear algebra only.

    """
    return ((a[0][0], a[1][0]), (a[0][1], a[1][1]))


def _add(a: Matrix, b: Matrix) -> Matrix:
    """Add two ``2 x 2`` matrices.

    Args:
        a: First term.
        b: Second term.

    Returns:
        The sum.

    References:
        [REF-KALMAN-1960]. Linear algebra only.

    """
    return tuple(tuple(a[i][j] + b[i][j] for j in range(2)) for i in range(2))


def _inverse(a: Matrix) -> Matrix:
    """Invert a ``2 x 2`` matrix.

    Args:
        a: A matrix with non-zero determinant.

    Returns:
        The inverse.

    Raises:
        ValueError: If the matrix is singular.

    References:
        [REF-KALMAN-1960]. Linear algebra only.

    """
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if det == 0.0 or not math.isfinite(det):
        raise ValueError("singular matrix in the doubling iteration")
    return ((a[1][1] / det, -a[0][1] / det), (-a[1][0] / det, a[0][0] / det))


def _doubling(
    model: GyroStarTrackerModel, tolerance: float, max_iterations: int
) -> tuple[Matrix, int, bool]:
    """Solve the filter Riccati equation by structure-preserving doubling.

    Iterates ``(A, G, H) -> (A W^-1 A, G + A W^-1 G A', H + A' H W^-1 A)`` with
    ``W = I + G H``, starting from ``A = F'``, ``G = H' R^-1 H`` and ``H = Q``;
    ``H`` converges quadratically to the predicted covariance ``P^-`` of the
    stabilising solution. It shares no code with the closed form.

    Args:
        model: The gyro and star-tracker model.
        tolerance: Relative change of ``H`` that stops the iteration.
        max_iterations: Iteration limit.

    Returns:
        ``(P^-, iterations, converged)``.

    References:
        [REF-KALMAN-1960], [EQ-AERO-ATT-003].

    """
    dt = model.dt
    a: Matrix = ((1.0, 0.0), (-dt, 1.0))
    g: Matrix = ((1.0 / model.sigma_n**2, 0.0), (0.0, 0.0))
    h: Matrix = attitude_process_noise(model)
    identity: Matrix = ((1.0, 0.0), (0.0, 1.0))
    for k in range(1, max_iterations + 1):
        w = _inverse(_add(identity, _mm(g, h)))
        aw = _mm(a, w)
        a_next = _mm(aw, a)
        g_next = _add(g, _mm(_mm(aw, g), _tr(a)))
        h_next = _add(h, _mm(_mm(_tr(a), h), _mm(w, a)))
        change = max(abs(h_next[i][j] - h[i][j]) for i in range(2) for j in range(2))
        a, g, h = a_next, g_next, h_next
        if change <= tolerance * max(abs(h[0][0]), abs(h[1][1])):
            return h, k, True
    return h, max_iterations, False


def steady_state_attitude_riccati(
    model: GyroStarTrackerModel,
    *,
    method: Literal["doubling", "iteration"] = "doubling",
    tolerance: float = 1.0e-14,
    max_iterations: int | None = None,
) -> SteadyStateAttitude:
    """Return the steady-state covariance by solving the Riccati equation.

    ``"doubling"`` runs a structure-preserving doubling iteration (a few dozen
    steps even for slowly converging cases); ``"iteration"`` runs the library's
    plain recursion :func:`probstat.filtering.steady_state_kalman`, whose
    convergence is only linear and may need very many steps for weak process
    noise. Both are independent of the closed form.

    Args:
        model: The gyro and star-tracker model.
        method: ``"doubling"`` or ``"iteration"``.
        tolerance: Relative-change stopping tolerance.
        max_iterations: Iteration limit (default 100 for doubling, 100000 for
            the plain recursion).

    Returns:
        The :class:`SteadyStateAttitude`; ``converged`` is false when the
        limit was hit, which is reported rather than hidden.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: If ``method``, ``tolerance`` or ``max_iterations`` is
            invalid, or the doubling iteration meets a singular matrix.

    Equation:
        [EQ-AERO-ATT-003].

    References:
        [REF-KALMAN-1960].

    """
    if not isinstance(model, GyroStarTrackerModel):
        raise TypeError("model must be a GyroStarTrackerModel")
    if method not in ("doubling", "iteration"):
        raise ValueError("method must be 'doubling' or 'iteration'")
    tol = _positive(tolerance, "tolerance")
    limit = (
        (100 if method == "doubling" else 100_000)
        if max_iterations is None
        else _count(max_iterations, "max_iterations", 1)
    )
    if method == "doubling":
        predicted, steps, ok = _doubling(model, tol, limit)
        return _from_predicted(predicted, model.sigma_n, "doubling", steps, ok)
    result = steady_state_kalman(
        attitude_state_space(model), tolerance=tol, max_iterations=limit
    )
    pred = result.predicted_covariance
    return _from_predicted(
        pred, model.sigma_n, "iteration", result.iterations, result.converged
    )


# --------------------------------------------------------------------------
# Synthetic data.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttitudeSimulation:
    """Synthetic error-state truth and star-tracker measurements.

    Args:
        truth: True error states ``[theta_k, b_k]`` for ``k = 0..T``.
        measurements: Star-tracker readings ``(y_k,)`` for ``k = 1..T``.

    References:
        [EQ-AERO-ATT-001].

    """

    truth: tuple[Vector, ...]
    measurements: tuple[Vector, ...]


def simulate_attitude_error(
    model: GyroStarTrackerModel,
    steps: int,
    *,
    seed: int | RandomStream,
    initial_std: tuple[float, float] = (1.0e-3, 1.0e-5),
) -> AttitudeSimulation:
    """Simulate the error state of the model and its star-tracker readings.

    The initial state is drawn from ``N(0, diag(initial_std^2))``, the state
    evolves as ``x_k = F x_(k-1) + w_k`` with ``w_k ~ N(0, Q)`` (sampled with
    the Cholesky factor of ``Q``), and ``y_k = theta_k + v_k``. This is the
    exact discrete image of the continuous model, so a consistent filter has
    chi-square distributed normalised errors.

    Args:
        model: The gyro and star-tracker model.
        steps: Number of steps ``T`` (at least 1).
        seed: Root seed or stream (see ``resolve_stream``).
        initial_std: Prior standard deviations of attitude (rad) and bias
            (rad/s).

    Returns:
        The truth and the measurements.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``steps`` or ``initial_std`` is invalid.

    Equation:
        [EQ-AERO-ATT-001].

    References:
        [REF-KALMAN-1960], [EQ-AERO-ATT-001].

    """
    if not isinstance(model, GyroStarTrackerModel):
        raise TypeError("model must be a GyroStarTrackerModel")
    n = _count(steps, "steps", 1)
    s0 = _positive(initial_std[0], "initial_std[0]")
    s1 = _positive(initial_std[1], "initial_std[1]")
    gen = resolve_stream(seed).generator()
    factor = cholesky_semidefinite(attitude_process_noise(model))
    theta = s0 * gen.gauss(0.0, 1.0)
    bias = s1 * gen.gauss(0.0, 1.0)
    truth: list[Vector] = [(theta, bias)]
    readings: list[Vector] = []
    for _ in range(n):
        e1 = gen.gauss(0.0, 1.0)
        e2 = gen.gauss(0.0, 1.0)
        w1 = factor[0][0] * e1
        w2 = factor[1][0] * e1 + factor[1][1] * e2
        theta, bias = theta - model.dt * bias + w1, bias + w2
        truth.append((theta, bias))
        readings.append((theta + model.sigma_n * gen.gauss(0.0, 1.0),))
    return AttitudeSimulation(truth=tuple(truth), measurements=tuple(readings))


def simulate_gyro_rate(
    sigma_v: float,
    sigma_u: float,
    dt: float,
    steps: int,
    *,
    seed: int | RandomStream,
    true_rate: float = 0.0,
) -> tuple[float, ...]:
    """Simulate gyro rate readings with white noise and a random-walk bias.

    ``g_k = omega + b_k + n_k`` with ``b_k = b_(k-1) + sigma_u sqrt(dt) xi_k``
    (``b_0 = 0``) and ``n_k ~ N(0, sigma_v^2 / dt)``, the sampled image of a
    rate noise of spectral density ``sigma_v^2``.

    Args:
        sigma_v: Angle random walk density, rad/sqrt(s), non-negative.
        sigma_u: Rate random walk density, rad/s/sqrt(s), non-negative.
        dt: Sample interval, s.
        steps: Number of readings.
        seed: Root seed or stream.
        true_rate: Constant true rate, rad/s.

    Returns:
        The readings, rad/s.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a parameter is out of range.

    Equation:
        [EQ-AERO-ATT-005].

    References:
        [EQ-AERO-ATT-005], [REF-KALMAN-1960].

    """
    sv = _non_negative(sigma_v, "sigma_v")
    su = _non_negative(sigma_u, "sigma_u")
    step = _positive(dt, "dt")
    n = _count(steps, "steps", 2)
    omega = require_finite_real(true_rate, name="true_rate")
    gen = resolve_stream(seed).generator()
    white = sv / math.sqrt(step)
    walk = su * math.sqrt(step)
    bias = 0.0
    out: list[float] = []
    for _ in range(n):
        bias += walk * gen.gauss(0.0, 1.0)
        out.append(omega + bias + white * gen.gauss(0.0, 1.0))
    return tuple(out)


# --------------------------------------------------------------------------
# Allan variance.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AllanResult:
    """Overlapping Allan variance of a rate series.

    Args:
        cluster_sizes: Cluster lengths ``m`` in samples.
        taus: Averaging times ``m dt``, s.
        variances: Overlapping Allan variances, (rad/s)^2.
        deviations: Square roots of ``variances``, rad/s.
        terms: Number of overlapping differences used at each ``m``.
        samples: Series length ``N``.
        dt: Sample interval, s.

    Equation:
        [EQ-AERO-ATT-004].

    References:
        [EQ-AERO-ATT-004].

    """

    cluster_sizes: tuple[int, ...]
    taus: tuple[float, ...]
    variances: tuple[float, ...]
    deviations: tuple[float, ...]
    terms: tuple[int, ...]
    samples: int
    dt: float


def overlapping_allan_deviation(
    rates: Sequence[float],
    dt: float,
    *,
    cluster_sizes: Sequence[int] | None = None,
) -> AllanResult:
    """Compute the overlapping Allan variance of rate readings.

    With ``ybar_j(m) = (1/m) sum_(i=j)^(j+m-1) g_i`` the mean of the ``m``
    readings starting at ``j``, the estimator is
    ``AVAR(m dt) = 1 / (2 (N - 2m + 1)) sum_(j=0)^(N-2m) (ybar_(j+m) -
    ybar_j)^2``. The series mean is removed first (the estimator does not
    depend on it, but it keeps the cumulative sums small); cluster means come
    from cumulative sums, so the cost is ``O(N)`` per cluster size.

    Args:
        rates: Gyro rate readings, rad/s (at least 4 values).
        dt: Sample interval, s.
        cluster_sizes: Cluster lengths ``m`` (each at least 1 and at most
            ``(N - 1) // 2``). Default: powers of two up to ``N // 9``, so that
            every estimate has at least about 7 independent differences.

    Returns:
        The :class:`AllanResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the series is too short, contains a non-finite value,
            is constant, or a cluster size is out of range.

    Equation:
        [EQ-AERO-ATT-004].

    References:
        [EQ-AERO-ATT-004].

    """
    step = _positive(dt, "dt")
    data = [require_finite_real(x, name="rates") for x in rates]
    n = len(data)
    if n < 4:
        raise ValueError("at least 4 readings are required")
    if cluster_sizes is None:
        sizes: list[int] = []
        m = 1
        while m <= max(n // 9, 1) and 2 * m <= n - 1:
            sizes.append(m)
            m *= 2
    else:
        sizes = [_count(m, "cluster_sizes", 1) for m in cluster_sizes]
        if not sizes:
            raise ValueError("cluster_sizes must not be empty")
    if any(2 * m > n - 1 for m in sizes):
        raise ValueError("a cluster size is too large for the series length")
    mean = math.fsum(data) / n
    cumulative = [0.0]
    running = 0.0
    for value in data:
        running += value - mean
        cumulative.append(running)
    variances: list[float] = []
    terms: list[int] = []
    for m in sizes:
        count = n - 2 * m + 1
        total = 0.0
        for j in range(count):
            diff = cumulative[j + 2 * m] - 2.0 * cumulative[j + m] + cumulative[j]
            total += diff * diff
        variances.append(total / (2.0 * count * m * m))
        terms.append(count)
    if not all(v > 0.0 for v in variances):
        raise ValueError("the series is degenerate: an Allan variance is zero")
    return AllanResult(
        cluster_sizes=tuple(sizes),
        taus=tuple(m * step for m in sizes),
        variances=tuple(variances),
        deviations=tuple(math.sqrt(v) for v in variances),
        terms=tuple(terms),
        samples=n,
        dt=step,
    )


def allan_model_variance(
    sigma_v: float, sigma_u: float, dt: float, cluster_size: int
) -> float:
    """Return the Allan variance of the sampled gyro model.

    For readings ``g_k = b_k + n_k`` with ``n_k`` white of variance
    ``sigma_v^2 / dt`` and ``b_k`` a random walk of step variance
    ``sigma_u^2 dt``, with ``tau = m dt``,

    ``AVAR(tau) = sigma_v^2 / tau + sigma_u^2 dt (2 m^2 + 1) / (6 m)``.

    The first term is exact for white noise. The second follows from the
    weights ``(j - 1)/m`` and ``(2m - j + 1)/m`` with which the ``j``-th
    increment enters the difference of two adjacent cluster means; it tends to
    ``sigma_u^2 tau / 3`` for ``m >> 1``. On a log-log plot the deviation has
    slope ``-1/2`` where the first term dominates and ``+1/2`` where the second
    does.

    Args:
        sigma_v: Angle random walk density, rad/sqrt(s).
        sigma_u: Rate random walk density, rad/s/sqrt(s).
        dt: Sample interval, s.
        cluster_size: Cluster length ``m`` (at least 1).

    Returns:
        The variance in (rad/s)^2.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-ATT-005].

    References:
        [EQ-AERO-ATT-005].

    """
    sv = _non_negative(sigma_v, "sigma_v")
    su = _non_negative(sigma_u, "sigma_u")
    step = _positive(dt, "dt")
    m = _count(cluster_size, "cluster_size", 1)
    return sv * sv / (m * step) + su * su * step * (2.0 * m * m + 1.0) / (6.0 * m)


@dataclass(frozen=True, slots=True)
class AllanFit:
    """Noise parameters recovered from an Allan variance.

    Args:
        sigma_v: Fitted angle random walk density, rad/sqrt(s).
        sigma_u: Fitted rate random walk density, rad/s/sqrt(s).
        sigma_v_error: Approximate standard error of ``sigma_v``.
        sigma_u_error: Approximate standard error of ``sigma_u``.
        converged: True when both components were identified as positive.
        stop_reason: ``"ok"``, ``"rate_random_walk_not_identified"``,
            ``"angle_random_walk_not_identified"`` or
            ``"no_noise_identified"``; a component not identified is reported
            as zero with ``converged`` false, never silently kept.
        residual_rms: Root mean square of the relative residuals.

    Equation:
        [EQ-AERO-ATT-006].

    References:
        [EQ-AERO-ATT-006].

    """

    sigma_v: float
    sigma_u: float
    sigma_v_error: float
    sigma_u_error: float
    converged: bool
    stop_reason: str
    residual_rms: float


_ERROR_INFLATION = 2.3
"""Factor applied to the least-squares standard errors of the Allan fit.

The weighted least squares treats the Allan variances at different cluster
sizes as independent, but they are built from the same samples and strongly
correlated, so its raw standard errors are too small. The factor was measured
by simulating this model (40 seeds, N = 40000, sigma_v = 2e-4, sigma_u = 3e-6
and 1e-6, dt = 0.1): the seed-to-seed standard deviation exceeded the raw error
by 2.04 to 2.55 times. It is an empirical correction for this fit, not a
derived constant; the tests check the resulting scatter.
"""


def _weighted_fit(
    columns: tuple[list[float], list[float]],
    target: list[float],
    weights: list[float],
) -> tuple[float, float, float, float, float]:
    """Solve a weighted two-column least-squares problem.

    Args:
        columns: The two regressor columns.
        target: Observations.
        weights: Positive weights.

    Returns:
        ``(a, b, cov_aa, cov_bb, cov_ab)`` of the estimate.

    Raises:
        ValueError: If the normal equations are singular.

    References:
        [EQ-AERO-ATT-006].

    """
    x1, x2 = columns
    s11 = math.fsum(w * u * u for w, u in zip(weights, x1, strict=True))
    s12 = math.fsum(w * u * v for w, u, v in zip(weights, x1, x2, strict=True))
    s22 = math.fsum(w * v * v for w, v in zip(weights, x2, strict=True))
    t1 = math.fsum(w * u * y for w, u, y in zip(weights, x1, target, strict=True))
    t2 = math.fsum(w * v * y for w, v, y in zip(weights, x2, target, strict=True))
    det = s11 * s22 - s12 * s12
    if not det > 0.0:
        raise ValueError("the Allan fit is ill-posed: use more cluster sizes")
    return (
        (s22 * t1 - s12 * t2) / det,
        (s11 * t2 - s12 * t1) / det,
        s22 / det,
        s11 / det,
        -s12 / det,
    )


def fit_allan_noise(
    result: AllanResult, *, min_cluster: int = 1, max_cluster: int | None = None
) -> AllanFit:
    """Fit ``sigma_v`` and ``sigma_u`` to an Allan variance.

    Weighted linear least squares of ``AVAR(m)`` on the two regressors of
    :func:`allan_model_variance` (``1/tau`` and ``dt (2 m^2 + 1) / (6 m)``),
    whose coefficients are ``sigma_v^2`` and ``sigma_u^2``. The weights are the
    inverse squared relative error, ``nu_m / AVAR_m^2`` (two passes, the second
    with the fitted curve instead of the noisy estimates), with the proxy
    ``nu_m = N / m - 1`` degrees of freedom of the estimate at cluster size
    ``m``. That proxy ignores the correlation between cluster sizes, so the raw
    least-squares errors are multiplied by an empirical factor 2.3 (see
    ``_ERROR_INFLATION``); the reported errors are approximate and the tests
    compare them with the observed seed-to-seed scatter.

    Args:
        result: Output of :func:`overlapping_allan_deviation`.
        min_cluster: Smallest cluster size used.
        max_cluster: Largest cluster size used (default: all).

    Returns:
        The :class:`AllanFit`. A component whose coefficient comes out
        non-positive is reported as zero with a ``stop_reason``.

    Raises:
        TypeError: If ``result`` has the wrong type.
        ValueError: If fewer than 3 cluster sizes remain or the problem is
            singular.

    Equation:
        [EQ-AERO-ATT-006].

    References:
        [EQ-AERO-ATT-006], [EQ-AERO-ATT-005].

    """
    if not isinstance(result, AllanResult):
        raise TypeError("result must be an AllanResult")
    low = _count(min_cluster, "min_cluster", 1)
    high = (
        max(result.cluster_sizes)
        if max_cluster is None
        else _count(max_cluster, "max_cluster", low)
    )
    keep = [i for i, m in enumerate(result.cluster_sizes) if low <= m <= high]
    if len(keep) < 3:
        raise ValueError("at least 3 cluster sizes are required")
    sizes = [result.cluster_sizes[i] for i in keep]
    avar = [result.variances[i] for i in keep]
    dt = result.dt
    col_v = [1.0 / (m * dt) for m in sizes]
    col_u = [dt * (2.0 * m * m + 1.0) / (6.0 * m) for m in sizes]
    nu = [max(result.samples / m - 1.0, 1.0) for m in sizes]
    scale = list(avar)
    solution = (0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(2):
        weights = [n_i / (s * s) for n_i, s in zip(nu, scale, strict=True)]
        solution = _weighted_fit((col_v, col_u), avar, weights)
        scale = [
            max(solution[0] * u + solution[1] * v, 1e-300)
            if solution[0] > 0.0 and solution[1] > 0.0
            else y
            for u, v, y in zip(col_v, col_u, avar, strict=True)
        ]
    a, b, caa, cbb, _ = solution
    reason = "ok"
    if a > 0.0 and b > 0.0:
        pass
    else:
        weights = [n_i / (y * y) for n_i, y in zip(nu, avar, strict=True)]
        sxx_v = math.fsum(w * u * u for w, u in zip(weights, col_v, strict=True))
        sxx_u = math.fsum(w * v * v for w, v in zip(weights, col_u, strict=True))
        a_only = (
            math.fsum(w * u * y for w, u, y in zip(weights, col_v, avar, strict=True))
            / sxx_v
        )
        b_only = (
            math.fsum(w * v * y for w, v, y in zip(weights, col_u, avar, strict=True))
            / sxx_u
        )
        if b <= 0.0 < a_only:
            a, b, caa, cbb = a_only, 0.0, 1.0 / sxx_v, 0.0
            reason = "rate_random_walk_not_identified"
        elif a <= 0.0 < b_only:
            a, b, caa, cbb = 0.0, b_only, 0.0, 1.0 / sxx_u
            reason = "angle_random_walk_not_identified"
        else:
            a, b, caa, cbb = 0.0, 0.0, 0.0, 0.0
            reason = "no_noise_identified"
    sigma_v = math.sqrt(max(a, 0.0))
    sigma_u = math.sqrt(max(b, 0.0))
    err_v = (
        _ERROR_INFLATION * math.sqrt(max(caa, 0.0)) / (2.0 * sigma_v)
        if sigma_v > 0.0
        else 0.0
    )
    err_u = (
        _ERROR_INFLATION * math.sqrt(max(cbb, 0.0)) / (2.0 * sigma_u)
        if sigma_u > 0.0
        else 0.0
    )
    rel = [
        (y - (sigma_v**2 * u + sigma_u**2 * v)) / y
        for u, v, y in zip(col_v, col_u, avar, strict=True)
    ]
    rms = math.sqrt(math.fsum(r * r for r in rel) / len(rel))
    return AllanFit(
        sigma_v=sigma_v,
        sigma_u=sigma_u,
        sigma_v_error=err_v,
        sigma_u_error=err_u,
        converged=reason == "ok",
        stop_reason=reason,
        residual_rms=rms,
    )


# --------------------------------------------------------------------------
# Pointing error budget.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointingBudgetResult:
    """Pointing error budget with its Monte Carlo cross-check.

    Args:
        budget: The :class:`~probstat.uq.UncertaintyBudget` of the four
            sources, in the order of :data:`POINTING_SOURCES`, with the
            correlation matrix.
        rss: Root-sum-square of the sources, ignoring correlation, rad.
        combined: Law-of-propagation standard uncertainty with correlation, rad.
        requirement: The 3-sigma pointing requirement, rad.
        margin: ``requirement / (3 combined)``; at least 1 means the 3-sigma
            requirement is met.
        meets_requirement: ``3 combined <= requirement``.
        probability_within_requirement: Gaussian ``P(|e| <= requirement)``,
            ``erf(requirement / (sqrt(2) combined))``.
        mc_standard_uncertainty: Standard deviation of the total error from
            :func:`probstat.uq.monte_carlo_budget`.
        mc_probability_within_requirement: Monte Carlo ``P(|e| <= requirement)``.
        mc_probability_error: Binomial standard error of that estimate.
        rss_coverage_mc: Monte Carlo ``P(|e| <= 3 rss)``: the real coverage of
            an interval built from the uncorrelated RSS.
        rss_coverage_gaussian: The coverage the RSS interval would have with
            the correct standard deviation, ``erf(3 / sqrt(2))``.
        mc_size: Monte Carlo sample size.

    Equation:
        [EQ-AERO-ATT-007], [EQ-AERO-ATT-008].

    References:
        [EQ-AERO-ATT-007], [EQ-AERO-ATT-008].

    """

    budget: UncertaintyBudget
    rss: float
    combined: float
    requirement: float
    margin: float
    meets_requirement: bool
    probability_within_requirement: float
    mc_standard_uncertainty: float
    mc_probability_within_requirement: float
    mc_probability_error: float
    rss_coverage_mc: float
    rss_coverage_gaussian: float
    mc_size: int


def _correlation_matrix(
    correlations: Mapping[tuple[str, str], float] | None,
) -> tuple[tuple[float, ...], ...] | None:
    """Build the symmetric correlation matrix from named pair values.

    Args:
        correlations: Mapping from a pair of source names to a correlation in
            ``[-1, 1]``, or ``None``.

    Returns:
        The matrix in :data:`POINTING_SOURCES` order, or ``None``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a name is unknown, a pair repeats or is a diagonal pair,
            or a value is outside ``[-1, 1]``.

    References:
        [EQ-AERO-ATT-007].

    """
    if not correlations:
        return None
    size = len(POINTING_SOURCES)
    matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    seen: set[frozenset[str]] = set()
    for (first, second), value in correlations.items():
        if first not in POINTING_SOURCES or second not in POINTING_SOURCES:
            raise ValueError(f"unknown source in pair ({first!r}, {second!r})")
        if first == second:
            raise ValueError("a correlation pair needs two different sources")
        key = frozenset((first, second))
        if key in seen:
            raise ValueError("a correlation pair is given twice")
        seen.add(key)
        rho = require_finite_real(value, name="correlation")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("a correlation must lie in [-1, 1]")
        i, j = POINTING_SOURCES.index(first), POINTING_SOURCES.index(second)
        matrix[i][j] = matrix[j][i] = rho
    return tuple(tuple(row) for row in matrix)


def pointing_budget(
    model: GyroStarTrackerModel,
    *,
    control_jitter: float,
    thermal_elastic: float,
    tracker_alignment: float,
    requirement_3sigma: float,
    correlations: Mapping[tuple[str, str], float] | None = None,
    size: int = 20_000,
    seed: int | RandomStream = 0,
) -> PointingBudgetResult:
    """Assemble the pointing error budget and check it by Monte Carlo.

    The total pointing error of one axis is the sum of four zero-mean sources
    (sensitivity 1 each): the attitude knowledge error left by the filter (the
    post-update steady-state standard deviation of
    :func:`steady_state_attitude_closed_form`), the control jitter, the
    thermal-elastic misalignment between the star tracker and the payload, and
    the star-tracker alignment error. Knowledge, alignment and thermal
    distortion can be correlated; the correlations are applied by the law of
    propagation ``u_c^2 = sum_ij u_i u_j r_ij`` ([EQ-AERO-ATT-007]) and by a
    Gaussian model in the Monte Carlo. Gaussian sources make the total
    Gaussian, so ``P(|e| <= requirement) = erf(requirement / (sqrt(2) u_c))``
    ([EQ-AERO-ATT-008]); the Monte Carlo estimate checks that, and the coverage
    of the uncorrelated RSS 3-sigma interval shows what ignoring correlation
    does to the stated 99.73 %.

    Args:
        model: The gyro and star-tracker model (supplies the knowledge term).
        control_jitter: Control jitter standard deviation, rad.
        thermal_elastic: Thermal-elastic misalignment standard deviation, rad.
        tracker_alignment: Star-tracker alignment standard deviation, rad.
        requirement_3sigma: The 3-sigma pointing requirement, rad: the budget
            must satisfy ``3 u_c <= requirement_3sigma``.
        correlations: Optional ``{(source_a, source_b): rho}`` over the names in
            :data:`POINTING_SOURCES`; unspecified pairs are uncorrelated.
        size: Monte Carlo sample size (at least 1000).
        seed: Root seed or stream.

    Returns:
        The :class:`PointingBudgetResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If a value is out of range or the correlation matrix is not
            positive semi-definite.

    Equation:
        [EQ-AERO-ATT-007], [EQ-AERO-ATT-008].

    References:
        [EQ-AERO-ATT-007], [EQ-AERO-ATT-008].

    """
    knowledge = steady_state_attitude_closed_form(model).attitude_std_filtered
    stds = (
        knowledge,
        _positive(control_jitter, "control_jitter"),
        _positive(thermal_elastic, "thermal_elastic"),
        _positive(tracker_alignment, "tracker_alignment"),
    )
    requirement = _positive(requirement_3sigma, "requirement_3sigma")
    n = _count(size, "size", 1000)
    matrix = _correlation_matrix(correlations)
    kinds = ("random", "random", "bias", "bias")
    entries = [
        BudgetEntry(
            name,
            u,
            1.0,
            evaluation="B",
            nature="aleatory",
            kind=kind,
            source="synthetic",
        )
        for name, u, kind in zip(POINTING_SOURCES, stds, kinds, strict=True)
    ]
    budget = UncertaintyBudget(entries, correlation=matrix)
    rss = budget.root_sum_square().value
    combined = budget.combined_standard_uncertainty()
    stream = resolve_stream(seed)
    mc = monte_carlo_budget(
        lambda x: math.fsum(x),
        [Normal(0.0, u) for u in stds],
        size=n,
        seed=stream.child("library"),
        correlation=matrix,
    )
    gen = stream.child("probability").generator()
    factor = cholesky(matrix) if matrix is not None else None
    inside = 0
    inside_rss = 0
    for _ in range(n):
        z = [gen.gauss(0.0, 1.0) for _ in stds]
        if factor is not None:
            z = [math.fsum(factor[i][k] * z[k] for k in range(i + 1)) for i in range(4)]
        total = math.fsum(u * v for u, v in zip(stds, z, strict=True))
        inside += abs(total) <= requirement
        inside_rss += abs(total) <= 3.0 * rss
    p_mc = inside / n
    return PointingBudgetResult(
        budget=budget,
        rss=rss,
        combined=combined,
        requirement=requirement,
        margin=requirement / (3.0 * combined),
        meets_requirement=3.0 * combined <= requirement,
        probability_within_requirement=math.erf(
            requirement / (math.sqrt(2.0) * combined)
        ),
        mc_standard_uncertainty=mc.standard_uncertainty,
        mc_probability_within_requirement=p_mc,
        mc_probability_error=math.sqrt(max(p_mc * (1.0 - p_mc), 1.0 / n) / n),
        rss_coverage_mc=inside_rss / n,
        rss_coverage_gaussian=math.erf(3.0 / math.sqrt(2.0)),
        mc_size=n,
    )


# --------------------------------------------------------------------------
# Sensor grades: Sobol ranking and requirement probability.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensorGrades:
    """Log-uniform ranges of the three sensor parameters (the sensor grades).

    Each parameter is log-uniform between a low and a high value, a synthetic
    description of a family of sensor grades from good to poor.

    Args:
        sigma_v: ``(low, high)`` angle random walk density, rad/sqrt(s).
        sigma_u: ``(low, high)`` rate random walk density, rad/s/sqrt(s).
        sigma_n: ``(low, high)`` star-tracker noise per sample, rad.

    Raises:
        TypeError: If a bound is not real.
        ValueError: If a range is not ``0 < low < high``.

    Equation:
        [EQ-AERO-ATT-009].

    References:
        [EQ-AERO-ATT-009].

    """

    sigma_v: tuple[float, float]
    sigma_u: tuple[float, float]
    sigma_n: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate the three ranges.

        Raises:
            TypeError: If a bound is not real.
            ValueError: If a range is not ``0 < low < high``.

        References:
            [EQ-AERO-ATT-009]. Validation only.

        """
        for name in ("sigma_v", "sigma_u", "sigma_n"):
            low, high = getattr(self, name)
            lo = _positive(low, f"{name} low")
            hi = _positive(high, f"{name} high")
            if not lo < hi:
                raise ValueError(f"{name} range must satisfy low < high")
            object.__setattr__(self, name, (lo, hi))

    def model(self, unit: Sequence[float], dt: float) -> GyroStarTrackerModel:
        """Map a point of the unit cube to a model.

        Args:
            unit: Three numbers in ``[0, 1]`` (``sigma_v``, ``sigma_u``,
                ``sigma_n``).
            dt: Sample interval, s.

        Returns:
            The model with log-uniformly mapped parameters.

        References:
            [EQ-AERO-ATT-009].

        """
        values = []
        for u, (low, high) in zip(
            unit, (self.sigma_v, self.sigma_u, self.sigma_n), strict=True
        ):
            values.append(
                math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
            )
        return GyroStarTrackerModel(values[0], values[1], values[2], dt)


@dataclass(frozen=True, slots=True)
class GradeSobolResult:
    """Sobol indices of the knowledge error over the sensor grades.

    Args:
        names: ``("sigma_v", "sigma_u", "sigma_n")``.
        first_order: First-order indices, in the order of ``names``.
        total: Total-effect indices.
        variance: Output variance of the knowledge standard deviation, rad^2.
        ranking: Names ordered by decreasing total index.
        evaluations: Model evaluations used.

    Equation:
        [EQ-AERO-ATT-009].

    References:
        [EQ-AERO-ATT-009], [REF-SOBOL-1993].

    """

    names: tuple[str, ...]
    first_order: tuple[float, ...]
    total: tuple[float, ...]
    variance: float
    ranking: tuple[str, ...]
    evaluations: int


def knowledge_sobol_ranking(
    grades: SensorGrades, dt: float, *, size: int, seed: int | RandomStream
) -> GradeSobolResult:
    """Rank the sensor parameters by their Sobol indices on the knowledge error.

    The model output is the steady-state post-update attitude standard
    deviation of :func:`steady_state_attitude_closed_form` as a function of
    ``(sigma_v, sigma_u, sigma_n)`` drawn independently log-uniform over
    ``grades``. First-order and total indices come from the library's
    pick-freeze estimator :func:`probstat.sensitivity.estimate_sobol_indices`
    on two independent plain Monte Carlo matrices.

    Args:
        grades: The parameter ranges.
        dt: Sample interval, s.
        size: Rows of each of the two sample matrices (at least 64).
        seed: Root seed or stream.

    Returns:
        The :class:`GradeSobolResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range or the output is constant.

    Equation:
        [EQ-AERO-ATT-009].

    References:
        [REF-SOBOL-1993], [EQ-AERO-ATT-009].

    """
    if not isinstance(grades, SensorGrades):
        raise TypeError("grades must be a SensorGrades")
    step = _positive(dt, "dt")
    n = _count(size, "size", 64)
    stream = resolve_stream(seed)
    rng_a = stream.child("A").generator()
    rng_b = stream.child("B").generator()
    sample_a = [tuple(rng_a.random() for _ in range(3)) for _ in range(n)]
    sample_b = [tuple(rng_b.random() for _ in range(3)) for _ in range(n)]

    def knowledge(row: tuple[float, ...]) -> float:
        """Return the closed-form knowledge standard deviation of one row.

        Args:
            row: Point of the unit cube.

        Returns:
            Steady-state post-update attitude standard deviation, rad.

        References:
            [EQ-AERO-ATT-009].

        """
        model = grades.model(row, step)
        return steady_state_attitude_closed_form(model).attitude_std_filtered

    sobol = estimate_sobol_indices(knowledge, sample_a, sample_b)
    names = ("sigma_v", "sigma_u", "sigma_n")
    order = sorted(range(3), key=lambda i: -sobol.total_effect[i])
    return GradeSobolResult(
        names=names,
        first_order=tuple(sobol.first_order),
        total=tuple(sobol.total_effect),
        variance=sobol.output_variance,
        ranking=tuple(names[i] for i in order),
        evaluations=sobol.evaluations,
    )


@dataclass(frozen=True, slots=True)
class GradeRequirementResult:
    """Probability that the 3-sigma pointing requirement is met over grades.

    Args:
        probability: Share of sampled sensor grades whose budget satisfies
            ``3 u_c <= requirement``.
        probability_error: Binomial standard error of ``probability``.
        mean_combined: Mean combined standard uncertainty over the grades, rad.
        size: Number of sampled grades.

    Equation:
        [EQ-AERO-ATT-009].

    References:
        [EQ-AERO-ATT-009].

    """

    probability: float
    probability_error: float
    mean_combined: float
    size: int


def grade_requirement_probability(
    grades: SensorGrades,
    dt: float,
    *,
    control_jitter: float,
    thermal_elastic: float,
    tracker_alignment: float,
    requirement_3sigma: float,
    size: int,
    seed: int | RandomStream,
) -> GradeRequirementResult:
    """Return the probability that the 3-sigma requirement holds over grades.

    Sensor parameters are drawn log-uniform over ``grades`` (a stand-in for not
    knowing which grade of sensor will be delivered); for each draw the
    knowledge error is added to the other three sources in quadrature
    (independent sources) and the requirement ``3 u_c <= requirement_3sigma``
    is tested.

    Args:
        grades: The parameter ranges.
        dt: Sample interval, s.
        control_jitter: Control jitter standard deviation, rad.
        thermal_elastic: Thermal-elastic standard deviation, rad.
        tracker_alignment: Star-tracker alignment standard deviation, rad.
        requirement_3sigma: The 3-sigma requirement, rad.
        size: Number of sampled grades (at least 100).
        seed: Root seed or stream.

    Returns:
        The :class:`GradeRequirementResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is out of range.

    Equation:
        [EQ-AERO-ATT-009].

    References:
        [EQ-AERO-ATT-009], [EQ-AERO-ATT-007].

    """
    if not isinstance(grades, SensorGrades):
        raise TypeError("grades must be a SensorGrades")
    step = _positive(dt, "dt")
    others = (
        _positive(control_jitter, "control_jitter") ** 2
        + _positive(thermal_elastic, "thermal_elastic") ** 2
        + _positive(tracker_alignment, "tracker_alignment") ** 2
    )
    limit = _positive(requirement_3sigma, "requirement_3sigma")
    n = _count(size, "size", 100)
    gen = resolve_stream(seed).generator()
    met = 0
    total = 0.0
    for _ in range(n):
        point = (gen.random(), gen.random(), gen.random())
        knowledge = steady_state_attitude_closed_form(
            grades.model(point, step)
        ).attitude_std_filtered
        combined = math.sqrt(knowledge**2 + others)
        total += combined
        met += 3.0 * combined <= limit
    p = met / n
    return GradeRequirementResult(
        probability=p,
        probability_error=math.sqrt(max(p * (1.0 - p), 1.0 / n) / n),
        mean_combined=total / n,
        size=n,
    )
