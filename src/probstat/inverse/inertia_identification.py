"""Identification of a rigid-body inertia tensor from manoeuvre data.

Experimental (added in 0.28).

Setting: a rigid spacecraft rotates about its centre of mass with body-frame
angular rate ``w`` (rad/s) under a known applied torque ``tau(t)`` (N m, for
example reaction wheels or thrusters) and an optional unknown constant
disturbance torque ``d``. Euler's equations
``J w' + w x (J w) = tau + d`` [EQ-INV-INERTIA-001] (the attitude-dynamics
setting of [REF-MARKLEY-CRASSIDIS-2014]) contain the symmetric inertia tensor
``J`` (kg m^2). The module identifies ``J`` (and ``d``) from noisy gyro
samples of ``w``. Everything is SI and every bundled data generator is
synthetic: nothing here is data from a real vehicle.

What is implemented (derived here, not copied from a text):

* Forward model: the torque is a zero-order hold on each sample interval and
  Euler's equations are integrated by classical RK4 with ``substeps`` steps
  per interval [EQ-INV-INERTIA-001]. Without torque the rotational energy
  ``T = w'Jw/2`` and the magnitude of the angular momentum ``|J w|`` are
  invariants (the body-frame vector ``J w`` itself rotates; it is constant only
  in the inertial frame) and are conserved by RK4 up to its truncation error
  [EQ-INV-INERTIA-009].
* Parameterization: ``J = L L'`` with ``L`` lower triangular and
  ``L_ii = exp(a_i)``; the six parameters are ``theta = (a_1, a_2, a_3,
  l_21, l_31, l_32)`` so that ``J`` is symmetric positive definite for every
  real ``theta`` [EQ-INV-INERTIA-002]. Positive definiteness is not enough for
  a rigid body: the principal moments must also satisfy the triangle
  inequalities ``I_1 + I_2 >= I_3`` (and permutations). Principal moments and
  axes come from the symmetric eigen-decomposition; :func:`check_realizable`
  reports the margins and the samplers reject draws that violate them
  [EQ-INV-INERTIA-002].
* Regression (a): Euler's equations are *linear* in the six distinct entries
  ``j = (J_xx, J_yy, J_zz, J_xy, J_xz, J_yz)`` and in ``d`` once the rate and
  its derivative are known: ``(M(w') + [w]x M(w)) j - d = tau``
  [EQ-INV-INERTIA-003]. With finite-differenced rates the derivative is noisy,
  and it enters the *regressors*, so ordinary least squares is an
  errors-in-variables estimator: biased even with unlimited data
  [EQ-INV-INERTIA-004]. :func:`regression_estimate` offers a central
  difference and an integral (windowed) form; the bias is measured in the
  tests, not assumed.
* Estimation (b): the output-error likelihood compares the integrated rates
  with the gyro samples, ``y_k = w(t_k; theta) + e_k`` with ``e_k ~
  N(0, sigma^2 I)``, and is maximized by Levenberg-Marquardt on the whitened
  residuals with central-difference sensitivities [EQ-INV-INERTIA-005].
  Because the integrated model is exact (up to RK4), the output-error
  estimator has no differentiation bias.
* Estimation (c): the Gauss-Newton Hessian at the MAP gives the Laplace
  covariance of ``theta``; its image under ``theta -> j`` (exact derivative of
  ``L L'``) gives the covariance of the inertia entries
  [EQ-INV-INERTIA-006]. Sampling the Laplace Gaussian and mapping through the
  eigen-decomposition gives the posterior of the principal moments and the
  principal-axis angles, with physical-realizability rejection
  [EQ-INV-INERTIA-007].
* Experiment design: the Fisher information of the output-error model
  evaluated at a nominal ``theta`` gives, with the prior precision, the
  Bayesian D-criterion ``log det(F + P)``; :func:`assess_excitation` compares
  one-axis, sequential three-axis and simultaneous multi-axis torque
  sequences [EQ-INV-INERTIA-008].

Common pitfalls:
    - Torques about one axis only identify the corresponding column of
      ``J^-1`` at small rates (``w' ~ J^-1 tau``); the other entries are only
      weakly constrained through the gyroscopic term ``w x (J w)``. The
      Laplace covariance without a prior is then singular; this is reported
      (``covariance_theta is None``), never regularized silently.
    - A constant disturbance torque is confounded with ``tau`` on every
      interval where ``tau`` is constant and the rate is small; torque
      reversals (``+A, 0, -A, 0``) and coasting arcs separate them.
    - The initial rate is taken as known (a rest start in the bundled
      generators); a noisy initial rate would bias both estimators.
    - Principal axes are ordered by ascending moment; nearly equal moments
      make the axes (and their posterior angles) ill defined.
    - The Laplace posterior is local and Gaussian in ``theta``; the posterior
      of ``J`` and of the moments is its nonlinear image, obtained by
      sampling, not Gaussian.
    - The gyro is modelled as white Gaussian noise of known standard
      deviation; bias, scale-factor and misalignment errors are ignored.

References:
    [REF-MARKLEY-CRASSIDIS-2014], [REF-NOCEDAL-WRIGHT-2006],
    [REF-GOLUB-VANLOAN-2013], [REF-HAIRER-NORSETT-WANNER-1993],
    [REF-TARANTOLA-2005].

"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    householder_qr,
    inverse_spd,
    matmul,
    symmetric_eigen,
    transpose,
)
from probstat.optimization import central_jacobian, least_squares

_SYMMETRY_TOLERANCE = 1e-10
_ZERO3: Vector = (0.0, 0.0, 0.0)
EXCITATION_KINDS = ("single_axis", "sequential", "multi_axis")
"""Torque-sequence families of :func:`excitation_torques`."""

_ENTRY_NAMES = ("Jxx", "Jyy", "Jzz", "Jxy", "Jxz", "Jyz")


def _positive(value: object, name: str) -> float:
    """Return a finite positive float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If it is not finite and positive.

    References:
        [EQ-INV-INERTIA-001]. No new equation is introduced.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _vec3(value: Sequence[float], name: str) -> Vector:
    """Return a finite 3-vector.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the length is not three or an entry is not finite.

    References:
        [EQ-INV-INERTIA-001]. No new equation is introduced.

    """
    vector = as_vector(value, name=name)
    if len(vector) != 3:
        raise ValueError(f"{name} must have three components")
    return vector


def _inertia_matrix(inertia: Sequence[Sequence[float]]) -> Matrix:
    """Return a validated symmetric positive-definite ``3 x 3`` inertia.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is not ``3 x 3``, symmetric and positive
            definite.

    References:
        [EQ-INV-INERTIA-002]. No new equation is introduced.

    """
    matrix = as_matrix(inertia, name="inertia", square=True)
    if len(matrix) != 3:
        raise ValueError("inertia must be a 3 x 3 matrix")
    scale = max(abs(v) for row in matrix for v in row)
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(matrix[i][j] - matrix[j][i]) > _SYMMETRY_TOLERANCE * scale:
                raise ValueError("inertia must be symmetric")
    sym = tuple(
        tuple(0.5 * (matrix[i][j] + matrix[j][i]) for j in range(3)) for i in range(3)
    )
    cholesky(sym)  # raises ValueError unless positive definite
    return sym


def _apply3(matrix: Matrix, v: Vector) -> Vector:
    """Return the product of a ``3 x 3`` matrix and a 3-vector.

    References:
        [EQ-INV-INERTIA-001]. No new equation is introduced.

    """
    return (
        matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
        matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
        matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2],
    )


def _cross(a: Vector, b: Vector) -> Vector:
    """Return the cross product of two 3-vectors.

    References:
        [EQ-INV-INERTIA-001]. No new equation is introduced.

    """
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _derivative(
    inertia: Matrix, inverse: Matrix, rate: Vector, torque: Vector
) -> Vector:
    """Return ``w' = J^-1 (u - w x J w)`` for total torque ``u``.

    References:
        [EQ-INV-INERTIA-001]. No new equation is introduced.

    """
    gyro = _cross(rate, _apply3(inertia, rate))
    return _apply3(
        inverse,
        (torque[0] - gyro[0], torque[1] - gyro[1], torque[2] - gyro[2]),
    )


def _rk4(
    inertia: Matrix, inverse: Matrix, rate: Vector, torque: Vector, step: float
) -> Vector:
    """Advance the rate by one classical RK4 step at constant torque.

    References:
        [EQ-INV-INERTIA-001], [REF-HAIRER-NORSETT-WANNER-1993]. The RK4
        formula is standard; it is not a new equation.

    """

    def shifted(base: Vector, slope: Vector, scale: float) -> Vector:
        """Return ``base + scale * slope`` componentwise.

        References:
            [REF-HAIRER-NORSETT-WANNER-1993]. No equation is evaluated.

        """
        return (
            base[0] + scale * slope[0],
            base[1] + scale * slope[1],
            base[2] + scale * slope[2],
        )

    k1 = _derivative(inertia, inverse, rate, torque)
    k2 = _derivative(inertia, inverse, shifted(rate, k1, 0.5 * step), torque)
    k3 = _derivative(inertia, inverse, shifted(rate, k2, 0.5 * step), torque)
    k4 = _derivative(inertia, inverse, shifted(rate, k3, step), torque)
    return tuple(
        rate[i] + step / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        for i in range(3)
    )


def rotational_energy(
    inertia: Sequence[Sequence[float]], rate: Sequence[float]
) -> float:
    """Return the rotational kinetic energy ``T = w' J w / 2`` in joules.

    Args:
        inertia: Symmetric positive-definite ``3 x 3`` inertia (kg m^2).
        rate: Body angular rate (rad/s).

    Returns:
        The energy.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the inertia is not symmetric positive definite or the
            rate does not have three components.

    Equation:
        [EQ-INV-INERTIA-009].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """
    matrix = _inertia_matrix(inertia)
    w = _vec3(rate, "rate")
    h = _apply3(matrix, w)
    return 0.5 * math.fsum(w[i] * h[i] for i in range(3))


def angular_momentum(
    inertia: Sequence[Sequence[float]], rate: Sequence[float]
) -> Vector:
    """Return the body-frame angular momentum ``h = J w`` in kg m^2/s.

    Args:
        inertia: Symmetric positive-definite ``3 x 3`` inertia (kg m^2).
        rate: Body angular rate (rad/s).

    Returns:
        The momentum vector.

    Raises:
        TypeError: If an entry is not real.
        ValueError: As :func:`rotational_energy`.

    Equation:
        [EQ-INV-INERTIA-009].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """
    return _apply3(_inertia_matrix(inertia), _vec3(rate, "rate"))


def simulate_rates(
    inertia: Sequence[Sequence[float]],
    torques: Sequence[Sequence[float]],
    dt: float,
    *,
    initial_rate: Sequence[float] = _ZERO3,
    disturbance: Sequence[float] = _ZERO3,
    substeps: int = 1,
) -> tuple[Vector, ...]:
    """Integrate Euler's equations by RK4 under a zero-order-hold torque.

    ``torques[k]`` acts on ``[t_k, t_(k+1)]`` with ``t_k = k dt``; each
    interval is split into ``substeps`` RK4 steps.

    Args:
        inertia: Symmetric positive-definite ``3 x 3`` inertia (kg m^2).
        torques: Applied torques (N m), one 3-vector per interval.
        dt: Sample interval (s).
        initial_rate: Rate at ``t_0`` (rad/s).
        disturbance: Constant extra torque ``d`` (N m).
        substeps: RK4 steps per interval.

    Returns:
        The ``N + 1`` rates at ``t_0 ... t_N`` (``N = len(torques)``).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the inertia is invalid, a vector has the wrong length,
            ``dt`` is not positive, ``substeps < 1``, or the state becomes
            non-finite.

    Equation:
        [EQ-INV-INERTIA-001].

    References:
        [REF-MARKLEY-CRASSIDIS-2014], [REF-HAIRER-NORSETT-WANNER-1993].

    """
    matrix = _inertia_matrix(inertia)
    step = _positive(dt, "dt")
    sub = require_non_negative_integer(substeps, name="substeps")
    if sub < 1:
        raise ValueError("substeps must be positive")
    rate = _vec3(initial_rate, "initial_rate")
    extra = _vec3(disturbance, "disturbance")
    inverse = inverse_spd(matrix)
    h = step / sub
    path = [rate]
    for index, raw in enumerate(torques):
        tau = _vec3(raw, "torque")
        total = (tau[0] + extra[0], tau[1] + extra[1], tau[2] + extra[2])
        for _ in range(sub):
            rate = _rk4(matrix, inverse, rate, total, h)
        if not all(math.isfinite(v) for v in rate):
            raise ValueError(f"rate is not finite after interval {index}")
        path.append(rate)
    return tuple(path)


@dataclass(frozen=True, slots=True)
class PrincipalAxes:
    """Principal moments and axes of an inertia tensor.

    Args:
        moments: Principal moments ``I_1 <= I_2 <= I_3`` (kg m^2).
        axes: Principal axes as COLUMNS of a ``3 x 3`` matrix, in the order of
            ``moments``; the sign of each is fixed so that its component of
            largest magnitude is positive.

    Equation:
        [EQ-INV-INERTIA-002].

    References:
        [REF-GOLUB-VANLOAN-2013], [REF-MARKLEY-CRASSIDIS-2014].

    """

    moments: Vector
    axes: Matrix

    def axis(self, index: int) -> Vector:
        """Return principal axis ``index`` (0-based, ascending moment).

        Args:
            index: Axis index in ``{0, 1, 2}``.

        Returns:
            The unit vector.

        Raises:
            ValueError: If the index is out of range.

        References:
            [EQ-INV-INERTIA-002]. No new equation is introduced.

        """
        if not isinstance(index, int) or not 0 <= index < 3:
            raise ValueError("axis index out of range")
        return tuple(row[index] for row in self.axes)


def principal_axes(inertia: Sequence[Sequence[float]]) -> PrincipalAxes:
    """Return the principal moments and axes by symmetric eigen-decomposition.

    Args:
        inertia: Symmetric positive-definite ``3 x 3`` inertia (kg m^2).

    Returns:
        Ascending moments and the matching unit axes.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the inertia is not symmetric positive definite.

    Equation:
        [EQ-INV-INERTIA-002].

    References:
        [REF-GOLUB-VANLOAN-2013], [REF-MARKLEY-CRASSIDIS-2014].

    """
    values, vectors = symmetric_eigen(_inertia_matrix(inertia))
    columns: list[Vector] = []
    for column in transpose(vectors):
        pivot = max(column, key=abs)
        columns.append(tuple(-v for v in column) if pivot < 0.0 else column)
    return PrincipalAxes(values, transpose(tuple(columns)))


@dataclass(frozen=True, slots=True)
class RealizabilityCheck:
    """Physical-realizability diagnostics of an inertia tensor.

    Args:
        positive_definite: Whether all principal moments are positive.
        moments: Principal moments (ascending).
        triangle_margins: ``(I_1 + I_2 - I_3, I_1 + I_3 - I_2, I_2 + I_3 -
            I_1)`` divided by ``I_1 + I_2 + I_3``; all are nonnegative for a
            realizable rigid body.
        realizable: ``positive_definite`` and every margin ``>= 0``.

    Equation:
        [EQ-INV-INERTIA-002].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """

    positive_definite: bool
    moments: Vector
    triangle_margins: Vector
    realizable: bool


def check_realizable(inertia: Sequence[Sequence[float]]) -> RealizabilityCheck:
    """Check that a symmetric matrix can be the inertia of a rigid body.

    A body has ``I_a = integral rho (r^2 - r_a^2) dV``, hence principal
    moments that are positive and satisfy the triangle inequalities
    ``I_a + I_b >= I_c``; a positive-definite matrix that violates them is
    not the inertia of any mass distribution.

    Args:
        inertia: Symmetric ``3 x 3`` matrix (kg m^2); it need not be positive
            definite.

    Returns:
        The diagnostics.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is not a finite symmetric ``3 x 3`` matrix.

    Equation:
        [EQ-INV-INERTIA-002].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """
    matrix = as_matrix(inertia, name="inertia", square=True)
    if len(matrix) != 3:
        raise ValueError("inertia must be a 3 x 3 matrix")
    scale = max(abs(v) for row in matrix for v in row) or 1.0
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(matrix[i][j] - matrix[j][i]) > _SYMMETRY_TOLERANCE * scale:
                raise ValueError("inertia must be symmetric")
    sym = tuple(
        tuple(0.5 * (matrix[i][j] + matrix[j][i]) for j in range(3)) for i in range(3)
    )
    moments, _ = symmetric_eigen(sym)
    total = math.fsum(moments)
    margins = (
        (moments[0] + moments[1] - moments[2]) / total,
        (moments[0] + moments[2] - moments[1]) / total,
        (moments[1] + moments[2] - moments[0]) / total,
    )
    positive = moments[0] > 0.0
    return RealizabilityCheck(
        positive, moments, margins, positive and min(margins) >= 0.0
    )


def inertia_from_theta(theta: Sequence[float]) -> Matrix:
    """Return ``J = L L'`` for ``theta = (a_1, a_2, a_3, l_21, l_31, l_32)``.

    ``L`` is lower triangular with ``L_ii = exp(a_i)``, ``L_21 = l_21``,
    ``L_31 = l_31`` and ``L_32 = l_32``. Only the first six entries of
    ``theta`` are used, so a nine-parameter vector (with disturbance) is
    accepted.

    Args:
        theta: At least six finite reals.

    Returns:
        The symmetric positive-definite inertia (kg m^2).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than six entries are given or the exponentials
            overflow.

    Equation:
        [EQ-INV-INERTIA-002].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    t = as_vector(theta, name="theta")
    if len(t) < 6:
        raise ValueError("theta needs at least six entries")
    try:
        d = (math.exp(t[0]), math.exp(t[1]), math.exp(t[2]))
    except OverflowError as error:
        raise ValueError("theta gives an overflowing inertia") from error
    low = ((d[0], 0.0, 0.0), (t[3], d[1], 0.0), (t[4], t[5], d[2]))
    full = matmul(low, transpose(low))
    if not all(math.isfinite(v) for row in full for v in row):
        raise ValueError("theta gives a non-finite inertia")
    return full


def theta_from_inertia(
    inertia: Sequence[Sequence[float]], disturbance: Sequence[float] | None = None
) -> Vector:
    """Return ``theta`` (optionally followed by ``d``) for an inertia tensor.

    Inverse of :func:`inertia_from_theta` through the Cholesky factor.

    Args:
        inertia: Symmetric positive-definite ``3 x 3`` inertia (kg m^2).
        disturbance: Optional disturbance torque appended to ``theta``.

    Returns:
        Six (or nine) parameters.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the inertia is not symmetric positive definite.

    Equation:
        [EQ-INV-INERTIA-002].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    low = cholesky(_inertia_matrix(inertia))
    theta = (
        math.log(low[0][0]),
        math.log(low[1][1]),
        math.log(low[2][2]),
        low[1][0],
        low[2][0],
        low[2][1],
    )
    if disturbance is None:
        return theta
    return theta + _vec3(disturbance, "disturbance")


def inertia_entries(inertia: Sequence[Sequence[float]]) -> Vector:
    """Return ``(J_xx, J_yy, J_zz, J_xy, J_xz, J_yz)``.

    Args:
        inertia: Symmetric ``3 x 3`` matrix (kg m^2).

    Returns:
        The six distinct entries.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrix is not ``3 x 3``.

    Equation:
        [EQ-INV-INERTIA-003].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """
    m = as_matrix(inertia, name="inertia", square=True)
    if len(m) != 3:
        raise ValueError("inertia must be a 3 x 3 matrix")
    return (m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2])


def _matrix_from_entries(entries: Vector) -> Matrix:
    """Return the symmetric matrix of ``(Jxx, Jyy, Jzz, Jxy, Jxz, Jyz)``.

    References:
        [EQ-INV-INERTIA-003]. No new equation is introduced.

    """
    xx, yy, zz, xy, xz, yz = entries
    return ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))


def _entry_jacobian(theta: Vector) -> Matrix:
    """Return ``d j / d theta`` (``6 x len(theta)``; zero in the ``d`` columns).

    ``dJ = dL L' + L dL'`` with ``dL`` the elementary matrix of the
    parameter (``L_ii e_i e_i'`` for ``a_i``, ``e_i e_j'`` for ``l_ij``).

    References:
        [EQ-INV-INERTIA-006]. The derivative of ``L L'`` is elementary.

    """
    d = tuple(math.exp(a) for a in theta[:3])
    low = ((d[0], 0.0, 0.0), (theta[3], d[1], 0.0), (theta[4], theta[5], d[2]))
    positions = ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))
    scales = (d[0], d[1], d[2], 1.0, 1.0, 1.0)
    columns: list[Vector] = []
    for (row, col), scale in zip(positions, scales, strict=True):
        dj = [[0.0] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                value = 0.0
                if r == row:
                    value += scale * low[c][col]
                if c == row:
                    value += low[r][col] * scale
                dj[r][c] = value
        columns.append(inertia_entries(dj))
    jac = [list(columns[p][i] for p in range(6)) for i in range(6)]
    for row_values in jac:
        row_values.extend([0.0] * (len(theta) - 6))
    return tuple(tuple(r) for r in jac)


@dataclass(frozen=True, slots=True)
class InertiaData:
    """Gyro samples of a torque manoeuvre.

    Args:
        dt: Sample interval (s).
        torques: Known torque per interval (N m); ``torques[k]`` is held on
            ``[t_k, t_(k+1)]``.
        rates: Gyro measurements ``y_k`` of the rate at ``t_k = k dt`` for
            ``k = 1 .. N`` (rad/s).
        initial_rate: Known rate at ``t_0`` (rad/s).
        noise_std: Declared white-noise standard deviation ``sigma`` (rad/s)
            of every gyro component.
        substeps: RK4 steps per interval used by the forward model.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the lengths disagree, a vector is not three-dimensional,
            or ``dt``, ``noise_std`` or ``substeps`` is not positive.

    Equation:
        [EQ-INV-INERTIA-005].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """

    dt: float
    torques: tuple[Vector, ...]
    rates: tuple[Vector, ...]
    initial_rate: Vector
    noise_std: float
    substeps: int = 1

    def __post_init__(self) -> None:
        """Validate the sample set.

        Raises:
            TypeError: If an entry is not real.
            ValueError: For inconsistent or invalid fields.

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        _positive(self.dt, "dt")
        _positive(self.noise_std, "noise_std")
        if (
            isinstance(self.substeps, bool)
            or not isinstance(self.substeps, int)
            or self.substeps < 1
        ):
            raise ValueError("substeps must be a positive integer")
        _vec3(self.initial_rate, "initial_rate")
        if len(self.torques) < 3 or len(self.torques) != len(self.rates):
            raise ValueError("need at least three intervals with one sample each")
        for row in self.torques:
            _vec3(row, "torque")
        for row in self.rates:
            _vec3(row, "rate")

    @property
    def samples(self) -> int:
        """Return the number ``N`` of rate samples.

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        return len(self.rates)


def simulate_inertia_data(
    inertia: Sequence[Sequence[float]],
    torques: Sequence[Sequence[float]],
    dt: float,
    *,
    noise_std: float,
    initial_rate: Sequence[float] = _ZERO3,
    disturbance: Sequence[float] = _ZERO3,
    substeps: int = 1,
    declared_noise_std: float | None = None,
    seed: int | RandomStream | None = None,
) -> InertiaData:
    """Simulate noisy gyro samples of a manoeuvre (synthetic data).

    Args:
        inertia: True inertia (kg m^2).
        torques: Applied torques (N m), one per interval.
        dt: Sample interval (s).
        noise_std: Standard deviation of the noise actually added (rad/s);
            zero gives noise-free samples.
        initial_rate: Rate at ``t_0`` (rad/s).
        disturbance: True constant disturbance torque (N m).
        substeps: RK4 steps per interval.
        declared_noise_std: Value stored as the likelihood ``sigma`` (default
            ``noise_std``; required when ``noise_std`` is zero).
        seed: Seed or stream for the noise.

    Returns:
        The data set.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the inputs are invalid or no positive declared noise
            level is available.

    Equation:
        [EQ-INV-INERTIA-005].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """
    sigma = require_non_negative_finite_real(noise_std, name="noise_std")
    declared = sigma if declared_noise_std is None else declared_noise_std
    path = simulate_rates(
        inertia,
        torques,
        dt,
        initial_rate=initial_rate,
        disturbance=disturbance,
        substeps=substeps,
    )
    generator = resolve_stream(seed).child("inertia-data").generator()
    measured = tuple(
        tuple(v + generator.gauss(0.0, sigma) if sigma > 0.0 else v for v in row)
        for row in path[1:]
    )
    return InertiaData(
        dt,
        tuple(_vec3(t, "torque") for t in torques),
        measured,
        _vec3(initial_rate, "initial_rate"),
        float(declared),
        substeps,
    )


def excitation_torques(
    kind: Literal["single_axis", "sequential", "multi_axis"],
    *,
    steps: int,
    amplitude: float,
    pulse_steps: int = 10,
) -> tuple[Vector, ...]:
    """Return a pulse-train torque sequence of one excitation family.

    The sequence is made of blocks of ``4 * pulse_steps`` intervals holding
    ``+A u, 0, -A u, 0`` for a direction ``u`` with ``|u| = 1``: the reversal
    returns the rate towards rest and separates a constant disturbance from
    the applied torque. ``single_axis`` uses ``u = e_x`` in every block,
    ``sequential`` cycles ``e_x, e_y, e_z``, and ``multi_axis`` cycles four
    simultaneous directions ``(+1, +1, +1), (+1, -1, +1), (-1, +1, +1),
    (+1, +1, -1)`` divided by ``sqrt 3``. All families have the same peak
    torque norm ``A``.

    Args:
        kind: Excitation family.
        steps: Number of intervals ``N``.
        amplitude: Peak torque norm ``A`` (N m).
        pulse_steps: Intervals per pulse.

    Returns:
        ``N`` torque vectors.

    Raises:
        ValueError: If ``kind`` is unknown or a count or amplitude is invalid.

    Equation:
        [EQ-INV-INERTIA-008].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """
    if kind not in EXCITATION_KINDS:
        raise ValueError(f"kind must be one of {EXCITATION_KINDS}")
    n = require_non_negative_integer(steps, name="steps")
    p = require_non_negative_integer(pulse_steps, name="pulse_steps")
    if n < 1 or p < 1:
        raise ValueError("steps and pulse_steps must be positive")
    level = _positive(amplitude, "amplitude")
    root3 = math.sqrt(3.0)
    multi = (
        (1.0 / root3, 1.0 / root3, 1.0 / root3),
        (1.0 / root3, -1.0 / root3, 1.0 / root3),
        (-1.0 / root3, 1.0 / root3, 1.0 / root3),
        (1.0 / root3, 1.0 / root3, -1.0 / root3),
    )
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    sign = (1.0, 0.0, -1.0, 0.0)
    out: list[Vector] = []
    for k in range(n):
        block, phase = divmod(k, 4 * p)
        if kind == "single_axis":
            u = axes[0]
        elif kind == "sequential":
            u = axes[block % 3]
        else:
            u = multi[block % 4]
        s = level * sign[phase // p]
        out.append((s * u[0], s * u[1], s * u[2]))
    return tuple(out)


def _entry_matrix(vector: Vector) -> Matrix:
    """Return ``M(v)`` (``3 x 6``) with ``J v = M(v) j``.

    References:
        [EQ-INV-INERTIA-003]. The identity is checked by expanding ``J v``.

    """
    x, y, z = vector
    return (
        (x, 0.0, 0.0, y, z, 0.0),
        (0.0, y, 0.0, x, 0.0, z),
        (0.0, 0.0, z, 0.0, x, y),
    )


def _skew_times(rate: Vector, block: Matrix) -> Matrix:
    """Return ``[w]x B`` for a ``3 x q`` matrix ``B``.

    References:
        [EQ-INV-INERTIA-003]. No new equation is introduced.

    """
    wx, wy, wz = rate
    cols = len(block[0])
    return tuple(
        tuple(
            (-wz * block[1][c] + wy * block[2][c])
            if r == 0
            else (wz * block[0][c] - wx * block[2][c])
            if r == 1
            else (-wy * block[0][c] + wx * block[1][c])
            for c in range(cols)
        )
        for r in range(3)
    )


@dataclass(frozen=True, slots=True)
class RegressionEstimate:
    """Linear-regression estimate of the inertia from Euler's equations.

    Args:
        entries: ``(J_xx, J_yy, J_zz, J_xy, J_xz, J_yz)`` (kg m^2).
        inertia: The symmetric matrix of ``entries`` (not repaired: it may be
            indefinite or unrealizable, see :func:`check_realizable`).
        disturbance: Estimated ``d`` (zeros when not estimated).
        residual_rms: Root mean square of the regression residual (N m).
        method: ``"central"`` or ``"integral"``.
        equations: Number of scalar equations used.

    Equation:
        [EQ-INV-INERTIA-003], [EQ-INV-INERTIA-004].

    References:
        [REF-MARKLEY-CRASSIDIS-2014], [REF-GOLUB-VANLOAN-2013].

    """

    entries: Vector
    inertia: Matrix
    disturbance: Vector
    residual_rms: float
    method: str
    equations: int


def regression_estimate(
    data: InertiaData,
    *,
    method: Literal["central", "integral"] = "central",
    window: int = 4,
    estimate_disturbance: bool = False,
) -> RegressionEstimate:
    """Estimate the inertia by linear least squares on Euler's equations.

    ``central`` uses ``w'(t_k) ~ (y_(k+1) - y_(k-1)) / (2 dt)``, the sample
    ``y_k`` for ``w(t_k)`` and the mean of the two adjacent held torques.
    ``integral`` integrates Euler's equations over non-overlapping windows of
    ``window`` intervals, ``J (y_b - y_a) + trapz([w]x J w) = integral tau``
    (the trapezoidal rule on the samples), which needs no derivative but
    still places noisy samples in the regressors. Both are errors-in-variables
    estimators [EQ-INV-INERTIA-004]: do not expect them to be unbiased at
    finite noise. The system is solved by Householder QR with a rank test; a
    rank-deficient design raises rather than being regularized.

    Args:
        data: Manoeuvre data.
        method: ``"central"`` or ``"integral"``.
        window: Intervals per window for ``integral`` (at least 2).
        estimate_disturbance: Also estimate the constant torque ``d``.

    Returns:
        The estimate.

    Raises:
        ValueError: If the method or window is invalid, there are too few
            equations, or the regression matrix is rank deficient.

    Equation:
        [EQ-INV-INERTIA-003], [EQ-INV-INERTIA-004].

    References:
        [REF-MARKLEY-CRASSIDIS-2014], [REF-GOLUB-VANLOAN-2013].

    """
    if method not in ("central", "integral"):
        raise ValueError("method must be 'central' or 'integral'")
    w = (data.initial_rate, *data.rates)  # w[k] is the sample at t_k
    h = data.dt
    rows: list[Vector] = []
    rhs: list[float] = []

    def add_block(block: Matrix, torque: Vector) -> None:
        """Append one three-equation block, with the ``-d`` columns if used.

        References:
            [EQ-INV-INERTIA-003]. No new equation is introduced.

        """
        for r in range(3):
            row = tuple(block[r])
            if estimate_disturbance:
                row += tuple(-1.0 if c == r else 0.0 for c in range(3))
            rows.append(row)
            rhs.append(torque[r])

    if method == "central":
        for k in range(1, len(w) - 1):
            accel = tuple((w[k + 1][i] - w[k - 1][i]) / (2.0 * h) for i in range(3))
            tau = tuple(
                0.5 * (data.torques[k - 1][i] + data.torques[k][i]) for i in range(3)
            )
            ma = _entry_matrix(accel)
            gy = _skew_times(w[k], _entry_matrix(w[k]))
            add_block(
                tuple(tuple(ma[r][c] + gy[r][c] for c in range(6)) for r in range(3)),
                tau,
            )
    else:
        m = require_non_negative_integer(window, name="window")
        if m < 2:
            raise ValueError("window must be at least 2")
        for a in range(0, len(w) - m, m):
            b = a + m
            span = m * h
            dw = tuple((w[b][i] - w[a][i]) / span for i in range(3))
            gyro = [[0.0] * 6 for _ in range(3)]
            for k in range(a, b):
                g0 = _skew_times(w[k], _entry_matrix(w[k]))
                g1 = _skew_times(w[k + 1], _entry_matrix(w[k + 1]))
                for r in range(3):
                    for c in range(6):
                        gyro[r][c] += 0.5 * h * (g0[r][c] + g1[r][c]) / span
            mdw = _entry_matrix(dw)
            tau = tuple(
                math.fsum(data.torques[k][i] for k in range(a, b)) / m for i in range(3)
            )
            add_block(
                tuple(
                    tuple(mdw[r][c] + gyro[r][c] for c in range(6)) for r in range(3)
                ),
                tau,
            )
    width = 9 if estimate_disturbance else 6
    if len(rows) < width:
        raise ValueError("too few equations for the unknowns")
    factor = householder_qr(tuple(rows))
    if not factor.full_rank:
        raise ValueError("the regression matrix is rank deficient")
    solution = factor.solve(tuple(rhs))
    entries = solution[:6]
    extra = solution[6:9] if estimate_disturbance else _ZERO3
    fitted = [math.fsum(r[c] * solution[c] for c in range(width)) for r in rows]
    rms = math.sqrt(
        math.fsum((f - t) ** 2 for f, t in zip(fitted, rhs, strict=True)) / len(rows)
    )
    return RegressionEstimate(
        entries, _matrix_from_entries(entries), extra, rms, method, len(rows)
    )


class InertiaProblem:
    """Output-error inertia identification with an optional Gaussian prior.

    The parameter vector is ``theta = (a_1, a_2, a_3, l_21, l_31, l_32)``
    [EQ-INV-INERTIA-002] followed, when ``estimate_disturbance`` is true, by
    the disturbance torque ``d`` (N m). The whitened residual vector stacks
    ``(y_k - w_k(theta)) / sigma`` for ``k = 1 .. N`` and, if a prior is
    given, ``(theta_i - m_i) / s_i`` [EQ-INV-INERTIA-005].

    Args:
        data: Manoeuvre data.
        estimate_disturbance: Include ``d`` in the parameters (otherwise
            ``d = 0``).
        prior_mean: Prior mean ``m`` of ``theta`` (with ``d`` if estimated);
            give together with ``prior_std`` or not at all.
        prior_std: Prior standard deviations ``s`` (positive).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If only one of the prior arguments is given or the prior
            has the wrong length or a non-positive scale.

    Equation:
        [EQ-INV-INERTIA-005].

    References:
        [REF-NOCEDAL-WRIGHT-2006], [REF-TARANTOLA-2005].

    """

    __slots__ = ("_data", "_dist", "_mean", "_std")

    def __init__(
        self,
        data: InertiaData,
        *,
        estimate_disturbance: bool = False,
        prior_mean: Sequence[float] | None = None,
        prior_std: Sequence[float] | None = None,
    ) -> None:
        """Validate and store the problem.

        Raises:
            TypeError: If an entry is not real.
            ValueError: For an invalid or incomplete prior.

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        self._data = data
        self._dist = bool(estimate_disturbance)
        count = 9 if self._dist else 6
        if (prior_mean is None) != (prior_std is None):
            raise ValueError("prior_mean and prior_std must be given together")
        if prior_mean is None or prior_std is None:
            self._mean: Vector | None = None
            self._std: Vector | None = None
            return
        mean = as_vector(prior_mean, name="prior_mean")
        std = as_vector(prior_std, name="prior_std")
        if len(mean) != count or len(std) != count:
            raise ValueError(f"the prior needs {count} entries")
        if any(s <= 0.0 for s in std):
            raise ValueError("prior_std must be positive")
        self._mean, self._std = mean, std

    @property
    def data(self) -> InertiaData:
        """Return the data set.

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        return self._data

    @property
    def dimension(self) -> int:
        """Return the number of parameters (6, or 9 with the disturbance).

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        return 9 if self._dist else 6

    @property
    def estimates_disturbance(self) -> bool:
        """Return whether ``d`` is a parameter.

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        return self._dist

    @property
    def prior_mean(self) -> Vector | None:
        """Return the prior mean (``None`` without a prior).

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        return self._mean

    @property
    def prior_std(self) -> Vector | None:
        """Return the prior standard deviations (``None`` without a prior).

        References:
            [EQ-INV-INERTIA-005]. No new equation is introduced.

        """
        return self._std

    def predicted_rates(self, theta: Sequence[float]) -> tuple[Vector, ...]:
        """Return the integrated rates at ``t_1 .. t_N`` for ``theta``.

        Args:
            theta: Parameter vector of length :attr:`dimension`.

        Returns:
            ``N`` predicted rates.

        Raises:
            ValueError: If ``theta`` has the wrong length or the integration
                fails.

        Equation:
            [EQ-INV-INERTIA-001].

        References:
            [REF-MARKLEY-CRASSIDIS-2014].

        """
        t = as_vector(theta, name="theta")
        if len(t) != self.dimension:
            raise ValueError(f"theta must have {self.dimension} entries")
        extra = t[6:9] if self._dist else _ZERO3
        path = simulate_rates(
            inertia_from_theta(t),
            self._data.torques,
            self._data.dt,
            initial_rate=self._data.initial_rate,
            disturbance=extra,
            substeps=self._data.substeps,
        )
        return path[1:]

    def data_residuals(self, theta: Sequence[float]) -> Vector:
        """Return the whitened data residuals ``(y - w(theta)) / sigma``.

        Args:
            theta: Parameter vector.

        Returns:
            ``3 N`` residuals.

        Raises:
            ValueError: As :meth:`predicted_rates`.

        Equation:
            [EQ-INV-INERTIA-005].

        References:
            [REF-NOCEDAL-WRIGHT-2006].

        """
        sigma = self._data.noise_std
        return tuple(
            (y[i] - p[i]) / sigma
            for y, p in zip(self._data.rates, self.predicted_rates(theta), strict=True)
            for i in range(3)
        )

    def residuals(self, theta: Sequence[float]) -> Vector:
        """Return the data residuals followed by the whitened prior residuals.

        Args:
            theta: Parameter vector.

        Returns:
            The stacked residual vector.

        Raises:
            ValueError: As :meth:`predicted_rates`.

        Equation:
            [EQ-INV-INERTIA-005].

        References:
            [REF-NOCEDAL-WRIGHT-2006].

        """
        out = self.data_residuals(theta)
        if self._mean is None or self._std is None:
            return out
        t = as_vector(theta, name="theta")
        return out + tuple(
            (a - m) / s for a, m, s in zip(t, self._mean, self._std, strict=True)
        )


def weak_prior(
    nominal_inertia: Sequence[Sequence[float]],
    *,
    log_std: float = 0.5,
    offdiagonal_fraction: float = 0.5,
    disturbance_std: float | None = None,
) -> tuple[Vector, Vector]:
    """Return a broad Gaussian prior of ``theta`` around a nominal inertia.

    The prior is a regularizer for the Laplace posterior, not a belief about
    a particular vehicle: the diagonal parameters ``a_i`` get standard
    deviation ``log_std`` (a factor ``exp(log_std)`` on the Cholesky
    diagonal) and the off-diagonal parameters ``offdiagonal_fraction`` times
    the root of the mean nominal diagonal. If ``disturbance_std`` is given the
    prior of ``d`` is ``N(0, disturbance_std^2 I)``.

    Args:
        nominal_inertia: Nominal inertia (kg m^2).
        log_std: Standard deviation of ``a_i``.
        offdiagonal_fraction: Relative standard deviation of ``l_ij``.
        disturbance_std: Prior standard deviation of ``d`` (N m), or ``None``
            to omit the disturbance.

    Returns:
        ``(mean, std)`` of length 6 (or 9).

    Raises:
        TypeError: If an entry is not real.
        ValueError: If a scale is not positive or the inertia is invalid.

    Equation:
        [EQ-INV-INERTIA-005].

    References:
        [REF-TARANTOLA-2005].

    """
    matrix = _inertia_matrix(nominal_inertia)
    s_log = _positive(log_std, "log_std")
    frac = _positive(offdiagonal_fraction, "offdiagonal_fraction")
    scale = frac * math.sqrt((matrix[0][0] + matrix[1][1] + matrix[2][2]) / 3.0)
    if disturbance_std is None:
        return theta_from_inertia(matrix), (s_log, s_log, s_log, scale, scale, scale)
    s_d = _positive(disturbance_std, "disturbance_std")
    return (
        theta_from_inertia(matrix, _ZERO3),
        (s_log, s_log, s_log, scale, scale, scale, s_d, s_d, s_d),
    )


def fisher_information(problem: InertiaProblem, theta: Sequence[float]) -> Matrix:
    """Return the Gauss-Newton information ``F = J_r' J_r`` of the data only.

    ``J_r`` is the central-difference Jacobian of the whitened data residuals,
    so ``F`` is the Fisher information of the Gaussian output-error model
    (the prior is not included).

    Args:
        problem: The problem (its prior is ignored).
        theta: Evaluation point.

    Returns:
        The ``p x p`` information matrix.

    Raises:
        ValueError: If the integration fails at ``theta``.

    Equation:
        [EQ-INV-INERTIA-008].

    References:
        [REF-NOCEDAL-WRIGHT-2006], [REF-TARANTOLA-2005].

    """
    jac, _ = central_jacobian(problem.data_residuals, as_vector(theta, name="theta"))
    return matmul(transpose(jac), jac)


@dataclass(frozen=True, slots=True)
class InertiaLaplace:
    """MAP estimate and Laplace posterior of an inertia-identification problem.

    Args:
        problem: The problem.
        theta_map: MAP parameters.
        inertia_map: ``J(theta_map)`` (kg m^2).
        disturbance_map: ``d`` at the MAP (zeros if not estimated).
        covariance_theta: ``(J_r' J_r)^-1`` with the Jacobian of the stacked
            residuals (``None`` if the Gauss-Newton Hessian is not positive
            definite, for example an unidentifiable parameter without a prior).
        cost: ``0.5 ||r||^2`` at the MAP.
        iterations: Levenberg-Marquardt iterations.
        converged: Optimizer tolerance met and the Hessian positive definite.
        message: Optimizer termination reason.

    Equation:
        [EQ-INV-INERTIA-005], [EQ-INV-INERTIA-006].

    References:
        [REF-NOCEDAL-WRIGHT-2006], [REF-TARANTOLA-2005].

    """

    problem: InertiaProblem
    theta_map: Vector
    inertia_map: Matrix
    disturbance_map: Vector
    covariance_theta: Matrix | None
    cost: float
    iterations: int
    converged: bool
    message: str

    def entry_covariance(self) -> Matrix:
        """Return the covariance of ``(Jxx, Jyy, Jzz, Jxy, Jxz, Jyz)``.

        Delta method with the exact derivative of ``L L'``:
        ``C_j = G C_theta G'``.

        Returns:
            The ``6 x 6`` covariance (kg^2 m^4).

        Raises:
            ValueError: If the Laplace covariance is unavailable.

        Equation:
            [EQ-INV-INERTIA-006].

        References:
            [REF-TARANTOLA-2005].

        """
        if self.covariance_theta is None:
            raise ValueError("the Laplace covariance is unavailable")
        g = _entry_jacobian(self.theta_map)
        return matmul(matmul(g, self.covariance_theta), transpose(g))

    def entry_std(self) -> dict[str, float]:
        """Return the posterior standard deviation of each distinct entry.

        Returns:
            A mapping from ``"Jxx" ... "Jyz"`` to the standard deviation.

        Raises:
            ValueError: If the Laplace covariance is unavailable.

        Equation:
            [EQ-INV-INERTIA-006].

        References:
            [REF-TARANTOLA-2005].

        """
        cov = self.entry_covariance()
        return {n: math.sqrt(cov[i][i]) for i, n in enumerate(_ENTRY_NAMES)}

    def sample_physical(
        self, count: int, *, seed: int | RandomStream | None = None
    ) -> InertiaPosterior:
        """Sample the Laplace Gaussian and map to moments and axis angles.

        Draws ``theta ~ N(theta_map, C_theta)``, forms ``J = L L'`` (positive
        definite by construction), rejects draws whose principal moments
        violate the triangle inequalities, and records the ascending moments
        and the angle of each principal axis to the MAP axis of the same rank
        [EQ-INV-INERTIA-007].

        Args:
            count: Number of draws (before rejection).
            seed: Seed or stream.

        Returns:
            The accepted samples and the number rejected.

        Raises:
            ValueError: If ``count`` is not positive or the covariance is
                unavailable or not positive definite.

        Equation:
            [EQ-INV-INERTIA-007].

        References:
            [REF-GOLUB-VANLOAN-2013], [REF-MARKLEY-CRASSIDIS-2014].

        """
        n = require_non_negative_integer(count, name="count")
        if n < 1:
            raise ValueError("count must be positive")
        if self.covariance_theta is None:
            raise ValueError("the Laplace covariance is unavailable")
        factor = cholesky(self.covariance_theta)
        reference = principal_axes(self.inertia_map)
        generator: random.Random = (
            resolve_stream(seed).child("inertia-posterior").generator()
        )
        dim = len(self.theta_map)
        moments: list[Vector] = []
        angles: list[Vector] = []
        rejected = 0
        for _ in range(n):
            z = [generator.gauss(0.0, 1.0) for _ in range(dim)]
            theta = tuple(
                self.theta_map[i] + math.fsum(factor[i][j] * z[j] for j in range(i + 1))
                for i in range(dim)
            )
            try:
                sample = inertia_from_theta(theta)
                report = check_realizable(sample)
            except ValueError:
                rejected += 1
                continue
            if not report.realizable:
                rejected += 1
                continue
            axes = principal_axes(sample)
            moments.append(axes.moments)
            angles.append(
                tuple(
                    math.acos(
                        min(
                            1.0,
                            abs(
                                math.fsum(
                                    a * b
                                    for a, b in zip(
                                        axes.axis(i), reference.axis(i), strict=True
                                    )
                                )
                            ),
                        )
                    )
                    for i in range(3)
                )
            )
        return InertiaPosterior(tuple(moments), tuple(angles), len(moments), rejected)


def _quantile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated ``q``-quantile of a sample.

    References:
        [EQ-INV-INERTIA-007]. No new equation is introduced.

    """
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


@dataclass(frozen=True, slots=True)
class InertiaPosterior:
    """Samples of the principal moments and axis angles.

    Args:
        moments: Ascending principal moments of each accepted sample (kg m^2).
        axis_angles: Angle (rad) between each sampled principal axis and the
            MAP axis of the same rank, per accepted sample.
        accepted: Number of physically realizable draws.
        rejected: Number of draws rejected as unrealizable.

    Equation:
        [EQ-INV-INERTIA-007].

    References:
        [REF-MARKLEY-CRASSIDIS-2014].

    """

    moments: tuple[Vector, ...]
    axis_angles: tuple[Vector, ...]
    accepted: int
    rejected: int

    @property
    def rejection_rate(self) -> float:
        """Return the fraction of draws rejected as unrealizable.

        References:
            [EQ-INV-INERTIA-007]. No new equation is introduced.

        """
        total = self.accepted + self.rejected
        return self.rejected / total

    def moment_interval(self, index: int, level: float = 0.95) -> tuple[float, float]:
        """Return the equal-tailed interval of principal moment ``index``.

        Args:
            index: Moment index (0 smallest).
            level: Coverage in ``(0, 1)``.

        Returns:
            Lower and upper quantiles (kg m^2).

        Raises:
            ValueError: If the index or level is invalid or no sample exists.

        Equation:
            [EQ-INV-INERTIA-007].

        References:
            [REF-MARKLEY-CRASSIDIS-2014].

        """
        return self._interval(self.moments, index, level)

    def angle_interval(self, index: int, level: float = 0.95) -> tuple[float, float]:
        """Return the equal-tailed interval of the angle of axis ``index``.

        Angles are nonnegative; use the upper quantile as the pointing-error
        bound.

        Args:
            index: Axis index (0 smallest moment).
            level: Coverage in ``(0, 1)``.

        Returns:
            Lower and upper quantiles (rad).

        Raises:
            ValueError: If the index or level is invalid or no sample exists.

        Equation:
            [EQ-INV-INERTIA-007].

        References:
            [REF-MARKLEY-CRASSIDIS-2014].

        """
        return self._interval(self.axis_angles, index, level)

    @staticmethod
    def _interval(
        rows: tuple[Vector, ...], index: int, level: float
    ) -> tuple[float, float]:
        """Return equal-tailed quantiles of one column of ``rows``.

        Raises:
            ValueError: For an invalid index or level or an empty sample.

        References:
            [EQ-INV-INERTIA-007]. No new equation is introduced.

        """
        if not rows:
            raise ValueError("no accepted samples")
        if not isinstance(index, int) or not 0 <= index < 3:
            raise ValueError("index must be 0, 1 or 2")
        lev = require_finite_real(level, name="level")
        if not 0.0 < lev < 1.0:
            raise ValueError("level must lie in (0, 1)")
        column = [row[index] for row in rows]
        tail = 0.5 * (1.0 - lev)
        return _quantile(column, tail), _quantile(column, 1.0 - tail)


def inertia_laplace(
    problem: InertiaProblem,
    initial: Sequence[float],
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> InertiaLaplace:
    """Find the MAP by Levenberg-Marquardt and form the Laplace covariance.

    The Hessian is the Gauss-Newton matrix ``J_r' J_r`` of the stacked
    whitened residuals at the MAP and ``C_theta = (J_r' J_r)^-1``. A local
    optimum is returned. If the Hessian is not positive definite (for example
    parameters the data cannot identify and no prior) ``covariance_theta`` is
    ``None`` and ``converged`` is false; nothing is regularized.

    Args:
        problem: The problem.
        initial: Starting ``theta`` of length ``problem.dimension``.
        max_iterations: Levenberg-Marquardt iteration limit.
        tolerance: Levenberg-Marquardt tolerance.

    Returns:
        MAP, covariance and status.

    Raises:
        ValueError: For invalid options or a start of the wrong length.

    Equation:
        [EQ-INV-INERTIA-005], [EQ-INV-INERTIA-006].

    References:
        [REF-NOCEDAL-WRIGHT-2006], [REF-TARANTOLA-2005].

    """
    steps = require_non_negative_integer(max_iterations, name="max_iterations")
    if steps < 1:
        raise ValueError("max_iterations must be positive")
    start = as_vector(initial, name="initial")
    if len(start) != problem.dimension:
        raise ValueError(f"initial must have {problem.dimension} entries")
    fit = least_squares(
        problem.residuals, start, tolerance=tolerance, max_iterations=steps
    )
    jac = fit.jacobian
    hessian = matmul(transpose(jac), jac)
    covariance: Matrix | None
    try:
        covariance = inverse_spd(hessian)
        spd = True
    except ValueError:
        covariance = None
        spd = False
    theta = fit.x
    return InertiaLaplace(
        problem,
        theta,
        inertia_from_theta(theta),
        theta[6:9] if problem.estimates_disturbance else _ZERO3,
        covariance,
        fit.cost,
        fit.iterations,
        bool(fit.converged and spd),
        fit.message if spd else fit.message + "; Hessian not positive definite",
    )


@dataclass(frozen=True, slots=True)
class ExcitationAssessment:
    """Information content of a torque sequence about the inertia entries.

    Args:
        kind: Label of the sequence.
        d_criterion: ``log det(F + P)`` with ``F`` the Fisher information and
            ``P`` the prior precision (larger is better).
        whitened_eigenvalues: Eigenvalues (ascending) of ``S F S`` with
            ``S = diag(prior std)``: a direction with eigenvalue ``<< 1`` is
            not learned from the data beyond the prior.
        entry_std: Posterior standard deviation of ``(Jxx, Jyy, Jzz, Jxy,
            Jxz, Jyz)`` from ``(F + P)^-1`` (kg m^2).
        prior_entry_std: Prior standard deviation of the same entries.

    Equation:
        [EQ-INV-INERTIA-008].

    References:
        [REF-TARANTOLA-2005], [REF-NOCEDAL-WRIGHT-2006].

    """

    kind: str
    d_criterion: float
    whitened_eigenvalues: Vector
    entry_std: Vector
    prior_entry_std: Vector

    @property
    def shrinkage(self) -> Vector:
        """Return posterior over prior standard deviation of each entry.

        A value near one means the data barely reduce the prior uncertainty
        of that entry.

        References:
            [EQ-INV-INERTIA-008]. No new equation is introduced.

        """
        return tuple(
            p / q for p, q in zip(self.entry_std, self.prior_entry_std, strict=True)
        )


def assess_excitation(
    kind: str,
    inertia: Sequence[Sequence[float]],
    torques: Sequence[Sequence[float]],
    dt: float,
    *,
    noise_std: float,
    prior_mean: Sequence[float],
    prior_std: Sequence[float],
    initial_rate: Sequence[float] = _ZERO3,
    estimate_disturbance: bool = False,
    substeps: int = 1,
) -> ExcitationAssessment:
    """Evaluate the Bayesian D-criterion and entry precision of a torque sequence.

    The Fisher information is computed at the true ``theta`` of ``inertia``
    (and zero disturbance); it does not depend on realized noise.

    Args:
        kind: Label stored in the result.
        inertia: Nominal/true inertia (kg m^2) at which to linearize.
        torques: Torque sequence (N m).
        dt: Sample interval (s).
        noise_std: Gyro noise standard deviation (rad/s).
        prior_mean: Prior mean of ``theta`` (see :func:`weak_prior`).
        prior_std: Prior standard deviations.
        initial_rate: Initial rate (rad/s).
        estimate_disturbance: Include ``d`` among the parameters.
        substeps: RK4 steps per interval.

    Returns:
        The assessment.

    Raises:
        ValueError: If the inputs are invalid or the posterior precision is
            not positive definite.

    Equation:
        [EQ-INV-INERTIA-008].

    References:
        [REF-TARANTOLA-2005], [REF-NOCEDAL-WRIGHT-2006].

    """
    data = simulate_inertia_data(
        inertia,
        torques,
        dt,
        noise_std=0.0,
        declared_noise_std=noise_std,
        initial_rate=initial_rate,
        substeps=substeps,
    )
    problem = InertiaProblem(
        data,
        estimate_disturbance=estimate_disturbance,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )
    theta = theta_from_inertia(inertia, _ZERO3 if estimate_disturbance else None)
    info = fisher_information(problem, theta)
    std = problem.prior_std
    assert std is not None  # set above
    dim = problem.dimension
    precision = tuple(
        tuple(info[i][j] + (1.0 / std[i] ** 2 if i == j else 0.0) for j in range(dim))
        for i in range(dim)
    )
    cov = inverse_spd(precision)
    low = cholesky(precision)
    logdet = 2.0 * math.fsum(math.log(low[i][i]) for i in range(dim))
    whitened = symmetric_eigen(
        tuple(
            tuple(std[i] * info[i][j] * std[j] for j in range(dim)) for i in range(dim)
        )
    )[0]
    g = _entry_jacobian(theta)
    post = matmul(matmul(g, cov), transpose(g))
    prior_cov = matmul(
        matmul(
            g,
            tuple(
                tuple(std[i] ** 2 if i == j else 0.0 for j in range(dim))
                for i in range(dim)
            ),
        ),
        transpose(g),
    )
    return ExcitationAssessment(
        kind,
        logdet,
        whitened,
        tuple(math.sqrt(post[i][i]) for i in range(6)),
        tuple(math.sqrt(prior_cov[i][i]) for i in range(6)),
    )
