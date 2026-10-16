"""Gaussian-process regression with ARD kernels and marginal-likelihood fitting.

A Gaussian process (GP) prior ``f ~ GP(0, k)`` conditioned on noisy training
data gives a Gaussian predictive law whose mean interpolates (or smooths) the
data and whose variance grows away from it. Outputs are centered and scaled
to unit sample variance before fitting; the reported hyperparameters and
predictions are transformed back to the original units.

The ``nugget`` is an explicit noise variance (in standardized output units)
added to the kernel diagonal. It regularizes nearly singular kernel matrices
for deterministic simulators and is never added silently; a nugget that is
too small for the data raises a clear error instead of being enlarged.

Hyperparameter regularization is opt-in (``regularization=``, see
:class:`GaussianProcessRegularization` and ADR-0054): it bounds the log length
scales relative to the spacing and extent of the training inputs, bounds the
signal variance, optionally adds a log-normal prior on the length scales, and
keeps the diagonal at least a fixed fraction of the signal variance. Without
it the fit is the plain type-II maximum likelihood estimate in the box
``|theta| <= 25``. Both report which parameters ended on a bound.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from operator import mul

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    solve_lower,
)
from probstat.modeling import ProbabilisticModel
from probstat.optimization import minimize_bfgs
from probstat.propagation import make_design

KERNELS = ("squared_exponential", "matern52")
_SQRT5 = math.sqrt(5.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_PARAMETER_BOUND = 25.0
_BOUND_TOLERANCE = 0.01
_CONDITION_WARNING = 1e12
_MAX_ADAPTIVE_NUGGET = 1e-4


def _fsum_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return ``sum l_i r_i`` with ``math.fsum`` (fallback for Python 3.11).

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=False))


# Python 3.12+ ``math.sumprod`` accumulates in extended precision in C.
_sumprod: Callable[[Sequence[float], Sequence[float]], float] = getattr(
    math, "sumprod", _fsum_product
)


class GaussianProcessConditioningWarning(UserWarning):
    """Warn that the fitted kernel matrix is ill-conditioned.

    Raised when the 1-norm condition number of the final kernel matrix
    exceeds ``1e12``: predictions and likelihood values then carry rounding
    noise of relative size ``cond(K) eps``; a larger nugget or
    :class:`GaussianProcessRegularization` bounds the condition number.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.6].

    """


class GaussianProcessJitterWarning(UserWarning):
    """Warn that the nugget was increased to obtain a positive-definite matrix.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

    """


@dataclass(frozen=True, slots=True)
class GaussianProcessRegularization:
    """Bounds, prior, and diagonal floor for GP hyperparameter estimation.

    With per-input training extent ``E_i = max x_i - min x_i`` and smallest
    positive gap ``h_i`` between sorted distinct coordinates of input ``i``,
    the log parameters are confined to

    * ``log ell_i in [log(a h_i), log(b E_i)]``: below the smallest gap all
      off-diagonal correlations vanish and the likelihood is flat in
      ``ell_i`` (a white-noise fit with no information about ``ell_i``);
      far above the extent the kernel matrix tends to a singular
      low-order-polynomial limit whose likelihood keeps rising while the
      posterior variance falls below rounding resolution;
    * ``log sigma_f^2 in [log s_lo, log s_hi]`` in standardized output units
      (the data have unit sample variance);
    * ``log sigma_n^2 in [-25, log s_hi]`` when noise is estimated.

    The diagonal of the kernel matrix is ``sigma_f^2 (1 + tau) + nugget
    [+ sigma_n^2]`` with ``tau = min_relative_nugget``: a noise floor
    proportional to the signal variance bounds the condition number of the
    correlation part by ``1 + n / tau`` whatever the length scales. With
    ``length_scale_prior_sd = s`` the objective adds the log-normal prior
    ``log ell_i ~ N(log E_i, s^2)`` (a maximum a posteriori fit). The
    optimizer works on a smooth periodic reparameterization of the box, so a
    maximum on a bound is approached smoothly and is reported in
    :attr:`GaussianProcessRegression.at_bound`.

    The default factors are computational choices, not published constants:
    ``a = 0.1`` keeps the full range the data can inform, ``b = 20`` admits
    near-linear trends across the design, ``[1e-3, 1e3]`` allows signal
    variances three orders of magnitude around the sample variance, and
    ``tau = 1e-8`` (a relative noise standard deviation of ``1e-4``) keeps
    ``1 + n / tau`` below about ``1e11`` for the tens to hundreds of points
    a pure-Python GP handles.

    Args:
        lower_spacing_factor: ``a > 0``.
        upper_extent_factor: ``b > 0``.
        signal_variance_bounds: ``(s_lo, s_hi)`` with ``0 < s_lo < s_hi``.
        min_relative_nugget: ``tau >= 0``.
        length_scale_prior_sd: Log-scale prior standard deviation ``s > 0``,
            or ``None`` for no prior (bounded maximum likelihood).

    Raises:
        ValueError: If a field is out of range.

    Equation:
        [EQ-SUR-GP-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Sections 5.2 and 5.4],
        [REF-ANDRIANAKIS-CHALLENOR-2012].

    """

    lower_spacing_factor: float = 0.1
    upper_extent_factor: float = 20.0
    signal_variance_bounds: tuple[float, float] = (1e-3, 1e3)
    min_relative_nugget: float = 1e-8
    length_scale_prior_sd: float | None = None

    def __post_init__(self) -> None:
        """Validate the regularization settings.

        Raises:
            ValueError: If a field is out of range.

        References:
            Project validation policy. No equation is evaluated.

        """
        for name in ("lower_spacing_factor", "upper_extent_factor"):
            value = require_finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        low, high = (
            require_finite_real(value, name="signal_variance_bounds")
            for value in self.signal_variance_bounds
        )
        if not 0.0 < low < high:
            raise ValueError("signal_variance_bounds must satisfy 0 < low < high")
        tau = require_finite_real(self.min_relative_nugget, name="min_relative_nugget")
        if tau < 0.0:
            raise ValueError("min_relative_nugget must be non-negative")
        if self.length_scale_prior_sd is not None:
            sd = require_finite_real(
                self.length_scale_prior_sd, name="length_scale_prior_sd"
            )
            if sd <= 0.0:
                raise ValueError("length_scale_prior_sd must be positive")


def gaussian_process_log_parameter_bounds(
    points: Sequence[Sequence[float]],
    regularization: GaussianProcessRegularization,
    *,
    estimate_noise: bool = False,
) -> tuple[tuple[float, float], ...]:
    """Return the box for ``theta = (log sigma_f^2, log ell_1..d[, log sigma_n^2])``.

    Length-scale bounds are ``[log(a h_i), log(b E_i)]`` with ``h_i`` the
    smallest positive gap between sorted coordinates of input ``i`` and
    ``E_i`` its extent; all bounds are clipped to ``|theta| <= 25``.

    Args:
        points: Training inputs.
        regularization: Regularization settings.
        estimate_noise: Whether ``theta`` ends with ``log sigma_n^2``.

    Returns:
        One ``(lower, upper)`` pair per log parameter.

    Raises:
        ValueError: If an input is constant across the points.

    Equation:
        [EQ-SUR-GP-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    x = tuple(as_vector(point, name="point") for point in points)
    cap = _LOG_PARAMETER_BOUND
    low, high = regularization.signal_variance_bounds
    bounds = [(max(math.log(low), -cap), min(math.log(high), cap))]
    for axis in range(len(x[0])):
        column = sorted({point[axis] for point in x})
        if len(column) < 2:
            raise ValueError(f"input {axis} is constant across the training points")
        gap = min(b - a for a, b in zip(column, column[1:], strict=False))
        extent = column[-1] - column[0]
        lower = math.log(regularization.lower_spacing_factor * gap)
        upper = math.log(regularization.upper_extent_factor * extent)
        lower, upper = max(lower, -cap), min(upper, cap)
        if lower >= upper:
            raise ValueError(f"length-scale bounds of input {axis} are empty")
        bounds.append((lower, upper))
    if estimate_noise:
        bounds.append((-cap, min(math.log(high), cap)))
    return tuple(bounds)


def _bound_hits(
    theta: Sequence[float], bounds: Sequence[tuple[float, float]], dimension: int
) -> tuple[str, ...]:
    """Name the log parameters within ``0.01`` of a bound.

    References:
        [EQ-SUR-GP-006]. No new equation.

    """
    names = (
        "signal_variance",
        *(f"length_scale[{axis}]" for axis in range(dimension)),
        "noise_variance",
    )
    hits: list[str] = []
    for name, value, (lower, upper) in zip(names, theta, bounds, strict=False):
        if value - lower <= _BOUND_TOLERANCE:
            hits.append(f"{name}:lower")
        elif upper - value <= _BOUND_TOLERANCE:
            hits.append(f"{name}:upper")
    return tuple(hits)


@dataclass(frozen=True, slots=True)
class GaussianProcessHyperparameters:
    """Kernel hyperparameters in original output and input units.

    Args:
        signal_variance: Prior process variance ``sigma_f^2``.
        length_scales: ARD length scales ``ell_i``, one per input.
        noise_variance: Diagonal noise variance (nugget plus any estimated
            noise).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2 and 5].

    """

    signal_variance: float
    length_scales: Vector
    noise_variance: float


def kernel_value(
    kernel: str,
    left: Sequence[float],
    right: Sequence[float],
    length_scales: Sequence[float],
    signal_variance: float,
) -> float:
    """Evaluate a stationary ARD covariance kernel.

    Squared exponential: ``sigma_f^2 exp(-r^2 / 2)``; Matern 5/2:
    ``sigma_f^2 (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)``, with
    ``r^2 = sum_i ((x_i - x'_i) / ell_i)^2``.

    Args:
        kernel: ``"squared_exponential"`` or ``"matern52"``.
        left: First input vector.
        right: Second input vector.
        length_scales: Positive length scales.
        signal_variance: Positive signal variance.

    Returns:
        Covariance ``k(x, x')``.

    Raises:
        ValueError: If the kernel name is unknown.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    squared = math.fsum(
        ((a - b) / ell) ** 2
        for a, b, ell in zip(left, right, length_scales, strict=True)
    )
    return signal_variance * _correlation(kernel, squared)


def _correlation(kernel: str, squared: float) -> float:
    """Return the unit-variance kernel as a function of ``r^2``.

    Raises:
        ValueError: If the kernel name is unknown.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    if kernel == "squared_exponential":
        return math.exp(-0.5 * squared)
    if kernel == "matern52":
        r = math.sqrt(squared)
        return (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * math.exp(-_SQRT5 * r)
    raise ValueError(f"kernel must be one of {KERNELS}")


def _length_scale_factor(kernel: str, squared: float) -> float:
    """Return ``(dk/d log ell_i) / (sigma_f^2 (Delta_i/ell_i)^2)``.

    Squared exponential: ``exp(-r^2/2)``. Matern 5/2:
    ``(5/3)(1 + sqrt(5) r) exp(-sqrt(5) r)``. Both are finite at ``r = 0``.

    Equation:
        [EQ-SUR-GP-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Sections 4.2 and 5.4].

    """
    if kernel == "squared_exponential":
        return math.exp(-0.5 * squared)
    r = math.sqrt(squared)
    return (5.0 / 3.0) * (1.0 + _SQRT5 * r) * math.exp(-_SQRT5 * r)


def _split_parameters(
    log_parameters: Sequence[float], dimension: int, estimate_noise: bool
) -> tuple[float, Vector, float]:
    """Decode ``(log sigma_f^2, log ell_1..d[, log sigma_n^2])``.

    Raises:
        ValueError: If the vector has the wrong length or leaves the
            documented box ``|theta_j| <= 25``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4]. No equation is evaluated.

    """
    theta = as_vector(log_parameters, name="log_parameters")
    expected = 1 + dimension + (1 if estimate_noise else 0)
    if len(theta) != expected:
        raise ValueError(f"log_parameters must have length {expected}")
    if any(abs(value) > _LOG_PARAMETER_BOUND for value in theta):
        raise ValueError("log hyperparameters left the box |theta| <= 25")
    signal = math.exp(theta[0])
    scales = tuple(math.exp(value) for value in theta[1 : 1 + dimension])
    extra = math.exp(theta[-1]) if estimate_noise else 0.0
    return signal, scales, extra


def _fast_cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Return the lower Cholesky factor as row lists (Cholesky-Banachiewicz).

    Bit-identical to :func:`probstat.linalg.cholesky` (each inner product is
    the correctly rounded ``math.fsum`` of the same rounded products) without
    its validation and tuple conversions. The identity matters: the
    likelihood, the fitted surrogate, and AK-MCS's incremental posterior
    (``reliability.active_learning``) must accept or reject the same
    near-singular kernel matrices and produce the same factor.

    Args:
        matrix: Symmetric matrix as row lists (only the lower triangle is read).

    Returns:
        Row lists ``L[i][:i+1]`` of the lower factor.

    Raises:
        ValueError: If a pivot is not positive (matrix not positive definite).

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    count = len(matrix)
    lower: list[list[float]] = []
    for i in range(count):
        row_i = matrix[i]
        current: list[float] = []
        for j in range(i):
            row_j = lower[j]
            current.append((row_i[j] - math.fsum(map(mul, current, row_j))) / row_j[j])
        pivot = row_i[i] - math.fsum(map(mul, current, current))
        if not pivot > 0.0:
            raise ValueError("kernel matrix is not positive definite")
        current.append(math.sqrt(pivot))
        lower.append(current)
    return lower


def _inverse_factor_columns(lower: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the columns of ``M = L^-1`` below the diagonal.

    ``columns[j][k - j] = M[k][j]`` for ``k >= j``, built row by row from
    ``M[i][j] = -(sum_(k=j..i-1) L[i][k] M[k][j]) / L[i][i]``: one
    triangular inversion (``n^3/6`` multiply-adds) of the existing factor.

    Equation:
        [EQ-SUR-GP-007].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    count = len(lower)
    columns: list[list[float]] = []
    for i in range(count):
        row = lower[i]
        inverse_diagonal = 1.0 / row[i]
        for j in range(i):
            column = columns[j]
            column.append(-_sumprod(row[j:i], column) * inverse_diagonal)
        columns.append([inverse_diagonal])
    return columns


def _precision_entry(columns: Sequence[Sequence[float]], a: int, b: int) -> float:
    """Return ``(K^-1)_ab = sum_(k >= max(a, b)) M[k][a] M[k][b]`` (``a <= b``).

    Equation:
        [EQ-SUR-GP-007].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    return _sumprod(columns[a][b - a :], columns[b])


def _triangular_solve(
    lower: Sequence[Sequence[float]], target: Sequence[float]
) -> Vector:
    """Solve ``L w = b`` then ``L' alpha = w`` and return ``alpha = K^-1 b``.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    count = len(lower)
    forward: list[float] = []
    for i in range(count):
        row = lower[i]
        forward.append((target[i] - _sumprod(row[:i], forward)) / row[i])
    backward = [0.0] * count
    for i in range(count - 1, -1, -1):
        total = forward[i]
        for k in range(i + 1, count):
            total -= lower[k][i] * backward[k]
        backward[i] = total / lower[i][i]
    return tuple(backward)


def _assemble(
    points: Sequence[Sequence[float]],
    scales: Sequence[float],
    signal: float,
    noise: float,
    kernel: str,
) -> tuple[
    list[list[float]],
    list[list[tuple[float, ...]]],
    list[list[float]],
    list[list[float]],
]:
    """Assemble the lower triangle of ``K`` and the pieces of its derivatives.

    The squared scaled differences ``((x_ia - x_ja)/ell_a)^2`` are formed
    exactly as in :func:`kernel_value`, so the fitted surrogate factorizes
    bit-for-bit the matrix the likelihood saw.

    Returns:
        Lower-triangle rows of ``K``, per-pair scaled squared differences,
        unit-variance correlations, and the length-scale derivative factors
        of :func:`_length_scale_factor`.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Sections 4.2 and 5.4].

    """
    count = len(points)
    parts: list[list[tuple[float, ...]]] = []
    correlation: list[list[float]] = []
    factors: list[list[float]] = []
    matrix: list[list[float]] = []
    squared_exponential = kernel == "squared_exponential"
    for i in range(count):
        row_parts: list[tuple[float, ...]] = []
        row_corr: list[float] = []
        row_factor: list[float] = []
        row_matrix: list[float] = []
        xi = points[i]
        for j in range(i):
            difference = tuple(
                ((a - b) / ell) ** 2
                for a, b, ell in zip(xi, points[j], scales, strict=True)
            )
            squared = math.fsum(difference)
            if squared_exponential:
                value = math.exp(-0.5 * squared)
                factor = value
            else:
                r = math.sqrt(squared)
                decay = math.exp(-_SQRT5 * r)
                value = (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * decay
                factor = (5.0 / 3.0) * (1.0 + _SQRT5 * r) * decay
            row_parts.append(difference)
            row_corr.append(value)
            row_factor.append(factor)
            row_matrix.append(signal * value)
        row_matrix.append(signal + noise)
        parts.append(row_parts)
        correlation.append(row_corr)
        factors.append(row_factor)
        matrix.append(row_matrix)
    return matrix, parts, correlation, factors


def gaussian_process_log_marginal_likelihood(
    points: Sequence[Sequence[float]],
    targets: Sequence[float],
    log_parameters: Sequence[float],
    *,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_noise: bool = False,
    relative_nugget: float = 0.0,
) -> tuple[float, Vector]:
    """Return the GP log marginal likelihood and its exact gradient.

    ``log p(y | theta) = -1/2 y' K^-1 y - 1/2 log det K - n/2 log 2 pi`` with
    ``K = K_f + (nugget + tau sigma_f^2 + sigma_n^2) I`` (``tau`` the
    relative nugget, ``0`` by default) and gradient
    ``d/d theta_j = 1/2 tr((alpha alpha' - K^-1) dK/d theta_j)``,
    ``alpha = K^-1 y``. Parameters are ``theta = (log sigma_f^2,
    log ell_1..d[, log sigma_n^2])``; targets are used as given (callers
    standardize them).

    Cost: one Cholesky factorization ``K = L L'`` per call, shared by the
    value and the gradient. The traces need every entry of ``K^-1`` because
    each ``dK/d theta_j`` is dense; they are formed as
    ``(K^-1)_ab = sum_k M_ka M_kb`` from one triangular inversion
    ``M = L^-1`` of the same factor [EQ-SUR-GP-007], over the upper
    triangle only, instead of ``n`` unit-vector Cholesky solves.

    Args:
        points: Training inputs.
        targets: Training targets.
        log_parameters: Log hyperparameter vector ``theta``.
        kernel: Kernel name.
        nugget: Fixed positive diagonal variance.
        estimate_noise: Whether ``theta`` ends with ``log sigma_n^2``.
        relative_nugget: Diagonal floor ``tau >= 0`` as a fraction of the
            signal variance (see :class:`GaussianProcessRegularization`).

    Returns:
        Log marginal likelihood and gradient with respect to ``theta``.

    Raises:
        ValueError: If the kernel matrix is not positive definite or the
            parameters leave the documented box.

    Equation:
        [EQ-SUR-GP-004], [EQ-SUR-GP-007].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(targets, name="targets")
    dimension = len(x[0])
    signal, scales, extra = _split_parameters(log_parameters, dimension, estimate_noise)
    tau = require_finite_real(relative_nugget, name="relative_nugget")
    if tau < 0.0:
        raise ValueError("relative_nugget must be non-negative")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    count = len(x)
    noise = nugget + extra
    if tau > 0.0:
        noise += tau * signal
    matrix, parts, correlation, factors = _assemble(x, scales, signal, noise, kernel)
    try:
        lower = _fast_cholesky(matrix)
    except ValueError as error:
        raise ValueError(
            "kernel matrix is not positive definite; increase the nugget"
        ) from error
    alpha = _triangular_solve(lower, y)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
    value = -0.5 * _sumprod(y, alpha)
    value += -0.5 * log_det - 0.5 * count * _LOG_TWO_PI
    columns = _inverse_factor_columns(lower)
    diagonal_weight = [
        alpha[a] * alpha[a] - _sumprod(columns[a], columns[a]) for a in range(count)
    ]
    trace_weight = math.fsum(diagonal_weight)
    signal_sum = 0.0
    axis_sums = [0.0] * dimension
    for b in range(count):
        alpha_b = alpha[b]
        column_b = columns[b]
        row_corr = correlation[b]
        row_factor = factors[b]
        row_parts = parts[b]
        for a in range(b):
            weight = alpha[a] * alpha_b - _sumprod(columns[a][b - a :], column_b)
            signal_sum += weight * row_corr[a]
            scaled_weight = weight * row_factor[a]
            difference = row_parts[a]
            for axis in range(dimension):
                axis_sums[axis] += scaled_weight * difference[axis]
    gradient = [0.5 * signal * trace_weight + signal * signal_sum]
    if tau > 0.0:
        gradient[0] += 0.5 * tau * signal * trace_weight
    gradient.extend(signal * total for total in axis_sums)
    if estimate_noise:
        gradient.append(0.5 * extra * trace_weight)
    return value, tuple(gradient)


@dataclass(frozen=True, slots=True)
class GaussianProcessLOO:
    """Closed-form leave-one-out diagnostics of a fitted GP.

    Args:
        means: LOO predictive means ``mu_(-i)``.
        variances: LOO predictive variances ``sigma^2_(-i)`` (with noise).
        residuals: ``y_i - mu_(-i)``.
        relative_error: ``mean(residual^2) / Var(y)``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """

    means: Vector
    variances: Vector
    residuals: Vector
    relative_error: float


@dataclass(frozen=True)
class GaussianProcessRegression:
    """A fitted Gaussian-process surrogate.

    Args:
        kernel: Kernel name.
        points: Training inputs.
        values: Training outputs in original units.
        hyperparameters: Fitted hyperparameters in original units.
        log_parameters: Fitted ``theta`` in standardized output units.
        nugget: Fixed diagonal variance in standardized output units.
        estimate_noise: Whether a noise variance was estimated.
        output_mean: Training-output sample mean used for centering.
        output_scale: Training-output sample standard deviation used for
            scaling.
        log_marginal_likelihood: ``log p(y | theta)`` in original units.
        converged: Whether the best restart met the optimizer tolerance.
        restarts: Number of optimizer starts attempted.
        evaluations: Total likelihood evaluations (kernel factorizations)
            over all restarts.
        relative_nugget: Diagonal floor ``tau`` as a fraction of the
            standardized signal variance (``0`` without regularization).
        regularization: Settings used, or ``None`` for the unregularized
            maximum likelihood fit.
        log_parameter_bounds: The box searched for ``theta`` (``|theta| <=
            25`` without regularization).
        at_bound: Parameters whose estimate lies within ``0.01`` (log
            units) of a bound, as ``"length_scale[0]:upper"`` and similar;
            empty when the optimum is interior. A length scale on its upper
            bound signals a nearly polynomial (or constant) trend, one on its
            lower bound a fit with almost no correlation between training
            points.
        condition_number: 1-norm condition number of the standardized kernel
            matrix (computed from the fit, not an argument).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2 and 5].

    """

    kernel: str
    points: tuple[Vector, ...]
    values: Vector
    hyperparameters: GaussianProcessHyperparameters
    log_parameters: Vector
    nugget: float
    estimate_noise: bool
    output_mean: float
    output_scale: float
    log_marginal_likelihood: float
    converged: bool
    restarts: int
    evaluations: int
    relative_nugget: float = 0.0
    regularization: GaussianProcessRegularization | None = None
    log_parameter_bounds: tuple[tuple[float, float], ...] = ()
    at_bound: tuple[str, ...] = ()
    _factorization: tuple[float, Vector, float, Matrix, Vector] = field(
        init=False, repr=False, compare=False
    )
    condition_number: float = field(init=False, compare=False, default=math.nan)

    def __post_init__(self) -> None:
        """Factorize the training kernel matrix once for all predictions.

        Also records ``condition_number``, the 1-norm condition number
        ``||K||_1 ||K^-1||_1`` of the standardized kernel matrix, with
        ``K^-1`` formed from the inverse Cholesky factor [EQ-SUR-GP-007].

        Raises:
            ValueError: If the kernel matrix is not positive definite.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1],
            [REF-GOLUB-VANLOAN-2013, Section 2.6].

        """
        factorization = self._standardized()
        object.__setattr__(self, "_factorization", factorization)
        signal, scales, noise, lower, _ = factorization
        rows, _, _, _ = _assemble(self.points, scales, signal, noise, self.kernel)
        object.__setattr__(self, "condition_number", _condition_number(rows, lower))

    def _standardized(self) -> tuple[float, Vector, float, Matrix, Vector]:
        """Return signal, scales, noise, Cholesky factor, and ``alpha``.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1]. No new equation.

        """
        dimension = len(self.points[0])
        signal, scales, extra = _split_parameters(
            self.log_parameters, dimension, self.estimate_noise
        )
        noise = self.nugget + extra
        if self.relative_nugget > 0.0:
            noise += self.relative_nugget * signal
        count = len(self.points)
        rows, _, _, _ = _assemble(self.points, scales, signal, noise, self.kernel)
        try:
            factor = _fast_cholesky(rows)
        except ValueError as error:
            raise ValueError("matrix must be positive definite") from error
        lower = tuple(tuple(factor[i]) + (0.0,) * (count - 1 - i) for i in range(count))
        targets = tuple(
            (value - self.output_mean) / self.output_scale for value in self.values
        )
        return signal, scales, noise, lower, _triangular_solve(factor, targets)

    @property
    def standardized_noise_variance(self) -> float:
        """Diagonal noise variance of the kernel matrix in standardized units.

        ``nugget + tau sigma_f^2 [+ sigma_n^2]``, exactly as factorized.

        References:
            [EQ-SUR-GP-006], [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2].

        """
        return self._factorization[2]

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and variance at one input.

        ``mu = m_y + s_y k_*' alpha`` and
        ``sigma^2 = s_y^2 (k_** - v'v [+ noise])`` with ``v = L^-1 k_*``.
        Tiny negative variances from cancellation are reported as zero. The
        subtraction cannot resolve variances below about ``1e-16 k_**``; with
        very long length scales (nearly flat posteriors) an exact zero means
        "below resolution", so variance-normalized conformal scores need an
        explicit offset (``conformal.conformalize_surrogate(scale_offset=...)``).

        Args:
            x: Input vector.
            include_noise: Add the diagonal noise variance (prediction of a
                new observation rather than of the latent function).

        Returns:
            Predictive mean and variance in original units.

        Equation:
            [EQ-SUR-GP-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        signal, scales, noise, lower, alpha = self._factorization
        cross = tuple(
            kernel_value(self.kernel, point, train, scales, signal)
            for train in self.points
        )
        mean = math.fsum(a * b for a, b in zip(cross, alpha, strict=True))
        whitened = solve_lower(lower, cross)
        variance = signal - math.fsum(v * v for v in whitened)
        if include_noise:
            variance += noise
        return (
            self.output_mean + self.output_scale * mean,
            self.output_scale**2 * max(variance, 0.0),
        )

    def predict(self, x: Sequence[float]) -> float:
        """Return the predictive mean at one input.

        Args:
            x: Input vector.

        Returns:
            Predictive mean in original units.

        Equation:
            [EQ-SUR-GP-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        return self.predict_with_variance(x)[0]

    def leave_one_out(self) -> GaussianProcessLOO:
        """Return closed-form LOO predictions at fixed hyperparameters.

        ``mu_(-i) = y_i - [K^-1 y]_i / [K^-1]_ii`` and
        ``sigma^2_(-i) = 1 / [K^-1]_ii`` (standardized units).

        Returns:
            LOO means, variances, residuals, and relative error.

        Equation:
            [EQ-SUR-GP-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        _, _, _, lower, alpha = self._factorization
        count = len(self.points)
        columns = _inverse_factor_columns(lower)
        means: list[float] = []
        variances: list[float] = []
        residuals: list[float] = []
        for i in range(count):
            diagonal = _sumprod(columns[i], columns[i])
            standardized = (self.values[i] - self.output_mean) / self.output_scale
            loo_mean = standardized - alpha[i] / diagonal
            means.append(self.output_mean + self.output_scale * loo_mean)
            variances.append(self.output_scale**2 / diagonal)
            residuals.append(self.values[i] - means[-1])
        center = math.fsum(self.values) / count
        spread = math.fsum((v - center) ** 2 for v in self.values) / (count - 1)
        return GaussianProcessLOO(
            tuple(means),
            tuple(variances),
            tuple(residuals),
            math.fsum(r * r for r in residuals) / count / spread,
        )


def _select_restart(
    candidates: Sequence[tuple[float, Vector, bool]], noise: float
) -> tuple[float, Vector, bool]:
    """Return the restart result to keep.

    The lowest objective wins, except that the lowest converged run is kept
    when the best overall run did not converge and is lower only by at most
    ``noise`` (the estimated rounding noise of the objective): such a
    difference cannot be resolved, and the converged run carries a verified
    stationarity certificate.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

    """
    best = min(candidates, key=lambda item: item[0])
    if best[2]:
        return best
    converged = [item for item in candidates if item[2]]
    if converged:
        alternative = min(converged, key=lambda item: item[0])
        if alternative[0] <= best[0] + noise:
            return alternative
    return best


def _objective_noise(
    points: Sequence[Vector],
    theta: Vector,
    dimension: int,
    estimate_noise: bool,
    nugget: float,
    tau: float,
    kernel: str,
    value: float,
) -> float:
    """Estimate the rounding noise of ``-log p`` as ``kappa eps (n + |f|)``.

    ``kappa = ||K||_1 ||K^-1||_1`` is the 1-norm condition number at
    ``theta``; perturbation theory for the quadratic form and the
    log-determinant gives errors of order ``kappa eps`` relative to their
    magnitudes. The floor is ``1e-9 max(1, |f|)``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.6]. No equation is evaluated.

    """
    floor = 1e-9 * max(1.0, abs(value))
    try:
        signal, scales, extra = _split_parameters(theta, dimension, estimate_noise)
        rows, _, _, _ = _assemble(
            points, scales, signal, nugget + extra + tau * signal, kernel
        )
        lower = _fast_cholesky(rows)
    except ValueError:
        return floor
    kappa = _condition_number(rows, lower)
    return max(floor, kappa * 2.0**-52 * (len(points) + abs(value)))


def _condition_number(
    rows: Sequence[Sequence[float]], lower: Sequence[Sequence[float]]
) -> float:
    """Return ``||K||_1 ||K^-1||_1`` from the lower triangle and its factor.

    Equation:
        [EQ-SUR-GP-007].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 2.6 and 3.1].

    """
    count = len(rows)
    norm_k = max(
        math.fsum(abs(rows[max(a, b)][min(a, b)]) for b in range(count))
        for a in range(count)
    )
    columns = _inverse_factor_columns(lower)
    norm_inverse = max(
        math.fsum(
            abs(_precision_entry(columns, min(a, b), max(a, b))) for b in range(count)
        )
        for a in range(count)
    )
    return norm_k * norm_inverse


def fit_gaussian_process(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_noise: bool = False,
    optimize: bool = True,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
    initial_log_parameters: Sequence[float] | None = None,
    regularization: GaussianProcessRegularization | None = None,
    adaptive_nugget: bool = True,
) -> GaussianProcessRegression:
    """Fit a zero-mean GP to standardized outputs by maximum marginal likelihood.

    The first start uses ``sigma_f^2 = 1`` (standardized units), length scales
    equal to half of each input's range, and ``sigma_n^2 = 1e-4`` when noise is
    estimated (or ``initial_log_parameters``). Further starts perturb the log
    length scales and log signal variance uniformly by up to ``+/- 1.5``. Each
    start runs BFGS with the exact gradient on ``-log p(y | theta)``; the best
    finite optimum is kept.

    With ``regularization`` the search is confined to the box of
    :func:`gaussian_process_log_parameter_bounds` through the map
    ``theta_j = l_j + (u_j - l_j) sin^2(phi_j / 2)`` (BFGS runs on ``phi``;
    starts are clipped to the inner 98 % of each interval). A maximum on a
    bound is then an ordinary stationary point ``phi_j = 0`` or ``pi``,
    reached at the usual BFGS rate, where a logistic map would push
    ``phi_j`` towards infinity with a vanishing gradient. The BFGS gradient
    tolerance is ``1e-5`` (relative) on this path instead of ``1e-6``: at
    an optimum on a bound of a nearly singular direction, rounding in the
    likelihood gradient kept its norm near ``3e-6`` and the search took 200
    iterations without further progress. The
    diagonal gains ``tau sigma_f^2`` with ``tau = min_relative_nugget``, and
    an optional log-normal length-scale prior is added to the objective
    [EQ-SUR-GP-006]. ``log_marginal_likelihood`` always reports the
    likelihood alone. Parameters that end within ``0.01`` of a bound are
    listed in ``at_bound`` in both modes.

    Args:
        points: Training inputs (at least two distinct points).
        values: Training outputs (not all equal).
        kernel: ``"squared_exponential"`` or ``"matern52"``.
        nugget: Explicit positive diagonal variance in standardized units.
        estimate_noise: Also estimate a noise variance above the nugget.
        optimize: Maximize the likelihood; ``False`` keeps the first start
            (with ``regularization``, the start is first clipped to the inner
            98% of the box, so it can move slightly).
        restarts: Total number of optimizer starts (at least one).
        seed: Integer seed or stream for the restart perturbations.
        initial_log_parameters: Optional first start ``theta``.
        regularization: Optional :class:`GaussianProcessRegularization`;
            ``None`` (default) gives the unregularized fit, unchanged from
            earlier releases (ADR-0054).
        adaptive_nugget: When no start gives a positive-definite matrix,
            retry with the nugget multiplied by ten (up to ``1e-4`` in
            standardized units, or the given nugget if larger). An increase
            is never silent: it emits :class:`GaussianProcessJitterWarning`
            and the nugget actually used is the result's ``nugget``.

    Among restarts the lowest objective wins, but a converged run is kept
    against a non-converged one that is lower only within the estimated
    rounding noise of the objective, ``max(1e-9 max(1, |f|), kappa eps
    (n + |f|))`` with ``kappa`` the 1-norm condition number of ``K`` at the
    lowest objective.

    Each likelihood evaluation shares one Cholesky factorization between the
    value and the gradient (cached per ``theta``); ``evaluations`` counts
    those factorizations. A final kernel matrix with 1-norm condition number
    above ``1e12`` emits :class:`GaussianProcessConditioningWarning`; the
    number is the result's ``condition_number``.

    Returns:
        Fitted regression surrogate.

    Raises:
        TypeError: If ``regularization`` has the wrong type.
        ValueError: If inputs are invalid or no start yields a positive
            definite kernel matrix.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-004], [EQ-SUR-GP-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2, 4, and 5],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if regularization is not None and not isinstance(
        regularization, GaussianProcessRegularization
    ):
        raise TypeError("regularization must be a GaussianProcessRegularization")
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(values, name="values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two points with one value each")
    dimension = len(x[0])
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError("points must be non-empty and equally long")
    jitter = require_finite_real(nugget, name="nugget")
    if jitter <= 0.0:
        raise ValueError("nugget must be positive")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    count = len(y)
    mean = math.fsum(y) / count
    scale = math.sqrt(math.fsum((v - mean) ** 2 for v in y) / (count - 1))
    if scale <= 0.0:
        raise ValueError("values are constant; nothing to regress")
    targets = tuple((v - mean) / scale for v in y)
    ranges = []
    for axis in range(dimension):
        column = [point[axis] for point in x]
        width = max(column) - min(column)
        if width <= 0.0:
            raise ValueError(f"input {axis} is constant across the training points")
        ranges.append(width)
    if initial_log_parameters is not None:
        first = as_vector(initial_log_parameters, name="initial_log_parameters")
    else:
        first = (0.0, *(math.log(0.5 * width) for width in ranges)) + (
            (math.log(1e-4),) if estimate_noise else ()
        )
    _split_parameters(first, dimension, estimate_noise)
    generator = resolve_stream(seed).child("gp-restarts").generator()
    starts = [first]
    for _ in range(restarts - 1 if optimize else 0):
        perturbed = [first[0] + generator.uniform(-1.5, 1.5)]
        perturbed.extend(
            first[1 + axis] + generator.uniform(-1.5, 1.5) for axis in range(dimension)
        )
        if estimate_noise:
            perturbed.append(first[-1])
        starts.append(tuple(perturbed))
    if regularization is None:
        tau = 0.0
        prior_sd: float | None = None
        bounds = tuple(
            (-_LOG_PARAMETER_BOUND, _LOG_PARAMETER_BOUND) for _ in range(len(first))
        )
    else:
        tau = regularization.min_relative_nugget
        prior_sd = regularization.length_scale_prior_sd
        bounds = gaussian_process_log_parameter_bounds(
            x, regularization, estimate_noise=estimate_noise
        )
    centres = tuple(math.log(width) for width in ranges)

    def log_prior(theta: Vector) -> tuple[float, list[float]]:
        """Return the log-normal length-scale log prior and its gradient.

        References:
            [EQ-SUR-GP-006], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.2].

        """
        gradient = [0.0] * len(theta)
        if prior_sd is None:
            return 0.0, gradient
        value = 0.0
        for axis, centre in enumerate(centres):
            offset = (theta[1 + axis] - centre) / prior_sd
            value -= 0.5 * offset * offset
            gradient[1 + axis] = -offset / prior_sd
        return value, gradient

    cache: dict[Vector, tuple[float, list[float]]] = {}
    factorizations = [0]

    def objective(theta: Vector) -> tuple[float, list[float]]:
        """Return ``-(log p(y | theta) + log prior)`` and its gradient.

        The value and gradient come from one factorization and are cached
        per ``theta``, so the optimizer's separate value and gradient
        requests at the same point cost a single factorization.

        References:
            [EQ-SUR-GP-004], [EQ-SUR-GP-006],
            [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        key = tuple(theta)
        hit = cache.get(key)
        if hit is not None:
            return hit
        factorizations[0] += 1
        value, gradient = gaussian_process_log_marginal_likelihood(
            x,
            targets,
            theta,
            kernel=kernel,
            nugget=jitter,
            estimate_noise=estimate_noise,
            relative_nugget=tau,
        )
        prior, prior_gradient = log_prior(theta)
        result = (
            -(value + prior),
            [-(g + p) for g, p in zip(gradient, prior_gradient, strict=True)],
        )
        if len(cache) > 8:
            cache.clear()
        cache[key] = result
        return result

    def to_theta(phi: Vector) -> Vector:
        """Map unconstrained ``phi`` to ``theta`` inside the box.

        References:
            [EQ-SUR-GP-006]. No new equation.

        """
        return tuple(
            min(max(low + (high - low) * math.sin(0.5 * p) ** 2, low), high)
            for p, (low, high) in zip(phi, bounds, strict=True)
        )

    def to_phi(theta: Vector) -> Vector:
        """Map ``theta`` (clipped to the inner 98 % of the box) to ``phi``.

        References:
            [EQ-SUR-GP-006]. No new equation.

        """
        phi = []
        for value, (low, high) in zip(theta, bounds, strict=True):
            fraction = min(max((value - low) / (high - low), 0.01), 0.99)
            phi.append(2.0 * math.asin(math.sqrt(fraction)))
        return tuple(phi)

    def negative(phi: Vector) -> float:
        """Return the objective in the optimizer's coordinates.

        References:
            [EQ-SUR-GP-004], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        theta = phi if regularization is None else to_theta(phi)
        return objective(theta)[0]

    def negative_gradient(phi: Vector) -> Vector:
        """Return the objective gradient in the optimizer's coordinates.

        References:
            [EQ-SUR-GP-004], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        if regularization is None:
            return tuple(objective(phi)[1])
        gradient = objective(to_theta(phi))[1]
        return tuple(
            0.5 * g * (high - low) * math.sin(p)
            for g, p, (low, high) in zip(gradient, phi, bounds, strict=True)
        )

    best: tuple[float, Vector, bool] | None = None
    candidates: list[tuple[float, Vector, bool]] = []
    initial_jitter = jitter
    while True:
        for start in starts:
            begin = start if regularization is None else to_phi(start)
            try:
                if optimize:
                    result = minimize_bfgs(
                        negative,
                        begin,
                        gradient=negative_gradient,
                        gradient_tolerance=1e-6 if regularization is None else 1e-5,
                        max_iterations=200,
                    )
                    candidate = (result.value, result.x, result.converged)
                else:
                    candidate = (negative(begin), begin, True)
            except ValueError:
                continue
            candidates.append(candidate)
        if candidates:
            lowest = min(candidates, key=lambda item: item[0])
            point = lowest[1] if regularization is None else to_theta(lowest[1])
            noise = _objective_noise(
                x, point, dimension, estimate_noise, jitter, tau, kernel, lowest[0]
            )
            best = _select_restart(candidates, noise)
        if best is not None or not adaptive_nugget:
            break
        if jitter * 10.0 > max(_MAX_ADAPTIVE_NUGGET, initial_jitter):
            break
        jitter *= 10.0
        cache.clear()
    if best is None:
        raise ValueError(
            "no start produced a positive-definite kernel matrix"
            + (
                f" (nugget raised to {jitter:.3g})"
                if adaptive_nugget and jitter > initial_jitter
                else ""
            )
        )
    if jitter > initial_jitter:
        warnings.warn(
            f"the nugget was increased from {initial_jitter:.3g} to {jitter:.3g} "
            "(standardized units) to make the kernel matrix positive definite; "
            "the fitted nugget is reported in GaussianProcessRegression.nugget",
            GaussianProcessJitterWarning,
            stacklevel=2,
        )
    evaluations = factorizations[0]
    value, optimum, converged = best
    theta = optimum if regularization is None else to_theta(optimum)
    if prior_sd is not None:
        value += log_prior(theta)[0]
    signal, scales, extra = _split_parameters(theta, dimension, estimate_noise)
    hyperparameters = GaussianProcessHyperparameters(
        signal * scale * scale, scales, (jitter + tau * signal + extra) * scale * scale
    )
    fitted = GaussianProcessRegression(
        kernel,
        x,
        y,
        hyperparameters,
        theta,
        jitter,
        estimate_noise,
        mean,
        scale,
        -value - count * math.log(scale),
        converged,
        len(starts),
        evaluations,
        relative_nugget=tau,
        regularization=regularization,
        log_parameter_bounds=bounds,
        at_bound=_bound_hits(theta, bounds, dimension),
    )
    if fitted.condition_number > _CONDITION_WARNING:
        warnings.warn(
            f"the fitted kernel matrix has 1-norm condition number "
            f"{fitted.condition_number:.3g}; likelihood values and predictions "
            "carry rounding noise of about cond(K) * 1e-16 relative. A larger "
            "nugget or GaussianProcessRegularization bounds it.",
            GaussianProcessConditioningWarning,
            stacklevel=2,
        )
    return fitted


def fit_gaussian_process_to_model(
    model: ProbabilisticModel,
    quantity: str | None = None,
    *,
    size: int,
    design: str = "lhs",
    seed: int | RandomStream | None = None,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_noise: bool = False,
    restarts: int = 3,
    regularization: GaussianProcessRegularization | None = None,
) -> GaussianProcessRegression:
    """Train a GP surrogate of one QoI on a design drawn from the input law.

    Args:
        model: Probabilistic model.
        quantity: QoI name; defaults to the first declared quantity.
        size: Number of training evaluations.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        seed: Root seed or stream (children ``"gp-design"`` and restarts).
        kernel: Kernel name.
        nugget: Explicit diagonal variance in standardized units.
        estimate_noise: Also estimate a noise variance.
        restarts: Optimizer starts.
        regularization: Optional hyperparameter regularization (see
            :func:`fit_gaussian_process`).

    Returns:
        Fitted regression on physical inputs; ``model.evaluations`` grows by
        ``size``.

    Equation:
        [EQ-SUR-GP-003], [EQ-SUR-GP-004], [EQ-SAMP-INV-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2 and 5].

    """
    name = model.quantity_names[0] if quantity is None else quantity
    model.quantity(name)
    root = resolve_stream(seed)
    unit = make_design(design, size, model.dimension, seed=root.child("gp-design"))
    points = model.inputs.sample_design(unit)
    values = model.evaluate_many(points)[name]
    return fit_gaussian_process(
        points,
        values,
        kernel=kernel,
        nugget=nugget,
        estimate_noise=estimate_noise,
        restarts=restarts,
        seed=root,
        regularization=regularization,
    )
