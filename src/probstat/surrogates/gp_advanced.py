"""Advanced Gaussian-process models: kernel algebra, derivatives, sparse, multi-output.

Experimental (added in 0.28).

This module extends the exact GP of :mod:`probstat.surrogates.gaussian_process`
with seven families of models, all written in pure Python and all following the
explicit-nugget policy of that module (a kernel matrix that is not positive
definite raises ``ValueError``; no jitter is ever added silently, and the
1-norm condition number of the factorized matrix is reported in every fit):

1. **Kernel algebra** (:class:`Kernel`): stationary ARD, periodic,
   rational-quadratic, linear, and white kernels, their sums and products, with
   exact gradients with respect to the log hyperparameters, and a composite
   kernel GP fitted by type-II maximum likelihood
   (:func:`fit_kernel_gp`) [EQ-SUR-GPX-001], [EQ-SUR-GPX-002],
   [EQ-SUR-GPX-003].
2. **Derivative-enhanced GP** (:func:`fit_derivative_gp`): joint observations of
   ``f`` and ``grad f`` through the derivative kernel blocks of the squared
   exponential, Matern 5/2, and Matern 3/2 kernels
   [EQ-SUR-GPX-004], [EQ-SUR-GPX-005].
3. **Predictive derivatives** of a fitted plain GP
   (:func:`gaussian_process_gradient`) [EQ-SUR-GPX-006].
4. **Sparse GP** (:func:`fit_sparse_gp`): DTC, FITC, and Titsias' collapsed
   variational free energy (VFE) with fixed or optimized inducing inputs and a
   cost model (:func:`sparse_gp_cost`) [EQ-SUR-GPX-007] to [EQ-SUR-GPX-010].
5. **Most-likely heteroscedastic GP** (:func:`fit_heteroscedastic_gp`)
   [EQ-SUR-GPX-011].
6. **Multi-output GP** with the linear model of coregionalization
   (:func:`fit_multi_output_gp`) [EQ-SUR-GPX-012].
7. **Linear-operator-constrained GP** (:func:`fit_operator_gp`): observations of
   ``L u`` for a constant-coefficient linear differential operator ``L`` under
   a squared-exponential prior on ``u`` [EQ-SUR-GPX-013], [EQ-SUR-GPX-014].

Hyperparameters are optimized in log units inside the box ``|theta| <= 15`` by
BFGS (:func:`probstat.optimization.minimize_bfgs`); parameters that end within
``0.01`` of the box are listed in ``at_bound``. Output standardization is
limited to centering (and, for the multi-output model, per-output scaling), so
kernel variances are in output units unless stated otherwise.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector
from probstat.optimization import minimize_bfgs

from .gaussian_process import (
    KERNELS,
    GaussianProcessConditioningWarning,
    GaussianProcessRegression,
    _correlation,
    _fast_cholesky,
    _inverse_factor_columns,
    _length_scale_factor,
    _precision_entry,
    _sumprod,
    _triangular_solve,
)

_LOG_TWO_PI = math.log(2.0 * math.pi)
_THETA_BOUND = 15.0
_BOUND_TOLERANCE = 0.01
_CONDITION_LIMIT = 1e12
_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)

Matrix2 = list[list[float]]


# ---------------------------------------------------------------------------
# validation and dense helpers
# ---------------------------------------------------------------------------


def _require_positive(value: float, name: str) -> float:
    """Return ``value`` as a finite positive float.

    Raises:
        ValueError: If the value is not finite and positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_count(value: int, name: str, minimum: int = 1) -> int:
    """Return ``value`` as an integer at least ``minimum``.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If it is smaller than ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_points(points: Sequence[Sequence[float]], name: str) -> tuple[Vector, ...]:
    """Return non-empty, equally long, finite input vectors.

    Raises:
        ValueError: If the set is empty or ragged.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    x = tuple(as_vector(point, name=name) for point in points)
    if not x:
        raise ValueError(f"{name} must not be empty")
    dimension = len(x[0])
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError(f"{name} must be non-empty and equally long")
    return x


def _factor(matrix: Matrix2, what: str) -> list[list[float]]:
    """Return the Cholesky rows of ``matrix``; never add jitter.

    Raises:
        ValueError: If the matrix is not positive definite.

    Equation:
        [EQ-LINALG-001].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    try:
        return _fast_cholesky(matrix)
    except ValueError as error:
        raise ValueError(
            f"{what} is not positive definite; increase the nugget or noise "
            "variance (no jitter is added)"
        ) from error


def _forward(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``L w = b`` for ragged lower rows.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    out: list[float] = []
    for i, row in enumerate(lower):
        out.append((rhs[i] - _sumprod(row[:i], out)) / row[i])
    return out


def _backward(lower: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``L' x = b`` for ragged lower rows.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    count = len(lower)
    out = [0.0] * count
    for i in range(count - 1, -1, -1):
        total = rhs[i]
        for k in range(i + 1, count):
            total -= lower[k][i] * out[k]
        out[i] = total / lower[i][i]
    return out


def _log_det(lower: Sequence[Sequence[float]]) -> float:
    """Return ``log det(L L') = 2 sum log L_ii``.

    Equation:
        [EQ-LINALG-006].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    return 2.0 * math.fsum(math.log(row[-1]) for row in lower)


def _precision(lower: Sequence[Sequence[float]]) -> Matrix2:
    """Return the full inverse ``K^-1`` from the Cholesky rows of ``K``.

    Equation:
        [EQ-SUR-GP-007].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    count = len(lower)
    columns = _inverse_factor_columns(lower)
    out = [[0.0] * count for _ in range(count)]
    for a in range(count):
        for b in range(a, count):
            out[a][b] = out[b][a] = _precision_entry(columns, a, b)
    return out


def _condition(matrix: Matrix2, precision: Matrix2) -> float:
    """Return the 1-norm condition number ``||K||_1 ||K^-1||_1``.

    Equation:
        [EQ-SUR-GP-007].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.6].

    """
    norm_k = max(math.fsum(abs(v) for v in row) for row in matrix)
    norm_inverse = max(math.fsum(abs(v) for v in row) for row in precision)
    return norm_k * norm_inverse


def _warn_conditioning(condition: float, what: str) -> None:
    """Emit the shared conditioning warning when ``cond > 1e12``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.6]. No equation is evaluated.

    """
    if condition > _CONDITION_LIMIT:
        warnings.warn(
            f"the {what} has 1-norm condition number {condition:.3g}; "
            "likelihood values and predictions carry rounding noise of about "
            "cond * 1e-16 relative (report only; nothing was modified)",
            GaussianProcessConditioningWarning,
            stacklevel=3,
        )


def _theta_names_at_bound(
    theta: Sequence[float], names: Sequence[str], bound: float = _THETA_BOUND
) -> tuple[str, ...]:
    """Name the log parameters within ``0.01`` of the box ``|theta| <= bound``.

    References:
        [REF-NOCEDAL-WRIGHT-2006]. No equation is evaluated.

    """
    hits: list[str] = []
    for name, value in zip(names, theta, strict=True):
        if value - (-bound) <= _BOUND_TOLERANCE:
            hits.append(f"{name}:lower")
        elif bound - value <= _BOUND_TOLERANCE:
            hits.append(f"{name}:upper")
    return tuple(hits)


def _check_theta(theta: Sequence[float], count: int) -> Vector:
    """Validate a log-parameter vector against the box and length.

    Raises:
        ValueError: If the length differs or the box is left.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation.

    """
    vector = as_vector(theta, name="log_parameters")
    if len(vector) != count:
        raise ValueError(f"log_parameters must have length {count}")
    if any(abs(value) > _THETA_BOUND for value in vector):
        raise ValueError(f"log hyperparameters left the box |theta| <= {_THETA_BOUND}")
    return vector


# ---------------------------------------------------------------------------
# kernel algebra
# ---------------------------------------------------------------------------


class Kernel(ABC):
    """Abstract covariance function with log hyperparameters and exact gradients.

    A kernel is immutable. :attr:`log_parameters` lists its hyperparameters in
    log units (all positive hyperparameters are stored as logs), and
    :meth:`gradient` returns ``dk/dtheta_j`` for every log parameter ``theta_j``
    so that a composite kernel's marginal-likelihood gradient is exact
    [EQ-SUR-GPX-001], [EQ-SUR-GPX-003]. The white kernel is special: it adds a
    variance to the diagonal of *training* matrices (:attr:`diagonal_variance`)
    and contributes nothing to cross-covariances or latent predictive
    variances.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """

    __slots__ = ()

    @property
    @abstractmethod
    def log_parameters(self) -> Vector:
        """Return the hyperparameters in log units.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Return one name per log parameter.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """

    @abstractmethod
    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return a kernel of the same structure with new log parameters.

        Args:
            log_parameters: Vector of the length of :attr:`log_parameters`.

        Returns:
            The updated kernel.

        Raises:
            ValueError: If the length is wrong or the box is left.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """

    @abstractmethod
    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return the latent covariance ``k(x, y)`` (zero for white noise).

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """

    @abstractmethod
    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return ``dk(x, y)/d theta_j`` for every log parameter.

        Args:
            x: First input.
            y: Second input.

        Returns:
            One derivative per log parameter, in the order of
            :attr:`log_parameters`.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """

    @property
    def parameter_count(self) -> int:
        """Return the number of log parameters.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 5].

        """
        return len(self.log_parameters)

    @property
    def diagonal_variance(self) -> float:
        """Return the white variance added to the diagonal of training matrices.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return 0.0

    @property
    def diagonal_variance_gradient(self) -> Vector:
        """Return ``d diagonal_variance / d theta_j`` for every log parameter.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return (0.0,) * self.parameter_count

    @property
    def has_white(self) -> bool:
        """Return whether a white component is present.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return False

    def matrix(self, points: Sequence[Sequence[float]]) -> Matrix2:
        """Return the training covariance ``K(X, X)`` including white noise.

        Args:
            points: Inputs.

        Returns:
            Full symmetric matrix as row lists.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        count = len(points)
        rows = [[0.0] * count for _ in range(count)]
        for i in range(count):
            for j in range(i + 1):
                rows[i][j] = rows[j][i] = self.evaluate(points[i], points[j])
        noise = self.diagonal_variance
        for i in range(count):
            rows[i][i] += noise
        return rows

    def matrix_gradients(self, points: Sequence[Sequence[float]]) -> list[Matrix2]:
        """Return ``dK/d theta_j`` (white noise included) for every parameter.

        Args:
            points: Inputs.

        Returns:
            One full symmetric matrix per log parameter.

        References:
            [EQ-SUR-GPX-001], [EQ-SUR-GPX-003].

        """
        count = len(points)
        size = self.parameter_count
        mats = [[[0.0] * count for _ in range(count)] for _ in range(size)]
        for i in range(count):
            for j in range(i + 1):
                derivative = self.gradient(points[i], points[j])
                for p in range(size):
                    mats[p][i][j] = mats[p][j][i] = derivative[p]
        for p, noise in enumerate(self.diagonal_variance_gradient):
            if noise != 0.0:
                for i in range(count):
                    mats[p][i][i] += noise
        return mats

    def cross(
        self, left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
    ) -> Matrix2:
        """Return the latent cross-covariance ``k(X_left, X_right)``.

        Args:
            left: Row inputs.
            right: Column inputs.

        Returns:
            Matrix with ``len(left)`` rows (white noise excluded).

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return [[self.evaluate(a, b) for b in right] for a in left]

    def __add__(self, other: object) -> Kernel:
        """Return the sum kernel ``k + other``.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if not isinstance(other, Kernel):
            return NotImplemented
        return SumKernel(self, other)

    def __mul__(self, other: object) -> Kernel:
        """Return the product kernel ``k * other``.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if not isinstance(other, Kernel):
            return NotImplemented
        return ProductKernel(self, other)


def _squared_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Return ``||x - y||^2`` after checking the lengths agree.

    Raises:
        ValueError: If the inputs have different lengths.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2]. No equation.

    """
    if len(x) != len(y):
        raise ValueError("inputs must have equal dimension")
    return math.fsum((a - b) ** 2 for a, b in zip(x, y, strict=True))


@dataclass(frozen=True, slots=True)
class StationaryKernel(Kernel):
    """Stationary ARD kernel (squared exponential, Matern 5/2, Matern 3/2).

    ``k(x, y) = sigma_f^2 phi(r)`` with ``r^2 = sum_i ((x_i - y_i)/ell_i)^2``
    [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-008]. Log parameters are
    ``(log sigma_f^2, log ell_1, ..., log ell_d)``.

    Args:
        name: One of ``KERNELS``.
        signal_variance: ``sigma_f^2 > 0``.
        length_scales: ``ell_i > 0``, one per input.

    Raises:
        ValueError: If the name is unknown or a value is not positive.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """

    name: str = "squared_exponential"
    signal_variance: float = 1.0
    length_scales: Vector = (1.0,)

    def __post_init__(self) -> None:
        """Validate the kernel name and hyperparameters.

        Raises:
            ValueError: If a value is out of range.

        References:
            [EQ-SUR-GP-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if self.name not in KERNELS:
            raise ValueError(f"name must be one of {KERNELS}")
        _require_positive(self.signal_variance, "signal_variance")
        scales = as_vector(self.length_scales, name="length_scales")
        if not scales:
            raise ValueError("length_scales must not be empty")
        for value in scales:
            _require_positive(value, "length_scales")

    @property
    def log_parameters(self) -> Vector:
        """Return ``(log sigma_f^2, log ell_i...)``.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        return (
            math.log(self.signal_variance),
            *(math.log(value) for value in self.length_scales),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the names of the log parameters.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        return (
            "signal_variance",
            *(f"length_scale[{i}]" for i in range(len(self.length_scales))),
        )

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the kernel with new log parameters.

        Args:
            log_parameters: ``(log sigma_f^2, log ell_i...)``.

        Returns:
            New :class:`StationaryKernel`.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        theta = _check_theta(log_parameters, 1 + len(self.length_scales))
        return StationaryKernel(
            self.name, math.exp(theta[0]), tuple(math.exp(v) for v in theta[1:])
        )

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return ``k(x, y)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        Raises:
            ValueError: If an input has the wrong dimension.

        References:
            [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-008].

        """
        if len(x) != len(self.length_scales) or len(y) != len(self.length_scales):
            raise ValueError("inputs must have one entry per length scale")
        squared = math.fsum(
            ((a - b) / ell) ** 2
            for a, b, ell in zip(x, y, self.length_scales, strict=True)
        )
        return self.signal_variance * _correlation(self.name, squared)

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return ``dk/d(log sigma_f^2, log ell_i...)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            Derivatives in the order of :attr:`log_parameters`.

        References:
            [EQ-SUR-GPX-001], [EQ-SUR-GP-004].

        """
        parts = [
            ((a - b) / ell) ** 2
            for a, b, ell in zip(x, y, self.length_scales, strict=True)
        ]
        squared = math.fsum(parts)
        value = self.signal_variance * _correlation(self.name, squared)
        factor = self.signal_variance * _length_scale_factor(self.name, squared)
        return (value, *(factor * part for part in parts))


@dataclass(frozen=True, slots=True)
class PeriodicKernel(Kernel):
    """Periodic (exp-sine-squared) kernel.

    ``k(x, y) = sigma^2 exp(-2 sin^2(pi r / p) / ell^2)`` with ``r = ||x - y||``
    [EQ-SUR-GPX-002]. Log parameters ``(log sigma^2, log ell, log p)``.

    Args:
        signal_variance: ``sigma^2 > 0``.
        length_scale: ``ell > 0``.
        period: ``p > 0`` (input units).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """

    signal_variance: float = 1.0
    length_scale: float = 1.0
    period: float = 1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If a value is not positive.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        _require_positive(self.signal_variance, "signal_variance")
        _require_positive(self.length_scale, "length_scale")
        _require_positive(self.period, "period")

    @property
    def log_parameters(self) -> Vector:
        """Return ``(log sigma^2, log ell, log p)``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        return (
            math.log(self.signal_variance),
            math.log(self.length_scale),
            math.log(self.period),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the parameter names.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        return ("signal_variance", "length_scale", "period")

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the kernel with new log parameters.

        Args:
            log_parameters: ``(log sigma^2, log ell, log p)``.

        Returns:
            New :class:`PeriodicKernel`.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        theta = _check_theta(log_parameters, 3)
        return PeriodicKernel(*(math.exp(v) for v in theta))

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return ``k(x, y)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        u = math.pi * math.sqrt(_squared_distance(x, y)) / self.period
        return self.signal_variance * math.exp(
            -2.0 * math.sin(u) ** 2 / self.length_scale**2
        )

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return ``dk/d(log sigma^2, log ell, log p)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The three derivatives.

        References:
            [EQ-SUR-GPX-002].

        """
        u = math.pi * math.sqrt(_squared_distance(x, y)) / self.period
        ell2 = self.length_scale**2
        value = self.signal_variance * math.exp(-2.0 * math.sin(u) ** 2 / ell2)
        return (
            value,
            value * 4.0 * math.sin(u) ** 2 / ell2,
            value * 2.0 * u * math.sin(2.0 * u) / ell2,
        )


@dataclass(frozen=True, slots=True)
class RationalQuadraticKernel(Kernel):
    """Rational-quadratic kernel (scale mixture of squared exponentials).

    ``k(x, y) = sigma^2 (1 + r^2 / (2 alpha ell^2))^-alpha`` [EQ-SUR-GPX-002].
    It tends to the squared exponential as ``alpha -> infinity``. Log
    parameters ``(log sigma^2, log ell, log alpha)``.

    Args:
        signal_variance: ``sigma^2 > 0``.
        length_scale: ``ell > 0``.
        alpha: Mixture shape ``alpha > 0``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """

    signal_variance: float = 1.0
    length_scale: float = 1.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If a value is not positive.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        _require_positive(self.signal_variance, "signal_variance")
        _require_positive(self.length_scale, "length_scale")
        _require_positive(self.alpha, "alpha")

    @property
    def log_parameters(self) -> Vector:
        """Return ``(log sigma^2, log ell, log alpha)``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        return (
            math.log(self.signal_variance),
            math.log(self.length_scale),
            math.log(self.alpha),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the parameter names.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        return ("signal_variance", "length_scale", "alpha")

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the kernel with new log parameters.

        Args:
            log_parameters: ``(log sigma^2, log ell, log alpha)``.

        Returns:
            New :class:`RationalQuadraticKernel`.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        theta = _check_theta(log_parameters, 3)
        return RationalQuadraticKernel(*(math.exp(v) for v in theta))

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return ``k(x, y)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        w = _squared_distance(x, y) / (2.0 * self.alpha * self.length_scale**2)
        return self.signal_variance * math.exp(-self.alpha * math.log1p(w))

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return ``dk/d(log sigma^2, log ell, log alpha)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The three derivatives.

        References:
            [EQ-SUR-GPX-002].

        """
        w = _squared_distance(x, y) / (2.0 * self.alpha * self.length_scale**2)
        value = self.signal_variance * math.exp(-self.alpha * math.log1p(w))
        ratio = w / (1.0 + w)
        return (
            value,
            value * 2.0 * self.alpha * ratio,
            value * self.alpha * (ratio - math.log1p(w)),
        )


@dataclass(frozen=True, slots=True)
class LinearKernel(Kernel):
    """Linear (dot-product) kernel ``k(x, y) = b + v x . y`` [EQ-SUR-GPX-002].

    Log parameters ``(log b, log v)``; the prior is a random affine function.

    Args:
        bias_variance: ``b > 0``.
        slope_variance: ``v > 0``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """

    bias_variance: float = 1.0
    slope_variance: float = 1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If a value is not positive.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        _require_positive(self.bias_variance, "bias_variance")
        _require_positive(self.slope_variance, "slope_variance")

    @property
    def log_parameters(self) -> Vector:
        """Return ``(log b, log v)``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        return (math.log(self.bias_variance), math.log(self.slope_variance))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the parameter names.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        return ("bias_variance", "slope_variance")

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the kernel with new log parameters.

        Args:
            log_parameters: ``(log b, log v)``.

        Returns:
            New :class:`LinearKernel`.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        theta = _check_theta(log_parameters, 2)
        return LinearKernel(*(math.exp(v) for v in theta))

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return ``k(x, y)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        Raises:
            ValueError: If the inputs differ in length.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

        """
        if len(x) != len(y):
            raise ValueError("inputs must have equal dimension")
        dot = math.fsum(a * b for a, b in zip(x, y, strict=True))
        return self.bias_variance + self.slope_variance * dot

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return ``dk/d(log b, log v)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The two derivatives.

        References:
            [EQ-SUR-GPX-002].

        """
        dot = math.fsum(a * b for a, b in zip(x, y, strict=True))
        return (self.bias_variance, self.slope_variance * dot)


@dataclass(frozen=True, slots=True)
class WhiteKernel(Kernel):
    """White-noise kernel ``k(x, y) = sigma_n^2 [x is y]`` [EQ-SUR-GPX-002].

    The variance is attached to *training indices*: it enters the diagonal of
    :meth:`Kernel.matrix` and its gradient, and contributes nothing to
    :meth:`Kernel.cross`, :meth:`Kernel.evaluate`, or latent predictive
    variances. It estimates observation noise when summed with a smooth
    kernel. Log parameter ``log sigma_n^2``.

    Args:
        variance: ``sigma_n^2 > 0``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """

    variance: float = 1.0

    def __post_init__(self) -> None:
        """Validate the variance.

        Raises:
            ValueError: If the variance is not positive.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        _require_positive(self.variance, "variance")

    @property
    def log_parameters(self) -> Vector:
        """Return ``(log sigma_n^2,)``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return (math.log(self.variance),)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the parameter names.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return ("noise_variance",)

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the kernel with a new log variance.

        Args:
            log_parameters: ``(log sigma_n^2,)``.

        Returns:
            New :class:`WhiteKernel`.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        theta = _check_theta(log_parameters, 1)
        return WhiteKernel(math.exp(theta[0]))

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return the latent covariance, which is zero.

        Args:
            x: First input (unused).
            y: Second input (unused).

        Returns:
            ``0.0``; the variance sits on the training diagonal.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return 0.0

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return the latent-covariance gradient, which is zero.

        Args:
            x: First input (unused).
            y: Second input (unused).

        Returns:
            ``(0.0,)``; the diagonal part is :attr:`diagonal_variance_gradient`.

        References:
            [EQ-SUR-GPX-002].

        """
        return (0.0,)

    @property
    def diagonal_variance(self) -> float:
        """Return ``sigma_n^2``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return self.variance

    @property
    def diagonal_variance_gradient(self) -> Vector:
        """Return ``d sigma_n^2 / d log sigma_n^2 = sigma_n^2``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return (self.variance,)

    @property
    def has_white(self) -> bool:
        """Return ``True``.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return True


@dataclass(frozen=True, slots=True)
class SumKernel(Kernel):
    """Sum ``k_1 + k_2`` of two kernels (a valid kernel) [EQ-SUR-GPX-001].

    Log parameters are those of ``left`` followed by those of ``right``;
    gradients concatenate, white variances add.

    Args:
        left: First kernel.
        right: Second kernel.

    Raises:
        TypeError: If an argument is not a :class:`Kernel`.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """

    left: Kernel
    right: Kernel

    def __post_init__(self) -> None:
        """Validate the operands.

        Raises:
            TypeError: If an operand is not a kernel.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if not isinstance(self.left, Kernel) or not isinstance(self.right, Kernel):
            raise TypeError("operands must be Kernel instances")

    @property
    def log_parameters(self) -> Vector:
        """Return the concatenated log parameters.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return self.left.log_parameters + self.right.log_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the names, prefixed by operand (``left.`` / ``right.``).

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return tuple(f"left.{n}" for n in self.left.parameter_names) + tuple(
            f"right.{n}" for n in self.right.parameter_names
        )

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the sum with new log parameters.

        Args:
            log_parameters: Concatenated parameters.

        Returns:
            New :class:`SumKernel`.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        theta = _check_theta(log_parameters, self.parameter_count)
        cut = self.left.parameter_count
        return SumKernel(
            self.left.with_log_parameters(theta[:cut]),
            self.right.with_log_parameters(theta[cut:]),
        )

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return ``k_1(x, y) + k_2(x, y)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return self.left.evaluate(x, y) + self.right.evaluate(x, y)

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return the concatenated parameter gradients.

        Args:
            x: First input.
            y: Second input.

        Returns:
            Derivatives of the sum.

        References:
            [EQ-SUR-GPX-001].

        """
        return self.left.gradient(x, y) + self.right.gradient(x, y)

    @property
    def diagonal_variance(self) -> float:
        """Return the sum of the operands' white variances.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return self.left.diagonal_variance + self.right.diagonal_variance

    @property
    def diagonal_variance_gradient(self) -> Vector:
        """Return the concatenated white-variance gradients.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return (
            self.left.diagonal_variance_gradient + self.right.diagonal_variance_gradient
        )

    @property
    def has_white(self) -> bool:
        """Return whether either operand is white.

        References:
            [EQ-SUR-GPX-002], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        return self.left.has_white or self.right.has_white


@dataclass(frozen=True, slots=True)
class ProductKernel(Kernel):
    """Product ``k_1 k_2`` of two kernels (a valid kernel) [EQ-SUR-GPX-001].

    The gradient follows the product rule. A white component inside a product
    has no training-index meaning here and is rejected.

    Args:
        left: First kernel.
        right: Second kernel.

    Raises:
        TypeError: If an argument is not a :class:`Kernel`.
        ValueError: If an operand contains a white component.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """

    left: Kernel
    right: Kernel

    def __post_init__(self) -> None:
        """Validate the operands.

        Raises:
            TypeError: If an operand is not a kernel.
            ValueError: If an operand contains a white component.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        if not isinstance(self.left, Kernel) or not isinstance(self.right, Kernel):
            raise TypeError("operands must be Kernel instances")
        if self.left.has_white or self.right.has_white:
            raise ValueError("a WhiteKernel cannot be a factor of a product")

    @property
    def log_parameters(self) -> Vector:
        """Return the concatenated log parameters.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return self.left.log_parameters + self.right.log_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the names, prefixed by operand (``left.`` / ``right.``).

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return tuple(f"left.{n}" for n in self.left.parameter_names) + tuple(
            f"right.{n}" for n in self.right.parameter_names
        )

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the product with new log parameters.

        Args:
            log_parameters: Concatenated parameters.

        Returns:
            New :class:`ProductKernel`.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        theta = _check_theta(log_parameters, self.parameter_count)
        cut = self.left.parameter_count
        return ProductKernel(
            self.left.with_log_parameters(theta[:cut]),
            self.right.with_log_parameters(theta[cut:]),
        )

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return ``k_1(x, y) k_2(x, y)``.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        References:
            [EQ-SUR-GPX-001], [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        return self.left.evaluate(x, y) * self.right.evaluate(x, y)

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return the product-rule gradient.

        Args:
            x: First input.
            y: Second input.

        Returns:
            Derivatives of the product.

        References:
            [EQ-SUR-GPX-001].

        """
        left_value = self.left.evaluate(x, y)
        right_value = self.right.evaluate(x, y)
        return tuple(g * right_value for g in self.left.gradient(x, y)) + tuple(
            g * left_value for g in self.right.gradient(x, y)
        )


# ---------------------------------------------------------------------------
# composite-kernel GP
# ---------------------------------------------------------------------------


def _kernel_marginal(
    kernel: Kernel,
    x: Sequence[Vector],
    y: Sequence[float],
    diagonal: Sequence[float],
    *,
    gradient: bool,
) -> tuple[float, Vector, list[list[float]], Vector, Matrix2]:
    """Return ``log p(y | theta)``, its gradient, the factor, ``alpha``, and ``K``.

    ``K = K_theta(X, X) + diag(diagonal)``; the gradient is
    ``1/2 tr((alpha alpha' - K^-1) dK/d theta_j)`` for every log parameter.

    Raises:
        ValueError: If ``K`` is not positive definite.

    Equation:
        [EQ-SUR-GPX-003].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    count = len(x)
    matrix = kernel.matrix(x)
    for i in range(count):
        matrix[i][i] += diagonal[i]
    lower = _factor(matrix, "kernel matrix")
    alpha = _triangular_solve(lower, y)
    value = -0.5 * _sumprod(y, alpha) - 0.5 * _log_det(lower)
    value -= 0.5 * count * _LOG_TWO_PI
    if not gradient:
        return value, (), lower, alpha, matrix
    precision = _precision(lower)
    weight = [
        [alpha[i] * alpha[j] - precision[i][j] for j in range(count)]
        for i in range(count)
    ]
    grads = tuple(
        0.5 * math.fsum(_sumprod(weight[i], dk[i]) for i in range(count))
        for dk in kernel.matrix_gradients(x)
    )
    return value, grads, lower, alpha, matrix


@dataclass(frozen=True)
class KernelGaussianProcess:
    """A GP fitted with a composite :class:`Kernel` [EQ-SUR-GPX-003].

    The outputs are centered by their sample mean (``output_mean``) and not
    scaled, so every variance is in output units. The diagonal of the kernel
    matrix is ``K_theta + (nugget + noise_variances_i) I``; white-kernel
    components of the kernel add to it as well.

    Args:
        kernel: Fitted kernel.
        points: Training inputs.
        values: Training outputs.
        nugget: Explicit positive diagonal variance (output units).
        noise_variances: Per-point extra noise variances (zeros if none).
        output_mean: Sample mean subtracted from the outputs.
        log_marginal_likelihood: ``log p(y | theta)`` at the fitted kernel.
        converged: Whether the best start met the optimizer tolerance.
        stop_reason: Optimizer message of the best start.
        restarts: Number of optimizer starts attempted.
        evaluations: Likelihood evaluations (factorizations) over all starts.
        condition_number: 1-norm condition number of the factorized matrix.
        at_bound: Log parameters within ``0.01`` of the box ``|theta| <= 15``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2 and Section 5.4].

    """

    kernel: Kernel
    points: tuple[Vector, ...]
    values: Vector
    nugget: float
    noise_variances: Vector
    output_mean: float
    log_marginal_likelihood: float
    converged: bool
    stop_reason: str
    restarts: int
    evaluations: int
    condition_number: float = field(init=False, compare=False, default=math.nan)
    at_bound: tuple[str, ...] = ()
    _lower: list[list[float]] = field(init=False, repr=False, compare=False)
    _alpha: Vector = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Factorize the training matrix once and record its condition number.

        Raises:
            ValueError: If the matrix is not positive definite.

        Equation:
            [EQ-SUR-GPX-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        centered = tuple(v - self.output_mean for v in self.values)
        diagonal = [self.nugget + extra for extra in self.noise_variances]
        _, _, lower, alpha, matrix = _kernel_marginal(
            self.kernel, self.points, centered, diagonal, gradient=False
        )
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_alpha", alpha)
        object.__setattr__(
            self, "condition_number", _condition(matrix, _precision(lower))
        )

    @property
    def ill_conditioned(self) -> bool:
        """Return whether ``cond_1(K) > 1e12`` (predictions carry rounding noise).

        References:
            [REF-GOLUB-VANLOAN-2013, Section 2.6]. No equation is evaluated.

        """
        return self.condition_number > _CONDITION_LIMIT

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and variance at one input.

        ``mu = m + k_*' alpha`` and ``sigma^2 = k(x, x) - v'v`` with
        ``v = L^-1 k_*`` (latent function); ``include_noise`` adds the nugget
        and the white variance of the kernel. Tiny negative variances from
        cancellation are reported as zero.

        Args:
            x: Input vector.
            include_noise: Predict a new observation instead of ``f``.

        Returns:
            Predictive mean and variance in output units.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-GPX-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        cross = [self.kernel.evaluate(point, train) for train in self.points]
        mean = self.output_mean + _sumprod(cross, self._alpha)
        whitened = _forward(self._lower, cross)
        variance = self.kernel.evaluate(point, point) - _sumprod(whitened, whitened)
        if include_noise:
            variance += self.nugget + self.kernel.diagonal_variance
        return mean, max(variance, 0.0)

    def predict(self, x: Sequence[float]) -> float:
        """Return the predictive mean at one input.

        Args:
            x: Input vector.

        Returns:
            Predictive mean.

        Equation:
            [EQ-SUR-GPX-003].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

        """
        return self.predict_with_variance(x)[0]


def fit_kernel_gp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    kernel: Kernel,
    *,
    nugget: float = 1e-8,
    noise_variances: Sequence[float] | None = None,
    optimize: bool = True,
    restarts: int = 2,
    seed: int | RandomStream | None = None,
    max_iterations: int = 100,
    log_bounds: Sequence[tuple[float, float]] | None = None,
) -> KernelGaussianProcess:
    """Fit a composite-kernel GP by type-II maximum likelihood.

    Maximizes ``log p(y | theta) = -1/2 y'K^-1 y - 1/2 log det K - n/2 log 2 pi``
    over the log parameters of ``kernel`` with BFGS and the exact gradient
    ``1/2 tr((alpha alpha' - K^-1) dK/d theta_j)`` [EQ-SUR-GPX-003], inside the
    box ``|theta| <= 15``. The first start is ``kernel`` as given; further
    starts perturb every log parameter uniformly by ``+/- 1`` (seeded). The
    start with the lowest objective is kept. A start whose kernel matrix is not
    positive definite is skipped; if every start fails, ``ValueError`` is
    raised and **no jitter is added**. The factorization's condition number is
    reported, and :class:`GaussianProcessConditioningWarning` is emitted above
    ``1e12``.

    Args:
        points: Training inputs.
        values: Training outputs (not all equal).
        kernel: Initial kernel (sum/product/white allowed).
        nugget: Explicit positive diagonal variance (output units).
        noise_variances: Optional per-point noise variances added to the
            diagonal (zeros allowed); used for heteroscedastic noise.
        optimize: Maximize the likelihood; ``False`` keeps ``kernel``.
        restarts: Total number of optimizer starts (at least one).
        seed: Integer seed or stream for the perturbations.
        max_iterations: BFGS iteration limit per start.
        log_bounds: Optional ``(low, high)`` per log parameter inside the
            default box ``|theta| <= 15``. Periodic kernels in particular have
            a multimodal likelihood in the period (short periods alias the
            data); bounding ``log p`` around the plausible range is advised.

    Returns:
        The fitted :class:`KernelGaussianProcess`.

    Raises:
        TypeError: If ``kernel`` is not a :class:`Kernel`.
        ValueError: If inputs are invalid or no start is positive definite.

    Equation:
        [EQ-SUR-GPX-003].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a Kernel")
    x = _require_points(points, "points")
    y = as_vector(values, name="values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two points with one value each")
    jitter = _require_positive(nugget, "nugget")
    _require_count(restarts, "restarts")
    _require_count(max_iterations, "max_iterations")
    count = len(y)
    extra = (
        (0.0,) * count
        if noise_variances is None
        else as_vector(noise_variances, name="noise_variances")
    )
    if len(extra) != count or any(v < 0.0 for v in extra):
        raise ValueError("noise_variances must be non-negative, one per point")
    mean = math.fsum(y) / count
    if max(y) == min(y):
        raise ValueError("values are constant; nothing to regress")
    centered = tuple(v - mean for v in y)
    diagonal = [jitter + v for v in extra]
    first = _check_theta(kernel.log_parameters, kernel.parameter_count)
    names = kernel.parameter_names
    size = len(first)
    box = [(-_THETA_BOUND, _THETA_BOUND)] * size
    if log_bounds is not None:
        if len(log_bounds) != size:
            raise ValueError(f"log_bounds must have {size} pairs")
        box = []
        for low, high in log_bounds:
            lo = require_finite_real(low, name="log_bounds")
            hi = require_finite_real(high, name="log_bounds")
            if not -_THETA_BOUND <= lo < hi <= _THETA_BOUND:
                raise ValueError("log_bounds must satisfy -15 <= low < high <= 15")
            box.append((lo, hi))
        if any(not lo < v < hi for v, (lo, hi) in zip(first, box, strict=True)):
            raise ValueError("the initial kernel must lie strictly inside log_bounds")
    generator = resolve_stream(seed).child("gpx-restarts").generator()
    starts = [first]
    for _ in range(restarts - 1 if optimize and size else 0):
        starts.append(
            tuple(
                min(
                    max(v + generator.uniform(-1.0, 1.0), lo + 0.01 * (hi - lo)),
                    hi - 0.01 * (hi - lo),
                )
                for v, (lo, hi) in zip(first, box, strict=True)
            )
        )
    cache: dict[Vector, tuple[float, Vector]] = {}
    evaluations = [0]

    def objective(theta: Vector) -> tuple[float, Vector]:
        """Return ``-log p`` and its gradient, cached per ``theta``.

        References:
            [EQ-SUR-GPX-003], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        key = tuple(theta)
        hit = cache.get(key)
        if hit is not None:
            return hit
        evaluations[0] += 1
        value, grads, _, _, _ = _kernel_marginal(
            kernel.with_log_parameters(theta), x, centered, diagonal, gradient=True
        )
        result = (-value, tuple(-g for g in grads))
        if len(cache) > 64:
            del cache[next(iter(cache))]
        cache[key] = result
        return result

    best: tuple[float, Vector, bool, str] | None = None
    for start in starts:
        try:
            if optimize and size:
                run = minimize_bfgs(
                    lambda t: objective(tuple(t))[0],
                    start,
                    gradient=lambda t: objective(tuple(t))[1],
                    gradient_tolerance=1e-5,
                    max_iterations=max_iterations,
                    bounds=box,
                )
                candidate = (run.value, run.x, run.converged, run.message)
            else:
                candidate = (objective(start)[0], start, True, "not optimized")
        except ValueError:
            continue
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError(
            "no start produced a positive-definite kernel matrix; increase the "
            "nugget (no jitter is added)"
        )
    theta = best[1]
    fitted = KernelGaussianProcess(
        kernel.with_log_parameters(theta),
        x,
        y,
        jitter,
        tuple(extra),
        mean,
        -best[0],
        best[2],
        best[3],
        len(starts),
        evaluations[0],
        at_bound=_bounded_hits(theta, names, box),
    )
    _warn_conditioning(fitted.condition_number, "fitted kernel matrix")
    return fitted


# ---------------------------------------------------------------------------
# derivative observations and predictive derivatives
# ---------------------------------------------------------------------------


def _phi_derivatives(name: str, squared: float) -> tuple[float, float, float]:
    """Return ``phi, dphi/dq, d2phi/dq2`` of the unit kernel at ``q = r^2``.

    Squared exponential: ``phi = e^(-q/2)``. Matern 5/2: ``phi'(q) =
    -(5/6)(1 + sqrt(5) r) e^(-sqrt(5) r)`` and ``phi''(q) = (25/12)
    e^(-sqrt(5) r)``. Matern 3/2: ``phi'(q) = -(3/2) e^(-sqrt(3) r)`` and
    ``phi''(q) = 3 sqrt(3) e^(-sqrt(3) r) / (4 r)``, which diverges at
    ``r = 0``; ``0.0`` is returned there because only the product
    ``phi'' Delta_i Delta_j`` (which tends to ``0``) enters the kernel blocks.

    Raises:
        ValueError: If the kernel name is unknown.

    Equation:
        [EQ-SUR-GPX-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """
    if name == "squared_exponential":
        e = math.exp(-0.5 * squared)
        return e, -0.5 * e, 0.25 * e
    r = math.sqrt(squared)
    if name == "matern52":
        e = math.exp(-_SQRT5 * r)
        return (
            (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * e,
            -(5.0 / 6.0) * (1.0 + _SQRT5 * r) * e,
            (25.0 / 12.0) * e,
        )
    if name == "matern32":
        e = math.exp(-_SQRT3 * r)
        second = 3.0 * _SQRT3 * e / (4.0 * r) if r > 0.0 else 0.0
        return (1.0 + _SQRT3 * r) * e, -1.5 * e, second
    raise ValueError(f"kernel must be one of {KERNELS}")


@dataclass(frozen=True, slots=True)
class DerivativeKernelBlocks:
    """Covariances of ``f`` and ``grad f`` between two inputs [EQ-SUR-GPX-004].

    For ``x`` and ``y`` with ``k(x, y)``:

    Args:
        value: ``cov(f(x), f(y)) = k``.
        gradient_x: ``cov(d_i f(x), f(y)) = dk/dx_i``.
        gradient_y: ``cov(f(x), d_j f(y)) = dk/dy_j = -dk/dx_j``.
        hessian: ``cov(d_i f(x), d_j f(y)) = d^2 k / dx_i dy_j``.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9], [REF-SOLAK-ETAL-2002].

    """

    value: float
    gradient_x: Vector
    gradient_y: Vector
    hessian: tuple[Vector, ...]


def derivative_kernel_blocks(
    kernel: str,
    x: Sequence[float],
    y: Sequence[float],
    length_scales: Sequence[float],
    signal_variance: float,
) -> DerivativeKernelBlocks:
    """Return the ``(f, grad f)`` covariance blocks of a stationary ARD kernel.

    With ``k = sigma_f^2 phi(q)``, ``q = sum_i (Delta_i/ell_i)^2``,
    ``Delta = x - y``:

    * ``dk/dx_i = sigma_f^2 phi'(q) 2 Delta_i / ell_i^2``;
    * ``d^2k/dx_i dy_j = sigma_f^2 [-2 phi'(q) delta_ij / ell_i^2
      - 4 phi''(q) Delta_i Delta_j / (ell_i^2 ell_j^2)]``

    [EQ-SUR-GPX-004]. For the squared exponential in one dimension this is
    ``k (1/ell^2 - Delta^2/ell^4)``. The Matern 3/2 process is only once
    mean-square differentiable, but the blocks above are finite and valid
    (the ``phi''`` term vanishes at ``Delta = 0``).

    Args:
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        x: First input.
        y: Second input.
        length_scales: Positive length scales.
        signal_variance: Positive signal variance.

    Returns:
        The four covariance blocks.

    Raises:
        ValueError: If the kernel is unknown or dimensions disagree.

    Equation:
        [EQ-SUR-GPX-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9], [REF-SOLAK-ETAL-2002].

    """
    scales = as_vector(length_scales, name="length_scales")
    if len(x) != len(scales) or len(y) != len(scales):
        raise ValueError("inputs must have one entry per length scale")
    delta = [a - b for a, b in zip(x, y, strict=True)]
    squared = math.fsum((d / ell) ** 2 for d, ell in zip(delta, scales, strict=True))
    phi, first, second = _phi_derivatives(kernel, squared)
    s2 = signal_variance
    gradient_x = tuple(
        s2 * first * 2.0 * d / ell**2 for d, ell in zip(delta, scales, strict=True)
    )
    hessian = tuple(
        tuple(
            s2
            * (
                (-2.0 * first / scales[i] ** 2 if i == j else 0.0)
                - 4.0 * second * delta[i] * delta[j] / (scales[i] ** 2 * scales[j] ** 2)
            )
            for j in range(len(delta))
        )
        for i in range(len(delta))
    )
    return DerivativeKernelBlocks(
        s2 * phi, gradient_x, tuple(-g for g in gradient_x), hessian
    )


def _block_entry(blocks: DerivativeKernelBlocks, row: int, column: int) -> float:
    """Return ``cov`` of component ``row`` at ``x`` and ``column`` at ``y``.

    Component ``-1`` is ``f`` and ``i >= 0`` is ``d_i f``.

    Equation:
        [EQ-SUR-GPX-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

    """
    if row < 0 and column < 0:
        return blocks.value
    if row < 0:
        return blocks.gradient_y[column]
    if column < 0:
        return blocks.gradient_x[row]
    return blocks.hessian[row][column]


@dataclass(frozen=True, slots=True)
class GradientPrediction:
    """Predictive law of ``grad f`` at one input [EQ-SUR-GPX-006].

    Args:
        mean: Posterior mean of ``grad f``.
        covariance: Posterior covariance matrix of ``grad f`` (``d x d``).
        variances: Its diagonal.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

    """

    mean: Vector
    covariance: tuple[Vector, ...]
    variances: Vector


def _groups(
    f_points: Sequence[Vector], g_points: Sequence[Vector], dimension: int
) -> list[tuple[Vector, tuple[int, ...], int]]:
    """Return ``(point, components, start)`` of the joint observation vector.

    Function observations occupy the first ``n`` rows (component ``-1``);
    each gradient point then adds ``d`` rows (components ``0..d-1``).

    References:
        [EQ-SUR-GPX-005]. No new equation.

    """
    groups: list[tuple[Vector, tuple[int, ...], int]] = []
    start = 0
    for point in f_points:
        groups.append((point, (-1,), start))
        start += 1
    for point in g_points:
        groups.append((point, tuple(range(dimension)), start))
        start += dimension
    return groups


def _joint_matrix(
    kernel: str,
    scales: Vector,
    s2: float,
    groups: Sequence[tuple[Vector, tuple[int, ...], int]],
    noises: Sequence[float],
) -> Matrix2:
    """Assemble the joint ``(f, grad f)`` covariance plus the noise diagonal.

    Equation:
        [EQ-SUR-GPX-005].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9], [REF-SOLAK-ETAL-2002].

    """
    total = groups[-1][2] + len(groups[-1][1])
    matrix = [[0.0] * total for _ in range(total)]
    for ga, (pa, ca, sa) in enumerate(groups):
        for pb, cb, sb in groups[: ga + 1]:
            blocks = derivative_kernel_blocks(kernel, pa, pb, scales, s2)
            for ia, row in enumerate(ca):
                for ib, column in enumerate(cb):
                    value = _block_entry(blocks, row, column)
                    matrix[sa + ia][sb + ib] = value
                    matrix[sb + ib][sa + ia] = value
    for i in range(total):
        matrix[i][i] += noises[i]
    return matrix


@dataclass(frozen=True)
class DerivativeGaussianProcess:
    """A GP conditioned on values and gradients of a smooth function.

    The prior is ``f ~ GP(m, k)`` with a stationary ARD kernel; ``f`` is
    observed with variance ``value_noise`` and ``grad f`` with variance
    ``gradient_noise`` per component [EQ-SUR-GPX-005]. Outputs are centered by
    ``output_mean`` (gradients are unaffected by a constant).

    Args:
        kernel: Kernel name.
        signal_variance: Fitted ``sigma_f^2`` (output units squared).
        length_scales: Fitted ARD length scales.
        value_noise: Variance of the function observations.
        gradient_noise: Variance of every gradient component observation.
        points: Inputs of the function observations.
        values: Function observations.
        gradient_points: Inputs of the gradient observations.
        gradients: Gradient observations, one vector per gradient point.
        output_mean: Mean subtracted from the function values.
        log_marginal_likelihood: Log evidence of the joint data.
        converged: Whether the hyperparameter optimizer met its tolerance.
        evaluations: Likelihood evaluations (joint factorizations).
        condition_number: 1-norm condition number of the joint matrix.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9], [REF-SOLAK-ETAL-2002].

    """

    kernel: str
    signal_variance: float
    length_scales: Vector
    value_noise: float
    gradient_noise: float
    points: tuple[Vector, ...]
    values: Vector
    gradient_points: tuple[Vector, ...]
    gradients: tuple[Vector, ...]
    output_mean: float
    log_marginal_likelihood: float
    converged: bool
    evaluations: int
    condition_number: float = field(init=False, compare=False, default=math.nan)
    _lower: list[list[float]] = field(init=False, repr=False, compare=False)
    _alpha: Vector = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Factorize the joint covariance once.

        Raises:
            ValueError: If the joint matrix is not positive definite.

        Equation:
            [EQ-SUR-GPX-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

        """
        dimension = len(self.length_scales)
        groups = _groups(self.points, self.gradient_points, dimension)
        noises = [self.value_noise] * len(self.points) + [self.gradient_noise] * (
            len(self.gradient_points) * dimension
        )
        matrix = _joint_matrix(
            self.kernel, self.length_scales, self.signal_variance, groups, noises
        )
        lower = _factor(matrix, "joint (f, grad f) covariance")
        data = [v - self.output_mean for v in self.values]
        for gradient in self.gradients:
            data.extend(gradient)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_alpha", _triangular_solve(lower, data))
        object.__setattr__(
            self, "condition_number", _condition(matrix, _precision(lower))
        )

    def _joint_prediction(self, x: Vector) -> tuple[Vector, Matrix2]:
        """Return the posterior mean and covariance of ``(f(x), grad f(x))``.

        Equation:
            [EQ-SUR-GPX-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

        """
        dimension = len(self.length_scales)
        if len(x) != dimension:
            raise ValueError("x must have the training input dimension")
        groups = _groups(self.points, self.gradient_points, dimension)
        components = tuple(range(-1, dimension))
        rows = [[0.0] * len(self._alpha) for _ in components]
        for point, comps, start in groups:
            blocks = derivative_kernel_blocks(
                self.kernel, x, point, self.length_scales, self.signal_variance
            )
            for r, row in enumerate(components):
                for ib, column in enumerate(comps):
                    rows[r][start + ib] = _block_entry(blocks, row, column)
        prior = derivative_kernel_blocks(
            self.kernel, x, x, self.length_scales, self.signal_variance
        )
        whitened = [_forward(self._lower, row) for row in rows]
        mean = [_sumprod(row, self._alpha) for row in rows]
        mean[0] += self.output_mean
        covariance = [
            [
                _block_entry(prior, a, b) - _sumprod(whitened[ia], whitened[ib])
                for ib, b in enumerate(components)
            ]
            for ia, a in enumerate(components)
        ]
        return tuple(mean), covariance

    def predict_with_variance(self, x: Sequence[float]) -> tuple[float, float]:
        """Return the posterior mean and variance of ``f(x)``.

        Args:
            x: Input vector.

        Returns:
            Latent mean and variance (negative rounding residue clipped to 0).

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-GPX-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

        """
        mean, covariance = self._joint_prediction(as_vector(x, name="x"))
        return mean[0], max(covariance[0][0], 0.0)

    def predict_gradient(self, x: Sequence[float]) -> GradientPrediction:
        """Return the posterior mean and covariance of ``grad f(x)``.

        Args:
            x: Input vector.

        Returns:
            Mean, covariance, and variances of the gradient.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-GPX-005].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

        """
        mean, covariance = self._joint_prediction(as_vector(x, name="x"))
        block = tuple(tuple(row[1:]) for row in covariance[1:])
        return GradientPrediction(
            mean[1:], block, tuple(max(block[i][i], 0.0) for i in range(len(block)))
        )


def fit_derivative_gp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    gradient_points: Sequence[Sequence[float]] = (),
    gradients: Sequence[Sequence[float]] = (),
    *,
    kernel: str = "matern52",
    value_noise: float = 1e-8,
    gradient_noise: float = 1e-8,
    optimize: bool = True,
    initial_signal_variance: float | None = None,
    initial_length_scales: Sequence[float] | None = None,
    max_iterations: int = 60,
) -> DerivativeGaussianProcess:
    """Fit a GP to function values and (optionally) gradients of a smooth function.

    The joint vector ``(f(X), grad f(X_g))`` is Gaussian with the blocks of
    :func:`derivative_kernel_blocks` [EQ-SUR-GPX-004], plus the explicit noise
    variances ``value_noise`` and ``gradient_noise`` on the diagonal
    [EQ-SUR-GPX-005]. The hyperparameters ``(log sigma_f^2, log ell_i)`` are
    fitted by maximizing the joint evidence with BFGS **using central
    finite-difference gradients** (the derivative-kernel gradients with
    respect to the hyperparameters are not implemented); with ``optimize``
    false the initial values are used. With no gradient data this is the
    ordinary GP. A joint matrix that is not positive definite raises
    ``ValueError`` (increase the noise variances; nothing is jittered), and the
    result reports its condition number.

    Args:
        points: Inputs of the function observations.
        values: Function observations (not all equal).
        gradient_points: Inputs of the gradient observations (may be empty).
        gradients: One gradient vector per gradient point.
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        value_noise: Positive variance of the function observations.
        gradient_noise: Positive variance of each gradient component.
        optimize: Fit the hyperparameters.
        initial_signal_variance: Start for ``sigma_f^2`` (default: sample
            variance of ``values``).
        initial_length_scales: Start for ``ell_i`` (default: a third of the
            extent of each input).
        max_iterations: BFGS iteration limit.

    Returns:
        The fitted :class:`DerivativeGaussianProcess`.

    Raises:
        ValueError: If inputs are invalid or the joint matrix is not positive
            definite.

    Equation:
        [EQ-SUR-GPX-004], [EQ-SUR-GPX-005].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9], [REF-SOLAK-ETAL-2002].

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    x = _require_points(points, "points")
    y = as_vector(values, name="values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two points with one value each")
    dimension = len(x[0])
    gx = tuple(as_vector(p, name="gradient_points") for p in gradient_points)
    grads = tuple(as_vector(g, name="gradients") for g in gradients)
    if len(gx) != len(grads):
        raise ValueError("need one gradient per gradient point")
    if any(len(p) != dimension for p in gx) or any(len(g) != dimension for g in grads):
        raise ValueError("gradient inputs and gradients must match the input dimension")
    v_noise = _require_positive(value_noise, "value_noise")
    g_noise = _require_positive(gradient_noise, "gradient_noise")
    count = len(y)
    mean = math.fsum(y) / count
    variance = math.fsum((v - mean) ** 2 for v in y) / (count - 1)
    if variance <= 0.0:
        raise ValueError("values are constant; nothing to regress")
    ranges = []
    for axis in range(dimension):
        column = [p[axis] for p in x + gx]
        width = max(column) - min(column)
        if width <= 0.0:
            raise ValueError(f"input {axis} is constant across the points")
        ranges.append(width)
    s2 = _require_positive(
        variance if initial_signal_variance is None else initial_signal_variance,
        "initial_signal_variance",
    )
    if initial_length_scales is None:
        scales = tuple(width / 3.0 for width in ranges)
    else:
        scales = as_vector(initial_length_scales, name="initial_length_scales")
        if len(scales) != dimension or any(v <= 0.0 for v in scales):
            raise ValueError("initial_length_scales must be positive, one per input")
    groups = _groups(x, gx, dimension)
    noises = [v_noise] * count + [g_noise] * (len(gx) * dimension)
    data = [v - mean for v in y]
    for gradient in grads:
        data.extend(gradient)
    evaluations = [0]

    def negative(theta: Vector) -> float:
        """Return ``-log p`` of the joint data at log parameters ``theta``.

        References:
            [EQ-SUR-GPX-005], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        evaluations[0] += 1
        signal = math.exp(theta[0])
        ells = tuple(math.exp(v) for v in theta[1:])
        lower = _factor(
            _joint_matrix(kernel, ells, signal, groups, noises), "joint covariance"
        )
        alpha = _triangular_solve(lower, data)
        total = len(data)
        return (
            0.5 * _sumprod(data, alpha)
            + 0.5 * _log_det(lower)
            + 0.5 * total * _LOG_TWO_PI
        )

    theta0 = (math.log(s2), *(math.log(v) for v in scales))
    converged = True
    theta = theta0
    if optimize:
        run = minimize_bfgs(
            negative,
            theta0,
            gradient_tolerance=1e-5,
            max_iterations=_require_count(max_iterations, "max_iterations"),
            bounds=[(-_THETA_BOUND, _THETA_BOUND)] * len(theta0),
        )
        theta, converged = run.x, run.converged
    final_value = negative(theta)
    fitted = DerivativeGaussianProcess(
        kernel,
        math.exp(theta[0]),
        tuple(math.exp(v) for v in theta[1:]),
        v_noise,
        g_noise,
        x,
        y,
        gx,
        grads,
        mean,
        -final_value,
        converged,
        evaluations[0],
    )
    _warn_conditioning(fitted.condition_number, "joint (f, grad f) matrix")
    return fitted


def gaussian_process_gradient(
    model: GaussianProcessRegression, points: Sequence[Sequence[float]]
) -> tuple[GradientPrediction, ...]:
    """Return the predictive law of ``grad f`` of a fitted plain GP.

    For a fitted :class:`GaussianProcessRegression` (squared exponential or
    Matern kernels) the gradient of the latent function at ``x_*`` is Gaussian
    with mean ``G' alpha`` and covariance ``H - G' K^-1 G`` where
    ``G_ji = dk(x_*, x_j)/dx_*,i`` and ``H_ij = d^2 k(x, y)/dx_i dy_j`` at
    ``y = x`` [EQ-SUR-GPX-006]. The kernel matrix is rebuilt in output units
    from ``model.hyperparameters`` (noise included) and factorized once for
    all ``points``; the result is therefore independent of the model's
    internal standardization. A kernel matrix that is no longer positive
    definite raises ``ValueError``.

    Args:
        model: Fitted exact GP.
        points: Evaluation inputs.

    Returns:
        One :class:`GradientPrediction` per point.

    Raises:
        TypeError: If ``model`` is not a :class:`GaussianProcessRegression`.
        ValueError: If a point has the wrong dimension.

    Equation:
        [EQ-SUR-GPX-006].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 9].

    """
    if not isinstance(model, GaussianProcessRegression):
        raise TypeError("model must be a GaussianProcessRegression")
    hyper = model.hyperparameters
    scales = hyper.length_scales
    s2 = hyper.signal_variance
    train = model.points
    count = len(train)
    matrix = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1):
            value = derivative_kernel_blocks(
                model.kernel, train[i], train[j], scales, s2
            ).value
            matrix[i][j] = matrix[j][i] = value
        matrix[i][i] += hyper.noise_variance
    lower = _factor(matrix, "kernel matrix")
    alpha = _triangular_solve(lower, [v - model.output_mean for v in model.values])
    results: list[GradientPrediction] = []
    for raw in points:
        x = as_vector(raw, name="point")
        if len(x) != len(scales):
            raise ValueError("point must have the training input dimension")
        cross = [
            derivative_kernel_blocks(model.kernel, x, t, scales, s2).gradient_x
            for t in train
        ]
        mean = tuple(_sumprod([c[i] for c in cross], alpha) for i in range(len(scales)))
        whitened = [_forward(lower, [c[i] for c in cross]) for i in range(len(scales))]
        prior = derivative_kernel_blocks(model.kernel, x, x, scales, s2).hessian
        covariance = tuple(
            tuple(
                prior[i][j] - _sumprod(whitened[i], whitened[j])
                for j in range(len(scales))
            )
            for i in range(len(scales))
        )
        results.append(
            GradientPrediction(
                mean,
                covariance,
                tuple(max(covariance[i][i], 0.0) for i in range(len(scales))),
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# sparse GP: DTC, FITC, VFE
# ---------------------------------------------------------------------------

SPARSE_METHODS = ("dtc", "fitc", "vfe")


def select_inducing_points(
    points: Sequence[Sequence[float]],
    count: int,
    *,
    method: str = "kmeans",
    seed: int | RandomStream | None = None,
) -> tuple[Vector, ...]:
    """Choose ``count`` inducing inputs from the training inputs.

    ``"subset"`` draws a seeded random subset without replacement (returned in
    data order); ``"kmeans"`` runs k-means++ seeding followed by Lloyd
    iterations (at most 100) and returns the cluster centres, so the inducing
    inputs need not be data points. ``count = n`` returns all inputs.

    Args:
        points: Training inputs.
        count: Number ``m`` of inducing inputs, at most the number of distinct
            inputs.
        method: ``"kmeans"`` or ``"subset"``.
        seed: Integer seed or stream.

    Returns:
        ``count`` input vectors.

    Raises:
        ValueError: If the method is unknown or ``count`` exceeds the number
            of (distinct) points.

    Equation:
        [EQ-SUR-GPX-007].

    References:
        [REF-TITSIAS-2009], [REF-SNELSON-GHAHRAMANI-2006].

    """
    x = _require_points(points, "points")
    m = _require_count(count, "count")
    if method not in ("kmeans", "subset"):
        raise ValueError("method must be 'kmeans' or 'subset'")
    distinct = sorted(set(x))
    if m > len(distinct):
        raise ValueError("count exceeds the number of distinct training inputs")
    if m == len(x):
        return x
    generator = resolve_stream(seed).child("gpx-inducing").generator()
    if method == "subset":
        return tuple(x[i] for i in sorted(generator.sample(range(len(x)), m)))

    def nearest(point: Vector, centres: Sequence[Vector]) -> tuple[int, float]:
        """Return the index and squared distance of the closest centre.

        References:
            [REF-TITSIAS-2009]. No equation is evaluated.

        """
        best = min(
            range(len(centres)), key=lambda c: _squared_distance(point, centres[c])
        )
        return best, _squared_distance(point, centres[best])

    centres: list[Vector] = [x[generator.randrange(len(x))]]
    while len(centres) < m:
        weights = [nearest(p, centres)[1] for p in x]
        total = math.fsum(weights)
        threshold = generator.random() * total
        running = 0.0
        chosen = len(x) - 1
        for index, weight in enumerate(weights):
            running += weight
            if running >= threshold and weight > 0.0:
                chosen = index
                break
        centres.append(x[chosen])
    dimension = len(x[0])
    assignment = [-1] * len(x)
    for _ in range(100):
        updated = [nearest(p, centres)[0] for p in x]
        if updated == assignment:
            break
        assignment = updated
        for c in range(m):
            members = [x[i] for i, a in enumerate(assignment) if a == c]
            if members:
                centres[c] = tuple(
                    math.fsum(p[axis] for p in members) / len(members)
                    for axis in range(dimension)
                )
    return tuple(centres)


def _sparse_terms(
    method: str,
    kernel: Kernel,
    x: Sequence[Vector],
    y: Sequence[float],
    z: Sequence[Vector],
    noise: float,
    nugget: float,
) -> tuple[float, list[list[float]], list[list[float]], list[float], Matrix2, float]:
    """Return the sparse objective and the factors needed for prediction.

    With ``V = L_m^-1 K_mn``, ``Q_nn = V'V`` and a diagonal ``D``
    (``sigma_n^2 I`` for DTC and VFE, ``diag(K_nn - Q_nn) + sigma_n^2 I`` for
    FITC), ``log N(y | 0, Q_nn + D) = -1/2 [n log 2 pi + sum log D_i + log det B
    + y'D^-1 y - c'c]`` with ``B = I + V D^-1 V'`` and ``c = L_B^-1 V D^-1 y``.
    VFE subtracts ``tr(K_nn - Q_nn) / (2 sigma_n^2)``.

    Returns:
        ``(objective, L_m, L_B, u = L_B^-T c, K_mm, trace_gap)``.

    Raises:
        ValueError: If ``K_mm`` or ``B`` is not positive definite, or a FITC
            diagonal entry is not positive.

    Equation:
        [EQ-SUR-GPX-007], [EQ-SUR-GPX-008].

    References:
        [REF-TITSIAS-2009], [REF-QUINONERO-CANDELA-RASMUSSEN-2005].

    """
    count, size = len(x), len(z)
    kmm = kernel.matrix(z)
    for a in range(size):
        kmm[a][a] += nugget
    lm = _factor(kmm, "inducing covariance K_mm")
    whitened: list[list[float]] = []
    gaps: list[float] = []
    for point in x:
        column = _forward(lm, [kernel.evaluate(zi, point) for zi in z])
        whitened.append(column)
        gaps.append(kernel.evaluate(point, point) - _sumprod(column, column))
    if method == "fitc":
        diag = [noise + gap for gap in gaps]
        if any(d <= 0.0 for d in diag):
            raise ValueError("FITC diagonal K_nn - Q_nn + sigma_n^2 is not positive")
    else:
        diag = [noise] * count
    b_matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for i in range(count):
        v, d = whitened[i], diag[i]
        scaled = [value / d for value in v]
        for a in range(size):
            row = b_matrix[a]
            sa = scaled[a]
            for b in range(a + 1):
                row[b] += sa * v[b]
            rhs[a] += scaled[a] * y[i]
    for a in range(size):
        b_matrix[a][a] += 1.0
        for b in range(a):
            b_matrix[b][a] = b_matrix[a][b]
    lb = _factor(b_matrix, "sparse system I + V D^-1 V'")
    c = _forward(lb, rhs)
    quad = math.fsum(yi * yi / d for yi, d in zip(y, diag, strict=True)) - _sumprod(
        c, c
    )
    log_det = math.fsum(math.log(d) for d in diag) + _log_det(lb)
    trace_gap = math.fsum(gaps)
    objective = -0.5 * (count * _LOG_TWO_PI + log_det + quad)
    if method == "vfe":
        objective -= trace_gap / (2.0 * noise)
    return objective, lm, lb, _backward(lb, c), kmm, trace_gap


@dataclass(frozen=True)
class SparseGaussianProcess:
    """A fitted sparse GP (DTC, FITC, or VFE) [EQ-SUR-GPX-009].

    ``objective`` is ``log N(y | 0, Q_nn + D)`` for DTC and FITC and Titsias'
    collapsed lower bound for VFE; it is a lower bound of the exact log
    marginal likelihood only for VFE. Outputs are centered (not scaled).

    Args:
        method: ``"dtc"``, ``"fitc"``, or ``"vfe"``.
        kernel: Fitted smooth kernel.
        points: Training inputs.
        values: Training outputs.
        inducing_points: Inducing inputs ``Z``.
        noise_variance: Observation noise variance ``sigma_n^2``.
        nugget: Explicit diagonal added to ``K_mm`` (``K_mm`` units).
        output_mean: Sample mean subtracted from the outputs.
        objective: Value of the sparse log likelihood / bound at the fit.
        trace_gap: ``tr(K_nn - Q_nn)``, the VFE regularizer's numerator.
        converged: Whether the optimizer met its tolerance.
        stop_reason: Optimizer message (``"not optimized"`` if none).
        evaluations: Objective evaluations used by the fit.
        kernel_evaluations: Kernel evaluations in one objective evaluation,
            ``m(m+1)/2 + n m + n`` (versus ``n(n+1)/2`` for the exact GP).
        inducing_condition: 1-norm condition number of ``K_mm`` (with the
            nugget).
        at_bound: Log parameters near the box ``|theta| <= 15``.

    References:
        [REF-TITSIAS-2009], [REF-QUINONERO-CANDELA-RASMUSSEN-2005].

    """

    method: str
    kernel: Kernel
    points: tuple[Vector, ...]
    values: Vector
    inducing_points: tuple[Vector, ...]
    noise_variance: float
    nugget: float
    output_mean: float
    objective: float
    trace_gap: float
    converged: bool
    stop_reason: str
    evaluations: int
    kernel_evaluations: int
    inducing_condition: float = field(init=False, compare=False, default=math.nan)
    at_bound: tuple[str, ...] = ()
    _lm: list[list[float]] = field(init=False, repr=False, compare=False)
    _lb: list[list[float]] = field(init=False, repr=False, compare=False)
    _u: list[float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Recompute the factors of the sparse system once.

        Raises:
            ValueError: If ``K_mm`` or the sparse system is not positive
                definite.

        Equation:
            [EQ-SUR-GPX-009].

        References:
            [REF-TITSIAS-2009].

        """
        centered = [v - self.output_mean for v in self.values]
        _, lm, lb, u, kmm, _ = _sparse_terms(
            self.method,
            self.kernel,
            self.points,
            centered,
            self.inducing_points,
            self.noise_variance,
            self.nugget,
        )
        object.__setattr__(self, "_lm", lm)
        object.__setattr__(self, "_lb", lb)
        object.__setattr__(self, "_u", u)
        object.__setattr__(self, "inducing_condition", _condition(kmm, _precision(lm)))

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and variance at one input.

        ``mu = m + s'u`` and ``sigma^2 = k(x,x) - |s|^2 + |L_B^-1 s|^2`` with
        ``s = L_m^-1 k_m(x)``, i.e. ``K_** - Q_** + K_*m Sigma K_m*`` of
        Titsias and of the DTC/FITC predictives [EQ-SUR-GPX-009].

        Args:
            x: Input vector.
            include_noise: Add ``sigma_n^2`` (a new observation).

        Returns:
            Predictive mean and variance (rounding residue clipped at 0).

        References:
            [REF-TITSIAS-2009], [REF-QUINONERO-CANDELA-RASMUSSEN-2005].

        """
        point = as_vector(x, name="x")
        s = _forward(
            self._lm, [self.kernel.evaluate(z, point) for z in self.inducing_points]
        )
        t = _forward(self._lb, s)
        variance = self.kernel.evaluate(point, point) - _sumprod(s, s) + _sumprod(t, t)
        if include_noise:
            variance += self.noise_variance
        return self.output_mean + _sumprod(s, self._u), max(variance, 0.0)

    def predict(self, x: Sequence[float]) -> float:
        """Return the predictive mean at one input.

        Args:
            x: Input vector.

        Returns:
            Predictive mean.

        References:
            [EQ-SUR-GPX-009], [REF-TITSIAS-2009].

        """
        return self.predict_with_variance(x)[0]


def fit_sparse_gp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    kernel: Kernel,
    *,
    method: str = "vfe",
    inducing_points: Sequence[Sequence[float]] | None = None,
    num_inducing: int | None = None,
    inducing_method: str = "kmeans",
    noise_variance: float | None = None,
    estimate_noise: bool = True,
    optimize: bool = True,
    optimize_inducing: bool = False,
    nugget: float = 1e-8,
    seed: int | RandomStream | None = None,
    max_iterations: int = 60,
) -> SparseGaussianProcess:
    """Fit a sparse GP with DTC, FITC, or the Titsias VFE bound.

    The prior over the ``m`` inducing values ``u = f(Z)`` is ``N(0, K_mm)`` and
    ``f | u`` is approximated by the Nystrom/projected-process mean
    ``K_nm K_mm^-1 u`` [EQ-SUR-GPX-007]. The training-conditional is dropped
    (DTC), kept on the diagonal (FITC), or kept in the bound as
    ``-tr(K_nn - Q_nn) / (2 sigma_n^2)`` (VFE) [EQ-SUR-GPX-008]. The cost is
    ``O(n m^2)`` per likelihood evaluation instead of ``O(n^3)``
    (:func:`sparse_gp_cost`). With ``Z = X`` and a tiny nugget the VFE bound
    equals the exact log marginal likelihood.

    Kernel parameters (and ``log sigma_n^2`` if ``estimate_noise``, and the
    inducing inputs if ``optimize_inducing``, kept inside the bounding box of
    the training inputs) are optimized with BFGS **using central
    finite-difference gradients**. ``K_mm`` gets the explicit ``nugget``;
    a ``K_mm`` or sparse system that is not positive definite raises
    ``ValueError`` (nothing is jittered), and ``inducing_condition`` reports
    the conditioning of ``K_mm``.

    Args:
        points: Training inputs.
        values: Training outputs (not all equal).
        kernel: Smooth kernel (no white component; noise is ``sigma_n^2``).
        method: ``"dtc"``, ``"fitc"``, or ``"vfe"``.
        inducing_points: Explicit inducing inputs, or ``None``.
        num_inducing: Number of inducing inputs if none are given.
        inducing_method: ``"kmeans"`` or ``"subset"`` (see
            :func:`select_inducing_points`).
        noise_variance: Noise variance (start value if estimated); default
            ``0.1`` times the sample variance of the outputs.
        estimate_noise: Optimize ``sigma_n^2``.
        optimize: Optimize the hyperparameters.
        optimize_inducing: Also optimize the inducing inputs.
        nugget: Positive diagonal added to ``K_mm``.
        seed: Seed for the inducing-point selection.
        max_iterations: BFGS iteration limit.

    Returns:
        The fitted :class:`SparseGaussianProcess`.

    Raises:
        TypeError: If ``kernel`` is not a :class:`Kernel`.
        ValueError: If inputs are invalid or the sparse system is singular.

    Equation:
        [EQ-SUR-GPX-007], [EQ-SUR-GPX-008], [EQ-SUR-GPX-009],
        [EQ-SUR-GPX-010].

    References:
        [REF-TITSIAS-2009], [REF-QUINONERO-CANDELA-RASMUSSEN-2005],
        [REF-SNELSON-GHAHRAMANI-2006].

    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a Kernel")
    if kernel.has_white:
        raise ValueError("the sparse kernel must not contain a WhiteKernel")
    if method not in SPARSE_METHODS:
        raise ValueError(f"method must be one of {SPARSE_METHODS}")
    x = _require_points(points, "points")
    y = as_vector(values, name="values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two points with one value each")
    jitter = _require_positive(nugget, "nugget")
    count = len(y)
    mean = math.fsum(y) / count
    variance = math.fsum((v - mean) ** 2 for v in y) / (count - 1)
    if variance <= 0.0:
        raise ValueError("values are constant; nothing to regress")
    centered = [v - mean for v in y]
    if inducing_points is not None:
        z = _require_points(inducing_points, "inducing_points")
        if len(z[0]) != len(x[0]):
            raise ValueError("inducing_points must match the input dimension")
    elif num_inducing is not None:
        z = select_inducing_points(x, num_inducing, method=inducing_method, seed=seed)
    else:
        raise ValueError("give inducing_points or num_inducing")
    sigma2 = _require_positive(
        0.1 * variance if noise_variance is None else noise_variance, "noise_variance"
    )
    dimension, size = len(x[0]), len(z)
    kernel_theta = _check_theta(kernel.log_parameters, kernel.parameter_count)
    names = kernel.parameter_names + (("noise_variance",) if estimate_noise else ())
    head = len(kernel_theta) + (1 if estimate_noise else 0)
    theta0 = (*kernel_theta, *((math.log(sigma2),) if estimate_noise else ()))
    flat_z = tuple(c for point in z for c in point)
    start = theta0 + (flat_z if optimize_inducing else ())
    lows = [min(p[a] for p in x) for a in range(dimension)]
    highs = [max(p[a] for p in x) for a in range(dimension)]
    box: list[tuple[float | None, float | None]] = [
        (-_THETA_BOUND, _THETA_BOUND)
    ] * head
    if optimize_inducing:
        for _ in range(size):
            box.extend((lows[a], highs[a]) for a in range(dimension))
    evaluations = [0]

    def decode(theta: Sequence[float]) -> tuple[Kernel, float, tuple[Vector, ...]]:
        """Split ``theta`` into kernel, noise variance, and inducing inputs.

        References:
            [EQ-SUR-GPX-008].

        """
        k = kernel.with_log_parameters(theta[: len(kernel_theta)])
        noise = math.exp(theta[len(kernel_theta)]) if estimate_noise else sigma2
        if optimize_inducing:
            tail = theta[head:]
            zs = tuple(
                tuple(tail[i * dimension : (i + 1) * dimension]) for i in range(size)
            )
        else:
            zs = z
        return k, noise, zs

    def negative(theta: Vector) -> float:
        """Return minus the sparse objective at ``theta``.

        References:
            [EQ-SUR-GPX-008].

        """
        evaluations[0] += 1
        k, noise, zs = decode(theta)
        return -_sparse_terms(method, k, x, centered, zs, noise, jitter)[0]

    converged, reason, theta = True, "not optimized", start
    if optimize and (head or optimize_inducing):
        run = minimize_bfgs(
            negative,
            start,
            gradient_tolerance=1e-5,
            max_iterations=_require_count(max_iterations, "max_iterations"),
            bounds=box,
        )
        converged, reason, theta = run.converged, run.message, run.x
    k_final, noise_final, z_final = decode(theta)
    terms = _sparse_terms(method, k_final, x, centered, z_final, noise_final, jitter)
    evaluations[0] += 1
    fitted = SparseGaussianProcess(
        method,
        k_final,
        x,
        y,
        z_final,
        noise_final,
        jitter,
        mean,
        terms[0],
        terms[5],
        converged,
        reason,
        evaluations[0],
        size * (size + 1) // 2 + count * size + count,
        at_bound=_theta_names_at_bound(theta[:head], names),
    )
    _warn_conditioning(fitted.inducing_condition, "inducing covariance K_mm")
    return fitted


@dataclass(frozen=True, slots=True)
class SparseGPCost:
    """Leading-order cost of one likelihood evaluation [EQ-SUR-GPX-010].

    Counts are multiply-adds (kernel-evaluation counts are exact).

    Args:
        n: Training size.
        m: Number of inducing inputs.
        dimension: Input dimension ``d``.
        exact_flops: ``n^2 d / 2 + n^3 / 6 + n^2`` (assembly, Cholesky, solve).
        sparse_flops: ``n m d + m^2 d / 2 + n m^2 + m^3 / 3 + n m``.
        exact_kernel_evaluations: ``n(n+1)/2``.
        sparse_kernel_evaluations: ``m(m+1)/2 + n m + n``.
        speedup: ``exact_flops / sparse_flops``.

    References:
        [REF-QUINONERO-CANDELA-RASMUSSEN-2005], [REF-GOLUB-VANLOAN-2013].

    """

    n: int
    m: int
    dimension: int
    exact_flops: float
    sparse_flops: float
    exact_kernel_evaluations: int
    sparse_kernel_evaluations: int
    speedup: float


def sparse_gp_cost(n: int, m: int, dimension: int = 1) -> SparseGPCost:
    """Return the leading-order cost model of exact versus sparse GP fitting.

    Exact: assembling ``K`` costs ``n^2 d / 2``, the Cholesky factorization
    ``n^3 / 6`` and the solve ``n^2`` multiply-adds. Sparse (DTC, FITC, VFE):
    ``K_nm`` costs ``n m d``, ``K_mm`` ``m^2 d / 2``, the whitening
    ``L_m^-1 K_mn`` ``n m^2 / 2``, the system ``V D^-1 V'`` ``n m^2 / 2``, two
    ``m x m`` Cholesky factors ``m^3 / 3``, and the right-hand side ``n m``
    [EQ-SUR-GPX-010]. The model is not a timing measurement.

    Args:
        n: Number of training points.
        m: Number of inducing points, ``1 <= m <= n``.
        dimension: Input dimension.

    Returns:
        The :class:`SparseGPCost`.

    Raises:
        ValueError: If ``m > n`` or a count is not positive.

    Equation:
        [EQ-SUR-GPX-010].

    References:
        [REF-QUINONERO-CANDELA-RASMUSSEN-2005], [REF-GOLUB-VANLOAN-2013].

    """
    n = _require_count(n, "n")
    m = _require_count(m, "m")
    d = _require_count(dimension, "dimension")
    if m > n:
        raise ValueError("m must not exceed n")
    exact = n * n * d / 2.0 + n**3 / 6.0 + n * n
    sparse = n * m * d + m * m * d / 2.0 + n * m * m + m**3 / 3.0 + n * m
    return SparseGPCost(
        n,
        m,
        d,
        exact,
        sparse,
        n * (n + 1) // 2,
        m * (m + 1) // 2 + n * m + n,
        exact / sparse,
    )


# ---------------------------------------------------------------------------
# most-likely heteroscedastic GP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeteroscedasticGaussianProcess:
    """A most-likely heteroscedastic GP (Kersting et al.) [EQ-SUR-GPX-011].

    ``y_i = f(x_i) + eps_i`` with ``eps_i ~ N(0, exp(r(x_i)))`` where ``f`` and
    the log noise level ``r`` are GPs. The two models are fitted by the
    alternation below until the stopping rule fires.

    Args:
        mean_model: GP of ``f`` fitted with the per-point noise
            ``training_noise_variances`` (:class:`KernelGaussianProcess`).
        noise_model: GP of the log noise variance, fitted to
            ``z_i = log((y_i - mu_i)^2 + s_i^2) + log_offset`` from the final
            ``mean_model``.
        training_noise_variances: Noise variances used to fit ``mean_model``.
        iterations: Number of ``mean_model`` fits performed.
        converged: ``True`` if the stopping tolerance was met (or the
            likelihood stopped increasing); ``False`` if the iteration limit
            ended the loop.
        stop_reason: ``"converged"``, ``"likelihood_decreased"`` (the earlier,
            best iterate is returned), or ``"max_iterations"``.
        log_likelihood_history: ``log p(y | theta_t, noise_t)`` per iteration.

    References:
        [REF-KERSTING-ETAL-2007].

    """

    mean_model: KernelGaussianProcess
    noise_model: KernelGaussianProcess
    training_noise_variances: Vector
    iterations: int
    converged: bool
    stop_reason: str
    log_likelihood_history: Vector

    def noise_variance(self, x: Sequence[float]) -> float:
        """Return the estimated noise variance ``exp(r(x))`` at ``x``.

        Args:
            x: Input vector.

        Returns:
            Positive noise variance (output units squared).

        Equation:
            [EQ-SUR-GPX-011].

        References:
            [REF-KERSTING-ETAL-2007].

        """
        return math.exp(self.noise_model.predict(x))

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the latent predictive mean and variance at ``x``.

        Args:
            x: Input vector.
            include_noise: Add the input-dependent noise variance.

        Returns:
            Mean and variance (output units).

        Equation:
            [EQ-SUR-GPX-011].

        References:
            [REF-KERSTING-ETAL-2007].

        """
        mean, variance = self.mean_model.predict_with_variance(x)
        if include_noise:
            variance += self.noise_variance(x)
        return mean, variance


def fit_heteroscedastic_gp(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    kernel: Kernel,
    *,
    noise_kernel: Kernel | None = None,
    nugget: float = 1e-8,
    initial_noise_variance: float | None = None,
    max_iterations: int = 8,
    tolerance: float = 1e-3,
    log_offset: float = 0.0,
    seed: int | RandomStream | None = None,
) -> HeteroscedasticGaussianProcess:
    """Fit a most-likely heteroscedastic GP by alternating two GPs.

    Starting from the constant noise ``sigma_0^2`` the loop is, for
    ``t = 0, 1, ...`` [EQ-SUR-GPX-011]:

    1. fit the main GP (kernel hyperparameters by type-II ML) with the
       per-point noise ``n_i^(t)`` and record ``L_t = log p(y | theta_t, n^(t))``;
    2. from this fit take ``mu_i`` and ``s_i^2`` (latent posterior mean and
       variance at ``x_i``) and form ``v_i = (y_i - mu_i)^2 + s_i^2 =
       E[eps_i^2 | y]``, the E-step of an EM update of a per-point noise
       variance; this is the infinite-sample limit of Kersting's sampled
       estimator;
    3. fit a GP (``noise_kernel`` + white noise) to ``z_i = log v_i +
       log_offset`` and set ``n_i^(t+1) = exp(m_z(x_i))``.

    **Stopping rule.** With ``rho_t = (L_t - L_(t-1)) / (1 + |L_(t-1)|)``: stop
    with ``"converged"`` when ``|rho_t| <= tolerance``; stop with
    ``"likelihood_decreased"`` and return the best earlier iterate when
    ``rho_t < -tolerance``; stop with ``"max_iterations"`` (``converged``
    false) otherwise after ``max_iterations`` fits. Because ``z`` is the log of
    a noisy positive quantity, ``E[log v]`` is below ``log E[v]`` (Jensen);
    ``log_offset`` (default ``0``, never applied silently) lets the caller add
    a correction such as ``1.2704 = -E[log chi^2_1]`` for the raw-residual
    case. Positive-definiteness failures raise ``ValueError``.

    Args:
        points: Training inputs.
        values: Training outputs.
        kernel: Smooth kernel of ``f`` (no white component).
        noise_kernel: Smooth kernel of the log noise (default: squared
            exponential with unit variance and length scale a third of each
            input range).
        nugget: Explicit positive diagonal variance of the main GP.
        initial_noise_variance: ``sigma_0^2`` (default: 0.1 times the sample
            variance of the outputs).
        max_iterations: Maximum number of main-GP fits.
        tolerance: Relative likelihood tolerance ``> 0``.
        log_offset: Constant added to ``log v_i``.
        seed: Seed for the optimizer restarts.

    Returns:
        The fitted :class:`HeteroscedasticGaussianProcess`.

    Raises:
        TypeError: If a kernel is not a :class:`Kernel`.
        ValueError: If inputs are invalid, a kernel contains white noise, a
            matrix is not positive definite, or a noise estimate is not
            finite and positive.

    Equation:
        [EQ-SUR-GPX-011].

    References:
        [REF-KERSTING-ETAL-2007], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a Kernel")
    if noise_kernel is not None and not isinstance(noise_kernel, Kernel):
        raise TypeError("noise_kernel must be a Kernel")
    if kernel.has_white or (noise_kernel is not None and noise_kernel.has_white):
        raise ValueError("kernels must not contain a WhiteKernel")
    x = _require_points(points, "points")
    y = as_vector(values, name="values")
    count = len(y)
    if len(x) != count or count < 4:
        raise ValueError("need at least four points with one value each")
    rounds = _require_count(max_iterations, "max_iterations")
    tol = _require_positive(tolerance, "tolerance")
    offset = require_finite_real(log_offset, name="log_offset")
    variance = math.fsum((v - math.fsum(y) / count) ** 2 for v in y) / (count - 1)
    if variance <= 0.0:
        raise ValueError("values are constant; nothing to regress")
    sigma0 = _require_positive(
        0.1 * variance if initial_noise_variance is None else initial_noise_variance,
        "initial_noise_variance",
    )
    if noise_kernel is None:
        widths = tuple(
            max(p[a] for p in x) - min(p[a] for p in x) for a in range(len(x[0]))
        )
        if any(w <= 0.0 for w in widths):
            raise ValueError("an input is constant across the training points")
        noise_kernel = StationaryKernel(
            "squared_exponential", 1.0, tuple(w / 3.0 for w in widths)
        )
    stream = resolve_stream(seed)
    noise: Vector = (sigma0,) * count
    current = kernel
    history: list[float] = []
    best: tuple[float, KernelGaussianProcess, KernelGaussianProcess, Vector] | None = (
        None
    )
    stop, converged = "max_iterations", False
    for step in range(rounds):
        main = fit_kernel_gp(
            x,
            y,
            current,
            nugget=nugget,
            noise_variances=noise,
            restarts=2 if step == 0 else 1,
            seed=stream.child("main", step),
        )
        likelihood = main.log_marginal_likelihood
        stats = [main.predict_with_variance(p) for p in x]
        targets = []
        for yi, (mu, var) in zip(y, stats, strict=True):
            second = (yi - mu) ** 2 + var
            if second <= 0.0:
                raise ValueError(
                    "zero residual and zero posterior variance: the noise level "
                    "is undefined"
                )
            targets.append(math.log(second) + offset)
        log_noise = fit_kernel_gp(
            x,
            targets,
            noise_kernel + WhiteKernel(1.0),
            nugget=nugget,
            restarts=1,
            seed=stream.child("noise", step),
        )
        history.append(likelihood)
        if best is None or likelihood > best[0]:
            best = (likelihood, main, log_noise, noise)
        if step > 0:
            rho = (likelihood - history[-2]) / (1.0 + abs(history[-2]))
            if rho < -tol:
                stop, converged = "likelihood_decreased", True
                break
            if abs(rho) <= tol:
                stop, converged = "converged", True
                break
        noise = tuple(math.exp(log_noise.predict(p)) for p in x)
        if not all(math.isfinite(v) and v > 0.0 for v in noise):
            raise ValueError("estimated noise variance is not finite and positive")
        current = main.kernel
    assert best is not None
    _, main, log_noise, used = best
    return HeteroscedasticGaussianProcess(
        main, log_noise, used, len(history), converged, stop, tuple(history)
    )


# ---------------------------------------------------------------------------
# multi-output GP: linear model of coregionalization
# ---------------------------------------------------------------------------


def _lmc_unpack(
    theta: Sequence[float],
    kernels: Sequence[Kernel],
    outputs: int,
    rank: int,
    estimate_noise: bool,
) -> tuple[list[Kernel], list[list[list[float]]], list[list[float]], list[float]]:
    """Split ``theta`` into kernels, ``W_r``, ``kappa_r`` and output noises.

    Layout per latent kernel ``r``: its log parameters, then ``W_r`` row-major
    (``outputs x rank``, raw values), then ``log kappa_r`` (``outputs``);
    finally ``log sigma_p^2`` per output if ``estimate_noise``.

    Equation:
        [EQ-SUR-GPX-012].

    References:
        [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

    """
    cursor = 0
    ks: list[Kernel] = []
    ws: list[list[list[float]]] = []
    kappas: list[list[float]] = []
    for kernel in kernels:
        size = kernel.parameter_count
        ks.append(kernel.with_log_parameters(theta[cursor : cursor + size]))
        cursor += size
        ws.append(
            [
                list(theta[cursor + p * rank : cursor + (p + 1) * rank])
                for p in range(outputs)
            ]
        )
        cursor += outputs * rank
        kappas.append([math.exp(v) for v in theta[cursor : cursor + outputs]])
        cursor += outputs
    noises = (
        [math.exp(v) for v in theta[cursor : cursor + outputs]]
        if estimate_noise
        else [0.0] * outputs
    )
    return ks, ws, kappas, noises


def _coregionalization(w: Sequence[Sequence[float]], kappa: Sequence[float]) -> Matrix2:
    """Return ``B = W W' + diag(kappa)``.

    Equation:
        [EQ-SUR-GPX-012].

    References:
        [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

    """
    size = len(w)
    return [
        [_sumprod(w[p], w[q]) + (kappa[p] if p == q else 0.0) for q in range(size)]
        for p in range(size)
    ]


def _lmc_marginal(
    theta: Sequence[float],
    kernels: Sequence[Kernel],
    rank: int,
    outputs: int,
    labels: Sequence[int],
    x: Sequence[Vector],
    y: Sequence[float],
    nugget: float,
    estimate_noise: bool,
    *,
    gradient: bool,
) -> tuple[float, Vector, list[list[float]], Vector, Matrix2]:
    """Return the LMC log marginal likelihood, its exact gradient, factor, alpha, K.

    ``K_ij = sum_r B^r[p_i, p_j] k_r(x_i, x_j) + delta_ij (sigma_(p_i)^2 +
    nugget)`` and ``d log p / d theta = 1/2 tr((alpha alpha' - K^-1) dK)`` with
    ``M^r = (alpha alpha' - K^-1) o K_r``:

    * kernel parameter ``j`` of ``r``: ``1/2 sum_ij W_ij B^r[p_i,p_j] dK_r,ij``;
    * ``W_r[p, a]``: ``sum_(i in p) (M^r W_r[:, a])_i`` with ``W_r[:, a]``
      indexed by the output of each row;
    * ``log kappa_r[p]``: ``1/2 kappa_r[p] sum_(i,j in p) M^r_ij``;
    * ``log sigma_p^2``: ``1/2 sigma_p^2 sum_(i in p) W_ii``.

    Raises:
        ValueError: If ``K`` is not positive definite.

    Equation:
        [EQ-SUR-GPX-012].

    References:
        [REF-ALVAREZ-ROSASCO-LAWRENCE-2012], [REF-RASMUSSEN-WILLIAMS-2006,
        Section 5.4].

    """
    ks, ws, kappas, noises = _lmc_unpack(theta, kernels, outputs, rank, estimate_noise)
    count = len(x)
    latent = [k.matrix(x) for k in ks]
    bs = [_coregionalization(w, kp) for w, kp in zip(ws, kappas, strict=True)]
    matrix = [[0.0] * count for _ in range(count)]
    for kr, br in zip(latent, bs, strict=True):
        for i in range(count):
            row, krow, brow = matrix[i], kr[i], br[labels[i]]
            for j in range(count):
                row[j] += brow[labels[j]] * krow[j]
    for i in range(count):
        matrix[i][i] += noises[labels[i]] + nugget
    lower = _factor(matrix, "multi-output kernel matrix")
    alpha = _triangular_solve(lower, y)
    value = -0.5 * _sumprod(y, alpha) - 0.5 * _log_det(lower)
    value -= 0.5 * count * _LOG_TWO_PI
    if not gradient:
        return value, (), lower, alpha, matrix
    precision = _precision(lower)
    weight = [
        [alpha[i] * alpha[j] - precision[i][j] for j in range(count)]
        for i in range(count)
    ]
    grads: list[float] = []
    members = [[i for i in range(count) if labels[i] == p] for p in range(outputs)]
    for r, kernel in enumerate(ks):
        br = bs[r]
        for dk in kernel.matrix_gradients(x):
            grads.append(
                0.5
                * math.fsum(
                    weight[i][j] * br[labels[i]][labels[j]] * dk[i][j]
                    for i in range(count)
                    for j in range(count)
                )
            )
        product = [
            [weight[i][j] * latent[r][i][j] for j in range(count)] for i in range(count)
        ]
        w_grad = [[0.0] * rank for _ in range(outputs)]
        for a in range(rank):
            column = [ws[r][labels[j]][a] for j in range(count)]
            for p in range(outputs):
                w_grad[p][a] = math.fsum(
                    _sumprod(product[i], column) for i in members[p]
                )
        for p in range(outputs):
            grads.extend(w_grad[p])
        for p in range(outputs):
            grads.append(
                0.5
                * kappas[r][p]
                * math.fsum(product[i][j] for i in members[p] for j in members[p])
            )
    if estimate_noise:
        for p in range(outputs):
            grads.append(0.5 * noises[p] * math.fsum(weight[i][i] for i in members[p]))
    return value, tuple(grads), lower, alpha, matrix


@dataclass(frozen=True)
class MultiOutputGaussianProcess:
    """A linear-model-of-coregionalization GP on heterotopic data.

    ``Cov(f_p(x), f_q(x')) = sum_r B^r[p, q] k_r(x, x')`` with
    ``B^r = W_r W_r' + diag(kappa_r)`` [EQ-SUR-GPX-012]; one latent kernel with
    ``rank 1`` is the intrinsic coregionalization model. Each output is
    standardized by its own sample mean and standard deviation; the matrices
    ``W_r``, ``kappa_r`` and the noise variances are in those standardized
    units, while predictions are returned in original units. Different outputs
    may be observed at different inputs (heterotopic data).

    Args:
        kernels: Fitted latent kernels ``k_r``.
        rank: Columns of every ``W_r``.
        w: ``w[r][p]`` is row ``p`` of ``W_r``.
        kappa: ``kappa[r][p]``.
        noise_variances: Observation noise per output (standardized units).
        nugget: Explicit diagonal variance (standardized units).
        inputs: Training inputs, one tuple per output.
        values: Training outputs, one tuple per output.
        output_means: Mean of every output.
        output_scales: Standard deviation of every output.
        log_marginal_likelihood: Evidence in standardized units.
        converged: Whether the best start met the optimizer tolerance.
        stop_reason: Optimizer message.
        evaluations: Likelihood evaluations over all starts.
        condition_number: 1-norm condition number of the joint matrix.
        at_bound: Log parameters near the box ``|theta| <= 15``.

    References:
        [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

    """

    kernels: tuple[Kernel, ...]
    rank: int
    w: tuple[tuple[Vector, ...], ...]
    kappa: tuple[Vector, ...]
    noise_variances: Vector
    nugget: float
    inputs: tuple[tuple[Vector, ...], ...]
    values: tuple[Vector, ...]
    output_means: Vector
    output_scales: Vector
    log_marginal_likelihood: float
    converged: bool
    stop_reason: str
    evaluations: int
    at_bound: tuple[str, ...] = ()
    condition_number: float = field(init=False, compare=False, default=math.nan)
    _labels: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _points: tuple[Vector, ...] = field(init=False, repr=False, compare=False)
    _lower: list[list[float]] = field(init=False, repr=False, compare=False)
    _alpha: Vector = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Factorize the joint kernel matrix once.

        Raises:
            ValueError: If the matrix is not positive definite.

        Equation:
            [EQ-SUR-GPX-012].

        References:
            [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

        """
        labels = tuple(p for p, block in enumerate(self.inputs) for _ in block)
        points = tuple(point for block in self.inputs for point in block)
        data = [
            (v - m) / s
            for block, m, s in zip(
                self.values, self.output_means, self.output_scales, strict=True
            )
            for v in block
        ]
        matrix = [[0.0] * len(points) for _ in points]
        for kernel, w, kappa in zip(self.kernels, self.w, self.kappa, strict=True):
            b = _coregionalization(w, kappa)
            latent = kernel.matrix(points)
            for i, li in enumerate(labels):
                for j, lj in enumerate(labels):
                    matrix[i][j] += b[li][lj] * latent[i][j]
        for i, li in enumerate(labels):
            matrix[i][i] += self.noise_variances[li] + self.nugget
        lower = _factor(matrix, "multi-output kernel matrix")
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_points", points)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_alpha", _triangular_solve(lower, data))
        object.__setattr__(
            self, "condition_number", _condition(matrix, _precision(lower))
        )

    def coregionalization_matrices(self) -> tuple[Matrix2, ...]:
        """Return ``B^r = W_r W_r' + diag(kappa_r)`` for every latent kernel.

        Returns:
            One ``P x P`` matrix per latent kernel (standardized units).

        Equation:
            [EQ-SUR-GPX-012].

        References:
            [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

        """
        return tuple(
            _coregionalization(w, kp) for w, kp in zip(self.w, self.kappa, strict=True)
        )

    def output_correlation(self, first: int, second: int) -> float:
        """Return the prior correlation of two outputs at one input.

        ``corr = S[p, q] / sqrt(S[p, p] S[q, q])`` with ``S = sum_r B^r
        k_r(x, x)`` at coincident inputs when every latent kernel has unit
        value there; for kernels with different variances the weights
        ``k_r(x, x)`` are included.

        Args:
            first: Output index ``p``.
            second: Output index ``q``.

        Returns:
            Correlation in ``[-1, 1]``.

        Raises:
            ValueError: If an index is out of range.

        Equation:
            [EQ-SUR-GPX-012].

        References:
            [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

        """
        size = len(self.inputs)
        if not (0 <= first < size and 0 <= second < size):
            raise ValueError("output index out of range")
        origin = (0.0,) * len(self.inputs[0][0])
        total = [[0.0] * size for _ in range(size)]
        for kernel, w, kappa in zip(self.kernels, self.w, self.kappa, strict=True):
            scale = kernel.evaluate(origin, origin)
            b = _coregionalization(w, kappa)
            for p in range(size):
                for q in range(size):
                    total[p][q] += scale * b[p][q]
        return total[first][second] / math.sqrt(
            total[first][first] * total[second][second]
        )

    def predict_with_variance(
        self, output: int, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and variance of one output at ``x``.

        ``mu = m_p + s_p c' alpha`` with ``c_j = sum_r B^r[p, p_j] k_r(x, x_j)``
        and ``sigma^2 = s_p^2 (sum_r B^r[p, p] k_r(x, x) - |L^-1 c|^2)``
        [EQ-SUR-GPX-012].

        Args:
            output: Output index ``p``.
            x: Input vector.
            include_noise: Add ``s_p^2 (sigma_p^2 + nugget)``.

        Returns:
            Mean and variance in original units.

        Raises:
            ValueError: If the index or the input is invalid.

        Equation:
            [EQ-SUR-GPX-012].

        References:
            [REF-ALVAREZ-ROSASCO-LAWRENCE-2012].

        """
        if not 0 <= output < len(self.inputs):
            raise ValueError("output index out of range")
        point = as_vector(x, name="x")
        cross = [0.0] * len(self._points)
        prior = 0.0
        for kernel, w, kappa in zip(self.kernels, self.w, self.kappa, strict=True):
            b = _coregionalization(w, kappa)
            prior += b[output][output] * kernel.evaluate(point, point)
            for j, (train, label) in enumerate(
                zip(self._points, self._labels, strict=True)
            ):
                cross[j] += b[output][label] * kernel.evaluate(point, train)
        whitened = _forward(self._lower, cross)
        variance = prior - _sumprod(whitened, whitened)
        if include_noise:
            variance += self.noise_variances[output] + self.nugget
        scale = self.output_scales[output]
        return (
            self.output_means[output] + scale * _sumprod(cross, self._alpha),
            scale * scale * max(variance, 0.0),
        )


def fit_multi_output_gp(
    inputs: Sequence[Sequence[Sequence[float]]],
    values: Sequence[Sequence[float]],
    kernels: Kernel | Sequence[Kernel],
    *,
    rank: int = 1,
    nugget: float = 1e-8,
    estimate_noise: bool = True,
    initial_noise_variance: float = 0.05,
    restarts: int = 2,
    seed: int | RandomStream | None = None,
    max_iterations: int = 150,
) -> MultiOutputGaussianProcess:
    """Fit an LMC / ICM multi-output GP by joint type-II maximum likelihood.

    The covariance of the stacked observations is ``K = sum_r (B^r[p_i, p_j]
    k_r(x_i, x_j)) + diag(sigma_(p_i)^2 + nugget)`` with ``B^r = W_r W_r' +
    diag(kappa_r)``; one kernel is the intrinsic coregionalization model (ICM),
    several are the linear model of coregionalization (LMC) [EQ-SUR-GPX-012].
    The log evidence is maximized over the latent-kernel parameters, ``W_r``,
    ``log kappa_r`` and (if ``estimate_noise``) the log output noises with
    BFGS and the exact gradient (see :func:`_lmc_marginal`). The first start
    uses ``W_r[p, 0] = 0.7`` (a positive correlation), further starts draw
    ``W_r`` from ``N(0, 0.7^2)``; the lowest objective is kept. ``W_r`` is
    identifiable only up to a rotation (and a joint scale with the kernel
    variance), which leaves the likelihood flat in those directions. A
    non-positive-definite start is skipped; if none works ``ValueError`` is
    raised and no jitter is added. Latent kernels must be smooth (no white).

    Args:
        inputs: Per output, its training inputs (may differ across outputs).
        values: Per output, its training values (at least two, not constant).
        kernels: One latent kernel or a sequence (LMC).
        rank: Columns of every ``W_r`` (``>= 1``).
        nugget: Explicit positive diagonal variance (standardized units).
        estimate_noise: Estimate one noise variance per output.
        initial_noise_variance: Start (or fixed value if not estimated) of
            every output's noise variance, in standardized units.
        restarts: Total number of optimizer starts.
        seed: Seed for the restart draws.
        max_iterations: BFGS iteration limit per start.

    Returns:
        The fitted :class:`MultiOutputGaussianProcess`.

    Raises:
        TypeError: If a kernel is not a :class:`Kernel`.
        ValueError: If the data are invalid or no start is positive definite.

    Equation:
        [EQ-SUR-GPX-012].

    References:
        [REF-ALVAREZ-ROSASCO-LAWRENCE-2012], [REF-RASMUSSEN-WILLIAMS-2006,
        Chapter 9].

    """
    latent = (kernels,) if isinstance(kernels, Kernel) else tuple(kernels)
    if not latent or not all(isinstance(k, Kernel) for k in latent):
        raise TypeError("kernels must be a Kernel or a non-empty sequence of Kernel")
    if any(k.has_white for k in latent):
        raise ValueError("latent kernels must not contain a WhiteKernel")
    q = _require_count(rank, "rank")
    jitter = _require_positive(nugget, "nugget")
    start_noise = _require_positive(initial_noise_variance, "initial_noise_variance")
    _require_count(restarts, "restarts")
    outputs = len(inputs)
    if outputs < 1 or len(values) != outputs:
        raise ValueError("need one input set and one value vector per output")
    xs = tuple(_require_points(block, "inputs") for block in inputs)
    ys = tuple(as_vector(block, name="values") for block in values)
    dimension = len(xs[0][0])
    means, scales, labels, points, data = [], [], [], [], []
    for p, (block, target) in enumerate(zip(xs, ys, strict=True)):
        if len(block) != len(target) or len(target) < 2:
            raise ValueError("every output needs equally many inputs and values (>= 2)")
        if len(block[0]) != dimension:
            raise ValueError("all outputs must share the input dimension")
        mean = math.fsum(target) / len(target)
        scale = math.sqrt(
            math.fsum((v - mean) ** 2 for v in target) / (len(target) - 1)
        )
        if scale <= 0.0:
            raise ValueError(f"output {p} is constant; nothing to regress")
        means.append(mean)
        scales.append(scale)
        for point, v in zip(block, target, strict=True):
            labels.append(p)
            points.append(point)
            data.append((v - mean) / scale)
    generator = resolve_stream(seed).child("gpx-lmc").generator()

    def build_start(random_w: bool) -> Vector:
        """Assemble a start vector (kernels, W, log kappa, log noise).

        References:
            [EQ-SUR-GPX-012].

        """
        theta: list[float] = []
        for kernel in latent:
            theta.extend(kernel.log_parameters)
            for _p in range(outputs):
                for a in range(q):
                    if random_w:
                        theta.append(generator.gauss(0.0, 0.7))
                    else:
                        theta.append(0.7 if a == 0 else 0.1)
            theta.extend([math.log(0.1)] * outputs)
        if estimate_noise:
            theta.extend([math.log(start_noise)] * outputs)
        return tuple(theta)

    first = build_start(False)
    names: list[str] = []
    box: list[tuple[float | None, float | None]] = []
    for r, kernel in enumerate(latent):
        for name in kernel.parameter_names:
            names.append(f"k{r}.{name}")
            box.append((-_THETA_BOUND, _THETA_BOUND))
        for p in range(outputs):
            for a in range(q):
                names.append(f"w{r}[{p},{a}]")
                box.append((None, None))
        for p in range(outputs):
            names.append(f"kappa{r}[{p}]")
            box.append((-_THETA_BOUND, _THETA_BOUND))
    if estimate_noise:
        for p in range(outputs):
            names.append(f"noise[{p}]")
            box.append((-_THETA_BOUND, _THETA_BOUND))
    cache: dict[Vector, tuple[float, Vector]] = {}
    evaluations = [0]

    def objective(theta: Vector) -> tuple[float, Vector]:
        """Return ``-log p`` and its gradient, cached per ``theta``.

        References:
            [EQ-SUR-GPX-012], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        key = tuple(theta)
        hit = cache.get(key)
        if hit is not None:
            return hit
        evaluations[0] += 1
        value, grads, _, _, _ = _lmc_marginal(
            key,
            latent,
            q,
            outputs,
            labels,
            points,
            data,
            jitter,
            estimate_noise,
            gradient=True,
        )
        result = (-value, tuple(-g for g in grads))
        if len(cache) > 64:
            del cache[next(iter(cache))]
        cache[key] = result
        return result

    starts = [first] + [build_start(True) for _ in range(restarts - 1)]
    best: tuple[float, Vector, bool, str] | None = None
    for begin in starts:
        try:
            run = minimize_bfgs(
                lambda t: objective(tuple(t))[0],
                begin,
                gradient=lambda t: objective(tuple(t))[1],
                gradient_tolerance=1e-5,
                max_iterations=_require_count(max_iterations, "max_iterations"),
                bounds=box,
            )
        except ValueError:
            continue
        if best is None or run.value < best[0]:
            best = (run.value, run.x, run.converged, run.message)
    if best is None:
        raise ValueError(
            "no start produced a positive-definite kernel matrix; increase the "
            "nugget (no jitter is added)"
        )
    ks, ws, kappas, noises = _lmc_unpack(best[1], latent, outputs, q, estimate_noise)
    if not estimate_noise:
        noises = [start_noise] * outputs
    fitted = MultiOutputGaussianProcess(
        tuple(ks),
        q,
        tuple(tuple(tuple(row) for row in w) for w in ws),
        tuple(tuple(kp) for kp in kappas),
        tuple(noises),
        jitter,
        xs,
        ys,
        tuple(means),
        tuple(scales),
        -best[0],
        best[2],
        best[3],
        evaluations[0],
        at_bound=_bounded_hits(best[1], names, box),
    )
    _warn_conditioning(fitted.condition_number, "joint multi-output matrix")
    return fitted


def _bounded_hits(
    theta: Sequence[float],
    names: Sequence[str],
    box: Sequence[tuple[float | None, float | None]],
) -> tuple[str, ...]:
    """Name the log parameters (those with finite bounds) near their bounds.

    References:
        [REF-NOCEDAL-WRIGHT-2006]. No equation is evaluated.

    """
    hits: list[str] = []
    for name, value, (low, high) in zip(names, theta, box, strict=True):
        if low is None or high is None:
            continue
        if value - low <= _BOUND_TOLERANCE:
            hits.append(f"{name}:lower")
        elif high - value <= _BOUND_TOLERANCE:
            hits.append(f"{name}:upper")
    return tuple(hits)


# ---------------------------------------------------------------------------
# linear-operator-constrained GP
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearDifferentialOperator:
    """Constant-coefficient operator ``L = sum_m a_m d^m / dx^m`` in one dimension.

    Args:
        coefficients: ``(a_0, a_1, ..., a_M)`` with ``a_M != 0``.

    Raises:
        ValueError: If the coefficients are empty, not finite, or all zero.

    References:
        [REF-JIDLING-ETAL-2017], [REF-RAISSI-ETAL-2017].

    """

    coefficients: Vector

    def __post_init__(self) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If they are empty, not finite, or all zero.

        References:
            [EQ-SUR-GPX-013], [REF-JIDLING-ETAL-2017].

        """
        coefficients = as_vector(self.coefficients, name="coefficients")
        if not coefficients or all(a == 0.0 for a in coefficients):
            raise ValueError("coefficients must be non-empty and not all zero")

    @classmethod
    def derivative(cls) -> LinearDifferentialOperator:
        """Return ``L = d/dx``.

        Returns:
            The first-derivative operator.

        References:
            [EQ-SUR-GPX-013], [REF-JIDLING-ETAL-2017].

        """
        return cls((0.0, 1.0))

    @classmethod
    def helmholtz(cls, c: float) -> LinearDifferentialOperator:
        """Return ``L = d^2/dx^2 - c`` (Helmholtz / screened Poisson operator).

        Args:
            c: Reaction coefficient.

        Returns:
            The operator ``u'' - c u``.

        References:
            [EQ-SUR-GPX-013], [REF-RAISSI-ETAL-2017].

        """
        return cls((-require_finite_real(c, name="c"), 0.0, 1.0))

    def apply(self, derivatives: Sequence[float]) -> float:
        """Return ``sum_m a_m u^(m)`` from the derivatives ``(u, u', u'', ...)``.

        Args:
            derivatives: Derivative values, at least ``M + 1`` of them.

        Returns:
            ``(L u)(x)``.

        Raises:
            ValueError: If too few derivatives are given.

        References:
            [EQ-SUR-GPX-013], [REF-JIDLING-ETAL-2017].

        """
        if len(derivatives) < len(self.coefficients):
            raise ValueError("need derivatives up to the order of the operator")
        return math.fsum(
            a * d for a, d in zip(self.coefficients, derivatives, strict=False)
        )


def _se_derivative(order: int, delta: float, scale: float, variance: float) -> float:
    """Return ``d^m/dDelta^m [s2 exp(-Delta^2/(2 ell^2))]`` for ``m = order``.

    Closed form ``s2 (-1/ell)^m He_m(Delta/ell) exp(-Delta^2/(2 ell^2))`` with
    the probabilists' Hermite polynomials ``He_(m+1) = t He_m - m He_(m-1)``.

    Equation:
        [EQ-SUR-GPX-013].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 4].

    """
    t = delta / scale
    previous, current = 1.0, t
    if order == 0:
        hermite = 1.0
    else:
        for m in range(1, order):
            previous, current = current, t * current - m * previous
        hermite = current
    return variance * (-1.0 / scale) ** order * hermite * math.exp(-0.5 * t * t)


def operator_covariances(
    operator: LinearDifferentialOperator,
    delta: float,
    length_scale: float,
    signal_variance: float,
) -> tuple[float, float, float]:
    """Return ``cov(Lu(x), Lu(y))``, ``cov(Lu(x), u(y))``, ``cov(u(x), Lu(y))``.

    For ``u ~ GP(0, k)`` with the squared-exponential ``k(Delta)``,
    ``Delta = x - y``, and ``D_m = d^m k / d Delta^m``:
    ``cov(Lu, u) = sum_m a_m D_m``, ``cov(u, Lu) = sum_n a_n (-1)^n D_n`` and
    ``cov(Lu, Lu) = sum_(m,n) a_m a_n (-1)^n D_(m+n)`` [EQ-SUR-GPX-013]. For
    ``L = d/dx`` the first is ``k (1/ell^2 - Delta^2/ell^4)``; for
    ``L = d^2/dx^2 - c`` it is ``D_4 - 2 c D_2 + c^2 D_0``.

    Args:
        operator: The operator ``L``.
        delta: ``x - y``.
        length_scale: Positive ``ell``.
        signal_variance: Positive ``sigma_f^2``.

    Returns:
        ``(cov_LL, cov_Lu, cov_uL)``.

    Equation:
        [EQ-SUR-GPX-013].

    References:
        [REF-JIDLING-ETAL-2017], [REF-RAISSI-ETAL-2017].

    """
    a = operator.coefficients
    order = len(a) - 1
    d = [
        _se_derivative(m, delta, length_scale, signal_variance)
        for m in range(2 * order + 1)
    ]
    cov_lu = math.fsum(a[m] * d[m] for m in range(order + 1))
    cov_ul = math.fsum(a[n] * (-1.0) ** n * d[n] for n in range(order + 1))
    cov_ll = math.fsum(
        a[m] * a[n] * (-1.0) ** n * d[m + n]
        for m in range(order + 1)
        for n in range(order + 1)
    )
    return cov_ll, cov_lu, cov_ul


@dataclass(frozen=True)
class OperatorGaussianProcess:
    """A GP prior on ``u`` conditioned on observations of ``L u`` and of ``u``.

    ``u ~ GP(0, k_SE)`` (zero prior mean, no centering), ``y_i^L = (L u)(x_i)
    + eps`` and ``y_j^u = u(x_j) + eps`` [EQ-SUR-GPX-013], [EQ-SUR-GPX-014].
    ``L`` of a GP is a GP, so the joint law is Gaussian with the closed-form
    covariances of :func:`operator_covariances`. If ``L`` annihilates some
    functions (for example constants for ``d/dx``), ``u`` is identified only up
    to that null space unless direct ``u`` observations are supplied; the
    posterior variance of ``u`` then stays large (see the tests).

    Args:
        operator: The operator ``L``.
        signal_variance: Fitted ``sigma_f^2``.
        length_scale: Fitted ``ell``.
        operator_points: Inputs of the ``L u`` observations.
        operator_values: The ``L u`` observations.
        value_points: Inputs of direct ``u`` observations.
        values: Direct observations of ``u``.
        operator_noise: Noise variance of the ``L u`` observations.
        value_noise: Noise variance of the ``u`` observations.
        log_marginal_likelihood: Log evidence of the joint data.
        converged: Whether the hyperparameter fit met its tolerance.
        evaluations: Likelihood evaluations.
        condition_number: 1-norm condition number of the joint matrix.

    References:
        [REF-JIDLING-ETAL-2017], [REF-RAISSI-ETAL-2017].

    """

    operator: LinearDifferentialOperator
    signal_variance: float
    length_scale: float
    operator_points: Vector
    operator_values: Vector
    value_points: Vector
    values: Vector
    operator_noise: float
    value_noise: float
    log_marginal_likelihood: float
    converged: bool
    evaluations: int
    condition_number: float = field(init=False, compare=False, default=math.nan)
    _lower: list[list[float]] = field(init=False, repr=False, compare=False)
    _alpha: Vector = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Factorize the joint covariance once.

        Raises:
            ValueError: If the joint matrix is not positive definite.

        Equation:
            [EQ-SUR-GPX-014].

        References:
            [REF-JIDLING-ETAL-2017].

        """
        matrix = _operator_matrix(
            self.operator,
            self.operator_points,
            self.value_points,
            self.length_scale,
            self.signal_variance,
            self.operator_noise,
            self.value_noise,
        )
        lower = _factor(matrix, "operator-GP covariance")
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(
            self, "_alpha", _triangular_solve(lower, self.operator_values + self.values)
        )
        object.__setattr__(
            self, "condition_number", _condition(matrix, _precision(lower))
        )

    def _cross(self, x: float, *, apply_operator: bool) -> tuple[Vector, float]:
        """Return the covariances of ``u(x)`` or ``Lu(x)`` with the data.

        Equation:
            [EQ-SUR-GPX-014].

        References:
            [REF-JIDLING-ETAL-2017].

        """
        cross: list[float] = []
        for xi in self.operator_points:
            ll, _, ul = operator_covariances(
                self.operator, x - xi, self.length_scale, self.signal_variance
            )
            cross.append(ll if apply_operator else ul)
        for xj in self.value_points:
            _, lu, _ = operator_covariances(
                self.operator, x - xj, self.length_scale, self.signal_variance
            )
            cross.append(
                lu
                if apply_operator
                else _se_derivative(0, x - xj, self.length_scale, self.signal_variance)
            )
        prior = (
            operator_covariances(
                self.operator, 0.0, self.length_scale, self.signal_variance
            )[0]
            if apply_operator
            else self.signal_variance
        )
        return tuple(cross), prior

    def _predict(self, x: float, *, apply_operator: bool) -> tuple[float, float]:
        """Return the posterior mean and variance of ``u(x)`` or ``Lu(x)``.

        Equation:
            [EQ-SUR-GPX-014].

        References:
            [REF-JIDLING-ETAL-2017].

        """
        point = require_finite_real(x, name="x")
        cross, prior = self._cross(point, apply_operator=apply_operator)
        whitened = _forward(self._lower, cross)
        return _sumprod(cross, self._alpha), max(
            prior - _sumprod(whitened, whitened), 0.0
        )

    def predict_with_variance(self, x: float) -> tuple[float, float]:
        """Return the posterior mean and variance of ``u(x)``.

        Args:
            x: Scalar input.

        Returns:
            Latent mean and variance of ``u``.

        Equation:
            [EQ-SUR-GPX-014].

        References:
            [REF-JIDLING-ETAL-2017].

        """
        return self._predict(x, apply_operator=False)

    def predict_operator_with_variance(self, x: float) -> tuple[float, float]:
        """Return the posterior mean and variance of ``(L u)(x)``.

        Args:
            x: Scalar input.

        Returns:
            Mean and variance of the operator image.

        Equation:
            [EQ-SUR-GPX-014].

        References:
            [REF-JIDLING-ETAL-2017].

        """
        return self._predict(x, apply_operator=True)


def _operator_matrix(
    operator: LinearDifferentialOperator,
    operator_points: Sequence[float],
    value_points: Sequence[float],
    length_scale: float,
    signal_variance: float,
    operator_noise: float,
    value_noise: float,
) -> Matrix2:
    """Assemble the joint covariance of ``(Lu(X_L), u(X_u))`` plus noise.

    Equation:
        [EQ-SUR-GPX-014].

    References:
        [REF-JIDLING-ETAL-2017].

    """
    nodes = [(x, True) for x in operator_points] + [(x, False) for x in value_points]
    total = len(nodes)
    matrix = [[0.0] * total for _ in range(total)]
    for i, (xi, li) in enumerate(nodes):
        for j in range(i + 1):
            xj, lj = nodes[j]
            ll, lu, ul = operator_covariances(
                operator, xi - xj, length_scale, signal_variance
            )
            if li and lj:
                value = ll
            elif li:
                value = lu
            elif lj:
                value = ul
            else:
                value = _se_derivative(0, xi - xj, length_scale, signal_variance)
            matrix[i][j] = matrix[j][i] = value
        matrix[i][i] += operator_noise if li else value_noise
    return matrix


def fit_operator_gp(
    operator: LinearDifferentialOperator,
    operator_points: Sequence[float],
    operator_values: Sequence[float],
    *,
    value_points: Sequence[float] = (),
    values: Sequence[float] = (),
    signal_variance: float = 1.0,
    length_scale: float = 1.0,
    operator_noise: float = 1e-6,
    value_noise: float = 1e-8,
    optimize: bool = False,
    max_iterations: int = 60,
) -> OperatorGaussianProcess:
    """Condition a squared-exponential GP on ``L u`` (and optionally ``u``) data.

    With ``u ~ GP(0, sigma_f^2 e^(-Delta^2/(2 ell^2)))`` the data
    ``(L u)(x_i) + eps`` and ``u(x_j) + eps`` are jointly Gaussian with the
    closed-form covariances of :func:`operator_covariances`
    [EQ-SUR-GPX-013]; the posterior of ``u`` (and of ``L u``) follows from
    ordinary Gaussian conditioning [EQ-SUR-GPX-014]. This is the
    linearly-constrained / physics-informed GP of Jidling et al. and
    Raissi et al. in one dimension, for example inferring ``u`` from noisy
    observations of a source term ``f = u'' - c u`` plus boundary values. By
    default the hyperparameters are used as given; with ``optimize`` the pair
    ``(log sigma_f^2, log ell)`` is fitted by BFGS **with central
    finite-difference gradients**. The joint matrix is never jittered: a
    non-positive-definite matrix raises ``ValueError``, and its condition
    number is reported (derivative data with a small noise variance and a long
    length scale can be very ill-conditioned).

    Args:
        operator: The operator ``L``.
        operator_points: Inputs of the ``L u`` observations (non-empty).
        operator_values: The ``L u`` observations.
        value_points: Inputs of direct ``u`` observations (e.g. boundaries).
        values: The direct observations.
        signal_variance: ``sigma_f^2`` (start value if optimized).
        length_scale: ``ell`` (start value if optimized).
        operator_noise: Positive noise variance of the ``L u`` data.
        value_noise: Positive noise variance of the ``u`` data.
        optimize: Fit ``sigma_f^2`` and ``ell`` by maximum likelihood.
        max_iterations: BFGS iteration limit.

    Returns:
        The fitted :class:`OperatorGaussianProcess`.

    Raises:
        TypeError: If ``operator`` is not a :class:`LinearDifferentialOperator`.
        ValueError: If inputs are invalid or the joint matrix is singular.

    Equation:
        [EQ-SUR-GPX-013], [EQ-SUR-GPX-014].

    References:
        [REF-JIDLING-ETAL-2017], [REF-RAISSI-ETAL-2017].

    """
    if not isinstance(operator, LinearDifferentialOperator):
        raise TypeError("operator must be a LinearDifferentialOperator")
    xl = as_vector(operator_points, name="operator_points")
    yl = as_vector(operator_values, name="operator_values")
    xu = as_vector(value_points, name="value_points")
    yu = as_vector(values, name="values")
    if not xl or len(xl) != len(yl) or len(xu) != len(yu):
        raise ValueError("need matching, non-empty operator data")
    s2 = _require_positive(signal_variance, "signal_variance")
    ell = _require_positive(length_scale, "length_scale")
    noise_l = _require_positive(operator_noise, "operator_noise")
    noise_u = _require_positive(value_noise, "value_noise")
    data = yl + yu
    evaluations = [0]

    def negative(theta: Vector) -> float:
        """Return ``-log p(data)`` at ``theta = (log sigma_f^2, log ell)``.

        References:
            [EQ-SUR-GPX-014], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        evaluations[0] += 1
        lower = _factor(
            _operator_matrix(
                operator,
                xl,
                xu,
                math.exp(theta[1]),
                math.exp(theta[0]),
                noise_l,
                noise_u,
            ),
            "operator-GP covariance",
        )
        alpha = _triangular_solve(lower, data)
        return (
            0.5 * _sumprod(data, alpha)
            + 0.5 * _log_det(lower)
            + 0.5 * len(data) * _LOG_TWO_PI
        )

    theta0 = (math.log(s2), math.log(ell))
    theta: Vector = theta0
    converged = True
    if optimize:
        run = minimize_bfgs(
            negative,
            theta0,
            gradient_tolerance=1e-5,
            max_iterations=_require_count(max_iterations, "max_iterations"),
            bounds=[(-_THETA_BOUND, _THETA_BOUND)] * 2,
        )
        theta, converged = run.x, run.converged
    final = negative(theta)
    fitted = OperatorGaussianProcess(
        operator,
        math.exp(theta[0]),
        math.exp(theta[1]),
        xl,
        yl,
        xu,
        yu,
        noise_l,
        noise_u,
        -final,
        converged,
        evaluations[0],
    )
    _warn_conditioning(fitted.condition_number, "operator-GP covariance")
    return fitted
